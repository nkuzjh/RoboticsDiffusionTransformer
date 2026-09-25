"""Entry-point protocol, checkpoint selection and coordinate export checks."""
import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from infer_seen10 import _checkpoint_path, _provenance, _resolve_batch_size, main as infer_main
from scripts.export_unilip_seen_predictions import convert_row
from train_seen10 import _native_argv, _resolved_paths, build_parser, main as train_main


class EntryPointTests(unittest.TestCase):
    def test_overwrite_error_identifies_actual_nonempty_directories(self):
        for populated in (('checkpoint_dir',), ('output_dir',), ('output_dir', 'checkpoint_dir')):
            with self.subTest(populated=populated), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = {'output_dir': root/'outputs', 'checkpoint_dir': root/'checkpoints'}
                for label, path in paths.items():
                    path.mkdir()
                    if label in populated:
                        (path/'run_metadata.json').write_text('{}')
                with patch('train_seen10._is_main_process', return_value=True), \
                        patch('train_seen10._native_argv') as native, \
                        self.assertRaises(FileExistsError) as raised:
                    train_main(['train', '--config', 'configs/csgo_seen10_aligned.yaml',
                                '--output-dir', str(paths['output_dir']),
                                '--checkpoint-dir', str(paths['checkpoint_dir'])])
                native.assert_not_called()
                message = str(raised.exception)
                for label, path in paths.items():
                    self.assertEqual(f'{label}={path}' in message, label in populated)
                self.assertNotIn('new --seed', message)

    def test_profile_paths_and_native_invocation(self):
        path = Path('configs/csgo_seen10_aligned.yaml').resolve()
        config = yaml.safe_load(path.read_text())
        cli = build_parser().parse_args(['train', '--config', str(path)])
        _, artifacts, checkpoints = _resolved_paths(config, path, cli)
        self.assertTrue(str(artifacts).endswith('csgo_aligned_aug_v1/RDT/seed_42'))
        self.assertEqual(checkpoints, artifacts/'checkpoints')
        args = _native_argv(config_path=path, checkpoint_dir=checkpoints, seed=42,
                            max_steps=19500, interval=4000, config=config, cli=cli, extra=[])
        for key, value in [('--checkpointing_period', '4000'), ('--max_train_steps', '19500'),
                           ('--lr_warmup_steps', '59')]:
            self.assertEqual(args[args.index(key)+1], value)
        # This invocation is single-process. Only the product is prescribed;
        # changing the microbatch/accumulation split must not break the test.
        effective_batch = (int(args[args.index('--train_batch_size')+1])
                           * int(args[args.index('--gradient_accumulation_steps')+1]))
        self.assertEqual(effective_batch, 128)
        self.assertIn('--image_aug', args)

    def test_training_accepts_any_split_with_effective_batch_128(self):
        for world, microbatch, accumulation in ((1, 4, 32), (1, 32, 4), (1, 128, 1), (2, 16, 4)):
            with self.subTest(world=world, microbatch=microbatch, accumulation=accumulation):
                output = io.StringIO()
                with patch.dict('os.environ', {'WORLD_SIZE': str(world)}), contextlib.redirect_stdout(output):
                    result = train_main(['train', '--config', 'configs/csgo_seen10_aligned.yaml', '--dry-run',
                                         '--train_batch_size', str(microbatch),
                                         '--gradient_accumulation_steps', str(accumulation)])
                self.assertEqual(result, 0)
                resolved = json.loads(output.getvalue())['args']
                self.assertEqual(world * resolved['train_batch_size'] * resolved['gradient_accumulation_steps'], 128)

    def test_training_rejects_effective_batch_other_than_128(self):
        for world, microbatch, accumulation in ((1, 32, 2), (2, 32, 4)):
            with self.subTest(world=world, microbatch=microbatch, accumulation=accumulation):
                with patch.dict('os.environ', {'WORLD_SIZE': str(world)}), \
                        self.assertRaisesRegex(ValueError, 'effective batch must be 128'):
                    train_main(['train', '--config', 'configs/csgo_seen10_aligned.yaml', '--dry-run',
                                '--train_batch_size', str(microbatch),
                                '--gradient_accumulation_steps', str(accumulation)])

    def test_inference_batch_is_independent_of_training_budget(self):
        config = {'training': {'train_batch_size': 32, 'gradient_accumulation_steps': 4, 'eval_batch_size': 32},
                  'inference': {'batch_size': 1}}
        self.assertEqual(_resolve_batch_size(config), 1)
        for batch in (1, 4, 32, 128, 256):
            with self.subTest(batch=batch):
                self.assertEqual(_resolve_batch_size(config, batch), batch)
                self.assertEqual(_resolve_batch_size({'inference': {'batch_size': batch}}), batch)
        self.assertEqual(_resolve_batch_size({'training': {'eval_batch_size': 32}}), 32)
        for batch in (0, -1, 1.5, True, 'bad'):
            with self.subTest(invalid_batch=batch), self.assertRaisesRegex(ValueError, 'positive integer'):
                _resolve_batch_size({'inference': {'batch_size': batch}})

    def test_inference_entry_accepts_batch_32_but_still_rejects_multiple_processes(self):
        # Stop at checkpoint lookup: exercise the real entry checks without
        # loading weights, creating outputs, or running model inference.
        command = ['infer', '--config', 'configs/csgo_seen10_aligned.yaml', '--batch-size', '32']
        with patch.dict('os.environ', {'WORLD_SIZE': '1'}), \
                patch('infer_seen10._checkpoint_path', side_effect=RuntimeError('checkpoint lookup reached')) as lookup, \
                self.assertRaisesRegex(RuntimeError, 'checkpoint lookup reached'):
            infer_main(command)
        lookup.assert_called_once()
        with patch.dict('os.environ', {'WORLD_SIZE': '2'}), \
                patch('infer_seen10._checkpoint_path') as lookup, \
                self.assertRaisesRegex(ValueError, 'single process'):
            infer_main(command)
        lookup.assert_not_called()

    def test_aligned_late_does_not_select_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['checkpoint-4000', 'checkpoint-19500']:
                (root/name).mkdir()
                (root/name/'config.json').write_text('{}')
            (root/'best').symlink_to('checkpoint-4000')
            (root/'late').symlink_to('checkpoint-19500')
            self.assertEqual(_checkpoint_path(root, rule='late').resolve(), root/'checkpoint-19500')
            self.assertEqual(_checkpoint_path(root).resolve(), root/'checkpoint-4000')

    def test_provenance_prevents_changed_inference_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = dict(checkpoint=root/'checkpoint-4000', seed=42, data_root=root)
            path = root/'provenance.json'
            _provenance(path, **args, extra={'batch_size': 1, 'num_processes': 1})
            _provenance(path, **args, extra={'batch_size': 1, 'num_processes': 1})
            with self.assertRaises(ValueError):
                _provenance(path, **args, extra={'batch_size': 4, 'num_processes': 1})

    def test_unilip_z_export_preserves_physical_prediction_without_gt(self):
        ranges = {'de_nuke': {'z_min': -15., 'z_max': 45.}}
        row = {'map':'de_nuke', 'file_frame':'file_num1_frame_0002',
               'pred_norm':[1.1, -0.2, 1.4, -.05, 1.2], 'gt_norm':'must never be consumed'}
        result = convert_row(row, ranges, 1e-6)
        self.assertAlmostEqual(result['pred_z']*60-15, 1.4*(60+1e-6)-15, places=12)
        self.assertEqual(result['pred_x'], 1.1)
        self.assertEqual(result['pred_yaw'], 1.2)
        self.assertNotIn('gt_norm', result)
        self.assertEqual(result['sample_id'], 'file_num1_frame_0002')


if __name__ == '__main__':
    unittest.main()
