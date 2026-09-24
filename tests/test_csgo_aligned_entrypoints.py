"""Entry-point protocol, checkpoint selection and coordinate export checks."""
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from infer_seen10 import _checkpoint_path, _provenance
from scripts.export_unilip_seen_predictions import convert_row
from train_seen10 import _native_argv, _resolved_paths, build_parser


class EntryPointTests(unittest.TestCase):
    def test_profile_paths_and_native_invocation(self):
        path = Path('configs/csgo_seen10_aligned.yaml').resolve()
        config = yaml.safe_load(path.read_text())
        cli = build_parser().parse_args(['train', '--config', str(path)])
        _, artifacts, checkpoints = _resolved_paths(config, path, cli)
        self.assertTrue(str(artifacts).endswith('csgo_aligned_aug_v1/RDT/seed_42'))
        args = _native_argv(config_path=path, checkpoint_dir=checkpoints, seed=42,
                            max_steps=19500, interval=4000, config=config, cli=cli, extra=[])
        for key, value in [('--checkpointing_period', '4000'), ('--max_train_steps', '19500'),
                           ('--gradient_accumulation_steps', '32'), ('--lr_warmup_steps', '59')]:
            self.assertEqual(args[args.index(key)+1], value)
        self.assertIn('--image_aug', args)

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
