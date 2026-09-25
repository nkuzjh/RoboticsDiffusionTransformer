"""Path resolution on both original and relocated checkouts; no model execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.csgo_paths import (
    LEGACY_DATA, LEGACY_EVAL, LEGACY_PYTHON,
    data_root, evaluator_root, evaluator_python,
)
from train_seen10 import _resolved_paths, build_parser


class PortablePathTests(unittest.TestCase):
    def test_original_machine_keeps_existing_yaml_paths(self):
        config = dict(data_root=LEGACY_DATA, shared_eval_dir=LEGACY_EVAL, unilip_python=LEGACY_PYTHON)
        with patch.object(Path, "exists", return_value=True):
            self.assertEqual(data_root(config, env={}), Path(LEGACY_DATA))
            self.assertEqual(evaluator_root(config, env={}), Path(LEGACY_EVAL))
            self.assertEqual(evaluator_python(config, env={}), Path(LEGACY_PYTHON))

    def test_other_machine_relocates_only_original_defaults(self):
        root = Path('/another user/task/RoboticsDiffusionTransformer')
        config = dict(data_root=LEGACY_DATA, shared_eval_dir=LEGACY_EVAL, unilip_python=LEGACY_PYTHON)
        with patch.object(Path, "exists", return_value=False):
            self.assertEqual(data_root(config, root=root, env={}), root.parent/'UniLIP/data/csgo_benchmark_v2')
            self.assertEqual(evaluator_root(config, root=root, env={}), root.parent/'csgo_benchmark_v2_eval_general')
            # Explicit Python paths, including the old UniLIP path, never fall back.
            self.assertEqual(evaluator_python(config, root=root, env={}), Path(LEGACY_PYTHON))
            self.assertEqual(evaluator_python({}, root=root, env={}),
                             root.parent/'csgo_benchmark_v2_eval_general/.venv/bin/python')
            self.assertEqual(data_root({'data_root': '/missing/custom'}, root=root, env={}), Path('/missing/custom'))
            self.assertEqual(evaluator_python({'unilip_python': '/missing/python'}, env={}), Path('/missing/python'))

    def test_explicit_environment_yaml_precedence_and_relative_paths(self):
        root = Path('/project')
        env = {'DATA_ROOT': 'environment', 'CSGO_DATA_ROOT': '/lower-priority'}
        config = {'data_root': 'yaml'}
        self.assertEqual(data_root(config, 'cli', root=root, env=env), root/'cli')
        self.assertEqual(data_root(config, root=root, env=env), root/'environment')
        self.assertEqual(data_root(config, root=root, env={}), root/'yaml')
        self.assertEqual(evaluator_python({}, '/venv/bin/python', env={}), Path('/venv/bin/python'))

    def test_direct_training_entry_honors_environment_and_cli(self):
        with patch.dict(os.environ, {'DATA_ROOT': '/new-server/data'}):
            config = {'data_root': LEGACY_DATA}
            cli = build_parser().parse_args([])
            self.assertEqual(_resolved_paths(config, Path('unused'), cli)[0], Path('/new-server/data'))
            cli = build_parser().parse_args(['--data-root', '/explicit/data'])
            self.assertEqual(_resolved_paths(config, Path('unused'), cli)[0], Path('/explicit/data'))

    def test_evaluator_python_matches_openvla_priority(self):
        root = Path('/project')
        config = {'unilip_python': 'yaml/bin/python', 'shared_eval_dir': '../evaluator'}
        env = {'CSGO_EVAL_PYTHON': 'preferred/bin/python', 'UNILIP_PYTHON': 'legacy/bin/python'}
        self.assertEqual(evaluator_python(config, 'cli/bin/python', root=root, env=env), root/'cli/bin/python')
        self.assertEqual(evaluator_python(config, root=root, env=env), root/'preferred/bin/python')
        self.assertEqual(evaluator_python(config, root=root, env={'UNILIP_PYTHON': 'legacy/bin/python'}),
                         root/'legacy/bin/python')
        self.assertEqual(evaluator_python(config, root=root, env={}), root/'yaml/bin/python')
        config['unilip_python'] = None
        self.assertEqual(evaluator_python(config, root=root, env={}), root/'../evaluator/.venv/bin/python')
        self.assertEqual(evaluator_python(config, root=root, env={'SHARED_EVAL_DIR': '/new/evaluator'}),
                         Path('/new/evaluator/.venv/bin/python'))

    def test_python_is_selected_without_probing_or_resolving_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root/'selected/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(sys.executable)
            with patch.object(Path, 'exists', side_effect=AssertionError('Python selection must not probe')):
                self.assertEqual(evaluator_python({}, interpreter, root=root, env={}), interpreter)
                self.assertEqual(evaluator_python({'unilip_python': LEGACY_PYTHON}, root=root, env={}),
                                 Path(LEGACY_PYTHON))
                self.assertEqual(evaluator_python({'shared_eval_dir': '/missing/evaluator'}, root=root, env={}),
                                 Path('/missing/evaluator/.venv/bin/python'))

    def test_current_profiles_use_shared_environment_by_default(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        for name in ('csgo_seen10.yaml', 'csgo_seen10_aligned.yaml'):
            config = yaml.safe_load((root/'configs'/name).read_text())
            self.assertIsNone(config['unilip_python'])
            self.assertEqual(evaluator_python(config, root=root, env={}),
                             evaluator_root(config, root=root, env={})/'.venv/bin/python')

    def test_wrapper_prints_configuration_without_launching_or_creating_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='rdt portable ') as temporary:
            directory = Path(temporary)
            config = directory/'profile.yaml'
            config.write_text('seed: 42\ndata_root: ../custom_data\nshared_eval_dir: ../custom_eval\n'
                              f'unilip_python: {sys.executable}\noutput_root: {directory}/outputs\n'
                              f'checkpoint_root: {directory}/checkpoints\n')
            env = {k:v for k,v in os.environ.items() if k not in (
                'DATA_ROOT', 'CSGO_DATA_ROOT', 'SHARED_EVAL_DIR', 'CSGO_EVAL_ROOT',
                'CSGO_EVAL_PYTHON', 'UNILIP_PYTHON', 'PYTHON')}
            result = subprocess.run(['bash', str(root/'scripts/run_csgo_seen10.sh'), 'eval',
                '--config', str(config), '--python', sys.executable, '--print-paths'],
                cwd=directory, env=env, text=True, capture_output=True, check=True)
            paths = json.loads(result.stdout)
            self.assertEqual(Path(paths['data_root']).resolve(), root.parent/'custom_data')
            self.assertEqual(Path(paths['shared_eval_dir']).resolve(), root.parent/'custom_eval')
            self.assertEqual(paths['unilip_python'], sys.executable)
            self.assertTrue(paths['checkpoint_dir'].endswith('RDT/seed_42'))
            self.assertFalse((directory/'outputs').exists())
            self.assertFalse((directory/'checkpoints').exists())

    def test_wrapper_python_overrides_and_eval_root_default(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='rdt eval ') as temporary:
            directory = Path(temporary)
            config = directory/'profile.yaml'
            config.write_text('seed: 42\ndata_root: ../data\nunilip_python: null\n'
                              f'output_root: {directory}/outputs\ncheckpoint_root: {directory}/checkpoints\n')
            env = {k:v for k,v in os.environ.items() if k not in (
                'DATA_ROOT', 'CSGO_DATA_ROOT', 'SHARED_EVAL_DIR', 'CSGO_EVAL_ROOT',
                'CSGO_EVAL_PYTHON', 'UNILIP_PYTHON', 'PYTHON')}
            command = ['bash', str(root/'scripts/run_csgo_seen10.sh'), 'eval', '--config', str(config),
                       '--python', sys.executable, '--eval-root', str(directory/'shared evaluator')]
            cases = [([], {}, directory/'shared evaluator/.venv/bin/python'),
                     ([], {'UNILIP_PYTHON': '/legacy/python'}, Path('/legacy/python')),
                     ([], {'UNILIP_PYTHON': '/legacy/python', 'CSGO_EVAL_PYTHON': '/preferred/python'}, Path('/preferred/python')),
                     (['--eval-python', '/cli/python'], {'CSGO_EVAL_PYTHON': '/env/python'}, Path('/cli/python')),
                     (['--unilip-python', '/alias/python'], {'CSGO_EVAL_PYTHON': '/env/python'}, Path('/alias/python'))]
            for args, overrides, expected in cases:
                with self.subTest(args=args, overrides=overrides):
                    result = subprocess.run(command+args+['--print-paths'], cwd=directory,
                                            env=dict(env, **overrides), text=True, capture_output=True, check=True)
                    paths = json.loads(result.stdout)
                    self.assertEqual(paths['unilip_python'], str(expected))
                    self.assertEqual(paths['evaluator_python'], str(expected))
            self.assertFalse((directory/'outputs').exists())
            self.assertFalse((directory/'checkpoints').exists())
            # An explicitly missing interpreter fails at launch; it is not
            # replaced with the project's working interpreter.
            result = subprocess.run(command+['--eval-python', str(directory/'missing-python')],
                                    cwd=directory, env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing-python', result.stderr)
            self.assertFalse(any(directory.rglob('summary_equal_map.json')))


if __name__ == '__main__':
    unittest.main()
