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
            self.assertEqual(evaluator_python(config, root=root, env={}), root/'.venv/bin/python')
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

    def test_wrapper_prints_configuration_without_launching_or_creating_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='rdt portable ') as temporary:
            directory = Path(temporary)
            config = directory/'profile.yaml'
            config.write_text('seed: 42\ndata_root: ../custom_data\nshared_eval_dir: ../custom_eval\n'
                              f'unilip_python: {sys.executable}\noutput_root: {directory}/outputs\n'
                              f'checkpoint_root: {directory}/checkpoints\n')
            env = {k:v for k,v in os.environ.items() if k not in (
                'DATA_ROOT', 'CSGO_DATA_ROOT', 'SHARED_EVAL_DIR', 'CSGO_EVAL_ROOT', 'UNILIP_PYTHON', 'PYTHON')}
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


if __name__ == '__main__':
    unittest.main()
