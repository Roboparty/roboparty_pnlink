"""Portable asset locations without opening PN, ROS, browsers, or servers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pn_rp1_paths as paths


class PortablePathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {name: '' for name in (
            'PN_RP1_MODEL', 'PN_RP1_GMR_ROOT', 'GMR_ROOT', 'PN_RP1_MESH_CACHE',
            'PN_RP1_SOLVER_PYTHON', 'PN_RP1_VISER_PYTHON', 'XDG_CACHE_HOME')}).start()

    def file(self, relative):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('local asset placeholder')
        return path

    def test_relative_model_path_uses_calibration_directory_and_does_not_modify_it(self):
        model = self.file('installation/models/robot.xml')
        config = self.file('installation/configs/pn.json')
        config.write_text(json.dumps({'robot_model_path': '../models/robot.xml'}))
        before = config.read_bytes()
        with patch('os.getcwd', return_value=str(self.root/'elsewhere')):
            selected = paths.resolve_model_path(configured_path='../models/robot.xml',
                                                 config_path=config)
        self.assertEqual(selected, model)
        self.assertEqual(config.read_bytes(), before)

    def test_explicit_model_wins_over_environment_and_environment_can_relocate_stale_json(self):
        explicit = self.file('explicit.xml')
        environment = self.file('environment.xml')
        with patch.dict(os.environ, {'PN_RP1_MODEL': str(environment)}):
            self.assertEqual(paths.resolve_model_path(explicit, configured_path='old/missing.xml'), explicit)
            self.assertEqual(paths.resolve_model_path(configured_path='old/missing.xml'), environment)
        with self.assertRaisesRegex(FileNotFoundError, 'PN_RP1_MODEL'):
            paths.resolve_model_path(configured_path=self.root/'missing.xml')

    def test_model_discovery_is_optional_and_import_default_does_not_require_assets(self):
        with patch.object(paths, 'PACKAGE_ROOT', self.root/'package'), \
             patch.object(paths.Path, 'home', return_value=self.root/'home'):
            absent = paths.default_model_path()
            self.assertFalse(absent.exists())
            packaged = self.file('package/assets/robots/rp1_description/mjcf/robot_28dof.teleop.xml')
            self.assertEqual(paths.default_model_path(), packaged)
            self.assertEqual(paths.resolve_model_path(), packaged)

    def test_gmr_uses_explicit_then_environment_then_installed_package(self):
        first = self.file('first/general_motion_retargeting/motion_retarget.py').parents[1]
        second = self.file('second/general_motion_retargeting/motion_retarget.py').parents[1]
        with patch.dict(os.environ, {'PN_RP1_GMR_ROOT': str(second)}), \
             patch.object(paths.importlib.util, 'find_spec', return_value=object()):
            self.assertEqual(paths.resolve_gmr_root(first), first)
            self.assertEqual(paths.resolve_gmr_root(), second)
        with patch.object(paths, 'PACKAGE_ROOT', self.root/'no-bundle'), \
             patch.object(paths.importlib.util, 'find_spec', return_value=object()):
            self.assertIsNone(paths.resolve_gmr_root())
        with patch.dict(os.environ, {'PN_RP1_GMR_ROOT': str(self.root/'missing')}):
            with self.assertRaisesRegex(FileNotFoundError, 'PN_RP1_GMR_ROOT'):
                paths.resolve_gmr_root()

    def test_bundled_gmr_is_preferred_over_an_installed_version(self):
        bundled = self.file('repo/third_party/GMR/general_motion_retargeting/motion_retarget.py').parents[1]
        with patch.object(paths, 'PACKAGE_ROOT', self.root/'repo/mocapapi_python'), \
             patch.object(paths.importlib.util, 'find_spec', return_value=object()):
            self.assertEqual(paths.resolve_gmr_root(), bundled)

    def test_repository_runtime_wins_over_legacy_home_layout(self):
        interpreter = self.root/'repo/.venv-ik/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        self.file('home/miniconda3/envs/gmr/bin/python')
        with patch.object(paths, 'PACKAGE_ROOT', self.root/'repo/mocapapi_python'), \
             patch.object(paths.Path, 'home', return_value=self.root/'home'):
            self.assertEqual(paths.resolve_python_runtime('solver'), interpreter)
            self.assertEqual(paths.resolve_python_runtime('viser'), interpreter)

    def test_legacy_home_gmr_discovery_and_environment_alias_are_compatible(self):
        legacy = self.file('home/dataset/GMR/general_motion_retargeting/motion_retarget.py').parents[1]
        with patch.object(paths, 'PACKAGE_ROOT', self.root/'package'), \
             patch.object(paths.Path, 'home', return_value=self.root/'home'), \
             patch.object(paths.importlib.util, 'find_spec', return_value=None):
            self.assertEqual(paths.resolve_gmr_root(), legacy)
        with patch.dict(os.environ, {'GMR_ROOT': str(legacy)}):
            self.assertEqual(paths.resolve_gmr_root(), legacy)

    def test_runtime_environment_preserves_venv_symlink_and_does_not_search_path(self):
        interpreter = self.root/'venv/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        with patch.dict(os.environ, {'PN_RP1_VISER_PYTHON': str(interpreter)}):
            self.assertEqual(paths.resolve_python_runtime('viser'), interpreter)
            self.assertNotEqual(paths.resolve_python_runtime('viser'), interpreter.resolve())
        with patch.object(paths, 'PACKAGE_ROOT', self.root/'package'), \
             patch.object(paths.Path, 'home', return_value=self.root/'home'):
            self.assertEqual(paths.resolve_python_runtime('solver'), Path(sys.executable))
            self.assertEqual(paths.resolve_python_runtime('viser'), Path(sys.executable))
            local = self.file('home/miniconda3/envs/gmr/bin/python')
            self.assertEqual(paths.resolve_python_runtime('solver'), local)
        with self.assertRaisesRegex(ValueError, 'runtime kind'):
            paths.resolve_python_runtime('unknown')

    def test_mesh_cache_environment_then_existing_local_cache_then_xdg(self):
        cached = self.file('analysis/rp1_web_model.json')
        self.assertEqual(paths.default_mesh_cache(cached), cached)
        with patch.dict(os.environ, {'PN_RP1_MESH_CACHE': str(self.root/'custom.json')}):
            self.assertEqual(paths.default_mesh_cache(cached), self.root/'custom.json')
        with patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.root/'xdg')}):
            self.assertEqual(paths.default_mesh_cache(self.root/'absent.json'),
                             self.root/'xdg/pn-rp1/rp1_web_model.json')

    def test_imports_and_cli_help_work_with_missing_model_and_gmr(self):
        env = dict(os.environ, PN_RP1_MODEL=str(self.root/'missing-model.xml'),
                   PN_RP1_GMR_ROOT=str(self.root/'missing-GMR'))
        package = Path(__file__).resolve().parents[1] / "mocapapi_python"
        script = ("import sys; import pn_rp1_calibration, pn_rp1_retarget, view_rp1_gmr_3d; "
                  "assert all(name not in sys.modules for name in "
                  "('mujoco','general_motion_retargeting','rclpy','viser','mocap_api'))")
        subprocess.run([sys.executable, '-c', script], cwd=package, env=env,
                       check=True, capture_output=True, text=True)
        for filename, arguments in (('pn_rp1_calibration.py', ['--help']),
                                    ('pn_rp1_retarget.py', ['worker', '--help']),
                                    ('view_rp1_gmr_3d.py', ['--help'])):
            result = subprocess.run([sys.executable, str(package/filename), *arguments],
                                    cwd=self.root, env=env, check=True,
                                    capture_output=True, text=True)
            self.assertIn('--model', result.stdout)


if __name__ == '__main__':
    unittest.main()
