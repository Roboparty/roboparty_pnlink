"""Resolve optional local assets without importing MuJoCo, GMR, ROS or the SDK.

Bundled model/GMR assets are preferred by default; custom installations can use
PN_RP1_MODEL and PN_RP1_GMR_ROOT. Calibration model paths are relative to the
calibration JSON, while explicit CLI/environment paths are relative to cwd.
Existing home-directory layouts are discovered only for local compatibility.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
_MODEL_RELATIVE = Path('assets/robots/rp1_description/mjcf/robot_28dof.teleop.xml')


def _absolute(value, *, base=None, follow_symlinks=True):
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve() if follow_symlinks else Path(os.path.abspath(path))


def _environment_path(*names, follow_symlinks=True):
    for name in names:
        value = os.environ.get(name)
        if value:
            return _absolute(value, follow_symlinks=follow_symlinks)
    return None


def default_model_path():
    """Select a model location; absence is reported only when loading it."""
    environment = _environment_path('PN_RP1_MODEL')
    if environment is not None:
        return environment
    candidates = (PACKAGE_ROOT / _MODEL_RELATIVE,
                  PACKAGE_ROOT.parent / _MODEL_RELATIVE,
                  Path.home() / 'rp/roboparty_teleop' / _MODEL_RELATIVE)
    return next((path for path in candidates if path.is_file()), candidates[0])


def resolve_model_path(model_path=None, *, configured_path=None, config_path=None):
    """Resolve CLI > environment > calibration path > discovered model.

    Relocating the same model is allowed; callers still verify its calibrated
    SHA before using it. This never rewrites the source calibration JSON.
    """
    environment = _environment_path('PN_RP1_MODEL')
    if model_path is not None:
        path = _absolute(model_path)
    elif environment is not None:
        path = environment
    elif configured_path is not None:
        base = _absolute(config_path).parent if config_path is not None else None
        path = _absolute(configured_path, base=base)
    else:
        path = _absolute(default_model_path())
    if not path.is_file():
        raise FileNotFoundError(f'RP1 model not found: {path}. Set PN_RP1_MODEL or pass --model '
                                '(keep the model XML and its referenced mesh assets together).')
    return path


def default_gmr_root():
    """Find a checkout, or return None for an installed GMR package."""
    environment = _environment_path('PN_RP1_GMR_ROOT', 'GMR_ROOT')
    if environment is not None:
        return environment
    bundled = (PACKAGE_ROOT / 'third_party/GMR',
               PACKAGE_ROOT.parent / 'third_party/GMR')
    selected = next((path for path in bundled
                     if (path / 'general_motion_retargeting/motion_retarget.py').is_file()), None)
    if selected is not None:
        return selected
    try:
        if importlib.util.find_spec('general_motion_retargeting') is not None:
            return None
    except (ImportError, ValueError):
        pass
    candidates = (*bundled, PACKAGE_ROOT.parent / 'GMR', Path.home() / 'dataset/GMR')
    return next((path for path in candidates
                 if (path / 'general_motion_retargeting/motion_retarget.py').is_file()),
                candidates[0])


def resolve_gmr_root(gmr_root=None):
    """Validate an external checkout, or use the installed Python package."""
    selected = default_gmr_root() if gmr_root is None else _absolute(gmr_root)
    if selected is None:
        return None
    path = _absolute(selected)
    if not (path / 'general_motion_retargeting/motion_retarget.py').is_file():
        raise FileNotFoundError(f'GMR checkout not found: {path}. Install general_motion_retargeting '
                                'or set PN_RP1_GMR_ROOT / pass --gmr-root to its repository root.')
    return path


def resolve_python_runtime(kind):
    """Choose an explicit environment runtime, an existing local one, or self.

    No PATH or conda-environment search is performed. Explicit CLI overrides
    belong to the caller. Environment values are not required to exist during
    argument parsing; launching them will report a missing executable normally.
    """
    if kind not in ('solver', 'viser'):
        raise ValueError("runtime kind must be 'solver' or 'viser'")
    # Resolving a venv's python symlink would launch the system interpreter
    # and silently drop the selected virtual environment's dependencies.
    environment = _environment_path(f'PN_RP1_{kind.upper()}_PYTHON', follow_symlinks=False)
    if environment is not None:
        return environment
    repository_runtime = PACKAGE_ROOT.parent / '.venv-ik/bin/python'
    if repository_runtime.is_file():
        return repository_runtime
    local = (Path.home() / 'miniconda3/envs/gmr/bin/python' if kind == 'solver' else
             PACKAGE_ROOT.parent / '.runtime/rp1-viser/bin/python')
    return local if local.is_file() else Path(sys.executable)


def default_mesh_cache(legacy_path=None):
    """Reuse an optional existing CAD cache; public installs need no cache."""
    environment = _environment_path('PN_RP1_MESH_CACHE')
    if environment is not None:
        return environment
    if legacy_path is not None and Path(legacy_path).is_file():
        return _absolute(legacy_path)
    root = _environment_path('XDG_CACHE_HOME') or Path.home() / '.cache'
    return root / 'pn-rp1/rp1_web_model.json'
