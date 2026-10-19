"""Registries for the bundled PN→RP1 runtime.

PNToRP1Retargeter temporarily registers its explicit RP1.3.2 model and generated
calibration before constructing GeneralMotionRetargeting, then removes the
entry. No other robot models or generic human-source mappings are bundled.
"""

from pathlib import Path

HERE = Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE.parent / "assets"

# Preserve the registry objects consumed by motion_retarget.py; the project
# adapter supplies the actual paths and task weights at construction time.
ROBOT_XML_DICT = {}
IK_CONFIG_DICT = {}
ROBOT_BASE_DICT = {}
VIEWER_CAM_DISTANCE_DICT = {}
