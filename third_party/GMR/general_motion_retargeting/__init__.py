"""PN→RP1 runtime subset of the pinned GMR fork.

Only GMR's task construction module is loaded here. The PN adapter owns the
source conversion, configuration registration and single-step mink solve.
The original GMR viewer, SMPL-X/XRobot loaders and other robots are excluded.
"""

from .params import (
    ASSET_ROOT,
    IK_CONFIG_DICT,
    IK_CONFIG_ROOT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)
from .motion_retarget import GeneralMotionRetargeting

__version__ = "0.2.0"

__all__ = [
    "GeneralMotionRetargeting",
    "ASSET_ROOT",
    "IK_CONFIG_ROOT",
    "ROBOT_XML_DICT",
    "IK_CONFIG_DICT",
    "ROBOT_BASE_DICT",
    "VIEWER_CAM_DISTANCE_DICT",
]
