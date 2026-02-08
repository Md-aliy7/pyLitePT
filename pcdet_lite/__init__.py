"""
PCDet Lite - Pure Python OpenPCDet Detection Components
========================================================
Standalone detection utilities adapted from OpenPCDet for LitePT integration.
No CUDA dependencies - works on CPU and GPU.

Original source: https://github.com/open-mmlab/OpenPCDet
"""

from .box_utils import (
    enlarge_box3d,
    boxes_to_corners_3d,
    rotate_points_along_z,
    points_in_boxes_cpu,
)

from .box_coder_utils import (
    PointResidualCoder,
    ResidualCoder,
)

from .loss_utils import (
    SigmoidFocalClassificationLoss,
    WeightedSmoothL1Loss,
)

from .detection_heads import PointHeadBox

__all__ = [
    'enlarge_box3d',
    'boxes_to_corners_3d', 
    'rotate_points_along_z',
    'points_in_boxes_cpu',
    'PointResidualCoder',
    'ResidualCoder',
    'SigmoidFocalClassificationLoss',
    'WeightedSmoothL1Loss',
    'PointHeadBox',
]
