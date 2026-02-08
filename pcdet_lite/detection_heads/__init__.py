"""
Detection Heads
===============
Point-based detection heads for 3D object detection.
"""

from .point_head_box import PointHeadBox
from .point_head_template import PointHeadTemplate

__all__ = ['PointHeadBox', 'PointHeadTemplate']
