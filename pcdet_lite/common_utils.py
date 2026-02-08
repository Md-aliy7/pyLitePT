"""
Common Utilities
================
General utility functions for 3D detection.
Adapted from OpenPCDet: pcdet/utils/common_utils.py
"""

import numpy as np
import torch


def rotate_points_along_z(points, angle):
    """
    Rotate points along the z-axis by given angles.
    
    Args:
        points: (..., 3) point coordinates
        angle: (...) rotation angles in radians
        
    Returns:
        rotated_points: (..., 3) rotated coordinates
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    
    # Apply 2D rotation in xy plane
    x_rot = x * cosa - y * sina
    y_rot = x * sina + y * cosa
    
    rotated = torch.stack([x_rot, y_rot, z], dim=-1)
    return rotated


def limit_period(val, offset=0.5, period=np.pi * 2):
    """
    Limit the value to a specified period.
    
    Args:
        val: tensor or array
        offset: offset factor (0.5 means center at 0)
        period: period length
        
    Returns:
        limited_val: values limited to [-period*offset, period*(1-offset)]
    """
    return val - torch.floor(val / period + offset) * period


# Unused functions (init_model_weights, drop_info, keep_arrays) removed during audit.


# Try to import nn for init_model_weights
try:
    import torch.nn as nn
except ImportError:
    pass
