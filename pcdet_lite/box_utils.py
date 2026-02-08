"""
Box Utilities
=============
Pure Python 3D bounding box operations.
Adapted from OpenPCDet: pcdet/utils/box_utils.py

Key functions:
- points_in_boxes_cpu: Pure Python box-point containment check (replaces CUDA op)
- enlarge_box3d: Expand boxes by margin
- boxes_to_corners_3d: Convert [x,y,z,dx,dy,dz,heading] to 8 corners
- rotate_points_along_z: Rotate points around Z axis
"""

import numpy as np
import torch


def rotate_points_along_z(points, angle):
    """
    Rotate points along Z-axis.
    
    Args:
        points: (N, 3) or (B, N, 3) tensor
        angle: (N,) or (B, N) rotation angles in radians
        
    Returns:
        rotated_points: same shape as input
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(angle.shape)
    ones = angle.new_ones(angle.shape)
    
    if points.dim() == 2:
        # (N, 3)
        rot_matrix = torch.stack([
            cosa, -sina, zeros,
            sina, cosa, zeros,
            zeros, zeros, ones
        ], dim=1).view(-1, 3, 3)
        points_rotated = torch.bmm(rot_matrix, points.unsqueeze(-1)).squeeze(-1)
    else:
        # (B, N, 3) - not commonly used in our case
        rot_matrix = torch.stack([
            cosa, -sina, zeros,
            sina, cosa, zeros,
            zeros, zeros, ones
        ], dim=-1).view(*angle.shape, 3, 3)
        points_rotated = torch.matmul(rot_matrix, points.unsqueeze(-1)).squeeze(-1)
    
    return points_rotated


def boxes_to_corners_3d(boxes):
    """
    Convert 3D boxes to 8 corner coordinates.
    
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading] or (N, 8) with class
        
    Returns:
        corners: (N, 8, 3) corner coordinates
    """
    template = torch.tensor([
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    ], dtype=boxes.dtype, device=boxes.device) / 2
    
    corners = boxes[:, None, 3:6] * template[None, :, :]  # (N, 8, 3)
    
    # Rotate around z-axis
    heading = boxes[:, 6]
    cosa = torch.cos(heading)
    sina = torch.sin(heading)
    
    # Rotation matrix for each box
    rot = boxes.new_zeros((boxes.shape[0], 3, 3))
    rot[:, 0, 0] = cosa
    rot[:, 0, 1] = -sina
    rot[:, 1, 0] = sina
    rot[:, 1, 1] = cosa
    rot[:, 2, 2] = 1
    
    # Apply rotation: (N, 8, 3) @ (N, 3, 3)^T = (N, 8, 3)
    corners = torch.bmm(corners, rot.transpose(1, 2))
    
    # Translate to box center
    corners += boxes[:, None, :3]
    
    return corners


def enlarge_box3d(boxes, extra_width):
    """
    Enlarge 3D boxes by adding extra width.
    
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        extra_width: [dx_extra, dy_extra, dz_extra] or scalar
        
    Returns:
        enlarged_boxes: (N, 7 or 8)
    """
    if isinstance(extra_width, (list, tuple)):
        extra_width = torch.tensor(extra_width, dtype=boxes.dtype, device=boxes.device)
    
    enlarged_boxes = boxes.clone()
    enlarged_boxes[:, 3:6] = boxes[:, 3:6] + extra_width * 2
    
    return enlarged_boxes


def points_in_boxes_cpu(points, boxes):
    """
    Check if points are inside 3D boxes - VECTORIZED PyTorch (no Python loops).
    
    This replaces the CUDA op `roiaware_pool3d_utils.points_in_boxes_gpu`.
    
    Args:
        points: (N, 3) point coordinates
        boxes: (M, 7) [x, y, z, dx, dy, dz, heading]
        
    Returns:
        box_idxs_of_pts: (N,) index of box containing each point, -1 if none
    """
    n_points = points.shape[0]
    n_boxes = boxes.shape[0]
    
    if n_boxes == 0:
        return points.new_full((n_points,), -1, dtype=torch.long)
    
    # Fully vectorized implementation - no Python loops
    # Broadcast: points (N, 1, 3), boxes (1, M, 7)
    
    # Extract box parameters
    cx = boxes[:, 0]  # (M,)
    cy = boxes[:, 1]
    cz = boxes[:, 2]
    dx = boxes[:, 3]
    dy = boxes[:, 4]
    dz = boxes[:, 5]
    heading = boxes[:, 6]
    
    # Translate points to box-centered coordinates (N, M, 3)
    local_x = points[:, 0:1] - cx.unsqueeze(0)  # (N, M)
    local_y = points[:, 1:2] - cy.unsqueeze(0)  # (N, M)
    local_z = points[:, 2:3] - cz.unsqueeze(0)  # (N, M)
    
    # Rotate points opposite to box heading (vectorized)
    cosa = torch.cos(-heading).unsqueeze(0)  # (1, M)
    sina = torch.sin(-heading).unsqueeze(0)  # (1, M)
    
    x_rot = local_x * cosa - local_y * sina  # (N, M)
    y_rot = local_x * sina + local_y * cosa  # (N, M)
    
    # Check if inside box (half-dimensions)
    in_box = (
        (torch.abs(x_rot) <= dx.unsqueeze(0) / 2) &
        (torch.abs(y_rot) <= dy.unsqueeze(0) / 2) &
        (torch.abs(local_z) <= dz.unsqueeze(0) / 2)
    )  # (N, M) boolean
    
    # Find first box containing each point (argmax on boolean gives first True)
    # If no box contains point, we need to return -1
    any_box = in_box.any(dim=1)  # (N,)
    
    # Get box index for each point
    box_idxs_of_pts = points.new_full((n_points,), -1, dtype=torch.long)
    
    # For points inside any box, find which box
    if any_box.any():
        # argmax on boolean: first True index
        box_idxs_of_pts[any_box] = in_box[any_box].to(torch.float).argmax(dim=1)
    
    return box_idxs_of_pts


def points_in_boxes_batch(points, boxes):
    """
    Batch version of points_in_boxes_cpu.
    
    Args:
        points: (B, N, 3) or list of (N_i, 3)
        boxes: (B, M, 7) [x, y, z, dx, dy, dz, heading]
        
    Returns:
        box_idxs: (B, N) or list of (N_i,)
    """
    if points.dim() == 2:
        # Single batch case
        return points_in_boxes_cpu(points, boxes.squeeze(0))
    
    batch_size = points.shape[0]
    results = []
    
    for b in range(batch_size):
        pts_b = points[b]
        boxes_b = boxes[b]
        
        # Filter out padding boxes (all zeros)
        valid_mask = boxes_b[:, 3:6].sum(dim=1) > 0
        valid_boxes = boxes_b[valid_mask]
        
        result = points_in_boxes_cpu(pts_b, valid_boxes)
        results.append(result)
    
    return torch.stack(results, dim=0)


def boxes_iou3d_cpu(boxes_a, boxes_b):
    """
    Calculate 3D IoU between boxes - VECTORIZED PyTorch (no Python loops).
    
    Uses axis-aligned bounding box approximation for speed.
    For accurate rotated IoU, use the full CUDA implementation.
    
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
        
    Returns:
        iou: (N, M) IoU matrix
    """
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    
    if n == 0 or m == 0:
        return boxes_a.new_zeros((n, m))
    
    # Fully vectorized implementation using broadcasting
    # boxes_a: (N, 1, 7), boxes_b: (1, M, 7)
    
    # Box A bounds (N, 1)
    ax_min = (boxes_a[:, 0] - boxes_a[:, 3] / 2).unsqueeze(1)
    ax_max = (boxes_a[:, 0] + boxes_a[:, 3] / 2).unsqueeze(1)
    ay_min = (boxes_a[:, 1] - boxes_a[:, 4] / 2).unsqueeze(1)
    ay_max = (boxes_a[:, 1] + boxes_a[:, 4] / 2).unsqueeze(1)
    az_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).unsqueeze(1)
    az_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).unsqueeze(1)
    
    # Box B bounds (1, M)
    bx_min = (boxes_b[:, 0] - boxes_b[:, 3] / 2).unsqueeze(0)
    bx_max = (boxes_b[:, 0] + boxes_b[:, 3] / 2).unsqueeze(0)
    by_min = (boxes_b[:, 1] - boxes_b[:, 4] / 2).unsqueeze(0)
    by_max = (boxes_b[:, 1] + boxes_b[:, 4] / 2).unsqueeze(0)
    bz_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).unsqueeze(0)
    bz_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).unsqueeze(0)
    
    # Calculate intersection dimensions (N, M)
    inter_x = torch.clamp(torch.min(ax_max, bx_max) - torch.max(ax_min, bx_min), min=0)
    inter_y = torch.clamp(torch.min(ay_max, by_max) - torch.max(ay_min, by_min), min=0)
    inter_z = torch.clamp(torch.min(az_max, bz_max) - torch.max(az_min, bz_min), min=0)
    
    # Intersection volume (N, M)
    inter_vol = inter_x * inter_y * inter_z
    
    # Individual volumes (N, 1) and (1, M)
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).unsqueeze(1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).unsqueeze(0)
    
    # Union volume
    union_vol = vol_a + vol_b - inter_vol
    
    # IoU
    iou = inter_vol / torch.clamp(union_vol, min=1e-6)
    
    return iou
