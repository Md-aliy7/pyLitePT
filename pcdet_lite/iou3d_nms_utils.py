"""
Pure Python implementation of 3D IoU and NMS.
Replaces OpenPCDet's CUDA-based logic for CPU-only environments.
"""

import torch
import numpy as np
from pcdet_lite.box_utils import boxes_to_corners_3d

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x -> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_bev_corners(boxes):
    """
    Convert boxes to 4 corners in BEV.
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
        corners: (N, 4, 2)
    """
    boxes, is_numpy = check_numpy_to_torch(boxes)
    
    x = boxes[:, 0]
    y = boxes[:, 1]
    dx = boxes[:, 3]
    dy = boxes[:, 4]
    heading = boxes[:, 6]
    
    # Corners in canonical frame (centered at 0, aligned)
    # 0: -dx/2, -dy/2
    # 1: -dx/2,  dy/2
    # 2:  dx/2,  dy/2
    # 3:  dx/2, -dy/2
    # Note: this order depends on definition, but consistent order matters for polygon
    
    zeros = torch.zeros_like(dx)
    
    # We use (N, 4, 3) logic but ignore Z
    # Canonical corners (N, 4, 2)
    # x signs: - - + +
    # y signs: - + + -
    x_signs = torch.tensor([-0.5, -0.5, 0.5, 0.5], device=boxes.device, dtype=boxes.dtype)
    y_signs = torch.tensor([-0.5, 0.5, 0.5, -0.5], device=boxes.device, dtype=boxes.dtype)
    
    # (N, 4)
    l = dx.unsqueeze(1) * x_signs
    w = dy.unsqueeze(1) * y_signs
    
    # Rotate
    cosa = torch.cos(heading)
    sina = torch.sin(heading)
    
    # (N, 1)
    cosa = cosa.unsqueeze(1)
    sina = sina.unsqueeze(1)
    
    # Rotated offsets
    # x_rot = x * cos - y * sin
    # y_rot = x * sin + y * cos
    r_x = l * cosa - w * sina
    r_y = l * sina + w * cosa
    
    # Add center
    corners_x = x.unsqueeze(1) + r_x
    corners_y = y.unsqueeze(1) + r_y
    
    corners = torch.stack([corners_x, corners_y], dim=-1) # (N, 4, 2)
    return corners

def poly_area(poly):
    """
    Calculate area of a convex polygon (N, 2)
    Shoelace formula
    """
    if len(poly) < 3: return 0.0
    # shift
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs(torch.dot(x, torch.roll(y, 1)) - torch.dot(y, torch.roll(x, 1)))

def intersect_poly(poly1, poly2):
    """
    Sutherland-Hodgman Polygon Clipping (Convex)
    poly1: (M, 2) clipper
    poly2: (N, 2) subject
    Returns area of intersection
    """
    # Converting to list for dynamic size
    def inside(cp1, cp2, p):
        # NOTE: Using '<' for clockwise winding (boxes_to_bev_corners outputs CW)
        return (cp2[0]-cp1[0])*(p[1]-cp1[1]) < (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def intersection(cp1, cp2, s, e):
        dc = [cp1[0]-cp2[0], cp1[1]-cp2[1]]
        dp = [s[0]-e[0], s[1]-e[1]]
        n1 = cp1[0]*cp2[1] - cp1[1]*cp2[0]
        n2 = s[0]*e[1] - s[1]*e[0]
        n3 = 1.0 / (dc[0]*dp[1] - dc[1]*dp[0] + 1e-6)
        return torch.tensor([(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3], device=poly1.device)

    outputList = poly2
    cp1 = poly1[-1]

    for clipVertex in poly1:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        if len(inputList) == 0:
            break
            
        s = inputList[-1]

        for e in inputList:
            if inside(cp1, cp2, e):
                if not inside(cp1, cp2, s):
                    outputList.append(intersection(cp1, cp2, s, e))
                outputList.append(e)
            elif inside(cp1, cp2, s):
                outputList.append(intersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
    
    if len(outputList) < 3:
        return 0.0
        
    # Calculate area of intersection polygon
    # Convert list of tensors to single tensor
    poly = torch.stack(outputList)
    return poly_area(poly)


def boxes_overlap_bev_cpu(boxes_a, boxes_b):
    """
    Calculate BEV intersection area. (Pure Python/PyTorch, Optimized with AABB check)
    Args:
        boxes_a: (N, 7)
        boxes_b: (M, 7)
    Returns:
        overlap: (N, M)
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    
    # Check device
    device = boxes_a.device
    
    overlaps = torch.zeros((N, M), device=device)
    
    # --- 1. Fast Vectorized AABB Check (On Original Device) ---
    # Radius check is good, AABB check is better and fully vectorizable
    
    # Calculate global AABB for each rotated box
    rad_a = ((boxes_a[:, 3]**2 + boxes_a[:, 4]**2).sqrt() / 2).unsqueeze(1) # (N, 1)
    rad_b = ((boxes_b[:, 3]**2 + boxes_b[:, 4]**2).sqrt() / 2).unsqueeze(0) # (1, M)
    
    center_a = boxes_a[:, :2].unsqueeze(1) # (N, 1, 2)
    center_b = boxes_b[:, :2].unsqueeze(0) # (1, M, 2)
    
    dist = torch.norm(center_a - center_b, dim=2) # (N, M)
    
    # Potential overlap mask
    possible_overlap = dist < (rad_a + rad_b)
    
    # If no overlaps possible at all, return
    if not possible_overlap.any():
        return overlaps
        
    # --- 2. Precise Polygon Intersection (On CPU) ---
    # Sutherland-Hodgman is sequential and branch-heavy.
    # Running it on CUDA scalars causes massive kernel launch latency and sync.
    # We MUST move candidates to CPU for this loop.
    
    # Find indices where overlap is possible
    # (n_idx, m_idx)
    n_indices, m_indices = torch.where(possible_overlap)
    
    if len(n_indices) == 0:
        return overlaps
        
    # Move relevant boxes to CPU for geometry calc
    # Converting the entire set is usually faster than slicing thousands of times on GPU
    # But if N is huge and candidates small, slicing first might be better. 
    # With AABB filter, candidates are sparse.
    
    # We need the corners.
    corners_a_gpu = boxes_to_bev_corners(boxes_a) # (N, 4, 2)
    corners_b_gpu = boxes_to_bev_corners(boxes_b) # (M, 4, 2)
    
    # Only transfer the specific polygons we need?
    # Or determining unique indices to transfer?
    # Simple approach: Transfer only the overlapping pairs' polygons?
    # Actually, slicing `corners_a[n_indices]` creates a new tensor of size K.
    # Transferring K polygons to CPU is one copy.
    
    poly_a_cpu = corners_a_gpu[n_indices].cpu()  # (K, 4, 2)
    poly_b_cpu = corners_b_gpu[m_indices].cpu()  # (K, 4, 2)
    
    # Iterate on CPU
    overlap_values = []
    for k in range(len(n_indices)):
        poly1 = poly_a_cpu[k]
        poly2 = poly_b_cpu[k]
        
        inter_area = intersect_poly(poly1, poly2)
        overlap_values.append(inter_area)
    
    # Move results back to GPU
    overlap_values = torch.tensor(overlap_values, device=device, dtype=overlaps.dtype)
    
    # Scatter into dense matrix
    overlaps[n_indices, m_indices] = overlap_values
            
    return overlaps

def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    API Match for OpenPCDet (CPU version).
    Args:
        boxes_a: (N, 7)
        boxes_b: (M, 7)
    Returns:
        iou3d: (N, M)
    """
    boxes_a, _ = check_numpy_to_torch(boxes_a)
    boxes_b, _ = check_numpy_to_torch(boxes_b)
    
    assert boxes_a.dim() == 2 and boxes_a.shape[1] == 7
    assert boxes_b.dim() == 2 and boxes_b.shape[1] == 7
    
    # Height overlap
    # (N, 1)
    max_h_a = boxes_a[:, 2] + boxes_a[:, 5] / 2
    min_h_a = boxes_a[:, 2] - boxes_a[:, 5] / 2
    # (1, M)
    max_h_b = boxes_b[:, 2] + boxes_b[:, 5] / 2
    min_h_b = boxes_b[:, 2] - boxes_b[:, 5] / 2
    
    # Expand
    max_of_min = torch.max(min_h_a.view(-1, 1), min_h_b.view(1, -1))
    min_of_max = torch.min(max_h_a.view(-1, 1), max_h_b.view(1, -1))
    overlap_h = torch.clamp(min_of_max - max_of_min, min=0)
    
    # BEV Overlap
    overlap_bev = boxes_overlap_bev_cpu(boxes_a, boxes_b)
    
    overlap_3d = overlap_bev * overlap_h
    
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    
    iou3d = overlap_3d / torch.clamp(vol_a + vol_b - overlap_3d, min=1e-6)
    
    return iou3d

def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    NMS on CPU (API Match).
    """
    boxes, _ = check_numpy_to_torch(boxes)
    scores, _ = check_numpy_to_torch(scores)
    
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
        
    boxes = boxes[order]
    scores = scores[order]
    
    keep = []
    
    # Greedy NMS
    # Since boxes_iou3d_gpu is O(N*M), and we want to select top, 
    # we can do standard NMS loop.
    
    # If N is large, this is slow. 
    # Optimize: only compute IoU with kept boxes
    
    suppressed = torch.zeros(len(boxes), dtype=torch.bool)
    
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
            
        keep.append(i)
        
        # Check IoU with remaining
        if i + 1 >= len(boxes):
            break
            
        # Get candidate boxes (all remaining)
        candidates = boxes[i+1:]
        current_box = boxes[i:i+1] # (1, 7)
        
        # Calc BEV IoU (Issue 6 Fix: Use rotated BEV IoU for NMS)
        overlap_bev = boxes_overlap_bev_cpu(current_box, candidates)
        vol_a_bev = (current_box[:, 3] * current_box[:, 4]).view(-1, 1)
        vol_b_bev = (candidates[:, 3] * candidates[:, 4]).view(1, -1)
        iou_bev = overlap_bev / torch.clamp(vol_a_bev + vol_b_bev - overlap_bev, min=1e-6)
        iou = iou_bev.view(-1)
        
        # Suppress
        remove = iou > thresh
        
        # Map back indices
        suppressed[i+1:] = suppressed[i+1:] | remove
        
    keep = torch.tensor(keep, dtype=torch.long)
    return order[keep], None
