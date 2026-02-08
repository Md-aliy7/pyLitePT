"""
=============================================================================
POINT OPERATIONS - Pure Python/PyTorch CPU Fallback
=============================================================================
Efficient implementations of point cloud operations using scipy and PyTorch.
Replaces CUDA pointops when not available.

All functions are vectorized for maximum performance.
"""

import torch
import numpy as np

# Try to import scipy for efficient KNN
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using fallback KNN (slower).")


def knn_query(k, query_coord, query_offset, support_coord, support_offset):
    """
    K-Nearest Neighbors query using scipy.spatial.cKDTree.
    
    This is a pure Python/scipy replacement for pointops.knn_query.
    Uses scipy.spatial.cKDTree for O(N log N) performance.
    
    Args:
        k: int, number of neighbors to query
        query_coord: (N, 3) query points
        query_offset: (B,) cumulative point counts per batch [n1, n1+n2, ...]
        support_coord: (M, 3) support points (point cloud to search)
        support_offset: (B,) cumulative point counts per batch for support
        
    Returns:
        idx: (N, k) indices of k-nearest neighbors in support_coord
        dist: (N, k) squared distances to k-nearest neighbors
    """
    device = query_coord.device
    dtype = query_coord.dtype
    
    query_np = query_coord.cpu().numpy()
    support_np = support_coord.cpu().numpy()
    query_off = query_offset.cpu().numpy() if isinstance(query_offset, torch.Tensor) else query_offset
    support_off = support_offset.cpu().numpy() if isinstance(support_offset, torch.Tensor) else support_offset
    
    n = query_np.shape[0]
    
    # Initialize output
    all_idx = np.zeros((n, k), dtype=np.int64)
    all_dist = np.zeros((n, k), dtype=np.float32)
    
    # Process each batch separately
    batch_size = len(query_off)
    q_start = 0
    s_start = 0
    
    for b in range(batch_size):
        q_end = int(query_off[b])
        s_end = int(support_off[b])
        
        query_batch = query_np[q_start:q_end]
        support_batch = support_np[s_start:s_end]
        
        if len(query_batch) == 0 or len(support_batch) == 0:
            q_start = q_end
            s_start = s_end
            continue
        
        # Build KD-Tree for support points
        if SCIPY_AVAILABLE:
            tree = cKDTree(support_batch)
            # Query k-nearest neighbors
            k_actual = min(k, len(support_batch))
            dist, idx = tree.query(query_batch, k=k_actual, workers=-1)
            
            # Handle case where k_actual < k (pad with last valid index)
            if k_actual < k:
                dist_padded = np.zeros((len(query_batch), k), dtype=np.float32)
                idx_padded = np.zeros((len(query_batch), k), dtype=np.int64)
                dist_padded[:, :k_actual] = dist
                idx_padded[:, :k_actual] = idx
                # Pad with last valid index
                if k_actual > 0:
                    dist_padded[:, k_actual:] = dist[:, -1:]
                    idx_padded[:, k_actual:] = idx[:, -1:]
                dist = dist_padded
                idx = idx_padded
        else:
            # Fallback: brute force (very slow for large point clouds)
            dist_matrix = np.linalg.norm(
                query_batch[:, None, :] - support_batch[None, :, :], axis=2
            )  # (Q, S)
            k_actual = min(k, len(support_batch))
            idx = np.argsort(dist_matrix, axis=1)[:, :k]
            dist = np.take_along_axis(dist_matrix, idx, axis=1)
            
            # Pad if needed
            if k_actual < k:
                idx = np.pad(idx, ((0, 0), (0, k - k_actual)), mode='edge')
                dist = np.pad(dist, ((0, 0), (0, k - k_actual)), mode='edge')
        
        # Map local indices to global (add batch offset)
        all_idx[q_start:q_end, :] = idx + s_start
        all_dist[q_start:q_end, :] = dist ** 2  # Return squared distances
        
        q_start = q_end
        s_start = s_end
    
    return (
        torch.from_numpy(all_idx).to(device=device, dtype=torch.long),
        torch.from_numpy(all_dist).to(device=device, dtype=dtype)
    )


def ball_query(radius, nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt):
    """
    Ball query - find points within radius of each query point.
    
    Args:
        radius: float, search radius
        nsample: int, max number of points to return per query
        xyz: (M, 3) support points
        xyz_batch_cnt: (B,) number of points per batch
        new_xyz: (N, 3) query points
        new_xyz_batch_cnt: (B,) number of query points per batch
        
    Returns:
        idx: (N, nsample) indices of points within radius, -1 for invalid
    """
    device = new_xyz.device
    N = new_xyz.shape[0]
    
    # Use KNN then filter by radius
    # Compute cumulative offsets
    offset_xyz = torch.cumsum(xyz_batch_cnt, dim=0)
    offset_new = torch.cumsum(new_xyz_batch_cnt, dim=0)
    
    # Query more neighbors than needed, then filter
    k_query = min(nsample * 2, xyz.shape[0])
    idx, dist_sq = knn_query(k_query, new_xyz, offset_new, xyz, offset_xyz)
    
    # Filter by radius (squared comparison for efficiency)
    radius_sq = radius ** 2
    valid = dist_sq <= radius_sq
    
    # Build output: keep first nsample valid indices per query
    result = torch.full((N, nsample), -1, dtype=torch.long, device=device)
    
    for i in range(N):
        valid_mask = valid[i]
        valid_idx = idx[i][valid_mask]
        n_valid = min(len(valid_idx), nsample)
        if n_valid > 0:
            result[i, :n_valid] = valid_idx[:n_valid]
    
    return result


def grouping(features, idx):
    """
    Group features by indices.
    
    Args:
        features: (M, C) point features
        idx: (N, K) indices to group
        
    Returns:
        grouped_features: (N, K, C) grouped features
    """
    # Handle -1 indices (invalid)
    idx_clamped = idx.clamp(min=0)
    grouped = features[idx_clamped]  # (N, K, C)
    
    # Zero out invalid positions
    invalid_mask = (idx == -1).unsqueeze(-1)
    grouped = grouped.masked_fill(invalid_mask, 0.0)
    
    return grouped


def interpolation(features, idx, weight):
    """
    Interpolate features using indices and weights.
    
    Args:
        features: (M, C) point features
        idx: (N, 3) indices of 3 nearest points
        weight: (N, 3) interpolation weights
        
    Returns:
        interp_features: (N, C) interpolated features
    """
    # Gather features
    gathered = features[idx]  # (N, 3, C)
    
    # Weighted sum
    weight = weight.unsqueeze(-1)  # (N, 3, 1)
    interp = (gathered * weight).sum(dim=1)  # (N, C)
    
    return interp


def furthest_point_sample(xyz, npoint):
    """
    Furthest Point Sampling.
    
    Iteratively select the point that is furthest from the already selected set.
    
    Args:
        xyz: (B, N, 3) input points
        npoint: int, number of points to sample
        
    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Random first point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)  # (B,)
    
    return centroids
