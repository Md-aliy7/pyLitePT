"""
Torch Scatter CPU Fallback
===========================
Optimized pure PyTorch implementation of torch_scatter operations.
Uses vectorized scatter operations instead of Python loops.
"""

import torch


def segment_csr(src, indptr, reduce="mean"):
    """
    Segment reduction using CSR (Compressed Sparse Row) format.
    
    Vectorized implementation using scatter_reduce_ (PyTorch 1.12+)
    with Python loop fallback.
    
    Args:
        src: [N, D] - Source tensor
        indptr: [M+1] - Index pointer array (CSR format)
        reduce: Reduction operation ("mean", "sum", "max", "min")
        
    Returns:
        out: [M, D] - Reduced tensor
    """
    num_segments = len(indptr) - 1
    device = src.device
    dtype = src.dtype
    
    if src.dim() == 1:
        src = src.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    D = src.shape[1]
    
    if num_segments == 0:
        out = torch.zeros(0, D, device=device, dtype=dtype)
        return out.squeeze(1) if squeeze_output else out
    
    # Convert indptr to proper tensor
    if not isinstance(indptr, torch.Tensor):
        indptr = torch.tensor(indptr, device=device, dtype=torch.long)
    else:
        indptr = indptr.long()
    
    # Build segment IDs from indptr: [0,0,0, 1,1, 2,2,2, ...]
    counts = indptr[1:] - indptr[:-1]  # [M]
    
    # Check for empty segments
    total = counts.sum().item()
    if total == 0:
        out = torch.zeros(num_segments, D, device=device, dtype=dtype)
        return out.squeeze(1) if squeeze_output else out
    
    seg_ids = torch.repeat_interleave(
        torch.arange(num_segments, device=device), counts
    )  # [N]
    
    # Map reduce names for scatter_reduce_
    reduce_map = {"sum": "sum", "mean": "mean", "max": "amax", "min": "amin"}
    scatter_reduce = reduce_map.get(reduce)
    
    if scatter_reduce is not None and hasattr(torch.Tensor, 'scatter_reduce_'):
        # Vectorized path using scatter_reduce_ (PyTorch 1.12+)
        if reduce in ("max", "min"):
            fill = float('-inf') if reduce == "max" else float('inf')
            out = torch.full((num_segments, D), fill, device=device, dtype=dtype)
        else:
            out = torch.zeros(num_segments, D, device=device, dtype=dtype)
        
        idx_expanded = seg_ids.unsqueeze(1).expand_as(src)
        out.scatter_reduce_(0, idx_expanded, src, reduce=scatter_reduce, include_self=False)
        
        # Handle empty segments for max/min (replace inf with 0)
        if reduce in ("max", "min"):
            empty_mask = counts == 0
            if empty_mask.any():
                out[empty_mask] = 0.0
    else:
        # Fallback for older PyTorch
        out = torch.zeros(num_segments, D, device=device, dtype=dtype)
        for i in range(num_segments):
            start = indptr[i].item()
            end = indptr[i + 1].item()
            if end > start:
                segment = src[start:end]
                if reduce == "mean":
                    out[i] = segment.mean(dim=0)
                elif reduce == "sum":
                    out[i] = segment.sum(dim=0)
                elif reduce == "max":
                    out[i] = segment.max(dim=0)[0]
                elif reduce == "min":
                    out[i] = segment.min(dim=0)[0]
    
    if squeeze_output:
        out = out.squeeze(1)
    
    return out


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    """
    Scatter add operation.
    
    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension along which to scatter
        out: Optional output tensor
        dim_size: Size of output dimension
        
    Returns:
        Scattered tensor
    """
    if out is None:
        if dim_size is None:
            dim_size = index.max().item() + 1
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    return out.scatter_add_(dim, index.expand_as(src), src.to(out.dtype))


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    """
    Scatter mean operation.
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    out_sum = torch.zeros(shape, dtype=src.dtype, device=src.device)
    out_sum = out_sum.scatter_add_(dim, index.expand_as(src), src.to(out_sum.dtype))
    
    ones = torch.ones_like(src)
    count = torch.zeros(shape, dtype=src.dtype, device=src.device)
    count = count.scatter_add_(dim, index.expand_as(ones), ones)
    count = count.clamp(min=1)
    
    return out_sum / count


def scatter_max(src, index, dim=0, out=None, dim_size=None, fill_value=float('-inf')):
    """
    Scatter max operation.
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    out = torch.full(shape, fill_value, dtype=src.dtype, device=src.device)
    
    if hasattr(out, 'scatter_reduce_'):
        out.scatter_reduce_(dim, index.expand_as(src), src.to(out.dtype), reduce='amax', include_self=False)
        return out, None
    else:
        for i in range(dim_size):
            mask = index == i
            if mask.any():
                out[i] = src[mask].max(dim=0)[0] if mask.sum() > 0 else fill_value
        return out, None


def scatter_min(src, index, dim=0, out=None, dim_size=None, fill_value=float('inf')):
    """
    Scatter min operation.
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    out = torch.full(shape, fill_value, dtype=src.dtype, device=src.device)

    if hasattr(out, 'scatter_reduce_'):
        out.scatter_reduce_(dim, index.expand_as(src), src.to(out.dtype), reduce='amin', include_self=False)
        return out, None
    else:
        for i in range(dim_size):
            mask = index == i
            if mask.any():
                out[i] = src[mask].min(dim=0)[0] if mask.sum() > 0 else fill_value
        return out, None


# Module-level exports
__all__ = [
    'segment_csr',
    'scatter_add',
    'scatter_mean', 
    'scatter_max',
    'scatter_min',
]
