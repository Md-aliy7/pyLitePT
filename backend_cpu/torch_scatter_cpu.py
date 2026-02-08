"""
Torch Scatter CPU Fallback
===========================
Pure PyTorch implementation of torch_scatter operations.
This provides CPU-compatible scatter operations without CUDA.
"""

import torch


def segment_csr(src, indptr, reduce="mean"):
    """
    Segment reduction using CSR (Compressed Sparse Row) format.
    
    This is a CPU-compatible replacement for torch_scatter.segment_csr.
    
    Args:
        src: [N, D] - Source tensor
        indptr: [M+1] - Index pointer array (CSR format)
                indptr[i] to indptr[i+1] defines the range for segment i
        reduce: Reduction operation ("mean", "sum", "max", "min")
        
    Returns:
        out: [M, D] - Reduced tensor
    
    Example:
        If indptr = [0, 3, 5, 8], then:
        - Segment 0: src[0:3]
        - Segment 1: src[3:5]
        - Segment 2: src[5:8]
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
    out = torch.zeros(num_segments, D, device=device, dtype=dtype)
    
    for i in range(num_segments):
        start = indptr[i].item() if isinstance(indptr[i], torch.Tensor) else indptr[i]
        end = indptr[i+1].item() if isinstance(indptr[i+1], torch.Tensor) else indptr[i+1]
        
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
            else:
                raise ValueError(f"Unknown reduce: {reduce}")
    
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
    
    # Ensure dtype matches for mixed precision compatibility
    return out.scatter_add_(dim, index.expand_as(src), src.to(out.dtype))


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    """
    Scatter mean operation.
    
    Args:
        src: Source tensor
        index: Index tensor  
        dim: Dimension along which to scatter
        out: Optional output tensor
        dim_size: Size of output dimension
        
    Returns:
        Mean of scattered values
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    # Compute sum
    shape = list(src.shape)
    shape[dim] = dim_size
    out_sum = torch.zeros(shape, dtype=src.dtype, device=src.device)
    # Ensure dtype matches for mixed precision compatibility
    out_sum = out_sum.scatter_add_(dim, index.expand_as(src), src.to(out_sum.dtype))
    
    # Compute count
    ones = torch.ones_like(src)
    count = torch.zeros(shape, dtype=src.dtype, device=src.device)
    count = count.scatter_add_(dim, index.expand_as(ones), ones)
    
    # Avoid division by zero
    count = count.clamp(min=1)
    
    return out_sum / count


def scatter_max(src, index, dim=0, out=None, dim_size=None, fill_value=float('-inf')):
    """
    Scatter max operation.
    
    Args:
        src: Source tensor
        index: Index tensor
        dim: Dimension along which to scatter
        out: Optional output tensor
        dim_size: Size of output dimension
        fill_value: Fill value for empty slots
        
    Returns:
        tuple: (max_values, argmax_indices)
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    out = torch.full(shape, fill_value, dtype=src.dtype, device=src.device)
    
    
    # Optimized implementation using pytorch 1.12+ scatter_reduce_ if available
    if hasattr(out, 'scatter_reduce_'):
        out.scatter_reduce_(dim, index.expand_as(src), src.to(out.dtype), reduce='amax', include_self=False)
        # Handle fill_value for indices that were not updated (stayed at fill_value)
        # Note: scatter_reduce_ with include_self=False and init value is correct
        return out, None
    else:
        # Fallback for older pytorch
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
