"""
PointROPE - Pure Python/PyTorch Implementation
================================================
3D Rotary Position Embeddings for point cloud transformers.

This is a faithful translation of the CUDA kernel in LitePT-main/libs/pointrope/kernels.cu
into fully vectorized PyTorch operations (no Python loops over tokens).

The CUDA kernel applies a 3D rotation to token embeddings based on spatial position:
  - Each head dimension D is split into 6 equal parts: [u_X, v_X, u_Y, v_Y, u_Z, v_Z]
  - Q = D/6 frequency components per spatial axis
  - For each axis (X,Y,Z): freq = pos * inv_freq, then rotate: u' = u*cos - v*sin, v' = v*cos + u*sin
"""

import torch
import torch.nn as nn


@torch.jit.script
def _pointrope_apply(tokens: torch.Tensor, positions: torch.Tensor, 
                     base: float, fwd: float) -> torch.Tensor:
    """
    Vectorized PointROPE core.
    
    Args:
        tokens: [B, N, H, D] - token embeddings (B=1 typically)
        positions: [B, N, 3] - integer grid coordinates (X, Y, Z)
        base: frequency base (e.g. 100.0)
        fwd: 1.0 for forward, -1.0 for backward
        
    Returns:
        tokens: [B, N, H, D] - rotated tokens (in-place modification)
    """
    B, N, H, D = tokens.shape
    Q = D // 6  # number of frequency components per axis
    rotated_dim = 6 * Q  # dimensions that get rotated
    
    # If Q == 0, no rotation is possible — return tokens unchanged
    if Q == 0:
        return tokens
    
    # Compute inverse frequencies: fwd / base^(i/Q) for i = 0..Q-1
    # Shape: [Q]
    i = torch.arange(Q, device=tokens.device, dtype=tokens.dtype)
    inv_freq = fwd / torch.pow(base, i / float(Q))  # [Q]
    
    # Compute rotation angles for each spatial dim (X, Y, Z)
    # positions: [B, N, 3] -> pos_x/y/z: [B, N, 1]
    # inv_freq: [Q] -> [1, 1, Q]
    # freq_x/y/z: [B, N, Q]
    inv_freq = inv_freq.unsqueeze(0).unsqueeze(0)  # [1, 1, Q]
    
    freq_x = positions[:, :, 0:1].to(tokens.dtype) * inv_freq  # [B, N, Q]
    freq_y = positions[:, :, 1:2].to(tokens.dtype) * inv_freq  # [B, N, Q]
    freq_z = positions[:, :, 2:3].to(tokens.dtype) * inv_freq  # [B, N, Q]
    
    cos_x = torch.cos(freq_x)  # [B, N, Q]
    sin_x = torch.sin(freq_x)
    cos_y = torch.cos(freq_y)
    sin_y = torch.sin(freq_y)
    cos_z = torch.cos(freq_z)
    sin_z = torch.sin(freq_z)
    
    # Expand for all heads: [B, N, 1, Q] -> broadcasts with [B, N, H, Q]
    cos_x = cos_x.unsqueeze(2)
    sin_x = sin_x.unsqueeze(2)
    cos_y = cos_y.unsqueeze(2)
    sin_y = sin_y.unsqueeze(2)
    cos_z = cos_z.unsqueeze(2)
    sin_z = sin_z.unsqueeze(2)
    
    # Split tokens into 6 parts: [u_X, v_X, u_Y, v_Y, u_Z, v_Z]
    # Each part has Q elements along the last dimension
    u_x = tokens[:, :, :, 0*Q:1*Q]
    v_x = tokens[:, :, :, 1*Q:2*Q]
    u_y = tokens[:, :, :, 2*Q:3*Q]
    v_y = tokens[:, :, :, 3*Q:4*Q]
    u_z = tokens[:, :, :, 4*Q:5*Q]
    v_z = tokens[:, :, :, 5*Q:6*Q]
    
    # Apply rotation for each axis
    # u' = u*cos - v*sin
    # v' = v*cos + u*sin
    new_u_x = u_x * cos_x - v_x * sin_x
    new_v_x = v_x * cos_x + u_x * sin_x
    new_u_y = u_y * cos_y - v_y * sin_y
    new_v_y = v_y * cos_y + u_y * sin_y
    new_u_z = u_z * cos_z - v_z * sin_z
    new_v_z = v_z * cos_z + u_z * sin_z
    
    # Reassemble rotated dimensions: [B, N, H, rotated_dim]
    rotated = torch.cat([new_u_x, new_v_x, new_u_y, new_v_y, new_u_z, new_v_z], dim=-1)
    
    # Pass through remaining dimensions (D - rotated_dim) unchanged
    if rotated_dim < D:
        remainder = tokens[:, :, :, rotated_dim:]
        return torch.cat([rotated, remainder], dim=-1)
    
    return rotated


class PointROPE_func(torch.autograd.Function):
    """Autograd Function for PointROPE with proper backward pass."""
    
    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1.0):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        return _pointrope_apply(tokens, positions, base, F0)
    
    @staticmethod
    def backward(ctx, grad_res):
        positions = ctx.saved_tensors[0]
        base = ctx.saved_base
        F0 = ctx.saved_F0
        # Backward is same rotation with negated F0
        grad_rotated = _pointrope_apply(grad_res.contiguous(), positions, base, -F0)
        return grad_rotated, None, None, None


class PointROPE(nn.Module):
    """
    3D Rotary Position Embeddings for point cloud transformers.
    
    Matches the interface of the CUDA PointROPE:
        forward(tokens, positions) where:
            tokens: [B, H, N, D] (note: H and N are transposed vs CUDA kernel)
            positions: [B, N, 3]
    
    The CUDA kernel expects [B, N, H, D] internally and the Python wrapper
    transposes (1,2) before calling. We replicate this exactly.
    """
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
    
    def forward(self, tokens, positions):
        """
        Args:
            tokens: [B, H, N, D] - input tokens
            positions: [B, N, 3] - grid coordinates
            
        Returns:
            tokens: [B, H, N, D] - rotated tokens
        """
        # Transpose to [B, N, H, D] for the kernel (matching CUDA layout)
        tokens_t = tokens.transpose(1, 2).contiguous()
        result = PointROPE_func.apply(tokens_t, positions, self.base, self.F0)
        # Transpose back to [B, H, N, D]
        return result.transpose(1, 2)
