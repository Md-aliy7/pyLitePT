"""
CPU-Compatible Attention for LitePT
====================================
Optimized pure PyTorch attention implementation to replace Flash Attention on CPU.
Uses torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+) as primary
path, with JIT-compiled manual attention as fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Check for PyTorch 2.0+ SDPA support
_HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')


@torch.jit.script
def _attention_core_jit(q_seq: torch.Tensor, k_seq: torch.Tensor, v_seq: torch.Tensor,
                        softmax_scale: float) -> torch.Tensor:
    """
    JIT-compiled attention core for maximum CPU performance.

    Args:
        q_seq, k_seq, v_seq: [seq_len, H, D]
        softmax_scale: float

    Returns:
        output: [seq_len, H, D]
    """
    # Transpose to [H, seq_len, D]
    q_t = q_seq.transpose(0, 1)
    k_t = k_seq.transpose(0, 1)
    v_t = v_seq.transpose(0, 1)

    # Compute attention: [H, seq_len, seq_len]
    attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale

    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Weighted sum: [H, seq_len, D]
    attn_output = torch.matmul(attn_weights, v_t)

    # Transpose back to [seq_len, H, D]
    return attn_output.transpose(0, 1)


def attention_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None):
    """
    Variable-length attention using packed QKV format.

    This is a CPU-compatible replacement for flash_attn.flash_attn_varlen_qkvpacked_func.

    Args:
        qkv: [total_N, 3, H, D] - Packed Q, K, V for all sequences
        cu_seqlens: [B+1] - Cumulative sequence lengths (int32)
        max_seqlen: Maximum sequence length in batch
        dropout_p: Dropout probability (ignored for inference)
        softmax_scale: Scale factor for attention (default: 1/sqrt(D))

    Returns:
        output: [total_N, H, D] - Attention output
    """
    total_N, _, H, D = qkv.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # Split into Q, K, V: each [total_N, H, D]
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

    # Output buffer
    output = torch.zeros(total_N, H, D, dtype=qkv.dtype, device=qkv.device)

    # Process each sequence
    num_seqs = len(cu_seqlens) - 1

    if _HAS_SDPA and num_seqs <= 32:
        # SDPA path: pad sequences into a batch for efficient processing
        # This avoids the Python loop for small batch counts
        seq_starts = cu_seqlens[:-1].long()
        seq_ends = cu_seqlens[1:].long()
        seq_lens = seq_ends - seq_starts

        for i in range(num_seqs):
            start = seq_starts[i].item()
            end = seq_ends[i].item()
            if end <= start:
                continue

            # [seq_len, H, D] -> [H, seq_len, D] (SDPA expects batch-first or we use 3D)
            q_seq = q[start:end].transpose(0, 1)  # [H, seq_len, D]
            k_seq = k[start:end].transpose(0, 1)
            v_seq = v[start:end].transpose(0, 1)

            # Use SDPA - internally optimized with C++ kernels
            attn_out = F.scaled_dot_product_attention(
                q_seq, k_seq, v_seq,
                dropout_p=dropout_p if qkv.requires_grad else 0.0,
                scale=softmax_scale,
            )  # [H, seq_len, D]

            output[start:end] = attn_out.transpose(0, 1)  # [seq_len, H, D]
    else:
        # JIT fallback path
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            if end <= start:
                continue

            q_seq = q[start:end]
            k_seq = k[start:end]
            v_seq = v[start:end]

            output[start:end] = _attention_core_jit(q_seq, k_seq, v_seq, softmax_scale)

    return output


# Module wrapper
class VariableLengthAttention(nn.Module):
    """Variable-length attention module for CPU."""

    def __init__(self, use_chunked=False, chunk_size=256):
        super().__init__()
        self.use_chunked = use_chunked
        self.chunk_size = chunk_size

    def forward(self, qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None):
        return attention_varlen_qkvpacked(
            qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale
        )


# Flash attention compatible interface
class FlashAttnCPU:
    """CPU-compatible flash attention interface."""

    @staticmethod
    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                          dropout_p=0.0, softmax_scale=None):
        """Drop-in replacement for flash_attn.flash_attn_varlen_qkvpacked_func."""
        return attention_varlen_qkvpacked(
            qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale
        )


# Mock flash_attn module
flash_attn_varlen_qkvpacked_func = FlashAttnCPU.flash_attn_varlen_qkvpacked_func
