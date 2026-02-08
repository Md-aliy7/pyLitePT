"""
CPU-Compatible Attention for LitePT
====================================
Pure PyTorch attention implementation to replace Flash Attention on CPU.
Supports variable-length sequences using masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    
    # Process each sequence separately
    num_seqs = len(cu_seqlens) - 1
    
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        
        if seq_len == 0:
            continue
        
        # Extract sequence: [seq_len, H, D]
        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]
        
        # Use optimized JIT attention kernel
        attn_output = _attention_core_jit(q_seq, k_seq, v_seq, softmax_scale)
        
        # Store
        output[start:end] = attn_output
    
    return output


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


def attention_varlen_qkvpacked_chunked(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, 
                                        softmax_scale=None, chunk_size=256):
    """
    Memory-efficient chunked attention for large sequences.
    
    Processes attention in chunks to reduce peak memory usage.
    
    Args:
        qkv: [total_N, 3, H, D] - Packed Q, K, V
        cu_seqlens: [B+1] - Cumulative sequence lengths
        max_seqlen: Maximum sequence length
        dropout_p: Dropout probability
        softmax_scale: Scale factor
        chunk_size: Process queries in chunks of this size
        
    Returns:
        output: [total_N, H, D]
    """
    total_N, _, H, D = qkv.shape
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    output = torch.zeros(total_N, H, D, dtype=qkv.dtype, device=qkv.device)
    
    num_seqs = len(cu_seqlens) - 1
    
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        
        if seq_len == 0:
            continue
        
        # Get full K, V for this sequence: [seq_len, H, D]
        k_seq = k[start:end]
        v_seq = v[start:end]
        
        # Process Q in chunks
        for q_start in range(0, seq_len, chunk_size):
            q_end = min(q_start + chunk_size, seq_len)
            
            # Query chunk: [chunk_len, H, D]
            q_chunk = q[start + q_start:start + q_end]
            
            # Transpose for matmul: [H, chunk_len, D] @ [H, D, seq_len] -> [H, chunk_len, seq_len]
            q_t = q_chunk.transpose(0, 1)
            k_t = k_seq.transpose(0, 1)
            v_t = v_seq.transpose(0, 1)
            
            attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            if dropout_p > 0 and qkv.requires_grad:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            
            attn_output = torch.matmul(attn_weights, v_t)
            attn_output = attn_output.transpose(0, 1)
            
            output[start + q_start:start + q_end] = attn_output
    
    return output


# Module wrapper
class VariableLengthAttention(nn.Module):
    """Variable-length attention module for CPU."""
    
    def __init__(self, use_chunked=True, chunk_size=256):
        super().__init__()
        self.use_chunked = use_chunked
        self.chunk_size = chunk_size
    
    def forward(self, qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None):
        if self.use_chunked:
            return attention_varlen_qkvpacked_chunked(
                qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, self.chunk_size
            )
        else:
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
        return attention_varlen_qkvpacked_chunked(
            qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale
        )


# Mock flash_attn module
flash_attn_varlen_qkvpacked_func = FlashAttnCPU.flash_attn_varlen_qkvpacked_func
