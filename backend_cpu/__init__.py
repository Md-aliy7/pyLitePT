"""
LitePT CPU Backend Package
==========================

This package contains pure Python/PyTorch implementations of specialized
CUDA operators used in 3D Deep Learning. These are used as fallbacks
when running on Windows or CPU-only environments.

Contents:
- spconv_cpu: Sparse Convolution implementation
- torch_scatter_cpu: Scatter/Gather operations
- attention_cpu: FlashAttention-v2 compatible implementation
- pointops: Point Cloud operations (KNN, Ball Query, Grouping)

These modules are typically loaded dynamically by `hybrid_backend.py`.
"""
