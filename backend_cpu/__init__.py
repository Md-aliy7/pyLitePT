"""
LitePT CPU Backend Package
==========================

Pure Python/PyTorch implementations of specialized CUDA operators.
Used as automatic fallbacks when running on CPU-only or Windows environments.
Loaded dynamically by `hybrid_backend.py`.

Contents:
- spconv_cpu: Sparse Convolution (vectorized dense conversion)
- torch_scatter_cpu: Scatter/Gather with vectorized segment_csr (scatter_reduce_)
- attention_cpu: SDPA-backed attention (PyTorch 2.0+) with JIT fallback
- pointops: Point Cloud operations (KNN, vectorized Ball Query, Grouping)

Also used via hybrid_backend:
- libs/pointrope.py: Real 3D Rotary Position Embeddings (vectorized, JIT-compiled)

Optimizations (2026-02):
- All Python loops replaced with vectorized PyTorch operations
- PyTorch 2.0+ scaled_dot_product_attention as primary attention path
- Multi-threaded CPU execution (auto-detects cores, TORCH_NUM_THREADS)
"""
