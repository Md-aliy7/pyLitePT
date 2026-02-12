"""
Verification tests for pyLitePT optimizations.
Tests correctness of PointROPE, attention, segment_csr, ball_query, spconv_cpu.dense
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import traceback


def test_pointrope():
    """Test that PointROPE applies real rotation (not identity)."""
    print("=" * 60)
    print("TEST: PointROPE")
    print("=" * 60)
    
    from libs.pointrope import PointROPE
    
    rope = PointROPE(freq=100.0)
    
    # D must be divisible by 6
    B, H, N, D = 1, 4, 16, 18  # D=18, Q=3
    tokens = torch.randn(B, H, N, D, requires_grad=True)
    positions = torch.randint(0, 100, (B, N, 3))
    
    # Forward
    out = rope(tokens, positions)
    
    # Check shape
    assert out.shape == tokens.shape, f"Shape mismatch: {out.shape} vs {tokens.shape}"
    
    # Check NOT identity (should differ from input)
    diff = (out.detach() - tokens.detach()).abs().mean().item()
    assert diff > 0.01, f"Output too similar to input (diff={diff:.6f}). RoPE may be identity!"
    
    # Check backward
    loss = out.sum()
    loss.backward()
    assert tokens.grad is not None, "No gradients!"
    assert tokens.grad.abs().mean().item() > 0, "Zero gradients!"
    
    # Check with zero positions (should still produce valid output)
    zero_pos = torch.zeros(B, N, 3, dtype=torch.long)
    out_zero = rope(tokens.detach().requires_grad_(True), zero_pos)
    # With zero positions, freq=0, cos=1, sin=0, so output should equal input
    diff_zero = (out_zero.detach() - tokens.detach()).abs().mean().item()
    assert diff_zero < 1e-5, f"Zero positions should be identity, diff={diff_zero:.6f}"
    
    print(f"  Shape: OK ({out.shape})")
    print(f"  Non-identity: OK (mean diff = {diff:.4f})")
    print(f"  Backward: OK")
    print(f"  Zero-pos identity: OK (diff = {diff_zero:.6f})")
    print("  PASSED\n")


def test_attention():
    """Test attention CPU fallback produces valid output."""
    print("=" * 60)
    print("TEST: Attention CPU")
    print("=" * 60)
    
    from backend_cpu.attention_cpu import attention_varlen_qkvpacked
    
    H, D = 4, 16
    # Two sequences: len=8 and len=12
    seq_lens = [8, 12]
    total_N = sum(seq_lens)
    
    qkv = torch.randn(total_N, 3, H, D)
    cu_seqlens = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(len(seq_lens))], dtype=torch.int32)
    max_seqlen = max(seq_lens)
    
    out = attention_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen)
    
    assert out.shape == (total_N, H, D), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in attention output!"
    assert not torch.isinf(out).any(), "Inf in attention output!"
    
    print(f"  Shape: OK ({out.shape})")
    print(f"  No NaN/Inf: OK")
    print("  PASSED\n")


def test_segment_csr():
    """Test vectorized segment_csr matches loop-based reference."""
    print("=" * 60)
    print("TEST: segment_csr (vectorized)")
    print("=" * 60)
    
    from backend_cpu.torch_scatter_cpu import segment_csr
    
    # Test data: 3 segments of sizes 3, 2, 4
    src = torch.randn(9, 8)  # [N=9, D=8]
    indptr = torch.tensor([0, 3, 5, 9])
    
    for reduce in ["sum", "mean", "max", "min"]:
        out = segment_csr(src, indptr, reduce=reduce)
        
        # Reference (manual loop)
        ref = torch.zeros(3, 8)
        for i in range(3):
            s, e = indptr[i].item(), indptr[i+1].item()
            seg = src[s:e]
            if reduce == "sum": ref[i] = seg.sum(0)
            elif reduce == "mean": ref[i] = seg.mean(0)
            elif reduce == "max": ref[i] = seg.max(0)[0]
            elif reduce == "min": ref[i] = seg.min(0)[0]
        
        diff = (out - ref).abs().max().item()
        assert diff < 1e-5, f"segment_csr({reduce}) mismatch: max diff = {diff}"
        print(f"  {reduce}: OK (max diff = {diff:.2e})")
    
    # Test empty segments
    indptr_empty = torch.tensor([0, 0, 3, 3])
    out_empty = segment_csr(src[:3], indptr_empty, reduce="sum")
    assert out_empty.shape == (3, 8), f"Empty segment shape: {out_empty.shape}"
    assert out_empty[0].abs().max() == 0, "Empty segment should be zero"
    print(f"  Empty segments: OK")
    
    print("  PASSED\n")


def test_ball_query():
    """Test vectorized ball_query."""
    print("=" * 60)
    print("TEST: ball_query (vectorized)")
    print("=" * 60)
    
    from backend_cpu.pointops import ball_query
    
    N = 50
    xyz = torch.randn(N, 3)
    xyz_batch_cnt = torch.tensor([N])
    new_xyz = torch.randn(10, 3)
    new_xyz_batch_cnt = torch.tensor([10])
    
    result = ball_query(1.0, 8, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
    
    assert result.shape == (10, 8), f"Shape mismatch: {result.shape}"
    # Valid indices should be in range [0, N) or -1
    valid_mask = result >= 0
    if valid_mask.any():
        assert result[valid_mask].max() < N, "Index out of range!"
    print(f"  Shape: OK ({result.shape})")
    print(f"  Indices valid: OK")
    print("  PASSED\n")


def test_spconv_dense():
    """Test vectorized dense() conversion."""
    print("=" * 60)
    print("TEST: spconv_cpu.dense() (vectorized)")
    print("=" * 60)
    
    from backend_cpu.spconv_cpu import SparseConvTensor
    
    N, C = 20, 8
    features = torch.randn(N, C)
    spatial_shape = [10, 10, 10]
    
    # Random valid indices
    b = torch.zeros(N, dtype=torch.int32)
    xyz = torch.randint(0, 10, (N, 3), dtype=torch.int32)
    indices = torch.cat([b.unsqueeze(1), xyz], dim=1)
    
    sct = SparseConvTensor(features, indices, spatial_shape, batch_size=1)
    
    # Test channels_first
    dense_cf = sct.dense(channels_first=True)
    assert dense_cf.shape == (1, C, 10, 10, 10), f"Shape: {dense_cf.shape}"
    
    # Verify some values
    for i in range(min(5, N)):
        bi, xi, yi, zi = indices[i].long().tolist()
        stored = dense_cf[bi, :, xi, yi, zi]
        expected = features[i]
        diff = (stored - expected).abs().max().item()
        assert diff < 1e-6, f"Value mismatch at point {i}: diff={diff}"
    
    # Test channels_last
    dense_cl = sct.dense(channels_first=False)
    assert dense_cl.shape == (1, 10, 10, 10, C), f"Shape: {dense_cl.shape}"
    
    print(f"  Channels-first shape: OK ({dense_cf.shape})")
    print(f"  Channels-last shape: OK ({dense_cl.shape})")
    print(f"  Values: OK")
    print("  PASSED\n")


def test_hybrid_backend():
    """Test hybrid_backend threading config."""
    print("=" * 60)
    print("TEST: hybrid_backend threading")
    print("=" * 60)
    
    from hybrid_backend import setup_backends
    
    status = setup_backends(verbose=False)
    
    threads = torch.get_num_threads()
    assert threads >= 1, f"Thread count should be >= 1, got {threads}"
    
    print(f"  Backend status: {status}")
    print(f"  Threads: {threads}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("pyLitePT Optimization Verification Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("PointROPE", test_pointrope),
        ("Attention CPU", test_attention),
        ("segment_csr", test_segment_csr),
        ("ball_query", test_ball_query),
        ("spconv_cpu.dense", test_spconv_dense),
        ("hybrid_backend", test_hybrid_backend),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            traceback.print_exc()
            print(f"  FAILED: {e}\n")
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{passed+failed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 60)
