"""
LitePT Hybrid Backend Loader
============================
Attempts to load native C++ CUDA-optimized libraries.
Falls back to Pure Python implementations when unavailable.

Usage:
    from litept.hybrid_backend import setup_backends
    setup_backends()  # Call BEFORE importing litept.model
"""

import sys
import torch

# Track what's being used for logging
BACKEND_STATUS = {
    'spconv': 'unknown',
    'torch_scatter': 'unknown', 
    'flash_attn': 'unknown',
    'pointrope': 'unknown',
}


def setup_backends(verbose=True):
    """
    Setup hybrid backends with graceful fallback.
    
    Returns dict with status of each backend.
    """
    global BACKEND_STATUS
    
    use_cuda = torch.cuda.is_available()
    
    if verbose:
        print("=" * 60)
        print("LitePT Backend Configuration")
        print("=" * 60)
        print(f"CUDA Available: {use_cuda}")
    
    # =========================================================================
    # 0. Mock SharedArray (Optional dependency for caching)
    # =========================================================================
    try:
        import SharedArray
    except ImportError:
        # Mock it to prevent hard crashes if imported elsewhere but not used
        # real usage in utils/cache.py handles ImportError gracefully too, but this is safer
        if 'SharedArray' not in sys.modules:
            from unittest.mock import MagicMock
            sys.modules["SharedArray"] = MagicMock()
            if verbose:
                print("- SharedArray: Not found (Mocked)")
    
    # =========================================================================
    # 1. SPCONV
    # =========================================================================
    try:
        if use_cuda:
            import spconv.pytorch as spconv_native
            # Test if it actually works
            _ = spconv_native.SparseConvTensor
            BACKEND_STATUS['spconv'] = 'native_cuda'
            if verbose:
                print(f"  spconv: Native CUDA [OK]")
        else:
            raise ImportError("No CUDA")
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        try:
            from backend_cpu import spconv_cpu
        except ImportError:
            import backend_cpu.spconv_cpu as spconv_cpu
            
        sys.modules['spconv'] = spconv_cpu
        sys.modules['spconv.pytorch'] = spconv_cpu
        BACKEND_STATUS['spconv'] = 'python_fallback'
        if verbose:
            print(f"  spconv: Python fallback [OK]")
    
    # =========================================================================
    # 2. TORCH_SCATTER
    # =========================================================================
    try:
        if use_cuda:
            import torch_scatter as scatter_native
            # Test basic function
            _ = scatter_native.segment_csr
            BACKEND_STATUS['torch_scatter'] = 'native_cuda'
            if verbose:
                print(f"  torch_scatter: Native CUDA [OK]")
        else:
            raise ImportError("No CUDA")
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        try:
            from backend_cpu import torch_scatter_cpu
        except ImportError:
            import backend_cpu.torch_scatter_cpu as torch_scatter_cpu
            
        sys.modules['torch_scatter'] = torch_scatter_cpu
        BACKEND_STATUS['torch_scatter'] = 'python_fallback'
        if verbose:
            print(f"  torch_scatter: Python fallback [OK]")
    
    # =========================================================================
    # 3. FLASH_ATTN
    # =========================================================================
    try:
        if use_cuda:
            from flash_attn import flash_attn_varlen_qkvpacked_func
            BACKEND_STATUS['flash_attn'] = 'native_cuda'
            if verbose:
                print(f"  flash_attn: Native CUDA [OK]")
        else:
            raise ImportError("No CUDA")
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        try:
            from backend_cpu import attention_cpu
        except ImportError:
            import backend_cpu.attention_cpu as attention_cpu
        
        class MockFlashAttn:
            """Mock flash_attn module using Python implementation."""
            @staticmethod
            def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, 
                                                  dropout_p=0.0, softmax_scale=None):
                return attention_cpu.flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale
                )
        
        sys.modules['flash_attn'] = MockFlashAttn()
        BACKEND_STATUS['flash_attn'] = 'python_fallback'
        if verbose:
            print(f"  flash_attn: Python fallback [OK]")
    
    # =========================================================================
    # 4. POINTROPE (handled in libs/pointrope/__init__.py)
    # =========================================================================
    # PointROPE fallback is already handled in its __init__.py
    # We just check which one was loaded
    try:
        if use_cuda:
            # Try to import the CUDA kernel wrapper
            # It seems the native extension is expected at libs.pointrope.pointrope
            # But we only have libs.pointrope (python)
            # We assume if pointrope_kernel is missing, it's not native
            try:
                import libs.pointrope.pointrope_kernel as _kernel
                BACKEND_STATUS['pointrope'] = 'native_cuda'
                if verbose:
                    print(f"  pointrope: Native CUDA [OK]")
            except ImportError:
                 raise ImportError("No CUDA kernel found")
        else:
            raise ImportError("No CUDA")
    except (ImportError, ModuleNotFoundError, OSError) as e:
        BACKEND_STATUS['pointrope'] = 'python_fallback'
        if verbose:
            print(f"  pointrope: Python fallback [OK]")
    
    if verbose:
        print("=" * 60)
        device_str = "CUDA" if use_cuda else "CPU"
        print(f"Device: {device_str}")
        print("=" * 60)
    
    # Configure CPU threading for optimal performance
    _configure_cpu_threading(verbose)
    
    return BACKEND_STATUS


def _configure_cpu_threading(verbose=True):
    """
    Configure CPU threading for optimal PyTorch performance.
    
    We use single-threaded mode to avoid overhead from Python's GIL
    and multiprocessing contention with the fallback implementations.
    """
    import os
    
    # Use only the main thread (no multiprocessing overhead)
    num_threads = 1
    
    # Set PyTorch threads
    torch.set_num_threads(num_threads)
    
    # Set OpenMP threads (used by PyTorch's CPU backend)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    
    # Set MKL threads (Intel Math Kernel Library)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    # Enable TensorFloat-32 for faster matmul on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if verbose:
            print(f"  TF32/cuDNN optimizations enabled")
    
    if verbose:
        print(f"  CPU threads: {num_threads}")


def get_backend_status():
    """Get current backend status."""
    return BACKEND_STATUS.copy()
