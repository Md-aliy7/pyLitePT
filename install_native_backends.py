import sys
import subprocess
import torch
import platform
import os

def check_cuda():
    print("Checking CUDA environment...")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available in this Python environment.")
        print(f"   PyTorch Version: {torch.__version__}")
        print("   Please install PyTorch with CUDA support first, or check your drivers.")
        return False
    
    version = torch.version.cuda
    print(f"✅ CUDA {version} detected.")
    return version

def install_package(package_name, index_url=None):
    cmd = [sys.executable, "-m", "pip", "install", package_name]
    if index_url:
        cmd.extend(["-f", index_url])
    
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call(cmd)
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def main():
    print("==================================================")
    print("   LitePT Native Backend Installer (Windows/Linux)")
    print("==================================================")
    
    cuda_ver = check_cuda()
    if not cuda_ver:
        print("\nSkipping native installation because CUDA is not available.")
        print("You can still run LitePT on CPU mode (slow but functional).")
        return

    # 1. SPCONV
    # spconv-cu120 works for CUDA 12.x
    spconv_pkg = "spconv-cu120" if cuda_ver.startswith("12") else "spconv-cu118"
    print(f"\n[1/3] Installing {spconv_pkg} for sparse convolution speedup...")
    install_package(spconv_pkg)

    # 2. TORCH-SCATTER
    # Need to find compatible wheel for PyTorch version
    pt_ver = torch.__version__.split("+")[0] # e.g. 2.5.1
    # PyG wheels naming convention
    pyg_url = f"https://data.pyg.org/whl/torch-{pt_ver}+{target_cuda_tag(cuda_ver)}.html"
    
    print(f"\n[2/3] Installing torch-scatter for faster aggregation...")
    print(f"   Looking for wheels at: {pyg_url}")
    # torch-scatter often lags behind. If not found, warn.
    if not install_package("torch-scatter", index_url=pyg_url):
        print("   -> torch-scatter binary not found for this PT/CUDA version.")
        print("   -> Using slower fallback (torch.scatter_reduce). This is fine.")

    # 3. FLASH-ATTENTION
    print(f"\n[3/3] Checking FlashAttention...")
    if platform.system() == "Windows":
        print("   -> FlashAttention on Windows is difficult to build/install.")
        print("   -> Skipping. PyTorch 2.0+ SDPA (F.scaled_dot_product_attention) is already fast!")
    else:
        print("   -> Attempting to install flash-attn...")
        install_package("flash-attn")

    print("\n==================================================")
    print("Installation Complete.")
    print("Please verify by running: python Custom/evaluate.py --help")

def target_cuda_tag(version):
    # 12.1 -> cu121, 11.8 -> cu118
    return "cu" + version.replace(".", "")

if __name__ == "__main__":
    main()
