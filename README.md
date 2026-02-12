# pyLitePT: Pure Python 3D Detection & Segmentation

**A highly optimized, "Pure Python" implementation of LitePT for 3D Object Detection and Semantic Segmentation.**

## 🚀 Key Features

-   **Flexible Architecture**: Multi-stage variants (with downsampling) for segmentation, single-stage variants (no downsampling) for detection.
-   **Dual-Path Unified Mode**: Optimal multi-task learning with separate backbones - segmentation gets hierarchical features, detection gets high-resolution features.
-   **Pure Python Backend**: Decoupled from legacy C++/CUDA engines; running natively on CPU and CUDA via PyTorch.
-   **Optimized Performance**: Vectorized operations, SDPA attention (PyTorch 2.0+), real PointROPE, multi-threaded CPU.
-   **Data Agnostic**: Built-in support for PLY (labelCloud) and NPY (PCDet-lite) formats with auto-recovery.
-   **Minimal Dependencies**: `torch`, `numpy`, `addict`, `timm`, `scipy`, `vispy`.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/pyLitePT
cd pyLitePT

# Install dependencies
pip install -r requirements.txt

# (Optional) Install native CUDA backends for 30-50% speedup on GPU
# Requires NVIDIA GPU + CUDA Drivers
python install_native_backends.py
```

## 📂 Project Structure

-   `Custom/`: **Primary entrance for users.** Contains training, evaluation, configuration, and model definitions.
    - `core.py`: Model architectures (LitePTUnifiedCustom, LitePTDualPathUnified)
    - `config.py`: Configuration settings
    - `train.py`: Training script
    - `evaluate.py`: Evaluation script
    - `visualize.py`: 3D visualization GUI
-   `models/`: Core LitePT backbone and detection head architectures.
-   `pcdet_lite/`: Lightweight 3D detection utility layer.
-   `backend_cpu/`: Optimized Python fallback implementations (SDPA attention, vectorized scatter/ball_query/spconv).
-   `hybrid_backend.py`: Auto-detects GPU libraries (spconv, flash_attn, torch_scatter) and falls back to `backend_cpu/`.
-   `libs/pointrope.py`: Real 3D Rotary Position Embeddings (vectorized PyTorch, JIT-compiled).
-   `tests/`: Optimization verification tests.

## 📈 Usage

### 1. Configure
Edit `Custom/config.py` to set your dataset paths, class names, and model variant:
```python
# Choose model variant based on your task
MODEL_VARIANT = 'small'  # For segmentation or unified mode
# MODEL_VARIANT = 'single_stage_small'  # For detection-only mode

# Enable dual-path for optimal unified mode
USE_DUAL_PATH_UNIFIED = True  # Separate backbones for seg/det
```

### 2. Train
```bash
# Segmentation only
python Custom/train.py --mode segmentation

# Detection only (use single_stage variant in config)
python Custom/train.py --mode detection

# Unified mode (dual-path recommended for best performance)
python Custom/train.py --mode unified
```

### 3. Visualize
```bash
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy
```

## 📚 Documentation

- **[Custom/README.md](Custom/README.md)**: Quick start guide
- **[Custom/TRAINING_WORKFLOW.md](Custom/TRAINING_WORKFLOW.md)**: Detailed training workflow
- **[Custom/ARCHITECTURE_GUIDE.md](Custom/ARCHITECTURE_GUIDE.md)**: Architecture decisions

## ⚙️ Backend Architecture

The system auto-detects optimal backends via `hybrid_backend.py`:

| Component | GPU (Native) | CPU (Fallback) |
|-----------|-------------|----------------|
| Sparse Conv | `spconv` CUDA | `backend_cpu/spconv_cpu.py` (vectorized dense) |
| Scatter Ops | `torch_scatter` | `backend_cpu/torch_scatter_cpu.py` (scatter_reduce_) |
| Attention | `flash_attn` | `backend_cpu/attention_cpu.py` (SDPA / JIT) |
| PointROPE | CUDA kernel | `libs/pointrope.py` (vectorized PyTorch) |

**Threading**: Auto-configures `cpu_count // 2` threads. Override with `TORCH_NUM_THREADS` env var.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
