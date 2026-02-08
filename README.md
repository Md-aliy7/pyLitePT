# pyLitePT: Pure Python 3D Detection & Segmentation

**A highly optimized, "Pure Python" implementation of LitePT for concurrent 3D Object Detection and Semantic Segmentation.**

## 🚀 Key Features

-   **Unified Architecture**: Concurrent Segmentation and Detection in a single forward pass.
-   **Pure Python Backend**: Decoupled from legacy C++/CUDA engines; running natively on CPU and CUDA via PyTorch.
-   **Optimized Performance**: Vectorized JoU/mAP calculation and intelligent class weight caching.
-   **Data Agnostic**: Built-in support for PLY (labelCloud) and NPY (PCDet-lite) formats with auto-recovery.
-   **Minimalist**: Just `torch`, `numpy`, and `vispy`.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/pyLitePT
cd pyLitePT

# Install dependencies
pip install -r requirements.txt
```

## 📂 Project Structure

-   `Custom/`: **Primary entrance for users.** Contains training, evaluation, and configuration.
-   `models/`: Core LitePT and Detection head architectures.
-   `pcdet_lite/`: Lightweight 3D detection utility layer.
-   `backend_cpu/`: Python fallback implementations for GPU-less environments.

## 📈 Usage

### 1. Configure
Edit `Custom/config.py` to set your dataset paths and class names.

### 2. Train
```bash
python Custom/train.py --mode unified --variant nano
```

### 3. Visualize
```bash
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy
```

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
