# LitePT Custom Training Pipeline (Unified)

This folder contains the **Unified LitePT** implementation for training on custom datasets (e.g., from **labelCloud**) with support for Segmentation, 3D Object Detection, and Unified Multi-Task Learning.

## 🚀 Latest Updates (v3.3)
-   **Metrics Accuracy**: AP calculation updated to all-point interpolation (VOC 2010+ / COCO standard).
-   **Code Accuracy**: Normalization reverted to per-point (matches original LitePT_det-main exactly).
-   **Backend Verification**: All CPU fallbacks (`spconv_cpu`, `torch_scatter_cpu`, `attention_cpu`) verified against original repos.
-   **Cleaner Output**: Backend setup messages silenced by default.

---

## 📚 Documentation Structure

| Document | Purpose |
|----------|---------|
| **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** | **Start Here**. Detailed step-by-step guide for Data Prep, Training, and Eval. |
| **[ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)** | Deep dive into Model Variants, PointRCNN Head, and Backend details. |
| **[Custom/config.py](config.py)** | Global configuration file (Classes, Paths, Hyperparameters). |

---

## ⚡ Quick Start

### 1. Data Preparation
Export from **labelCloud** as **NPY Folder** (Recommended) or **PLY**.
Structure: `data/train/scene1/...`, `data/val/...`

### 2. Auto-Configuration
Analyze your dataset to automatically set class weights and anchor sizes:
```bash
python Custom/auto_optimize.py --data_path data --apply
```

### 3. Training
Run training (System automatically selects the optimal architecture):

```bash
# Segmentation Only (uses Multi-Stage architecture)
python Custom/train.py --mode segmentation --format npy

# Detection Only (uses Single-Stage architecture)
python Custom/train.py --mode detection --format npy

# Unified Mode (Dual-Path Recommended ⭐)
# Set USE_DUAL_PATH_UNIFIED = True in config.py
python Custom/train.py --mode unified --format npy
```

### 4. Visualization
Inspect results with the interactive 3D viewer:
```bash
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy

# Evaluate metrics
python Custom/evaluate.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy
```

---

## 📁 Directory Structure

```
Custom/
├── config.py           # Configuration (classes, paths, params)
├── core.py             # Model definitions (LitePTUnified, DualPath)
├── dataset.py          # CustomDataset loader
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
├── visualize.py        # Visualization GUI
├── postprocess.py      # NMS and filtering logic
├── auto_optimize.py    # Auto-configuration tool
├── verify_params.py    # Parameter count verification
├── TRAINING_WORKFLOW.md# Detailed How-To
├── ARCHITECTURE_GUIDE.md # Technical Deep Dive
└── README.md           # This file
```

---

**(c) 2026 LitePT Custom Implementation**
