# LitePT Custom Training Pipeline (Unified)

This folder contains the custom unified implementation for training LitePT on your own datasets (e.g., from **labelCloud**) with support for:

1.  **Semantic Segmentation**
2.  **3D Object Detection**
3.  **Unified (Combined)**

It utilizes the `LitePT-nano` backbone ensuring efficiency, but supports `micro` (~2M), `tiny` (~6M), `small` (S ~12M), `base` (B ~45M), and `large` (L ~86M).

## 📁 Directory Structure

```
Custom/
├── config.py           # Configuration (classes, paths, params)
├── dataset.py          # CustomDataset (PLY & NPY support)
├── train.py            # Main training script
├── visualize.py        # Visualization GUI
├── auto_optimize.py    # Auto-configuration tool
├── TRAINING_WORKFLOW.md# Detailed step-by-step guide
├── Readme.md           # This file
```

For a detailed step-by-step guide, see [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md).

## ⚡ Quick Start (Auto-Optimize)

Run this to automatically analyze your data and configure the training:
```bash
python Custom/auto_optimize.py --data_path data --apply
```

## 🚀 Usage

### 1. Data Preparation
Export your data from **labelCloud** (same as KPConvX).
*   **Format**: PLY (`.ply` + `_gt_boxes.npy`) OR NPY Folder.
*   **Structure**: `train/`, `val/`, `test/`.

### 2. Training

Run `train.py` with your desired mode and format.

**Command Line Arguments:**
*   `--mode`: `segmentation`, `detection`, or `unified` (default depends on config)
*   `--format`: `ply` or `npy`. **Make sure to set this correctly!**
*   `--data_path`: Path to your dataset root.

**Examples:**

```bash
# Train Semantic Segmentation
python Custom/train.py --mode segmentation --format ply --data_path "path/to/data"

# Train 3D Object Detection
python Custom/train.py --mode detection --format npy --data_path "path/to/data"

# Train Unified
python Custom/train.py --mode unified --format ply --data_path "path/to/data"
```

### 3. Visualization

Use `visualize.py` to inspect results. Includes 3D bounding box visualization.

```bash
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth --data_format ply
```

## 🛠️ Configuration
Edit `Custom/config.py`:
*   `NUM_CLASSES_SEG`: Set to >0 to enable segmentation head.
*   `NUM_CLASSES_DET`: Set to >0 to enable detection head.
*   `DETECTION_CONFIG`: Configure anchor sizes.
