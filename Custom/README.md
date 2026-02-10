# LitePT Custom Training Pipeline (Unified)

This folder contains the custom unified implementation for training LitePT on your own datasets (e.g., from **labelCloud**) with support for:

1.  **Semantic Segmentation**
2.  **3D Object Detection**
3.  **Unified (Combined)** - with optional dual-path architecture

## 🎯 Model Variants

Choose a size based on your needs - the architecture is automatically selected based on your mode:

| Variant | Params (Multi-stage) | Params (Single-stage) | Use Case |
|---------|---------------------|----------------------|----------|
| `nano` | ~1M | ~0.5M | Lightweight, fast inference, quick experiments |
| `micro` | ~2M | ~1M | Small datasets, edge devices |
| `tiny` | ~6M | ~2M | Balanced speed/accuracy, development |
| `small` | ~12M | ~5M | **Recommended** for production |
| `base` | ~45M | ~15M | High accuracy, more compute |
| `large` | ~86M | ~30M | Maximum accuracy |

### Architecture Selection (Automatic)

The system automatically selects the optimal architecture based on your mode:

**Segmentation Mode** (`NUM_CLASSES_SEG > 0`, `NUM_CLASSES_DET = 0`):
- Uses multi-stage architecture with downsampling
- Encoder-decoder with hierarchical features
- Optimal for dense prediction

**Detection Mode** (`NUM_CLASSES_SEG = 0`, `NUM_CLASSES_DET > 0`):
- Automatically uses single-stage architecture (no downsampling)
- Preserves spatial resolution for small objects
- Follows author's recommendation

**Unified Mode** (both > 0):
- **Single-Path** (`USE_DUAL_PATH_UNIFIED = False`): Shared multi-stage backbone
- **Dual-Path** (`USE_DUAL_PATH_UNIFIED = True`) ⭐: Seg uses multi-stage, Det uses single-stage

## 📁 Directory Structure

```
Custom/
├── config.py           # Configuration (classes, paths, params)
├── core.py             # Model definitions (LitePTUnifiedCustom, LitePTDualPathUnified)
├── dataset.py          # CustomDataset (PLY & NPY support)
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
├── visualize.py        # Visualization GUI
├── auto_optimize.py    # Auto-configuration tool
├── verify_params.py    # Parameter count verification
├── TRAINING_WORKFLOW.md# Detailed step-by-step guide
├── README.md           # This file
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
# Train Semantic Segmentation (automatically uses multi-stage architecture)
python Custom/train.py --mode segmentation --format ply --data_path "path/to/data"

# Train 3D Object Detection (automatically uses single-stage architecture)
python Custom/train.py --mode detection --format npy --data_path "path/to/data"

# Train Unified (Single-path: one backbone for both tasks)
python Custom/train.py --mode unified --format ply --data_path "path/to/data"

# Train Unified (Dual-path: optimal architecture for each task) ⭐
# Set USE_DUAL_PATH_UNIFIED = True in config.py first
python Custom/train.py --mode unified --format ply --data_path "path/to/data"
```

**Note:** Just set `MODEL_VARIANT = 'small'` (or any size) in config.py. The system automatically selects:
- Segmentation mode → multi-stage architecture
- Detection mode → single-stage architecture (no downsampling)
- Unified dual-path → both architectures (optimal for each task)

### 3. Visualization

Use `visualize.py` to inspect results. Includes 3D bounding box visualization.

```bash
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth --data_format ply
```

---

## 📚 Documentation

- **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** - Detailed step-by-step training guide with troubleshooting

---

## 🛠️ Configuration

### Essential Settings in `Custom/config.py`

```python
# 1. Dataset Path
DATA_PATH = "data"

# 2. Classes
NUM_CLASSES_SEG = 20  # Set to 0 to disable segmentation
NUM_CLASSES_DET = 5   # Set to 0 to disable detection

# 3. Model Size (same names for all modes)
MODEL_VARIANT = 'small'  # Choose: nano, micro, tiny, small, base, large

# 4. Dual-Path for Unified Mode (optional)
USE_DUAL_PATH_UNIFIED = False  # Set True for optimal unified performance

# 5. Training Parameters
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.001
```

### Mode Selection Guide

| Mode | Config | Architecture | When to Use |
|------|--------|--------------|-------------|
| **Segmentation** | `NUM_CLASSES_SEG > 0`, `NUM_CLASSES_DET = 0` | Multi-stage (with downsampling) | Dense prediction, scene understanding |
| **Detection** | `NUM_CLASSES_SEG = 0`, `NUM_CLASSES_DET > 0` | Single-stage (no downsampling) - automatic | Object detection, small objects |
| **Unified (Single-Path)** | Both > 0, `USE_DUAL_PATH_UNIFIED = False` | Multi-stage shared backbone | Parameter efficiency, limited resources |
| **Unified (Dual-Path)** ⭐ | Both > 0, `USE_DUAL_PATH_UNIFIED = True` | Seg: multi-stage, Det: single-stage | Best performance for both tasks |

### How Architecture Selection Works

**You only choose the size** - the system automatically selects the architecture:

```python
MODEL_VARIANT = 'small'  # Just choose size
```

**System automatically uses:**
- **Segmentation mode** → `small` (multi-stage, 12M params)
- **Detection mode** → `single_stage_small` (single-stage, 5M params) - automatic!
- **Unified dual-path** → `small` for seg + `single_stage_small` for det

### Configuration Examples

**Example 1: Segmentation for Indoor Scenes**
```python
NUM_CLASSES_SEG = 13  # ScanNet classes
NUM_CLASSES_DET = 0
MODEL_VARIANT = 'small'
BATCH_SIZE = 4
EPOCHS = 100
```

**Example 2: Detection for Autonomous Driving**
```python
NUM_CLASSES_SEG = 0
NUM_CLASSES_DET = 3  # Car, Pedestrian, Cyclist
MODEL_VARIANT = 'small'  # System uses single-stage automatically
BATCH_SIZE = 2
EPOCHS = 80
```

**Example 3: Unified for Robotics (Optimal Performance)**
```python
NUM_CLASSES_SEG = 20
NUM_CLASSES_DET = 10
MODEL_VARIANT = 'small'
USE_DUAL_PATH_UNIFIED = True  # Each task gets optimal architecture
BATCH_SIZE = 2
EPOCHS = 150
```

**Example 4: Unified for Edge Device (Parameter Efficient)**
```python
NUM_CLASSES_SEG = 10
NUM_CLASSES_DET = 5
MODEL_VARIANT = 'nano'
USE_DUAL_PATH_UNIFIED = False  # Single backbone
BATCH_SIZE = 8
EPOCHS = 100
```

---

## 🎓 Common Commands

### Training
```bash
# Basic training
python Custom/train.py --mode unified

# With custom data path
python Custom/train.py --mode unified --data_path /path/to/data

# With specific format
python Custom/train.py --mode unified --format npy

# Resume from checkpoint
python Custom/train.py --mode unified --resume
```

### Evaluation
```bash
# Evaluate on test set
python Custom/evaluate.py --checkpoint exp/custom_training/best_unified_model.pth

# With specific format
python Custom/evaluate.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy
```

### Visualization
```bash
# Launch 3D viewer
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth

# With specific format
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy
```

### Auto-Optimization
```bash
# Analyze dataset and suggest config
python Custom/auto_optimize.py --data_path data

# Apply optimizations automatically
python Custom/auto_optimize.py --data_path data --apply
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **OOM Error** | Reduce `BATCH_SIZE` or use smaller variant (nano/micro) |
| **Zero mAP** | Check `gt_boxes.npy` exists; system auto-uses single-stage for detection |
| **Slow training** | Use smaller variant, enable `USE_AMP = True` |
| **Poor seg accuracy** | Verify segmentation mode; check class weights |
| **Poor det accuracy** | System auto-uses single-stage; check anchor sizes (`MEAN_SIZE`) |
| **NaN Loss** | Reduce learning rate; disable AMP; check input data |

---

## 📊 Data Format

### NPY Folder (Recommended)
```
data/
├── train/
│   ├── scene1/
│   │   ├── coord.npy      # (N, 3) XYZ coordinates
│   │   ├── color.npy      # (N, 3) RGB colors
│   │   ├── segment.npy    # (N,) segmentation labels
│   │   └── gt_boxes.npy   # (M, 8) detection boxes [x,y,z,dx,dy,dz,heading,class]
│   └── scene2/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### PLY Format
```
data/
├── train/
│   ├── scene1.ply         # Point cloud with labels
│   ├── scene1_gt_boxes.npy  # Detection boxes
│   └── scene2.ply
├── val/
│   └── ...
└── test/
    └── ...
```

---

## 💡 Performance Tips

### 1. Start Small, Scale Up
```python
# Phase 1: Quick validation (1-2 hours)
MODEL_VARIANT = 'nano'
EPOCHS = 20

# Phase 2: Full training (overnight)
MODEL_VARIANT = 'small'
EPOCHS = 100
```

### 2. Memory Optimization
If you get OOM (Out of Memory):
```python
BATCH_SIZE = 2  # Reduce batch size
MODEL_VARIANT = 'nano'  # Use smaller variant
USE_AMP = True  # Enable mixed precision
```

### 3. Speed Optimization
For faster training:
```python
MODEL_VARIANT = 'nano'  # Smaller model
NUM_WORKERS = 4  # More data loading workers (not on Windows)
USE_AMP = True  # Mixed precision
TRAIN_STEPS_PER_EPOCH = 100  # Limit steps per epoch
```

### 4. Accuracy Optimization
For best results:
```python
MODEL_VARIANT = 'small'  # Or base/large
USE_DUAL_PATH_UNIFIED = True  # If unified mode
EPOCHS = 200  # More training
BATCH_SIZE = 4  # Larger batch if possible
LOSS_BALANCING_METHOD = 'uncertainty'  # Automatic balancing
```

---

## 📚 Documentation

- **[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)** - Detailed step-by-step training guide with troubleshooting

---

## 🛠️ Configuration (Detailed)
---

## 🛠️ Advanced Configuration

Edit `Custom/config.py`:
*   `NUM_CLASSES_SEG`: Set to >0 to enable segmentation head.
*   `NUM_CLASSES_DET`: Set to >0 to enable detection head.
*   `MODEL_VARIANT`: Choose size: `nano`, `micro`, `tiny`, `small`, `base`, or `large`
*   `USE_DUAL_PATH_UNIFIED`: Set to `True` for optimal unified mode (separate backbones for seg/det).
*   `DETECTION_CONFIG`: Configure anchor sizes and loss weights.
*   `CLASS_WEIGHTS`: Auto-calculated or manually set for class balancing
*   `LOSS_BALANCING_METHOD`: `'uncertainty'` (auto), `'gradnorm'`, or `'none'` (static)
*   `DETECTION_LOSS_WEIGHT`: Balance factor for detection vs segmentation loss

---

**Last Updated:** 2026-02-10
**Version:** 3.0 (Simplified Variant System)
