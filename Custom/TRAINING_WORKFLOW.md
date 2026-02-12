# LitePT Custom Training Workflow

Complete workflow for training LitePT on your own labeled data.
This workflow supports **Unified Training** (Segmentation + Detection), **Dual-Path Architecture**, **Auto-Configuration**, and **Stability Optimizations**.

---

## 🚀 Key Features

- **Flexible Architecture**: Choose between multi-stage (with downsampling) or single-stage (no downsampling) variants.
- **Dual-Path Unified Mode**: Optimal architecture for multi-task learning - segmentation gets hierarchical features, detection gets high-resolution features.
- **Auto-Configuration**: Automatically calculates class weights, input channels, and anchor sizes from your data.
- **Robustness**: Built-in support for CPU/CUDA, BFloat16/FP16 (AMP), and automatic error handling.
- **Dynamic Visualization**: Interactive GUI with adjustable NMS and confidence thresholds.

---

## 1. Data Preparation

### Supported Formats

| Format | Structure | Use Use Case |
|--------|-----------|----------|
| **NPY Folder** | Folder with `coord.npy`, `color.npy`, `segment.npy`, `gt_boxes.npy` | **Recommended**. Full features, fast loading. |
| **PLY** | Single `.ply` file with embedded labels | Quick testing, simple datasets. |

### Directory Structure

```
<ProjectRoot>/data/
├── train/           # Training scenes
│   ├── scene1/      # NPY folder format
│   │   ├── coord.npy       # (N, 3) XYZ
│   │   ├── color.npy       # (N, 3) RGB
│   │   ├── segment.npy     # (N,) Labels
│   │   └── gt_boxes.npy    # (M, 8) Boxes
│   └── scene2.ply   # OR PLY format
├── val/             # Validation scenes (same structure)
└── test/            # Test scenes (optional)
```

> [!TIP]
> Use **labelCloud** to label your data. We support its PLY export and NPY format directly.
> To use it:
> ​Navigate to the directory:
> cd "path/to/labelCloud"
​> Run the application:
> python labelCloud.py

---

## 2. Configuration (`Custom/config.py`)

Edit `Custom/config.py` to set up your dataset. Most settings can be set to `'auto'`.

### Essential Settings

```python
# Dataset Path
DATA_PATH = "data"

# Classes (Auto-loaded from classes.json if available)
CLASS_NAMES = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid', 'torus', 'wall']
NUM_CLASSES_SEG = 7
NUM_CLASSES_DET = 6  # Examples: Exclude 'wall' from detection

# Model Architecture
MODEL_VARIANT = 'small'  # Choose size: 'nano', 'micro', 'tiny', 'small', 'base', 'large'
                         # System automatically selects architecture based on mode
USE_DUAL_PATH_UNIFIED = False  # Set True for optimal unified mode (separate backbones)

# Training
EPOCHS = 100
BATCH_SIZE = 4       # Reduce if OOM
LEARNING_RATE = 0.001
```

### Model Variant Selection Guide

**For Segmentation Mode:**
- Use any size: `nano`, `micro`, `tiny`, `small`, `base`, `large`
- System automatically uses multi-stage architecture
- Recommended: `small` (~12M params)

**For Detection Mode:**
- Use any size: `nano`, `micro`, `tiny`, `small`, `base`, `large`
- System automatically uses single-stage architecture (no downsampling)
- Recommended: `small` (~5M params single-stage)
- Why? No downsampling preserves spatial resolution for small objects

**For Unified Mode:**
- Single-path (parameter efficient): Use any size like `small`, set `USE_DUAL_PATH_UNIFIED = False`
- Dual-path (optimal performance) ⭐: Use any size like `small`, set `USE_DUAL_PATH_UNIFIED = True`
  - Segmentation branch: Uses multi-stage architecture
  - Detection branch: Automatically uses single-stage architecture
  - Each task gets its optimal architecture!

### Auto-Configuration Features

Set these to `'auto'` in `config.py` to let the script calculate optimal values:

- **`INPUT_CHANNELS = 'auto'`**: Detects if your data has color, normals, intensity, etc.
- **`CLASS_WEIGHTS = 'auto'`**: Balances loss for rare classes.
- **`MEAN_SIZE = 'auto'`**: Calculates optimal 3D anchor sizes for detection.

---

## 3. Training (`Custom/train.py`)

The training script `Custom/train.py` handles the entire loop, utilizing `Custom/core.py` for model definitions. It automatically selects the appropriate model class based on your configuration:
- **Single-path unified**: Uses `LitePTUnifiedCustom` (one backbone for both tasks)
- **Dual-path unified**: Uses `LitePTDualPathUnified` (separate backbones for optimal performance)

### Run Training

```bash
# Segmentation only (uses multi-stage architecture)
python Custom/train.py --mode segmentation --epochs 100 --format npy

# Detection only (recommended: use single-stage variant in config.py)
python Custom/train.py --mode detection --epochs 100 --format npy

# Unified single-path (one backbone, parameter efficient)
python Custom/train.py --mode unified --epochs 100 --format npy

# Unified dual-path (two backbones, optimal performance) ⭐
# Set USE_DUAL_PATH_UNIFIED = True in config.py first
python Custom/train.py --mode unified --epochs 100 --format npy
```

### Modes

| Mode | Command Flag | Description | Recommended Config |
|------|--------------|-------------|-------------------|
| **Unified (Dual-Path)** ⭐ | `--mode unified` | Separate backbones for seg/det (optimal) | `small` + `USE_DUAL_PATH_UNIFIED=True` |
| **Unified (Single-Path)** | `--mode unified` | Shared backbone (parameter efficient) | `small` + `USE_DUAL_PATH_UNIFIED=False` |
| **Seg Only** | `--mode segmentation` | Train only segmentation | `small` (auto: multi-stage) |
| **Det Only** | `--mode detection` | Train only detection | `small` (auto: single-stage) |

### Key Flags

- `--data_path`: Override data directory.
- `--model_variant`: Choose model size (`nano` is good default).
- `--workers`: Number of dataloader workers (0 for Windows/Debug).

---

## 4. Evaluation & Visualization

### Evaluate Metrics

Run batch evaluation to get mAP and Accuracy:

```bash
python Custom/evaluate.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy
```

### Interactive Visualization

Launch the 3D viewer to inspect predictions vs Ground Truth:

```bash
python Custom/visualize.py --checkpoint exp/custom_training/best_unified_model.pth --data_format npy
```

**Controls:**
- **Scene Selection**: Dropdown to switch scenes.
- **Show Boxes**: Toggle bounding boxes.
- **Confidence Slider**: Filter predictions by score.
- **NMS Slider**: Adjust Non-Maximum Suppression overlap threshold.

---

## 5. Analysis Tools

We provide tools to understand your dataset before/after training.

### `auto_optimize.py`
Calculates optimal class weights and updates `config.py`.

```bash
python Custom/auto_optimize.py --apply
```

---

## 6. Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Zero mAP (0.0%)** | Missing detection boxes or wrong architecture | 1. Check `gt_boxes.npy` exists<br>2. System auto-uses single-stage for detection<br>3. Ensure `DETECTION_LOSS_WEIGHT > 0` |
| **NaN Loss** | Learning rate too high or invalid data | 1. Reduce learning rate to 0.0001<br>2. Disable AMP (`USE_AMP = False`)<br>3. Check for NaN/Inf in input data |
| **OOM (Out of Memory)** | Batch size too large or model too big | 1. Reduce `BATCH_SIZE` (try 2 or 1)<br>2. Use smaller variant (`nano` or `micro`)<br>3. Enable `USE_AMP = True` |
| **Slow Training** | Large model or inefficient settings | 1. Use smaller variant<br>2. Enable `USE_AMP = True`<br>3. Set `TRAIN_STEPS_PER_EPOCH` to limit steps<br>4. Increase `NUM_WORKERS` (not on Windows) |
| **Poor Segmentation Accuracy** | Wrong architecture or class imbalance | 1. Verify segmentation mode (not detection)<br>2. Check class weights are balanced<br>3. Increase model size (`small` → `base`) |
| **Poor Detection Accuracy** | Wrong anchor sizes or architecture | 1. System auto-uses single-stage for detection<br>2. Check `MEAN_SIZE` in config (run auto_optimize.py)<br>3. Verify coordinate system (XYZ vs XZY) |
| **BFloat16 Error** | Incompatible hardware | Code automatically handles this - if persists, set `USE_AMP = False` |
| **Import Errors** | Missing dependencies | Run `pip install -r requirements.txt` |

### Debugging Steps

1. **Check Data Integrity**
   ```bash
   python Custom/analyze_data.py --path data/train/scene1/
   ```

2. **Verify Configuration**
   ```bash
   python Custom/verify_params.py
   ```

3. **Run Auto-Optimization**
   ```bash
   python Custom/auto_optimize.py --data_path data --apply
   ```

4. **Check Logs**
   - Training logs: `exp/custom_training/training.log`
   - Console output for real-time debugging

### Performance Optimization

**Memory Optimization:**
```python
# Custom/config.py
BATCH_SIZE = 2  # Reduce if OOM
MODEL_VARIANT = 'nano'  # Smaller model
USE_AMP = True  # Mixed precision
```

**Speed Optimization:**
```python
# Custom/config.py
MODEL_VARIANT = 'nano'  # Faster model
TRAIN_STEPS_PER_EPOCH = 100  # Limit steps
USE_AMP = True  # Faster computation
```

**Accuracy Optimization:**
```python
# Custom/config.py
MODEL_VARIANT = 'small'  # Better model
USE_DUAL_PATH_UNIFIED = True  # Optimal for unified
EPOCHS = 200  # More training
LOSS_BALANCING_METHOD = 'uncertainty'  # Auto-balance
```

---

## 7. Output Artifacts

Results are saved to `exp/custom_training/`:

- **`best_unified_model.pth`**: Model with highest **Stable Unified Score**
   - Formula: `0.5 * SegAcc + 0.5 * EMA(mAP@0.5)`
   - Uses Exponential Moving Average (EMA) for detection to reduce noise
- **`training.log`**: Detailed logs of loss and metrics per epoch
- **`checkpoints/`**: Periodic checkpoints during training

### Model Checkpointing

The system saves:
- Best model based on validation metrics
- Last checkpoint for resuming training
- Periodic checkpoints every N epochs (configurable)

To resume training:
```bash
python Custom/train.py --mode unified --resume
```

---

## 8. Advanced Features

### Multi-Task Loss Balancing

The implementation supports automatic loss balancing for unified mode:

**Uncertainty Weighting (Recommended):**
```python
# Custom/config.py
LOSS_BALANCING_METHOD = 'uncertainty'  # Kendall et al. 2018
```
- Learns task-specific uncertainty automatically
- No hyperparameter tuning needed
- Balances segmentation and detection losses dynamically

**GradNorm:**
```python
# Custom/config.py
LOSS_BALANCING_METHOD = 'gradnorm'  # Chen et al. 2018
GRADNORM_ALPHA = 1.5  # Asymmetry parameter
```
- Balances gradient magnitudes across tasks
- Requires tuning `GRADNORM_ALPHA`
- More complex but can be more stable

**Static Weights:**
```python
# Custom/config.py
LOSS_BALANCING_METHOD = 'none'
DETECTION_LOSS_WEIGHT = 2.0  # Manual weight
```
- Simple manual control
- Requires tuning for your dataset

### Detection Configuration

**Anchor Sizes (Mean Sizes):**
```python
# Custom/config.py
DETECTION_CONFIG = {
    'MEAN_SIZE': 'auto',  # Auto-calculate from training data
    # Or manually specify:
    # 'MEAN_SIZE': [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0], ...]  # Per class
}
```

**Loss Weights:**
```python
DETECTION_CONFIG = {
    'LOSS_CONFIG': {
        'LOSS_REG': 'smooth-l1',
        'LOSS_WEIGHTS': {
            'point_cls_weight': 1.0,  # Classification loss weight
            'point_box_weight': 1.0,  # Box regression loss weight
            'code_weights': [1.0] * 8  # Per-parameter weights
        }
    }
}
```

---

## 9. Workspace Structure

Understanding the file layout helps in customization and debugging.

### Core Scripts (`Custom/`)
This is where 99% of your interaction happens.
- **`train.py`**: Main training loop. Handles data loading, unified loss calculation, and checkpointing.
- **`evaluate.py`**: Validation script. Calculates mAP, Recall, and Segmentation Accuracy on test sets.
- **`visualize.py`**: Interactive 3D GUI. Visualizes Ground Truth vs Predictions with NMS/Confidence controls.
- **`core.py`**: **Model Definitions**. Defines:
  - `LitePTUnifiedCustom`: Single-path unified model (one backbone for both tasks)
  - `LitePTDualPathUnified`: Dual-path unified model (separate backbones for optimal performance)
  - `MODEL_CONFIGS`: All model variants (nano through large, plus single-stage variants)
- **`config.py`**: **Global Settings**. Controls paths, hyperparameters, model variants, and dual-path mode.
- **`auto_optimize.py`**: Helper to calculate class weights and analyze distributions.
- **`verify_params.py`**: Utility to verify model parameter counts.
- **`dataset.py`**: Custom dataset loader for NPY/PLY formats.

### Architecture (`models/`)
Deep learning modules and backbone definitions.
- **`litept/litept.py`**: The LitePT Backbone implementation (Encoder-Decoder Point Transformer).
- **`detection.py`**: Adapter connecting LitePT backbone to detection heads.
- **`modules.py`**: Basic building blocks (PointModule, PointSequential).

### Detection Utilities (`pcdet_lite/`)
Lightweight, pure-Python implementation of 3D detection tools (No CUDA compile needed).
- **`detection_heads/`**: Box prediction heads.
- **`box_utils.py`**: Box decoding and encoding logic.
- **`iou3d_nms_utils.py`**: Intersection over Union (IoU) and NMS calculations (Note: We use fast `torchvision` NMS in scripts).

### Data Loading (`datasets/`)
This folder contains both essential utilities and legacy loaders.

**Essential Files**:
- **`utils.py`**: Contains `collate_fn` which handles batching of variable-sized 3D point clouds. **CRITICAL**.
- **`Shapes3D/shapes3d_generator.py`**: A script to generating synthetic training data (Cubes, Spheres, etc.) with perfect labels. **CRITICAL for local testing**.

**Main Loader**:
- **`Custom/dataset.py`**: The primary loader for your NPY/PLY data. It serves the same purpose as the legacy loaders but is more flexible.

**Legacy Loaders (Reference Only)**:
- `shapes3d.py`, `scannet.py`, `waymo.py`, etc.: Loaders from the original codebase. Not used by `Custom/train.py`.
- `defaults.py`: Base class for legacy loaders.

### Other Folders
- **`exp/`**: **Results Directory**. All logs and checkpoints are saved here.
- **`metrics/`**: Evaluation logic (mAP calculation).
- **`libs/`**: Low-level point cloud operations (PointOps).
- **`utils/`**: General helper scripts (Logging, distributed setup).


---

## 10. Quick Reference

### Essential Configuration
```python
# Custom/config.py
DATA_PATH = "data"
NUM_CLASSES_SEG = 20  # 0 to disable
NUM_CLASSES_DET = 5   # 0 to disable
MODEL_VARIANT = 'small'  # nano, micro, tiny, small, base, large
USE_DUAL_PATH_UNIFIED = True  # For optimal unified mode
EPOCHS = 100
BATCH_SIZE = 4
```

### Training Commands
```bash
# Segmentation only
python Custom/train.py --mode segmentation --format npy

# Detection only (auto-uses single-stage)
python Custom/train.py --mode detection --format npy

# Unified (dual-path recommended)
python Custom/train.py --mode unified --format npy
```

### Architecture Selection (Automatic)
- **Segmentation mode** → Multi-stage architecture (with downsampling)
- **Detection mode** → Single-stage architecture (no downsampling) - automatic!
- **Unified single-path** → Multi-stage shared backbone
- **Unified dual-path** → Seg: multi-stage, Det: single-stage (optimal!)

### Model Sizes
| Variant | Multi-stage | Single-stage | Use Case |
|---------|-------------|--------------|----------|
| nano | ~1M | ~0.5M | Quick experiments |
| micro | ~2M | ~1M | Small datasets |
| tiny | ~6M | ~2M | Development |
| small | ~12M | ~5M | **Production** ⭐ |
| base | ~45M | ~15M | High accuracy |
| large | ~86M | ~30M | Maximum accuracy |

### Common Issues
- **OOM** → Reduce `BATCH_SIZE`, use smaller variant
- **Zero mAP** → Check `gt_boxes.npy`, system auto-uses single-stage
- **NaN Loss** → Reduce learning rate, disable AMP
- **Slow** → Use smaller variant, enable AMP

---

**Last Updated:** 2026-02-10
**Version:** 3.0 (Simplified Variant System)


