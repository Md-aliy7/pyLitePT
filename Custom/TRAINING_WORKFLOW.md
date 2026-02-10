# LitePT Custom Training Workflow

Complete workflow for training LitePT on your own labeled data.
This workflow supports **Unified Training** (Segmentation + Detection), **Auto-Configuration**, and **Stability Optimizations**.

---

## 🚀 Key Features

- **Unified Architecture**: Single model for both Semantic Segmentation and 3D Object Detection.
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

# Training
EPOCHS = 100
BATCH_SIZE = 4       # Reduce if OOM
LEARNING_RATE = 0.001
MODEL_VARIANT = 'nano' # 'nano', 'micro', 'tiny' (~6M), 'small' (S), 'base' (B), 'large' (L)
```

### Auto-Configuration Features

Set these to `'auto'` in `config.py` to let the script calculate optimal values:

- **`INPUT_CHANNELS = 'auto'`**: Detects if your data has color, normals, intensity, etc.
- **`CLASS_WEIGHTS = 'auto'`**: Balances loss for rare classes.
- **`MEAN_SIZE = 'auto'`**: Calculates optimal 3D anchor sizes for detection.

---

## 3. Training (`Custom/train.py`)

The training script `Custom/train.py` handles the entire loop, utilizing `Custom/core.py` for model definitions. It is completely independent of the legacy `engines/` directory.

### Run Unified Training

```bash
python Custom/train.py --mode unified --epochs 100 --data_format npy
```

### Modes

| Mode | Command Flag | Description |
|------|--------------|-------------|
| **Unified** | `--mode unified` | Train Segmentation + Detection heads together. |
| **Seg Only** | `--mode segmentation` | Train only segmentation (faster). |
| **Det Only** | `--mode detection` | Train only detection. |

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

| Issue | Solution |
|-------|----------|
| **Zero mAP (0.0%)** | 1. Check `gt_boxes.npy` exists.<br>2. Ensure `LOSS_WEIGHTS['point_box_weight']` is > 0.<br>3. Check for coordinate system mismatch (XYZ vs XZY). |
| **NaN Loss** | 1. Reduce learning rate.<br>2. Disable AMP (`USE_AMP = False` in config).<br>3. Check for invalid values in input data (`analyze_data.py`). |
| **BFloat16 Error** | We fixed this! If you see it, ensure you pulled the latest `train.py` which casts to float32 before numpy conversion. |
| **OOM (Out of Memory)** | 1. Reduce `BATCH_SIZE`.<br>2. Use `MODEL_VARIANT = 'micro'`.<br>3. Enable `USE_AMP = True`. |

---

## 7. Output Artifacts

Results are saved to `exp/custom_training/`:

- `best_unified_model.pth`: Model with highest **Stable Unified Score**.
   - Formula: `0.5 * SegAcc + 0.5 * EMA(mAP@0.5)`
   - Note: We use Exponential Moving Average (EMA) for detection to ignore noise.
- `training.log`: Detailed logs of loss and metrics.

---

## 8. Workspace Structure

Understanding the file layout helps in customization and debugging.

### Core Scripts (`Custom/`)
This is where 99% of your interaction happens.
- **`train.py`**: Main training loop. Handles data loading, unified loss calculation, and checkpointing.
- **`evaluate.py`**: Validation script. Calculates mAP, Recall, and Segmentation Accuracy on test sets.
- **`visualize.py`**: Interactive 3D GUI. Visualizes Ground Truth vs Predictions with NMS/Confidence controls.
- **`core.py`**: **Model Definitions**. Defines the `LitePTUnifiedCustom` model and architecture variants (`nano`...`large`).
- **`config.py`**: **Global Settings**. Controls paths, hyperparameters, and model variants.
- **`auto_optimize.py`**: Helper to calculate class weights and analyze distributions.
- **`verify_params.py`**: Utility to verify model parameter counts.
- **`hybrid_backend.py`**: Handles switching between `spconv` (CUDA) and naive (CPU) backends.
- **`*_cpu.py`**: Pure Python fallbacks for `spconv`, `torch_scatter`, and `attention` (enables running on non-CUDA machines).

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

