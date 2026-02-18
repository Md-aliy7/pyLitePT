"""
=============================================================================
DATA CONFIGURATION FILE - EDIT THIS FILE FOR YOUR DATASET
=============================================================================
This file contains ALL the settings you need to change for your specific data.
The rest of the training code should NOT need any modifications.

To use with a new dataset:
1. Edit the settings below
2. Run: python Custom/train.py

Supports both Semantic Segmentation and 3D Object Detection.
"""

import os

# =============================================================================
# REQUIRED: DATASET PATHS
# =============================================================================

# Path to your labeled NPY folder data
# Can be a single folder (will auto-split 80/10/10) or have subfolders:
#   your_data/
#   ├── train/        (training data)
#   ├── val/          (validation data)  
#   └── test/         (test data)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')

# Where to save trained models
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'exp', 'custom_training')


# =============================================================================
# REQUIRED: CLASS DEFINITIONS
# =============================================================================

# Class names (Auto-loaded from classes.json if available)
# Default to empty list - will be populated from dataset
CLASS_NAMES = ['ground', 'cart']
_classes_json = os.path.join(DATA_PATH, 'classes.json')
if os.path.exists(_classes_json):
    import json
    with open(_classes_json, 'r') as f:
        CLASS_NAMES = json.load(f)
elif os.path.exists(DATA_PATH):
    # Try to auto-detect from data directory
    import json
    # Check for any scene with classes.json
    for root, dirs, files in os.walk(DATA_PATH):
        if 'classes.json' in files:
            with open(os.path.join(root, 'classes.json'), 'r') as f:
                CLASS_NAMES = json.load(f)
            break

# Number of segmentation classes (Auto-derived)
NUM_CLASSES_SEG = 2

# Validation: Ensure classes are defined
if NUM_CLASSES_SEG == 0 and not os.path.exists(os.path.join(DATA_PATH, 'classes.json')):
    import warnings
    warnings.warn(
        f"No classes.json found in {DATA_PATH}. "
        "CLASS_NAMES will be auto-detected from dataset during training. "
        "For better control, create a classes.json file with your class names."
    )

# Label Mapping (Optional): Map raw dataset labels to training IDs
# Example: {-1: 0} maps label -1 to class 0
# Leave empty if your labels are already correct
LABEL_MAPPING = {}

# Number of detection classes
# Use all segmentation classes for detection by default
# The auto-optimizer will set this based on actual detection boxes in the dataset
NUM_CLASSES_DET = 1

# =============================================================================
# TRAINING HYPERPARAMETERS (adjust as needed)
# =============================================================================

# Number of training epochs
EPOCHS = 900

# Batch size (reduce if running out of memory)
BATCH_SIZE = 2

# Learning rate
LEARNING_RATE = 0.001

# Weight decay for AdamW
WEIGHT_DECAY = 0.01

# Gradient Clipping (1.0 recommended for stability)
GRAD_CLIP_NORM = 1.0

# Class weights (Auto-calculated if 'auto')
# Cart (few points) -> High weight, Ground (many points) -> Low weight
CLASS_WEIGHTS = [16.6, 1.0]

# Training steps per epoch (set to None for full epoch)
TRAIN_STEPS_PER_EPOCH = None

# =============================================================================
# MODEL ARCHITECTURE (usually don't need to change)
# =============================================================================

# Model variant - Choose size based on your needs:
#
# AVAILABLE SIZES (same names for all modes):
#   'nano'   (~1M params)  - Lightweight, fast inference, quick experiments
#   'micro'  (~2M params)  - Small datasets, edge devices
#   'tiny'   (~6M params)  - Balanced speed/accuracy, development
#   'small'  (~12M params) - RECOMMENDED for production
#   'base'   (~45M params) - High accuracy, more compute
#   'large'  (~86M params) - Maximum accuracy
#
# ARCHITECTURE SELECTION (automatic based on mode):
#
#   Segmentation Mode (NUM_CLASSES_SEG > 0, NUM_CLASSES_DET = 0):
#     → Uses multi-stage architecture with downsampling
#     → Encoder-decoder with hierarchical features
#     → Optimal for dense prediction tasks
#
#   Detection Mode (NUM_CLASSES_SEG = 0, NUM_CLASSES_DET > 0):
#     → Automatically uses single-stage architecture (no downsampling)
#     → Preserves spatial resolution for small objects
#     → Follows author's recommendation for detection
#
#   Unified Mode (both > 0):
#     → Single-Path (USE_DUAL_PATH_UNIFIED=False):
#       - Uses multi-stage architecture with downsampling
#       - Shared backbone for both tasks (parameter efficient)
#     → Dual-Path (USE_DUAL_PATH_UNIFIED=True): ⭐ OPTIMAL
#       - Segmentation: multi-stage with downsampling
#       - Detection: single-stage without downsampling
#       - Each task gets optimal architecture!
#
# EXAMPLES:
#   Segmentation:  MODEL_VARIANT='small' → Uses multi-stage 'small'
#   Detection:     MODEL_VARIANT='small' → Automatically uses 'single_stage_small'
#   Unified:       MODEL_VARIANT='small' + USE_DUAL_PATH_UNIFIED=True
#                  → Seg uses 'small', Det uses 'single_stage_small'
#
MODEL_VARIANT = 'nano'

# Dual-Path Architecture for Unified Mode
# 
# When True and NUM_CLASSES_SEG > 0 and NUM_CLASSES_DET > 0:
#   - Uses LitePTDualPathUnified with separate backbones
#   - Segmentation branch: Multi-stage architecture (MODEL_VARIANT)
#   - Detection branch: Single-stage architecture (single_stage_{MODEL_VARIANT})
#   - OPTIMAL performance for both tasks (recommended for production)
#   - Trade-off: More parameters (~1.5x) and slightly slower training
#
# When False:
#   - Uses LitePTUnifiedCustom with single shared backbone
#   - Both tasks use same features (downsampled if multi-stage variant)
#   - Parameter efficient but detection may be suboptimal
#   - Good for: Limited resources, parameter budget constraints
#
# Example configurations:
#   Best performance:     MODEL_VARIANT='small', USE_DUAL_PATH_UNIFIED=True
#   Parameter efficient:  MODEL_VARIANT='small', USE_DUAL_PATH_UNIFIED=False
#   Fast development:     MODEL_VARIANT='nano',  USE_DUAL_PATH_UNIFIED=False
#
USE_DUAL_PATH_UNIFIED = True

# Input feature channels (Auto-detected if 'auto')
# Set to 'auto' to let dataset detect (e.g. 6 for coord+color)
INPUT_CHANNELS = 6

# Grid size for voxelization (Auto-detected)
GRID_SIZE = 0.015


# =============================================================================
# DETECTION SETTINGS (Only used if NUM_CLASSES_DET > 0)
# =============================================================================

# 3D Box Regression Configuration
DETECTION_CONFIG = {
    # Point Cloud Range [x_min, y_min, z_min, x_max, y_max, z_max]
    'POINT_CLOUD_RANGE': [-1000, -1000, -1000, 1000, 1000, 1000],
    
    # Anchor/Mean Sizes for Box Regression (Order matches class IDs)
    # Set to 'auto' to calculate from training data
    'MEAN_SIZE': 'auto',
    
    # Extra width for target assignment (ignore region around GT)
    'GT_EXTRA_WIDTH': [0.2, 0.2, 0.2],    # [x, y, z] margins for target assignment

    # Loss Configuration (Weighted to balance Seg vs Det)
    'LOSS_CONFIG': {
        'LOSS_REG': 'weighted-smooth-l1',
        'LOSS_WEIGHTS': {
            'point_cls_weight': 1.0,
            'point_box_weight': 1.0,  # Restored to 1.0 to force regression learning
            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    }
}

# Loss Weighting
# Balance factor for detection loss relative to segmentation loss
# loss = seg_loss + (det_loss * DETECTION_LOSS_WEIGHT)
# Issue 3: Increased to 2.0x to balance detection vs segmentation
# Segmentation operates on ~100k points while detection on ~100-1000 positive points
DETECTION_LOSS_WEIGHT = 2.0

# Automatic Multi-Task Balancing
# Options: 'none' (static weights), 'uncertainty' (Kendall 2018), 'gradnorm' (Chen 2018)
# Issue 5: Using static balancing for stability and predictability
LOSS_BALANCING_METHOD = 'uncertainty'

# GradNorm Settings (Only used if LOSS_BALANCING_METHOD == 'gradnorm')
GRADNORM_ALPHA = 1.5  # Asymmetry parameter (common value: 1.5)


# =============================================================================
# ADVANCED SETTINGS (rarely need to change)
# =============================================================================

# Number of workers for data loading (0 = main thread only)
# Set to 0 for Windows compatibility and debugging
NUM_WORKERS = 0

# Labels to ignore during training
IGNORED_LABELS = [-1]

# Use mixed precision training (faster on GPU)
USE_AMP = True

# Early stopping: stop if no improvement for N epochs (0 to disable)
EARLY_STOPPING_PATIENCE = 100

# Auto-resume: automatically resume from last checkpoint if exists
AUTO_RESUME = True


# =============================================================================
# AUTOMATIC ADAPTIVE CONFIG OVERRIDE
# =============================================================================
# Loads 'dataset_status.json' generated by auto_optimize.py if it exists.
# This ensures training uses the latest optimized parameters without manual editing.

_json_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_status.json')
if os.path.exists(_json_config_path):
    import json
    try:
        with open(_json_config_path, 'r') as _f:
            _opt_params = json.load(_f)
            
        print(f"⚙️  Loading Optimized Config from: {_json_config_path}")
        
        # Override Globals
        if 'NUM_CLASSES_SEG' in _opt_params: NUM_CLASSES_SEG = _opt_params['NUM_CLASSES_SEG']
        if 'NUM_CLASSES_DET' in _opt_params: NUM_CLASSES_DET = _opt_params['NUM_CLASSES_DET']
        # Note: CLASS_NAMES mapping from JSON might need care if list order matters vs classes.json
        if 'CLASS_NAMES' in _opt_params: CLASS_NAMES = _opt_params['CLASS_NAMES'] 
        if 'GRID_SIZE' in _opt_params: GRID_SIZE = float(_opt_params['GRID_SIZE'])
        if 'INPUT_CHANNELS' in _opt_params: INPUT_CHANNELS = _opt_params['INPUT_CHANNELS']
        if 'BATCH_SIZE' in _opt_params: BATCH_SIZE = _opt_params['BATCH_SIZE']
        
        # Deep Override for MEAN_SIZE
        if 'MEAN_SIZE' in _opt_params and NUM_CLASSES_DET > 0:
            DETECTION_CONFIG['MEAN_SIZE'] = _opt_params['MEAN_SIZE']
            
        # Handle Class Weights
        if 'CLASS_WEIGHTS' in _opt_params:
            _cw = _opt_params['CLASS_WEIGHTS']
            if isinstance(_cw, list): 
                import numpy as _np
                CLASS_WEIGHTS = _np.array(_cw, dtype=_np.float32)
            else:
                CLASS_WEIGHTS = _cw

    except Exception as _e:
        print(f"⚠️  Failed to load optimized config: {_e}")

# =============================================================================
# VALIDATION LOGIC
# =============================================================================
def _validate_config():
    """Confirms critical settings are valid on import."""
    errors = []
    warnings = []
    
    if NUM_CLASSES_SEG <= 0:
        errors.append(f"NUM_CLASSES_SEG must be > 0, got {NUM_CLASSES_SEG}")
        
    if BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE must be > 0, got {BATCH_SIZE}")
        
    if EPOCHS <= 0:
        errors.append(f"EPOCHS must be > 0, got {EPOCHS}")
        
    if LEARNING_RATE <= 0:
        errors.append(f"LEARNING_RATE must be > 0, got {LEARNING_RATE}")
        
    if WEIGHT_DECAY < 0:
        errors.append(f"WEIGHT_DECAY cannot be negative, got {WEIGHT_DECAY}")
        
    valid_variants = ['nano', 'micro', 'tiny', 'small', 'base', 'large']
    if MODEL_VARIANT not in valid_variants:
        errors.append(f"Invalid MODEL_VARIANT '{MODEL_VARIANT}'. Must be one of {valid_variants}")
        
    if not os.path.exists(DATA_PATH):
        warnings.append(f"DATA_PATH check failed: '{DATA_PATH}' does not exist.")
        
    if NUM_CLASSES_DET > 0:
        if DETECTION_CONFIG is None or 'MEAN_SIZE' not in DETECTION_CONFIG:
            errors.append("NUM_CLASSES_DET > 0 but MEAN_SIZE missing in DETECTION_CONFIG")
            
    # Print Validation Results
    if warnings:
        for w in warnings: print(f"⚠️  Config Warning: {w}")
        
    if errors:
        print("\n❌  CRITICAL CONFIG ERRORS:")
        for e in errors: print(f"   - {e}")
        print("Please fix Custom/config.py before training.\n")
        # raise ValueError("Invalid Configuration") # Optional: crash on error

_validate_config()
