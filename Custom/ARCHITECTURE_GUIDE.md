# LitePT Architecture Guide

## Overview

This guide explains the LitePT architecture options available in this implementation and helps you choose the right configuration for your task.

---

## 🎯 Three Questions to Answer

1. **What task do you want to perform?**
   - Segmentation only
   - Detection only
   - Both (unified mode)

2. **What's your priority?**
   - Best performance (dual-path unified)
   - Parameter efficiency (single-path unified)
   - Speed (smaller variants)

3. **What's your hardware?**
   - High-end GPU → Use larger variants (base, large)
   - Mid-range GPU → Use small or tiny
   - Limited resources → Use nano or micro

---

## 📊 Model Variants

**Choose a size - the system automatically selects the optimal architecture based on your mode!**

| Variant | Multi-Stage Params | Single-Stage Params | Speed | Use Case |
|---------|-------------------|---------------------|-------|----------|
| `nano` | ~1M | ~0.5M | ⚡⚡⚡⚡⚡ | Quick experiments, edge devices |
| `micro` | ~2M | ~1M | ⚡⚡⚡⚡ | Small datasets, prototyping |
| `tiny` | ~6M | ~2M | ⚡⚡⚡ | Balanced development |
| `small` | ~12M | ~5M | ⚡⚡ | **Production (Recommended)** ⭐ |
| `base` | ~45M | ~15M | ⚡ | High accuracy needs |
| `large` | ~86M | ~30M | ⚡ | Maximum accuracy |

### Architecture Selection (Automatic)

The system automatically selects the optimal architecture based on your mode:

**Multi-Stage Architecture** (with downsampling):
```
Input → Encoder (5 stages, downsample 4x) → Decoder (4 stages, upsample) → Output
```
- Used for: Segmentation mode, Unified single-path mode
- Features: Hierarchical features, encoder-decoder structure
- Best for: Dense prediction tasks

**Single-Stage Architecture** (no downsampling):
```
Input → Encoder (single resolution, no downsampling) → Output
```
- Used for: Detection mode (automatic), Unified dual-path mode (detection branch)
- Features: Preserves spatial resolution, encoder-only
- Best for: Small object detection

**You don't need to choose the architecture - just set your mode and the system handles it!**

---

## 🏗️ Architecture Modes

### 1. Segmentation Mode

**Configuration:**
```python
# Custom/config.py
NUM_CLASSES_SEG = 20  # Your number of classes
NUM_CLASSES_DET = 0   # Disable detection
MODEL_VARIANT = 'small'  # Just choose size
```

**System automatically uses:**
- Multi-stage architecture with downsampling
- Encoder-decoder with hierarchical features
- Variant: `small` (12M params)

**Architecture:**
```
Input → Multi-stage Encoder-Decoder → Segmentation Head → Per-point labels
```

**Recommended Sizes:**
- Development: `nano` or `micro`
- Production: `small` ⭐
- High accuracy: `base` or `large`

---

### 2. Detection Mode

**Configuration:**
```python
# Custom/config.py
NUM_CLASSES_SEG = 0   # Disable segmentation
NUM_CLASSES_DET = 5   # Your number of classes
MODEL_VARIANT = 'small'  # Just choose size
```

**System automatically uses:**
- Single-stage architecture (no downsampling)
- Encoder-only, preserves spatial resolution
- Variant: `single_stage_small` (5M params) - automatic!

**Architecture:**
```
Input → Single-stage Encoder → Detection Head → 3D bounding boxes
```

**Why single-stage?**
- Preserves spatial resolution (no downsampling)
- Better for small object detection
- Follows author's recommendation

**Recommended Sizes:**
- Development: `nano` or `micro`
- Production: `small` ⭐
- High accuracy: `base` or `large`

---

### 2. Detection Mode

**Configuration:**
```python
# Custom/config.py
NUM_CLASSES_SEG = 0   # Disable segmentation
NUM_CLASSES_DET = 5   # Your number of classes
MODEL_VARIANT = 'small'  # Just choose size - system uses single-stage automatically
```

**System automatically uses:**
- Single-stage architecture (no downsampling)
- Encoder-only, preserves spatial resolution
- Variant: `single_stage_small` (5M params) - automatic!

**Architecture:**
```
Input → Single-stage Encoder → Detection Head → 3D bounding boxes
```

**Why single-stage?**
- Preserves spatial resolution (no downsampling)
- Better for small object detection
- Follows author's recommendation

**Recommended Sizes:**
- Development: `nano` or `micro`
- Production: `small` ⭐
- High accuracy: `base` or `large`

---

### 3. Unified Mode - Single-Path

**Configuration:**
```python
# Custom/config.py
NUM_CLASSES_SEG = 20  # Segmentation classes
NUM_CLASSES_DET = 5   # Detection classes
MODEL_VARIANT = 'small'  # Multi-stage variant
USE_DUAL_PATH_UNIFIED = False  # Single backbone
```

**Architecture:**
```
Input → Multi-stage Encoder-Decoder → Features
                                        ├→ Segmentation Head
                                        └→ Detection Head
```

**Pros:**
- ✅ Parameter efficient (one backbone)
- ✅ Faster training
- ✅ Shared feature learning

**Cons:**
- ⚠️ Detection gets downsampled features (not optimal for small objects)

**Use when:**
- Parameter budget is tight
- Training time is limited
- Objects are relatively large

---

### 4. Unified Mode - Dual-Path ⭐ OPTIMAL

**Configuration:**
```python
# Custom/config.py
NUM_CLASSES_SEG = 20  # Segmentation classes
NUM_CLASSES_DET = 5   # Detection classes
MODEL_VARIANT = 'small'  # Base variant name
USE_DUAL_PATH_UNIFIED = True  # Enable dual-path
```

**Architecture:**
```
        ┌→ Multi-stage Encoder-Decoder → Segmentation Head
Input ──┤
        └→ Single-stage Encoder → Detection Head
```

**How it works:**
- Segmentation branch: Uses `MODEL_VARIANT` (e.g., `small`)
- Detection branch: Automatically uses `single_stage_{MODEL_VARIANT}` (e.g., `single_stage_small`)
- Each task gets its optimal architecture!

**Pros:**
- ✅ Optimal architecture for BOTH tasks
- ✅ Segmentation gets hierarchical features
- ✅ Detection gets high-resolution features
- ✅ Follows author's best practices

**Cons:**
- ⚠️ More parameters (~1.5x)
- ⚠️ Slightly slower training

**Use when:**
- You need best performance on both tasks
- You have sufficient GPU memory
- Production deployment with quality requirements

**Parameter Counts (Dual-Path):**
- `nano`: ~1M + ~0.5M = ~1.5M total
- `small`: ~12M + ~5M = ~17M total
- `base`: ~45M + ~15M = ~60M total

---

## 🎓 Decision Tree

```
What's your task?
│
├─ Segmentation only
│  └─ Use: MODEL_VARIANT = 'small'
│     System uses: multi-stage architecture
│
├─ Detection only
│  └─ Use: MODEL_VARIANT = 'small'
│     System uses: single-stage architecture (automatic!)
│
└─ Both (Unified)
   │
   ├─ Priority: Best Performance
   │  └─ Use: MODEL_VARIANT = 'small' + USE_DUAL_PATH_UNIFIED = True ⭐
   │     System uses: Seg (multi-stage) + Det (single-stage)
   │
   └─ Priority: Parameter Efficiency
      └─ Use: MODEL_VARIANT = 'small' + USE_DUAL_PATH_UNIFIED = False
         System uses: multi-stage for both
```

---

## 📝 Configuration Examples

### Example 1: Segmentation for Indoor Scenes
```python
# Custom/config.py
NUM_CLASSES_SEG = 13  # ScanNet classes
NUM_CLASSES_DET = 0
MODEL_VARIANT = 'small'
BATCH_SIZE = 4
EPOCHS = 100
```

### Example 2: Detection for Autonomous Driving
```python
# Custom/config.py
NUM_CLASSES_SEG = 0
NUM_CLASSES_DET = 3  # Car, Pedestrian, Cyclist
MODEL_VARIANT = 'small'  # System automatically uses single-stage
BATCH_SIZE = 2
EPOCHS = 80
```

### Example 3: Unified for Robotics (Optimal)
```python
# Custom/config.py
NUM_CLASSES_SEG = 20  # Scene understanding
NUM_CLASSES_DET = 10  # Object detection
MODEL_VARIANT = 'small'
USE_DUAL_PATH_UNIFIED = True  # Optimal performance
BATCH_SIZE = 2  # Larger model needs smaller batch
EPOCHS = 150
```

### Example 4: Unified for Edge Device (Efficient)
```python
# Custom/config.py
NUM_CLASSES_SEG = 10
NUM_CLASSES_DET = 5
MODEL_VARIANT = 'nano'
USE_DUAL_PATH_UNIFIED = False  # Parameter efficient
BATCH_SIZE = 8
EPOCHS = 100
```

---

## 🔧 Advanced Configuration

### Multi-Task Loss Balancing

The implementation supports automatic loss balancing:

```python
# Custom/config.py
LOSS_BALANCING_METHOD = 'uncertainty'  # Kendall et al. 2018 (Recommended)
# LOSS_BALANCING_METHOD = 'gradnorm'   # Chen et al. 2018
# LOSS_BALANCING_METHOD = 'none'       # Static weights
```

**Uncertainty Weighting (Recommended):**
- Learns task-specific uncertainty
- Automatically balances seg/det losses
- No hyperparameter tuning needed

**GradNorm:**
- Balances gradient magnitudes
- Requires tuning `GRADNORM_ALPHA`
- More complex but can be more stable

**Static Weights:**
- Manual control via `DETECTION_LOSS_WEIGHT`
- Simple but requires tuning

---

## 🚀 Performance Tips

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

## 📊 Expected Performance

### Segmentation (ScanNet-like dataset)
| Variant | mIoU | Training Time (100 epochs) |
|---------|------|---------------------------|
| nano | ~60% | ~2 hours |
| micro | ~65% | ~3 hours |
| small | ~70% | ~6 hours |
| base | ~75% | ~20 hours |

### Detection (KITTI-like dataset)
| Variant | mAP@0.5 | Training Time (80 epochs) |
|---------|---------|--------------------------|
| nano | ~50% | ~1.5 hours |
| small | ~65% | ~4 hours |
| base | ~75% | ~12 hours |

*Note: System automatically uses single-stage architecture for detection mode*

### Unified (Custom dataset)
| Mode | Seg mIoU | Det mAP | Training Time |
|------|----------|---------|---------------|
| Single-path (small) | ~68% | ~60% | ~8 hours |
| Dual-path (small) | ~70% | ~65% | ~10 hours |

*Note: Actual performance depends on your dataset, hardware, and hyperparameters.*

---

## 🐛 Troubleshooting

### Issue: Detection mAP is 0%
**Solution:**
1. Verify `gt_boxes.npy` files exist and are correct format
2. Ensure `DETECTION_LOSS_WEIGHT > 0`
3. System automatically uses single-stage for detection mode

### Issue: OOM during training
**Solution:**
1. Reduce `BATCH_SIZE`
2. Use smaller variant (nano or micro)
3. Enable `USE_AMP = True`

### Issue: Slow training
**Solution:**
1. Use smaller variant
2. Enable `USE_AMP = True`
3. Set `TRAIN_STEPS_PER_EPOCH` to limit steps

### Issue: Poor segmentation accuracy
**Solution:**
1. Check class weights are balanced
2. Verify using segmentation mode (not detection mode)
3. Increase model size (small → base)

### Issue: Poor detection accuracy
**Solution:**
1. System automatically uses single-stage for detection mode
2. Check anchor sizes (`MEAN_SIZE` in config)
3. Verify coordinate system (XYZ vs XZY)

---

## 📚 Further Reading

- **[Custom/README.md](README.md)**: Quick start guide
- **[Custom/TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)**: Detailed workflow
- **[MODEL_VARIANTS_GUIDE.md](../MODEL_VARIANTS_GUIDE.md)**: Complete variant reference
- **[IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)**: Implementation details

---

## ✅ Quick Reference

### Segmentation Mode
```python
MODEL_VARIANT = 'small'  # System uses multi-stage
NUM_CLASSES_SEG = 20
NUM_CLASSES_DET = 0
```

### Detection Mode
```python
MODEL_VARIANT = 'small'  # System automatically uses single-stage
NUM_CLASSES_SEG = 0
NUM_CLASSES_DET = 5
```

### Unified Mode (Optimal)
```python
MODEL_VARIANT = 'small'  # System uses both architectures
USE_DUAL_PATH_UNIFIED = True
NUM_CLASSES_SEG = 20
NUM_CLASSES_DET = 5
```

---

**Last Updated:** 2026-02-10
**Version:** 2.0
