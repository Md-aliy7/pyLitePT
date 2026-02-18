"""
=============================================================================
EVALUATION SCRIPT
=============================================================================
Evaluates trained models on the test set.
Supports both Semantic Segmentation and Unified models.
    
Usage:
    python Custom/evaluate.py --checkpoint results/best_model.pth --split val
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0, parent)
sys.path.insert(0, current)

import config as cfg
from dataset import CustomDataset
from datasets.utils import collate_fn
from hybrid_backend import setup_backends
from core import create_unified_model

# Import Detection Metrics
try:
    from metrics.detection_metrics import DetectionMetrics
except ImportError:
    DetectionMetrics = None

def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint handling unified/legacy keys."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # Handle state dict mismatch if loading legacy into unified or vice versa
    # For now assuming we are evaluating the Unified model trained by train_unified.py
    model.load_state_dict(state_dict, strict=False)
    
    return ckpt.get('epoch', -1), ckpt.get('accuracy', 0.0)

def evaluate(model, loader, device, num_classes, class_names):
    model.eval()
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward
            outputs = model(batch)
            
            # Segmentation predictions
            if 'seg_logits' in outputs:
                preds = outputs['seg_logits'].argmax(1).cpu().numpy()
                targets = batch['segment'].cpu().numpy()
                
                # Filter ignored
                mask = targets != -1
                preds = preds[mask]
                targets = targets[mask]
                
                np.add.at(confusion, (targets, preds), 1)
    
    # Metrics
    ious, precisions, recalls, f1s = [], [], [], []
    per_class = {}
    
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        union = tp + fp + fn
        
        iou = tp / union * 100 if union > 0 else 0
        p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        r = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        name = class_names[c] if c < len(class_names) else f"Class {c}"
        per_class[name] = {'iou': iou, 'precision': p, 'recall': r, 'f1': f1, 'support': int(confusion[c, :].sum())}
        ious.append(iou)
    
    overall = {
        'accuracy': np.trace(confusion) / confusion.sum() * 100 if confusion.sum() > 0 else 0,
        'miou': np.nanmean(ious),
        'total_samples': int(confusion.sum())
    }
    
    # Print
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall['accuracy']:.2f}%")
    print(f"Mean IoU:         {overall['miou']:.2f}%")
    print(f"\n{'-'*60}")
    print(f"{'Class':<12} {'IoU':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*60}")
    for name, m in per_class.items():
        print(f"{name:<12} {m['iou']:>10.2f} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f}")
    print(f"{'='*60}\n")
    
    return overall, per_class


def evaluate_detection(model, loader, device, num_classes, class_names):
    """Evaluate detection performance."""
    if DetectionMetrics is None:
        print("Warning: DetectionMetrics not available. Skipping detection evaluation.")
        return {}, {}
    
    model.eval()
    
    det_metrics = DetectionMetrics(
        num_classes=num_classes,
        iou_thresholds=[0.25, 0.5, 0.75],
        class_names=class_names
    )
    
    print("Evaluating detection...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward
            outputs = model(batch)
            
            # Detection predictions
            if model.det_head is not None and 'batch_box_preds' in outputs:
                det_out = outputs
                if 'batch_box_preds' in det_out and 'gt_boxes' in batch:
                    pred_boxes = det_out['batch_box_preds'].float().cpu().numpy()
                    pred_scores = det_out.get('point_cls_scores', torch.zeros(len(pred_boxes))).float().cpu().numpy()
                    
                    
                    # Filter noisy boxes for faster metric calculation
                    # We use the model's own score_thresh (now 0.01) which is low enough.

                    # Check alignment
                    if len(pred_scores) != len(pred_boxes):
                        print(f"ERROR: Score alignment mismatch! Boxes: {len(pred_boxes)}, Scores: {len(pred_scores)}")
                        pred_scores = np.zeros(len(pred_boxes))
                    
                    if 'batch_cls_preds' in det_out:
                        cls_preds = det_out['batch_cls_preds'].argmax(dim=-1).cpu().numpy()
                        if len(cls_preds) == len(det_out['batch_box_preds']):
                            # pred_labels = cls_preds[score_mask]
                            pred_labels = cls_preds 
                        else:
                            pred_labels = np.zeros(len(pred_boxes), dtype=np.int32)
                    else:
                        pred_labels = np.zeros(len(pred_boxes), dtype=np.int32)
                    
                    gt_boxes = batch['gt_boxes'].float().cpu().numpy()
                    if gt_boxes.ndim == 3:
                        gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[-1])
                    valid_mask = gt_boxes[:, 3:6].sum(axis=1) > 0
                    gt_boxes = gt_boxes[valid_mask]
                    gt_labels = gt_boxes[:, 7].astype(np.int32) if gt_boxes.shape[1] > 7 else np.zeros(len(gt_boxes), dtype=np.int32)
                    
                    if len(pred_boxes) > 0 or len(gt_boxes) > 0:
                        det_metrics.add_batch(
                            pred_boxes[:, :7] if len(pred_boxes) > 0 else np.zeros((0, 7)),
                            pred_scores,
                            pred_labels,
                            gt_boxes[:, :7] if len(gt_boxes) > 0 else np.zeros((0, 7)),
                            gt_labels
                        )
    
    # Compute and print results
    results = det_metrics.compute()
    
    print(f"\n{'='*60}")
    print(f"DETECTION EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total GT: {results['total_gt']} | Total Pred: {results['total_pred']}")
    print(f"\n{'-'*60}")
    for iou_thresh in [0.25, 0.5, 0.75]:
        mAP = results.get(f'mAP@{iou_thresh}', 0.0)
        recall = results.get(f'recall@{iou_thresh}', 0.0)
        print(f"mAP@{iou_thresh}: {mAP*100:.2f}% | Recall@{iou_thresh}: {recall*100:.2f}%")
    print(f"{'='*60}\n")
    
    return results, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to .pth file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--data_format', type=str, default='auto', choices=['auto', 'ply', 'npy'])
    args = parser.parse_args()
    
    setup_backends(verbose=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility (Best Practice)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Dataset
    # OPTIMIZATION: Check if we need to auto-optimize first
    from auto_optimize import ensure_optimization
    import importlib
    
    if ensure_optimization(cfg.DATA_PATH):
        importlib.reload(cfg)
        cfg.DATA_PATH = args.data_path if args.data_format == 'auto' else cfg.DATA_PATH # Respect overrides

    fmt_arg = args.data_format if args.data_format != 'auto' else 'npy'
    dataset = CustomDataset(cfg.DATA_PATH, split=args.split, cfg=cfg, data_format=fmt_arg)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS)
    
    # Model
    # Auto-resolve config params
    input_channels = dataset.input_channels
    if cfg.INPUT_CHANNELS != 'auto': input_channels = cfg.INPUT_CHANNELS
    
    # Auto-calculate MEAN_SIZE if needed
    det_config = cfg.DETECTION_CONFIG.copy() # Copy to avoid modifying global config permanently
    if cfg.NUM_CLASSES_DET > 0 and det_config.get('MEAN_SIZE') == 'auto':
        print("Auto-calculating box mean sizes from dataset...")
        det_config['MEAN_SIZE'] = dataset.calculate_mean_sizes(cfg.NUM_CLASSES_DET)
    
    # Use model factory to ensure architecture matches training
    model = create_unified_model(
        cfg=cfg,
        input_channels=input_channels,
        det_config=det_config,
        device=device
    )
    
    # Checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.RESULTS_DIR, 'best_unified_model.pth')
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        # Optimization: Map to CPU first, strip optimizer, then GC
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        del ckpt
        import gc
        gc.collect()
        
        # Robust loading: filter out shape mismatches
        model_state = model.state_dict()
        filtered_state = {}
        msg = []
        for k, v in state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    msg.append(f"Skipping {k} (shape mismatch: ckpt {v.shape} vs model {model_state[k].shape})")
        
        if msg:
            print("\nWarning: Some weights were not loaded due to shape mismatches:")
            for m in msg: print("  " + m)
            
        model.load_state_dict(filtered_state, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using random weights.")
    
    # Segmentation Evaluation
    seg_acc = 0.0
    if cfg.NUM_CLASSES_SEG > 0:
        print("\n[SEGMENTATION EVALUATION]")
        seg_results, _ = evaluate(model, loader, device, cfg.NUM_CLASSES_SEG, cfg.CLASS_NAMES)
        seg_acc = seg_results['accuracy'] / 100.0 # Convert % to 0-1
    
    # Detection Evaluation
    det_mAP_50 = 0.0
    if cfg.NUM_CLASSES_DET > 0 and model.det_head is not None:
        print("\n[DETECTION EVALUATION]")
        det_class_names = getattr(cfg, 'CLASS_NAMES_DET', cfg.CLASS_NAMES)
        det_results, _ = evaluate_detection(model, loader, device, cfg.NUM_CLASSES_DET, det_class_names)
        det_mAP_50 = det_results.get('mAP@0.5', 0.0)

    # UNIFIED SCORE CALCULATION
    # ==========================
    # Score = 0.5 * SegAcc + 0.5 * mAP@0.5
    unified_score = (seg_acc * 0.5) + (det_mAP_50 * 0.5)
    
    print("\n" + "="*60)
    print("UNIFIED PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Segmentation Accuracy: {seg_acc*100:.2f}%")
    print(f"Detection mAP@0.5:     {det_mAP_50*100:.2f}%")
    print("-" * 30)
    print(f"STABLE UNIFIED SCORE:  {unified_score:.4f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
