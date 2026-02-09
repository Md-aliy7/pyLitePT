"""
=============================================================================
UNIFIED TRAINING - SEMANTIC SEGMENTATION + 3D DETECTION
=============================================================================
Run this script to train LitePT for both segmentation and detection.
Automatically handles input dimensions, class weights, and hyperparameter checks.

Usage:
    python Custom/train.py [args]

    Args:
        --mode MODE             segmentation, detection, or unified
        --format FMT            ply or npy
        --data_path PATH        Path to dataset
        --num_classes_seg N     Number of segmentation classes
        --num_classes_det N     Number of detection classes
        --model_variant V       Model variant (nano, micro, tiny, small, base, large)
        --epochs N              Number of epochs
        --batch_size N          Batch size
        --lr LR                 Learning rate
        --weight_decay WD       Weight decay
        --num_workers N         Number of data loading workers
        --results_dir DIR       Directory to save results

Features:
    - Semantic Segmentation (Weighted CrossEntropy)
    - 3D Object Detection (pcdet_lite integration)
    - Auto input channel detection
    - Auto class weight calculation
    - Learning rate scheduling (OneCycleLR with correct per-batch stepping)
    - Full Validation Loop
    - Verbose per-batch logging
=============================================================================
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Progress bar

# Standardized seeding for reproducibility
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Standardized Error Logger/Notifier
# (Loggers removed as they were unused)

# Add project root to path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
if parent not in sys.path:
    sys.path.insert(0, parent)
if current not in sys.path:
    sys.path.insert(0, current)

# Import config and dataset
import config as cfg
import Custom.auto_optimize as auto_opt
import importlib

# Ensure optimized config exists and reload if generated
if auto_opt.ensure_optimization(cfg.DATA_PATH):
    print("🔄 Reloading config after auto-optimization...")
    importlib.reload(cfg)

from dataset import CustomDataset

# Import LitePT components
from hybrid_backend import setup_backends
from datasets.utils import collate_fn
# core module imports
from core import LitePTUnifiedCustom

# Import Detection Metrics
try:
    from metrics.detection_metrics import DetectionMetrics
except ImportError as e:
    print(f"Warning: Could not import DetectionMetrics: {e}")
    DetectionMetrics = None


# ============================================================================
# UTILS
# ============================================================================

class EMATracker:
    """Exponential Moving Average Tracker for smoothing noisy metrics."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
    
    def update(self, new_val):
        if not self.initialized:
            # First value, or if we get NaN/Inf, reset
            if not np.isfinite(new_val):
                 return self.value
            self.value = new_val
            self.initialized = True
        else:
            if np.isfinite(new_val):
                self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value


# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, seg_criterion, device, epoch, scaler=None, num_det_classes=0, class_names=None):
    """Train one epoch with verbose per-batch logging and memory optimizations."""
    import gc  # Memory management
    model.train()
    
    total_metrics = {'loss': 0, 'seg_loss': 0, 'det_loss': 0, 'seg_acc': 0}
    count = 0
    epoch_start = time.time()
    
    use_amp = scaler is not None and device.type == 'cuda'
    # Auto-detect best AMP dtype
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    # Initialize detection metrics if detection is enabled
    det_metrics = None
    if model.det_head is not None and num_det_classes > 0 and DetectionMetrics is not None:
        det_metrics = DetectionMetrics(
            num_classes=num_det_classes,
            iou_thresholds=[0.25, 0.5, 0.75],
            class_names=class_names
        )
    
    # Progress bar
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [TRAIN]", leave=False)
    
    for i, batch in pbar:
        batch_start = time.time()
        
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        n_points = batch['coord'].shape[0] if 'coord' in batch else 0
        
        # Optimized zero_grad (faster than zero_grad())
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=amp_dtype):
            # Forward
            outputs = model(batch)
            
            # 1. Segmentation Loss
            if model.seg_head is not None:
                seg_loss = seg_criterion(outputs['seg_logits'], batch['segment'])
            else:
                seg_loss = torch.tensor(0.0, device=device)
            
            # 2. Detection Loss
            if model.det_head is not None:
                det_loss, det_tb_dict = model.det_head.get_loss()

                # Accumulate detection metrics for training (careful with overhead)
                if 'point_pos_num' in det_tb_dict:
                    total_metrics['det_pos'] = total_metrics.get('det_pos', 0) + det_tb_dict['point_pos_num']

                if det_metrics is not None:
                     # Detach to avoid graph retention
                    with torch.no_grad():
                        det_out = outputs 
                        if 'batch_box_preds' in det_out and 'gt_boxes' in batch:
                            pred_boxes = det_out['batch_box_preds'].detach().float().cpu().numpy()
                            pred_scores = det_out.get('point_cls_scores', torch.zeros(len(pred_boxes))).detach().float().cpu().numpy()
                            
                            if 'batch_cls_preds' in det_out:
                                cls_preds = det_out['batch_cls_preds'].argmax(dim=-1).detach().cpu().numpy()
                                if len(cls_preds) == len(pred_boxes):
                                    pred_labels = cls_preds
                                else:
                                    pred_labels = np.zeros(len(pred_boxes), dtype=np.int32)
                            else:
                                pred_labels = np.zeros(len(pred_boxes), dtype=np.int32)
                            
                            gt_boxes = batch['gt_boxes'].detach().float().cpu().numpy()
                            if gt_boxes.ndim == 3: gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[-1])
                            valid_mask = gt_boxes[:, 3:6].sum(axis=1) > 0
                            gt_boxes = gt_boxes[valid_mask]
                            gt_labels = gt_boxes[:, 7].astype(np.int32) if gt_boxes.shape[1] > 7 else np.zeros(len(gt_boxes), dtype=np.int32)
                            # Issue 6: Increased threshold to 0.3 for higher quality metrics
                            # Filter noisy boxes before geometry evaluation
                            score_mask = pred_scores > 0.3
                            pred_boxes = pred_boxes[score_mask]
                            pred_scores = pred_scores[score_mask]
                            
                            # Labels are already 0-based from dataset - no conversion needed

                            if len(pred_boxes) > 0 or len(gt_boxes) > 0:
                                if len(pred_boxes) > 500: # Final safety ceiling for training metrics
                                    top_idx = np.argsort(-pred_scores)[:500]
                                    pred_boxes = pred_boxes[top_idx]
                                    pred_scores = pred_scores[top_idx]
                                    pred_labels = pred_labels[score_mask][top_idx]
                                else:
                                    pred_labels = pred_labels[score_mask]

                                det_metrics.add_batch(
                                    pred_boxes[:, :7] if len(pred_boxes) > 0 else np.zeros((0, 7)),
                                    pred_scores,
                                    pred_labels,
                                    gt_boxes[:, :7] if len(gt_boxes) > 0 else np.zeros((0, 7)),
                                    gt_labels
                                )
            else:
                det_loss = torch.tensor(0.0, device=device)
            
            # Total Loss
            # Total Loss Calculation
            loss_balancing_method = getattr(cfg, 'LOSS_BALANCING_METHOD', 'uncertainty')
            
            if loss_balancing_method == 'uncertainty' and hasattr(model, 'log_vars'):
                # Kendall et al. (CVPR 2018): Multi-Task Learning Using Uncertainty
                # Loss = 1/(2*sigma^2) * L + log(sigma)
                # We use log_var = log(sigma^2)
                # Loss = exp(-log_var) * L + 0.5 * log_var
                
                log_vars = model.log_vars
                
                # Segmentation (Index 0)
                precision_seg = torch.exp(-log_vars[0])
                loss_seg_weighted = 0.5 * seg_loss * precision_seg + 0.5 * log_vars[0]
                    
                # Detection (Index 1)
                precision_det = torch.exp(-log_vars[1])
                loss_det_weighted = 0.5 * det_loss * precision_det + 0.5 * log_vars[1]
                
                loss = loss_seg_weighted + loss_det_weighted
                
                total_metrics['w_seg'] = precision_seg.item()
                total_metrics['w_det'] = precision_det.item()
                
            elif loss_balancing_method == 'gradnorm':
                # GradNorm Implementation (Chen et al. 2018)
                
                # 0. Initialize initial losses if first batch
                if model.initial_losses is None:
                    model.initial_losses = torch.tensor([seg_loss.item(), det_loss.item()], device=device)
                
                # 1. Weighted Loss for Model Update
                # Renormalize weights so they sum to 2 (number of tasks)
                # "The weighted loss L is computed as \sum w_i(t) * L_i(t)"
                # "In every step, we also renormalize w_i(t) so that \sum w_i(t) = T"
                
                # Detach weights for the model graph (treated as constants for model backprop)
                w_seg = model.task_weights[0].detach()
                w_det = model.task_weights[1].detach()
                
                # Ensure renormalization (optional but standard in GradNorm)
                # norm_factor = 2.0 / (w_seg + w_det + 1e-6)
                # w_seg = w_seg * norm_factor
                # w_det = w_det * norm_factor
                
                weighted_seg_loss = w_seg * seg_loss
                weighted_det_loss = w_det * det_loss
                loss = weighted_seg_loss + weighted_det_loss
                
                total_metrics['w_seg'] = w_seg.item()
                total_metrics['w_det'] = w_det.item()
                
                # 2. GradNorm Weight Update (Deferred)
                # We need to compute gradients w.r.t shared layer *before* graph is consumed by main backward
                # and before we update any weights.
                
                # We store these purely for the GradNorm step later
                shared_layer = model.get_last_shared_layer()
                g1_norm, g2_norm = None, None
                
                if shared_layer is not None:
                     # Compute raw gradients for each task
                     # We use retain_graph=True so main backward can still run
                     # Note: seg_loss and det_loss are the raw losses
                     
                     # We need to handle AMP scaling if active, but autograd.grad usually handles unscaled?
                     # No, if use_amp, losses are small. 
                     # GradNorm paper uses standard gradients. 
                     # If we use scaler.scale(loss), we get scaled gradients.
                     # For GradNorm, we should probably use unscaled gradients or consistently scaled ones.
                     # Let's use unscaled to match the paper's definition of "gradient norm".
                     # However, if using AMP, unscaled gradients might be underflowing? 
                     # Usually we can just use the raw loss.
                     
                     g1 = torch.autograd.grad(seg_loss, shared_layer, retain_graph=True, create_graph=True)[0]
                     g2 = torch.autograd.grad(det_loss, shared_layer, retain_graph=True, create_graph=True)[0]
                     
                     g1_norm = torch.norm(g1, 2)
                     g2_norm = torch.norm(g2, 2)
            else:
                # Standard Static Weighting
                det_weight = getattr(cfg, 'DETECTION_LOSS_WEIGHT', 1.0)
                loss = seg_loss + (det_loss * det_weight)
        
        # Backward with gradient clipping
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # --- GRADNORM UPDATE STEP ---
            if loss_balancing_method == 'gradnorm' and g1_norm is not None:
                 model.weight_optimizer.zero_grad()
                 
                 # Calculate Target
                 g_avg = (g1_norm + g2_norm) / 2.0
                 
                 # Relative Inverse Training Rate
                 l1_hat = seg_loss.detach() / (model.initial_losses[0] + 1e-8)
                 l2_hat = det_loss.detach() / (model.initial_losses[1] + 1e-8)
                 l_avg = (l1_hat + l2_hat) / 2.0
                 
                 inv_rate1 = l1_hat / l_avg
                 inv_rate2 = l2_hat / l_avg
                 
                 alpha = getattr(cfg, 'GRADNORM_ALPHA', 1.5)
                 t1 = g_avg * (inv_rate1 ** alpha)
                 t2 = g_avg * (inv_rate2 ** alpha)
                 
                 # L_grad: Minimize difference between (w_i * g_i) and target
                 # Note: We use the attached task_weights here to get gradients for them
                 l_grad = torch.abs(model.task_weights[0] * g1_norm.detach() - t1.detach()) + \
                          torch.abs(model.task_weights[1] * g2_norm.detach() - t2.detach())
                 
                 l_grad.backward()
                 model.weight_optimizer.step()
                 
                 # Renormalize
                 with torch.no_grad():
                    w_sum = model.task_weights.sum()
                    model.task_weights.div_(w_sum / 2.0)
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(cfg, 'GRAD_CLIP_NORM', 10.0))
            scaler.step(optimizer)
            scaler.update()
        else:
             loss.backward()
             
             # --- GRADNORM UPDATE STEP (Non-AMP) ---
             if loss_balancing_method == 'gradnorm' and g1_norm is not None:
                 model.weight_optimizer.zero_grad()
                 g_avg = (g1_norm + g2_norm) / 2.0
                 l1_hat = seg_loss.detach() / (model.initial_losses[0] + 1e-8)
                 l2_hat = det_loss.detach() / (model.initial_losses[1] + 1e-8)
                 l_avg = (l1_hat + l2_hat) / 2.0
                 inv_rate1, inv_rate2 = l1_hat / l_avg, l2_hat / l_avg
                 alpha = getattr(cfg, 'GRADNORM_ALPHA', 1.5)
                 t1, t2 = g_avg * (inv_rate1 ** alpha), g_avg * (inv_rate2 ** alpha)
                 
                 l_grad = torch.abs(model.task_weights[0] * g1_norm.detach() - t1.detach()) + \
                          torch.abs(model.task_weights[1] * g2_norm.detach() - t2.detach())
                 
                 l_grad.backward()
                 model.weight_optimizer.step()
                 with torch.no_grad():
                    model.task_weights.div_(model.task_weights.sum() / 2.0)
             
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(cfg, 'GRAD_CLIP_NORM', 10.0))
             optimizer.step()

        
        # Stability Monitoring (Check Feature Norms)
        # As per Issue #4/#6: Large norms indicate explosion
        if i % 100 == 0:  # Check every 100 batches
            try:
                # Traverse back from output point to parents
                point = outputs['point']
                max_norm = 0.0
                depth = 0
                while True:
                    if hasattr(point, 'feat') and point.feat is not None:
                        norm = point.feat.norm(p=2, dim=-1).mean().item()
                        max_norm = max(max_norm, norm)
                        if norm > 100.0:
                            print(f"⚠️  WARNING: High Feature Norm ({norm:.1f}) at depth {depth}. Possible instability!")
                    
                    if hasattr(point, 'pooling_parent'):
                         point = point.pooling_parent
                         depth += 1
                    else:
                        break
            except Exception:
                pass # Fail silently in monitoring
        
        if scheduler:
            scheduler.step()
        
        # Metrics & Logging
        batch_loss = loss.item()
        batch_seg_loss = seg_loss.item()
        batch_det_loss = det_loss.item()
        
        total_metrics['loss'] += batch_loss
        total_metrics['seg_loss'] += batch_seg_loss
        total_metrics['det_loss'] += batch_det_loss
        
        # Segmentation Accuracy
        acc = 0.0
        if model.seg_head is not None:
            with torch.no_grad():
                preds = outputs['seg_logits'].argmax(1)
                mask = batch['segment'] != seg_criterion.ignore_index
                if mask.sum() > 0:
                    acc = (preds[mask] == batch['segment'][mask]).float().mean().item()
        total_metrics['seg_acc'] += acc
        count += 1
        
        current_lr = scheduler.get_last_lr()[0] if scheduler else 0.0
        
        # Update progress bar
        avg_loss = total_metrics['loss'] / count
        avg_seg_acc = total_metrics['seg_acc'] / count
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'seg_loss': f'{batch_seg_loss:.4f}',
            'det_loss': f'{batch_det_loss:.4f}',
            'acc': f'{avg_seg_acc*100:.1f}%',
            'pos': f"{total_metrics.get('det_pos', 0) / count:.1f}",
            'lr': f'{current_lr:.6f}'
        })
        
        # Periodic memory cleanup (every 50 batches)
        if i % 50 == 49 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    pbar.close()
    
    epoch_time = time.time() - epoch_start
    avg_metrics = {k: v / count for k, v in total_metrics.items()}
    
    det_mAP_50 = 0.0
    det_mAP_25 = 0.0
    det_recall_50 = 0.0
    det_recall_25 = 0.0
    if det_metrics is not None:
        det_results = det_metrics.compute()
        det_mAP_50 = det_results.get('mAP@0.5', 0.0)
        det_mAP_25 = det_results.get('mAP@0.25', 0.0)
        det_recall_50 = det_results.get('recall@0.5', 0.0)
        det_recall_25 = det_results.get('recall@0.25', 0.0)

    # Structured Logging
    det_mAP_75 = 0.0
    det_recall_75 = 0.0
    
    log_msg = (
        f"  [Epoch {epoch}] {epoch_time:.1f}s | "
        f"Loss: {avg_metrics['loss']:.4f} "
        f"(Seg: {avg_metrics['seg_loss']:.4f}, Det: {avg_metrics['det_loss']:.4f}) | "
        f"SegAcc: {avg_metrics['seg_acc']*100:.2f}%"
    )
    
    if det_metrics is not None:
        det_mAP_75 = det_results.get('mAP@0.75', 0.0)
        det_recall_75 = det_results.get('recall@0.75', 0.0)
        
        log_msg += f"\n    Det mAP@0.5:  {det_mAP_50*100:.2f}% | Recall@0.5:  {det_recall_50*100:.2f}%"
        log_msg += f"\n    Det mAP@0.25: {det_mAP_25*100:.2f}% | Recall@0.25: {det_recall_25*100:.2f}%"
        log_msg += f"\n    Det mAP@0.75: {det_mAP_75*100:.2f}% | Recall@0.75: {det_recall_75*100:.2f}%"
        log_msg += f"\n    Avg Pos: {total_metrics.get('det_pos', 0) / count:.1f}"

    if 'w_seg' in avg_metrics and 'w_det' in avg_metrics:
        log_msg += f"\n    [Auto-Weighting] Seg Weight: {avg_metrics['w_seg']:.4f}, Det Weight: {avg_metrics['w_det']:.4f}"

    print(log_msg)
    
    # Final cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return avg_metrics['loss'], avg_metrics['seg_acc'], det_mAP_50, det_recall_50


def validate_one_epoch(model, loader, seg_criterion, device, num_det_classes=0, class_names=None):
    """Validate one epoch with detection accuracy metrics and memory optimizations."""
    import gc  # Memory management
    
    # CRITICAL FIX for Single-Scene Verification
    # If dealing with a single scene (overfitting test), model.eval() fails because 
    # BatchNorm running stats lag behind. We force model.train() to use batch stats.
    if len(loader.dataset) <= 1:
        model.train()
    else:
        model.eval()
    
    total_metrics = {'loss': 0, 'seg_loss': 0, 'det_loss': 0, 'seg_acc': 0, 'det_pos': 0}
    count = 0
    start = time.time()
    
    # Initialize detection metrics if detection is enabled
    det_metrics = None
    if model.det_head is not None and num_det_classes > 0 and DetectionMetrics is not None:
        det_metrics = DetectionMetrics(
            num_classes=num_det_classes,
            iou_thresholds=[0.25, 0.5, 0.75],
            class_names=class_names
        )
    
    # Progress bar for validation
    pbar = tqdm(enumerate(loader), total=len(loader), desc="[VAL]", leave=False)
    
    # Use inference_mode for maximum efficiency (faster than no_grad)
    with torch.inference_mode():
        for i, batch in pbar:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward
            outputs = model(batch)
            
            # 1. Segmentation Loss
            if model.seg_head is not None:
                seg_loss = seg_criterion(outputs['seg_logits'], batch['segment'])
            else:
                seg_loss = torch.tensor(0.0, device=device)
            
            # 2. Detection Loss & Metrics
            if model.det_head is not None:
                det_loss, det_tb_dict = model.det_head.get_loss()
                
                # Accumulate Positives (Use get_loss output, NOT outputs)
                if 'point_pos_num' in det_tb_dict:
                     total_metrics['det_pos'] = total_metrics.get('det_pos', 0) + det_tb_dict['point_pos_num']
                
                # Accumulate detection metrics
                if det_metrics is not None:
                    # Model merges det_out into outputs, so check keys directly
                    det_out = outputs 
                    
                    if 'batch_box_preds' in det_out and 'gt_boxes' in batch:
                        pred_boxes = det_out['batch_box_preds'].float().cpu().numpy()
                        pred_scores = det_out.get('point_cls_scores', torch.zeros(len(pred_boxes))).float().cpu().numpy()
                        
                        # Issue 6: Increased threshold to 0.3 for higher quality metrics
                        # Filter noisy boxes before greedy evaluation
                        score_mask = pred_scores > 0.3
                        pred_boxes = pred_boxes[score_mask]
                        pred_scores = pred_scores[score_mask]
                        
                        # Extract class from predictions
                        if 'batch_cls_preds' in det_out:
                            cls_preds = det_out['batch_cls_preds'].argmax(dim=-1).cpu().numpy()
                            if len(cls_preds) == len(det_out['batch_box_preds']):
                                pred_labels = cls_preds[score_mask]
                            else:
                                pred_labels = np.zeros(len(pred_boxes), dtype=np.int32)
                        else:
                            pred_labels = np.zeros(len(pred_boxes), dtype=np.int32)
                        
                        
                        # Get ground truth boxes and labels
                        gt_boxes = batch['gt_boxes'].float().cpu().numpy()
                        # Flatten batch
                        if gt_boxes.ndim == 3: gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[-1])
                        
                        # Filter valid
                        valid_mask = gt_boxes[:, 3:6].sum(axis=1) > 0
                        gt_boxes = gt_boxes[valid_mask]
                        gt_labels = gt_boxes[:, 7].astype(np.int32) if gt_boxes.shape[1] > 7 else np.zeros(len(gt_boxes), dtype=np.int32)
                        
                        # Labels are already 0-based from dataset - no conversion needed

                        if len(pred_boxes) > 0 or len(gt_boxes) > 0:
                            det_metrics.add_batch(
                                pred_boxes[:, :7] if len(pred_boxes) > 0 else np.zeros((0, 7)),
                                pred_scores,
                                pred_labels,
                                gt_boxes[:, :7] if len(gt_boxes) > 0 else np.zeros((0, 7)),
                                gt_labels
                            )

            else:
                det_loss = torch.tensor(0.0, device=device)
            
            det_weight = getattr(cfg, 'DETECTION_LOSS_WEIGHT', 1.0)
            loss = seg_loss + (det_loss * det_weight)
            
            # Metrics
            total_metrics['loss'] += loss.item()
            total_metrics['seg_loss'] += seg_loss.item()
            total_metrics['det_loss'] += det_loss.item()
            
            # Segmentation Accuracy
            acc = 0.0
            if model.seg_head is not None:
                preds = outputs['seg_logits'].argmax(1)
                mask = batch['segment'] != seg_criterion.ignore_index
                if mask.sum() > 0:
                    acc = (preds[mask] == batch['segment'][mask]).float().mean().item()
            total_metrics['seg_acc'] += acc
            count += 1
            
            # Update progress bar
            avg_loss = total_metrics['loss'] / count
            avg_seg_acc = total_metrics['seg_acc'] / count
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_seg_acc*100:.1f}%'})
            
    pbar.close()
    
    val_time = time.time() - start
    avg_metrics = {k: v / count for k, v in total_metrics.items()} if count > 0 else total_metrics
    
    det_results = {}
    if det_metrics is not None:
        det_results = det_metrics.compute()
    
    print(f"  Validation Completed in {val_time:.1f}s")
    print(f"    Val Loss: {avg_metrics['loss']:.4f} | Seg: {avg_metrics['seg_loss']:.4f} | Det: {avg_metrics['det_loss']:.4f}")
    print(f"    Val Seg Accuracy: {avg_metrics['seg_acc']*100:.2f}%")
    
    if det_metrics is not None:
        val_mAP_50 = det_results.get('mAP@0.5', 0.0)
        val_mAP_25 = det_results.get('mAP@0.25', 0.0)
        val_mAP_75 = det_results.get('mAP@0.75', 0.0)
        val_recall_50 = det_results.get('recall@0.5', 0.0)
        val_recall_25 = det_results.get('recall@0.25', 0.0)
        val_recall_75 = det_results.get('recall@0.75', 0.0)
        
        print(f"    Val Det mAP@0.5:  {val_mAP_50*100:.2f}% | Recall@0.5:  {val_recall_50*100:.2f}%")
        print(f"    Val Det mAP@0.25: {val_mAP_25*100:.2f}% | Recall@0.25: {val_recall_25*100:.2f}%")
        print(f"    Val Det mAP@0.75: {val_mAP_75*100:.2f}% | Recall@0.75: {val_recall_75*100:.2f}%")
        print(f"    Val Avg Positives: {total_metrics.get('det_pos', 0) / count:.1f}")
    
    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return avg_metrics['loss'], avg_metrics['seg_acc'], det_results


def main(args):
    # Setup
    print("=" * 80)
    print("LitePT UNIFIED TRAINING - Segmentation + Detection".center(80))
    print("=" * 80)
    
    setup_backends(verbose=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configuration Overrides
    epochs = args.epochs or cfg.EPOCHS
    batch_size = args.batch_size or cfg.BATCH_SIZE
    lr = args.lr or cfg.LEARNING_RATE
    
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    
    # 1. Dataset Initialization
    print("\n[DATASET]")
    print(f"Loading data from: {cfg.DATA_PATH}")

    # OPTIMIZATION: Check if we need to auto-optimize first
    from auto_optimize import ensure_optimization
    import importlib
    
    if ensure_optimization(cfg.DATA_PATH):
        # RELOAD CONFIG to pick up new values if optimization occurred
        importlib.reload(cfg)
        # Re-apply args that might have been overwritten by reload
        cfg.DATA_PATH = args.data_path 
    
    train_set = CustomDataset(cfg.DATA_PATH, split='train', cfg=cfg, data_format=args.format)
    val_set = CustomDataset(cfg.DATA_PATH, split='val', cfg=cfg, data_format=args.format)
    
    if len(train_set) == 0:
        print("ERROR: No training scenes found!")
        return
        
    # Auto-detect input channels
    input_channels = train_set.input_channels
    if cfg.INPUT_CHANNELS != 'auto':
        if input_channels != cfg.INPUT_CHANNELS:
            print(f"WARNING: Dataset has {input_channels} channels but config expects {cfg.INPUT_CHANNELS}")
        input_channels = cfg.INPUT_CHANNELS
        
    print(f"Input Channels: {input_channels}")
    
    # Auto-calculate class weights
    class_weights = None
    if cfg.NUM_CLASSES_SEG > 0:
        if isinstance(cfg.CLASS_WEIGHTS, str) and cfg.CLASS_WEIGHTS == 'auto':
            class_weights = train_set.calculate_class_weights(cfg.NUM_CLASSES_SEG)
            print(f"Auto Class Weights: {np.round(class_weights, 3)}")
        else:
            class_weights = cfg.CLASS_WEIGHTS
            print(f"Manual Class Weights: {class_weights}")
    
    # Auto-calculate detection mean sizes
    if cfg.NUM_CLASSES_DET > 0:
        det_config = cfg.DETECTION_CONFIG
        mean_size = det_config.get('MEAN_SIZE')
        if isinstance(mean_size, str) and mean_size == 'auto':
            print("Auto-calculating box mean sizes...")
            mean_sizes = train_set.calculate_mean_sizes(cfg.NUM_CLASSES_DET)
            det_config['MEAN_SIZE'] = mean_sizes
            print(f"Auto Mean Sizes: {mean_sizes}")

    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    # 2. Model Creation
    print("\n[MODEL]")
    print(f"Creating Unified Model: LitePT-{cfg.MODEL_VARIANT}")
    
    model = LitePTUnifiedCustom(
        in_channels=input_channels,
        num_classes_seg=cfg.NUM_CLASSES_SEG,
        num_classes_det=cfg.NUM_CLASSES_DET,
        variant=cfg.MODEL_VARIANT,
        det_config=cfg.DETECTION_CONFIG
    )
    
    model.to(device)
    
    # Model Verification / Param Counting
    from verify_params import log_model_info
    log_model_info(model)
    
    # 3. Training Setup
    # Issue 4: Detection head uses 2x learning rate for faster convergence
    # Create separate parameter groups for backbone and detection head
    param_groups = []
    
    # Backbone and segmentation head parameters (base LR)
    backbone_params = []
    for name, param in model.named_parameters():
        if 'det_head' not in name:
            backbone_params.append(param)
    
    param_groups.append({
        'params': backbone_params,
        'lr': lr,
        'weight_decay': cfg.WEIGHT_DECAY
    })
    
    # Detection head parameters (2x LR)
    if model.det_head is not None:
        det_params = list(model.det_head.parameters())
        param_groups.append({
            'params': det_params,
            'lr': lr * 2.0,
            'weight_decay': cfg.WEIGHT_DECAY
        })
        print(f"[Optimizer] Detection head LR: {lr * 2.0:.6f} (2x backbone)")
    
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
    
    seg_criterion = None
    if cfg.NUM_CLASSES_SEG > 0:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights is not None else None
        seg_criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=cfg.IGNORED_LABELS[0])
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and cfg.USE_AMP else None

    # GradNorm Setup
    gradnorm_optimizer = None
    if getattr(cfg, 'LOSS_BALANCING_METHOD', 'uncertainty') == 'gradnorm':
        print("[GradNorm] Initializing task weights...")
        # Attach to model for easy access in train loop (optional, but convenient)
        model.task_weights = torch.ones(2, device=device, requires_grad=True)
        model.weight_optimizer = torch.optim.Adam([model.task_weights], lr=0.025)
        model.initial_losses = None # Will be set in first batch
        gradnorm_optimizer = model.weight_optimizer # Alias for saving

    
    # 4. Training Loop
    print("\n[TRAINING]")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    
    best_val_acc = 0.0  # Combined score (seg_acc + det_mAP)
    best_seg_acc = 0.0
    best_det_mAP = 0.0
    
    # EMA Tracker for Detection
    det_ema = EMATracker(alpha=0.3)
    
    start_epoch = 1
    patience_counter = 0
    early_stop_patience = getattr(cfg, 'EARLY_STOPPING_PATIENCE', 0)
    
    # Auto-resume from last checkpoint
    last_ckpt_path = os.path.join(cfg.RESULTS_DIR, 'last_unified_model.pth')
    if getattr(cfg, 'AUTO_RESUME', False) and os.path.exists(last_ckpt_path):
        print(f"\n[AUTO-RESUME] Loading checkpoint: {last_ckpt_path}")
        try:
            ckpt = torch.load(last_ckpt_path, map_location='cpu', weights_only=False)
            state = ckpt['model_state_dict']
            
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
                print(f"  Partial load detected. Mismatched layers ignored:")
                for m in msg: print("    " + m)
            
            model.load_state_dict(filtered_state, strict=False)
            
            if not msg:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                
                # Load GradNorm state if available
                if gradnorm_optimizer is not None and 'gradnorm_weights' in ckpt:
                    try:
                        model.task_weights.data = ckpt['gradnorm_weights'].to(device)
                        if 'gradnorm_optimizer_state_dict' in ckpt:
                            gradnorm_optimizer.load_state_dict(ckpt['gradnorm_optimizer_state_dict'])
                        if 'gradnorm_initial_losses' in ckpt:
                             model.initial_losses = ckpt['gradnorm_initial_losses'].to(device)
                        print("  > GradNorm state restored.")
                    except Exception as e:
                        print(f"  > Warning: Failed to load GradNorm state: {e}")
                    
            else:
                 print("  > Optimizer state skipped due to architecture change (starting fresh optimizer)")
            start_epoch = ckpt.get('epoch', 0) + 1
            
            # Load EMA state if available
            det_ema.value = ckpt.get('val_det_mAP_ema', ckpt.get('val_det_mAP', 0.0))
            if det_ema.value > 0: det_ema.initialized = True
            
            # Recalculate Best Score using NEW formula based on MODE
            b_acc = ckpt.get('val_accuracy', 0)
            b_map50 = ckpt.get('val_det_mAP', 0)
            b_map50_ema = det_ema.value
            
            if args.mode == 'segmentation':
                best_val_acc = b_acc
            elif args.mode == 'detection':
                # Weighted Detection Score (Pure EMA of mAP@0.5)
                best_val_acc = b_map50_ema
            else: # unified
                # New Stable Formula: 50% Seg + 50% EMA(mAP@0.5)
                best_val_acc = (b_acc * 0.5) + (b_map50_ema * 0.5)
            
            best_seg_acc = b_acc
            best_det_mAP = b_map50
            print(f"  Resuming from epoch {start_epoch}, best_seg_acc={best_seg_acc*100:.1f}%, best_det_mAP={best_det_mAP*100:.1f}%")
            print(f"  > Metrics recalibrated to new Weighted Score ({args.mode}) with EMA: {best_val_acc:.4f}")
            # Recreate scheduler at correct step
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs,
                last_epoch=(start_epoch - 1) * len(train_loader) - 1
            )
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            start_epoch = 1
    
    for epoch in range(start_epoch, epochs + 1):
        # TRAIN
        train_loss, train_acc, train_det_mAP, train_det_recall = train_one_epoch(
            model, train_loader, optimizer, scheduler, seg_criterion, device, epoch, scaler,
            num_det_classes=cfg.NUM_CLASSES_DET,
            class_names=getattr(cfg, 'CLASS_NAMES_DET', None)
        )
        
        # VALIDATE
        if len(val_set) > 0:
            val_loss, val_acc, det_results = validate_one_epoch(
                model, val_loader, seg_criterion, device,
                num_det_classes=cfg.NUM_CLASSES_DET,
                class_names=getattr(cfg, 'CLASS_NAMES_DET', None)
            )
            
            # Extract metrics safe for scoring
            det_mAP_50 = det_results.get('mAP@0.5', 0.0)
            det_mAP_25 = det_results.get('mAP@0.25', 0.0)
            det_mAP_75 = det_results.get('mAP@0.75', 0.0)
            det_recall = det_results.get('recall@0.5', 0.0) # For logging
        else:
            val_acc = 0.0
            det_mAP_50 = 0.0
            det_mAP_25 = 0.0
            det_mAP_75 = 0.0
            det_recall = 0.0
            det_results = {}
            print("  Skipping validation (no validation data)")
        
        # Update EMA
        det_mAP_50_ema = det_ema.update(det_mAP_50)
        
        # Save checkpoints
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': train_acc,
            'train_det_mAP': train_det_mAP,
            'val_accuracy': val_acc,
            'val_det_mAP': det_mAP_50,
            'val_det_mAP_ema': det_mAP_50_ema,  # Save EMA state
            'val_det_mAP_25': det_mAP_25, 
            'val_det_mAP_75': det_mAP_75, 
            'val_det_recall': det_recall,
            'val_det_recall': det_recall,
            'config': {k: v for k, v in cfg.__dict__.items() if k.isupper()}
        }
        
        # Save GradNorm state
        if gradnorm_optimizer is not None:
             state['gradnorm_weights'] = model.task_weights.detach().cpu()
             state['gradnorm_optimizer_state_dict'] = gradnorm_optimizer.state_dict()
             if model.initial_losses is not None:
                 state['gradnorm_initial_losses'] = model.initial_losses.detach().cpu()
        
        # Save last
        torch.save(state, os.path.join(cfg.RESULTS_DIR, 'last_unified_model.pth'))
        
        # Save best (COMPOSITE SCORE) based on MODE
        if args.mode == 'segmentation':
            combined_score = val_acc
        elif args.mode == 'detection':
            # Weighted Detection Score (Pure EMA)
            combined_score = det_mAP_50_ema
        else:
            # Unified
            # New Stable Formula: 50% SegAcc + 50% EMA(mAP@0.5)
            # Removes noisy @0.75 and @0.25 from selection logic
            combined_score = (val_acc * 0.5) + (det_mAP_50_ema * 0.5)
        
        # 1. Best Unified (Composite)
        if combined_score > best_val_acc:
            print(f"  >>> New Best Unified Model! Score: {combined_score:.4f} (Prev: {best_val_acc:.4f})")
            print(f"      Breakdown: Seg={val_acc*100:.1f}%, EMA(mAP@50)={det_mAP_50_ema*100:.1f}% (Raw={det_mAP_50*100:.1f}%)")
            best_val_acc = combined_score
            patience_counter = 0
            torch.save(state, os.path.join(cfg.RESULTS_DIR, 'best_unified_model.pth'))
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{early_stop_patience if early_stop_patience > 0 else 'Inf'} | Best Score: {best_val_acc:.4f}")
            
        # 2. Best Segmentation (Pure Accuracy)
        if val_acc > best_seg_acc:
             print(f"  >>> New Best Segmentation Model! Acc: {val_acc*100:.2f}%")
             best_seg_acc = val_acc
             torch.save(state, os.path.join(cfg.RESULTS_DIR, 'best_seg_model.pth'))
             
        # 3. Best Detection (Pure mAP@0.5)
        if det_mAP_50 > best_det_mAP:
             print(f"  >>> New Best Detection Model! mAP@0.5: {det_mAP_50*100:.2f}%")
             best_det_mAP = det_mAP_50
             torch.save(state, os.path.join(cfg.RESULTS_DIR, 'best_det_model.pth'))
             
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            print(f"\n  Early stopping triggered (no improvement for {early_stop_patience} epochs)")
            break
            
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE".center(80))
    print("=" * 80)
    print(f"Best Val Seg Accuracy: {best_seg_acc*100:.1f}%")
    print(f"Best Val Det mAP@0.5: {best_det_mAP*100:.1f}%")
    print(f"Results saved to: {cfg.RESULTS_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LitePT Unified (Seg + Det)')
    
    # Task Mode
    parser.add_argument('--mode', type=str, default='unified', 
                        choices=['segmentation', 'detection', 'unified'],
                        help='Training mode: segmentation only, detection only, or unified (both)')
    
    # Dataset & Model
    parser.add_argument('--data_path', type=str, default=cfg.DATA_PATH, help='Path to dataset')
    parser.add_argument('--num_classes_seg', type=int, default=cfg.NUM_CLASSES_SEG, help='Number of segmentation classes')
    parser.add_argument('--num_classes_det', type=int, default=cfg.NUM_CLASSES_DET, help='Number of detection classes')
    parser.add_argument('--model_variant', type=str, default=cfg.MODEL_VARIANT, choices=['micro', 'nano', 'small', 'base'], help='Model variant')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=cfg.WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS, help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--results_dir', type=str, default=cfg.RESULTS_DIR, help='Directory to save results')
    
    # Data Format
    parser.add_argument('--format', type=str, default='npy', 
                        choices=['ply', 'npy'],
                        help='Data format: ply for labelCloud PLY files, npy for NPY folder format (default: npy)')
    
    args = parser.parse_args()
    
    # Handle mode: override class counts based on mode
    if args.mode == 'segmentation':
        args.num_classes_det = 0
        print(f"Mode: SEGMENTATION ONLY (detection disabled)")
    elif args.mode == 'detection':
        args.num_classes_seg = 0
        print(f"Mode: DETECTION ONLY (segmentation disabled)")
    else:
        print(f"Mode: UNIFIED (segmentation + detection)")
    
    # Apply args to config
    cfg.DATA_PATH = args.data_path
    cfg.NUM_CLASSES_SEG = args.num_classes_seg
    cfg.NUM_CLASSES_DET = args.num_classes_det
    cfg.MODEL_VARIANT = args.model_variant
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.NUM_WORKERS = args.num_workers
    cfg.RESULTS_DIR = args.results_dir
    
    main(args)
