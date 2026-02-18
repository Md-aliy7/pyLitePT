"""
=============================================================================
DETECTION POST-PROCESSING
=============================================================================
Reusable post-processing utilities for 3D detection predictions.
Extracts confidence filtering + NMS logic used by both visualize.py and
detection.py.
"""

import torch
import numpy as np


def filter_and_nms(boxes, scores, conf_thresh=0.5, iou_thresh=0.1, 
                   cls_preds=None, max_boxes=None):
    """
    Apply confidence filtering and NMS to detection predictions.
    
    Uses torchvision's axis-aligned BEV NMS on the AABB of rotated boxes
    for fast, GPU-compatible NMS.
    
    Args:
        boxes: (N, 7+) [x, y, z, dx, dy, dz, heading, ...]
        scores: (N,) confidence scores
        conf_thresh: Minimum confidence threshold
        iou_thresh: IoU threshold for NMS suppression
        cls_preds: (N, C) optional class logits for per-box class labels
        max_boxes: Optional maximum number of boxes to return
        
    Returns:
        dict with keys:
            'boxes': (K, 7) filtered boxes
            'scores': (K,) filtered scores  
            'labels': (K,) predicted class labels (0-based)
            'keep_indices': (K,) indices into the original input
    """
    if boxes is None or len(boxes) == 0:
        return {
            'boxes': np.zeros((0, 7), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32),
            'labels': np.zeros(0, dtype=np.int64),
            'keep_indices': np.zeros(0, dtype=np.int64),
        }

    # Ensure float tensors
    boxes = boxes.float()
    scores = scores.float()
    if scores.dim() == 2:
        scores = scores.squeeze(1)

    # 1. Confidence filter
    mask = scores > conf_thresh
    if mask.sum() == 0:
        return {
            'boxes': np.zeros((0, 7), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32),
            'labels': np.zeros(0, dtype=np.int64),
            'keep_indices': np.zeros(0, dtype=np.int64),
        }
    
    boxes_f = boxes[mask]
    scores_f = scores[mask]

    # 2. Compute AABB for BEV NMS
    x, y = boxes_f[:, 0], boxes_f[:, 1]
    dx, dy = boxes_f[:, 3], boxes_f[:, 4]
    heading = boxes_f[:, 6]

    cos_h = torch.abs(torch.cos(heading))
    sin_h = torch.abs(torch.sin(heading))

    w_aabb = dx * cos_h + dy * sin_h
    h_aabb = dx * sin_h + dy * cos_h

    x1 = x - w_aabb / 2
    y1 = y - h_aabb / 2
    x2 = x + w_aabb / 2
    y2 = y + h_aabb / 2

    boxes_bev = torch.stack([x1, y1, x2, y2], dim=1)

    # 3. NMS via torchvision
    import torchvision.ops
    keep_idx = torchvision.ops.nms(boxes_bev, scores_f, iou_threshold=iou_thresh)

    if max_boxes is not None and len(keep_idx) > max_boxes:
        keep_idx = keep_idx[:max_boxes]

    # 4. Extract results
    result_boxes = boxes_f[keep_idx].cpu().numpy()
    result_scores = scores_f[keep_idx].cpu().numpy()

    # 5. Class labels
    if cls_preds is not None and cls_preds.dim() == 2 and cls_preds.shape[0] == boxes.shape[0]:
        cls_preds_f = cls_preds[mask][keep_idx]
        result_labels = cls_preds_f.argmax(dim=-1).cpu().numpy()
    else:
        result_labels = np.zeros(len(result_boxes), dtype=np.int64)

    # 6. Map indices back to original
    orig_indices = torch.where(mask)[0][keep_idx].cpu().numpy()

    return {
        'boxes': result_boxes,
        'scores': result_scores,
        'labels': result_labels,
        'keep_indices': orig_indices,
    }
