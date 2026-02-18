"""
=============================================================================
DETECTION METRICS - 3D Object Detection Evaluation
=============================================================================
Provides 3D box IoU calculation, mAP computation, and recall metrics.
Adapted from OpenPCDet's KITTI evaluation but pure Python (no numba).

Usage:
    from metrics.detection_metrics import DetectionMetrics, box3d_iou
    
    metrics = DetectionMetrics(num_classes=7, iou_thresholds=[0.25, 0.5])
    metrics.add_batch(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    results = metrics.compute()
=============================================================================
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """
    Convert 3D boxes [x, y, z, dx, dy, dz, heading] to 8 corner points.
    
    Args:
        boxes: (N, 7+) array with [x, y, z, dx, dy, dz, heading, ...]
        
    Returns:
        corners: (N, 8, 3) array of corner coordinates
    """
    if len(boxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    
    # Template corners for unit box centered at origin
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
    ], dtype=np.float32) / 2  # Half-size template
    
    N = boxes.shape[0]
    corners = np.zeros((N, 8, 3), dtype=np.float32)
    
    for i in range(N):
        cx, cy, cz = boxes[i, 0:3]
        dx, dy, dz = boxes[i, 3:6]
        heading = boxes[i, 6]
        
        # Scale template by box dimensions
        scaled = template * np.array([dx, dy, dz])
        
        # Rotation around Z axis
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rot = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Rotate and translate
        rotated = scaled @ rot.T
        corners[i] = rotated + np.array([cx, cy, cz])
    
    return corners


def box3d_vol(corners: np.ndarray) -> np.ndarray:
    """
    Compute volume of a 3D box from corners.
    
    Args:
        corners: (N, 8, 3) corner coordinates
        
    Returns:
        volumes: (N,) array of volumes
    """
    if len(corners) == 0:
        return np.zeros(0, dtype=np.float32)
    
    # Volume = dx * dy * dz
    # dx = distance between corners 0 and 3
    # dy = distance between corners 0 and 1
    # dz = distance between corners 0 and 4
    dx = np.linalg.norm(corners[:, 0] - corners[:, 3], axis=1)
    dy = np.linalg.norm(corners[:, 0] - corners[:, 1], axis=1)
    dz = np.linalg.norm(corners[:, 0] - corners[:, 4], axis=1)
    
    return dx * dy * dz


def box3d_iou_single(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute approximate 3D IoU between two boxes.
    
    Uses axis-aligned bounding box approximation for speed.
    For rotated boxes, this is an approximation but works well
    when boxes are roughly aligned.
    
    Args:
        box1: (7,) [x, y, z, dx, dy, dz, heading]
        box2: (7,) [x, y, z, dx, dy, dz, heading]
        
    Returns:
        iou: float, IoU value
    """
    # Get corners
    corners1 = boxes_to_corners_3d(box1[None])[0]  # (8, 3)
    corners2 = boxes_to_corners_3d(box2[None])[0]  # (8, 3)
    
    # AABB approximation
    min1 = corners1.min(axis=0)
    max1 = corners1.max(axis=0)
    min2 = corners2.min(axis=0)
    max2 = corners2.max(axis=0)
    
    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]
    
    # Union (Using AABB volumes to be consistent with Intersection)
    vol1 = (max1 - min1).prod()
    vol2 = (max2 - min2).prod()
    union_vol = vol1 + vol2 - inter_vol
    
    if union_vol < 1e-6:
        return 0.0
    
    return float(inter_vol / union_vol)


def box3d_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute 3D IoU matrix between two sets of boxes using Vectorized PyTorch.
    
    Args:
        boxes1: (N, 7+) prediction boxes
        boxes2: (M, 7+) ground truth boxes
        
    Returns:
        iou_matrix: (N, M) IoU values
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    
    # Use pcdet_lite's rotated 3D IoU implementation
    try:
        from pcdet_lite.iou3d_nms_utils import boxes_iou3d_gpu
        
        b1_torch = torch.from_numpy(boxes1[:, :7]).float()
        b2_torch = torch.from_numpy(boxes2[:, :7]).float()
        
        iou_matrix_torch = boxes_iou3d_gpu(b1_torch, b2_torch)
        return iou_matrix_torch.cpu().numpy()
        
    except ImportError:
        # Fallback to loop (only if pcdet_lite is missing, which involves other errors usually)
        print("Warning: pcdet_lite not found, using slow IoU")
        iou_matrix = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                iou_matrix[i, j] = box3d_iou_single(boxes1[i, :7], boxes2[j, :7])
        return iou_matrix


def compute_precision_recall(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray, 
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute precision-recall curve for one class.
    
    Args:
        pred_boxes: (N, 7) predicted boxes
        pred_scores: (N,) confidence scores
        gt_boxes: (M, 7) ground truth boxes
        iou_threshold: IoU threshold for matching
        
    Returns:
        precision: (K,) precision values at each threshold
        recall: (K,) recall values at each threshold
        ap: Average Precision (area under PR curve)
    """
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return np.array([1.0]), np.array([0.0]), 1.0
        else:
            return np.zeros(len(pred_boxes)), np.zeros(len(pred_boxes)), 0.0
    
    if len(pred_boxes) == 0:
        return np.array([0.0]), np.array([0.0]), 0.0
    
    # Sort predictions by score (descending)
    sorted_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]
    
    # Compute IoU matrix
    iou_matrix = box3d_iou_matrix(pred_boxes, gt_boxes)
    
    # Match predictions to ground truth
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    gt_matched = np.zeros(num_gt, dtype=bool)
    
    tp = np.zeros(num_pred)
    fp = np.zeros(num_pred)
    
    for i in range(num_pred):
        # Find best matching GT box
        ious = iou_matrix[i]
        
        # Mask already matched GT boxes
        ious_masked = np.where(gt_matched, 0, ious)
        
        if ious_masked.max() >= iou_threshold:
            best_gt = ious_masked.argmax()
            tp[i] = 1
            gt_matched[best_gt] = True
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / num_gt
    
    # Compute AP (All-Point Interpolation, VOC 2010+ / COCO standard)
    # Make precision monotonically decreasing from right to left
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mrec = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Integrate area under the envelope at recall change points
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    
    return precision, recall, ap


class DetectionMetrics:
    """
    Accumulator for detection metrics over multiple batches.
    
    Usage:
        metrics = DetectionMetrics(num_classes=7, iou_thresholds=[0.25, 0.5])
        
        for batch in dataloader:
            pred_boxes, pred_scores, pred_labels = model(batch)
            gt_boxes, gt_labels = batch['gt_boxes'], batch['gt_labels']
            metrics.add_batch(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        
        results = metrics.compute()
        print(f"mAP@0.5: {results['mAP@0.5']:.3f}")
    """
    
    def __init__(
        self, 
        num_classes: int,
        iou_thresholds: List[float] = [0.25, 0.5, 0.75],
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            num_classes: Number of object classes
            iou_thresholds: List of IoU thresholds for evaluation
            class_names: Optional list of class names for reporting
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Per-class storage
        self.all_pred_boxes = {c: [] for c in range(num_classes)}
        self.all_pred_scores = {c: [] for c in range(num_classes)}
        self.all_gt_boxes = {c: [] for c in range(num_classes)}
        
        # Counts
        self.total_gt = 0
        self.total_pred = 0
    
    def reset(self):
        """Reset all accumulated data."""
        self.all_pred_boxes = {c: [] for c in range(self.num_classes)}
        self.all_pred_scores = {c: [] for c in range(self.num_classes)}
        self.all_gt_boxes = {c: [] for c in range(self.num_classes)}
        self.total_gt = 0
        self.total_pred = 0
    
    def add_batch(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ):
        """
        Add predictions and ground truth from one batch.
        
        Args:
            pred_boxes: (N, 7) predicted boxes [x, y, z, dx, dy, dz, heading]
            pred_scores: (N,) confidence scores
            pred_labels: (N,) predicted class labels (0-indexed)
            gt_boxes: (M, 7) ground truth boxes
            gt_labels: (M,) ground truth class labels (0-indexed)
        """
        # Convert tensors to numpy if needed
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.float().cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.float().cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.float().cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.float().cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.float().cpu().numpy()
        
        self.total_pred += len(pred_boxes)
        self.total_gt += len(gt_boxes)
        
        # Store per-class
        for c in range(self.num_classes):
            pred_mask = pred_labels == c
            gt_mask = gt_labels == c
            
            if pred_mask.any():
                self.all_pred_boxes[c].append(pred_boxes[pred_mask])
                self.all_pred_scores[c].append(pred_scores[pred_mask])
            
            if gt_mask.any():
                self.all_gt_boxes[c].append(gt_boxes[gt_mask])
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with:
                - mAP@{threshold} for each IoU threshold
                - AP_{class}@{threshold} for each class and threshold
                - recall@{threshold} overall recall
                - total_gt: total ground truth count
                - total_pred: total prediction count
        """
        results = {
            'total_gt': self.total_gt,
            'total_pred': self.total_pred,
        }
        
        for iou_thresh in self.iou_thresholds:
            aps = []
            recalls = []
            
            for c in range(self.num_classes):
                # Concatenate all per-class predictions and GT
                if len(self.all_pred_boxes[c]) > 0:
                    pred_boxes = np.concatenate(self.all_pred_boxes[c], axis=0)
                    pred_scores = np.concatenate(self.all_pred_scores[c], axis=0)
                else:
                    pred_boxes = np.zeros((0, 7), dtype=np.float32)
                    pred_scores = np.zeros(0, dtype=np.float32)
                
                if len(self.all_gt_boxes[c]) > 0:
                    gt_boxes = np.concatenate(self.all_gt_boxes[c], axis=0)
                else:
                    gt_boxes = np.zeros((0, 7), dtype=np.float32)
                
                # Compute AP for this class
                precision, recall, ap = compute_precision_recall(
                    pred_boxes, pred_scores, gt_boxes, iou_thresh
                )
                
                aps.append(ap)
                if len(recall) > 0:
                    recalls.append(recall[-1])  # Final recall value
                else:
                    recalls.append(0.0)
                
                # Store per-class AP
                class_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                results[f'AP_{class_name}@{iou_thresh}'] = ap
            
            # Compute mAP (mean over classes with GT)
            valid_aps = [ap for i, ap in enumerate(aps) if len(self.all_gt_boxes[i]) > 0]
            if len(valid_aps) > 0:
                results[f'mAP@{iou_thresh}'] = np.mean(valid_aps)
            else:
                results[f'mAP@{iou_thresh}'] = 0.0
                
            # Overall recall (Average only over classes present in GT)
            valid_recalls = [recall for i, recall in enumerate(recalls) if len(self.all_gt_boxes[i]) > 0]
            
            if len(valid_recalls) > 0:
                results[f'recall@{iou_thresh}'] = np.mean(valid_recalls)
            else:
                results[f'recall@{iou_thresh}'] = 0.0
        
        return results
    
    def get_summary_string(self) -> str:
        """Get a formatted summary string of metrics."""
        results = self.compute()
        
        lines = ["Detection Metrics:"]
        lines.append(f"  GT: {results['total_gt']} | Pred: {results['total_pred']}")
        
        for iou_thresh in self.iou_thresholds:
            mAP = results.get(f'mAP@{iou_thresh}', 0.0)
            recall = results.get(f'recall@{iou_thresh}', 0.0)
            lines.append(f"  mAP@{iou_thresh}: {mAP*100:.1f}% | Recall@{iou_thresh}: {recall*100:.1f}%")
        
        return "\n".join(lines)
