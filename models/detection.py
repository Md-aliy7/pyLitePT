"""
LitePT Detection Adapter
========================
Integrates detection heads with LitePT backbone.

This module adapts the LitePT 'Point' structure to the dictionary format
expected by detection heads. Uses pcdet_lite for pure Python detection
(no CUDA dependencies required).
"""

import torch
import torch.nn as nn
from addict import Dict

# Import from pcdet_lite (pure Python, no CUDA dependencies)
from pcdet_lite.detection_heads import PointHeadBox



class LitePTDetectionHead(nn.Module):
    """
    Detection head adapter for LitePT backbone.
    
    Converts LitePT Point structure to detection head format and
    provides a unified interface for training and inference.
    """
    
    def __init__(self, in_channels, num_classes, model_cfg=None):
        """
        Args:
            in_channels: Number of input feature channels from backbone
            num_classes: Number of object classes to detect
            model_cfg: Optional configuration dict for detection head
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Default config if not provided
        if model_cfg is None:
            # Generic default config - mean_size should be provided by config or auto-calculated
            self.model_cfg = Dict({
                'CLS_FC': [256, 256],
                'REG_FC': [256, 256],
                'TARGET_CONFIG': {
                    'BOX_CODER': 'PointResidualCoder',
                    'BOX_CODER_CONFIG': {
                        'use_mean_size': True,
                        'mean_size': [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]  # Generic defaults (will be overridden by config)
                    },
                    'GT_EXTRA_WIDTH': [0.2, 0.2, 0.2]
                },
                'LOSS_CONFIG': {
                    'LOSS_REG': 'smooth-l1',
                    'LOSS_WEIGHTS': {
                        'point_cls_weight': 1.0,
                        'point_box_weight': 1.0,
                        'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    }
                }
            })
        else:
            self.model_cfg = model_cfg if isinstance(model_cfg, Dict) else Dict(model_cfg)
            
        # Create detection head
        self.head = PointHeadBox(
            num_class=num_classes,
            input_channels=in_channels,
            model_cfg=self.model_cfg,
            predict_boxes_when_training=True
        )

    def forward(self, point, gt_boxes=None, original_coord=None, original_batch=None):
        """
        Forward pass through detection head.
        
        Args:
            point: LitePT Point object with:
                - .feat: (N, C) point features
                - .coord: (N, 3) point coordinates (may be modified by pooling)
                - .batch: (N,) batch indices
            gt_boxes: Optional (B, M, 8) ground truth boxes
                [x, y, z, dx, dy, dz, heading, label]
            original_coord: Optional (N, 3) original point coordinates before pooling
            original_batch: Optional (N,) original batch indices
            
        Returns:
            dict: Detection results including:
                - point_cls_scores: (N,) classification scores
                - batch_cls_preds: (N, num_class) class predictions
                - batch_box_preds: (N, 7) box predictions (if not training)
        """
        # Use original coordinates if provided, otherwise fall back to point coords
        # This is critical because point.coord may be averaged during encoder-decoder
        coord = original_coord if original_coord is not None else point.coord
        batch = original_batch if original_batch is not None else point.batch
        
        # Prepare input dict for PointHeadBox
        batch_idx = batch.float().unsqueeze(1)  # (N, 1)
        point_coords = torch.cat([batch_idx, coord], dim=1)  # (N, 4)
        
        input_dict = {
            'point_features': point.feat,
            'point_coords': point_coords,
            'batch_size': int(batch.max().item() + 1)
        }
        
        if gt_boxes is not None:
            input_dict['gt_boxes'] = gt_boxes
            
        # Forward through detection head
        ret_dict = self.head(input_dict)
        
        if not self.training:
            ret_dict = self.post_process(ret_dict)
        
        return ret_dict

    def post_process(self, ret_dict, use_nms=False, nms_thresh=0.7, 
                      score_thresh=0.05, max_boxes=100):
        """
        Apply post-processing to predictions with per-class rotated BEV NMS.
        
        Args:
            ret_dict: Detection output dictionary
            use_nms: bool, whether to use full rotated NMS (slower but more accurate)
            nms_thresh: float, IoU threshold for NMS (default 0.7)
            score_thresh: float, minimum score threshold (default 0.3)
            max_boxes: int, maximum boxes per batch (default 100)
            
        Returns:
            ret_dict: Updated dictionary with filtered predictions
        """
        # Issue 1: Import rotated BEV NMS instead of AABB approximation
        from pcdet_lite.iou3d_nms_utils import nms_gpu
        
        if 'batch_box_preds' not in ret_dict:
            return ret_dict
        
        box_preds = ret_dict['batch_box_preds']
        cls_scores = ret_dict['point_cls_scores']
        cls_preds = ret_dict['batch_cls_preds']
        batch_idx = ret_dict['batch_index']
        
        # Iterate over batch
        batch_size = int(batch_idx.max().item() + 1)
        final_boxes = []
        final_scores = []
        final_cls_preds = []
        final_batch_idx = []
        
        for k in range(batch_size):
            mask = (batch_idx == k)
            cur_boxes = box_preds[mask]
            cur_scores = cls_scores[mask]
            cur_cls_preds = cls_preds[mask]
            
            if cur_scores.dim() == 2: 
                cur_scores = cur_scores.squeeze(1)
            
            # Step 1: Score threshold filter
            score_mask = cur_scores > score_thresh
            
            if score_mask.sum() == 0:
                continue
                
            cur_boxes = cur_boxes[score_mask]
            cur_scores = cur_scores[score_mask]
            cur_cls_preds = cur_cls_preds[score_mask]
            
            # Step 2: TopK filter for efficiency
            if cur_boxes.shape[0] > max_boxes:
                topk_scores, topk_inds = torch.topk(cur_scores, k=max_boxes)
                cur_boxes = cur_boxes[topk_inds]
                cur_scores = topk_scores
                cur_cls_preds = cur_cls_preds[topk_inds]
            
            # Step 3: Per-Class Rotated BEV NMS
            # Issue 2: Per-class NMS to prevent cross-suppression between different classes
            if use_nms and cur_boxes.shape[0] > 1:
                # Get predicted class for each box
                pred_classes = cur_cls_preds.argmax(dim=-1)
                
                # Per-class NMS: run NMS independently for each class
                selected_indices = []
                for class_id in range(self.num_classes):
                    class_mask = (pred_classes == class_id)
                    if class_mask.sum() == 0:
                        continue
                    
                    # Get boxes and scores for this class
                    class_boxes = cur_boxes[class_mask]
                    class_scores = cur_scores[class_mask]
                    class_indices = torch.where(class_mask)[0]
                    
                    # Issue 1: Using rotated BEV NMS instead of AABB approximation
                    # This correctly handles rotated 3D boxes using polygon intersection
                    keep_idx, _ = nms_gpu(class_boxes, class_scores, nms_thresh)
                    
                    # Map back to original indices
                    selected_indices.append(class_indices[keep_idx])
                
                # Merge results from all classes
                if len(selected_indices) > 0:
                    selected = torch.cat(selected_indices)
                else:
                    selected = torch.tensor([], dtype=torch.long, device=cur_boxes.device)
            else:
                # TopK already applied or single box
                selected = torch.arange(cur_boxes.shape[0], device=cur_boxes.device)
            
            if len(selected) > 0:
                final_boxes.append(cur_boxes[selected])
                final_scores.append(cur_scores[selected])
                final_cls_preds.append(cur_cls_preds[selected])
                final_batch_idx.append(torch.full((len(selected),), k, 
                                                   device=batch_idx.device, 
                                                   dtype=batch_idx.dtype))
        
        if len(final_boxes) > 0:
            ret_dict['batch_box_preds'] = torch.cat(final_boxes, dim=0)
            ret_dict['point_cls_scores'] = torch.cat(final_scores, dim=0)
            ret_dict['batch_cls_preds'] = torch.cat(final_cls_preds, dim=0)
            ret_dict['batch_index'] = torch.cat(final_batch_idx, dim=0)
        else:
            ret_dict['batch_box_preds'] = torch.zeros((0, 7), device=box_preds.device)
            ret_dict['point_cls_scores'] = torch.zeros((0,), device=box_preds.device)
            ret_dict['batch_cls_preds'] = torch.zeros((0, self.num_classes), device=box_preds.device)
            ret_dict['batch_index'] = torch.zeros((0,), device=batch_idx.device)
            
        return ret_dict

    def get_loss(self, tb_dict=None):
        """
        Calculate detection loss.
        
        Args:
            tb_dict: Optional tensorboard dict for logging
            
        Returns:
            loss: Scalar detection loss
            tb_dict: Updated tensorboard dict
        """
        return self.head.get_loss(tb_dict)
