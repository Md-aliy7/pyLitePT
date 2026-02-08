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
            self.model_cfg = Dict({
                'CLS_FC': [256, 256],
                'REG_FC': [256, 256],
                'TARGET_CONFIG': {
                    'BOX_CODER': 'PointResidualCoder',
                    'BOX_CODER_CONFIG': {
                        'use_mean_size': True,
                        'mean_size': [
                            [10.0, 10.0, 10.0],  # cube
                            [16.0, 16.0, 16.0],  # sphere
                            [12.0, 12.0, 15.0],  # cylinder
                            [14.0, 14.0, 15.0],  # cone
                            [12.0, 12.0, 12.0],  # pyramid
                            [22.0, 22.0, 6.0],   # torus
                        ]
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
                      score_thresh=0.01, max_boxes=100):
        """
        Apply post-processing to predictions.
        
        Args:
            ret_dict: Detection output dictionary
            use_nms: bool, whether to use full rotated NMS (slower but more accurate)
            nms_thresh: float, IoU threshold for NMS (default 0.7)
            score_thresh: float, minimum score threshold (default 0.3)
            max_boxes: int, maximum boxes per batch (default 100)
            
        Returns:
            ret_dict: Updated dictionary with filtered predictions
        """
        if 'batch_box_preds' not in ret_dict:
            return ret_dict
        
        box_preds = ret_dict['batch_box_preds']
        cls_scores = ret_dict['point_cls_scores']
        cls_preds = ret_dict['batch_cls_preds']  # Filter this too
        batch_idx = ret_dict['batch_index']
        
        # Iterate over batch
        batch_size = int(batch_idx.max().item() + 1)
        final_boxes = []
        final_scores = []
        final_cls_preds = []  # Store filtered class logits
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
            
            # Step 3: Optional NMS
            # Step 3: Fast BEV NMS (torchvision)
            if use_nms and cur_boxes.shape[0] > 1:
                # 1. Project to BEV (x, y, dx, dy, heading) -> (x1, y1, x2, y2)
                # Box: [x, y, z, dx, dy, dz, heading]
                x, y = cur_boxes[:, 0], cur_boxes[:, 1]
                dx, dy = cur_boxes[:, 3], cur_boxes[:, 4]
                heading = cur_boxes[:, 6]
                
                # Rotate extents to get AABB size
                cos_h = torch.abs(torch.cos(heading))
                sin_h = torch.abs(torch.sin(heading))
                
                w_aabb = dx * cos_h + dy * sin_h
                h_aabb = dx * sin_h + dy * cos_h
                
                x1 = x - w_aabb / 2
                y1 = y - h_aabb / 2
                x2 = x + w_aabb / 2
                y2 = y + h_aabb / 2
                
                boxes_bev = torch.stack([x1, y1, x2, y2], dim=1)
                
                # 2. Apply NMS
                import torchvision.ops
                keep_idx = torchvision.ops.nms(boxes_bev, cur_scores, iou_threshold=nms_thresh)
                selected = keep_idx
            else:
                # TopK already applied or single box
                selected = torch.arange(cur_boxes.shape[0], device=cur_boxes.device)
            
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
