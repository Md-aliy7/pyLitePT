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

    def forward(self, point, gt_boxes=None, original_coord=None, original_batch=None, 
                use_nms=True, nms_thresh=0.7, score_thresh=0.01):
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
            use_nms: bool, whether to apply NMS (default True)
            nms_thresh: float, IoU threshold for NMS (default 0.7)
            score_thresh: float, minimum score threshold (default 0.01)
            
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
        ret_dict['batch_index'] = batch
        
        # Semantic Refinement (Ground boxes in segmentation results)
        seg_logits = point.get('seg_logits', None)
        if seg_logits is not None and not self.training:
            # We can refine scores here or inside post_process
            ret_dict['seg_logits'] = seg_logits
            ret_dict['coords'] = coord
        
        if not self.training:
            ret_dict = self.post_process(ret_dict, use_nms=use_nms, nms_thresh=nms_thresh, score_thresh=score_thresh)
        
        return ret_dict

    def post_process(self, ret_dict, use_nms=True, nms_thresh=0.7, 
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
        batch_idx = ret_dict.get('batch_index')
        seg_logits = ret_dict.get('seg_logits')
        coords = ret_dict.get('coords')
        
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
            
            # Step 2: PRE-NMS TOPK (Optimization Fix!)
            # Take more than final max_boxes to allow refinement to prune correctly
            pre_nms_topk = max_boxes * 4 
            if cur_boxes.shape[0] > pre_nms_topk:
                cur_scores, topk_inds = torch.topk(cur_scores, k=pre_nms_topk)
                cur_boxes = cur_boxes[topk_inds]
                cur_cls_preds = cur_cls_preds[topk_inds]
            
            # Step 3: Semantic Refinement (Now on much fewer boxes)
            if seg_logits is not None and coords is not None:
                # Refine scores for the current batch item
                cur_seg = seg_logits[mask]
                cur_coords = coords[mask]
                cur_scores = self.semantic_refinement(cur_boxes, cur_scores, cur_seg, cur_coords, cur_cls_preds)

            # Step 4: Oriented BEV NMS (Class-Aware)
            if use_nms and cur_boxes.shape[0] > 1:
                from pcdet_lite.iou3d_nms_utils import nms_gpu
                cur_labels = cur_cls_preds.argmax(dim=-1)
                keep_idx = nms_gpu(cur_boxes, cur_scores, thresh=nms_thresh, labels=cur_labels)
                selected = keep_idx[:max_boxes] # Final top boxes
            else:
                # TopK already applied, just sort and cap
                _, selected = torch.topk(cur_scores, k=min(len(cur_scores), max_boxes))
            
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

    def semantic_refinement(self, boxes, scores, seg_logits, coords, cls_preds):
        """
        Refines detection scores based on semantic point occupancy.
        FULLY VECTORIZED: No Python loops.
        """
        if len(boxes) == 0 or len(coords) == 0:
            return scores
            
        from pcdet_lite.box_utils import points_in_boxes_cpu
        
        # 1. Assign each point to a box (N, 3), (M, 7) -> (N,)
        box_idxs = points_in_boxes_cpu(coords, boxes)
        
        # 2. Get Predicted Seg labels (N,)
        seg_labels = seg_logits.argmax(dim=-1)
        
        # 3. Get Target Class for each box (M,)
        box_target_classes = cls_preds.argmax(dim=-1)
        
        # 4. Filter for points inside any box
        valid_pts_mask = box_idxs >= 0
        if not valid_pts_mask.any():
            return scores * 0.5 # No points in boxes -> ghost box penalty
            
        in_box_idxs = box_idxs[valid_pts_mask]
        in_box_seg_labels = seg_labels[valid_pts_mask]
        
        # 5. Map box target class to the points inside it
        point_target_classes = box_target_classes[in_box_idxs]
        
        # 6. Calculate consistency
        is_consistent = (in_box_seg_labels == point_target_classes)
        
        # 7. Aggregate counts per box using vectorized bincount
        num_boxes = len(boxes)
        total_counts = torch.bincount(in_box_idxs, minlength=num_boxes).float()
        consistent_counts = torch.bincount(in_box_idxs[is_consistent], minlength=num_boxes).float()
        
        # 8. Calculate score multiplier
        # occupancy: percentage of correct semantic points in the box
        occupancy = (consistent_counts / torch.clamp(total_counts, min=1.0))
        # density: log-scaled count of grounding points
        density_bonus = torch.log1p(consistent_counts) / 5.0
        
        # Penalize boxes with NO points or NO consistent points
        multiplier = (0.5 + 0.5 * occupancy) * (1.0 + density_bonus)
        
        return scores * multiplier

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
