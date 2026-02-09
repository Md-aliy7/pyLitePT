"""
Point Head Template
===================
Base class for point-based detection heads.
Adapted from OpenPCDet: pcdet/models/dense_heads/point_head_template.py

Uses pure Python box operations (no CUDA dependencies).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..box_utils import points_in_boxes_cpu, enlarge_box3d
from ..loss_utils import SigmoidFocalClassificationLoss
from ..common_utils import rotate_points_along_z


class PointHeadTemplate(nn.Module):
    """Base class for point-based detection heads."""
    
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.get('LOSS_CONFIG', {}))
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        """Build loss functions."""
        self.add_module(
            'cls_loss_func',
            SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        
        reg_loss_type = losses_cfg.get('LOSS_REG', 'smooth-l1')
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'weighted-smooth-l1':
            from ..loss_utils import WeightedSmoothL1Loss
            self.reg_loss_func = WeightedSmoothL1Loss(
                code_weights=losses_cfg.get('LOSS_WEIGHTS', {}).get('code_weights', None)
            )

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        """Create fully connected layers."""
        fc_layers = []
        c_in = input_channels
        for k in range(len(fc_cfg)):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, 
                             central_radius=2.0):
        """
        Assign targets to points based on ground-truth boxes.
        
        Pure Python implementation replacing CUDA roiaware_pool3d.
        
        Args:
            points: (N, 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8) [x, y, z, dx, dy, dz, heading, label]
            extend_gt_boxes: (B, M, 8) enlarged boxes for ignore region
            ret_box_labels: whether to return box regression labels
            ret_part_labels: whether to return part labels
            set_ignore_flag: whether to set ignore flag
            use_ball_constraint: use ball constraint for positive samples
            central_radius: radius for ball constraint

        Returns:
            targets_dict: dictionary with target labels
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, f'points.shape={points.shape}'
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, f'gt_boxes.shape={gt_boxes.shape}'
        
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        gt_box_of_points = points.new_full((points.shape[0],), -1).long()
        num_gt_so_far = 0
        
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]  # (N_k, 3)
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            
            # Get valid boxes (non-zero size)
            valid_mask = gt_boxes[k, :, 3:6].sum(dim=1) > 0
            gt_boxes_k = gt_boxes[k, valid_mask, :7]
            gt_classes_k = gt_boxes[k, valid_mask, 7].long()
            
            if gt_boxes_k.shape[0] == 0:
                point_cls_labels[bs_mask] = point_cls_labels_single
                continue
            
            # Use pure Python points_in_boxes
            box_idxs_of_pts = points_in_boxes_cpu(points_single, gt_boxes_k)
            box_fg_flag = (box_idxs_of_pts >= 0)
            
            if set_ignore_flag and extend_gt_boxes is not None:
                # Check extended boxes for ignore region
                extend_boxes_k = extend_gt_boxes[k, valid_mask, :7]
                extend_box_idxs = points_in_boxes_cpu(points_single, extend_boxes_k)
                fg_flag = box_fg_flag
                ignore_flag = (~fg_flag) & (extend_box_idxs >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                # Ball constraint for positive samples
                fg_idx = box_idxs_of_pts[box_fg_flag]
                box_centers = gt_boxes_k[fg_idx, :3].clone()
                box_centers[:, 2] += gt_boxes_k[fg_idx, 5] / 2
                ball_flag = ((box_centers - points_single[box_fg_flag]).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag.clone()
                fg_flag[box_fg_flag] = ball_flag
            else:
                fg_flag = box_fg_flag
                
            # Assign class labels
            fg_idx = box_idxs_of_pts[fg_flag]
            gt_box_of_fg_points = gt_boxes_k[fg_idx]
            gt_class_of_fg_points = gt_classes_k[fg_idx]
            
            if self.num_class == 1:
                point_cls_labels_single[fg_flag] = 1
            else:
                # Convert 0-based class indices to 1-based for detection
                # (0 = background, 1+ = foreground classes)
                point_cls_labels_single[fg_flag] = (gt_class_of_fg_points + 1).long()
            
            point_cls_labels[bs_mask] = point_cls_labels_single
            
            # Global indexing for loss normalization
            batch_box_idxs = box_idxs_of_pts.clone()
            batch_box_idxs[box_idxs_of_pts >= 0] += num_gt_so_far
            gt_box_of_points[bs_mask] = batch_box_idxs
            num_gt_so_far += gt_boxes_k.shape[0]

            # Box regression labels
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points, 
                    points=points_single[fg_flag],
                    gt_classes=gt_class_of_fg_points
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            # Part labels (for part-aware methods)
            if ret_part_labels and gt_box_of_fg_points.shape[0] > 0:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = rotate_points_along_z(
                    transformed_points.view(-1, 3), -gt_box_of_fg_points[:, 6]
                )
                offset = torch.tensor([0.5, 0.5, 0.5], device=transformed_points.device, dtype=transformed_points.dtype)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels,
            'gt_box_of_points': gt_box_of_points
        }
        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        """Calculate classification loss."""
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.get('LOSS_CONFIG', {}).get('LOSS_WEIGHTS', {})
        point_loss_cls = point_loss_cls * loss_weights_dict.get('point_cls_weight', 1.0)
        
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        """Calculate box regression loss."""
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']

        reg_weights = pos_mask.float()
        
        # Issue 3 Fix: Normalize per GT box, not per point
        gt_box_indices = self.forward_ret_dict['gt_box_of_points']
        pos_gt_indices = gt_box_indices[pos_mask]
        if len(pos_gt_indices) > 0:
            # Shift indices to be 0-based within batch if needed, 
            # but bincount works on non-negative. we just need to handle the counts.
            # We use a trick to get points per gt:
            pts_per_gt = torch.bincount(pos_gt_indices)
            reg_weights[pos_mask] /= pts_per_gt[pos_gt_indices].float()
            pos_normalizer = (pts_per_gt > 0).sum().float()
        else:
            pos_normalizer = torch.tensor(0.0, device=reg_weights.device)
            
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], 
            point_box_labels[None, ...], 
            reduction='none'
        )
        point_loss_box = (point_loss_box_src * reg_weights[None, :, None]).sum()

        loss_weights_dict = self.model_cfg.get('LOSS_CONFIG', {}).get('LOSS_WEIGHTS', {})
        point_loss_box = point_loss_box * loss_weights_dict.get('point_box_weight', 1.0)
        
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Generate predicted boxes from network outputs.
        
        Args:
            points: (N, 3) point coordinates
            point_cls_preds: (N, num_class) class predictions
            point_box_preds: (N, box_code_size) box predictions

        Returns:
            point_cls_preds: (N, num_class) class predictions
            point_box_preds: (N, 7) decoded boxes
        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        # Use 0-based class indices (no +1 conversion needed)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes)
        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
