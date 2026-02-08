"""
Point Head Box
==============
Point-based 3D object detection head.
Adapted from OpenPCDet: pcdet/models/dense_heads/point_head_box.py

Uses pure Python box operations (no CUDA dependencies).
"""

import torch
from addict import Dict

from ..box_utils import enlarge_box3d
from ..box_coder_utils import PointResidualCoder
from .point_head_template import PointHeadTemplate


class PointHeadBox(PointHeadTemplate):
    """
    Point-based detection head for 3D object detection.
    
    Used in PointRCNN and similar architectures.
    Reference: https://arxiv.org/abs/1812.04244
    """
    
    def __init__(self, num_class, input_channels, model_cfg=None, 
                 predict_boxes_when_training=False, **kwargs):
        """
        Args:
            num_class: Number of object classes
            input_channels: Number of input feature channels
            model_cfg: Configuration dict or object with:
                - CLS_FC: List of FC layer sizes for classification
                - REG_FC: List of FC layer sizes for regression
                - TARGET_CONFIG: Target assignment configuration
                - LOSS_CONFIG: Loss configuration
            predict_boxes_when_training: Whether to predict boxes during training
        """
        # Create default config if not provided
        if model_cfg is None:
            model_cfg = self._get_default_config()
        elif isinstance(model_cfg, dict):
            # Merge with defaults
            default_cfg = self._get_default_config()
            for key, value in model_cfg.items():
                default_cfg[key] = value
            model_cfg = default_cfg
            
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        
        # Classification layers
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.get('CLS_FC', [256, 256]),
            input_channels=input_channels,
            output_channels=num_class
        )

        # Box regression layers
        target_cfg = self.model_cfg.get('TARGET_CONFIG', {})
        box_coder_cfg = target_cfg.get('BOX_CODER_CONFIG', {})
        
        self.box_coder = PointResidualCoder(
            code_size=8,
            use_mean_size=box_coder_cfg.get('use_mean_size', True),
            mean_size=box_coder_cfg.get('mean_size', [
                [3.9, 1.6, 1.56],   # Car-like
                [0.8, 0.6, 1.73],   # Pedestrian-like
                [1.76, 0.6, 1.73],  # Cyclist-like
                [10.0, 10.0, 10.0], # Large object (cube)
                [8.0, 8.0, 8.0],    # Medium object (sphere)
                [6.0, 6.0, 15.0],   # Tall object (cylinder)
            ])
        )
        
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.get('REG_FC', [256, 256]),
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

    def _get_default_config(self):
        """Return default configuration."""
        return Dict({
            'CLS_FC': [256, 256],
            'REG_FC': [256, 256],
            'TARGET_CONFIG': {
                'BOX_CODER': 'PointResidualCoder',
                'BOX_CODER_CONFIG': {
                    'use_mean_size': True,
                    'mean_size': [
                        [3.9, 1.6, 1.56],
                        [0.8, 0.6, 1.73],
                        [1.76, 0.6, 1.73],
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

    def assign_targets(self, input_dict):
        """
        Assign ground-truth targets to points.
        
        Args:
            input_dict: Dictionary with:
                - point_features: (N, C) point features
                - batch_size: int
                - point_coords: (N, 4) [bs_idx, x, y, z]
                - gt_boxes: (B, M, 8) ground truth boxes
                
        Returns:
            targets_dict: Dictionary with target labels
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        
        assert gt_boxes.dim() == 3, f'gt_boxes.shape={gt_boxes.shape}'
        assert point_coords.dim() == 2, f'point_coords.shape={point_coords.shape}'

        batch_size = gt_boxes.shape[0]
        
        # Enlarge boxes for ignore region
        target_cfg = self.model_cfg.get('TARGET_CONFIG', {})
        extra_width = target_cfg.get('GT_EXTRA_WIDTH', [0.2, 0.2, 0.2])
        
        extend_gt_boxes = enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), 
            extra_width=extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1])
        
        targets_dict = self.assign_stack_targets(
            points=point_coords, 
            gt_boxes=gt_boxes, 
            extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, 
            use_ball_constraint=True,
            central_radius=2.0,
            ret_part_labels=False, 
            ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        """
        Calculate total loss (classification + box regression).
        
        Args:
            tb_dict: Optional tensorboard dict for logging
            
        Returns:
            point_loss: Total loss
            tb_dict: Updated tensorboard dict
        """
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)

        point_loss = point_loss_cls + point_loss_box
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Forward pass.
        
        Args:
            batch_dict: Dictionary with:
                - batch_size: int
                - point_features: (N, C) point features
                - point_coords: (N, 4) [bs_idx, x, y, z]
                - gt_boxes (optional): (B, M, 8) ground truth boxes
                
        Returns:
            batch_dict: Updated dictionary with predictions
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
            
        # Forward through classification layers
        point_cls_preds = self.cls_layers(point_features)  # (N, num_class)
        
        # Forward through box regression layers
        point_box_preds = self.box_layers(point_features)  # (N, box_code_size)

        # Compute classification scores
        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            'point_box_preds': point_box_preds
        }
        
        if self.training or 'gt_boxes' in batch_dict:
            # Assign targets if training or if gt_boxes are provided (for validation loss)
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['gt_box_of_points'] = targets_dict['gt_box_of_points']

        if not self.training or self.predict_boxes_when_training:
            # Generate predicted boxes
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, 
                point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
