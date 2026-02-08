"""
LitePT Unified Model
====================
Combines LitePT backbone with OpenPCDet detection head.

This model performs simultaneous Semantic Segmentation and 3D Object Detection.
"""

import torch
import torch.nn as nn
from addict import Dict

from models.builder import MODELS
from models.litept.litept import LitePT
from models.detection import LitePTDetectionHead

@MODELS.register_module("LitePTUnified")
class LitePTUnified(LitePT):
    def __init__(
        self,
        # Detection args
        num_classes_det=3,
        det_head_cfg=None,
        # Original LitePT args (passed down)
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        # Seg Head args
        num_classes_seg=13,
        cls_mode=True, # enable classification/segmentation head
    ):
        # Initialize LitePT backbone
        super().__init__(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            enc_conv=enc_conv,
            enc_attn=enc_attn,
            enc_rope_freq=enc_rope_freq,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            dec_conv=dec_conv,
            dec_attn=dec_attn,
            dec_rope_freq=dec_rope_freq,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enc_mode=False
        )
        
        # Segmentation Head (Linear)
        if cls_mode:
            self.seg_head = nn.Linear(dec_channels[0], num_classes_seg)
        else:
            self.seg_head = None
            
        # Detection Head
        # Use final decoder output channels for detection
        det_in_channels = dec_channels[0]
        self.det_head = LitePTDetectionHead(
            in_channels=det_in_channels,
            num_classes=num_classes_det,
            model_cfg=det_head_cfg
        )

        # Loss Balancing Parameters (Kendall et al. 2018)
        # log_vars = log(sigma^2) -> initialized to 0 (sigma=1)
        # We need 2 values: [seg_uncertainty, det_uncertainty]
        self.log_vars = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self, data_dict):
        # Run backbone + decoder
        point = super().forward(data_dict)
        
        results = {}
        
        # Semantic Segmentation
        if self.seg_head is not None:
             # point.feat is (N, C)
            seg_logits = self.seg_head(point.feat)
            results['seg_logits'] = seg_logits
            
        # Object Detection
        # We need to extract GT boxes if available
        gt_boxes = None
        if isinstance(data_dict, dict) and 'gt_boxes' in data_dict:
            gt_boxes = data_dict['gt_boxes'] 
            
        det_results = self.det_head(point, gt_boxes=gt_boxes)
        results.update(det_results)

        # Return results directly, loss calculation moved to train.py
        # This allows for flexible external loss balancing (GradNorm etc.)
        
        return results

    def get_last_shared_layer(self):
        """Returns the last layer of the shared backbone (decoder) for GradNorm."""
        # Check if using decoder or encoder-only
        if hasattr(self, 'dec') and len(self.dec) > 0:
            # Last block of the decoder
            # self.dec is PointSequential containing 'dec0', 'dec1', etc.
            # Inside each 'decX' (PointSequential), there are 'blockY' (Block)
            # We want the weights of the last block
            last_stage = self.dec[-1] # This is a PointSequential or Module
            if isinstance(last_stage, nn.Sequential) or isinstance(last_stage, list): # PointSequential behaves like container
                 # Iterate to find last Block
                 for m in reversed(last_stage):
                     if hasattr(m, 'mlp'): # Block has mlp
                         return m.mlp[-1].fc2 # Last Linear of MLP
            
            # Fallback: just return the parameters of the last module in dec
            return list(self.dec.parameters())[-1]
            
        elif hasattr(self, 'enc') and len(self.enc) > 0:
             # Last block of encoder (if enc_mode)
            last_stage = self.enc[-1]
            return list(last_stage.parameters())[-1]
            
        return None
