"""
=============================================================================
LITEPT UNIFIED MODEL CORE
=============================================================================
This file contains the core model definitions for LitePT Unified.
It separates the architecture from the training logic (train.py).
"""

import torch
import torch.nn as nn
from models.litept.litept import LitePT

# Import Detection Head
try:
    from models.detection import LitePTDetectionHead
except ImportError as e:
    print(f"Warning: Could not import LitePTDetectionHead: {e}")
    LitePTDetectionHead = None

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'nano': {  # ~1M params (Formerly Micro)
        'stride': (2, 2, 2, 2),
        'enc_depths': (1, 1, 1, 1, 1),
        'enc_channels': (16, 32, 64, 112, 224),
        'enc_num_head': (2, 4, 8, 14, 28),
        'enc_patch_size': (128, 128, 128, 128, 128),
        'dec_depths': (0, 0, 0, 0),
        'dec_channels': (32, 32, 64, 112),
        'dec_num_head': (4, 4, 8, 14),
        'dec_patch_size': (128, 128, 128, 128),
    },
    'micro': {  # ~2M params (Formerly Nano)
        'stride': (2, 2, 2, 2),
        'enc_depths': (1, 1, 1, 1, 1),
        'enc_channels': (24, 48, 96, 168, 336),
        'enc_num_head': (2, 4, 8, 14, 28),
        'enc_patch_size': (128, 128, 128, 128, 128),
        'dec_depths': (0, 0, 0, 0),
        'dec_channels': (48, 48, 96, 168),
        'dec_num_head': (4, 4, 8, 14),
        'dec_patch_size': (128, 128, 128, 128),
    },
    'tiny': {  # ~6M params (Targeting ~6M)
        'stride': (2, 2, 2, 2),
        'enc_depths': (1, 1, 1, 2, 1),
        'enc_channels': (32, 64, 128, 256, 512),
        'enc_num_head': (2, 4, 8, 16, 32),
        'enc_patch_size': (256, 256, 256, 256, 256),
        'dec_depths': (0, 0, 0, 0),
        'dec_channels': (64, 64, 128, 256),
        'dec_num_head': (4, 4, 8, 16),
        'dec_patch_size': (256, 256, 256, 256),
    },
    'small': {  # ~12.7M params - (LitePT-S in Paper)
        'stride': (2, 2, 2, 2),
        'enc_depths': (2, 2, 2, 6, 2),
        'enc_channels': (36, 72, 144, 252, 504),
        'enc_num_head': (3, 6, 12, 21, 42),
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'dec_depths': (2, 2, 2, 2),
        'dec_channels': (36, 72, 144, 252),
        'dec_num_head': (3, 6, 12, 21),
        'dec_patch_size': (1024, 1024, 1024, 1024),
    },
    'base': { # ~45.1M params - (LitePT-B in Paper)
        'stride': (2, 2, 2, 2),
        'enc_depths': (3, 3, 3, 12, 3),
        'enc_channels': (54, 108, 216, 432, 576),
        'enc_num_head': (3, 6, 12, 24, 32),
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'dec_depths': (2, 2, 2, 2),
        'dec_channels': (54, 108, 216, 432),
        'dec_num_head': (4, 8, 16, 32),
        'dec_patch_size': (1024, 1024, 1024, 1024),
    },
    'large': { # ~85.9M params - (LitePT-L in Paper)
        'stride': (2, 2, 2, 2),
        'enc_depths': (3, 3, 3, 12, 3),
        'enc_channels': (72, 144, 288, 576, 864),
        'enc_num_head': (6, 12, 24, 48, 72),
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'dec_depths': (2, 2, 2, 2),
        'dec_channels': (72, 144, 288, 576),
        'dec_num_head': (6, 12, 24, 48),
        'dec_patch_size': (1024, 1024, 1024, 1024),
    },
    'single_stage': {  # Detection Variant (No Downsampling, ~2M params)
        'stride': (), # No Downsampling
        'enc_depths': (8,), # 8 Blocks
        'enc_channels': (128,),
        'enc_num_head': (8,),
        'enc_patch_size': (128,),
        'dec_depths': (), 
        'dec_channels': (), # No Decoder
        'dec_num_head': (),
        'dec_patch_size': (),
    }
}


# ============================================================================
# UNIFIED MODEL
# ============================================================================

class LitePTUnifiedCustom(nn.Module):
    """LitePT with both segmentation and detection heads."""
    
    def __init__(
        self, 
        in_channels: int, 
        num_classes_seg: int, 
        num_classes_det: int, 
        variant: str = 'micro', 
        det_config: dict = None
    ):
        """
        Initialize LitePT Unified Model.

        Args:
            in_channels (int): Number of input point features (e.g., 3 for RGB, 6 for RGB+Normal).
            num_classes_seg (int): Number of semantic segmentation classes. Set to 0 to disable.
            num_classes_det (int): Number of detection classes. Set to 0 to disable.
            variant (str): Model variant ('nano', 'micro', 'tiny', 'small', 'base', 'large').
            det_config (dict, optional): Configuration dictionary for detection head. Defaults to None.
        """
        super().__init__()
        
        model_cfg = MODEL_CONFIGS.get(variant, MODEL_CONFIGS['micro'])
        det_config = det_config or {}
        
        # Backbone
        self.backbone = LitePT(
            in_channels=in_channels,
            **model_cfg,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.1,
            enc_mode=False,
        )
        
        # Determine feature dimension
        if len(model_cfg['dec_channels']) > 0:
            feat_dim = model_cfg['dec_channels'][0]
        else:
            # Single Stage: Output is last encoder channel
            feat_dim = model_cfg['enc_channels'][-1]
        
        # Segmentation head (always enabled if num_classes_seg > 0)
        self.num_classes_seg = num_classes_seg
        if num_classes_seg > 0:
            self.seg_head = nn.Linear(feat_dim, num_classes_seg)
        else:
            self.seg_head = None
        
        # Detection head (optional)
        self.num_classes_det = num_classes_det
        if num_classes_det > 0 and LitePTDetectionHead is not None:
            # Configure detection head
            det_head_cfg = {
                'CLS_FC': [256, 256],
                'REG_FC': [256, 256],
                'TARGET_CONFIG': {
                    'BOX_CODER': 'PointResidualCoder',
                    'BOX_CODER_CONFIG': {
                        'use_mean_size': True,
                        'mean_size': det_config.get('MEAN_SIZE', [[1.0, 1.0, 1.0]])
                    },
                    'GT_EXTRA_WIDTH': det_config.get('GT_EXTRA_WIDTH', [0.2, 0.2, 0.2])
                },
                'LOSS_CONFIG': det_config.get('LOSS_CONFIG', {
                    'LOSS_REG': 'smooth-l1',
                    'LOSS_WEIGHTS': {
                        'point_cls_weight': 1.0,
                        'point_box_weight': 1.0,
                        'code_weights': [1.0] * 8
                    }
                })
            }
            self.det_head = LitePTDetectionHead(
                in_channels=feat_dim,
                num_classes=num_classes_det,
                model_cfg=det_head_cfg
            )
        else:
            self.det_head = None
            
        # Multi-Task Uncertainty Weighting (Kendall et al. 2018)
        # Learnable log variance parameters: [log_var_seg, log_var_det]
        # Initialized to 0.0 (variance=1.0)
        self.log_vars = nn.Parameter(torch.zeros(2))
    
    def forward(self, batch):
        """Forward pass through backbone and heads."""
        point = self.backbone(batch)
        outputs = {'point': point}
        
        # Segmentation
        if self.seg_head is not None:
            outputs['seg_logits'] = self.seg_head(point.feat)
        
        # Detection
        if self.det_head is not None:
            gt_boxes = batch.get('gt_boxes', None)
            # CRITICAL: Use original coordinates from batch, not the modified
            # point.coord which has been averaged during encoder-decoder pooling
            original_coord = batch['coord'].to(point.feat.device)
            original_batch = batch['batch'].to(point.feat.device) if 'batch' in batch else point.batch
            # Detection
            # CRITICAL: Pass seg_logits for Semantic-Aware NMS refinement
            point['seg_logits'] = outputs.get('seg_logits')
            
            det_out = self.det_head(point, gt_boxes=gt_boxes, 
                                    original_coord=original_coord,
                                    original_batch=original_batch)
            outputs.update(det_out)
            
        return outputs

    def get_last_shared_layer(self):
        """Returns the last layer of the shared backbone (decoder) for GradNorm."""
        # Check if using decoder or encoder-only
        if hasattr(self.backbone, 'dec') and len(self.backbone.dec) > 0:
            # Last block of the decoder
            last_stage = self.backbone.dec[-1] 
            if isinstance(last_stage, nn.Sequential) or isinstance(last_stage, list):
                 # Iterate to find last Block
                 for m in reversed(last_stage):
                     if hasattr(m, 'mlp'): # Block has mlp
                         return m.mlp[-1].fc2 # Last Linear of MLP
            
            return list(self.backbone.dec.parameters())[-1]
            
        elif hasattr(self.backbone, 'enc') and len(self.backbone.enc) > 0:
             # Last block of encoder (if enc_mode)
            last_stage = self.backbone.enc[-1]
            return list(last_stage.parameters())[-1]
            
        return None
