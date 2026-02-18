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
    'nano': {  # ~1M params
        'stride': (2, 2, 2, 2),
        'enc_depths': (1, 1, 1, 1, 1),
        'enc_channels': (16, 32, 64, 108, 216),  # Adjusted for head_dim=18
        'enc_num_head': (2, 4, 8, 6, 12),
        'enc_patch_size': (128, 128, 128, 128, 128),
        'enc_conv': (True, True, True, False, False),
        'enc_attn': (False, False, False, True, True),
        'enc_rope_freq': (100.0, 100.0, 100.0, 100.0, 100.0),
        'dec_depths': (0, 0, 0, 0),
        'dec_channels': (32, 32, 64, 108),
        'dec_num_head': (4, 4, 8, 6),
        'dec_patch_size': (128, 128, 128, 128),
        'dec_conv': (False, False, False, False),
        'dec_attn': (False, False, False, False),
        'dec_rope_freq': (100.0, 100.0, 100.0, 100.0),
    },
    'micro': {  # ~2M params
        'stride': (2, 2, 2, 2),
        'enc_depths': (1, 1, 1, 1, 1),
        'enc_channels': (24, 48, 96, 168, 336),
        'enc_num_head': (2, 4, 8, 14, 28),
        'enc_patch_size': (128, 128, 128, 128, 128),
        'enc_conv': (True, True, True, False, False),
        'enc_attn': (False, False, False, True, True),
        'enc_rope_freq': (100.0, 100.0, 100.0, 100.0, 100.0),
        'dec_depths': (0, 0, 0, 0),
        'dec_channels': (48, 48, 96, 168),
        'dec_num_head': (4, 4, 8, 14),
        'dec_patch_size': (128, 128, 128, 128),
        'dec_conv': (False, False, False, False),
        'dec_attn': (False, False, False, False),
        'dec_rope_freq': (100.0, 100.0, 100.0, 100.0),
    },
    'tiny': {  # ~6M params
        'stride': (2, 2, 2, 2),
        'enc_depths': (1, 1, 1, 2, 1),
        'enc_channels': (32, 64, 128, 252, 504),  # Adjusted for head_dim=18
        'enc_num_head': (2, 4, 8, 14, 28),
        'enc_patch_size': (256, 256, 256, 256, 256),
        'enc_conv': (True, True, True, False, False),
        'enc_attn': (False, False, False, True, True),
        'enc_rope_freq': (100.0, 100.0, 100.0, 100.0, 100.0),
        'dec_depths': (0, 0, 0, 0),
        'dec_channels': (64, 64, 128, 252),
        'dec_num_head': (4, 4, 8, 14),
        'dec_patch_size': (256, 256, 256, 256),
        'dec_conv': (False, False, False, False),
        'dec_attn': (False, False, False, False),
        'dec_rope_freq': (100.0, 100.0, 100.0, 100.0),
    },
    'small': {  # ~12.7M params - (LitePT-S in Paper)
        'stride': (2, 2, 2, 2),
        'enc_depths': (2, 2, 2, 6, 2),
        'enc_channels': (36, 72, 144, 252, 504),
        'enc_num_head': (2, 4, 8, 14, 28),  # FIXED: was (3, 6, 12, 21, 42)
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'enc_conv': (True, True, True, False, False),  # ADDED
        'enc_attn': (False, False, False, True, True),  # ADDED
        'enc_rope_freq': (100.0, 100.0, 100.0, 100.0, 100.0),  # ADDED
        'dec_depths': (0, 0, 0, 0),  # FIXED: was (2, 2, 2, 2)
        'dec_channels': (72, 72, 144, 252),  # FIXED: was (36, 72, 144, 252)
        'dec_num_head': (4, 4, 8, 14),  # FIXED: was (3, 6, 12, 21)
        'dec_patch_size': (1024, 1024, 1024, 1024),
        'dec_conv': (False, False, False, False),  # ADDED
        'dec_attn': (False, False, False, False),  # ADDED
        'dec_rope_freq': (100.0, 100.0, 100.0, 100.0),  # ADDED
    },
    'base': { # ~45.1M params - (LitePT-B in Paper)
        'stride': (2, 2, 2, 2),
        'enc_depths': (3, 3, 3, 12, 3),
        'enc_channels': (54, 108, 216, 432, 576),
        'enc_num_head': (3, 6, 12, 24, 32),
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'enc_conv': (True, True, True, False, False),  # ADDED
        'enc_attn': (False, False, False, True, True),  # ADDED
        'enc_rope_freq': (100.0, 100.0, 100.0, 100.0, 100.0),  # ADDED
        'dec_depths': (0, 0, 0, 0),  # FIXED: was (2, 2, 2, 2)
        'dec_channels': (54, 108, 216, 432),
        'dec_num_head': (4, 8, 16, 32),
        'dec_patch_size': (1024, 1024, 1024, 1024),
        'dec_conv': (False, False, False, False),  # ADDED
        'dec_attn': (False, False, False, False),  # ADDED
        'dec_rope_freq': (100.0, 100.0, 100.0, 100.0),  # ADDED
    },
    'large': { # ~85.9M params - (LitePT-L in Paper)
        'stride': (2, 2, 2, 2),
        'enc_depths': (3, 3, 3, 12, 3),
        'enc_channels': (72, 144, 288, 576, 864),
        'enc_num_head': (4, 8, 16, 32, 48),  # FIXED: was (6, 12, 24, 48, 72)
        'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
        'enc_conv': (True, True, True, False, False),  # ADDED
        'enc_attn': (False, False, False, True, True),  # ADDED
        'enc_rope_freq': (100.0, 100.0, 100.0, 100.0, 100.0),  # ADDED
        'dec_depths': (0, 0, 0, 0),  # FIXED: was (2, 2, 2, 2)
        'dec_channels': (72, 144, 288, 576),
        'dec_num_head': (6, 12, 24, 48),
        'dec_patch_size': (1024, 1024, 1024, 1024),
        'dec_conv': (False, False, False, False),  # ADDED
        'dec_attn': (False, False, False, False),  # ADDED
        'dec_rope_freq': (100.0, 100.0, 100.0, 100.0),  # ADDED
    },
    # ========================================================================
    # SINGLE-STAGE VARIANTS (Detection-Optimized: No Downsampling)
    # ========================================================================
    'single_stage_nano': {  # Single-stage nano (~0.5M params)
        'stride': (),
        'enc_depths': (6,),
        'enc_channels': (72,),  # Adjusted for head_dim=18
        'enc_num_head': (4,),
        'enc_patch_size': (128,),
        'enc_conv': (False,),
        'enc_attn': (True,),
        'enc_rope_freq': (100.0,),
        'dec_depths': (), 
        'dec_channels': (),
        'dec_num_head': (),
        'dec_patch_size': (),
        'dec_conv': (),
        'dec_attn': (),
        'dec_rope_freq': (),
    },
    'single_stage_micro': {  # Single-stage micro (~1M params)
        'stride': (),
        'enc_depths': (8,),
        'enc_channels': (108,),  # Adjusted for head_dim=18
        'enc_num_head': (6,),
        'enc_patch_size': (128,),
        'enc_conv': (False,),
        'enc_attn': (True,),
        'enc_rope_freq': (100.0,),
        'dec_depths': (), 
        'dec_channels': (),
        'dec_num_head': (),
        'dec_patch_size': (),
        'dec_conv': (),
        'dec_attn': (),
        'dec_rope_freq': (),
    },
    'single_stage_tiny': {  # Single-stage tiny (~2M params)
        'stride': (),
        'enc_depths': (8,),
        'enc_channels': (144,),  # Adjusted for head_dim=18
        'enc_num_head': (8,),
        'enc_patch_size': (128,),
        'enc_conv': (False,),
        'enc_attn': (True,),
        'enc_rope_freq': (100.0,),
        'dec_depths': (), 
        'dec_channels': (),
        'dec_num_head': (),
        'dec_patch_size': (),
        'dec_conv': (),
        'dec_attn': (),
        'dec_rope_freq': (),
    },
    'single_stage_small': {  # Single-stage small (~5M params)
        'stride': (),
        'enc_depths': (8,),
        'enc_channels': (216,),  # Adjusted for head_dim=18
        'enc_num_head': (12,),
        'enc_patch_size': (256,),
        'enc_conv': (False,),
        'enc_attn': (True,),
        'enc_rope_freq': (100.0,),
        'dec_depths': (), 
        'dec_channels': (),
        'dec_num_head': (),
        'dec_patch_size': (),
        'dec_conv': (),
        'dec_attn': (),
        'dec_rope_freq': (),
    },
    'single_stage_base': {  # Single-stage base (~15M params)
        'stride': (),
        'enc_depths': (12,),
        'enc_channels': (288,),  # Adjusted for head_dim=18
        'enc_num_head': (16,),
        'enc_patch_size': (512,),
        'enc_conv': (False,),
        'enc_attn': (True,),
        'enc_rope_freq': (100.0,),
        'dec_depths': (), 
        'dec_channels': (),
        'dec_num_head': (),
        'dec_patch_size': (),
        'dec_conv': (),
        'dec_attn': (),
        'dec_rope_freq': (),
    },
    'single_stage_large': {  # Single-stage large (~30M params)
        'stride': (),
        'enc_depths': (12,),
        'enc_channels': (432,),  # Adjusted for head_dim=18
        'enc_num_head': (24,),
        'enc_patch_size': (1024,),
        'enc_conv': (False,),
        'enc_attn': (True,),
        'enc_rope_freq': (100.0,),
        'dec_depths': (), 
        'dec_channels': (),
        'dec_num_head': (),
        'dec_patch_size': (),
        'dec_conv': (),
        'dec_attn': (),
        'dec_rope_freq': (),
    },
    # Legacy single_stage (alias for single_stage_tiny for backward compatibility)
    'single_stage': {  # Detection Variant (No Downsampling, ~2M params) - DEPRECATED: Use single_stage_tiny
        'stride': (),
        'enc_depths': (8,),
        'enc_channels': (128,),
        'enc_num_head': (8,),
        'enc_patch_size': (128,),
        'enc_conv': (False,),
        'enc_attn': (True,),
        'enc_rope_freq': (100.0,),
        'dec_depths': (), 
        'dec_channels': (),
        'dec_num_head': (),
        'dec_patch_size': (),
        'dec_conv': (),
        'dec_attn': (),
        'dec_rope_freq': (),
    }
}


# ============================================================================
# UNIFIED MODEL
# ============================================================================

class LitePTUnifiedCustom(nn.Module):
    """LitePT with both segmentation and detection heads.
    
    Automatically selects appropriate architecture based on task:
    - Segmentation: Uses multi-stage encoder-decoder
    - Detection: Can use multi-stage (shared) or single-stage (optimal)
    - Unified: Uses multi-stage for both (single backbone)
    """
    
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
            variant (str): Model variant name. Can be:
                          - Base names: 'nano', 'micro', 'tiny', 'small', 'base', 'large'
                          - Single-stage: 'single_stage_nano', 'single_stage_micro', etc.
                          The architecture (multi-stage vs single-stage) is automatically
                          selected based on the variant name.
            det_config (dict, optional): Configuration dictionary for detection head. Defaults to None.
        """
        super().__init__()
        
        model_cfg = MODEL_CONFIGS.get(variant, MODEL_CONFIGS['micro'])
        det_config = det_config or {}
        
        # Backbone
        self.backbone = LitePT(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),  # ADDED: serialization orders
            **model_cfg,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.3,  # FIXED: was 0.1, should be 0.3 for proper regularization
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
            det_out = self.det_head(point, gt_boxes=gt_boxes, 
                                    original_coord=original_coord,
                                    original_batch=original_batch)
            outputs.update(det_out)
            
        return outputs

    def get_last_shared_layer(self):
        """Returns the last shared nn.Module of the backbone for GradNorm.
        
        GradNorm needs a layer (nn.Module) so it can call .parameters()
        to compute per-task gradient norms: ||∇_W(w_i * L_i)||
        
        Returns:
            nn.Module or None: The last decoder/encoder stage module.
        """
        if hasattr(self.backbone, 'dec') and len(self.backbone.dec) > 0:
            return self.backbone.dec[-1]  # Return the module itself
        elif hasattr(self.backbone, 'enc') and len(self.backbone.enc) > 0:
            return self.backbone.enc[-1]  # Return the module itself
        return None


# ============================================================================
# DUAL-PATH UNIFIED MODEL (Optimal for Multi-Task Learning)
# ============================================================================

class LitePTDualPathUnified(nn.Module):
    """
    Dual-Path LitePT for optimal multi-task learning.
    
    Architecture:
        - Segmentation Branch: Multi-stage encoder-decoder with downsampling
        - Detection Branch: Single-stage encoder without downsampling
        - Each task gets its optimal architecture for best performance
    
    This architecture follows the principle that:
        - Segmentation benefits from hierarchical features (downsampling)
        - Detection benefits from high-resolution features (no downsampling)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes_seg: int,
        num_classes_det: int,
        variant: str = 'small',
        det_config: dict = None
    ):
        """
        Initialize Dual-Path LitePT Unified Model.
        
        Args:
            in_channels: Number of input point features
            num_classes_seg: Number of segmentation classes
            num_classes_det: Number of detection classes
            variant: Base variant name (nano, micro, tiny, small, base, large)
                    Automatically uses:
                    - 'variant' for segmentation (multi-stage)
                    - 'single_stage_variant' for detection (single-stage)
            det_config: Detection head configuration
        """
        super().__init__()
        
        # Get configurations for both branches
        seg_cfg = MODEL_CONFIGS.get(variant, MODEL_CONFIGS['small'])
        det_variant = f'single_stage_{variant}'
        det_cfg = MODEL_CONFIGS.get(det_variant, MODEL_CONFIGS['single_stage_tiny'])
        
        det_config = det_config or {}
        
        # Segmentation Branch (Multi-stage with downsampling)
        self.seg_backbone = LitePT(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            **seg_cfg,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.3,
            enc_mode=False,
        )
        
        # Detection Branch (Single-stage without downsampling)
        self.det_backbone = LitePT(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            **det_cfg,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.3,
            enc_mode=False,
        )
        
        # Feature dimensions for each branch
        seg_feat_dim = seg_cfg['dec_channels'][0] if len(seg_cfg['dec_channels']) > 0 else seg_cfg['enc_channels'][-1]
        det_feat_dim = det_cfg['enc_channels'][-1]  # Single-stage always uses encoder output
        
        # Segmentation Head
        self.num_classes_seg = num_classes_seg
        if num_classes_seg > 0:
            self.seg_head = nn.Linear(seg_feat_dim, num_classes_seg)
        else:
            self.seg_head = None
        
        # Detection Head
        self.num_classes_det = num_classes_det
        if num_classes_det > 0 and LitePTDetectionHead is not None:
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
                in_channels=det_feat_dim,
                num_classes=num_classes_det,
                model_cfg=det_head_cfg
            )
        else:
            self.det_head = None
        
        # Multi-Task Uncertainty Weighting
        self.log_vars = nn.Parameter(torch.zeros(2))
        
        # Store variant info
        self.variant = variant
        self.det_variant = det_variant
    
    def forward(self, batch):
        """
        Dual-path forward pass.
        
        Both branches process the input independently:
            - Segmentation branch: Gets hierarchical features
            - Detection branch: Gets high-resolution features
        """
        outputs = {}
        
        # Segmentation Branch Forward
        if self.seg_head is not None:
            seg_point = self.seg_backbone(batch)
            outputs['seg_logits'] = self.seg_head(seg_point.feat)
            outputs['seg_point'] = seg_point
        
        # Detection Branch Forward
        if self.det_head is not None:
            det_point = self.det_backbone(batch)
            gt_boxes = batch.get('gt_boxes', None)
            original_coord = batch['coord'].to(det_point.feat.device)
            original_batch = batch['batch'].to(det_point.feat.device) if 'batch' in batch else det_point.batch
            
            det_out = self.det_head(det_point, gt_boxes=gt_boxes,
                                   original_coord=original_coord,
                                   original_batch=original_batch)
            outputs.update(det_out)
            outputs['det_point'] = det_point
        
        # For compatibility, set 'point' to segmentation point if available
        if 'seg_point' in outputs:
            outputs['point'] = outputs['seg_point']
        elif 'det_point' in outputs:
            outputs['point'] = outputs['det_point']
        
        return outputs
    
    def get_last_shared_layer(self):
        """Returns the last layer of segmentation backbone for GradNorm.
        
        Note: In dual-path, there's no truly "shared" layer, but we use
        the segmentation backbone's last layer as the reference point.
        
        Returns:
            nn.Module or None: The last decoder/encoder stage module.
        """
        if hasattr(self.seg_backbone, 'dec') and len(self.seg_backbone.dec) > 0:
            return self.seg_backbone.dec[-1]  # Return the module itself
        elif hasattr(self.seg_backbone, 'enc') and len(self.seg_backbone.enc) > 0:
            return self.seg_backbone.enc[-1]  # Return the module itself
        return None


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_unified_model(
    cfg,
    input_channels: int,
    det_config: dict,
    device: str = 'cuda'
):
    """
    Create unified model with correct architecture based on configuration.
    
    This function ensures consistent model instantiation across training,
    evaluation, and visualization by using the same logic to determine
    which model class and variant to use.
    
    Args:
        cfg: Configuration object with MODEL_VARIANT, NUM_CLASSES_SEG, 
             NUM_CLASSES_DET, USE_DUAL_PATH_UNIFIED
        input_channels: Number of input feature channels
        det_config: Detection configuration dictionary
        device: Device to place model on
        
    Returns:
        Initialized model (LitePTUnifiedCustom or LitePTDualPathUnified)
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate configuration
    if cfg.NUM_CLASSES_SEG < 0 or cfg.NUM_CLASSES_DET < 0:
        raise ValueError(f"Invalid class counts: seg={cfg.NUM_CLASSES_SEG}, det={cfg.NUM_CLASSES_DET}")
    
    if cfg.NUM_CLASSES_SEG == 0 and cfg.NUM_CLASSES_DET == 0:
        raise ValueError("At least one of NUM_CLASSES_SEG or NUM_CLASSES_DET must be > 0")
    
    # Determine architecture
    use_dual_path = (
        cfg.NUM_CLASSES_SEG > 0 and 
        cfg.NUM_CLASSES_DET > 0 and 
        getattr(cfg, 'USE_DUAL_PATH_UNIFIED', False)
    )
    
    # Log architecture selection
    if cfg.NUM_CLASSES_SEG == 0 and cfg.NUM_CLASSES_DET > 0:
        # Detection only - use single-stage variant
        variant_to_use = f'single_stage_{cfg.MODEL_VARIANT}' if not cfg.MODEL_VARIANT.startswith('single_stage_') else cfg.MODEL_VARIANT
        print(f"[Model Factory] Detection Mode: Using single-stage architecture (no downsampling)")
        print(f"[Model Factory] Creating Model: LitePT-{variant_to_use}")
        print(f"[Model Factory] Input Channels: {input_channels}")
        print(f"[Model Factory] Detection Classes: {cfg.NUM_CLASSES_DET}")
        
        model = LitePTUnifiedCustom(
            in_channels=input_channels,
            num_classes_seg=cfg.NUM_CLASSES_SEG,
            num_classes_det=cfg.NUM_CLASSES_DET,
            variant=variant_to_use,
            det_config=det_config
        )
    elif use_dual_path:
        # Unified dual-path mode
        print(f"[Model Factory] Unified Mode (Dual-Path): Optimal architecture for both tasks")
        print(f"[Model Factory]   - Segmentation branch: LitePT-{cfg.MODEL_VARIANT} (multi-stage with downsampling)")
        print(f"[Model Factory]   - Detection branch: LitePT-single_stage_{cfg.MODEL_VARIANT} (single-stage, no downsampling)")
        print(f"[Model Factory] Input Channels: {input_channels}")
        print(f"[Model Factory] Segmentation Classes: {cfg.NUM_CLASSES_SEG}")
        print(f"[Model Factory] Detection Classes: {cfg.NUM_CLASSES_DET}")
        
        model = LitePTDualPathUnified(
            in_channels=input_channels,
            num_classes_seg=cfg.NUM_CLASSES_SEG,
            num_classes_det=cfg.NUM_CLASSES_DET,
            variant=cfg.MODEL_VARIANT,
            det_config=det_config
        )
    else:
        # Segmentation only or unified single-path mode
        if cfg.NUM_CLASSES_SEG > 0 and cfg.NUM_CLASSES_DET > 0:
            print(f"[Model Factory] Unified Mode (Single-Path): Shared backbone for both tasks")
            print(f"[Model Factory] Creating Model: LitePT-{cfg.MODEL_VARIANT} (multi-stage with downsampling)")
        else:
            print(f"[Model Factory] Segmentation Mode: Multi-stage architecture")
            print(f"[Model Factory] Creating Model: LitePT-{cfg.MODEL_VARIANT}")
        
        print(f"[Model Factory] Input Channels: {input_channels}")
        print(f"[Model Factory] Segmentation Classes: {cfg.NUM_CLASSES_SEG}")
        if cfg.NUM_CLASSES_DET > 0:
            print(f"[Model Factory] Detection Classes: {cfg.NUM_CLASSES_DET}")
        
        model = LitePTUnifiedCustom(
            in_channels=input_channels,
            num_classes_seg=cfg.NUM_CLASSES_SEG,
            num_classes_det=cfg.NUM_CLASSES_DET,
            variant=cfg.MODEL_VARIANT,
            det_config=det_config
        )
    
    # Log detection head configuration
    if model.det_head is not None:
        # Infer detection head input channels
        if hasattr(model, 'det_backbone'):
            # Dual-path: detection head gets features from detection backbone
            det_cfg = MODEL_CONFIGS.get(f'single_stage_{cfg.MODEL_VARIANT}', MODEL_CONFIGS['single_stage_tiny'])
            det_feat_dim = det_cfg['enc_channels'][-1]
        else:
            # Single-path: detection head gets features from shared decoder
            model_cfg = MODEL_CONFIGS.get(cfg.MODEL_VARIANT, MODEL_CONFIGS['micro'])
            if len(model_cfg['dec_channels']) > 0:
                det_feat_dim = model_cfg['dec_channels'][0]
            else:
                det_feat_dim = model_cfg['enc_channels'][-1]
        
        print(f"[Model Factory] Detection Head Input Channels: {det_feat_dim}")
    
    return model.to(device)
