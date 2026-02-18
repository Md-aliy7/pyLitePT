
import os
import sys

# Add project root
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0, parent)
sys.path.insert(0, current)

from core import LitePTUnifiedCustom, MODEL_CONFIGS

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_models():
    print(f"{'='*60}")
    print(f"LITEPT PARAMETER COUNT VERIFICATION")
    print(f"{'='*60}")
    print(f"{'Variant':<15} | {'Params':>15} | {'Millions':>10}")
    print(f"{'-'*46}")
    
    variants = list(MODEL_CONFIGS.keys())
    # Sort by size logic - show main variants first, then single-stage variants
    ordered = ['nano', 'micro', 'tiny', 'small', 'base', 'large',
               'single_stage_nano', 'single_stage_micro', 'single_stage_tiny', 
               'single_stage_small', 'single_stage_base', 'single_stage_large']
    
    # Dummy config for detection head
    det_config = {
        'MEAN_SIZE': [[1.0, 1.0, 1.0]], 
        'ANCHOR_RANGES': [[0, 50]]
    }
    
    for variant in ordered:
        if variant not in MODEL_CONFIGS: continue
        
        try:
            # Instantiate model
            model = LitePTUnifiedCustom(
                in_channels=6,        # Standard coord+color
                num_classes_seg=10,   # Dummy
                num_classes_det=10,   # Dummy
                variant=variant,
                det_config=det_config
            )
            
            total_params = count_parameters(model)
            millions = total_params / 1e6
            
            print(f"{variant:<15} | {total_params:>15,} | {millions:>9.2f} M")
            
            del model
            
        except Exception as e:
            print(f"{variant:<15} | {'ERROR':>15} | {str(e)}")

    print(f"{'='*60}")

def log_model_info(model):
    """Logs detailed parameter count for a specific model instance."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*40}")
    print(f"MODEL SUMMARY")
    print(f"{'='*40}")
    print(f"Total Parameters:     {total_params:>12,}")
    print(f"Trainable Parameters: {trainable_params:>12,}")
    print(f"Segmentation Head:    {'Enabled' if getattr(model, 'seg_head', None) else 'Disabled'}")
    print(f"Detection Head:       {'Enabled' if getattr(model, 'det_head', None) else 'Disabled'}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    verify_models()
