"""
=============================================================================
AUTO OPTIMIZE - Automatic Hyperparameter Optimization for LitePT
=============================================================================
Analyzes entire dataset and generates optimized configuration.

Usage:
    python Custom/auto_optimize.py                        # Use default DATA_PATH
    python Custom/auto_optimize.py --data_path /path      # Custom path
    python Custom/auto_optimize.py --apply                # Auto-apply to config.py
=============================================================================
"""

import os
import sys
import argparse
import numpy as np

# Add project root
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0, parent)


def read_ply_simple(path):
    """Read a PLY file and return structured data."""
    ply_dtypes = {
        b'int8': 'i1', b'char': 'i1',
        b'uint8': 'u1', b'uchar': 'u1',
        b'int32': 'i4', b'int': 'i4',
        b'float32': 'f4', b'float': 'f4',
    }
    
    with open(path, 'rb') as f:
        if b'ply' not in f.readline():
            raise ValueError('Not a PLY file')
        
        fmt_line = f.readline().split()
        fmt = fmt_line[1].decode()
        ext = '<' if 'little' in fmt else '>'
        
        properties = []
        num_points = 0
        
        line = b''
        while b'end_header' not in line:
            line = f.readline()
            if b'element vertex' in line:
                num_points = int(line.split()[2])
            elif b'property' in line and b'list' not in line:
                parts = line.split()
                if len(parts) >= 3 and parts[1] in ply_dtypes:
                    properties.append((parts[2].decode(), ext + ply_dtypes[parts[1]]))
        
        data = np.fromfile(f, dtype=properties, count=num_points)
    
    return data


def analyze_dataset(data_path):
    """Analyze all files in dataset directory."""
    stats = {
        'num_files': 0,
        'total_points': 0,
        'seg_class_counts': {},
        'det_class_counts': {},
        'min_bounds': None,
        'max_bounds': None,
        'has_color': False,
        'has_detection': False,
        'format': None,  # 'ply' or 'npy'
    }
    
    # Find all data files
    files = []
    for split in ['train', 'val', 'test', 'training', 'validation']:
        split_dir = os.path.join(data_path, split)
        if os.path.isdir(split_dir):
            for item in os.listdir(split_dir):
                item_path = os.path.join(split_dir, item)
                
                # NPY folder
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'coord.npy')):
                    files.append((item_path, 'npy'))
                # PLY file
                elif item.endswith('.ply') and 'input_grid' not in item:
                    files.append((item_path, 'ply'))
    
    if not files:
        print(f"❌ No data files found in {data_path}")
        return None
    
    stats['format'] = files[0][1]
    print(f"📂 Analyzing {len(files)} files ({stats['format'].upper()} format)...")
    
    for path, fmt in files:
        try:
            if fmt == 'npy':
                pts = np.load(os.path.join(path, 'coord.npy'))
                seg_path = os.path.join(path, 'segment.npy')
                seg_labels = np.load(seg_path) if os.path.exists(seg_path) else None
                
                color_path = os.path.join(path, 'color.npy')
                if os.path.exists(color_path):
                    stats['has_color'] = True
                
                gt_boxes_path = os.path.join(path, 'gt_boxes.npy')
                if os.path.exists(gt_boxes_path):
                    gt_boxes = np.load(gt_boxes_path)
                    stats['has_detection'] = True
                    if len(gt_boxes) > 0 and gt_boxes.shape[1] > 7:
                        dims = gt_boxes[:, 3:6]
                        det_labels = gt_boxes[:, 7].astype(int)
                        
                        for i, lbl in enumerate(det_labels):
                            stats['det_class_counts'][lbl] = stats['det_class_counts'].get(lbl, 0) + 1
                            if lbl not in stats.get('det_box_sums', {}):
                                stats.setdefault('det_box_sums', {})[lbl] = np.zeros(3)
                            stats['det_box_sums'][lbl] += dims[i]
                
                # Use seg_labels for segmentation class counting
                labels = seg_labels
            else:
                data = read_ply_simple(path)
                pts = np.column_stack([data['x'], data['y'], data['z']])
                
                seg_labels = None
                if 'class' in data.dtype.names:
                    seg_labels = data['class']
                elif 'label' in data.dtype.names:
                    seg_labels = data['label']
                
                if 'red' in data.dtype.names:
                    stats['has_color'] = True
                
                # Check sidecar gt_boxes
                gt_boxes_path = path.replace('.ply', '_gt_boxes.npy')
                if os.path.exists(gt_boxes_path):
                    gt_boxes = np.load(gt_boxes_path)
                    stats['has_detection'] = True
                    if len(gt_boxes) > 0 and gt_boxes.shape[1] > 7:
                        dims = gt_boxes[:, 3:6]
                        det_labels = gt_boxes[:, 7].astype(int)
                        
                        for i, lbl in enumerate(det_labels):
                            stats['det_class_counts'][lbl] = stats['det_class_counts'].get(lbl, 0) + 1
                            if lbl not in stats.get('det_box_sums', {}):
                                stats.setdefault('det_box_sums', {})[lbl] = np.zeros(3)
                            stats['det_box_sums'][lbl] += dims[i]
                
                # Use seg_labels for segmentation class counting
                labels = seg_labels
            
            # Update stats
            stats['num_files'] += 1
            stats['total_points'] += len(pts)
            
            min_pt = pts.min(axis=0)
            max_pt = pts.max(axis=0)
            if stats['min_bounds'] is None:
                stats['min_bounds'] = min_pt.copy()
                stats['max_bounds'] = max_pt.copy()
            else:
                np.minimum(stats['min_bounds'], min_pt, out=stats['min_bounds'])
                np.maximum(stats['max_bounds'], max_pt, out=stats['max_bounds'])
            
            if labels is not None:
                unique, counts = np.unique(labels, return_counts=True)
                for u, c in zip(unique.astype(int), counts.astype(int)):
                    stats['seg_class_counts'][u] = stats['seg_class_counts'].get(u, 0) + c
                    
        except Exception as e:
            print(f"  ⚠️  Error reading {os.path.basename(path)}: {e}")
            import traceback
            traceback.print_exc()
    
    if stats['num_files'] == 0:
        return None
    
    stats['extent'] = stats['max_bounds'] - stats['min_bounds']
    stats['max_extent'] = np.max(stats['extent'])
    stats['avg_points'] = stats['total_points'] / stats['num_files']
    
    return stats


def calculate_optimal_params(stats, data_path=None):
    """Calculate optimal hyperparameters."""
    params = {}
    
    # Check for classes.json (Ground Truth)
    classes_json_path = os.path.join(data_path, 'classes.json') if data_path else None
    labelcloud_classes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'labelCloud', 'labels', '_classes.json')
    
    gt_classes = None
    import json
    if classes_json_path and os.path.exists(classes_json_path):
        with open(classes_json_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and "classes" in data:
                gt_classes = [c["name"] for c in sorted(data["classes"], key=lambda x: x.get("id", 0))]
            else:
                gt_classes = data

    if not gt_classes and os.path.exists(labelcloud_classes_path):
        with open(labelcloud_classes_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and "classes" in data:
                gt_classes = [c["name"] for c in sorted(data["classes"], key=lambda x: x.get("id", 0))]
    
    # Segmentation classes - from segment.npy files
    # CRITICAL FIX: LabelCloud and annotation tools often use 0-based indexing
    # where class 0 is background/default and may not be saved in data
    # We need to ensure class 0 is included in the count
    if gt_classes:
        params['NUM_CLASSES_SEG'] = len(gt_classes)
        params['CLASS_NAMES'] = gt_classes
    else:
        # Auto-detect from segmentation labels
        if stats['seg_class_counts']:
            # Get the maximum class ID found in data
            max_class_id = max(stats['seg_class_counts'].keys())
            # NUM_CLASSES should be max_id + 1 to include class 0
            # Example: if we have classes [1, 2, 3], we need 4 classes (0, 1, 2, 3)
            num_seg_classes = max_class_id + 1
            params['NUM_CLASSES_SEG'] = num_seg_classes
            
            # Generate class names for ALL classes including 0
            class_names = []
            for i in range(num_seg_classes):
                if i == 0:
                    class_names.append('background')  # Class 0 is typically background
                else:
                    class_names.append(f'class_{i}')
            params['CLASS_NAMES'] = class_names
        else:
            # Fallback: use detection classes if no segmentation labels found
            if stats['det_class_counts']:
                max_class_id = max(stats['det_class_counts'].keys())
                num_seg_classes = max_class_id + 1
            else:
                num_seg_classes = 1
            params['NUM_CLASSES_SEG'] = num_seg_classes
            params['CLASS_NAMES'] = ['background'] + [f'class_{i}' for i in range(1, num_seg_classes)]
    
    # Class weights for segmentation
    # CRITICAL FIX: Include class 0 (background) in weight calculation
    if params['NUM_CLASSES_SEG'] > 0 and stats['seg_class_counts']:
        total = sum(stats['seg_class_counts'].values())
        
        # Calculate weights for ALL classes (0 to NUM_CLASSES_SEG-1)
        weights = []
        for cls_id in range(params['NUM_CLASSES_SEG']):
            count = stats['seg_class_counts'].get(cls_id, 0)
            
            if count == 0:
                # Class not found in data (e.g., class 0 background)
                # Assign a neutral weight (1.0) since we don't have data
                weights.append(1.0)
            else:
                # Standard inverse frequency weighting
                weight = total / (params['NUM_CLASSES_SEG'] * count)
                weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        if weights.min() > 0:
            weights = weights / weights.min()
        params['CLASS_WEIGHTS'] = [round(float(w), 2) for w in weights]
    else:
        params['CLASS_WEIGHTS'] = 'auto'
    
    # Detection classes - use max class ID + 1 to include class 0
    # CRITICAL FIX: Detection boxes use 0-based indexing
    # If we have boxes with class IDs [1, 2, 3], we need 4 classes (0, 1, 2, 3)
    if stats['det_class_counts']:
        max_det_class = max(stats['det_class_counts'].keys())
        params['NUM_CLASSES_DET'] = max_det_class + 1
    else:
        params['NUM_CLASSES_DET'] = 0
    
    # Mean Sizes - based on actual detection boxes found
    # CRITICAL FIX: Create mean sizes for ALL classes (0 to NUM_CLASSES_DET-1)
    mean_sizes = []
    if params['NUM_CLASSES_DET'] > 0:
        for cls_id in range(params['NUM_CLASSES_DET']):
            if cls_id in stats.get('det_box_sums', {}) and stats['det_class_counts'].get(cls_id, 0) > 0:
                # Class has boxes - use actual mean size
                s = stats['det_box_sums'][cls_id]
                c = stats['det_class_counts'][cls_id]
                mean_sizes.append((s / c).tolist())
            else:
                # Class has no boxes (e.g., class 0 background) - use default
                mean_sizes.append([1.0, 1.0, 1.0])
    
    params['MEAN_SIZE'] = mean_sizes
    
    # Grid size
    max_extent = stats['max_extent']
    grid_size = np.clip(max_extent / 200, 0.005, 1.0)
    params['GRID_SIZE'] = round(grid_size, 4)
    
    # Input channels
    params['INPUT_CHANNELS'] = 6 if stats['has_color'] else 3
    
    # Batch size based on average points
    avg_pts = stats['avg_points']
    if avg_pts < 10000:
        params['BATCH_SIZE'] = 8
    elif avg_pts < 50000:
        params['BATCH_SIZE'] = 4
    elif avg_pts < 100000:
        params['BATCH_SIZE'] = 2
    else:
        params['BATCH_SIZE'] = 1
        
    # Data format
    params['DATA_FORMAT'] = stats['format']
    
    return params


def print_analysis(stats, params):
    """Print analysis results."""
    print(f"\n{'='*70}")
    print("📊 DATASET ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n📁 Files: {stats['num_files']}")
    print(f"📍 Total Points: {stats['total_points']:,}")
    print(f"📍 Avg Points/File: {stats['avg_points']:,.0f}")
    print(f"📐 Max Extent: {stats['max_extent']:.2f}")
    print(f"🎨 Has Color: {stats['has_color']}")
    print(f"📦 Has Detection: {stats['has_detection']}")
    print(f"📂 Format: {stats['format'].upper()}")
    
    if stats['seg_class_counts']:
        total = sum(stats['seg_class_counts'].values())
        print(f"\n🏷️  Segmentation Classes:")
        print(f"    Detected in data: {sorted(stats['seg_class_counts'].keys())}")
        print(f"    Total classes (including class 0): {params['NUM_CLASSES_SEG']}")
        print(f"    Class distribution:")
        for cls_id in range(params['NUM_CLASSES_SEG']):
            count = stats['seg_class_counts'].get(cls_id, 0)
            if count > 0:
                pct = 100 * count / total
                print(f"      Class {cls_id}: {count:>10,} ({pct:>5.2f}%)")
            else:
                print(f"      Class {cls_id}: {'NOT IN DATA':>10} (background/unlabeled)")
    
    if stats['det_class_counts']:
        total = sum(stats['det_class_counts'].values())
        print(f"\n📦 Detection Classes:")
        print(f"    Detected in data: {sorted(stats['det_class_counts'].keys())}")
        print(f"    Total classes (including class 0): {params['NUM_CLASSES_DET']}")
        print(f"    Box distribution:")
        for cls_id in range(params['NUM_CLASSES_DET']):
            count = stats['det_class_counts'].get(cls_id, 0)
            if count > 0:
                print(f"      Class {cls_id}: {count:>10,} boxes")
                if 'det_box_sums' in stats and cls_id in stats['det_box_sums']:
                    s = stats['det_box_sums'][cls_id]
                    c = count
                    m = s / c
                    print(f"        Mean Size: [{m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f}]")
            else:
                print(f"      Class {cls_id}: {'NOT IN DATA':>10} (background/no boxes)")
    
    print(f"\n{'='*70}")
    print("⚙️  RECOMMENDED CONFIGURATION")
    print(f"{'='*70}\n")
    
    print(f"NUM_CLASSES_SEG = {params['NUM_CLASSES_SEG']}")
    print(f"NUM_CLASSES_DET = {params['NUM_CLASSES_DET']}")
    print(f"CLASS_NAMES = {params['CLASS_NAMES']}")
    print(f"CLASS_WEIGHTS = {params['CLASS_WEIGHTS']}")
    print(f"")
    print(f"INPUT_CHANNELS = {params['INPUT_CHANNELS']}")
    print(f"GRID_SIZE = {params['GRID_SIZE']}")
    print(f"BATCH_SIZE = {params['BATCH_SIZE']}")
    # print(f"EPOCHS = {params['EPOCHS']}")
    if params['MEAN_SIZE']:
        print(f"MEAN_SIZE (Detection) = {params['MEAN_SIZE']}")
    
    print(f"\n{'='*70}")


def apply_to_config(params, config_path):
    """Apply parameters to config.py."""
    import re
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    replacements = {
        'NUM_CLASSES_SEG': str(params['NUM_CLASSES_SEG']),
        'NUM_CLASSES_DET': str(params['NUM_CLASSES_DET']),
        'CLASS_NAMES': str(params['CLASS_NAMES']),
        'CLASS_WEIGHTS': repr(params['CLASS_WEIGHTS']),
        'INPUT_CHANNELS': str(params['INPUT_CHANNELS']),
        'GRID_SIZE': str(params['GRID_SIZE']),
        'BATCH_SIZE': str(params['BATCH_SIZE']),
        # 'EPOCHS': str(params['EPOCHS']),
    }
    
    for key, value in replacements.items():
        pattern = rf'^{key}\s*=.*$'
        replacement = f'{key} = {value}'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Special handling for MEAN_SIZE inside DETECTION_CONFIG
    if params['MEAN_SIZE']:
        lines = content.splitlines()
        new_lines = []
        skip = False
        inserted = False
        
        for line in lines:
            if "'MEAN_SIZE': [" in line:
                skip = True
                new_lines.append(f"    'MEAN_SIZE': [")
                for ms in params['MEAN_SIZE']:
                    new_lines.append(f"        {ms},")
                new_lines.append(f"    ],") # Corrected closing: just end the list
                inserted = True
                continue
            
            if skip:
                # Strictly match duplication end
                if line.strip() in ['],', ']']: 
                    skip = False
                    # Don't append this line, because we already appended '    ],' above
                continue
            
            new_lines.append(line)
        
        if inserted:
            content = "\n".join(new_lines)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Applied parameters to {config_path}")
    print("⚠️  Please review CLASS_NAMES and set meaningful names!")


def main():
    parser = argparse.ArgumentParser(description='Auto-optimize LitePT hyperparameters')
    parser.add_argument('--data_path', type=str, default=None, help='Path to data directory')
    parser.add_argument('--apply', action='store_true', help='Apply to config.py')
    args = parser.parse_args()
    
    # Import config
    import config as cfg
    data_path = args.data_path or cfg.DATA_PATH
    
    print(f"🔍 Analyzing: {data_path}")
    
    stats = analyze_dataset(data_path)
    if stats is None:
        return
    
    params = calculate_optimal_params(stats, data_path)
    print_analysis(stats, params)
    
    if args.apply:
        # 1. Update config.py text (Legacy/Permanent)
        config_path = os.path.join(current, 'config.py')
        apply_to_config(params, config_path)
        
        # 2. Save to JSON (Dynamic)
        import json
        json_path = os.path.join(current, 'dataset_status.json')
        
        # Helper to convert internal types to JSON serializable
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                clean_params[k] = v.tolist()
            elif isinstance(v, np.generic):
                clean_params[k] = v.item()
            else:
                clean_params[k] = v
                
        with open(json_path, 'w') as f:
            json.dump(clean_params, f, indent=4)
        print(f"✅ Saved optimized parameters to {json_path}")


def ensure_optimization(data_path, force=False):
    """
    Checks for optimization cache. If missing or forced, runs optimization.
    Returns: True if optimization ran (and config needs reload), False otherwise.
    """
    json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_status.json')
    
    if os.path.exists(json_path) and not force:
        return False
        
    print("⚠️  No optimized config found (or forced). Running Auto-Optimization...")
    
    stats = analyze_dataset(data_path)
    if stats is None:
        print("❌ Optimization failed: could not analyze dataset.")
        return False
        
    params = calculate_optimal_params(stats, data_path)
    
    # Save to JSON
    import json
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            clean_params[k] = v.tolist()
        elif isinstance(v, np.generic):
            clean_params[k] = v.item()
        else:
            clean_params[k] = v
            
    with open(json_path, 'w') as f:
        json.dump(clean_params, f, indent=4)
    
    print(f"✅ Generated optimized config: {json_path}")
    return True


if __name__ == '__main__':
    main()
