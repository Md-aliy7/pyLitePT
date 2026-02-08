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
                labels = np.load(seg_path) if os.path.exists(seg_path) else None
                
                color_path = os.path.join(path, 'color.npy')
                if os.path.exists(color_path):
                    stats['has_color'] = True
                
                gt_boxes_path = os.path.join(path, 'gt_boxes.npy')
                if os.path.exists(gt_boxes_path):
                    gt_boxes = np.load(gt_boxes_path)
                    stats['has_detection'] = True
                    if len(gt_boxes) > 0 and gt_boxes.shape[1] > 7:
                        dims = gt_boxes[:, 3:6]
                        labels = gt_boxes[:, 7].astype(int)
                        
                        for i, lbl in enumerate(labels):
                            stats['det_class_counts'][lbl] = stats['det_class_counts'].get(lbl, 0) + 1
                            if lbl not in stats.get('det_box_sums', {}):
                                stats.setdefault('det_box_sums', {})[lbl] = np.zeros(3)
                            stats['det_box_sums'][lbl] += dims[i]
            else:
                data = read_ply_simple(path)
                pts = np.column_stack([data['x'], data['y'], data['z']])
                
                labels = None
                if 'class' in data.dtype.names:
                    labels = data['class']
                elif 'label' in data.dtype.names:
                    labels = data['label']
                
                if 'red' in data.dtype.names:
                    stats['has_color'] = True
                
                # Check sidecar gt_boxes
                gt_boxes_path = path.replace('.ply', '_gt_boxes.npy')
                if os.path.exists(gt_boxes_path):
                    gt_boxes = np.load(gt_boxes_path)
                    stats['has_detection'] = True
                    if len(gt_boxes) > 0 and gt_boxes.shape[1] > 7:
                        dims = gt_boxes[:, 3:6]
                        labels = gt_boxes[:, 7].astype(int)
                        
                        for i, lbl in enumerate(labels):
                            stats['det_class_counts'][lbl] = stats['det_class_counts'].get(lbl, 0) + 1
                            if lbl not in stats.get('det_box_sums', {}):
                                stats.setdefault('det_box_sums', {})[lbl] = np.zeros(3)
                            stats['det_box_sums'][lbl] += dims[i]
            
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
    
    gt_classes = None
    if classes_json_path and os.path.exists(classes_json_path):
        import json
        with open(classes_json_path, 'r') as f:
            gt_classes = json.load(f)
    
    # Segmentation classes
    if gt_classes:
        params['NUM_CLASSES_SEG'] = len(gt_classes)
        params['CLASS_NAMES'] = gt_classes
    else:
        num_seg_classes = len(stats['seg_class_counts'])
        params['NUM_CLASSES_SEG'] = num_seg_classes
        params['CLASS_NAMES'] = [f'class_{i}' for i in sorted(stats['seg_class_counts'].keys())]
    
    # Class weights
    if params['NUM_CLASSES_SEG'] > 0:
        total = sum(stats['seg_class_counts'].values())
        counts = []
        for i in range(params['NUM_CLASSES_SEG']):
            # Use stats count or 0 (to avoid div by zero)
            c = stats['seg_class_counts'].get(i, 0)
            counts.append(c)
        counts = np.array(counts, dtype=np.float32)
        
        # Avoid zero division
        weights = total / (params['NUM_CLASSES_SEG'] * np.maximum(counts, 1))
        # Normalize min to 1.0 (approx) or keep raw. Let's normalize around mean=1 or min=1
        if weights.min() > 0:
             weights = weights / weights.min()
        params['CLASS_WEIGHTS'] = [round(float(w), 2) for w in weights]
    else:
        params['CLASS_WEIGHTS'] = 'auto'
    
    # Detection classes
    # If we have gt_classes, check if 'wall' is in there. Usually last class.
    # We want NUM_CLASSES_DET to exclude 'wall' if it's strictly background.
    # But user wants 100% correctness.
    # Shapes3D: 6 shapes + wall. Detection usually only for shapes.
    # We will assume "wall" is background for detection IF it exists at the end.
    
    is_shapes3d = gt_classes and 'wall' in gt_classes
    if is_shapes3d:
         # 6 shapes
         params['NUM_CLASSES_DET'] = len(gt_classes) - 1
    else:
         params['NUM_CLASSES_DET'] = len(stats['det_class_counts'])
    
    # Mean Sizes
    mean_sizes = []
    if params['NUM_CLASSES_DET'] > 0:
        for i in range(params['NUM_CLASSES_DET']):
            # Assuming classes are 0..N-1
            if i in stats.get('det_box_sums', {}):
                 s = stats['det_box_sums'][i]
                 c = stats['det_class_counts'][i]
                 mean_sizes.append((s / c).tolist())
            else:
                 mean_sizes.append([1.0, 1.0, 1.0]) # Default
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
        print(f"\n🏷️  Segmentation Classes ({len(stats['seg_class_counts'])}):")
        for cls_id, count in sorted(stats['seg_class_counts'].items()):
            pct = 100 * count / total
            print(f"    Class {cls_id}: {count:>10,} ({pct:>5.2f}%)")
    
    if stats['det_class_counts']:
        total = sum(stats['det_class_counts'].values())
        print(f"\n📦 Detection Classes ({len(stats['det_class_counts'])}):")
        for cls_id, count in sorted(stats['det_class_counts'].items()):
            print(f"    Class {cls_id}: {count:>10,}")
            if 'det_box_sums' in stats and cls_id in stats['det_box_sums']:
                 s = stats['det_box_sums'][cls_id]
                 c = count
                 m = s / c
                 print(f"      Mean Size: [{m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f}]")
    
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
