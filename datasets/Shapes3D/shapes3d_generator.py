"""
Shapes3D Dataset Generator for LitePT
======================================
Generates synthetic 3D geometric shape point clouds for semantic segmentation
and 3D object detection.

Features:
    - Unified NPY folder format (compatible with KPConvX and labelCloud)
    - 6 shape classes + wall background (7 total)
    - RGB colors with noise augmentation  
    - Bounding boxes (gt_boxes) for 3D detection
    - Rotation and scale augmentation

Classes:
    0: cube, 1: sphere, 2: cylinder, 3: cone, 4: pyramid, 5: torus, 6: wall

Usage:
    python shapes3d_generator.py --output data/shapes3d --train 2500 --val 300 --test 200

Arguments:
    --output    Output directory path
    --train     Number of training scenes (default: 2500)
    --val       Number of validation scenes (default: 300)
    --test      Number of test scenes (default: 200)
    --workers   Number of parallel workers (default: auto)

Output Format (per scene folder):
    scene_XXXXX/
    ├── coord.npy     # (N, 3) float32 coordinates
    ├── color.npy     # (N, 3) float32 RGB [0-1]
    ├── segment.npy   # (N,) int32 labels
    └── gt_boxes.npy  # (M, 8) [x,y,z,dx,dy,dz,heading,label]
"""


import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

# Shape classes (6 shapes + wall)
SHAPE_CLASSES = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid', 'torus']
SHAPE_LABELS = {name: i for i, name in enumerate(SHAPE_CLASSES)}
WALL_LABEL = 6  # Background wall label

# RGB colors for each shape
SHAPE_COLORS = {
    'cube':     (255, 50, 50),      # Red
    'sphere':   (50, 255, 50),      # Green
    'cylinder': (50, 50, 255),      # Blue
    'cone':     (255, 255, 50),     # Yellow
    'pyramid':  (255, 50, 255),     # Magenta
    'torus':    (50, 255, 255),     # Cyan
}
WALL_COLOR = (128, 128, 128)  # Gray


def generate_cube(center, size=10.0, num_points=2000):
    """Generate points on all 6 faces of a cube."""
    points = []
    half = size / 2
    pts_per_face = num_points // 6
    
    for axis in range(3):
        for sign in [-1, 1]:
            face_pts = np.random.uniform(-half, half, (pts_per_face, 3))
            face_pts[:, axis] = sign * half
            points.append(face_pts)
    
    pts = np.vstack(points).astype(np.float32)
    pts += np.array(center)
    return pts


def generate_sphere(center, radius=8.0, num_points=2000):
    """Generate points on sphere surface using Fibonacci distribution."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = np.cos(theta) * np.sin(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(phi) * radius
    
    pts = np.column_stack([x, y, z]).astype(np.float32)
    pts += np.array(center)
    return pts


def generate_cylinder(center, radius=6.0, height=15.0, num_points=2000):
    """Generate points on cylinder surface (top, bottom, side)."""
    pts_caps = num_points // 3
    pts_side = num_points - 2 * pts_caps
    
    # Top cap
    r_top = np.sqrt(np.random.uniform(0, 1, pts_caps)) * radius
    theta_top = np.random.uniform(0, 2*np.pi, pts_caps)
    top = np.column_stack([
        r_top * np.cos(theta_top),
        r_top * np.sin(theta_top),
        np.full(pts_caps, height/2)
    ])
    
    # Bottom cap
    r_bot = np.sqrt(np.random.uniform(0, 1, pts_caps)) * radius
    theta_bot = np.random.uniform(0, 2*np.pi, pts_caps)
    bottom = np.column_stack([
        r_bot * np.cos(theta_bot),
        r_bot * np.sin(theta_bot),
        np.full(pts_caps, -height/2)
    ])
    
    # Side
    theta_side = np.random.uniform(0, 2*np.pi, pts_side)
    z_side = np.random.uniform(-height/2, height/2, pts_side)
    side = np.column_stack([
        radius * np.cos(theta_side),
        radius * np.sin(theta_side),
        z_side
    ])
    
    pts = np.vstack([top, bottom, side]).astype(np.float32)
    pts += np.array(center)
    return pts


def generate_cone(center, radius=7.0, height=15.0, num_points=2000):
    """Generate points on cone surface (base + side)."""
    pts_base = num_points // 3
    pts_side = num_points - pts_base
    
    # Base (circle at z=0)
    r_base = np.sqrt(np.random.uniform(0, 1, pts_base)) * radius
    theta_base = np.random.uniform(0, 2*np.pi, pts_base)
    base = np.column_stack([
        r_base * np.cos(theta_base),
        r_base * np.sin(theta_base),
        np.zeros(pts_base)
    ])
    
    # Side (cone surface from base to apex)
    t = np.random.uniform(0, 1, pts_side)
    theta_side = np.random.uniform(0, 2*np.pi, pts_side)
    r_side = radius * (1 - t)
    side = np.column_stack([
        r_side * np.cos(theta_side),
        r_side * np.sin(theta_side),
        t * height
    ])
    
    pts = np.vstack([base, side]).astype(np.float32)
    pts += np.array(center)
    return pts


def generate_pyramid(center, base_size=12.0, height=12.0, num_points=2000):
    """Generate points on 4-sided pyramid (square base + 4 triangular faces)."""
    half = base_size / 2
    pts_per_face = num_points // 5
    
    points = []
    
    # Base (square)
    base_x = np.random.uniform(-half, half, pts_per_face)
    base_y = np.random.uniform(-half, half, pts_per_face)
    base_z = np.zeros(pts_per_face)
    points.append(np.column_stack([base_x, base_y, base_z]))
    
    # 4 triangular faces
    apex = np.array([0, 0, height])
    corners = [
        np.array([-half, -half, 0]),
        np.array([half, -half, 0]),
        np.array([half, half, 0]),
        np.array([-half, half, 0]),
    ]
    
    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i + 1) % 4]
        
        u = np.random.uniform(0, 1, pts_per_face)
        v = np.random.uniform(0, 1, pts_per_face)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v
        
        face_pts = u[:, None] * apex + v[:, None] * c1 + w[:, None] * c2
        points.append(face_pts)
    
    pts = np.vstack(points).astype(np.float32)
    pts += np.array(center)
    return pts


def generate_torus(center, major_radius=8.0, minor_radius=3.0, num_points=2000):
    """Generate points on torus surface."""
    u = np.random.uniform(0, 2*np.pi, num_points)
    v = np.random.uniform(0, 2*np.pi, num_points)
    
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    
    pts = np.column_stack([x, y, z]).astype(np.float32)
    pts += np.array(center)
    return pts


SHAPE_GENERATORS = {
    'cube': generate_cube,
    'sphere': generate_sphere,
    'cylinder': generate_cylinder,
    'cone': generate_cone,
    'pyramid': generate_pyramid,
    'torus': generate_torus,
}


def generate_wall(wall_size, num_points=3000):
    """Generate wall plane at Z=0."""
    w_width, w_height = wall_size
    
    wall_x = np.random.uniform(-w_width/2, w_width/2, num_points)
    wall_y = np.random.uniform(-w_height/2, w_height/2, num_points)
    wall_z = np.zeros(num_points)
    
    wall_points = np.column_stack([wall_x, wall_y, wall_z]).astype(np.float32)
    return wall_points


def add_rgb_with_noise(labels, noise_level=20):
    """Add RGB colors based on labels with noise."""
    n = len(labels)
    colors = np.zeros((n, 3), dtype=np.float32)
    
    for label in np.unique(labels):
        mask = labels == label
        if label == WALL_LABEL:
            base_color = WALL_COLOR
        else:
            shape_name = None
            for name, lbl in SHAPE_LABELS.items():
                if lbl == label:
                    shape_name = name
                    break
            base_color = SHAPE_COLORS.get(shape_name, (128, 128, 128))
        
        noise = np.random.randint(-noise_level, noise_level + 1, (mask.sum(), 3))
        noisy_color = np.clip(np.array(base_color) + noise, 0, 255).astype(np.float32)
        colors[mask] = noisy_color
    
    return colors


def generate_scene(scene_idx, out_dir, split, wall_size=(100, 100), 
                   points_per_shape=2000, wall_points=3000):
    """Generate a single scene with 3-5 floating shapes above a wall."""
    np.random.seed(scene_idx * 1000 + hash(split) % 1000)
    
    # Shape defaults (from generator functions)
    # cube: size=10
    # sphere: radius=8
    # cylinder: radius=6, height=15
    # cone: radius=7, height=15
    # pyramid: base=12, height=12
    # torus: major=8, minor=3
    shape_dims = {
        'cube': (10.0, 10.0, 10.0),
        'sphere': (16.0, 16.0, 16.0), # 2*r
        'cylinder': (12.0, 12.0, 15.0), # 2*r, 2*r, h
        'cone': (14.0, 14.0, 15.0),
        'pyramid': (12.0, 12.0, 12.0),
        'torus': (22.0, 22.0, 6.0) # approx box: 2*(major+minor), 2*(major+minor), 2*minor
    }

    gt_boxes = []

    # Random 3-5 shapes
    n_shapes = np.random.randint(3, 6)
    selected_shapes = np.random.choice(SHAPE_CLASSES, n_shapes, replace=True)
    
    # Position shapes in a grid
    if n_shapes == 3:
        offsets = [(-25, 0), (0, 0), (25, 0)]
    elif n_shapes == 4:
        offsets = [(-25, 15), (25, 15), (-25, -15), (25, -15)]
    else:  # 5
        offsets = [(-30, 18), (0, 18), (30, 18), (-18, -18), (18, -18)]

    all_shape_points = []
    all_shape_labels = []

    for i, shape_name in enumerate(selected_shapes):
        offset_x, offset_y = offsets[i]
        offset_x += np.random.uniform(-4, 4)
        offset_y += np.random.uniform(-4, 4)
        
        # Height above wall: 15-25 units
        height_above_wall = np.random.uniform(15, 25)
        
        generator = SHAPE_GENERATORS[shape_name]
        center_pos = [offset_x, offset_y, height_above_wall + 8]
        center = np.array(center_pos)
        pts = generator(center, num_points=points_per_shape)
        
        label = SHAPE_LABELS[shape_name]
        labels = np.full(len(pts), label, dtype=np.int32)
        
        all_shape_points.append(pts)
        all_shape_labels.append(labels)

        # Create BBox [x, y, z, dx, dy, dz, heading, label]
        dims = shape_dims[shape_name]
        gt_boxes.append([center[0], center[1], center[2], dims[0], dims[1], dims[2], 0.0, float(label)])
    
    gt_boxes = np.array(gt_boxes, dtype=np.float32)

    # Generate wall
    wall_pts = generate_wall(wall_size, wall_points)
    wall_labels = np.full(len(wall_pts), WALL_LABEL, dtype=np.int32)
    
    # Combine all points
    all_points = np.vstack(all_shape_points + [wall_pts])
    all_labels = np.concatenate(all_shape_labels + [wall_labels])
    
    # Random rotation around Z axis
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    all_points = all_points @ R.T
    
    # Apply rotation to boxes
    # Center rotation
    gt_boxes[:, :3] = gt_boxes[:, :3] @ R.T
    # Heading rotation
    gt_boxes[:, 6] += theta
    # Normalize heading to [-pi, pi]
    gt_boxes[:, 6] = (gt_boxes[:, 6] + np.pi) % (2 * np.pi) - np.pi

    # Random scale
    scale = np.random.uniform(0.9, 1.1)
    all_points *= scale
    
    # Apply scale to boxes
    gt_boxes[:, :6] *= scale # Scale centers (0-2) and dims (3-5)

    # Small position noise
    shift = np.random.normal(0, 0.05, all_points.shape).astype(np.float32)
    all_points += shift
    
    # Apply mean noise shift to boxes to keep alignment
    # (Approximation: using mean shift of the scene is usually close to 0, 
    # but strictly we should track the shift per point. 
    # Since shift is random per point, the box center technically stays the same in expectation.
    # We will skip adding per-point noise to the box center to maintain ground truth "cleanliness" 
    # relative to the underlying shape, or we could add the mean shift.)
    mean_shift = np.mean(shift, axis=0)
    gt_boxes[:, :3] += mean_shift

    # Add RGB colors
    colors = add_rgb_with_noise(all_labels, noise_level=20)
    
    # Save as .npy files in scene folder
    scene_name = f'scene_{scene_idx:05d}'
    scene_dir = os.path.join(out_dir, split, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    
    np.save(os.path.join(scene_dir, 'coord.npy'), all_points.astype(np.float32))
    np.save(os.path.join(scene_dir, 'color.npy'), colors.astype(np.float32))
    np.save(os.path.join(scene_dir, 'segment.npy'), all_labels.astype(np.int32))
    np.save(os.path.join(scene_dir, 'gt_boxes.npy'), gt_boxes.astype(np.float32))
    
    return len(all_points), n_shapes


def generate_split(args):
    """Generate scenes for a split (for multiprocessing)."""
    scene_idx, out_dir, split = args
    n_pts, n_shapes = generate_scene(scene_idx, out_dir, split)
    return scene_idx, n_pts, n_shapes


def main():
    parser = argparse.ArgumentParser(description='Generate Shapes3D dataset for LitePT')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--train', type=int, default=30,
                        help='Number of training scenes')
    parser.add_argument('--val', type=int, default=5,
                        help='Number of validation scenes')
    parser.add_argument('--test', type=int, default=5,
                        help='Number of test scenes')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0=sequential)')
    args = parser.parse_args()
    
    # Get project root
    current = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(os.path.dirname(current))
    out_dir = os.path.join(project_root, args.output)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    
    print(f"Generating Shapes3D dataset in {out_dir}")
    print(f"  Train: {args.train} scenes")
    print(f"  Val:   {args.val} scenes")
    print(f"  Test:  {args.test} scenes")
    
    # Save classes.json
    import json
    class_names = SHAPE_CLASSES + ['wall']
    with open(os.path.join(out_dir, 'classes.json'), 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"  Classes: {class_names}")
    print()
    
    splits = [
        ('train', args.train),
        ('val', args.val),
        ('test', args.test),
    ]
    
    for split_name, n_scenes in splits:
        print(f"Generating {split_name} split ({n_scenes} scenes)...")
        
        tasks = [(i, out_dir, split_name) for i in range(n_scenes)]
        
        if args.workers > 0:
            with Pool(min(args.workers, cpu_count())) as pool:
                results = list(pool.imap(generate_split, tasks))
        else:
            results = []
            for i, task in enumerate(tasks):
                result = generate_split(task)
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{n_scenes} scenes generated")
        
        total_pts = sum(r[1] for r in results)
        avg_pts = total_pts / n_scenes
        print(f"  Completed: {n_scenes} scenes, avg {avg_pts:.0f} pts/scene")
        print()
    
    print("Dataset generation complete!")
    print(f"Data saved to: {out_dir}")


if __name__ == '__main__':
    main()
