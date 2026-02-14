"""
=============================================================================
DATASET LOADER - GENERIC & FLEXIBLE
=============================================================================
Generic dataset loader for NPY folder format.
Automatically detects available features (color, normal, etc.) and stacks them.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Dataset for PLY files and NPY folder format.
    
    Features:
    - Supports both PLY files (labelCloud) and NPY folders (Pointcept)
    - Auto-detects input channels (coord, color, normal, intensity, etc.)
    - Handles flexible features stacking
    - Auto-splits validation if not provided
    - Calculates class weights automatically
    """
    
    def __init__(self, data_root, split='train', cfg=None, data_format='npy'):
        """
        Args:
            data_root: Path to dataset root
            split: 'train', 'val', or 'test'
            cfg: Config module with settings
            data_format: 'ply' for PLY files, 'npy' for NPY folder format
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.data_format = data_format
        self.cfg = cfg
        self.ignore_index = getattr(cfg, 'IGNORED_LABELS', [-1])[0]
        self.load_seg = (getattr(cfg, 'NUM_CLASSES_SEG', 0) > 0)
        self.label_mapping = getattr(cfg, 'LABEL_MAPPING', {})
        self.grid_size = getattr(cfg, 'GRID_SIZE', 0.02)
        
        # 1. Discover Scenes
        self.scenes = self._discover_scenes(data_root, split)
        print(f"CustomDataset [{split}]: Found {len(self.scenes)} scenes in {data_root} (format: {data_format})")
        
        # 2. Detect Features from first scene
        self.feature_names = ['coord'] # Always start with coord
        self.input_channels = 3 # x, y, z
        
        if len(self.scenes) > 0:
            first_scene = self.scenes[0]
            
            if self.data_format == 'ply':
                # PLY loader always returns coord+color (6 channels)
                self.feature_names = ['coord', 'color']
                self.input_channels = 6
            else:
                # Check for common feature files (NPY folder)
                potential_features = {
                    'color.npy': 3,      # RGB
                    'normal.npy': 3,     # nx, ny, nz
                    'intensity.npy': 1,  # i
                    'feature.npy': None  # custom (check shape)
                }
            
                for fname, chans in potential_features.items():
                    fpath = os.path.join(first_scene, fname)
                    if os.path.exists(fpath):
                        self.feature_names.append(fname.replace('.npy', ''))
                        if chans is None:
                            # Infer channels from file
                            data = np.load(fpath)
                            chans = data.shape[1] if len(data.shape) > 1 else 1
                        self.input_channels += chans
        
        print(f"CustomDataset [{split}]: Input features: {self.feature_names} -> Total Channels: {self.input_channels}")

    def _discover_scenes(self, data_root, split):
        """Discover scenes handling both split folders and flat structure.
        
        For PLY format: looks for .ply files
        For NPY format: looks for folders with coord.npy
        """
        split_dir = os.path.join(data_root, split)
        
        if self.data_format == 'ply':
            # Discover PLY files
            if os.path.exists(split_dir):
                scenes = sorted(glob.glob(os.path.join(split_dir, '*.ply')))
            else:
                all_scenes = sorted(glob.glob(os.path.join(data_root, '*.ply')))
                n = len(all_scenes)
                if n == 0: return []
                
                n_train = int(n * 0.8)
                n_val = int(n * 0.1)
                
                if split == 'train':
                    scenes = all_scenes[:n_train]
                elif split == 'val':
                    scenes = all_scenes[n_train:n_train+n_val]
                else:
                    scenes = all_scenes[n_train+n_val:]
        else:
            # Discover NPY folders
            if os.path.exists(split_dir):
                scenes = sorted([
                    d for d in glob.glob(os.path.join(split_dir, '*'))
                    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'coord.npy'))
                ])
            else:
                all_scenes = sorted([
                    d for d in glob.glob(os.path.join(data_root, '*'))
                    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'coord.npy'))
                ])
                n = len(all_scenes)
                if n == 0: return []
                
                n_train = int(n * 0.8)
                n_val = int(n * 0.1)
                
                if split == 'train':
                    scenes = all_scenes[:n_train]
                elif split == 'val':
                    scenes = all_scenes[n_train:n_train+n_val]
                else:
                    scenes = all_scenes[n_train+n_val:]
        return scenes
    
    def __len__(self):
        """Return the total number of scenes in the dataset."""
        return len(self.scenes)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the scene.

        Returns:
            dict: Dictionary containing:
                - coord (Tensor): Point coordinates (N, 3)
                - segment (Tensor, optional): Semantic labels (N)
                - instance (Tensor, optional): Instance labels (N)
                - gt_boxes (Tensor, optional): Ground truth boxes (M, 7)
                - ... and other features/metadata.
        """
        scene_path = self.scenes[idx]
        
        if self.data_format == 'ply':
            # Load PLY file
            coord, features, segment, gt_boxes = self._load_ply(scene_path)
        else:
            # Load NPY folder
            coord, features, segment, gt_boxes = self._load_npy(scene_path)
        
        # 3. Augmentation (Training only)
        if self.split == 'train' and getattr(self.cfg, 'AUGMENT', True):
            coord, features, gt_boxes = self._augment(coord, features, gt_boxes)
            
        # Build Item
        data = {
            'coord': torch.from_numpy(coord),
            'feat': torch.from_numpy(features),
            'segment': torch.from_numpy(segment),
            'gt_boxes': torch.from_numpy(gt_boxes),
            'name': os.path.basename(scene_path),
            'split': self.split,
            'grid_size': np.array([self.grid_size], dtype=np.float32),
        }
        
        # For backward compatibility
        if features.shape[1] >= 6:
            data['color'] = torch.from_numpy(features[:, 3:6])

        # 4. Grid Sampling (Voxelization) - OPTIMIZED for speed
        # We perform this manually here to avoid complex imports from datasets.transform
        # and to ensure deterministic behavior for validation.
        use_grid_sample = getattr(self.cfg, 'USE_GRID_SAMPLE', True)
        
        if use_grid_sample:
            scale_coord = coord / self.grid_size
            grid_coord = np.floor(scale_coord).astype(np.int32)  # Use int32 instead of int64 for speed
            
            # OPTIMIZATION: Use hash-based unique instead of np.unique (much faster!)
            # Create unique voxel hash using bit shifting (faster than multiplication)
            # Assumes coordinates are within reasonable range (-2^10 to 2^10)
            offset = 1024  # Offset to handle negative coordinates
            grid_coord_offset = grid_coord + offset
            
            # Create hash: x + y*2048 + z*2048*2048 (bit-shift equivalent)
            hash_vals = (grid_coord_offset[:, 0] + 
                        grid_coord_offset[:, 1] * 2048 + 
                        grid_coord_offset[:, 2] * (2048 * 2048))
            
            if self.split == 'train':
                # Training: Random point per voxel
                # Shuffle indices first for randomness
                perm = np.random.permutation(len(hash_vals))
                hash_vals_shuffled = hash_vals[perm]
                _, unique_idx = np.unique(hash_vals_shuffled, return_index=True)
                indices = perm[unique_idx]
            else:
                # Validation/Test: Deterministic (First point per voxel)
                _, indices = np.unique(hash_vals, return_index=True)
            
            # Subsample data
            data['coord'] = data['coord'][indices]
            data['feat'] = data['feat'][indices]
            data['segment'] = data['segment'][indices]
            # data['gt_boxes'] remains scene-level, no subsampling needed
            
            # Re-generate color if it exists
            if 'color' in data:
                data['color'] = data['color'][indices]

        return data

    def _augment(self, coord, features, gt_boxes):
        """Standard point cloud and box augmentations."""
        # 1. Random Flip (X and Y axis)
        if np.random.random() > 0.5:
            coord[:, 0] = -coord[:, 0]
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                gt_boxes[:, 6] = -gt_boxes[:, 6]
        if np.random.random() > 0.5:
            coord[:, 1] = -coord[:, 1]
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]
        
        # 2. Random Rotation (Around Z axis)
        angle = np.random.uniform(-np.pi/4, np.pi/4) # +/- 45 degrees
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        coord = coord @ rot_mat.T
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, 0:3] = gt_boxes[:, 0:3] @ rot_mat.T
            gt_boxes[:, 6] += angle
            
        # 3. Random Scaling
        scale = np.random.uniform(0.9, 1.1)
        coord = coord * scale
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, 0:6] *= scale # Scale center and dims
            
        # 4. Point Jitter
        noise = np.random.normal(0, 0.005, size=coord.shape).astype(np.float32)
        coord += noise
        
        # Update features (coords are first 3)
        features[:, 0:3] = coord
        
        return coord, features, gt_boxes
    
    def _load_ply(self, ply_path):
        """Load data from PLY file (labelCloud format)."""
        data = self._read_ply_binary(ply_path)
        
        coord = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
        
        # Load colors
        if 'red' in data.dtype.names:
            r = data['red'].astype(np.float32)
            g = data['green'].astype(np.float32)
            b = data['blue'].astype(np.float32)
            if r.max() > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            color = np.vstack((r, g, b)).T
        else:
            color = np.ones((coord.shape[0], 3), dtype=np.float32) * 0.5
        
        features = np.concatenate([coord, color], axis=1)
        
        # Load labels
        if 'label' in data.dtype.names and self.load_seg:
            segment = data['label'].astype(np.int64)
            # Optional: Apply label mapping from config
            label_mapping = getattr(self, 'label_mapping', {})
            if label_mapping:
                for k, v in label_mapping.items():
                    segment[segment == k] = v
        else:
            # Default to ignore_index if no labels
            segment = np.full(coord.shape[0], self.ignore_index, dtype=np.int64)
        
        # Check for sidecar gt_boxes (exported by labelCloud)
        gt_boxes = np.zeros((0, 8), dtype=np.float32)
        gt_boxes_path = ply_path.replace('.ply', '_gt_boxes.npy')
        if os.path.exists(gt_boxes_path):
            gt_boxes = np.load(gt_boxes_path).astype(np.float32)
            # Labels are already 0-based from annotation tools - no conversion needed
        
        return coord, features, segment, gt_boxes
    
    def _load_npy(self, scene_path):
        """Load data from NPY folder."""
        try:
            # 1. Load Coordinates
            coord = np.load(os.path.join(scene_path, 'coord.npy')).astype(np.float32)
            
            # Optional: Clip to range if configured (for detection only)
            if hasattr(self.cfg, 'DETECTION_CONFIG'):
                 pc_range = self.cfg.DETECTION_CONFIG.get('POINT_CLOUD_RANGE')
                 if pc_range:
                     mask = (coord[:, 0] >= pc_range[0]) & (coord[:, 0] <= pc_range[3]) & \
                            (coord[:, 1] >= pc_range[1]) & (coord[:, 1] <= pc_range[4]) & \
                            (coord[:, 2] >= pc_range[2]) & (coord[:, 2] <= pc_range[5])
                     coord = coord[mask]
            
            # 2. Load Features (Dynamic)
            features_list = [coord]
            
            for fname in self.feature_names:
                if fname == 'coord': continue
                
                fpath = os.path.join(scene_path, f'{fname}.npy')
                try:
                    feat = np.load(fpath).astype(np.float32)
                except FileNotFoundError:
                    continue
                    
                if fname == 'color' and feat.max() > 1.0:
                    feat = feat * (1.0 / 255.0)
                
                if len(feat.shape) == 1:
                    feat = feat[:, None]
                
                features_list.append(feat)
            
            if len(features_list) > 1:
                features = np.concatenate(features_list, axis=1)
            else:
                features = coord
    
            # Force input channels if specified in config
            cfg_channels = getattr(self.cfg, 'INPUT_CHANNELS', 'auto')
            if cfg_channels != 'auto' and isinstance(cfg_channels, int):
                 if features.shape[1] > cfg_channels:
                     features = features[:, :cfg_channels]
            
            # 3. Load Labels
            segment = None
            if self.load_seg:
                seg_path = os.path.join(scene_path, 'segment.npy')
                if os.path.exists(seg_path):
                    segment = np.load(seg_path).astype(np.int64)
            
            if segment is None:
                segment = np.full(coord.shape[0], self.ignore_index, dtype=np.int64)
    
            # 4. Load Detection Boxes
            gt_boxes_path = os.path.join(scene_path, 'gt_boxes.npy')
            if os.path.exists(gt_boxes_path):
                gt_boxes = np.load(gt_boxes_path).astype(np.float32)
                # Labels are already 0-based from annotation tools - no conversion needed
            else:
                gt_boxes = np.zeros((0, 8), dtype=np.float32)
            
            return coord, features, segment, gt_boxes

        except Exception as e:
            print(f"Error loading NPY scene {scene_path}: {e}")
            # Return dummy data
            return np.zeros((1,3), np.float32), np.zeros((1,3), np.float32), np.zeros(1, np.int64), np.zeros((0,8), np.float32)
    
    @staticmethod
    def _read_ply_binary(filename):
        """Read binary PLY file and return structured numpy array."""
        ply_dtypes = {
            b'int8': 'i1', b'char': 'i1', b'uint8': 'u1', b'uchar': 'u1',
            b'int16': 'i2', b'short': 'i2', b'uint16': 'u2', b'ushort': 'u2',
            b'int32': 'i4', b'int': 'i4', b'uint32': 'u4', b'uint': 'u4',
            b'float32': 'f4', b'float': 'f4', b'float64': 'f8', b'double': 'f8'
        }
        valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}
        
        with open(filename, 'rb') as f:
            if b'ply' not in f.readline():
                raise ValueError('Not a PLY file')
            
            fmt_line = f.readline().split()
            fmt = fmt_line[1].decode() if len(fmt_line) > 1 else 'binary_little_endian'
            ext = valid_formats.get(fmt, '<')
            
            properties = []
            num_points = None
            line = b''
            
            while b'end_header' not in line:
                line = f.readline()
                if b'element vertex' in line:
                    num_points = int(line.split()[2])
                elif b'property' in line and b'list' not in line:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] in ply_dtypes:
                        properties.append((parts[2].decode(), ext + ply_dtypes[parts[1]]))
            
            if num_points is None or len(properties) == 0:
                raise ValueError('Could not parse PLY header')
            
            data = np.fromfile(f, dtype=properties, count=num_points)
        
        return data
    
    def get_class_distribution(self):
        """Calculate class distribution across all scenes."""
        distribution = {}
        for scene_path in self.scenes:
            seg_path = os.path.join(scene_path, 'segment.npy')
            if os.path.exists(seg_path):
                segment = np.load(seg_path)
                unique, counts = np.unique(segment, return_counts=True)
                for u, c in zip(unique, counts):
                    if u != self.ignore_index:
                        distribution[int(u)] = distribution.get(int(u), 0) + int(c)
        return distribution
    
    def calculate_class_weights(self, num_classes):
        """Calculate balanced class weights with caching."""
        import json
        cache_path = os.path.join(self.data_root, 'class_weights_cache.json')
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    if len(cached) == num_classes:
                        print(f"Loading cached class weights from {cache_path}")
                        return cached
            except Exception:
                pass

        print("Calculating class weights (this may take a moment for large datasets)...")
        dist = self.get_class_distribution()
        if not dist:
            return [1.0] * num_classes
        
        total = sum(dist.values())
        weights = []
        for i in range(num_classes):
            count = dist.get(i, 0)
            weight = total / (num_classes * (count + 1e-6))
            # DAMPENING: sqrt to prevent extreme weights
            weights.append(float(weight ** 0.5))
            
        # Normalize to mean 1.0
        weights = np.array(weights)
        weights = weights / weights.mean()
        weights_list = weights.tolist()
        
        # Save cache
        try:
            with open(cache_path, 'w') as f:
                json.dump(weights_list, f)
        except Exception:
            pass
            
        return weights_list

    def calculate_mean_sizes(self, num_classes):
        """Calculate mean box sizes per class with caching."""
        import json
        
        # Check for cache file in data root
        cache_path = os.path.join(self.data_root, 'mean_sizes_cache.json')
        if os.path.exists(cache_path):
            print(f"Loading cached mean sizes from {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if len(cached_data) == num_classes:
                         return cached_data
                    else:
                        print("Cache mismatch (class count), recalculating...")
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        sums = {}
        counts = {}
        
        print("Calculating mean box sizes from scratch...")
        for scene_path in self.scenes:
            # Check for different gt_boxes paths to be robust
            if self.data_format == 'ply':
                 gt_path = scene_path.replace('.ply', '_gt_boxes.npy')
            else:
                 gt_path = os.path.join(scene_path, 'gt_boxes.npy')
                 
            if os.path.exists(gt_path):
                boxes = np.load(gt_path)
                if boxes.shape[0] > 0 and boxes.shape[1] > 7:
                    dims = boxes[:, 3:6]  # dx, dy, dz
                    labels = boxes[:, 7].astype(int)
                    
                    for i in range(len(labels)):
                        lbl = labels[i]
                        if lbl not in sums:
                            sums[lbl] = np.zeros(3)
                            counts[lbl] = 0
                        sums[lbl] += dims[i]
                        counts[lbl] += 1
                        
        mean_sizes = []
        for i in range(num_classes):
            target_label = i + 1
            
            if target_label in sums and counts[target_label] > 0:
                mean_sizes.append((sums[target_label] / counts[target_label]).tolist())
            elif i in sums and counts[i] > 0:
                 # Fallback to 0-based if found (mixed data?)
                 mean_sizes.append((sums[i] / counts[i]).tolist())
            else:
                mean_sizes.append([1.0, 1.0, 1.0]) # Default fallback
        
        # Save cache
        try:
            with open(cache_path, 'w') as f:
                json.dump(mean_sizes, f)
            print(f"Saved mean sizes cache to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
                
        return mean_sizes
