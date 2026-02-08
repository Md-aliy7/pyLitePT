"""
PLY Segmentation Handler for KPConvX compatibility.

This handler exports/imports PLY files with embedded segmentation labels,
compatible with the KPConvX model training format.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from .base import BaseSegmentationHandler


class PLYSegmentationHandler(BaseSegmentationHandler):
    """
    Handler for PLY files with embedded segmentation labels.
    
    This is designed for compatibility with KPConvX model input format,
    which expects PLY files containing: x, y, z, red, green, blue, label
    """
    EXTENSIONS = {".ply"}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _create_labels(self, num_points: int) -> npt.NDArray[np.int8]:
        """Create default labels array for new point clouds."""
        return np.ones(shape=(num_points,), dtype=np.int8) * self.default_label

    def _read_labels(self, label_path: Path) -> npt.NDArray[np.int8]:
        """
        Read segmentation labels from a PLY file.
        
        The PLY file must contain a 'label' property.
        """
        try:
            # Try reading with the KPConvX-style PLY reader
            data = self._read_ply_binary(label_path)
            if 'label' in data.dtype.names:
                return data['label'].astype(np.int8)
            else:
                raise ValueError(f"PLY file {label_path} does not contain 'label' property")
        except Exception as e:
            raise ValueError(f"Failed to read labels from PLY: {e}")

    def _write_labels(self, label_path: Path, labels: npt.NDArray[np.int8]) -> None:
        """
        Write segmentation labels to a PLY file.
        
        This method requires the original point cloud data to be available.
        It creates a new PLY file with points, colors, and labels.
        
        Note: This is called from PointCloud.save_segmentation_labels() which
        needs to be updated to pass the full point cloud data.
        """
        if not label_path.parent.exists():
            label_path.parent.mkdir(parents=True)
        
        # Labels-only fallback: save as binary for now
        # The full PLY export is handled by write_labeled_ply()
        labels.astype(np.int32).tofile(label_path.with_suffix('.bin'))

    @staticmethod
    def write_labeled_ply(
        output_path: Path,
        points: npt.NDArray[np.float32],
        colors: Optional[npt.NDArray[np.float32]],
        labels: npt.NDArray[np.int8]
    ) -> None:
        """
        Write a complete labeled PLY file in KPConvX format.
        
        Args:
            output_path: Path to write the PLY file
            points: Nx3 array of point coordinates (float32)
            colors: Nx3 array of RGB colors (float32, 0-1 range) or None
            labels: N array of per-point labels (int8)
        """
        import sys
        
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        
        n_points = points.shape[0]
        
        # Prepare colors (default to green if not provided for visibility)
        if colors is None:
            colors_uint8 = np.zeros((n_points, 3), dtype=np.uint8)
            colors_uint8[:, 1] = 255  # Green channel
        else:
            # Convert from 0-1 float to 0-255 uint8
            if colors.max() <= 1.0:
                colors_uint8 = (colors * 255).astype(np.uint8)
            else:
                colors_uint8 = colors.astype(np.uint8)
        
        # Convert labels to int32 for KPConvX compatibility
        labels_int32 = labels.astype(np.int32)
        
        # Build header
        header_lines = [
            'ply',
            f'format binary_{sys.byteorder}_endian 1.0',
            f'element vertex {n_points}',
            'property float32 x',
            'property float32 y',
            'property float32 z',
            'property uint8 red',
            'property uint8 green',
            'property uint8 blue',
            'property int32 label',
            'end_header'
        ]
        
        # Write header
        with open(output_path, 'w') as f:
            for line in header_lines:
                f.write(f"{line}\n")
        
        # Create structured array for binary data
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('label', '<i4')
        ])
        
        data = np.empty(n_points, dtype=dtype)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        data['red'] = colors_uint8[:, 0]
        data['green'] = colors_uint8[:, 1]
        data['blue'] = colors_uint8[:, 2]
        data['label'] = labels_int32
        
        # Append binary data
        with open(output_path, 'ab') as f:
            data.tofile(f)

    @staticmethod
    def _read_ply_binary(filename: Path) -> np.ndarray:
        """
        Read a binary PLY file and return structured numpy array.
        
        This is a simplified version of KPConvX's read_ply function.
        """
        ply_dtypes = {
            b'int8': 'i1', b'char': 'i1',
            b'uint8': 'u1', b'uchar': 'u1',
            b'int16': 'i2', b'short': 'i2',
            b'uint16': 'u2', b'ushort': 'u2',
            b'int32': 'i4', b'int': 'i4',
            b'uint32': 'u4', b'uint': 'u4',
            b'float32': 'f4', b'float': 'f4',
            b'float64': 'f8', b'double': 'f8'
        }
        valid_formats = {
            'ascii': '', 
            'binary_big_endian': '>',
            'binary_little_endian': '<'
        }
        
        with open(filename, 'rb') as plyfile:
            # Check PLY magic
            if b'ply' not in plyfile.readline():
                raise ValueError('File does not start with ply')
            
            # Get format
            fmt_line = plyfile.readline().split()
            if len(fmt_line) < 2:
                raise ValueError('Invalid PLY format line')
            fmt = fmt_line[1].decode()
            
            if fmt == "ascii":
                raise ValueError('ASCII PLY not supported, use binary format')
            
            ext = valid_formats[fmt]
            
            # Parse header
            properties = []
            num_points = None
            line = b''
            
            while b'end_header' not in line and line != b'':
                line = plyfile.readline()
                
                if b'element vertex' in line or b'element' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            num_points = int(parts[2])
                        except ValueError:
                            pass
                
                elif b'property' in line and b'list' not in line:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] in ply_dtypes:
                        properties.append((parts[2].decode(), ext + ply_dtypes[parts[1]]))
            
            if num_points is None or len(properties) == 0:
                raise ValueError('Could not parse PLY header')
            
            # Read data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)
        
        return data
