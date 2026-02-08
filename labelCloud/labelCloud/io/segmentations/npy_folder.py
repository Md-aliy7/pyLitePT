"""
NPY Folder Segmentation Handler for LitePT compatibility.

This handler exports/imports segmentation data in the LitePT/Pointcept format:
- coord.npy: Point coordinates (N, 3)
- color.npy: Point colors (N, 3), values 0-255 or 0-1
- segment.npy: Per-point labels (N,)
"""
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from .base import BaseSegmentationHandler


class NPYFolderSegmentationHandler(BaseSegmentationHandler):
    """
    Handler for NPY folder format (LitePT/Pointcept compatible).
    
    Each point cloud is stored as a folder containing:
    - coord.npy: (N, 3) float32 coordinates
    - color.npy: (N, 3) float32 RGB (0-1 range)
    - segment.npy: (N,) int32 labels
    
    Optionally:
    - normal.npy: (N, 3) float32 normals
    - instance.npy: (N,) int32 instance IDs
    """
    EXTENSIONS = {".npy_folder", ".npyfolder"}  # Virtual extension for folder mode

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _create_labels(self, num_points: int) -> npt.NDArray[np.int8]:
        """Create default labels array for new point clouds."""
        return np.ones(shape=(num_points,), dtype=np.int8) * self.default_label

    def _read_labels(self, label_path: Path) -> npt.NDArray[np.int8]:
        """
        Read segmentation labels from NPY folder.
        
        Args:
            label_path: Path to the folder containing segment.npy
        """
        segment_file = label_path / "segment.npy"
        if not segment_file.exists():
            raise ValueError(f"segment.npy not found in {label_path}")
        
        return np.load(segment_file).astype(np.int8)

    def _write_labels(self, label_path: Path, labels: npt.NDArray[np.int8]) -> None:
        """
        Write segmentation labels to NPY file.
        
        Note: Full folder export is handled by write_npy_folder().
        This just saves labels for existing folders.
        """
        if not label_path.exists():
            label_path.mkdir(parents=True)
        
        segment_file = label_path / "segment.npy"
        np.save(segment_file, labels.astype(np.int32))

    @staticmethod
    def write_npy_folder(
        output_path: Path,
        points: npt.NDArray[np.float32],
        colors: Optional[npt.NDArray[np.float32]],
        labels: npt.NDArray[np.int8],
        normals: Optional[npt.NDArray[np.float32]] = None,
        instances: Optional[npt.NDArray[np.int32]] = None,
        gt_boxes: Optional[npt.NDArray[np.float32]] = None,
    ) -> None:
        """
        Write a complete LitePT-compatible NPY folder.
        
        Args:
            output_path: Path to create the folder
            points: Nx3 array of point coordinates (float32)
            colors: Nx3 array of RGB colors (float32, 0-1 or 0-255 range)
            labels: N array of per-point labels
            normals: Optional Nx3 array of point normals
            instances: Optional N array of instance IDs
            gt_boxes: Optional Mx7+ array of bounding boxes [x, y, z, dx, dy, dz, heading, label, ...]
        """
        if not output_path.exists():
            output_path.mkdir(parents=True)
        
        n_points = points.shape[0]
        
        # Save coordinates
        np.save(output_path / "coord.npy", points.astype(np.float32))
        
        # Save colors (normalize to 0-1 if needed)
        if colors is None:
            # Default gray color
            colors = np.ones((n_points, 3), dtype=np.float32) * 0.5
        else:
            colors = colors.astype(np.float32)
            # Normalize to 0-1 if in 0-255 range
            if colors.max() > 1.0:
                colors = colors / 255.0
        np.save(output_path / "color.npy", colors)
        
        # Save segmentation labels
        np.save(output_path / "segment.npy", labels.astype(np.int32))
        
        # Save optional normals
        if normals is not None:
            np.save(output_path / "normal.npy", normals.astype(np.float32))
        
        # Save optional instance IDs
        if instances is not None:
            np.save(output_path / "instance.npy", instances.astype(np.int32))
            
        # Save optional bounding boxes
        if gt_boxes is not None:
            np.save(output_path / "gt_boxes.npy", gt_boxes.astype(np.float32))

    @staticmethod
    def read_npy_folder(
        folder_path: Path,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """
        Read an NPY folder and return (coord, color, segment).
        
        Args:
            folder_path: Path to the NPY folder
            
        Returns:
            Tuple of (coord, color, segment) arrays
        """
        coord = np.load(folder_path / "coord.npy").astype(np.float32)
        
        color_file = folder_path / "color.npy"
        if color_file.exists():
            color = np.load(color_file).astype(np.float32)
        else:
            color = np.ones((coord.shape[0], 3), dtype=np.float32) * 0.5
        
        segment_file = folder_path / "segment.npy"
        if segment_file.exists():
            segment = np.load(segment_file).astype(np.int32)
        else:
            segment = np.zeros(coord.shape[0], dtype=np.int32)
            
        gt_boxes = None
        gt_boxes_file = folder_path / "gt_boxes.npy"
        if gt_boxes_file.exists():
            gt_boxes = np.load(gt_boxes_file).astype(np.float32)
        
        return coord, color, segment, gt_boxes
