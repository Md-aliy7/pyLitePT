"""
=============================================================================
LITEPT DYNAMIC VISUALIZATION TOOL
=============================================================================
Visualizes model predictions using PySide6 and VisPy.
Robustly handles Unified, Segmentation-only, and Detection-only modes.
Dynamically loads settings from 'Custom/config.py'.

Usage:
    python Custom/visualize.py --checkpoint results/best_unified_model.pth --data_format ply
"""

import os
import sys
import argparse
import numpy as np
import torch
import gc
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QLabel, QPushButton, QSplitter, QCheckBox, QSlider,
    QFrame, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# VisPy
from vispy import scene
from vispy.color import Color

# Add project root
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.insert(0, parent)
sys.path.insert(0, current)

# Import Config & data utils
import config as cfg
from dataset import CustomDataset
from datasets.utils import collate_fn
from core import LitePTUnifiedCustom
from pcdet_lite.box_utils import boxes_to_corners_3d


# =============================================================================
# COLOR UTILS (DYNAMIC)
# =============================================================================

_COLOR_CACHE = {}

def get_colors(num_classes):
    """Get cached colors for given number of classes."""
    if num_classes not in _COLOR_CACHE:
        _COLOR_CACHE[num_classes] = _generate_colors_fast(num_classes)
    return _COLOR_CACHE[num_classes]

def _generate_colors_fast(num_classes):
    """Generate distinct colors using vectorized HSV conversion."""
    if num_classes == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    colors = np.ones((num_classes, 3), dtype=np.float32)
    
    # Vectorized HSV to RGB
    hues = np.arange(num_classes) / max(num_classes, 1)
    s, v = 0.8, 0.9
    
    i_h = (hues * 6).astype(int) % 6
    f = hues * 6 - np.floor(hues * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    for idx in range(num_classes):
        ih = i_h[idx]
        if ih == 0: colors[idx] = [v, t[idx], p]
        elif ih == 1: colors[idx] = [q[idx], v, p]
        elif ih == 2: colors[idx] = [p, v, t[idx]]
        elif ih == 3: colors[idx] = [p, q[idx], v]
        elif ih == 4: colors[idx] = [t[idx], p, v]
        else: colors[idx] = [v, p, q[idx]]
    
    return colors

# =============================================================================
# VISPY CANVAS
# =============================================================================

class PointCloudCanvas(scene.SceneCanvas):
    """VisPy canvas for rendering point clouds."""
    
    def __init__(self, parent=None):
        super().__init__(keys=None, show=False, parent=parent, bgcolor='white')
        self.unfreeze()
        
        self.view = self.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=45, azimuth=45, elevation=30)
        
        # Point Cloud Visual
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.set_gl_state('translucent', depth_test=True)
        
        # Box Visual
        self.box_visual = scene.visuals.Line(method='gl', width=2, parent=self.view.scene)
        
        # Axis
        scene.visuals.XYZAxis(parent=self.view.scene)
        
        self.freeze()
        
    @property
    def native(self):
        return self.canvas.native if hasattr(self, 'canvas') else super().native
        
    def set_data(self, points, labels=None, colors=None, default_color=(0.5, 0.5, 0.5)):
        if points is None or len(points) == 0:
            self.scatter.set_data(pos=np.zeros((0, 3)), face_color='white')
            return
            
        if colors is not None and labels is not None:
            # Map labels to colors
            # Handle -1 (ignore)
            valid_mask = labels >= 0
            n_points = len(points)
            point_colors = np.ones((n_points, 3), dtype=np.float32) * default_color
            
            valid_labels = labels[valid_mask]
            # Clip to range
            if len(colors) > 0:
                 valid_labels = np.clip(valid_labels, 0, len(colors)-1)
                 point_colors[valid_mask] = colors[valid_labels]
        else:
            point_colors = np.ones((len(points), 3)) * default_color
                
        self.scatter.set_data(pos=points, face_color=point_colors, size=5, edge_width=0)
        
        # Auto-center camera on first data load
        center = points.mean(axis=0)
        extent = np.max(points.max(axis=0) - points.min(axis=0))
        self.view.camera.center = center
        self.view.camera.distance = extent * 1.5
        
    def set_boxes(self, boxes, colors='lime'):
        """
        Args:
            boxes: (N, 7) [x,y,z,dx,dy,dz,rot]
            colors: str or (N, 3)
        """
        if boxes is None or len(boxes) == 0:
            self.box_visual.set_data(pos=np.zeros((0, 3)), color='black', connect='segments')
            return
            
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu()
        else:
            boxes = torch.from_numpy(boxes).float()

        corners = boxes_to_corners_3d(boxes).numpy()
        
        edges = np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ])
        
        start_corners = corners[:, edges[:, 0], :]
        end_corners = corners[:, edges[:, 1], :]
        all_lines = np.stack([start_corners, end_corners], axis=2).reshape(-1, 3)
        
        # Expand colors if provided per-box
        if not isinstance(colors, str) and hasattr(colors, 'shape') and len(colors) == len(boxes):
            # (N, 3) -> (N*24, 3) for 12 edges * 2 vertices
            colors = np.repeat(colors, 24, axis=0)
        
        self.box_visual.set_data(pos=all_lines, color=colors, connect='segments')


# =============================================================================
# MAIN WINDOW
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self, dataset, model, device):
        super().__init__()
        self.setWindowTitle("LitePT Dynamic Visualizer")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("background-color: white;")
        
        self.dataset = dataset
        self.model = model
        self.device = device
        self.idx = 0
        self.show_boxes = True
        self.conf_thresh = 0.5
        
        # Setup Classes & Colors
        self.class_names = getattr(cfg, 'CLASS_NAMES', [f'Class {i}' for i in range(cfg.NUM_CLASSES_SEG)])
        self.colors = get_colors(len(self.class_names))
        
        self._init_ui()
        # Initial load
        if len(self.dataset) > 0:
            self.load_scene(0)
            
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # =========================================================
        # LEFT SIDEBAR
        # =========================================================
        ctrl_frame = QFrame()
        ctrl_frame.setFixedWidth(320)
        ctrl_frame.setStyleSheet("QFrame { background-color: #f0f0f0; border-radius: 8px; }")
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(15, 15, 15, 15)
        ctrl_layout.setSpacing(15)
        
        # 1. Info
        info_box = QGroupBox("Model Info")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(QLabel(f"Device: {self.device}"))
        info_layout.addWidget(QLabel(f"Variant: {cfg.MODEL_VARIANT}"))
        info_layout.addWidget(QLabel(f"Scenes: {len(self.dataset)}"))
        ctrl_layout.addWidget(info_box)
        
        # 2. Navigation
        scene_box = QGroupBox("Scene Selection")
        scene_layout = QVBoxLayout(scene_box)
        
        self.scene_combo = QComboBox()
        self.scene_combo.setMinimumHeight(30)
        for s in self.dataset.scenes:
            self.scene_combo.addItem(os.path.basename(str(s)))
        self.scene_combo.currentIndexChanged.connect(self.combo_scene_changed)
        scene_layout.addWidget(self.scene_combo)
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.prev_scene)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_scene)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        scene_layout.addLayout(nav_layout)
        ctrl_layout.addWidget(scene_box)
        
        # 3. Detection Controls (Dynamic)
        if cfg.NUM_CLASSES_DET > 0:
            det_box = QGroupBox("Detection Controls")
            det_layout = QVBoxLayout(det_box)
            
            self.box_check = QCheckBox("Show Boxes")
            self.box_check.setChecked(True)
            self.box_check.stateChanged.connect(self.toggle_boxes)
            det_layout.addWidget(self.box_check)
            
            # Confidence Slider
            conf_layout = QHBoxLayout()
            conf_layout.addWidget(QLabel("Conf:"))
            self.conf_slider = QSlider(Qt.Horizontal)
            self.conf_slider.setRange(0, 100)
            self.conf_slider.setValue(int(self.conf_thresh * 100))
            self.conf_slider.valueChanged.connect(self.update_conf)
            conf_layout.addWidget(self.conf_slider)
            self.conf_val_label = QLabel(f"{self.conf_thresh:.2f}")
            conf_layout.addWidget(self.conf_val_label)
            det_layout.addLayout(conf_layout)
            
            # IoU Slider (for NMS threshold)
            iou_layout = QHBoxLayout()
            iou_layout.addWidget(QLabel("NMS:"))
            self.iou_slider = QSlider(Qt.Horizontal)
            self.iou_slider.setRange(0, 100)
            self.iou_thresh = 0.1  # Default NMS IoU threshold
            self.iou_slider.setValue(int(self.iou_thresh * 100))
            self.iou_slider.valueChanged.connect(self.update_iou)
            iou_layout.addWidget(self.iou_slider)
            self.iou_val_label = QLabel(f"{self.iou_thresh:.2f}")
            iou_layout.addWidget(self.iou_val_label)
            det_layout.addLayout(iou_layout)
            
            ctrl_layout.addWidget(det_box)
        
        # 4. Legend (Connects to Class Names)
        cbar_box = QGroupBox("Class Legend")
        cbar_layout = QVBoxLayout(cbar_box)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll_widget = QWidget()
        self.legend_layout = QVBoxLayout(scroll_widget)
        self.legend_layout.setSpacing(5)
        
        self.class_labels = []
        for i, name in enumerate(self.class_names):
            row = QHBoxLayout()
            color_box = QLabel()
            color_box.setFixedSize(20, 20)
            if i < len(self.colors):
                r, g, b = int(self.colors[i,0]*255), int(self.colors[i,1]*255), int(self.colors[i,2]*255)
            else:
                r,g,b = 127,127,127
            color_box.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #333;")
            
            lbl = QLabel(f"{name}: --")
            self.class_labels.append(lbl)
            
            row.addWidget(color_box)
            row.addWidget(lbl)
            self.legend_layout.addLayout(row)
            
        scroll.setWidget(scroll_widget)
        cbar_layout.addWidget(scroll)
        ctrl_layout.addWidget(cbar_box, stretch=1)
        
        ctrl_layout.addStretch()
        main_layout.addWidget(ctrl_frame)

        # =========================================================
        # RIGHT: VISUALIZATION
        # =========================================================
        view_splitter = QSplitter(Qt.Horizontal)
        
        # GT
        gt_container = QWidget()
        gt_layout = QVBoxLayout(gt_container)
        gt_layout.setContentsMargins(0, 0, 0, 0)
        gt_layout.setSpacing(2)
        l_gt = QLabel("Ground Truth")
        l_gt.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px; color: #444;")
        l_gt.setAlignment(Qt.AlignCenter)
        gt_layout.addWidget(l_gt)
        self.canvas_gt = PointCloudCanvas()
        gt_layout.addWidget(self.canvas_gt.native)
        view_splitter.addWidget(gt_container)
        
        # Pred
        pred_container = QWidget()
        pred_layout = QVBoxLayout(pred_container)
        pred_layout.setContentsMargins(0, 0, 0, 0)
        pred_layout.setSpacing(2)
        l_pred = QLabel("Prediction")
        l_pred.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px; color: #444;")
        l_pred.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(l_pred)
        self.canvas_pred = PointCloudCanvas()
        pred_layout.addWidget(self.canvas_pred.native)
        view_splitter.addWidget(pred_container)
        
        main_layout.addWidget(view_splitter, stretch=1)
        
        # Link cameras
        self.canvas_pred.view.camera = self.canvas_gt.view.camera

    def prev_scene(self):
        if self.idx > 0:
            self.idx -= 1
            self.scene_combo.setCurrentIndex(self.idx)

    def next_scene(self):
        if self.idx < len(self.dataset) - 1:
            self.idx += 1
            self.scene_combo.setCurrentIndex(self.idx)
            
    def combo_scene_changed(self, index):
        self.idx = index
        self.load_scene(index)
        
    def toggle_boxes(self):
        self.show_boxes = self.box_check.isChecked()
        self.update_visuals()
        
    def update_conf(self):
        val = self.conf_slider.value() / 100.0
        self.conf_thresh = val
        self.conf_val_label.setText(f"{val:.2f}")
        self.update_visuals()
        
    def update_iou(self):
        val = self.iou_slider.value() / 100.0
        self.iou_thresh = val
        self.iou_val_label.setText(f"{val:.2f}")
        self.update_visuals()
        
    def load_scene(self, idx):
        """Loads data and runs inference ONCE. Caches results."""
        # Clean up previous cache to free memory
        if hasattr(self, 'cached_result'):
            del self.cached_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # 1. Prepare Batch
        data = self.dataset[idx]
        batch = collate_fn([data])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        # 2. Inference
        self.model.eval() # consistent visuals
        with torch.no_grad():
            outputs = self.model(batch)
        
        # 3. Cache Results
        self.cached_result = {
            'data': data,
            'outputs': outputs
        }
        
        # 4. Trigger Visualization update
        self.update_visuals()
            
    def update_visuals(self):
        """Re-draws the scene using cached results and current thresholds."""
        if not hasattr(self, 'cached_result'): return
        
        data = self.cached_result['data']
        outputs = self.cached_result['outputs']
        pts = data['coord'].numpy()
            
        # 3. Extract Segmentation
        if 'seg_logits' in outputs:
            preds = outputs['seg_logits'].argmax(1).cpu().numpy()
        else:
            # If model treats as det-only, preds might be missing. Use dummy.
            preds = np.zeros(len(pts), dtype=np.int64)
            
        # 4. Extract Detection
        pred_boxes = None
        # Clear previous text
        if hasattr(self, 'text_visual'):
            self.text_visual.parent = None
        self.text_visual = scene.visuals.Text(parent=self.canvas_pred.view.scene, color='black', font_size=300) # Large font for 3D
        
        if cfg.NUM_CLASSES_DET > 0 and self.show_boxes and 'batch_box_preds' in outputs:
            boxes = outputs['batch_box_preds']
            scores = outputs['point_cls_scores']
            if scores.dim() == 2: scores = scores.squeeze(1)
            mask = scores > self.conf_thresh
            
            if mask.sum() > 0:
                boxes_f = boxes[mask].float()
                scores_f = scores[mask].float()
                
                # --- SEMANTIC-AWARE REFINEMENT (Centralized & Optimized) ---
                if hasattr(self.model, 'det_head') and 'seg_logits' in outputs:
                    # Determine target classes for these points
                    if 'batch_cls_preds' in outputs:
                         cur_cls_preds = outputs['batch_cls_preds'][mask]
                    else:
                         cur_cls_preds = torch.zeros((len(boxes_f), cfg.NUM_CLASSES_DET), device=boxes_f.device)
                         
                    coords_f = torch.from_numpy(pts).to(boxes_f.device)
                    seg_logits_f = outputs['seg_logits'] # This is (N, C) for the batch
                    
                    # Refine scores using the model's optimized vectorized logic
                    scores_f = self.model.det_head.semantic_refinement(
                        boxes_f, scores_f, seg_logits_f, coords_f, cur_cls_preds
                    )
                
                # Apply Oriented NMS
                t_nms_start = time.time()
                from pcdet_lite.iou3d_nms_utils import nms_gpu
                
                keep_idx = nms_gpu(boxes_f, scores_f, thresh=getattr(self, 'iou_thresh', 0.1))
                
                if (time.time() - t_nms_start) > 0.1:
                    print(f"  [Perf] Oriented NMS took {(time.time() - t_nms_start):.3f}s (Boxes: {len(boxes_f)})")
                
                pred_boxes = boxes_f[keep_idx].cpu().numpy()
                final_scores = scores_f[keep_idx].cpu().numpy()
                
                # Get class labels for boxes
                if 'batch_cls_preds' in outputs:
                     cls_preds = outputs['batch_cls_preds']
                     if cls_preds.dim() == 2:
                         # Ensure shapes match before indexing
                         if cls_preds.shape[0] == boxes.shape[0]:
                             cls_preds_f = cls_preds[mask][keep_idx]
                             pred_labels = cls_preds_f.argmax(dim=-1).cpu().numpy()
                         else:
                             pred_labels = np.zeros(len(pred_boxes), dtype=int)
                     else:
                         pred_labels = np.zeros(len(pred_boxes), dtype=int)
                else:
                    pred_labels = np.zeros(len(pred_boxes), dtype=int)
                
                # Assign colors based on class
                if len(pred_labels) > 0 and len(self.colors) > 0:
                     box_colors = self.colors[np.clip(pred_labels, 0, len(self.colors)-1)]
                else:
                     box_colors = 'red'
                     
                self.canvas_pred.set_boxes(pred_boxes, colors=box_colors)
                
                # Add Text Labels (Class + Conf)
                text_pos = pred_boxes[:, :3] # Center
                text_str = []
                # Vectorized text creation if possible? VisPy Text needs list of strings.
                # Optimization: Limit text labels if too many (e.g. > 50) to prevent lag
                MAX_LABELS = 50
                if len(pred_boxes) > MAX_LABELS:
                     print(f"  [Perf] Limiting text labels to {MAX_LABELS} (found {len(pred_boxes)})")
                
                for i in range(min(len(pred_boxes), MAX_LABELS)):
                    c_name = self.class_names[pred_labels[i]] if pred_labels[i] < len(self.class_names) else f"C{pred_labels[i]}"
                    text_str.append(f"{c_name}\n{final_scores[i]:.2f}")
                
                # Reuse existing Text visual
                if not hasattr(self, 'text_visual_pred'):
                     self.text_visual_pred = scene.visuals.Text(parent=self.canvas_pred.view.scene, color='black', font_size=300)
                
                self.text_visual_pred.text = text_str
                self.text_visual_pred.pos = text_pos[:len(text_str)]
                self.text_visual_pred.visible = True
            else:
                 self.canvas_pred.set_boxes(None)
                 if hasattr(self, 'text_visual_pred'): 
                      self.text_visual_pred.visible = False 
        
        gt = data['segment'].numpy() if 'segment' in data else None
        
        # Set Points
        self.canvas_gt.set_data(pts, labels=gt, colors=self.colors)
        self.canvas_pred.set_data(pts, labels=preds, colors=self.colors)
        
        # Set Boxes
        gt_boxes = data.get('gt_boxes', None)
        if gt_boxes is not None and hasattr(gt_boxes, 'numpy'):
                gt_boxes = gt_boxes.numpy()

        if self.show_boxes:
            if gt_boxes is not None:
                # Color GT boxes by their class (index 7)
                if gt_boxes.shape[1] > 7:
                    gt_labels = gt_boxes[:, 7].astype(int)
                    # Data labels are 1-based (1..N). Colors are 0-based (0..N-1).
                    # Shift back -1 for visualization
                    color_indices = np.clip(gt_labels - 1, 0, len(self.colors)-1)
                    gt_colors = self.colors[color_indices]
                    self.canvas_gt.set_boxes(gt_boxes, colors=gt_colors)
                else:
                    self.canvas_gt.set_boxes(gt_boxes, colors='lime')
            else:
                 self.canvas_gt.set_boxes(None)
                 
            if pred_boxes is not None:
                # Colors already set above
                pass 
            else:
                self.canvas_pred.set_boxes(None)
        else:
             self.canvas_gt.set_boxes(None)
             self.canvas_pred.set_boxes(None)
             
        # 6. Update Legend Stats
        if gt is not None:
            for i, name in enumerate(self.class_names):
                # Simple IoU/Acc calc
                support = (gt == i).sum()
                pred_cnt = (preds == i).sum()
                intersection = ((gt == i) & (preds == i)).sum()
                union = support + pred_cnt - intersection
                
                iou = intersection / union if union > 0 else 0.0
                acc = intersection / support if support > 0 else 0.0
                precision = intersection / pred_cnt if pred_cnt > 0 else 0.0
                recall = acc # Recall is the same as Accuracy (TP / TP+FN) for this class context
                
                # Format: Name: IoU=XX P=XX R=XX
                self.class_labels[i].setText(f"{name}: IoU={iou*100:.0f}% P={precision*100:.0f}% R={recall*100:.0f}%")
                if iou > 0.5: 
                    self.class_labels[i].setStyleSheet(f"color: green;")
                else:
                    self.class_labels[i].setStyleSheet(f"color: red;")
                
        self.setWindowTitle(f"LitePT - Scene {self.idx} - Found {len(pred_boxes) if pred_boxes is not None else 0} boxes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--data_format', type=str, default='auto', choices=['auto', 'ply', 'npy'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    # OPTIMIZATION: Check if we need to auto-optimize first
    from auto_optimize import ensure_optimization
    import importlib
    
    if ensure_optimization(cfg.DATA_PATH):
        importlib.reload(cfg)
    
    fmt_arg = args.data_format if args.data_format != 'auto' else 'npy'
    dataset = CustomDataset(cfg.DATA_PATH, split=args.split, cfg=cfg, data_format=fmt_arg)
    print(f"Loaded {len(dataset)} scenes from {cfg.DATA_PATH}")
    
    # Model
    input_channels = dataset.input_channels
    if cfg.INPUT_CHANNELS != 'auto': input_channels = cfg.INPUT_CHANNELS
    
    # Auto-calculate MEAN_SIZE if needed
    det_config = cfg.DETECTION_CONFIG.copy() # Copy to avoid modifying global config permanently
    if cfg.NUM_CLASSES_DET > 0 and det_config.get('MEAN_SIZE') == 'auto':
        print("Auto-calculating box mean sizes from dataset...")
        det_config['MEAN_SIZE'] = dataset.calculate_mean_sizes(cfg.NUM_CLASSES_DET)

    model = LitePTUnifiedCustom(
        in_channels=input_channels,
        num_classes_seg=cfg.NUM_CLASSES_SEG,
        num_classes_det=cfg.NUM_CLASSES_DET,
        variant=cfg.MODEL_VARIANT,
        det_config=det_config
    )
    model.to(device)
    
    # Checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.RESULTS_DIR, 'best_unified_model.pth')
        
    if os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}")
        # Optimization: Map to CPU first, strip optimizer, then GC
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        del ckpt
        import gc
        gc.collect()
        # Robust loading: filter out shape mismatches
        model_state = model.state_dict()
        filtered_state = {}
        msg = []
        for k, v in state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    msg.append(f"Skipping {k} (shape mismatch: ckpt {v.shape} vs model {model_state[k].shape})")
            else:
                pass # Ignore keys not in model
        
        if msg:
            print("\nWarning: Some weights were not loaded due to shape mismatches:")
            for m in msg: print("  " + m)
        
        model.load_state_dict(filtered_state, strict=False)
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Using random weights.")
    
    app = QApplication(sys.argv)
    win = MainWindow(dataset, model, device)
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
