import torch
import torch.nn as nn
try:
    import torch_scatter
except ImportError:
    # Fallback to local CPU implementation
    import sys
    import os
    
    # Try to find torch_scatter_cpu in backend_cpu
    try:
        import backend_cpu.torch_scatter_cpu as torch_scatter
        found = True
    except ImportError:
        pass

    if not found:
        # Try to find torch_scatter_cpu in parent directory components (legacy)
        params = os.path.dirname(os.path.realpath(__file__)).split(os.sep)
        # Search up to 3 levels up
        for i in range(3):
            path = os.sep.join(params[:-i-1])
            if os.path.exists(os.path.join(path, 'torch_scatter_cpu.py')):
                if path not in sys.path:
                    sys.path.insert(0, path)
                try:
                    import torch_scatter_cpu as torch_scatter
                    found = True
                    break
                except ImportError:
                    pass
    
    if not found:
        # Last resort: assume it's in python path
        try:
            import torch_scatter_cpu as torch_scatter
        except ImportError:
            torch_scatter = None
try:
    import torch_cluster
except ImportError:
    torch_cluster = None

# from models.losses import build_criteria # Legacy, removed
build_criteria = None # Placeholder
from models.utils.structure import Point
from models.utils import offset2batch
from .builder import MODELS, build_model

from models.modules import PointModel, PointSequential
try:
    import spconv.pytorch as spconv
except ImportError:
    # Fallback to local CPU implementation
    import sys
    import os
    
    # Try to find spconv_cpu in backend_cpu
    try:
        import backend_cpu.spconv_cpu as spconv_cpu
        spconv = spconv_cpu.pytorch
        found = True
    except ImportError:
        pass

    if not found:
        # Try to find spconv_cpu in parent directory components (legacy)
        params = os.path.dirname(os.path.realpath(__file__)).split(os.sep)
        # Search up to 3 levels up
        for i in range(3):
            path = os.sep.join(params[:-i-1])
            if os.path.exists(os.path.join(path, 'spconv_cpu.py')):
                if path not in sys.path:
                    sys.path.insert(0, path)
                try:
                    import spconv_cpu
                    spconv = spconv_cpu.pytorch
                    found = True
                    break
                except ImportError:
                    pass
    
    if not found:
        # Last resort: assume it's in python path
        try:
            import spconv_cpu
            spconv = spconv_cpu.pytorch
        except ImportError:
            spconv = None

import torch.distributed as dist
from tqdm import tqdm
try:
    import pointops
except ImportError:
    try:
        import backend_cpu.pointops as pointops
    except ImportError:
        # Fallback to libs
        import libs.pointops.functions.pointops as pointops

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                # parent.feat = point.feat[inverse]
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict