from enum import Enum
from typing import Dict, List, Type

from . import ObjectDetectionFormat, SemanticSegmentationFormat
from .label_formats.base import BaseLabelFormat


class LabelingMode(str, Enum):
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    UNIFIED = "unified"  # Combined seg + det mode

    def get_available_formats(
        self,
    ) -> List[BaseLabelFormat]:
        return LABELING_MODE_TO_FORMAT.get(self, SemanticSegmentationFormat).list()


LABELING_MODE_TO_FORMAT: Dict[LabelingMode, Type[BaseLabelFormat]] = {
    LabelingMode.OBJECT_DETECTION: ObjectDetectionFormat,
    LabelingMode.SEMANTIC_SEGMENTATION: SemanticSegmentationFormat,
    LabelingMode.UNIFIED: SemanticSegmentationFormat,  # Uses seg format with gt_boxes
}
