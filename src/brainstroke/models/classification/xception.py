from .timm_backbone import TimmBackboneClassifier
from ...core.config import NUM_CLASSES


class Xception(TimmBackboneClassifier):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, pretrained=True):
        super().__init__("xception", num_classes=num_classes, dropout=dropout, pretrained=pretrained, input_size=299)
        self.input_size = 299
