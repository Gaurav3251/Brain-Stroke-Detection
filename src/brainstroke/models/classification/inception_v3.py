from .timm_backbone import TimmBackboneClassifier
from ...core.config import NUM_CLASSES


class InceptionV3(TimmBackboneClassifier):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, pretrained=True):
        super().__init__("inception_v3", num_classes=num_classes, dropout=dropout, pretrained=pretrained, input_size=299)
        self.input_size = 299
