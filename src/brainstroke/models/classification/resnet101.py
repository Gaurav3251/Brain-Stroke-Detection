from .timm_backbone import TimmBackboneClassifier
from ...core.config import NUM_CLASSES


class ResNet101(TimmBackboneClassifier):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, pretrained=True):
        super().__init__("resnet101", num_classes=num_classes, dropout=dropout, pretrained=pretrained, input_size=224)
        self.input_size = 224
