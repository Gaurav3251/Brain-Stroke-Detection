from .densenet121_se import DenseNet121SE
from .densenet201_se import DenseNet201SE
from .efficientnet_b4_spp import EfficientNetB4SPP
from .efficientnetv2_s import EfficientNetV2S
from .mobilenet_v2 import MobileNetV2
from .inception_v3 import InceptionV3
from .xception import Xception
from .convnext_small import ConvNeXtSmall
from .resnet50 import ResNet50
from .resnet101 import ResNet101
from .timm_backbone import TIMM_AVAILABLE

MODEL_REGISTRY = {
    "resnet50": {"cls": ResNet50, "img_size": 224},
    "resnet101": {"cls": ResNet101, "img_size": 224},
    "densenet121_se": {"cls": DenseNet121SE, "img_size": 224},
    "densenet201_se": {"cls": DenseNet201SE, "img_size": 224},
    "efficientnet_b4_spp": {"cls": EfficientNetB4SPP, "img_size": 224},
    "efficientnetv2_s": {"cls": EfficientNetV2S, "img_size": 224},
    "mobilenet_v2": {"cls": MobileNetV2, "img_size": 224},
    "inception_v3": {"cls": InceptionV3, "img_size": 299},
    "xception": {"cls": Xception, "img_size": 299},
    "convnext_small": {"cls": ConvNeXtSmall, "img_size": 224},
}


def get_classifier(model_key: str, pretrained: bool = True):
    key = model_key.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")
    entry = MODEL_REGISTRY[key]
    model = entry["cls"](pretrained=pretrained)
    model.input_size = entry["img_size"]
    return model, entry["img_size"]
