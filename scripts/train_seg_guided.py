import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.models import SegGuidedDenseNet, SegGuidedEfficientNet, UNet
from brainstroke.training import train_seg_guided_classifier
from brainstroke.core.utils import load_ckpt
from brainstroke.core.config import PRETRAINED_DIR


def main():
    # Load pretrained UNet (used to generate seg maps)
    unet = UNet(deep_sup=True)
    unet, _, _, _ = load_ckpt(str(PRETRAINED_DIR / "unet_best.pth"), unet)

    sg_dense = SegGuidedDenseNet(pretrained=True)
    train_seg_guided_classifier(sg_dense, "sg_densenet121", unet)

    sg_eff = SegGuidedEfficientNet(pretrained=True)
    train_seg_guided_classifier(sg_eff, "sg_efficientnet_b4", unet)


if __name__ == "__main__":
    main()
