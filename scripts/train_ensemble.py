import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.models import SegGuidedDenseNet, SegGuidedEfficientNet, ConfidenceEnsemble, UNet
from brainstroke.training import train_ensemble_fusion
from brainstroke.core.utils import load_ckpt
from brainstroke.core.config import PRETRAINED_DIR


def main():
    # Load pretrained models
    unet = UNet(deep_sup=True)
    unet, _, _, _ = load_ckpt(str(PRETRAINED_DIR / "unet_best.pth"), unet)

    sg_dense = SegGuidedDenseNet(pretrained=False)
    sg_dense, _, _, _ = load_ckpt(str(PRETRAINED_DIR / "sg_densenet121_best.pth"), sg_dense)

    sg_eff = SegGuidedEfficientNet(pretrained=False)
    sg_eff, _, _, _ = load_ckpt(str(PRETRAINED_DIR / "sg_efficientnet_b4_best.pth"), sg_eff)

    ensemble = ConfidenceEnsemble(sg_dense, sg_eff)
    ensemble.freeze_backbones()

    train_ensemble_fusion(ensemble, unet)


if __name__ == "__main__":
    main()
