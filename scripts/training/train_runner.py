import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.models import SwinUNet, UNet, ResUNet, AttentionUNet, SegGuidedClassifier
from brainstroke.model_io import (
    build_classifier,
    build_segmenter,
    load_model_checkpoint,
    get_input_size,
    load_champions,
)
from brainstroke.training import train_model, train_seg_guided_classifier, train_ensemble_fusion
from brainstroke.models import ConfidenceEnsemble


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train runner (one model per config)")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg.get("task", "classify").lower()

    if task == "classify":
        model_key = cfg["model"]
        pretrained = bool(cfg.get("pretrained", True))
        epochs = cfg.get("epochs")
        model = build_classifier(model_key, pretrained=pretrained)
        img_size = get_input_size(model_key)
        train_model(model, model_key, task="classify", img_size=img_size, num_epochs=epochs)

    elif task == "segment":
        model_key = cfg["model"]
        run_name = cfg.get("run_name", model_key)
        pretrained = bool(cfg.get("pretrained", True))
        epochs = cfg.get("epochs")
        stroke_only = cfg.get("stroke_only")
        if model_key == "swin_unet":
            model = SwinUNet(pretrained=pretrained, img_size=224)
        elif model_key == "unet":
            model = UNet(deep_sup=True)
        elif model_key == "resunet":
            model = ResUNet(pretrained=pretrained)
        elif model_key == "attention_unet":
            model = AttentionUNet()
        else:
            raise ValueError(f"Unknown segmenter: {model_key}")
        train_model(
            model,
            run_name,
            task="segment",
            img_size=224,
            num_epochs=epochs,
            stroke_only_override=stroke_only,
        )

    elif task == "seg_guided":
        model_key = cfg["model"]
        champs = load_champions()
        seg_prior = cfg.get("seg_prior", champs.get("best_segmentation", "unet"))
        epochs = cfg.get("epochs", 10)

        base = build_classifier(model_key, pretrained=False)
        base = load_model_checkpoint(base, model_key)
        sg = SegGuidedClassifier(base)
        seg_model = build_segmenter(seg_prior, pretrained=False)
        seg_model = load_model_checkpoint(seg_model, seg_prior)

        img_size = get_input_size(model_key)
        use_amp = model_key != "xception"
        train_seg_guided_classifier(sg, f"sg_{model_key}", seg_model, img_size=img_size, n_epochs=epochs, use_amp=use_amp)

    elif task == "ensemble":
        fine_tune = bool(cfg.get("fine_tune", False))
        epochs = cfg.get("epochs", 15)
        champs = load_champions()
        group_champs = champs.get("group_champions", {})
        best_seg = champs.get("best_segmentation", "unet")

        seg_model = build_segmenter(best_seg, pretrained=False)
        seg_model = load_model_checkpoint(seg_model, best_seg)

        models = []
        sizes = []
        for grp in ["A", "B", "C", "D", "E"]:
            key = group_champs.get(grp)
            if key is None:
                continue
            base = build_classifier(key, pretrained=False)
            base = load_model_checkpoint(base, key)
            sg = SegGuidedClassifier(base)
            sg = load_model_checkpoint(sg, f"sg_{key}")
            models.append((f"sg_{key}", sg, True))
            sizes.append(get_input_size(key))

        if len(models) == 0:
            raise RuntimeError("No seg-guided models found. Train seg-guided champions first.")

        ensemble = ConfidenceEnsemble(models)
        max_size = max(sizes) if sizes else 224
        train_ensemble_fusion(ensemble, seg_model, fine_tune_backbones=fine_tune, img_size=max_size, epochs=epochs)

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
