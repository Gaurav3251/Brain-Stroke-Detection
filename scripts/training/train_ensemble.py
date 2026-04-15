import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.model_io import load_champions, build_classifier, build_segmenter, load_model_checkpoint, get_input_size
from brainstroke.models import SegGuidedClassifier, ConfidenceEnsemble
from brainstroke.training import train_ensemble_fusion


def main():
    parser = argparse.ArgumentParser(description="Train confidence-weighted ensemble")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune backbones end-to-end")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

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
    train_ensemble_fusion(ensemble, seg_model, fine_tune_backbones=args.fine_tune, img_size=max_size, epochs=args.epochs)


if __name__ == "__main__":
    main()
