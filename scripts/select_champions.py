import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from brainstroke.core.config import METRIC_DIR, CHAMPION_FILE

GROUPS = {
    "A": ["densenet121_se", "densenet201_se"],
    "B": ["efficientnet_b4_spp", "efficientnetv2_s", "mobilenet_v2"],
    "C": ["resnet50", "resnet101"],
    "D": ["inception_v3", "xception"],
    "E": ["convnext_small"],
}

SEG_MODELS = ["unet", "resunet", "attention_unet", "swin_unet_stroke_only"]


def _load_metric(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    group_champions = {}
    for grp, models in GROUPS.items():
        best = None
        best_score = -1
        for m in models:
            mpath = METRIC_DIR / f"{m}_metrics.json"
            data = _load_metric(str(mpath))
            if data is None:
                continue
            score = data.get("val_macro_recall_best", data.get("macro_recall", 0.0))
            if score > best_score:
                best_score = score
                best = m
        group_champions[grp] = best

    best_seg = None
    best_dice = -1
    for m in SEG_MODELS:
        mpath = METRIC_DIR / f"{m}_seg_metrics.json"
        data = _load_metric(str(mpath))
        if data is None:
            continue
        score = data.get("val_dice_best", data.get("mean_dice", 0.0))
        if score > best_dice:
            best_dice = score
            best_seg = m
    if best_seg is None:
        best_seg = "unet"

    payload = {
        "group_champions": group_champions,
        "best_segmentation": best_seg,
    }
    CHAMPION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved champions ->", CHAMPION_FILE)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
