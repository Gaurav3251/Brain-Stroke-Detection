import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from ..core.config import PLOT_DIR, CLASSES, NUM_CLASSES
from ..models import get_seg_output

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def save_plot(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] -> {path}")


def plot_class_dist(tr_s, va_s, te_s):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")
    for ax, samples, title in zip(axes, [tr_s, va_s, te_s], ["Train", "Val", "Test"]):
        counts = [sum(s["label"] == i for s in samples) for i in range(NUM_CLASSES)]
        bars = ax.bar(CLASSES, counts, color=["#2563EB", "#DC2626", "#16A34A"])
        ax.bar_label(bars, padding=3)
        ax.set_title(title)
        ax.set_ylim(0, max(counts) * 1.15)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "class_distribution")


def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name} - Training Curves", fontsize=14, fontweight="bold")
    axes[0].plot(history["train_loss"], label="Train", color="#2563EB")
    axes[0].plot(history["val_loss"], label="Val", color="#DC2626")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train", color="#2563EB")
        axes[1].plot(history["val_acc"], label="Val", color="#DC2626")
        axes[1].set_title("Accuracy")
        axes[1].set_ylabel("Accuracy")
    else:
        axes[1].plot(history["val_dice"], label="Val Dice", color="#16A34A")
        axes[1].set_title("Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    save_plot(fig, f"{model_name}_training_curves")


def plot_confusion_matrix(cm, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold")
    for ax, norm in zip(axes, [False, True]):
        data = cm.astype("float") / cm.sum(axis=1, keepdims=True) if norm else cm
        sns.heatmap(data, annot=True, fmt=".2f" if norm else "d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, linewidths=0.5, square=True)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        ax.set_title("Normalized" if norm else "Raw Counts")
    plt.tight_layout()
    save_plot(fig, f"{model_name}_confusion_matrix")


def plot_roc(y_true, y_proba, model_name):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(CLASSES, ["#2563EB", "#DC2626", "#16A34A"])):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{model_name} - ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_plot(fig, f"{model_name}_roc")


def plot_pr(y_true, y_proba, model_name):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(CLASSES, ["#2563EB", "#DC2626", "#16A34A"])):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f"{cls} (AP={auc(rec, prec):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} - PR Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, f"{model_name}_pr")


def plot_confidence_hist(y_proba, y_true, model_name):
    max_conf = y_proba.max(1)
    correct = (y_proba.argmax(1) == y_true)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_conf[correct], bins=30, alpha=0.7, color="#16A34A", label="Correct")
    ax.hist(max_conf[~correct], bins=30, alpha=0.7, color="#DC2626", label="Incorrect")
    ax.set_xlabel("Max Confidence")
    ax.set_title(f"{model_name} - Confidence")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, f"{model_name}_confidence_hist")


def plot_threshold_curves(model, loader, model_name):
    thresholds = np.linspace(0, 1, 21)
    model.eval()
    all_p, all_m = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, masks, _ = batch
            out = model(imgs)
            seg = get_seg_output(out, model)
            all_p.append(torch.sigmoid(seg).cpu())
            all_m.append(masks)
    all_p = torch.cat(all_p)
    all_m = torch.cat(all_m)
    dice_s, iou_s = [], []
    for t in thresholds:
        pb = (all_p > t).float()
        inter = (pb * all_m).sum()
        union = pb.sum() + all_m.sum()
        dice_s.append(((2 * inter + 1) / (union + 1)).item())
        iou_s.append(((inter + 1) / (union - inter + 1)).item())
    best_t = thresholds[np.argmax(dice_s)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, dice_s, "b-o", label="Dice", markersize=4)
    ax.plot(thresholds, iou_s, "g-s", label="IoU", markersize=4)
    ax.axvline(best_t, color="red", linestyle="--", label=f"Best t={best_t:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_title(f"{model_name} - Dice/IoU vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    save_plot(fig, f"{model_name}_threshold")


def plot_model_comparison(results):
    metrics = ["accuracy", "macro_f1", "macro_auc", "cohen_kappa"]
    n = len(results)
    x = np.arange(len(metrics))
    w = 0.8 / n
    colors = ["#2563EB", "#DC2626", "#16A34A", "#CA8A04"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, vals) in enumerate(results.items()):
        bars = ax.bar(x + i * w - (n - 1) * w / 2, [vals.get(m, 0) for m in metrics],
                      w, label=name, color=colors[i % len(colors)], alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_plot(fig, "model_comparison")


def plot_sample_preds(model, dataset, model_name, task="classify", n=9):
    idx = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    cols = 3
    rows = (len(idx) + cols - 1) // cols
    model.eval()
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, j in enumerate(idx):
        img_t, lbl = dataset[j]
        with torch.no_grad():
            lg = model(img_t.unsqueeze(0))
            pred = lg.argmax(1).item()
            conf = F.softmax(lg, -1)[0, pred].item()
        disp = np.clip(img_t.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN), 0, 1)
        axes[i].imshow(disp)
        axes[i].set_title(
            f"GT:{CLASSES[lbl]} | P:{CLASSES[pred]}({conf:.2f})",
            color="green" if pred == lbl.item() else "red",
            fontsize=8,
        )
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{model_name} - Sample Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, f"{model_name}_sample_preds")
