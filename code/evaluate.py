r"""
Оцiнка натренованої моделi.
Запуск:  python code\evaluate.py
"""
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import timm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLASSES, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, MODEL_NAME,
    MODELS_DIR, RESULTS_DIR, EXPERIMENT_TAG,
)
from dataset import make_datasets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {EXPERIMENT_TAG} ({NUM_CLASSES} classes)")

    _, val_ds = make_datasets()
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    ckpt_path = MODELS_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"Loaded {ckpt_path}, val_acc was {ckpt['val_acc']:.3f}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=3)
    print("\n=== Classification Report ===")
    print(report)
    with open(RESULTS_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_size = (16, 14) if NUM_CLASSES > 10 else (8, 6)
    fig, ax = plt.subplots(figsize=fig_size)
    annot_fmt = "d" if NUM_CLASSES <= 10 else ""
    sns.heatmap(cm_norm, annot=(cm if NUM_CLASSES <= 10 else False), fmt=annot_fmt, cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                cbar_kws={"label": "Normalized rate"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {MODEL_NAME} ({EXPERIMENT_TAG})")
    plt.xticks(rotation=60, ha="right", fontsize=8 if NUM_CLASSES > 10 else 10)
    plt.yticks(rotation=0, fontsize=8 if NUM_CLASSES > 10 else 10)
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix -> {cm_path}")

if __name__ == "__main__":
    main()