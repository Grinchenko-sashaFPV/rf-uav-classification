r"""
Оцiнка робастностi моделi до шуму на спектрограмах.
Запуск:  $env:ALL_CLASSES = "1"; python code\evaluate_snr.py
"""
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLASSES, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, MODEL_NAME,
    MODELS_DIR, RESULTS_DIR, EXPERIMENT_TAG,
)
from dataset import make_datasets


# Рiвнi SNR для тестування
SNR_LEVELS_DB = [-20, -16, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 14, 18]


def add_pixel_noise_snr(images: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Додає гаусiвський шум до тензора зображень з заданим SNR (dB).
    Шум додається в нормалiзованому просторi, що приблизно вiдповiдає
    дiї шуму на пiксельну iнтенсивнiсть.
    """
    # Потужнiсть сигналу = середньоквадратичне значення зображення
    signal_power = images.pow(2).mean()
    # snr_linear = signal_power / noise_power  =>  noise_power = signal_power / snr_linear
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = noise_power.sqrt()
    noise = torch.randn_like(images) * noise_std
    return images + noise


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
    print(f"Loaded {ckpt_path}, baseline val_acc was {ckpt['val_acc']:.3f}")

    # Для вiдтворюваностi результатiв
    torch.manual_seed(42)

    results = {"snr_db": [], "accuracy": []}

    print(f"\nTesting at {len(SNR_LEVELS_DB)} SNR levels...")
    for snr_db in SNR_LEVELS_DB:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"SNR={snr_db:+3d}dB", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # Додаємо шум
                noisy_imgs = add_pixel_noise_snr(imgs, snr_db)
                logits = model(noisy_imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
        acc = correct / total
        results["snr_db"].append(snr_db)
        results["accuracy"].append(acc)
        print(f"  SNR={snr_db:+3d}dB:  accuracy = {acc:.3f}")

    out_json = RESULTS_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_snr_curve.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_json}")

    # Графiк
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["snr_db"], [100 * a for a in results["accuracy"]],
            marker="o", linewidth=2, markersize=8, color="#1f77b4")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy vs SNR — {MODEL_NAME} ({EXPERIMENT_TAG})", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    # Horizontal reference lines
    ax.axhline(y=100/NUM_CLASSES, color="gray", linestyle="--", linewidth=1,
               label=f"Random chance ({100/NUM_CLASSES:.1f}%)")
    ax.axhline(y=95, color="green", linestyle=":", linewidth=1, label="95% threshold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_png = RESULTS_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_snr_curve.png"
    plt.savefig(out_png, dpi=150)
    print(f"Plot   -> {out_png}")


if __name__ == "__main__":
    main()