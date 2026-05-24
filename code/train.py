r"""Тренування ResNet18 на RFUAV."""
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NUM_CLASSES, CLASSES, BATCH_SIZE, NUM_WORKERS, EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, MODEL_NAME, PRETRAINED,
    MODELS_DIR, RESULTS_DIR, LOG_EVERY, SEED, EXPERIMENT_TAG,
)
from dataset import make_datasets


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Experiment: {EXPERIMENT_TAG} ({NUM_CLASSES} classes)")

    train_ds, val_ds = make_datasets()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {MODEL_NAME} ({n_params:.1f} M parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

            if batch_idx % LOG_EVERY == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{train_correct/train_total:.3f}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(f"E{epoch:02d} | train_loss {train_loss:.3f} acc {train_acc:.3f}"
              f" | val_loss {val_loss:.3f} acc {val_acc:.3f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = MODELS_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": CLASSES,
            }, ckpt_path)
            print(f"  saved best -> {ckpt_path}")

    hist_path = RESULTS_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val acc: {best_val_acc:.3f}")
    print(f"History saved: {hist_path}")


if __name__ == "__main__":
    main()