"""
Конфiгурацiя експериментiв RFUAV.
Перемикається мiж 5 DJI або 37 класами через ALL_CLASSES=1.
"""
import os
from pathlib import Path

# Шляхи
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "rfuav_subset" / "ImageSet-AllDrones-MatlabPipeline"
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Режим
USE_ALL_CLASSES = os.environ.get("ALL_CLASSES", "0") == "1"

if USE_ALL_CLASSES:
    CLASSES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    EXPERIMENT_TAG = "all37"
else:
    CLASSES = [
        "DJI FPV COMBO",
        "DJI AVATA2",
        "DJI MINI3",
        "DJI MINI4 PRO",
        "DJI MAVIC3 PRO",
    ]
    EXPERIMENT_TAG = "dji5"

NUM_CLASSES = len(CLASSES)
LABEL_MAP = {name: idx for idx, name in enumerate(CLASSES)}

# Данi
IMG_SIZE = 224
VAL_SPLIT = 0.2
SEED = 42

# Тренування
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_NAME = os.environ.get("MODEL_NAME", "resnet18")
PRETRAINED = True

# Вивiд
LOG_EVERY = 20