"""
Завантажує тільки спектрограмні зображення RFUAV
Скачуються:
  - ImageSet-AllDrones-MatlabPipeline/train/ - тренувальні спектрограми
  - ValidationSet_5Drones/ - валідаційні архіви для 5 DJI
  - weight/ - готові ваги моделей для порівняння

Запуск:  python code\download_rfuav.py
"""
from huggingface_hub import snapshot_download
from pathlib import Path

DATA_DIR = Path("data/rfuav_subset")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Завантажую у {DATA_DIR.absolute()}")

snapshot_download(
    repo_id="kitofrank/RFUAV",
    repo_type="dataset",
    local_dir=str(DATA_DIR),
    allow_patterns=[
        "ImageSet-AllDrones-MatlabPipeline/**",
        "ValidationSet_5Drones/**",
        "weight/**",
        "*.md",
        ".gitattributes",
    ],
)

for p in sorted(DATA_DIR.iterdir())[:10]:
    print(f"  {p.name}")