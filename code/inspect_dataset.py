"""
Інспекція скачаного RFUAV subset.
Показує train і valid класи з кількістю зображень.
"""
from pathlib import Path

ROOT = Path("data/rfuav_subset/ImageSet-AllDrones-MatlabPipeline")


def show(split_name: str):
    split_dir = ROOT / split_name
    if not split_dir.exists():
        print(f"[{split_name}] папка не існує")
        return
    print(f"\n=== {split_name.upper()} ===")
    classes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    total = 0
    for cls in classes:
        n = len(list(cls.glob("*.jpg")))
        total += n
        print(f"  {cls.name}: {n}")
    print(f"  --- Усього: {len(classes)} класів, {total} зображень ---")


show("train")
show("valid")