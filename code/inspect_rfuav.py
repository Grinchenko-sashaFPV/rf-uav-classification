"""
Перевіряє структуру RFUAV перед завантаженням.
"""
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files(repo_id="kitofrank/RFUAV", repo_type="dataset")

print(f"Всього файлів: {len(files)}\n")

top_level = {}
for f in files:
    parts = f.split("/")
    key = parts[0] if len(parts) > 1 else "(root)"
    top_level.setdefault(key, []).append(f)

print("Структура верхнього рівня:")
for key, items in sorted(top_level.items()):
    print(f"  {key}/  ({len(items)} файлів)")
    for sample in items[:3]:
        print(f"    - {sample}")
    if len(items) > 3:
        print(f"    ... та ще {len(items) - 3}")