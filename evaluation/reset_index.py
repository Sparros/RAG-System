from pathlib import Path

paths = [
    Path("data/faiss.index"),
    Path("data/metadata.pkl"),
]

for p in paths:
    if p.exists():
        p.unlink()
        print(f"Deleted {p}")
    else:
        print(f"{p} does not exist")
