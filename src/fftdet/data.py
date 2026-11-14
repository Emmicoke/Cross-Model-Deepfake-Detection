"""Dataset discovery and image loading. A dataset = folder with real/ and fake/ sub-folders."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _find_class_dir(root: Path, name: str) -> Path:
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.lower() == name:
            return p
    raise FileNotFoundError(f"{root} has no '{name}' sub-folder (case-insensitive)")


def list_images(root: str | Path, max_per_class: int | None = None, seed: int = 0):
    """Return (paths, labels). label 0 = real, 1 = fake. Subsamples deterministically."""
    root = Path(root)
    paths, labels = [], []
    rng = np.random.default_rng(seed)
    for label, name in ((0, "real"), (1, "fake")):
        d = _find_class_dir(root, name)
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS)
        if not files:
            raise FileNotFoundError(f"no images under {d}")
        if max_per_class and len(files) > max_per_class:
            idx = rng.choice(len(files), max_per_class, replace=False)
            files = [files[i] for i in sorted(idx)]
        paths += files
        labels += [label] * len(files)
    return paths, np.array(labels, dtype=np.int64)


def load_gray(path, size: int) -> np.ndarray:
    """Grayscale float32 in [0,1], centre-cropped to the largest square, then resized to size x size
    only if the crop is not already that size (native 32x32 CIFAKE stays untouched)."""
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    im = im.crop((l, t, l + s, t + s))
    if s != size:
        im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_dataset(root, size: int, max_per_class=None, seed=0):
    """Return (X [N,size,size] float32, y [N], paths)."""
    from tqdm import tqdm
    paths, y = list_images(root, max_per_class, seed)
    X = np.stack([load_gray(p, size) for p in tqdm(paths, desc=f"load {Path(root).name}", leave=False)])
    return X, y, paths


def image_formats(root, max_per_class=2000, seed=0):
    """Count file extensions per class: used to detect JPEG-vs-PNG format bias."""
    paths, y = list_images(root, max_per_class, seed)
    out = {}
    for p, l in zip(paths, y):
        out.setdefault("fake" if l else "real", {}).setdefault(p.suffix.lower(), 0)
        out["fake" if l else "real"][p.suffix.lower()] += 1
    return out
