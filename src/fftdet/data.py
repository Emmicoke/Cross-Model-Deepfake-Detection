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


def list_images(root: str | Path, max_per_class: int | None = None, seed: int = 0, fake_only: bool = False):
    """Return (paths, labels). label 0 = real, 1 = fake. Subsamples deterministically.
    fake_only=True: the folder has only fake/ (no reals), e.g. an extra generator without matched reals."""
    root = Path(root)
    paths, labels = [], []
    rng = np.random.default_rng(seed)
    for label, name in ((0, "real"), (1, "fake")):
        if fake_only and label == 0:
            continue
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
    """Grayscale float32 in [0,1]. Images at least size x size are CENTRE-CROPPED to size x size with no
    resampling (resizing would erase the generator fingerprints we measure). Only images smaller than
    size are up-sampled (bicubic); prepare_* scripts skip such images instead."""
    im = Image.open(path).convert("L")
    w, h = im.size
    if min(w, h) >= size:
        l, t = (w - size) // 2, (h - size) // 2
        im = im.crop((l, t, l + size, t + size))
    else:
        s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        im = im.crop((l, t, l + s, t + s)).resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_dataset(root, size: int, max_per_class=None, seed=0, fake_only: bool = False):
    """Return (X [N,size,size] float32, y [N], paths)."""
    from tqdm import tqdm
    paths, y = list_images(root, max_per_class, seed, fake_only)
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
