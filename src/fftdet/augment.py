"""Pixel-domain robustness augmentation (Wang et al., 2020 style): random JPEG, blur, resize.
Applied to *training* images before the FFT so the detector cannot rely on fragile high frequencies."""
from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageFilter


def augment_batch(X: np.ndarray, seed: int, p: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = X.copy()
    for i in range(len(X)):
        if rng.random() > p:
            continue
        im = Image.fromarray((X[i] * 255).astype(np.uint8))
        op = rng.integers(3)
        if op == 0:                      # JPEG
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=int(rng.integers(60, 100)))
            im = Image.open(buf)
        elif op == 1:                    # Gaussian blur
            im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
        else:                            # down/up resize
            s = im.size[0]
            f = float(rng.uniform(0.6, 0.95))
            im = im.resize((max(int(s * f), 8),) * 2, Image.BILINEAR).resize((s, s), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.float32) / 255.0
    return out
