#!/usr/bin/env python
"""SMOKE-TEST ONLY. Builds a tiny fake dataset (noise images; 'fake' ones get a periodic up-sampling-like
grid artefact) so the pipeline can be tested without downloads. Results on it mean NOTHING scientifically."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def make(root, n, size, period, seed):
    rng = np.random.default_rng(seed)
    for split in ("train", "test"):
        for cls in ("REAL", "FAKE"):
            d = Path(root) / split / cls; d.mkdir(parents=True, exist_ok=True)
            for i in range(n if split == "train" else n // 4):
                base = rng.normal(0.5, 0.15, (size // 4, size // 4))
                img = np.kron(base, np.ones((4, 4))) + rng.normal(0, 0.05, (size, size))
                if cls == "FAKE":
                    img += 0.03 * np.cos(2 * np.pi * np.arange(size) / period)[None, :]
                Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / f"{i}.png")

if __name__ == "__main__":
    out = sys.argv[1]
    make(f"{out}/cifake", 400, 32, 2, 0)
    # OOD "generator" with a different artefact period, same folder layout (real/ fake/)
    d = Path(out) / "ood" / "toygen"
    for cls in ("real", "fake"):
        (d / cls).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    for cls in ("real", "fake"):
        for i in range(100):
            img = np.kron(rng.normal(0.5, 0.15, (8, 8)), np.ones((4, 4))) + rng.normal(0, 0.05, (32, 32))
            if cls == "fake":
                img += 0.03 * np.cos(2 * np.pi * np.arange(32) / 4)[None, :]
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / cls / f"{i}.png")
