#!/usr/bin/env python
"""Fake-only generator folders (no matched reals, e.g. SD3.5 / Flux-schnell from Zenodo, Synthbuster fakes)
-> data/ood_fake_only/<name>/fake/*.png, centre-cropped to size x size (never resized; smaller images skipped).

  python scripts/prepare_fake_only.py --src <extracted>/FLUX.1-schnell --name flux_schnell --out data
  python scripts/prepare_fake_only.py --src <extracted>/synthbuster/dalle3 --name synthbuster_dalle3

Cropping here gives exactly the pixels the loader would use (it centre-crops too), saved losslessly as PNG,
so large originals are not re-decoded in every experiment.
"""
import argparse, json
from pathlib import Path
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    dest = Path(a.out) / "ood_fake_only" / a.name / "fake"
    dest.mkdir(parents=True, exist_ok=True)
    st = {"name": a.name, "found": 0, "kept": 0, "skipped_small": 0, "unreadable": 0, "src_ext": {}}
    for p in sorted(Path(a.src).rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        st["found"] += 1
        st["src_ext"][p.suffix.lower()] = st["src_ext"].get(p.suffix.lower(), 0) + 1
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            st["unreadable"] += 1; continue
        w, h = im.size
        if min(w, h) < a.size:
            st["skipped_small"] += 1; continue
        l, t = (w - a.size) // 2, (h - a.size) // 2
        im.crop((l, t, l + a.size, t + a.size)).save(dest / (p.stem + ".png"))
        st["kept"] += 1
    print(json.dumps(st))


if __name__ == "__main__":
    main()
