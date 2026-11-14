#!/usr/bin/env python
"""Convert Tiny-GenImage parquet shards (columns: image, label, generator) into the folder layout used here.

  python scripts/prepare_tiny_genimage.py --parquet-dir ~/Downloads/tiny/data --out data \
      --train-gen SD15 --size 256

Result (all PNG, centre-cropped to size x size; images with a shorter side < size are SKIPPED, never resized,
because resizing would destroy the very artefacts we study):
  data/tinygen/train/{real,fake}   train generator, 80 % of its fakes + a disjoint real pool
  data/tinygen/test/{real,fake}    held-out in-distribution test (20 % of its fakes + disjoint reals)
  data/ood/<GENERATOR>/{real,fake} every other generator, with its own disjoint real chunk
Prints per-generator counts (incl. how many were skipped for being too small) -> report these.
"""
import argparse, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image


def class_names(pf):
    """ClassLabel id -> name maps from the HuggingFace schema metadata ({} if the column is not a ClassLabel)."""
    md = pf.schema_arrow.metadata or {}
    feats = json.loads(md[b"huggingface"].decode()).get("info", {}).get("features", {}) if b"huggingface" in md else {}
    return {col: dict(enumerate(f["names"])) for col, f in feats.items()
            if isinstance(f, dict) and f.get("_type") == "ClassLabel" and "names" in f}


def rows(parquet_dir):
    """Yield (image bytes, label name, generator name, file). Tiny-GenImage stores label and generator as int
    ClassLabels (label: 0=real 1=fake; generator: 0=Real 1=ADM ... 6=SD15 ...), so ids are mapped to names."""
    import pyarrow.parquet as pq
    for f in sorted(Path(parquet_dir).glob("*.parquet")):
        pf = pq.ParquetFile(f)
        cn = class_names(pf)
        for batch in pf.iter_batches(batch_size=256):
            d = batch.to_pydict()
            for i in range(len(d["image"])):
                lab, gen = d["label"][i], d["generator"][i]
                lab = cn.get("label", {}).get(lab, lab)
                gen = cn.get("generator", {}).get(gen, gen)
                yield d["image"][i]["bytes"], lab, gen, f.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-gen", default="SD15", help="generator name used for training")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-per-gen", type=int, default=2000, help="cap fakes kept per generator")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)

    # pass 1: decode, crop, collect (gen, is_fake, array) -> write to a temp pool on disk
    pool = out / "_pool"; pool.mkdir(parents=True, exist_ok=True)
    meta = defaultdict(lambda: {"kept": 0, "skipped_small": 0})
    items = []                      # (gen_name, is_fake, path)
    label_names = gen_names = None
    kept_fake = defaultdict(int)
    for n, (b, label, gen, _) in enumerate(rows(a.parquet_dir)):
        # label/generator are mapped to names in rows() when stored as ClassLabel ints
        gen = str(gen); is_fake = str(label).lower() in ("1", "fake")
        key = "REAL" if not is_fake else gen
        if is_fake and kept_fake[gen] >= a.max_per_gen:
            continue
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        if min(w, h) < a.size:
            meta[key]["skipped_small"] += 1
            continue
        l, t = (w - a.size) // 2, (h - a.size) // 2
        p = pool / f"{key}_{n}.png"
        im.crop((l, t, l + a.size, t + a.size)).save(p)
        meta[key]["kept"] += 1
        if is_fake:
            kept_fake[gen] += 1
        items.append((key, is_fake, p))
    print(json.dumps(meta, indent=2))

    reals = [p for k, f, p in items if not f]
    rng.shuffle(reals)
    fakes = defaultdict(list)
    for k, f, p in items:
        if f:
            fakes[k].append(p)
    gens = sorted(fakes)
    if a.train_gen not in fakes:
        sys.exit(f"train generator {a.train_gen!r} not found. Available: {gens}")

    def put(paths, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for p in paths:
            (dest / p.name).write_bytes(p.read_bytes())

    # real pool: 50 % train gen (80/20 train/test), 50 % shared across OOD generators in equal disjoint chunks
    ood_gens = [g for g in gens if g != a.train_gen]
    half = len(reals) // 2
    id_reals, ood_reals = reals[:half], reals[half:]
    ntr = int(0.8 * len(id_reals))
    tf = fakes[a.train_gen]; rng.shuffle(tf); nf = int(0.8 * len(tf))
    def balanced(r, f):             # equal real/fake counts so accuracy/F1 are not driven by class prior
        k = min(len(r), len(f)); return r[:k], f[:k]
    r, f = balanced(id_reals[:ntr], tf[:nf]); put(r, out / "tinygen/train/real"); put(f, out / "tinygen/train/fake")
    r, f = balanced(id_reals[ntr:], tf[nf:]); put(r, out / "tinygen/test/real"); put(f, out / "tinygen/test/fake")
    chunk = len(ood_reals) // max(len(ood_gens), 1)
    for i, g in enumerate(ood_gens):
        r, f = balanced(ood_reals[i * chunk:(i + 1) * chunk], fakes[g])
        put(r, out / "ood" / g / "real"); put(f, out / "ood" / g / "fake")
    for p in pool.glob("*.png"):
        p.unlink()
    pool.rmdir()
    print(f"train gen={a.train_gen}; OOD gens={ood_gens}; reals: {len(id_reals)} in-dist / {len(ood_reals)} OOD pool")


if __name__ == "__main__":
    main()
