#!/usr/bin/env python
"""Build the folder layout used here from the HuggingFace dataset jhutter2/281_Genimage (a small GenImage subset).

  python scripts/prepare_281_genimage.py --out data            # downloads the repo snapshot (~1.4 GB)
  python scripts/prepare_281_genimage.py --raw-dir <snapshot> --out data

Repo layout: <generator>/ai/*.PNG (fakes) and <generator>/nature/*.PNG (ImageNet reals) for
generator in adm, stable_diffusion_v_1_5, wukong. Its HF configs adm / stable_diffusion / wukong point at the
ai/ folders and the config "nature" points at adm/nature only. By default (--reals all) the reals of all three
nature/ folders are pooled (exact duplicates removed by MD5), which roughly triples the real pool;
--reals nature_cfg uses only the "nature" config (adm/nature) as load_dataset(..., "nature") would.

Result (PNG, centre-cropped to size x size, never resized; images smaller than size are skipped and counted):
  data/g281/train/{real,fake}        80 % of the SD v1.5 fakes + 80 % of half the real pool
  data/g281/test/{real,fake}         the other 20 % of both
  data/ood/g281_adm/{real,fake}      ADM fakes + a disjoint 25 % chunk of the real pool
  data/ood/g281_wukong/{real,fake}   Wukong fakes + the last disjoint 25 % chunk
Every folder is balanced (equal real/fake counts). Seed 0 by default.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

FAKE_DIRS = {"sd": "stable_diffusion_v_1_5/ai", "adm": "adm/ai", "wukong": "wukong/ai"}
NATURE_DIRS = ["adm/nature", "stable_diffusion_v_1_5/nature", "wukong/nature"]
EXTS = {".png", ".jpg", ".jpeg"}


def files(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None, help="local snapshot of jhutter2/281_Genimage (downloaded if missing)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--reals", choices=["all", "nature_cfg"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    raw = Path(a.raw_dir) if a.raw_dir else None
    if raw is None or not raw.exists():
        from huggingface_hub import snapshot_download
        raw = Path(snapshot_download("jhutter2/281_Genimage", repo_type="dataset",
                                     local_dir=str(raw) if raw else None, max_workers=16))

    stats = {}

    def crop_ok(p):
        """True if decodable and both sides >= size; None if the file cannot be decoded (the repo has some)."""
        try:
            with Image.open(p) as im:
                im.load()
                return min(im.size) >= a.size
        except Exception:
            return None

    # reals: pool, dedupe by content hash, drop too-small images
    seen, reals = set(), []
    for d in (NATURE_DIRS if a.reals == "all" else NATURE_DIRS[:1]):
        fs = files(raw / d)
        st = {"found": len(fs), "duplicate": 0, "unreadable": 0, "skipped_small": 0, "kept": 0}
        for p in fs:
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen:
                st["duplicate"] += 1; continue
            seen.add(h)
            ok = crop_ok(p)
            if ok is None:
                st["unreadable"] += 1; continue
            if not ok:
                st["skipped_small"] += 1; continue
            reals.append(p); st["kept"] += 1
        stats[f"real:{d}"] = st
    fakes = {}
    for g, d in FAKE_DIRS.items():
        fs = files(raw / d)
        ok = [crop_ok(p) for p in fs]
        keep = [p for p, o in zip(fs, ok) if o]
        stats[f"fake:{d}"] = {"found": len(fs), "unreadable": sum(o is None for o in ok),
                              "skipped_small": sum(o is False for o in ok), "kept": len(keep)}
        fakes[g] = keep
    print(json.dumps(stats, indent=2))

    reals = [reals[i] for i in rng.permutation(len(reals))]
    for g in fakes:
        fakes[g] = [fakes[g][i] for i in rng.permutation(len(fakes[g]))]
    n = len(reals)
    id_r, adm_r, wk_r = reals[: n // 2], reals[n // 2: n // 2 + n // 4], reals[n // 2 + n // 4:]
    ntr_r, ntr_f = int(0.8 * len(id_r)), int(0.8 * len(fakes["sd"]))

    def put(paths, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for p in paths:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            l, t = (w - a.size) // 2, (h - a.size) // 2
            name = p.parent.parent.name + "_" + p.stem + ".png"
            im.crop((l, t, l + a.size, t + a.size)).save(dest / name)

    def balanced(name, r, f, dest):
        k = min(len(r), len(f))
        put(r[:k], dest / "real"); put(f[:k], dest / "fake")
        counts[name] = {"real": k, "fake": k, "unused_real": len(r) - k, "unused_fake": len(f) - k}

    counts = {}
    balanced("g281/train", id_r[:ntr_r], fakes["sd"][:ntr_f], out / "g281/train")
    balanced("g281/test", id_r[ntr_r:], fakes["sd"][ntr_f:], out / "g281/test")
    balanced("ood/g281_adm", adm_r, fakes["adm"], out / "ood/g281_adm")
    balanced("ood/g281_wukong", wk_r, fakes["wukong"], out / "ood/g281_wukong")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
