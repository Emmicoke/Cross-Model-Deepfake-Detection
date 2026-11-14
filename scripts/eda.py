#!/usr/bin/env python
"""EDA: class counts, format check, sample grid, mean spectrum + radial profile (real vs fake)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fftdet import data, features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with real/ and fake/ (e.g. data/cifake/train)")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--max-per-class", type=int, default=5000)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    name = a.name or Path(a.root).parent.name + "_" + Path(a.root).name
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    paths_all, y_all = data.list_images(a.root)
    counts = {"real": int((y_all == 0).sum()), "fake": int((y_all == 1).sum())}
    fmts = data.image_formats(a.root)
    X, y, _ = data.load_dataset(a.root, a.size, a.max_per_class)
    stats = {"name": name, "counts": counts, "formats": fmts, "size": a.size,
             "pixel_mean": {c: float(X[y == i].mean()) for i, c in enumerate(("real", "fake"))},
             "pixel_var": {c: float(X[y == i].var()) for i, c in enumerate(("real", "fake"))}}
    S = F.log_spectrum(X)
    prof = F.azimuthal_average(S)
    mean_spec = {c: S[y == i].mean(0) for i, c in enumerate(("real", "fake"))}
    mean_prof = {c: prof[y == i].mean(0) for i, c in enumerate(("real", "fake"))}
    stats["radial_profile_l2_gap"] = float(np.linalg.norm(mean_prof["fake"] - mean_prof["real"]))
    (out / f"{name}_eda.json").write_text(json.dumps(stats, indent=2))

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for i, c in enumerate(("real", "fake")):
        im = ax[i].imshow(mean_spec[c], cmap="magma"); ax[i].set_title(f"Mean log-spectrum: {c}"); ax[i].axis("off")
    d = mean_spec["fake"] - mean_spec["real"]; v = np.abs(d).max()
    ax[2].imshow(d, cmap="coolwarm", vmin=-v, vmax=v); ax[2].set_title("fake - real"); ax[2].axis("off")
    for c in ("real", "fake"):
        ax[3].plot(mean_prof[c], label=c)
    ax[3].set_xlabel("radial frequency bin"); ax[3].set_ylabel("mean log-magnitude"); ax[3].legend(); ax[3].set_title("Azimuthal average")
    plt.tight_layout(); plt.savefig(out / f"{name}_spectra.png", dpi=150); plt.close()

    k = 8
    fig, ax = plt.subplots(2, k, figsize=(2 * k, 4.2))
    for r, c in enumerate((0, 1)):
        for j, idx in enumerate(np.where(y == c)[0][:k]):
            ax[r, j].imshow(X[idx], cmap="gray", vmin=0, vmax=1); ax[r, j].axis("off")
        ax[r, 0].set_title("real" if c == 0 else "fake", loc="left")
    plt.tight_layout(); plt.savefig(out / f"{name}_samples.png", dpi=150); plt.close()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
