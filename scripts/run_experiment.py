#!/usr/bin/env python
"""Train one model and evaluate in-distribution + OOD. Appends one row to results/results.csv.

Example:
  python scripts/run_experiment.py --model fft_resnet18 --train-root data/cifake \
      --ood data/ood/synthbuster_mj data/ood/genimage_adm --size 32 --seed 42
`--train-root` must contain train/ and test/ (each with real/ and fake/).
"""
import argparse, resource, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fftdet import data, models, metrics
from fftdet.augment import augment_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=models.ALL_KINDS)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--ood", nargs="*", default=[], help="OOD dataset folders (each with real/ fake/)")
    ap.add_argument("--ood-fake-only", nargs="*", default=[],
                    help="folders with only fake/ (no reals): reports recall at thr 0.5, excluded from OOD gap")
    ap.add_argument("--size", type=int, default=32, help="image side; 32 = native CIFAKE")
    ap.add_argument("--max-per-class", type=int, default=None, help="cap images per class (train)")
    ap.add_argument("--max-test-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", action="store_true", help="JPEG/blur/resize augmentation on train images")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    root = Path(a.train_root)
    exp_id = a.exp_id or f"{a.model}_s{a.seed}{'_aug' if a.augment else ''}"
    Xtr, ytr, _ = data.load_dataset(root / "train", a.size, a.max_per_class, a.seed)
    # stratified 80/20 train/val split
    rng = np.random.default_rng(a.seed)
    va = np.zeros(len(ytr), bool)
    for c in (0, 1):
        idx = np.where(ytr == c)[0]
        va[rng.choice(idx, int(0.2 * len(idx)), replace=False)] = True
    Xva, yva, Xtr, ytr = Xtr[va], ytr[va], Xtr[~va], ytr[~va]
    if a.augment:
        Xtr = augment_batch(Xtr, a.seed)
    print(f"[{exp_id}] train={len(ytr)} val={len(yva)} fake_frac={ytr.mean():.3f}")

    model = models.build(a.model, a.size, a.seed, epochs=a.epochs, bs=a.bs, lr=a.lr)
    t0 = time.time()
    model.fit(Xtr, ytr, Xva, yva)
    train_s = time.time() - t0

    sets = {"id": str(root / "test")}
    sets.update({Path(p).name: p for p in a.ood})
    row = dict(exp_id=exp_id, model=a.model, seed=a.seed, size=a.size, augment=a.augment,
               n_train=len(ytr), epochs=a.epochs if a.model in models.TORCH_KINDS else "", notes=a.notes)
    Path("results/scores").mkdir(parents=True, exist_ok=True)
    aucs = {}
    for name, path in sets.items():
        X, y, _ = data.load_dataset(path, a.size, a.max_test_per_class, a.seed)
        s = model.score(X)
        m = metrics.evaluate(y, s)
        aucs[name] = m["auc"]
        for k, v in m.items():
            row[f"{k}_{name}"] = round(float(v), 4)
        row[f"n_{name}"] = len(y)
        np.savez_compressed(f"results/scores/{exp_id}__{name}.npz", y=y, score=s)
        print(f"  {name:>20s}: acc={m['acc']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")
        if name == "id":
            Xt = X[:256]
    for path in a.ood_fake_only:
        name = Path(path).name
        X, y, _ = data.load_dataset(path, a.size, a.max_test_per_class, a.seed, fake_only=True)
        s = model.score(X)
        row[f"recall_{name}"] = round(float((s >= 0.5).mean()), 4)
        row[f"meanscore_{name}"] = round(float(s.mean()), 4)
        row[f"n_{name}"] = len(y)
        np.savez_compressed(f"results/scores/{exp_id}__{name}.npz", y=y, score=s)
        print(f"  {name:>20s}: recall(fake)={row[f'recall_{name}']:.4f} (fake-only set, no AUC)")
    ood = [v for k, v in aucs.items() if k != "id"]
    row["auc_ood_mean"] = round(float(np.mean(ood)), 4) if ood else ""
    row["ood_gap"] = round(float(metrics.ood_gap(aucs["id"], np.mean(ood))), 4) if ood else ""
    row["params"] = model.n_params()
    row["train_time_s"] = round(train_s, 1)
    row["ms_per_img"] = round(metrics.timed(lambda: model.score(Xt), len(Xt)), 4)
    row["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    try:
        import torch
        row["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else ""
    except Exception:
        row["peak_gpu_mb"] = ""

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if out.exists():
        old = pd.read_csv(out)
        old = old[old.exp_id != exp_id]          # re-running an experiment replaces its row
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"-> {out}  (ood_gap={row['ood_gap']}, {row['ms_per_img']} ms/img, {train_s:.0f}s train)")


if __name__ == "__main__":
    main()
