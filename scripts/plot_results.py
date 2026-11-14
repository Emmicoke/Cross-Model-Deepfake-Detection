#!/usr/bin/env python
"""Tables + ROC curves for one results directory (holding results.csv and scores/*.npz).

  python scripts/plot_results.py results/tinygen            # -> results/tinygen/{experiment_table,summary_table}.md
  python scripts/plot_results.py results/g281 --seed 42     #    and results/tinygen/figures/roc_<set>.png

experiment_table.md: one row per run (in-dist acc/F1/AUC, AUC per OOD set, mean OOD AUC, OOD gap,
                     fake recall per fake-only set, params, ms/img, train time).
summary_table.md:    mean +- std over seeds for each (model, augment) pair.
ROC curves: in-distribution test and every OOD set with both classes, one curve per model (the given seed).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def table_cols(df):
    ood_auc = sorted(c for c in df.columns if c.startswith("auc_") and c not in ("auc_id", "auc_ood_mean"))
    rec = sorted(c for c in df.columns if c.startswith("recall_"))
    base = ["exp_id", "model", "seed", "augment", "n_train", "acc_id", "f1_id", "auc_id"]
    tail = ["auc_ood_mean", "ood_gap"] + rec + ["params", "ms_per_img", "train_time_s"]
    return [c for c in base + ood_auc + tail if c in df.columns], ood_auc, rec


def to_md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res", nargs="?", default="results", help="dir with results.csv and scores/")
    ap.add_argument("--seed", type=int, default=None, help="seed whose scores are plotted (default: smallest)")
    a = ap.parse_args()
    res = Path(a.res); fig_dir = res / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / "results.csv").sort_values(["model", "augment", "seed"])
    cols, ood_auc, rec = table_cols(df)
    (res / "experiment_table.md").write_text(to_md(df[cols]) + "\n", encoding="utf-8")

    num = [c for c in cols if c not in ("exp_id", "model", "seed", "augment") and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(["model", "augment"], sort=False)
    mean, std, n = g[num].mean(), g[num].std(), g.size()
    summ = mean.copy().astype(object)
    for c in num:
        for idx in mean.index:
            m, s = mean.loc[idx, c], std.loc[idx, c]
            if c in ("params", "n_train"):
                summ.loc[idx, c] = f"{m:.0f}"
            elif c == "train_time_s":
                summ.loc[idx, c] = f"{m:.0f}"
            elif n[idx] > 1 and not np.isnan(s):
                summ.loc[idx, c] = f"{m:.3f} ± {s:.3f}"
            else:
                summ.loc[idx, c] = f"{m:.3f}"
    summ.insert(0, "n_seeds", n)
    (res / "summary_table.md").write_text(to_md(summ.reset_index()) + "\n", encoding="utf-8")

    seed = a.seed if a.seed is not None else int(df.seed.min())
    runs = df[df.seed == seed].exp_id.tolist()
    sets = ["id"] + [c[len("auc_"):] for c in ood_auc]
    for s in sets:
        plt.figure(figsize=(5, 4.5))
        for eid in runs:
            f = res / "scores" / f"{eid}__{s}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            if len(np.unique(d["y"])) < 2:
                continue
            fpr, tpr, _ = roc_curve(d["y"], d["score"])
            plt.plot(fpr, tpr, lw=1, label=eid)
        plt.plot([0, 1], [0, 1], "k:", lw=.8); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC ({s}, seed {seed})"); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(fig_dir / f"roc_{s}.png", dpi=150); plt.close()

    # OOD AUC per set, bar chart (mean over seeds) for a quick visual of cross-model generalisation
    if ood_auc:
        m = df.groupby(["model", "augment"], sort=False)[["auc_id"] + ood_auc].mean()
        m.index = [f"{mo}{'+aug' if au else ''}" for mo, au in m.index]
        ax = m.plot.bar(figsize=(max(8, 1.2 * len(m)), 4.5), width=.85)
        ax.axhline(.5, color="k", lw=.6, ls=":"); ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6, ncol=2); plt.xticks(rotation=35, ha="right"); plt.tight_layout()
        plt.savefig(fig_dir / "auc_per_set.png", dpi=150); plt.close()
    print(f"wrote {res/'experiment_table.md'}, {res/'summary_table.md'}, ROC for {len(sets)} sets (seed {seed})")


if __name__ == "__main__":
    main()
