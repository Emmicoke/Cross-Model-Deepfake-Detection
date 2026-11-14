#!/usr/bin/env python
"""ROC curves from results/scores/*.npz and a markdown experiment table from results/results.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def main(res="results", exp_ids=None, sets=("id",)):
    res = Path(res); (res / "figures").mkdir(exist_ok=True)
    df = pd.read_csv(res / "results.csv")
    cols = [c for c in ["exp_id", "model", "seed", "augment", "acc_id", "f1_id", "auc_id", "auc_ood_mean",
                        "ood_gap", "params", "ms_per_img", "train_time_s"] if c in df.columns]
    (res / "experiment_table.md").write_text(df[cols].to_markdown(index=False) if hasattr(df, "to_markdown") else df[cols].to_string(index=False))
    for s in sets:
        plt.figure(figsize=(5, 4.5))
        for f in sorted((res / "scores").glob(f"*__{s}.npz")):
            if exp_ids and f.name.split("__")[0] not in exp_ids:
                continue
            d = np.load(f); fpr, tpr, _ = roc_curve(d["y"], d["score"])
            plt.plot(fpr, tpr, label=f.name.split("__")[0])
        plt.plot([0, 1], [0, 1], "k:", lw=.8); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({s})")
        plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(res / "figures" / f"roc_{s}.png", dpi=150); plt.close()


if __name__ == "__main__":
    main(*sys.argv[1:2])
