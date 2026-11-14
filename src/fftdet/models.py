"""Models: sklearn baselines + torch MLP / ResNet-18. All expose fit(Xtr,ytr,Xva,yva) and score(X)->P(fake)."""
from __future__ import annotations
import copy
import numpy as np
from . import features as F


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _block_downsample(X, out=32):
    n, h, w = X.shape
    if h == out:
        return X
    k = h // out
    return X[:, : out * k, : out * k].reshape(n, out, k, out, k).mean(axis=(2, 4))


# ----------------------------------------------------------------------------- sklearn
class SkModel:
    """feature_fn: raw images [N,H,W] -> feature matrix. Fitted on training statistics only."""

    def __init__(self, kind: str, seed: int = 0, svm_max_train: int = 10000):
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        self.kind, self.seed, self.svm_max_train = kind, seed, svm_max_train
        self.decision = False
        if kind == "naive_majority":
            self.feat, self.clf = lambda X: np.zeros((len(X), 1)), DummyClassifier(strategy="most_frequent")
        elif kind == "naive_stats":      # spatial mean/variance, depth-1 threshold rule
            self.feat, self.clf = F.spatial_stats, DecisionTreeClassifier(max_depth=1, random_state=seed)
        elif kind == "pixel_logreg":
            self.feat = lambda X: _block_downsample(X).reshape(len(X), -1)
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
        elif kind == "lbp_logreg":
            self.feat = F.lbp_hist
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
        elif kind == "lbp_svm":
            self.feat = F.lbp_hist
            self.clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
            self.decision = True
        elif kind == "fft_radial_logreg":   # ablation: frequency profile + linear classifier
            self.feat = lambda X: F.azimuthal_average(F.log_spectrum(X))
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        else:
            raise ValueError(kind)

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        if self.kind == "lbp_svm" and len(Xtr) > self.svm_max_train:
            idx = np.random.default_rng(self.seed).choice(len(Xtr), self.svm_max_train, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        self.clf.fit(self.feat(Xtr), ytr)
        return self

    def score(self, X):
        f = self.feat(X)
        if self.decision:
            return _sigmoid(self.clf.decision_function(f))
        return self.clf.predict_proba(f)[:, 1]

    def n_params(self):
        return 0


# ----------------------------------------------------------------------------- torch
def make_resnet18(size: int):
    import torch.nn as nn
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    if size <= 64:                       # small-image stem: keep resolution
        m.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
    else:
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.fc = nn.Linear(512, 1)
    return m


def make_mlp(d_in: int, hidden=(256, 128), p=0.3):
    import torch.nn as nn
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


class TorchModel:
    """kind: fft_mlp (radial profile) | fft_resnet18 (2-D spectrum map) | pixel_resnet18 (spatial control)."""

    def __init__(self, kind: str, size: int, seed=0, epochs=15, lr=1e-3, bs=128, wd=1e-4,
                 patience=4, device=None):
        import torch
        self.torch, self.kind, self.size, self.seed = torch, kind, size, seed
        self.epochs, self.lr, self.bs, self.wd, self.patience = epochs, lr, bs, wd, patience
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if kind == "fft_mlp":
            self.net = make_mlp(size // 2)
        elif kind in ("fft_resnet18", "pixel_resnet18"):
            self.net = make_resnet18(size)
        else:
            raise ValueError(kind)
        self.net.to(self.dev)
        self.mu = self.sd = None

    def _prep(self, X, fit=False):
        if self.kind == "fft_mlp":
            Z = F.azimuthal_average(F.log_spectrum(X))
            if fit:
                self.mu, self.sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
            Z = (Z - self.mu) / self.sd
        elif self.kind == "fft_resnet18":
            Z = F.log_spectrum(X)
            if fit:
                self.mu, self.sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
            Z = ((Z - self.mu) / self.sd)[:, None]
        else:
            Z = ((X - 0.5) / 0.25)[:, None]
        return self.torch.tensor(Z, dtype=self.torch.float32)

    def fit(self, Xtr, ytr, Xva, yva):
        from sklearn.metrics import roc_auc_score
        torch = self.torch
        Ztr, Zva = self._prep(Xtr, fit=True), self._prep(Xva)
        ytr_t = torch.tensor(ytr, dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad = -1.0, None, 0
        self.history = []
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Ztr), generator=g)
            tot = 0.0
            for i in range(0, len(perm), self.bs):
                idx = perm[i:i + self.bs]
                xb, yb = Ztr[idx].to(self.dev), ytr_t[idx].to(self.dev)
                opt.zero_grad()
                loss = lossf(self.net(xb).squeeze(1), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            auc = roc_auc_score(yva, self._score_tensor(Zva))
            self.history.append((ep + 1, tot / len(perm), auc))
            print(f"  epoch {ep+1:02d} loss={tot/len(perm):.4f} val_auc={auc:.4f}", flush=True)
            if auc > best:
                best, best_state, bad = auc, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    print("  early stop"); break
        self.net.load_state_dict(best_state)
        self.best_val_auc = best
        return self

    def _score_tensor(self, Z):
        torch = self.torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 512):
                out.append(torch.sigmoid(self.net(Z[i:i + 512].to(self.dev)).squeeze(1)).cpu())
        return torch.cat(out).numpy()

    def score(self, X):
        return self._score_tensor(self._prep(X))

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


SK_KINDS = ["naive_majority", "naive_stats", "pixel_logreg", "lbp_logreg", "lbp_svm", "fft_radial_logreg"]
TORCH_KINDS = ["fft_mlp", "fft_resnet18", "pixel_resnet18"]
ALL_KINDS = SK_KINDS + TORCH_KINDS


def build(kind, size, seed, **kw):
    if kind in SK_KINDS:
        return SkModel(kind, seed, kw.get("svm_max_train", 10000))
    return TorchModel(kind, size, seed, epochs=kw.get("epochs", 15), lr=kw.get("lr", 1e-3),
                      bs=kw.get("bs", 128), device=kw.get("device"))
