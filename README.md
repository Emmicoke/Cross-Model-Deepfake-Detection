# Cross-Model Deepfake Detection via Frequency-Domain Fourier Artifact Analysis

BIL539 Artificial Intelligence term project (2026-2027 Fall), Fahrettin Emin Korkut.
First stage of a master's thesis on cross-model deepfake detection: can a light detector trained on one
generator (e.g. Stable Diffusion) tell real from synthetic images made by *other* generators, using only
2-D FFT spectrum artefacts?

## Pipeline

```
image -> gray, centre crop -> (train only: JPEG/blur/resize augmentation)
      -> 2-D FFT, log(1+|F|), Hann window -> [2-D map | azimuthal average (1-D radial profile)]
      -> classifier -> P(fake)
```

| Model id | Role | Input |
|---|---|---|
| `naive_majority` | naive baseline (lower bound) | none |
| `naive_stats` | naive baseline, depth-1 threshold rule | pixel mean/variance |
| `pixel_logreg` | classical baseline | raw pixels (32x32) |
| `lbp_logreg`, `lbp_svm` | classical baselines | uniform LBP histogram |
| `fft_radial_logreg` | ablation: frequency profile + linear model | azimuthal average |
| `fft_mlp` | main approach, light | azimuthal average |
| `fft_resnet18` | main approach | 2-D log spectrum (1 channel) |
| `pixel_resnet18` | control: same CNN on pixels, no FFT | grayscale image |

Metrics: accuracy, F1, ROC-AUC (threshold fixed at 0.5 for acc/F1, also on OOD sets),
**OOD generalisation gap** = AUC(in-distribution test) - mean AUC(OOD sets), ms/image, parameters, peak memory.

## Setup

```bash
conda env create -f environment.yml && conda activate deepfake     # or: pip install -r requirements.txt
# GPU: install the CUDA build of PyTorch from https://pytorch.org first, then check
python -c "import torch; print(torch.cuda.is_available())"
```

## Data (not committed, check each licence before use)

* **CIFAKE** (Bird & Lotfi, 2024): 32x32, CIFAR-10 reals vs Stable Diffusion v1.4 fakes. Put it at
  `data/cifake/{train,test}/{REAL,FAKE}`.
* **OOD generators**: any folder with `real/` and `fake/` sub-folders under `data/ood/<name>/`
  (e.g. a Synthbuster or GenImage subset, or your own Flux/SD outputs). Each folder becomes one OOD test set,
  nothing from them is used for training. Images are centre-cropped and resized to `--size`.
* **Multi-generator set (recommended for the cross-model claim):** Tiny-GenImage parquet shards from HuggingFace
  (`TheKernel01/Tiny-GenImage`, CC-BY-NC-SA-4.0) -> `python scripts/prepare_tiny_genimage.py --parquet-dir <dir> --out data --train-gen SD15 --size 256`
  writes `data/tinygen/{train,test}` and `data/ood/<generator>`. Images are centre-cropped, never resized; smaller ones are skipped and counted.
* **Fake-only extra generators** (no matched reals, e.g. Flux / SD3.5): put them in `data/ood_fake_only/<name>/fake/`;
  they are reported as recall of the fake class at threshold 0.5 (no AUC, excluded from the OOD gap).
* Do not train at 32x32 (CIFAKE) and test on larger images resized down: resizing destroys the artefacts.
  Use CIFAKE for the pipeline/baselines only.
* See `data/README.md`. Check JPEG-vs-PNG format bias between real and fake: `scripts/eda.py` reports file formats.

## Run

```bash
bash run_all.sh                                  # EDA + all baselines + main models, seed 42
SEEDS="42 43 44" EPOCHS=20 bash run_all.sh       # 3 seeds
MAXPC=20000 bash run_all.sh                      # cap images per class for a quick run
python scripts/run_experiment.py --model fft_resnet18 --train-root data/cifake --ood data/ood/mygen --size 32
```

Outputs: `results/results.csv` (one row per experiment: Exp-ID, model, seed, metrics per test set, OOD gap,
params, ms/img, memory), `results/experiment_table.md`, `results/figures/` (mean spectra, radial profiles,
ROC curves), `results/scores/` (raw scores per test set).

`scripts/make_synthetic_smoke_data.py` builds a toy dataset to test the code without downloads;
its numbers are meaningless.

## M2 experiments (as run)

```bash
python scripts/prepare_281_genimage.py --out data                       # jhutter2/281_Genimage -> data/g281, data/ood/g281_{adm,wukong}
python scripts/prepare_tiny_genimage.py --parquet-dir <Tiny-GenImage>/data --out data --train-gen SD15 --size 256 --max-per-gen 2500
python scripts/prepare_fake_only.py --src <dir of fakes> --name <name>  # -> data/ood_fake_only/<name>/fake (Flux, SD3.5, Synthbuster)
EXP=e2 SEEDS="42 43 44" EPOCHS=20 bash scripts/run_m2.sh                # train g281 (SD v1.5)  -> results/g281/
EXP=e1 SEEDS="42 43 44" EPOCHS=20 bash scripts/run_m2.sh                # train Tiny-GenImage SD15 -> results/tinygen/
python scripts/plot_results.py results/g281; python scripts/plot_results.py results/tinygen
```

Each results dir holds `results.csv`, `experiment_table.md` (per run), `summary_table.md` (mean ± std over seeds),
`scores/` and `figures/`. EDA output is in `results/figures/`. Note: in both GenImage-derived sets the real images
are ImageNet JPEGs and the fakes are PNGs, so JPEG traces are a possible shortcut (see the `--augment` variant).

## Layout

```
src/fftdet/   data.py features.py augment.py models.py metrics.py
scripts/      run_experiment.py eda.py plot_results.py run_m2.sh prepare_tiny_genimage.py prepare_281_genimage.py
              prepare_fake_only.py make_synthetic_smoke_data.py
run_all.sh    environment.yml  requirements.txt  data/README.md  results/
```

## Ethics

The detector is decision *support*, not forensic proof. False positives (real image flagged synthetic) have
real costs. Datasets are used under their research licences; no personal data is collected.
