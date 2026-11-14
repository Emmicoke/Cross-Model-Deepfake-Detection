# Data layout

Every dataset used by the code is a folder with a `real/` and a `fake/` sub-folder
(case-insensitive, searched recursively for .png/.jpg/.jpeg/.webp).

```
data/cifake/train/REAL  data/cifake/train/FAKE     # CIFAKE (SD v1.4 fakes, CIFAR-10 reals)
data/cifake/test/REAL   data/cifake/test/FAKE
data/ood/<generator_name>/real  data/ood/<generator_name>/fake   # e.g. synthbuster_mj, genimage_adm
```

Datasets are not committed. See the main README for download sources and licences.
