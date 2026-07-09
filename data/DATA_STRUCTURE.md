# Data structure

Raw medical data is never committed. This describes the layout the toolkit
expects so you can point it at your own copies of the public datasets.

## Layout

```
data/
├── raw/                     # as downloaded
│   ├── vindr/
│   ├── INBreast/
│   ├── CMMD/
│   ├── CBIS_DDSM/
│   └── mosmed/
└── processed/               # produced by run_filtering.py + run_split.py
    ├── vindr/{train,valid,test}.csv
    ├── INBreast/...
    ├── CMMD/...
    ├── CBIS_DDSM/...
    └── mosmed/test.csv       # external test only
```

## Annotation CSV

One row per image:

| column           | required | description                                      |
|------------------|----------|--------------------------------------------------|
| `image`          | yes      | DICOM path relative to that source's `root`      |
| `label`          | yes      | label mapped to `{0,1}` via the config class map |
| `dataset`        | rec.     | source name, used to stratify splits             |
| `birads`         | opt.     | BI-RADS category (metadata filtering)            |
| `density`        | opt.     | ACR breast-density category                      |
| `mean_intensity` | opt.     | precomputed; skips re-reading the DICOM          |

## Label mapping

Binary target: absence (0) vs presence (1) of a malignant lesion.

| dataset   | source labels                                            | → class |
|-----------|----------------------------------------------------------|---------|
| VinDr     | BI-RADS 1,2 → 0 · 5 → 1 · (0,3,4,6 dropped)              | 0 / 1   |
| INBreast  | BI-RADS 1,2 → 0 · 5,6 → 1 · (3,4a-c dropped)            | 0 / 1   |
| CMMD      | B → 0 · M → 1                                            | 0 / 1   |
| CBIS-DDSM | NO_OBJECT / BENIGN(_WITHOUT_CALLBACK) → 0 · MALIGNANT → 1| 0 / 1   |
| MosMed    | benign → 0 · malignant → 1 (external test)              | 0 / 1   |

## Preparing one source

```bash
python scripts/download_datasets.py --info

python scripts/run_filtering.py \
    --ann data/raw/vindr/all.csv --root data/raw/vindr \
    --out data/processed/vindr/filtered.csv

python scripts/run_split.py \
    --ann data/processed/vindr/filtered.csv \
    --out-dir data/processed/vindr --seed 42
```
