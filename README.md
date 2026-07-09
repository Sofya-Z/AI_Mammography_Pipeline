# mammokit

[![CI](https://github.com/ziminasofya-glitch/mammography-ai-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ziminasofya-glitch/mammography-ai-pipeline/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A toolkit for building mammography classification models from public FFDM
datasets. It covers the whole path from raw DICOMs to a trained classifier:
reading and normalizing images, filtering out bad samples, applying augmentation
presets, splitting the data, and training a CNN or CLIP-style model — all driven
by plain YAML configs.

> Datasets are not bundled. Each one is public but has its own data-use terms;
> see [`data/DATA_STRUCTURE.md`](data/DATA_STRUCTURE.md).

## What's inside

| Step | Where | CLI |
|------|-------|-----|
| Fetch public datasets | `mammokit/data/download.py` | `scripts/download_datasets.py` |
| Read & normalize DICOMs | `mammokit/data/dicom_reader.py` | — |
| Filter by metadata + intensity | `mammokit/data/filtering.py` | `scripts/run_filtering.py` |
| Augmentation presets | `mammokit/data/augmentations.py` | via config |
| Stratified split | `mammokit/data/splits.py` | `scripts/run_split.py` |
| Train / evaluate | `mammokit/train.py`, `mammokit/models/` | `scripts/train.py` |

## Layout

```
mammokit/
├── configs/
│   ├── default.yaml            # main config
│   ├── datasets/               # per-dataset class maps
│   ├── models/                 # efficientnet, mammo_clip, medclip, ...
│   └── transforms/             # light / standard / full presets + params
├── mammokit/
│   ├── data/                   # download, dicom_reader, filtering,
│   │                           #   augmentations, splits, dataset
│   ├── models/                 # efficientnet, clip_models, factory
│   ├── utils/                  # config, losses, metrics
│   └── train.py
├── scripts/                    # CLI entry points
├── tests/                      # synthetic-data smoke test
└── data/DATA_STRUCTURE.md
```

## Install

```bash
git clone https://github.com/ziminasofya-glitch/mammography-ai-pipeline.git
cd mammography-ai-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# 0. Check the pipeline runs on synthetic DICOMs (no real data needed)
python tests/test_smoke.py

# 1. See how to obtain each dataset
python scripts/download_datasets.py --info

# 2. Filter: metadata anomalies + darkest/brightest 5% by intensity
python scripts/run_filtering.py \
    --ann data/raw/vindr/all.csv --root data/raw/vindr \
    --out data/processed/vindr/filtered.csv --low-pct 5 --high-pct 5

# 3. Stratified split
python scripts/run_split.py \
    --ann data/processed/vindr/filtered.csv \
    --out-dir data/processed/vindr --seed 42

# 4. Train (EfficientNet-B3 + standard augmentation preset)
python scripts/train.py --config configs/default.yaml
```

Override any config value on the command line:

```bash
python scripts/train.py --config configs/default.yaml \
    model.name=mammo_clip transforms.preset=light train.num_epochs=50
```

## Notes on the steps

**Reading.** DICOMs are decoded, optionally passed through the modality/VOI LUTs,
forced to MONOCHROME2 (MONOCHROME1 images are inverted), scaled to `[0, 1]` and
returned as 3-channel arrays. Z-score normalization uses statistics from the
training split.

**Filtering.** Two independent filters: metadata (implausible density,
out-of-range BI-RADS, positive labels paired with a benign BI-RADS) and pixel
intensity (drop the darkest/brightest tails by mean intensity).

**Augmentation presets.** `light`, `standard`, `full`, or `none`, selected with
`transforms.preset`. Per-transform parameters live in
`configs/transforms/augmentations/`.

**Models.** `EfficientNetClassifier` (ImageNet-pretrained) plus CLIP-style
classifiers (`mammo_clip`, `medclip`, `biomedclip`, `openai_clip`) that combine a
visual encoder with a small MLP head. CLIP encoder weights are optional — a stub
encoder keeps everything runnable without them.

## Branches

`main` (stable), `develop` (integration), `feature/*` (per-step work). Run
`bash scripts/init_branches.sh` after `git init` to create them.

## License

MIT — see [LICENSE](LICENSE).
