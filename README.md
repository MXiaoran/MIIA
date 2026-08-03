# MIIA Remote-Sensing Image-Text Retrieval 

## Environment

Python 3.10 and an NVIDIA GPU with at least 16 GB of VRAM are recommended:

```powershell
conda create -n miia python=3.10 -y
conda activate miia
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

## Data Preparation

```powershell
python -m miia.data prepare --datasets rsicd rsitmd ucm
```

The complete datasets should contain 10,921 RSICD images, 4,743 RSITMD images,
and 2,100 UCM-Captions images, with five captions per image. If automatic
download is unavailable, download and extract the datasets manually into the
following layout:

```text
data/raw/
├── rsicd/
│   ├── dataset_rsicd.json
│   └── RSICD_images/...
├── rsitmd/
│   ├── dataset_RSITMD.json
│   └── images/...
└── ucm/
    ├── dataset_ucm.json
    └── imgs/...
```

Data preparation generates the complete manifest, the leakage-free training
manifest, and the de-duplication audit log:

```text
data/processed/ret3_manifest.json
data/processed/ret3_train.json
data/processed/ret3_dedup_audit.json
```

Before formal training, verify that no cross-dataset near-duplicates exist
between the training data and any test split:

```powershell
python scripts/audit_leakage.py
```

The reported `passed` value must be `true`.

## Training

```powershell
python -m miia.train --config configs/ret3_single_gpu.yaml
```

The first run automatically downloads OpenAI CLIP ViT-B/16 and the DALL-E dVAE
encoder. To resume training from the latest checkpoint:

```powershell
python -m miia.train `
  --config configs/ret3_single_gpu.yaml `
  --resume outputs/ret3_miia_vit_b16_seed23/checkpoints/last.pth
```

`last.pth` stores the latest training state. The checkpoint with the highest
validation-set mR is saved as `best.pth`.

## Testing

Run the unit tests:

```powershell
python -m pytest
```

Evaluate the best checkpoint on all three test sets:

```powershell
python -m miia.evaluate `
  --config configs/ret3_single_gpu.yaml `
  --checkpoint outputs/ret3_miia_vit_b16_seed23/checkpoints/best.pth `
  --datasets rsicd rsitmd ucm
```

Results are written to `reports/retrieval_results.json`,
`reports/retrieval_results.csv`, and `reports/retrieval_results.md`.
