# CMA-Net: Conditioned Morphology-Aware Network (PathMNIST)

> Official PyTorch implementation (starter) for the paper code artifact.

This repo provides a **clean, reproducible** training script and a modular CMA-Net implementation tested on **PathMNIST (32×32)**.

## Quick Start

```bash
# 1) (Recommended) create a fresh virtual env (conda or venv)
# conda create -n cmanet python=3.10 -y && conda activate cmanet

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train on PathMNIST (RGB)
python scripts/train.py --rgb --epochs 200 --batch 128

# 4) Evaluate a saved checkpoint
python scripts/eval.py --ckpt results/cmanet_pathmnist_best.pth --rgb
```

## Expected Results (PathMNIST @ 32×32)

We provide a simple baseline configuration:
- Optimizer: AdamW (lr=3e-4, wd=1e-4)
- Scheduler: CosineAnnealingWarmRestarts (T0=10, Tmult=2)
- Loss: CrossEntropyLoss (label smoothing=0.1)
- AMP: enabled by default on CUDA

Fill your **exact** metrics here after running:
| Model  | Top-1 | Top-5 | AUC (macro) | FLOPs | Params |
|--------|------:|------:|------------:|------:|-------:|
| CMA-Net (ours) | xx.xx | 99.9x | 0.9xx | 0.xxG | xx.xM |

## Reproducibility
- Fixed seed and deterministic cuDNN (benchmark disabled).
- Config file in `configs/pathmnist.yaml` captures key hyperparameters.
- FLOPs & Params via `thop` in `scripts/profile.py`.

## Data
`medmnist` will auto-download PathMNIST to `./data` at first run.

## Citation
Add your preferred citation (or edit `CITATION.cff`).

## License
This starter uses **AGPL-3.0** by default *before publication* to discourage code plagiarism. 
You can switch to a more permissive license (e.g., MIT) **after acceptance**.
See `LICENSE` and `PROVENANCE.md` for more details.
