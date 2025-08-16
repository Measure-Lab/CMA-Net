# CMA-Net: Conditioned Medical Attention Network for AI-Assisted Clinical Diagnosis of Fine-Grained Histopathological Lesions

[![Paper](https://img.shields.io/badge/Paper-Medical%20Image%20Analysis-blue)](link_to_pdf)  
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](./LICENSE)  

This repository provides the **official PyTorch implementation** of **CMA-Net**, introduced in our paper:  

> **CMA-Net: Conditioned Medical Attention Network for AI-Assisted Clinical Diagnosis of Fine-Grained Histopathological Lesions**  
> Submitted to *Medical Image Analysis*, 2025.  

---

## 🔬 Motivation
Fine-grained histopathological image classification remains challenging due to **subtle inter-class differences** and **high intra-class variability**. Existing CNN- or Transformer-based approaches often fail to capture discriminative lesion cues.  

**CMA-Net** introduces a **Conditioned Medical Attention (CMA) module**, which:  
- Learns **condition-guided feature modulation** to highlight discriminative lesion regions.  
- Integrates **multi-scale local-global dependencies** through hybrid convolution-attention blocks.  
- Achieves state-of-the-art results on benchmark medical datasets (e.g., PathMNIST).  

---

## 🚀 Quick Start

### 1. Environment
```bash
# create a new environment
conda create -n cmanet python=3.10 -y
conda activate cmanet

# install dependencies
pip install -r requirements.txt

# Train CMA-Net on PathMNIST (RGB, 32×32)
python scripts/train.py --config configs/pathmnist.yaml --rgb

python scripts/eval.py --ckpt results/cmanet_pathmnist_best.pth --rgb

CMA-Net/
│── configs/         # YAML configs for datasets/experiments
│── scripts/         # Training, evaluation, profiling scripts
│── models/          # CMA-Net implementation
│── data/            # Auto-downloaded medmnist datasets
│── results/         # Checkpoints & logs
│── README.md        # This file

