# CMA-Net: Conditioned Medical Attention Network for AI-Assisted Clinical Diagnosis of Fine-Grained Histopathological Lesions

This repository provides the official PyTorch implementation of **CMA-Net**, developed for our submission:

> **CMA-Net: Conditioned Medical Attention Network for AI-Assisted Clinical Diagnosis of Fine-Grained Histopathological Lesions**  
> Submitted to *Medical Image Analysis*, 2025.

---
![CMA-Net Framework](src/framework.png)

## Motivation
Fine-grained histopathological image classification is challenging due to subtle inter-class differences and large intra-class variability. Conventional CNNs or Transformers often struggle to capture highly discriminative lesion cues.  
CMA-Net introduces a **Conditioned Medical Attention (CMA) module**, enabling condition-guided feature modulation and multi-scale convolution–attention fusion. This results in improved robustness and accuracy on medical benchmarks such as PathMNIST.

---

## Environment Setup
We recommend creating a clean Python environment:
```bash
conda create -n cmanet python=3.10 -y
conda activate cmanet
pip install -r requirements.txt

<img width="1323" height="770" alt="image" src="https://github.com/user-attachments/assets/d5f99f4b-39f3-4cf0-879b-dec2a23fcfb7" />
