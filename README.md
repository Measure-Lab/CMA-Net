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

## Results

CMA-Net achieves state-of-the-art performance across multiple medical imaging benchmarks:

### Benchmark Comparison (Top Section)

| Model               | PathMNIST Acc | Acc Impro | BreastMNIST Acc | Acc Impro | DermaMNIST Acc | Acc Impro | OCTMNIST Acc | Acc Impro |
|---------------------|--------------:|----------:|----------------:|----------:|----------------:|----------:|--------------:|----------:|
| ResNet-18           | 90.7%         | -0.9%     | 86.3%           | +4.1%     | 73.5%           | +5.3%     | 74.3%         | +7.4%     |
| ResNet-18           | 90.9%         | -1.1%     | 83.3%           | +7.1%     | 73.4%           | +5.2%     | 73.6%         | +5.4%     |
| ResNet-50           | 91.1%         | -1.3%     | 84.0%           | +6.4%     | 73.9%           | +4.7%     | 73.1%         | +6.1%     |
| ResNet-50           | 89.2%         | +0.6%     | 84.2%           | +6.2%     | 73.1%           | +5.7%     | 67.7%         | +1.4%     |
| auto-sklearn        | 71.6%         | +18.2%    | 80.6%           | +9.8%     | 59.3%           | +19.3%    | 60.1%         | +21.6%    |
| AutoKeras           | 83.4%         | +6.4%     | 83.1%           | +7.3%     | 69.3%           | +9.7%     | 63.6%         | +18.1%    |
| Google AutoML Vision| 72.8%         | +17.0%    | 66.9%           | +23.5%    | 58.2%           | +20.8%    | 61.2%         | +20.5%    |
| MedViT-V1-T         | **93.8%**     | -4.0%     | 89.7%           | +1.0%     | 73.7%           | +10.2%    | 72.7%         | +9.0%     |
| ConvNeXt-T          | 93.0%         | -3.2%     | 73.1%           | +17.3%    | 68.9%           | +15.6%    | 72.3%         | +9.4%     |
| SwinTrans-T         | 90.5%         | -0.7%     | 68.4%           | +16.8%    | 72.0%           | +6.1%     | 67.7%         | +14.0%    |
| **CMA-Net (Ours)**  | **93.8%**     | —         | **90.4%**       | —         | **78.8%**       | —         | **81.7%**     | —         |

---

### Benchmark Comparison (Bottom Section)

| Model               | PneumoniaMNIST Acc | Acc Impro | RetinaMNIST Acc | Acc Impro | BloodMNIST Acc | Acc Impro | TissueMNIST Acc | Acc Impro |
|---------------------|-------------------:|----------:|----------------:|----------:|----------------:|----------:|----------------:|----------:|
| ResNet-18           | 85.4%              | +8.4%     | 52.4%           | +7.0%     | 95.8%           | +1.8%     | 67.6%           | +4.0%     |
| ResNet-18           | 86.4%              | +7.4%     | 52.8%           | +6.6%     | 96.3%           | +1.3%     | 68.0%           | +3.5%     |
| ResNet-50           | 85.4%              | +8.6%     | 52.8%           | +6.6%     | 96.3%           | +1.3%     | 68.0%           | +3.5%     |
| ResNet-50           | 84.8%              | +5.4%     | 52.6%           | +6.8%     | 96.1%           | +1.5%     | 68.2%           | +3.3%     |
| auto-sklearn        | 85.5%              | +8.3%     | 51.5%           | +8.0%     | 96.0%           | +1.6%     | 52.3%           | +18.4%    |
| AutoKeras           | 87.8%              | +6.0%     | 50.3%           | +9.2%     | 96.8%           | +0.8%     | 63.7%           | +7.1%     |
| Google AutoML Vision| 94.6%              | -0.8%     | 53.1%           | +6.4%     | 96.5%           | +1.1%     | 63.6%           | +7.2%     |
| MedViT-V1-T         | **94.9%**          | -1.1%     | 53.1%           | +6.4%     | 96.6%           | +1.0%     | 63.6%           | +7.2%     |
| ConvNeXt-T          | 79.3%              | +16.6%    | 49.6%           | +10.0%    | 96.0%           | +1.6%     | 67.2%           | +4.6%     |
| SwinTrans-T         | 84.0%              | +9.8%     | 72.4%           | -15.6%    | 86.5%           | +11.6%    | 67.0%           | +4.8%     |
| **CMA-Net (Ours)**  | **93.6%**          | —         | **56.3%**       | —         | **97.6%**       | —         | **71.6%**       | —         |


---

## Environment Setup
We recommend creating a clean Python environment:
```bash
conda create -n cmanet python=3.10 -y
conda activate cmanet
pip install -r requirements.txt
