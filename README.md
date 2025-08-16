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

| Model              | PathMNIST |        | BreastMNIST |        | DermaMNIST |        | OCTMNIST |        |
|--------------------|-----------|--------|-------------|--------|------------|--------|----------|--------|
|                    | Acc↑      | Acc Impro | Acc↑      | Acc Impro | Acc↑      | Acc Impro | Acc↑    | Acc Impro |
| ResNet-18          | 90.7%   | -0.9%     | 86.3%   | **+4.1%**   | 73.5%   | **+5.3%**   | 74.3%   | **+7.4%**   |
| ResNet-18          | 90.9%   | -1.1%     | 83.3%   | **+7.1%**   | 75.4%   | **+3.4%**   | 76.3%   | **+5.4%**   |
| ResNet-50          | 91.1%   | -1.3%     | 81.2%   | **+9.2%**   | 73.5%   | **+5.3%**   | 76.2%   | **+5.5%**   |
| ResNet-50          | 89.2%   | **+0.6%** | 84.2%   | **+6.2%**   | 73.1%   | **+5.7%**   | 77.0%   | **+3.6%**   |
| auto-sklearn       | 71.6%   | **+18.2%**| 80.0%   | **+10.1%**  | 71.9%   | **+6.9%**   | 60.1%   | **+21.6%**  |
| AutoKeras          | 83.4%   | **+6.4%** | 83.1%   | **+7.3%**   | 74.9%   | **+3.9%**   | 76.3%   | **+5.4%**   |
| Google AutoML Vision  | 72.8% | **+17.0%**| 86.1% | **+4.3%**   | 76.8%   | **+2.0%**   | 77.1%   | **+4.6%**   |
| MedViT-V1-T        | **93.8%**| -4.0%     | 89.7%   | **+0.7%**   | 76.8%   | **+2.0%**   | 76.8%   | **+4.9%**   |
| ConvNeXt-T         | 93.0%   | -3.2%     | 73.1%   | **+17.3%**  | 68.9%   | **+9.9%**   | 72.3%   | **+9.4%**   |
| SwinTrans-T        | 90.5%   | -0.7%     | 68.4%   | **+22.0%**  | 72.0%   | **+6.8%**   | 67.7%   | **+14.0%**  |
| **CMA-Net (Ours)**   | **89.8%** |          | **90.4%** |          | **78.8%** |          | **81.7%** |          |


| Model              | PneumoniaMNIST |        | RetinaMNIST |        | BloodMNIST |        | TissueMNIST |        |
|--------------------|----------------|--------|-------------|--------|------------|--------|-------------|--------|
|                    | Acc↑           | Acc Impro | Acc↑      | Acc Impro | Acc↑      | Acc Impro | Acc↑       | Acc Impro |
| ResNet-18          | 85.4%        | **+8.4%** | 52.4%     | **+3.9%**   | 95.8%   | **+1.8%**   | 67.6%   | **+4.0%**   |
| ResNet-18          | 86.4%        | **+7.4%** | 49.3%     | **+7.0%**   | 96.3%   | **+1.3%**   | 68.1%   | **+3.5%**   |
| ResNet-50          | 85.4%        | **+8.4%** | 52.8%     | **+3.5%**   | 95.8%   | **+2.0%**   | 68.0%   | **+3.6%**   |
| ResNet-50          | 88.4%        | **+5.4%** | 51.1%     | **+5.2%**   | 95.0%   | **+2.6%**   | 68.0%   | **+3.6%**   |
| auto-sklearn       | 85.5%        | **+8.3%** | 51.5%     | **+4.8%**   | 87.8%   | **+9.8%**   | 52.3%   | **+18.4%**  |
| AutoKeras          | 87.8%        | **+6.0%** | 50.3%     | **+6.0%**   | 95.6%   | **+2.2%**   | 63.7%   | **+7.0%**   |
| Google AutoML Vision | 94.6%   | -0.8%     | 53.1%     | **+3.2%**   | 96.6%   | **+1.0%**   | 63.7%   | **+7.0%**   |
| MedViT-V1-T      | **94.9%**    | -1.1%     | 53.4%     | **+2.9%**   | 96.5%   | **+1.1%**   | 67.0%   | **+4.6%**   |
| ConvNeXt-T       | 79.3%        | **+14.5%**| 65.8%     | -9.5%       | 95.0%   | **+2.8%**   | 67.2%   | **+3.6%**   |
| SwinTrans-T      | 84.0%        | **+9.8%** | **72.4%** | -16.1%      | 86.5%   | **+11.1%**  | 67.0%   | **+4.6%**   |
| **CMA-Net (Ours)**   | **93.8%**    |          | **56.3%** |          | **97.6%** |          | **71.6%** |          |

---

## Environment Setup
We recommend creating a clean Python environment:
```bash
conda create -n cmanet python=3.10 -y
conda activate cmanet
pip install -r requirements.txt
```
---
## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).  
See the [LICENSE](./LICENSE) file for details.
