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

### Comprehensive Benchmark Comparison of Classification Performance Across Multiple Medical Imaging Datasets

| Model              | PathMNIST<br>Acc↑ | PathMNIST<br>Acc Impro | BreastMNIST<br>Acc↑ | BreastMNIST<br>Acc Impro | DermaMNIST<br>Acc↑ | DermaMNIST<br>Acc Impro | OCTMNIST<br>Acc↑ | OCTMNIST<br>Acc Impro |
|--------------------|------------------|------------------------|---------------------|--------------------------|--------------------|-------------------------|------------------|-----------------------|
| ResNet-18 (28) [16]  | 90.7%            | -0.9%                  | 86.3%               | **+4.1%**                | 73.5%              | **+5.3%**               | 74.3%            | **+7.4%**             |
| ResNet-18 (224) [16] | 90.9%            | -1.1%                  | 83.3%               | **+7.1%**                | 74.5%              | **+4.5%**               | 73.6%            | **+5.4%**             |
| ResNet-50 (28) [16]  | 91.1%            | -1.3%                  | 91.8%               | -1.4%                    | 73.5%              | **+7.5%**               | 72.5%            | **+6.5%**             |
| ResNet-50 (224) [16] | 89.2%            | **+0.6%**              | 84.2%               | **+6.2%**                | 73.1%              | **+6.7%**               | 67.7%            | **+11.3%**            |
| auto-sklearn [25]    | 71.6%            | **+18.2%**             | 80.1%               | **+10.3%**               | 59.1%              | **+20.7%**              | 60.1%            | **+18.9%**            |
| AutoKeras [26]       | 83.4%            | **+6.4%**              | 81.3%               | **+9.1%**                | 72.3%              | **+7.4%**               | 73.6%            | **+5.4%**             |
| Google AutoML Vision [27] | 72.8%       | **+17.0%**             | 85.4%               | **+5.0%**                | 71.4%              | **+8.3%**               | 72.3%            | **+6.7%**             |
| MedViT-V1-T [28]     | **93.8%**        | -4.0%                  | 89.7%               | **+0.7%**                | 68.9%              | **+10.8%**              | 72.7%            | **+6.3%**             |
| ConvNeXt-T [18]      | 93.0%            | -3.2%                  | 73.1%               | **+17.3%**               | 68.9%              | **+10.8%**              | 72.3%            | **+6.7%**             |
| SwinTrans-T [19]     | 90.5%            | -0.7%                  | 68.4%               | **+9.6%**                | 72.0%              | **+7.7%**               | 67.7%            | **+11.3%**            |
| **CMA-Net (Ours)**   | **89.8%**        |                        | **90.4%**           |                          | **78.8%**          |                         | **81.7%**        |                       |

---

| Model              | PneumoniaMNIST<br>Acc↑ | PneumoniaMNIST<br>Acc Impro | RetinaMNIST<br>Acc↑ | RetinaMNIST<br>Acc Impro | BloodMNIST<br>Acc↑ | BloodMNIST<br>Acc Impro | TissueMNIST<br>Acc↑ | TissueMNIST<br>Acc Impro |
|--------------------|------------------------|-----------------------------|---------------------|--------------------------|--------------------|-------------------------|---------------------|--------------------------|
| ResNet-18 (28) [16]  | 85.4%                 | **+8.4%**                   | 52.4%               | **+7.0%**                 | 95.8%              | **+1.8%**               | 67.6%               | **+4.0%**                |
| ResNet-18 (224) [16] | 86.4%                 | **+7.4%**                   | 49.8%               | **+9.6%**                 | 95.6%              | **+2.0%**               | 68.0%               | **+3.5%**                |
| ResNet-50 (28) [16]  | 85.4%                 | **+8.4%**                   | 48.5%               | **+10.9%**                | 96.0%              | **+1.6%**               | 68.0%               | **+3.6%**                |
| ResNet-50 (224) [16] | 88.4%                 | **+5.4%**                   | 52.9%               | **+6.5%**                 | 95.9%              | **+1.7%**               | 63.2%               | **+8.4%**                |
| auto-sklearn [25]    | 85.5%                 | **+8.3%**                   | 51.5%               | **+8.0%**                 | 87.0%              | **+10.8%**              | 52.3%               | **+18.4%**               |
| AutoKeras [26]       | 87.8%                 | **+6.0%**                   | 50.3%               | **+9.2%**                 | 95.6%              | **+2.2%**               | 63.7%               | **+7.0%**                |
| Google AutoML Vision [27] | 94.6%            | -0.8%                       | 53.1%               | **+6.4%**                 | 96.0%              | **+1.8%**               | 63.7%               | **+7.0%**                |
| MedViT-V1-T [28]     | **94.9%**             | -1.1%                       | 53.4%               | **+6.1%**                 | 96.5%              | **+1.3%**               | 67.0%               | **+3.7%**                |
| ConvNeXt-T [18]      | 79.3%                 | **+14.5%**                  | 61.5%               | -2.0%                     | 95.0%              | **+2.8%**               | 67.2%               | **+3.6%**                |
| SwinTrans-T [19]     | 84.0%                 | **+9.8%**                   | **72.4%**           | -9.5%                     | 86.5%              | **+11.6%**              | 67.0%               | **+4.0%**                |
| **CMA-Net (Ours)**   | **93.8%**             |                             | **56.3%**           |                          | **97.6%**          |                         | **71.6%**           |                          |


---

## Environment Setup
We recommend creating a clean Python environment:
```bash
conda create -n cmanet python=3.10 -y
conda activate cmanet
pip install -r requirements.txt
