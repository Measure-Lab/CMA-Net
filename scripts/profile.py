#!/usr/bin/env python3
import torch
from src.models.cmanet import CMANet
from src.utils import compute_model_metrics, get_device

if __name__ == "__main__":
    device = get_device()
    model = CMANet(num_classes=9, n_channels=3).to(device)
    flops, params = compute_model_metrics(model, input_size=(32,32), device=device)
    print(f"Model FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f} M")
