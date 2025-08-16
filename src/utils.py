"""
Utilities: seeding, FLOPs/Params profiling.
"""
import random, numpy as np, torch, torch.nn as nn
import thop

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_model_metrics(model, input_size=(32,32), device=None):
    if device is None:
        device = get_device()
    model = model.to(device)
    in_ch = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            in_ch = m.in_channels
            break
    if in_ch is None:
        in_ch = 3
    inputs = torch.randn(1, in_ch, *input_size).to(device)
    flops, params = thop.profile(model, inputs=(inputs,), verbose=False)
    return flops, params
