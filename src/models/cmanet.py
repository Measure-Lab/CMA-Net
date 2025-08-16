"""
CMA-Net model definition.
Copyright (c) 2025
License: AGPL-3.0 (pre-publication)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ALME(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        z = self.dwconv(x)
        z = self.bn(z)
        z = z * self.se(z)
        z = self.proj(z)
        return z

class ConditionedSelfAttention(nn.Module):
    def __init__(self, dim, nhead=4, mlp_ratio=4, dropout=0.1, use_conditioned=True, alpha=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * mlp_ratio, dropout=dropout
        )
        self.use_conditioned = use_conditioned
        self.alpha = alpha
        self.gate_q = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.gate_k = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        self.gate_v = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=True)
        nn.init.zeros_(self.gate_q.weight); nn.init.zeros_(self.gate_q.bias)
        nn.init.zeros_(self.gate_k.weight); nn.init.zeros_(self.gate_k.bias)
        nn.init.zeros_(self.gate_v.weight); nn.init.zeros_(self.gate_v.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        L = h * w
        seq = x.view(b, c, L).permute(2, 0, 1)  # [L, B, C]
        if not self.use_conditioned:
            out = self.encoder(seq)
            return out.permute(1, 2, 0).view(b, c, h, w)

        pooled = x.mean(dim=(2, 3), keepdim=True)
        phi_q = 1.0 + self.alpha * torch.tanh(self.gate_q(pooled)).view(b, c)
        phi_k = 1.0 + self.alpha * torch.tanh(self.gate_k(pooled)).view(b, c)
        phi_v = 1.0 + self.alpha * torch.tanh(self.gate_v(pooled)).view(b, c)
        phi_q = phi_q.unsqueeze(0); phi_k = phi_k.unsqueeze(0); phi_v = phi_v.unsqueeze(0)
        enc = self.encoder
        q = seq * phi_q; k = seq * phi_k; v = seq * phi_v
        attn_out, _ = enc.self_attn(q, k, v, attn_mask=None, key_padding_mask=None, need_weights=False)
        src = seq + enc.dropout1(attn_out)
        src = enc.norm1(src)
        ffn_out = enc.linear2(enc.dropout(enc.activation(enc.linear1(src))))
        src = src + enc.dropout2(ffn_out)
        src = enc.norm2(src)
        return src.permute(1, 2, 0).view(b, c, h, w)

class CMABlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.alme = ALME(dim, reduction=reduction)
        self.attn = ConditionedSelfAttention(dim, nhead=4, mlp_ratio=4, dropout=0.1, use_conditioned=True, alpha=0.1)

    def forward(self, x):
        res = x
        x = self.alme(x)
        x = self.attn(x)
        return x + res

class CMANet(nn.Module):
    def __init__(self, num_classes=9, n_channels=3):
        super(CMANet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            CMABlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            CMABlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            CMABlock(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x

__all__ = ["ALME", "ConditionedSelfAttention", "CMABlock", "CMANet"]
