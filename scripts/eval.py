#!/usr/bin/env python3
import argparse, torch, os
from torch import nn
from src.models.cmanet import CMANet
from src.data import get_dataloaders
from src.engine import evaluate
from src.utils import set_seed, get_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    _, test_loader, in_ch = get_dataloaders(args.data, args.batch, rgb=args.rgb)
    model = CMANet(num_classes=9, n_channels=(3 if args.rgb else 1)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    top1, top5, te_loss, auc = evaluate(model, test_loader, criterion, device, num_classes=9)
    print(f"Eval | Top1 {top1:.2f} | Top5 {top5:.2f} | AUC {auc:.4f} | Loss {te_loss:.4f}")

if __name__ == "__main__":
    main()
