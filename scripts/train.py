#!/usr/bin/env python3
import argparse, os, json, torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.models.cmanet import CMANet
from src.data import get_dataloaders
from src.engine import train_one_epoch, evaluate
from src.utils import set_seed, get_device, compute_model_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.outdir, exist_ok=True)

    train_loader, test_loader, in_ch = get_dataloaders(args.data, args.batch, rgb=args.rgb)
    model = CMANet(num_classes=9, n_channels=(3 if args.rgb else 1)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    flops, params = compute_model_metrics(model, input_size=(32,32), device=device)
    print(f"Model FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f} M")

    best1=0.0; best5=0.0; best_path=os.path.join(args.outdir, "cmanet_pathmnist_best.pth")
    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        top1, top5, te_loss, auc = evaluate(model, test_loader, criterion, device, num_classes=9)
        scheduler.step()
        if top1 > best1:
            best1 = top1; best5 = top5
            torch.save({"model": model.state_dict(),
                        "best_top1": best1, "best_top5": best5},
                       best_path)
        print(f"Epoch {ep+1:03d} | Train {tr_loss:.4f} | Val {te_loss:.4f} | Top1 {top1:.2f} | Top5 {top5:.2f} | AUC {auc:.4f} | Best1 {best1:.2f}")

    # Save final metrics JSON
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"best_top1": best1, "best_top5": best5, "FLOPs(G)": round(flops/1e9,3),
                   "Params(M)": round(params/1e6,3)}, f, indent=2)

if __name__ == "__main__":
    main()
