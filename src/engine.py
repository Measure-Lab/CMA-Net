"""
Training and evaluation loops for CMA-Net.
"""
import torch, numpy as np, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train(); epoch_loss=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device).squeeze().long()
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total=0; correct1=0; correct5=0; loss_sum=0.0
    y_all=[]; p_all=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device).squeeze().long()
        out = model(x)
        loss_sum += criterion(out,y).item()
        _, pred5 = out.topk(5,1,True,True)
        correct1 += (pred5[:,0]==y).sum().item()
        correct5 += (pred5==y.view(-1,1)).sum().item()
        total += y.size(0)
        y_all.append(y.cpu().numpy())
        p_all.append(F.softmax(out,dim=1).cpu().numpy())
    top1 = 100.0*correct1/total; top5 = 100.0*correct5/total
    y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
    try:
        auc = roc_auc_score(y_all, p_all, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return top1, top5, loss_sum/len(loader), auc
