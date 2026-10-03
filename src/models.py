import copy

import numpy as np
import torch
import torch.nn as nn


class PrepayNet(nn.Module):
    """Feed-forward net following the paper: in -> 128 -> 256 -> 1, BatchNorm + ReLU,
    sigmoid output (applied inside BCEWithLogitsLoss / predict_nn)."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_nn(Xtr, ytr, Xva, yva, epochs=30, bs=1024, lr=0.1, gamma=0.91,
             pos_weight=10.0, seed=42, device=None):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PrepayNet(Xtr.shape[1]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva).to(device); yv = torch.tensor(yva, dtype=torch.float32).to(device)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        print(f"  epoch {ep+1:02d} train_loss {tot/len(Xt):.4f} val_loss {vl:.4f}")
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu()


def predict_nn(model, X, bs=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.tensor(X[i:i + bs]))).numpy())
    return np.concatenate(out)
