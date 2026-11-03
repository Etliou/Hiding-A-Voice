# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 03_model_A.py: Encoder A (SpeakerNet): 4 conv blocks, statistics pooling, 192-number fingerprint.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import torch, torch.nn as nn, torch.nn.functional as F

class SpeakerNet(nn.Module):
    def __init__(self, n_classes, emb_dim=192):
        super().__init__()
        def blk(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        # stride (freq, time): shrink frequency, keep time resolution
        self.conv = nn.Sequential(
            blk(1,   32, (2,1)),     # 80 -> 40 mel rows
            blk(32,  64, (2,1)),     # 40 -> 20
            blk(64, 128, (2,1)),     # 20 -> 10
            blk(128,128, (2,1)),     # 10 -> 5
        )
        self.proj = nn.Linear(128*5*2, emb_dim)   # *5 freq rows, *2 for mean+std
        self.bn   = nn.BatchNorm1d(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):                 # x: (B, 80, T)
        x = x.unsqueeze(1)                # (B, 1, 80, T)
        h = self.conv(x)                  # (B, 128, 5, T)
        B, C, Fq, T = h.shape
        h = h.reshape(B, C*Fq, T)         # (B, 640, T)  frequency folded in
        mu  = h.mean(dim=2)
        sd  = h.var(dim=2, unbiased=False).clamp_min(1e-8).sqrt()
        pooled = torch.cat([mu, sd], dim=1)        # (B, 1280)
        e = self.bn(self.proj(pooled))             # (B, 192) embedding
        return e, self.head(e)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SpeakerNet(n_classes=len(spk2idx)).to(dev)
print("params (M):", round(sum(p.numel() for p in model.parameters())/1e6, 2))

e, logits = model(xb.to(dev))
print("embedding:", tuple(e.shape), "| logits:", tuple(logits.shape))
