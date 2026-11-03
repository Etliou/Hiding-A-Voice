# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 14_model_C_and_ensemble.py: Encoder C (dilated convolutions, seed 777) and the ensemble protection A + C.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import random, glob, os, time, torch, torchaudio, torch.nn as nn
torch.manual_seed(777); random.seed(777)

class SpeakerNetC(nn.Module):
    """Deliberately unlike A and B: dilated convs, wider, 320-dim."""
    def __init__(s, nc, d=320):
        super().__init__()
        b = lambda ci,co,k,st,dl=1: nn.Sequential(
            nn.Conv2d(ci,co,k,st,(k//2)*dl,dilation=dl), nn.BatchNorm2d(co), nn.ReLU(True))
        s.conv = nn.Sequential(b(1,64,3,(2,1)), b(64,96,3,(2,1),2), b(96,128,3,(2,1),2),
                               b(128,160,3,(2,2)), b(160,192,3,(1,1),4))
        s.proj = nn.Sequential(nn.Linear(192*5*2,640), nn.ReLU(True), nn.Linear(640,d))
        s.bn = nn.BatchNorm1d(d); s.head = nn.Linear(d, nc)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)); B,C,F,T = h.shape; h = h.reshape(B, C*F, T)
        e = s.bn(s.proj(torch.cat([h.mean(2),
              h.var(2, unbiased=False).clamp_min(1e-8).sqrt()], 1)))
        return e, s.head(e)

class SpkDS(torch.utils.data.Dataset):
    def __init__(s, items): s.items = items
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, lab = s.items[i]
        x, _ = torchaudio.load(p); x = x.mean(0)
        if x.shape[0] < CHUNK: x = torch.nn.functional.pad(x, (0, CHUNK - x.shape[0]))
        o = random.randint(0, x.shape[0] - CHUNK)
        return x[o:o+CHUNK], lab

train_items = [(p, spk2idx[s]) for p, s in pairs if s in train_spk]
dl = torch.utils.data.DataLoader(SpkDS(train_items), batch_size=64, shuffle=True,
                                 num_workers=2, drop_last=True, persistent_workers=True)

EPOCHS, CKPT = 15, f'{R}/code/modelC_ckpt.pt'
modelC = SpeakerNetC(len(spk2idx)).to(dev)
opt = torch.optim.Adam(modelC.parameters(), 1e-3)
start = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=dev)
    modelC.load_state_dict(ck['m']); opt.load_state_dict(ck['o']); start = ck['e']

for ep in range(start, EPOCHS):
    modelC.train(); tot = n = corr = 0
    for xb, yb in dl:
        xb, yb = xb.to(dev), yb.to(dev)
        _, lg = modelC(logmel(xb))
        loss = nn.functional.cross_entropy(lg, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*yb.shape[0]; n += yb.shape[0]
        corr += (lg.argmax(1) == yb).sum().item()
    torch.save({'m': modelC.state_dict(), 'o': opt.state_dict(), 'e': ep+1}, CKPT)
    print(f'epoch {ep+1:2d}/{EPOCHS}  loss {tot/n:.3f}  acc {corr/n:.3f}')
torch.save(modelC.state_dict(), f'{R}/code/modelC_ensemble_member.pt')

# ===== ensemble (A + C) measured-channel EOT, judged by B =====
def protect_ens(wav, models, eps=0.002, K=300, eot=4, rir_pool=None):
    alpha = eps/10; wav = wav.to(dev)
    pool = [h.to(dev) for h in rir_pool]
    e0 = [emb(M, wav) for M in models]
    d = torch.zeros_like(wav, requires_grad=True)
    for _ in range(K):
        x = (wav + d).clamp(-1, 1); tot = 0
        for _b in range(max(1, eot)):
            xb = apply_rir(x, pool[random.randrange(len(pool))]) if eot else x
            for M, e in zip(models, e0):
                ee, _ = M(logmel(xb).unsqueeze(0))
                tot = tot + (torch.nn.functional.normalize(ee, dim=1)*e.unsqueeze(0)).sum()
        g, = torch.autograd.grad(tot/(max(1,eot)*len(models)), d)
        d = (d - alpha*g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
    return (wav + d).clamp(-1, 1).detach().cpu(), d.detach().cpu()
