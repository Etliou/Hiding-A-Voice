# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 06_model_B.py: Encoder B, the independent judge, in a fresh session: restore the data and train (seed 1234).
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import os, glob, random, time, json, torch, torchaudio, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from google.colab import drive; drive.mount('/content/drive')

OUT = '/content/drive/MyDrive/phc351'; os.makedirs(OUT, exist_ok=True)
os.chdir('/content')
if not os.path.exists('/content/LibriSpeech/train-clean-100'):
    os.system("curl -sL -C - -o tc100.tar.gz "
              "https://www.openslr.org/resources/12/train-clean-100.tar.gz && tar -xzf tc100.tar.gz")

SR, N_FFT, HOP, WIN, N_MELS = 16000, 512, 160, 400, 80
CHUNK = 3*SR
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
_window = torch.hann_window(WIN)
_fb = torchaudio.functional.melscale_fbanks(N_FFT//2+1, 0.0, SR/2, N_MELS, SR,
                                            norm=None, mel_scale="htk")

def logmel(wav):
    X = torch.stft(wav, N_FFT, HOP, WIN, _window.to(wav.device),
                   center=True, return_complex=True)
    return torch.log(_fb.T.to(wav.device) @ (X.abs()**2) + 1e-6)

files = sorted(glob.glob('/content/LibriSpeech/train-clean-100/*/*/*.flac'))
pairs = [(p, p.split('/')[-3]) for p in files]
speakers = sorted({s for _, s in pairs}); random.Random(0).shuffle(speakers)
n_tr = int(0.8*len(speakers))
train_spk, held_spk = set(speakers[:n_tr]), set(speakers[n_tr:])
spk2idx = {s:i for i,s in enumerate(sorted(train_spk))}
print("files", len(files), "| train spk", len(train_spk), "| held", len(held_spk))

class SpeakerSet(Dataset):
    def __init__(self, pairs, keep, labelled=True):
        self.items = [(p,s) for p,s in pairs if s in keep]; self.labelled = labelled
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, spk = self.items[i]
        wav, _ = torchaudio.load(path); wav = wav.mean(0)
        if wav.shape[0] < CHUNK:
            wav = torch.nn.functional.pad(wav, (0, CHUNK-wav.shape[0]))
        else:
            st = random.randint(0, wav.shape[0]-CHUNK); wav = wav[st:st+CHUNK]
        return logmel(wav), (spk2idx[spk] if self.labelled else -1)

train_dl = DataLoader(SpeakerSet(pairs, train_spk), batch_size=32, shuffle=True,
                      num_workers=2, drop_last=True)

class SpeakerNetB(nn.Module):
    def __init__(self, n_classes, emb_dim=256):
        super().__init__()
        def blk(ci, co, k, s):
            return nn.Sequential(nn.Conv2d(ci, co, k, s, k//2),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(blk(1,48,5,(2,1)), blk(48,96,3,(2,1)),
                                  blk(96,160,3,(2,2)), blk(160,160,3,(2,1)),
                                  blk(160,160,3,(1,1)))
        self.proj = nn.Sequential(nn.Linear(160*5*2,512), nn.ReLU(inplace=True),
                                  nn.Linear(512, emb_dim))
        self.bn = nn.BatchNorm1d(emb_dim); self.head = nn.Linear(emb_dim, n_classes)
    def forward(self, x):
        h = self.conv(x.unsqueeze(1)); B,C,F,T = h.shape
        h = h.reshape(B, C*F, T)
        e = self.bn(self.proj(torch.cat([h.mean(2),
              h.var(2, unbiased=False).clamp_min(1e-8).sqrt()], 1)))
        return e, self.head(e)

torch.manual_seed(1234); random.seed(1234)
modelB = SpeakerNetB(len(spk2idx)).to(dev)
opt = torch.optim.Adam(modelB.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
print("B params(M):", round(sum(p.numel() for p in modelB.parameters())/1e6,2))

for ep in range(5):
    modelB.train(); t0=time.time(); tot=cor=0; run=0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        _, lg = modelB(xb); loss = lossf(lg, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        run += loss.item()*yb.size(0); cor += (lg.argmax(1)==yb).sum().item(); tot += yb.size(0)
    print(f"B ep {ep+1}/5 | loss {run/tot:.3f} | acc {100*cor/tot:.1f}% | {time.time()-t0:.0f}s", flush=True)
    torch.save(modelB.state_dict(), f'{OUT}/modelB.pt')

print("DONE — modelB.pt saved to Drive")
