# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 16_sim_vs_measured.py: Simulated versus measured rooms (leave one room out): the session cell with simulated rooms and protect(), then the comparison. This is the first, unmatched comparison; the matched one is in experiments_v2/ (paper Section VII-B).
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import os, glob, random, time, json, csv, shutil, math
import torch, torchaudio, torch.nn as nn, numpy as np
from google.colab import drive; drive.mount('/content/drive')

R   = '/content/drive/MyDrive/PHC351_24324008_VoiceProtection'
REC = f'{R}/testing_data/recordings'
RES = f'{R}/testing_data/results'
FIG = f'{R}/figures'
for d in [REC, RES, FIG, f'{R}/code', f'{R}/libraries',
          f'{R}/training_data', f'{R}/testing_data/audio_samples']:
    os.makedirs(d, exist_ok=True)

os.chdir('/content')
if not os.path.exists('/content/LibriSpeech/train-clean-100'):
    os.system("curl -sL -C - -o tc100.tar.gz https://www.openslr.org/resources/12/"
              "train-clean-100.tar.gz && tar -xzf tc100.tar.gz")
if not os.path.exists('/content/RIRS_NOISES'):
    os.system("curl -sL -C - -o rirs.zip https://www.openslr.org/resources/28/"
              "rirs_noises.zip && unzip -q rirs.zip")

SR, N_FFT, HOP, WIN, N_MELS = 16000, 512, 160, 400, 80
CHUNK = 3 * SR
TAU   = 0.46                      # model B decision threshold (from its EER)
dev   = 'cuda' if torch.cuda.is_available() else 'cpu'

_w  = torch.hann_window(WIN)
_fb = torchaudio.functional.melscale_fbanks(N_FFT//2+1, 0., SR/2, N_MELS, SR,
                                            norm=None, mel_scale="htk")
def logmel(x):
    X = torch.stft(x, N_FFT, HOP, WIN, _w.to(x.device),
                   center=True, return_complex=True)
    return torch.log(_fb.T.to(x.device) @ (X.abs()**2) + 1e-6)

files = sorted(glob.glob('/content/LibriSpeech/train-clean-100/*/*/*.flac'))
pairs = [(p, p.split('/')[-3]) for p in files]
spks  = sorted({s for _, s in pairs}); random.Random(0).shuffle(spks)
n = int(0.8*len(spks)); train_spk, held_spk = set(spks[:n]), set(spks[n:])
spk2idx = {s: i for i, s in enumerate(sorted(train_spk))}

class SpeakerNet(nn.Module):
    def __init__(s, nc, d=192):
        super().__init__()
        b = lambda ci,co,st: nn.Sequential(nn.Conv2d(ci,co,3,st,1),
                                           nn.BatchNorm2d(co), nn.ReLU(True))
        s.conv = nn.Sequential(b(1,32,(2,1)), b(32,64,(2,1)),
                               b(64,128,(2,1)), b(128,128,(2,1)))
        s.proj = nn.Linear(128*5*2, d); s.bn = nn.BatchNorm1d(d)
        s.head = nn.Linear(d, nc)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)); B,C,F,T = h.shape; h = h.reshape(B, C*F, T)
        e = s.bn(s.proj(torch.cat([h.mean(2),
              h.var(2, unbiased=False).clamp_min(1e-8).sqrt()], 1)))
        return e, s.head(e)

class SpeakerNetB(nn.Module):
    def __init__(s, nc, d=256):
        super().__init__()
        b = lambda ci,co,k,st: nn.Sequential(nn.Conv2d(ci,co,k,st,k//2),
                                             nn.BatchNorm2d(co), nn.ReLU(True))
        s.conv = nn.Sequential(b(1,48,5,(2,1)), b(48,96,3,(2,1)), b(96,160,3,(2,2)),
                               b(160,160,3,(2,1)), b(160,160,3,(1,1)))
        s.proj = nn.Sequential(nn.Linear(160*5*2,512), nn.ReLU(True), nn.Linear(512,d))
        s.bn = nn.BatchNorm1d(d); s.head = nn.Linear(d, nc)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)); B,C,F,T = h.shape; h = h.reshape(B, C*F, T)
        e = s.bn(s.proj(torch.cat([h.mean(2),
              h.var(2, unbiased=False).clamp_min(1e-8).sqrt()], 1)))
        return e, s.head(e)

ck = glob.glob('/content/drive/MyDrive/**/model*.pt', recursive=True)
model  = SpeakerNet(len(spk2idx)).to(dev)
model.load_state_dict(torch.load([p for p in ck if 'modelA' in p][0])); model.eval()
modelB = SpeakerNetB(len(spk2idx)).to(dev)
modelB.load_state_dict(torch.load([p for p in ck if 'modelB' in p][0])); modelB.eval()
for p in model.parameters():  p.requires_grad_(False)
for p in modelB.parameters(): p.requires_grad_(False)

rirs = []
for f in sorted(glob.glob('/content/RIRS_NOISES/simulated_rirs/**/*.wav',
                          recursive=True))[:400]:
    h, sr = torchaudio.load(f); h = h.mean(0)
    if sr != SR: h = torchaudio.functional.resample(h, sr, SR)
    h = h[:SR//2]
    if h.abs().max() > 1e-6: rirs.append((h / h.abs().max()).to(dev))
random.Random(1).shuffle(rirs)
RIR_SIM_TRAIN, RIR_SIM_TEST = rirs[:200], rirs[200:]

def apply_rir(x, h):
    L = x.shape[0] + h.shape[0] - 1; nfft = 1 << (L-1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)[:x.shape[0]]
    return y / y.abs().max().clamp_min(1e-9) * x.abs().max()

def emb(net, x):
    with torch.no_grad(): e, _ = net(logmel(x.to(dev)).unsqueeze(0))
    return torch.nn.functional.normalize(e, dim=1)[0]
cos = lambda a, b: float((a*b).sum())

def fit(x, n=CHUNK):
    x = torch.nn.functional.pad(x, (0, max(0, n - x.shape[0])))
    return x[:n]

def protect(wav, eps=0.001, K=300, eot=0, rir_pool=None):
    """eot=0 -> plain PGD.  eot=B -> average gradient over B random channels."""
    alpha = eps/10; wav = wav.to(dev); e0 = emb(model, wav)
    d = torch.zeros_like(wav, requires_grad=True)
    pool = rir_pool if rir_pool is not None else RIR_SIM_TRAIN
    for _ in range(K):
        x = (wav + d).clamp(-1, 1)
        if eot:
            tot = 0
            for _b in range(eot):
                h = pool[random.randrange(len(pool))]
                e, _ = model(logmel(apply_rir(x, h)).unsqueeze(0))
                tot = tot + (torch.nn.functional.normalize(e,dim=1)*e0.unsqueeze(0)).sum()
            loss = tot/eot
        else:
            e, _ = model(logmel(x).unsqueeze(0))
            loss = (torch.nn.functional.normalize(e,dim=1)*e0.unsqueeze(0)).sum()
        g, = torch.autograd.grad(loss, d)
        d = (d - alpha*g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
    return (wav + d).clamp(-1, 1).detach(), d.detach()

snr_db = lambda x, d: 20*torch.log10(x.std()/d.std().clamp_min(1e-9)).item()
print("restored |", len(files), "files |", len(held_spk), "held-out speakers | dev:", dev)

# ===== next cell =====
EPSS = [0.001, 0.002]
bank, t0 = {}, time.time()

# protections that don't depend on which room is held out
for eps in EPSS:
    bank[('pgd', eps, None)] = protect(src, eps=eps, K=300, eot=0)[0].cpu()
    bank[('sim', eps, None)] = protect(src, eps=eps, K=300, eot=4,
                                       rir_pool=RIR_SIM_TRAIN)[0].cpu()
    print(f'  eps={eps}  pgd+sim done   {time.time()-t0:5.0f}s')

# measured-channel EOT: train on the other rooms, never on the held-out one
for held in ready:
    pool = [RIR_MEAS[r] for r in ready if r != held]
    for eps in EPSS:
        bank[('meas', eps, held)] = protect(src, eps=eps, K=300, eot=4,
                                            rir_pool=pool)[0].cpu()
    print(f'  held={held:14s} done      {time.time()-t0:5.0f}s')

torch.save(bank, f'{RES}/loo_bank.pt')

# ---- evaluate through the held-out MEASURED channel, channel-matched enrolment ----
print(f'\n{"held-out":14s}{"eps":>8s}{"method":>8s}{"SNR":>7s}{"cosB":>8s}{"cosA":>8s}')
for held in ready:
    train = [r for r in ready if r != held]
    ht    = RIR_MEAS[held]
    EB = torch.nn.functional.normalize(torch.stack(
        [emb(modelB, apply_rir(src, RIR_MEAS[r])) for r in train]
        + [emb(modelB, src)]).mean(0), dim=0)
    EA = torch.nn.functional.normalize(torch.stack(
        [emb(model,  apply_rir(src, RIR_MEAS[r])) for r in train]
        + [emb(model,  src)]).mean(0), dim=0)
    base = apply_rir(src, ht)
    print(f'{held:14s}{"-":>8s}{"none":>8s}{"-":>7s}'
          f'{cos(EB, emb(modelB, base)):8.3f}{cos(EA, emb(model, base)):8.3f}')
    for eps in EPSS:
        for m, key in [('pgd', ('pgd',eps,None)), ('sim', ('sim',eps,None)),
                       ('meas', ('meas',eps,held))]:
            w = bank[key]
            y = apply_rir(w, ht)
            print(f'{held:14s}{eps:8.4f}{m:>8s}{snr_db(src, w-src):7.1f}'
                  f'{cos(EB, emb(modelB, y)):8.3f}{cos(EA, emb(model, y)):8.3f}')
