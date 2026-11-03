# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 09_main_run.py: Restore from Drive, protect_meas, and the main run: 20 speakers x 4 held-out rooms, eps 0.004 (98% -> 28%).
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

# ============ RESTORE + MULTI-SPEAKER RUN ============
import os, glob, math, random, time, json, torch, torchaudio, torch.nn as nn
import numpy as np
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev); assert dev == 'cuda', 'STOP — switch to T4 GPU first'
!pip install -q pesq
from pesq import pesq as pesq_fn

R = '/content/drive/MyDrive/PHC351_24324008_VoiceProtection'
REC, RES, FIG = f'{R}/testing_data/recordings', f'{R}/testing_data/results', f'{R}/figures'
SR, N_FFT, HOP, WIN, N_MELS = 16000, 512, 160, 400, 80
CHUNK = 3*SR
_w  = torch.hann_window(WIN)
_fb = torchaudio.functional.melscale_fbanks(N_FFT//2+1, 0., SR/2, N_MELS, SR,
                                            norm=None, mel_scale="htk")
def logmel(x):
    X = torch.stft(x, N_FFT, HOP, WIN, _w.to(x.device), center=True, return_complex=True)
    return torch.log(_fb.T.to(x.device) @ (X.abs()**2) + 1e-6)

class SpeakerNet(nn.Module):
    def __init__(s, nc, d=192):
        super().__init__()
        b = lambda ci,co,st: nn.Sequential(nn.Conv2d(ci,co,3,st,1), nn.BatchNorm2d(co), nn.ReLU(True))
        s.conv = nn.Sequential(b(1,32,(2,1)), b(32,64,(2,1)), b(64,128,(2,1)), b(128,128,(2,1)))
        s.proj = nn.Linear(128*5*2, d); s.bn = nn.BatchNorm1d(d); s.head = nn.Linear(d, nc)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)); B,C,F,T = h.shape; h = h.reshape(B, C*F, T)
        e = s.bn(s.proj(torch.cat([h.mean(2), h.var(2,unbiased=False).clamp_min(1e-8).sqrt()],1)))
        return e, s.head(e)

class SpeakerNetB(nn.Module):
    def __init__(s, nc, d=256):
        super().__init__()
        b = lambda ci,co,k,st: nn.Sequential(nn.Conv2d(ci,co,k,st,k//2), nn.BatchNorm2d(co), nn.ReLU(True))
        s.conv = nn.Sequential(b(1,48,5,(2,1)), b(48,96,3,(2,1)), b(96,160,3,(2,2)),
                               b(160,160,3,(2,1)), b(160,160,3,(1,1)))
        s.proj = nn.Sequential(nn.Linear(160*5*2,512), nn.ReLU(True), nn.Linear(512,d))
        s.bn = nn.BatchNorm1d(d); s.head = nn.Linear(d, nc)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)); B,C,F,T = h.shape; h = h.reshape(B, C*F, T)
        e = s.bn(s.proj(torch.cat([h.mean(2), h.var(2,unbiased=False).clamp_min(1e-8).sqrt()],1)))
        return e, s.head(e)

def load_net(cls, pats):
    for pat in pats:
        hits = glob.glob(f'{R}/**/{pat}', recursive=True)
        if hits:
            f_ = sorted(hits)[-1]; sd = torch.load(f_, map_location=dev)
            sd = sd['m'] if isinstance(sd, dict) and 'm' in sd else sd
            net = cls(sd['head.weight'].shape[0]).to(dev); net.load_state_dict(sd)
            print(f'  {cls.__name__:12s} <- {os.path.basename(f_)}'); return net.eval()
    print(f'  !! {cls.__name__} NOT FOUND'); return None
model  = load_net(SpeakerNet,  ['modelA*.pt'])
modelB = load_net(SpeakerNetB, ['modelB_independent*.pt','modelB*.pt'])

def load16k(p):
    x, sr = torchaudio.load(p); x = x.mean(0)
    return torchaudio.functional.resample(x, sr, SR) if sr != SR else x
def fit(x, n=CHUNK): return torch.nn.functional.pad(x,(0,max(0,n-x.shape[0])))[:n]
def apply_rir(x, h):
    L = x.shape[0]+h.shape[0]-1; nf = 1 << (L-1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x,nf)*torch.fft.rfft(h.to(x.device),nf), nf)[:x.shape[0]]
    return y / y.abs().max().clamp_min(1e-9) * x.abs().max()
def emb(net, x):
    with torch.no_grad(): e,_ = net(logmel(x.to(dev)).unsqueeze(0))
    return torch.nn.functional.normalize(e, dim=1)[0]
cos    = lambda a,b: float((a*b).sum())
snr_db = lambda x,d: 20*torch.log10(x.std()/d.std().clamp_min(1e-9)).item()
nrm    = lambda v: torch.nn.functional.normalize(v, dim=0)
def pq(ref, deg):
    try: return pesq_fn(SR, ref.numpy(), deg.numpy(), 'wb')
    except Exception: return float('nan')

def protect_meas(wav, eps, models, K=300, eot=4, rir_pool=None):
    wav = wav.to(dev); alpha = eps/10
    pool = [h.to(dev) for h in rir_pool]
    e0 = [emb(M, wav) for M in models]
    d = torch.zeros_like(wav, requires_grad=True)
    for _ in range(K):
        x = (wav + d).clamp(-1,1); tot = 0
        for _b in range(eot):
            xb = apply_rir(x, pool[random.randrange(len(pool))])
            for M, a0 in zip(models, e0):
                ee = torch.nn.functional.normalize(M(logmel(xb).unsqueeze(0))[0], dim=1)[0]
                tot = tot + (ee*a0).sum()
        g, = torch.autograd.grad(tot/(eot*len(models)), d)
        d = (d - alpha*g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
    return (wav + d).clamp(-1,1).detach().cpu()

md = torch.load(f'{REC}/measured_rirs.pt'); RIR_MEAS, ready = md['rirs'], md['rooms']
th = torch.load(f'{RES}/thresholds_measured.pt'); TAU_A, TAU_B = th['TAU_A'], th['TAU_B']
src = fit(load16k(f'{R}/demo_best_config/A_original.wav'))
held = {}
for p in sorted(glob.glob(f'{R}/training_data/heldout_subset/*.flac')):
    held.setdefault(os.path.basename(p).split('_')[0], []).append(p)
held = {k:v for k,v in held.items() if len(v) >= 3}
print(f'rooms {ready}   tau_B {TAU_B:.3f}   {len(held)} cached held-out speakers')

# ---- save the operating-point clip for re-recording ----
EPS_OP = 0.004
w_op = protect_meas(src, EPS_OP, [model], rir_pool=list(RIR_MEAS.values()))
torchaudio.save(f'{R}/demo_best_config/C_protected_eps004.wav', w_op.unsqueeze(0), SR)
print(f'saved C_protected_eps004.wav   SNR {snr_db(src, w_op-src):.1f} dB  PESQ {pq(src, w_op):.2f}')

# ---- multi-speaker ----
NSPK = 20
spk_run = [s for s in held][:NSPK]
res, t0 = [], time.time()
for si, s in enumerate(spk_run):
    ps = random.Random(si).sample(held[s], 2)
    s_src, s_enr = fit(load16k(ps[0])), fit(load16k(ps[1]))
    for hr in ready:
        pool  = [RIR_MEAS[r] for r in ready if r != hr]
        train = [r for r in ready if r != hr]; ht = RIR_MEAS[hr]
        EB = nrm(torch.stack([emb(modelB, apply_rir(s_enr, RIR_MEAS[r])) for r in train]
                             + [emb(modelB, s_enr)]).mean(0))
        EA = nrm(torch.stack([emb(model,  apply_rir(s_enr, RIR_MEAS[r])) for r in train]
                             + [emb(model,  s_enr)]).mean(0))
        w  = protect_meas(s_src, EPS_OP, [model], rir_pool=pool)
        yc, yp = apply_rir(s_src, ht), apply_rir(w, ht)
        res.append(dict(spk=s, room=hr,
                        cleanB=cos(EB, emb(modelB, yc)), protB=cos(EB, emb(modelB, yp)),
                        cleanA=cos(EA, emb(model,  yc)), protA=cos(EA, emb(model,  yp)),
                        snr=snr_db(s_src, w-s_src), pesq=pq(s_src, w)))
    json.dump(res, open(f'{RES}/multispeaker.json','w'))
    print(f'  {si+1:2d}/{NSPK}  {s:>8s}  {(time.time()-t0)/60:5.1f} min')

A = {k: np.array([r[k] for r in res]) for k in ('cleanB','protB','cleanA','protA','snr','pesq')}
ci = lambda v: 1.96*v.std(ddof=1)/np.sqrt(len(v))
print(f'\nn = {len(res)} trials ({NSPK} speakers x {len(ready)} rooms), eps={EPS_OP}')
for k in ('cleanB','protB','cleanA','protA'):
    print(f'  {k:8s} {A[k].mean():.3f} +- {ci(A[k]):.3f}')
print(f'  SNR {A["snr"].mean():.1f} dB   PESQ {np.nanmean(A["pesq"]):.2f}')
print(f'  attacker succeeds (cosB > {TAU_B:.3f}):  clean {100*(A["cleanB"]>TAU_B).mean():.0f}%'
      f'   protected {100*(A["protB"]>TAU_B).mean():.0f}%')
