# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 13_full_chain.py: EOT through the whole recording chain (overdrive, AGC, drift, noise, codec), evaluated with a real AAC encoder.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

# ============ AUGMENTED PHYSICAL-CHAIN EOT ============
import os, glob, math, random, json, subprocess, torch, torchaudio, torch.nn as nn
import torch.nn.functional as F, numpy as np
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev); assert dev == 'cuda', 'STOP — switch to T4 GPU'

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
        h = s.conv(x.unsqueeze(1)); B,C,Fq,T = h.shape; h = h.reshape(B, C*Fq, T)
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
        h = s.conv(x.unsqueeze(1)); B,C,Fq,T = h.shape; h = h.reshape(B, C*Fq, T)
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
    raise SystemExit(f'{cls.__name__} NOT FOUND')
model  = load_net(SpeakerNet,  ['modelA*.pt'])
modelB = load_net(SpeakerNetB, ['modelB_independent*.pt','modelB*.pt'])

def load16k(p):
    x, sr = torchaudio.load(p); x = x.mean(0)
    return torchaudio.functional.resample(x, sr, SR) if sr != SR else x
def fit(x, n=CHUNK): return F.pad(x,(0,max(0,n-x.shape[0])))[:n]
def apply_rir(x, h):
    L = x.shape[0]+h.shape[0]-1; nf = 1 << (L-1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x,nf)*torch.fft.rfft(h.to(x.device),nf), nf)[:x.shape[0]]
    return y / y.abs().max().clamp_min(1e-9) * x.abs().max()
def emb(net, x):
    with torch.no_grad(): e,_ = net(logmel(x.to(dev)).unsqueeze(0))
    return torch.nn.functional.normalize(e, dim=1)[0]
cos = lambda a,b: float((a*b).sum())
nrm = lambda v: torch.nn.functional.normalize(v, dim=0)
snr_db = lambda x,d: 20*torch.log10(x.std()/d.std().clamp_min(1e-9)).item()

# ---------- the extended channel model ----------
def saturate(y, k):                      # loudspeaker squashes peaks
    return torch.tanh(k*y)/k
def compress(y, thr=0.08, ratio=3.0):    # microphone AGC
    e = F.avg_pool1d(y.abs().view(1,1,-1), 400, 160, padding=200)
    e = F.interpolate(e, size=y.shape[0], mode='linear', align_corners=False)[0,0]
    g = (e.clamp_min(1e-5)/thr).clamp_min(1.0)**(1.0/ratio - 1.0)
    return y*g
def drift(y, d):                         # the two clocks disagree slightly
    n = y.shape[0]
    z = F.interpolate(y.view(1,1,-1), scale_factor=1.0+d, mode='linear',
                      align_corners=False)[0,0]
    return F.pad(z,(0,max(0,n-z.shape[0])))[:n]
def codec_proxy(y, floor_db=-45.0):      # a lossy codec discards quiet detail
    X = torch.stft(y, N_FFT, HOP, WIN, _w.to(y.device), center=True, return_complex=True)
    mag = X.abs().clamp_min(1e-9)
    rel = 20*torch.log10(mag/mag.max(0,keepdim=True).values.clamp_min(1e-9))
    m = torch.sigmoid((rel - floor_db)/3.0)
    return torch.istft(X*m, N_FFT, HOP, WIN, _w.to(y.device), center=True, length=y.shape[0])

def phys_chain(x, h, rng, train=True):
    y = apply_rir(x, h)
    y = y * (0.7 + 0.6*rng.random())
    y = saturate(y, 1.2 + 2.3*rng.random())
    y = compress(y)
    y = drift(y, (rng.random()-0.5)*0.002)
    y = y + torch.randn_like(y)*y.std()*(0.004 + 0.012*rng.random())
    y = codec_proxy(y)
    return y / y.abs().max().clamp_min(1e-9) * 0.7

def real_aac(y):                          # the actual codec, evaluation only
    torchaudio.save('/content/_t.wav', y.cpu().unsqueeze(0), SR)
    subprocess.run(['ffmpeg','-y','-v','error','-i','/content/_t.wav','-c:a','aac',
                    '-b:a','64k','/content/_t.m4a'], capture_output=True)
    subprocess.run(['ffmpeg','-y','-v','error','-i','/content/_t.m4a','-ac','1',
                    '-ar',str(SR),'/content/_t2.wav'], capture_output=True)
    return fit(load16k('/content/_t2.wav'), y.shape[0]).to(y.device)

def eval_chain(x, h, rng):                # full chain WITH the real codec
    y = apply_rir(x, h)
    y = saturate(y*(0.8+0.4*rng.random()), 2.0)
    y = compress(y)
    y = drift(y, (rng.random()-0.5)*0.002)
    y = y + torch.randn_like(y)*y.std()*0.01
    y = y / y.abs().max().clamp_min(1e-9) * 0.7
    return real_aac(y)

# ---------- protection ----------
def protect_meas(wav, eps, K=300, eot=4, pool=None):      # old: room only
    wav = wav.to(dev); a = eps/10; e0 = emb(model, wav)
    d = torch.zeros_like(wav, requires_grad=True)
    for _ in range(K):
        x = (wav+d).clamp(-1,1); tot = 0
        for _ in range(eot):
            y = apply_rir(x, pool[random.randrange(len(pool))])
            ee = torch.nn.functional.normalize(model(logmel(y).unsqueeze(0))[0], dim=1)[0]
            tot = tot + (ee*e0).sum()
        g, = torch.autograd.grad(tot/eot, d)
        d = (d - a*g.sign()).clamp(-eps,eps).detach().requires_grad_(True)
    return (wav+d).clamp(-1,1).detach().cpu()

def protect_phys(wav, eps, K=300, eot=4, pool=None, seed=0):   # new: full chain
    wav = wav.to(dev); a = eps/10; e0 = emb(model, wav); rng = random.Random(seed)
    d = torch.zeros_like(wav, requires_grad=True)
    for _ in range(K):
        x = (wav+d).clamp(-1,1); tot = 0
        for _ in range(eot):
            y = phys_chain(x, pool[rng.randrange(len(pool))], rng)
            ee = torch.nn.functional.normalize(model(logmel(y).unsqueeze(0))[0], dim=1)[0]
            tot = tot + (ee*e0).sum()
        g, = torch.autograd.grad(tot/eot, d)
        d = (d - a*g.sign()).clamp(-eps,eps).detach().requires_grad_(True)
    return (wav+d).clamp(-1,1).detach().cpu()

md = torch.load(f'{REC}/measured_rirs.pt'); RIR_MEAS, ready = md['rirs'], md['rooms']
th = torch.load(f'{RES}/thresholds_measured.pt'); TAU_B = th['TAU_B']
src_n = fit(load16k(f'{R}/demo_best_config/C_original_norm.wav'))
print(f'rooms {ready}   tau_B {TAU_B:.3f}')

# ================= THE GATE =================
import time; t0 = time.time()
print(f'\n=== GATE: evaluated through the FULL chain incl. real AAC ===')
print(f'{"held-out":15s}{"method":>10s}{"SNR":>7s}{"cosB":>8s}')
gate = {}
for hr in ready:
    pool = [RIR_MEAS[r] for r in ready if r != hr]; tr = [r for r in ready if r != hr]
    rngE = random.Random(99)
    EB = nrm(torch.stack([emb(modelB, eval_chain(src_n.to(dev), RIR_MEAS[q], rngE))
                          for q in tr]).mean(0))
    yc = eval_chain(src_n.to(dev), RIR_MEAS[hr], random.Random(7))
    print(f'{hr:15s}{"none":>10s}{"-":>7s}{cos(EB, emb(modelB, yc)):8.3f}')
    for nm, fn in [('room-only', protect_meas), ('full-chain', protect_phys)]:
        w = fn(src_n, 0.004, pool=pool)
        y = eval_chain(w.to(dev), RIR_MEAS[hr], random.Random(7))
        c = cos(EB, emb(modelB, y)); gate.setdefault(nm, []).append(c)
        print(f'{hr:15s}{nm:>10s}{snr_db(src_n, w-src_n):7.1f}{c:8.3f}')
print(f'\nmean cosB   room-only {np.mean(gate["room-only"]):.3f}'
      f'   full-chain {np.mean(gate["full-chain"]):.3f}   (tau_B {TAU_B:.3f})')
PASS = np.mean(gate['full-chain']) < np.mean(gate['room-only']) - 0.03
print('GATE:', 'PASS — re-record tomorrow' if PASS else 'FAIL — do not re-record')
print(f'gate took {(time.time()-t0)/60:.1f} min')

# ================= if it passed, the statistics =================
if PASS:
    held = {}
    for p in sorted(glob.glob(f'{R}/training_data/heldout_subset/*.flac')):
        held.setdefault(os.path.basename(p).split('_')[0], []).append(p)
    held = {k:v for k,v in held.items() if len(v) >= 3}
    res = []
    for si, s in enumerate(list(held)[:20]):
        ps = random.Random(si).sample(held[s], 2)
        a_, b_ = fit(load16k(ps[0])), fit(load16k(ps[1]))
        for hr in ready:
            pool = [RIR_MEAS[r] for r in ready if r != hr]; tr = [r for r in ready if r != hr]
            rngE = random.Random(99)
            EB = nrm(torch.stack([emb(modelB, eval_chain(b_.to(dev), RIR_MEAS[q], rngE))
                                  for q in tr]).mean(0))
            w = protect_phys(a_, 0.004, pool=pool, seed=si)
            yc = eval_chain(a_.to(dev), RIR_MEAS[hr], random.Random(7))
            yp = eval_chain(w.to(dev),  RIR_MEAS[hr], random.Random(7))
            res.append(dict(spk=s, room=hr, cleanB=cos(EB, emb(modelB, yc)),
                            protB=cos(EB, emb(modelB, yp)), snr=snr_db(a_, w-a_)))
        json.dump(res, open(f'{RES}/physchain_multispeaker.json','w'))
        print(f'  {si+1:2d}/20  {(time.time()-t0)/60:5.1f} min')
    cB = np.array([r['cleanB'] for r in res]); pB = np.array([r['protB'] for r in res])
    ci = lambda v: 1.96*v.std(ddof=1)/np.sqrt(len(v))
    print(f'\nn={len(res)}   clean {cB.mean():.3f}+-{ci(cB):.3f}   prot {pB.mean():.3f}+-{ci(pB):.3f}')
    print(f'attacker succeeds: clean {100*(cB>TAU_B).mean():.0f}%  protected {100*(pB>TAU_B).mean():.0f}%')
    w_final = protect_phys(src_n, 0.004, pool=list(RIR_MEAS.values()), seed=1)
    torchaudio.save(f'{R}/demo_best_config/D_protected_physchain.wav',
                    w_final.unsqueeze(0), SR)
    print('saved D_protected_physchain.wav  — record this tomorrow')
