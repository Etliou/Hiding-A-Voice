# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 07_rooms.py: Room measurement: the sine sweep; trimming, alignment and RT60; the regularised deconvolution and the measurement loop.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import torch, torchaudio, math, os
R='/content/drive/MyDrive/PHC351_24324008_VoiceProtection'
os.makedirs(f'{R}/testing_data/recordings', exist_ok=True)
SR=16000
T,f1,f2 = 10.0, 20.0, 8000.0
t = torch.linspace(0, T, int(SR*T))
K = 2*math.pi*f1*T/math.log(f2/f1)
sweep = torch.sin(K*(torch.exp(t/T*math.log(f2/f1)) - 1))
sweep = torch.cat([torch.zeros(SR), sweep*0.8, torch.zeros(2*SR)])   # 1s lead, 2s tail for reverb
torchaudio.save(f'{R}/testing_data/recordings/sweep_reference.wav', sweep.unsqueeze(0), SR)
print("saved sweep_reference.wav —", sweep.shape[0]/SR, "seconds")

# ===== next cell =====
def load16k(path):
    x, sr = torchaudio.load(path); x = x.mean(0)
    if sr != SR: x = torchaudio.functional.resample(x, sr, SR)
    return x

# convert m4a -> 16 kHz mono wav via ffmpeg (turn 446)
def convert_all():
    for p in sorted(glob.glob(f'{REC}/*')):
        base = os.path.basename(p); stem, ext = os.path.splitext(base)
        if ext.lower() not in ('.m4a','.mp3','.mov','.mp4','.aac','.caf'): continue
        s = stem.lower().replace('pro_', 'prot_')
        if not re.match(r'(sweep|clean|prot)_', s): continue
        out = f'{REC}/{s}.wav'
        if os.path.exists(out): continue
        subprocess.run(['ffmpeg','-y','-v','error','-i',p,'-ac','1','-ar',str(SR),'-c:a','pcm_s16le', out])

# cut the recording to the part that contains the sweep (turn 454)
def trim_to_sweep(rec, ref, pad=SR):
    n = 1 << (rec.shape[0] + ref.shape[0] - 1).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(rec, n) * torch.fft.rfft(ref.flip(0), n), n)
    lag = int(c[:rec.shape[0]].argmax()) - (ref.shape[0] - 1)
    a = max(0, lag - pad); b = min(rec.shape[0], max(0, lag) + ref.shape[0] + 2*pad)
    return rec[a:b], lag

# reverberation time: Schroeder backward integration, T20 x 3, truncated at the noise floor (turn 450)
def rt60(h):
    e  = h**2
    nf = e[-e.shape[0]//10:].mean()                       # noise floor
    m  = (e > 10*nf).nonzero()
    e  = e[:int(m[-1])+1] if len(m) else e
    s  = torch.flip(torch.cumsum(torch.flip(e,[0]),0),[0])
    db = 10*torch.log10(s/s[0] + 1e-12)
    try: i5, i25 = int((db<=-5).nonzero()[0]), int((db<=-25).nonzero()[0])
    except IndexError: return float('nan')
    return 3.0*(i25-i5)/SR

# line a re-recording up with the original: cross-correlate, crop, match RMS (turn 456)
def align_to(rec, ref):
    n = 1 << (rec.shape[0] + ref.shape[0] - 1).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(rec, n) * torch.fft.rfft(ref.flip(0), n), n)
    lag = max(0, int(c[:rec.shape[0]].argmax()) - (ref.shape[0] - 1))
    y = rec[lag : lag + ref.shape[0]]
    y = torch.nn.functional.pad(y, (0, max(0, ref.shape[0] - y.shape[0])))
    return y * (ref.std() / y.std().clamp_min(1e-9))

# ===== next cell =====
def deconvolve(rec, ref, f_lo=250., f_hi=7000., lam=1e-2, tail=SR):
    n = 1 << (max(rec.shape[0], ref.shape[0])*2 - 1).bit_length()
    Y = torch.fft.rfft(rec, n); X = torch.fft.rfft(ref, n)
    f  = torch.linspace(0, SR/2, X.shape[0])
    bp = 0.5*(1+torch.tanh((f-f_lo)/(0.3*f_lo))) * 0.5*(1-torch.tanh((f-f_hi)/(0.1*f_hi)))
    reg = lam*(X.abs()**2).max() * (1 + 1e4*(1-bp))
    h = torch.fft.irfft(Y*X.conj()/(X.abs()**2 + reg) * bp, n)
    k = int(h[:SR*5].abs().argmax()); h = h[max(0,k-64) : max(0,k-64)+tail]
    h = h / h.abs().max().clamp_min(1e-9)        # <-- direct path = 1 BEFORE the blend
    Hc = torch.fft.rfft(h, tail)
    fc = torch.linspace(0, SR/2, Hc.shape[0])
    bc = 0.5*(1+torch.tanh((fc-f_lo)/(0.3*f_lo))) * 0.5*(1-torch.tanh((fc-f_hi)/(0.1*f_hi)))
    d  = torch.zeros(tail); d[64] = 1.
    h  = torch.fft.irfft(Hc*bc + torch.fft.rfft(d, tail)*(1-bc), tail)
    return h / h.abs().max().clamp_min(1e-9)

RIR_MEAS = {}
print(f'{"room":15s}{"src-real":>10s}{"src-sim":>9s}{"real-sim":>10s}{"LTAS":>7s}{"RT60":>7s}')
for r in ready:
    trimmed, _ = trim_to_sweep(load16k(f'{REC}/sweep_{r}.wav'), ref)
    h = deconvolve(trimmed, ref); RIR_MEAS[r] = h          # no truncate()
    real = align_to(load16k(f'{REC}/clean_{r}.wav'), src); sim = apply_rir(src, h)
    e0, e1, e2 = emb(model,src), emb(model,real), emb(model,sim)
    Lc = {k: (lambda v: v - v.mean())(logmel(v).mean(1)) for k,v in [('real',real),('sim',sim)]}
    print(f'{r:15s}{cos(e0,e1):10.3f}{cos(e0,e2):9.3f}{cos(e1,e2):10.3f}'
          f'{float((Lc["real"]-Lc["sim"]).abs().mean()):7.2f}{rt60(h):7.2f}')

torch.save({'rirs': RIR_MEAS, 'rooms': ready, 'held': HELD}, f'{REC}/measured_rirs.pt')
