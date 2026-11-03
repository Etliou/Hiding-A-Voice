# =============================================================================
#  PHC-351 · Hiding a Voice · Bhavishay (24324008)
#  DEVICE CONTROL RUN: is the advantage of measured rooms really the rooms, or
#  the laptop loudspeaker and phone microphone that every measurement shares?
#
#  HOW TO RUN (Google Colab)
#    1. Runtime -> Change runtime type -> T4 GPU -> Save.
#    2. Run this cell. Allow Google Drive access when the pop-up asks.
#    3. Leave the tab open (laptop on charger, sleep turned off). About 1.5-2 h.
#    If Colab disconnects, run the cell again: everything already finished is
#    skipped and the run continues where it stopped.
#    Nothing is recorded and nothing physical is needed: it uses the sweep
#    recordings, rooms, encoders and thresholds already on your Drive.
#
#  WHAT IT DOES (in this order)
#    B. Loads YOUR encoders A and B, YOUR four measured rooms and thresholds.
#    A. Downloads the OpenSLR-28 simulated rooms (LibriSpeech only if the
#       held-out clips of the main run are not on Drive).
#    C. Prepares the simulated rooms exactly as in the overnight run v2: the
#       same 200 processed responses and the same RT60-matched picks.
#    D. Estimates the response that every measurement shares - the laptop
#       loudspeaker and the phone microphone - for each held-out room from the
#       THREE TRAINING rooms only (the held-out room is never used): the average
#       of their 1/3-octave-smoothed magnitude responses, as a minimum-phase
#       filter. A figure of it is saved for checking.
#    E. Main loop: the 20 speakers of the main run x 4 held-out rooms, five
#       conditions, each with the SAME random draws as in the overnight run:
#         meas_full         measured, 3 rooms (the paper's method)
#         sim3_rt           simulated, 3 rooms, RT60-matched (as overnight)
#         sim3_rt_dev       the same 3 simulated rooms + the device response
#         sim_proc_200      simulated, 200 processed (as overnight)
#         sim_proc_200_dev  the same 200 + the device response
#       eps 0.004 with 3 runs first, then eps 0.002 with 3 runs.
#    F. Analysis -> device_control/analysis/REPORT.md
#
#  RULE FIXED BEFORE THE RUN (key comparison: sim3_rt_dev against meas_full)
#    D = cos_B(sim3_rt_dev) - cos_B(meas_full), averaged over the 80 speaker x
#    room trials, with its 95% CI, at equal budget AND at equal SNR.
#    - CI above 0 in both: measured rooms still protect better than simulated
#      rooms that include the device response -> not only the devices.
#    - CI includes 0 in either: with the device response added, measured and
#      simulated rooms are not distinguishable -> the devices explain it.
#    - CI below 0 in both: simulated rooms + device protect better.
#
#  OUTPUT: MyDrive/PHC351_24324008_VoiceProtection/testing_data/results/device_control/
#  Nothing here invents or edits numbers: every number in the report is
#  computed from device_control/results_rows.csv, written as the runs finish.
# =============================================================================

import os, sys, re, gc, json, csv, glob, math, time, zlib, random, shutil
import zipfile, platform, subprocess, traceback, datetime

T_START = time.time()

# ----------------------------- settings --------------------------------------
SMOKE = os.environ.get('PHC_SMOKE', '0') == '1'          # tiny local test only
CONTENT = os.environ.get('PHC_CONTENT', '/content')
DRIVE = os.environ.get('PHC_DRIVE', '/content/drive/MyDrive')
R = os.environ.get('PHC_ROOT', f'{DRIVE}/PHC351_24324008_VoiceProtection')
REC, RES = f'{R}/testing_data/recordings', f'{R}/testing_data/results'
OUT = f'{RES}/device_control'
MAIN_ROWS = f'{RES}/overnight_v2/results_rows.csv'      # only for the reproduction check

EPS_MAIN = 0.004                     # the paper's operating point
N_CLIPS = 20                         # the main run's 20 held-out speakers
K_ITERS = 300                        # PGD iterations, as in the paper
B_EOT = 4                            # channels per iteration, as in the paper
PASSES = [(0.004, 0), (0.004, 1), (0.004, 2),        # in priority order
          (0.002, 0), (0.002, 1), (0.002, 2)]
TIME_BUDGET_H = float(os.environ.get('PHC_HOURS', '3.0'))
SIM_SCAN_PER_SIZE = 1000             # as in the overnight run (same RT60-matched picks)
TAU_PAPER = {'A': 0.734, 'B': 0.793}
IMPOSTOR_MEAN_B = 0.700
DEV_SMOOTH_FRAC = 3                  # 1/3-octave smoothing of the device response
DEV_TAPS = 4096                      # length of the device filter (256 ms)

if SMOKE:
    N_CLIPS, K_ITERS = 2, 3
    PASSES = [(0.004, 0), (0.002, 0)]
    SIM_SCAN_PER_SIZE = 6

SR, N_FFT, HOP, WIN, N_MELS = 16000, 512, 160, 400, 80
CHUNK = 3 * SR
F_LO, F_HI = 250., 7000.

os.makedirs(CONTENT, exist_ok=True)

# ------------------------------ helpers --------------------------------------
_LOG = None
def log(*a):
    msg = ' '.join(str(x) for x in a)
    stamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(msg, flush=True)
    if _LOG:
        try:
            with open(_LOG, 'a') as f: f.write(f'[{stamp}] {msg}\n')
        except Exception: pass

def hours(): return (time.time() - T_START) / 3600

def crc(s): return zlib.crc32(str(s).encode())

def mount_drive():
    try:
        from google.colab import drive
    except ImportError:
        print('Not running in Colab: Drive mount skipped (local test).'); return
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')

def pip_install():
    for pkg, mod in [('pesq', 'pesq'), ('pystoi', 'pystoi'), ('soundfile', 'soundfile')]:
        try: __import__(mod); continue
        except Exception: pass
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                           capture_output=True, text=True)
        print(f'  pip install {pkg}:', 'ok' if r.returncode == 0 else 'FAILED (continuing without it)')

mount_drive()
if not os.path.isdir(R):
    raise SystemExit(f'Project folder not found: {R}\nCheck that Drive is mounted and the folder name is right.')
os.makedirs(OUT, exist_ok=True); os.makedirs(f'{OUT}/cache', exist_ok=True)
_LOG = f'{OUT}/run_log.txt'
log('=' * 70); log(f'DEVICE CONTROL RUN started {datetime.datetime.now():%Y-%m-%d %H:%M}  (smoke={SMOKE})')
pip_install()

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cuda':
    torch.backends.cudnn.benchmark = True
    log('GPU:', torch.cuda.get_device_name(0))
else:
    log('!! No GPU found. It will still run, but very slowly. Runtime -> Change runtime type -> T4 GPU.')

try:
    from pesq import pesq as _pesq
except Exception:
    _pesq = None; log('  (pesq not available: PESQ columns will be empty)')
try:
    from pystoi import stoi as _stoi
except Exception:
    _stoi = None; log('  (pystoi not available: STOI/ESTOI columns will be empty)')

# ------------------- front end: identical to the project ---------------------
def _mel_fbanks_htk(n_freqs, f_min, f_max, n_mels, sample_rate):
    """Same formula as torchaudio.functional.melscale_fbanks(norm=None, mel_scale='htk')."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.max(torch.zeros(1), torch.min(down, up))

try:
    import torchaudio
    _fb = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0., SR / 2, N_MELS, SR,
                                                norm=None, mel_scale='htk')
    FB_SOURCE = f'torchaudio {torchaudio.__version__}'
except Exception:
    torchaudio = None
    _fb = _mel_fbanks_htk(N_FFT // 2 + 1, 0., SR / 2, N_MELS, SR)
    FB_SOURCE = 'own copy of the torchaudio formula'
_w = torch.hann_window(WIN)
_dev_cache = {}
def _consts(device):
    k = str(device)
    if k not in _dev_cache: _dev_cache[k] = (_w.to(device), _fb.T.contiguous().to(device))
    return _dev_cache[k]

def logmel(x):                                   # (N,) or (B, N) -> (80, T) or (B, 80, T)
    w, fbT = _consts(x.device)
    X = torch.stft(x, N_FFT, HOP, WIN, w, center=True, return_complex=True)
    return torch.log(fbT @ (X.abs() ** 2) + 1e-6)

class SpeakerNet(nn.Module):                      # encoder A (03_model_A.py)
    def __init__(s, nc, d=192):
        super().__init__()
        b = lambda ci, co, st: nn.Sequential(nn.Conv2d(ci, co, 3, st, 1), nn.BatchNorm2d(co), nn.ReLU(True))
        s.conv = nn.Sequential(b(1, 32, (2, 1)), b(32, 64, (2, 1)), b(64, 128, (2, 1)), b(128, 128, (2, 1)))
        s.proj = nn.Linear(128 * 5 * 2, d); s.bn = nn.BatchNorm1d(d); s.head = nn.Linear(d, nc)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)); B, C, Fq, T = h.shape; h = h.reshape(B, C * Fq, T)
        e = s.bn(s.proj(torch.cat([h.mean(2), h.var(2, unbiased=False).clamp_min(1e-8).sqrt()], 1)))
        return e, s.head(e)

class SpeakerNetB(nn.Module):                     # encoder B (06_model_B.py)
    def __init__(s, nc, d=256):
        super().__init__()
        b = lambda ci, co, k, st: nn.Sequential(nn.Conv2d(ci, co, k, st, k // 2), nn.BatchNorm2d(co), nn.ReLU(True))
        s.conv = nn.Sequential(b(1, 48, 5, (2, 1)), b(48, 96, 3, (2, 1)), b(96, 160, 3, (2, 2)),
                               b(160, 160, 3, (2, 1)), b(160, 160, 3, (1, 1)))
        s.proj = nn.Sequential(nn.Linear(160 * 5 * 2, 512), nn.ReLU(True), nn.Linear(512, d))
        s.bn = nn.BatchNorm1d(d); s.head = nn.Linear(d, nc)
    def forward(s, x):
        h = s.conv(x.unsqueeze(1)); B, C, Fq, T = h.shape; h = h.reshape(B, C * Fq, T)
        e = s.bn(s.proj(torch.cat([h.mean(2), h.var(2, unbiased=False).clamp_min(1e-8).sqrt()], 1)))
        return e, s.head(e)

def torch_load(p):
    for kw in ({'weights_only': True}, {'weights_only': False}, {}):
        try: return torch.load(p, map_location='cpu', **kw)
        except TypeError: continue
        except Exception:
            if kw == {}: raise
            continue
    raise RuntimeError(f'could not load {p}')

def find_ckpt(patterns):
    for pat in patterns:                                  # 1. inside the project folder (as 09 did)
        hits = [h for h in glob.glob(f'{R}/**/{pat}', recursive=True) if '/overnight_v2/' not in h]
        if hits: return sorted(hits)[-1]
    for root in (DRIVE, f'{DRIVE}/phc351'):              # 2. where 04 and 06 saved them
        for pat in patterns:
            hits = glob.glob(f'{root}/{pat}')
            if hits: return sorted(hits)[-1]
    for pat in patterns:                                  # 3. anywhere in My Drive
        hits = glob.glob(f'{DRIVE}/**/{pat}', recursive=True)
        if hits: return sorted(hits)[-1]
    return None

def load_net(cls, patterns):
    p = find_ckpt(patterns)
    if p is None: raise SystemExit(f'{cls.__name__}: no file matching {patterns} found on Drive.')
    sd = torch_load(p)
    if isinstance(sd, dict) and 'm' in sd and isinstance(sd['m'], dict): sd = sd['m']
    net = cls(sd['head.weight'].shape[0]).to(dev); net.load_state_dict(sd); net.eval()
    for q in net.parameters(): q.requires_grad_(False)
    log(f'  {cls.__name__:12s} <- {p}')
    return net, p

def load16k(p):
    x, sr = sf.read(p, dtype='float32', always_2d=True)
    x = torch.from_numpy(np.ascontiguousarray(x.mean(1)))
    if sr != SR:
        if torchaudio is not None:
            x = torchaudio.functional.resample(x, sr, SR)
        else:
            from scipy.signal import resample_poly
            g = math.gcd(int(sr), SR)
            x = torch.from_numpy(resample_poly(x.numpy(), SR // g, int(sr) // g).astype(np.float32))
    return x

def save_wav(p, x):
    sf.write(p, np.clip(x.detach().cpu().numpy(), -1, 1), SR, subtype='PCM_16')

def fit(x, n=CHUNK): return F.pad(x, (0, max(0, n - x.shape[0])))[:n]
def nrm(v): return F.normalize(v, dim=0)
def cos(a, b): return float((a * b).sum())
def snr_db(x, d): return 20 * torch.log10(x.std() / d.std().clamp_min(1e-9)).item()

def apply_rir(x, h):                               # exactly as in 09 / 16
    L = x.shape[0] + h.shape[0] - 1; nf = 1 << (L - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nf) * torch.fft.rfft(h.to(x.device), nf), nf)[:x.shape[0]]
    return y / y.abs().max().clamp_min(1e-9) * x.abs().max()

@torch.no_grad()
def emb(net, x):
    e, _ = net(logmel(x.to(dev)).unsqueeze(0))
    return F.normalize(e, dim=1)[0]

@torch.no_grad()
def emb_batch(net, X):                              # X: (n, N) -> (n, D), in chunks
    out = []
    for i in range(0, X.shape[0], 16):
        e, _ = net(logmel(X[i:i + 16].to(dev)))
        out.append(F.normalize(e, dim=1))
    return torch.cat(out)

def pesq_wb(ref, deg):
    if _pesq is None: return float('nan')
    try: return float(_pesq(SR, ref.cpu().numpy().astype(np.float64), deg.cpu().numpy().astype(np.float64), 'wb'))
    except Exception: return float('nan')

def stoi_pair(ref, deg):
    if _stoi is None: return float('nan'), float('nan')
    try:
        r, d = ref.cpu().numpy().astype(np.float64), deg.cpu().numpy().astype(np.float64)
        return float(_stoi(r, d, SR, extended=False)), float(_stoi(r, d, SR, extended=True))
    except Exception: return float('nan'), float('nan')

# --------------- room processing (07_rooms.py, with switches) ----------------
def bandweights(n_bins):
    f = torch.linspace(0, SR / 2, n_bins)
    return 0.5 * (1 + torch.tanh((f - F_LO) / (0.3 * F_LO))) * 0.5 * (1 - torch.tanh((f - F_HI) / (0.1 * F_HI)))

def trim_to_sweep(rec, ref, pad=SR):
    n = 1 << (rec.shape[0] + ref.shape[0] - 1).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(rec, n) * torch.fft.rfft(ref.flip(0), n), n)
    lag = int(c[:rec.shape[0]].argmax()) - (ref.shape[0] - 1)
    a = max(0, lag - pad); b = min(rec.shape[0], max(0, lag) + ref.shape[0] + 2 * pad)
    return rec[a:b], lag

def deconv_raw(rec, ref, freq_reg=True, bandpass=True, lam=1e-2):
    n = 1 << (max(rec.shape[0], ref.shape[0]) * 2 - 1).bit_length()
    Y = torch.fft.rfft(rec, n); X = torch.fft.rfft(ref, n)
    bp = bandweights(X.shape[0])
    if freq_reg: reg = lam * (X.abs() ** 2).max() * (1 + 1e4 * (1 - bp))   # paper: frequency dependent
    else:        reg = lam * (X.abs() ** 2).max()                          # ablation: constant
    Hf = Y * X.conj() / (X.abs() ** 2 + reg)
    if bandpass: Hf = Hf * bp
    return torch.fft.irfft(Hf, n)

def post_process(h_long, prenorm=True, blend=True, tail=SR):
    k = int(h_long[:SR * 5].abs().argmax())
    h = h_long[max(0, k - 64): max(0, k - 64) + tail]
    h = F.pad(h, (0, max(0, tail - h.shape[0])))
    if prenorm: h = h / h.abs().max().clamp_min(1e-9)            # direct path = 1 before the blend
    if blend:
        Hc = torch.fft.rfft(h, tail); bc = bandweights(Hc.shape[0])
        d = torch.zeros(tail); d[64] = 1.
        h = torch.fft.irfft(Hc * bc + torch.fft.rfft(d, tail) * (1 - bc), tail)
    return h / h.abs().max().clamp_min(1e-9)

def sim_process(h):
    """A simulated response given the SAME frequency treatment and normalisation as the measured ones."""
    n = 1 << (2 * max(h.shape[0], SR) - 1).bit_length()
    hl = torch.fft.irfft(torch.fft.rfft(h, n) * bandweights(n // 2 + 1), n)
    return post_process(hl, prenorm=True, blend=True)

def rt60(h):                                      # 07_rooms.py
    e = h ** 2
    nf = e[-e.shape[0] // 10:].mean()
    m = (e > 10 * nf).nonzero()
    e = e[:int(m[-1]) + 1] if len(m) else e
    s = torch.flip(torch.cumsum(torch.flip(e, [0]), 0), [0])
    db = 10 * torch.log10(s / s[0] + 1e-12)
    try: i5, i25 = int((db <= -5).nonzero()[0]), int((db <= -25).nonzero()[0])
    except IndexError: return float('nan')
    return 3.0 * (i25 - i5) / SR

def early_energy(h, ms=50):
    k = int(h.abs().argmax()); n = int(SR * ms / 1000)
    return float((h[:k + n] ** 2).sum() / (h ** 2).sum().clamp_min(1e-12))

def best_xcorr(a, b, maxlag=400):
    n = 1 << (a.shape[0] + b.shape[0]).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(a, n) * torch.fft.rfft(b.flip(0), n), n)
    c = c / (a.norm() * b.norm()).clamp_min(1e-12)
    mid = b.shape[0] - 1
    return float(c[max(0, mid - maxlag): mid + maxlag + 1].max())

# ---------------- batched EOT: same maths as protect_meas in 09 --------------
class Pool:
    """A set of channels, Fourier-transformed once, applied B at a time."""
    def __init__(self, rirs, N):
        L = max(int(h.shape[0]) for h in rirs)
        self.nf = 1 << (N + L - 2).bit_length(); self.N = N
        self.H = torch.stack([torch.fft.rfft(h.float().to(dev), self.nf) for h in rirs])
    def apply(self, x, idx, outnorm=True):
        y = torch.fft.irfft(torch.fft.rfft(x, self.nf).unsqueeze(0) * self.H[idx], self.nf)[:, :self.N]
        if outnorm: y = y / y.abs().amax(1, keepdim=True).clamp_min(1e-9) * x.abs().max()
        return y

def protect(wav, eps, pool=None, rng=None, K=None, B=None, outnorm=True):
    """PGD (pool=None) or EOT over `pool`: K steps, step eps/10, B channels per step, untargeted."""
    K = K or K_ITERS; B = B or B_EOT
    wav = wav.to(dev).float(); alpha = eps / 10
    e0 = emb(modelA, wav)
    d = torch.zeros_like(wav, requires_grad=True)
    P = 0 if pool is None else pool.H.shape[0]
    for _ in range(K):
        x = (wav + d).clamp(-1, 1)
        if pool is None: xb = x.unsqueeze(0)
        else:
            idx = torch.tensor([rng.randrange(P) for _ in range(B)], device=dev)
            xb = pool.apply(x, idx, outnorm)
        e, _ = modelA(logmel(xb))
        loss = (F.normalize(e, dim=1) * e0.unsqueeze(0)).sum(1).mean()
        g, = torch.autograd.grad(loss, d)
        d = (d - alpha * g.sign()).clamp(-eps, eps).detach().requires_grad_(True)
    return (wav + d).clamp(-1, 1).detach().cpu()

# ------------------------------ downloads ------------------------------------
def fetch(name, urls, dest, min_bytes):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes: return True
    for u in urls:
        log(f'  downloading {name} from {u} ...')
        r = subprocess.run(['curl', '-sS', '-L', '-f', '-C', '-', '--retry', '5', '--retry-delay', '10',
                            '--connect-timeout', '30', '-o', dest, u])
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) >= min_bytes: return True
        log(f'    failed (curl code {r.returncode}); trying the next mirror')
        if os.path.exists(dest) and r.returncode in (18, 22, 33, 36): os.remove(dest)
    return False

def get_librispeech():
    base = f'{CONTENT}/LibriSpeech/train-clean-100'
    n = len(glob.glob(f'{base}/*/*/*.flac')) if os.path.isdir(base) else 0
    if n >= 28000 or (SMOKE and n > 0): return True
    tgz = f'{CONTENT}/tc100.tar.gz'
    urls = [f'https://{h}/resources/12/train-clean-100.tar.gz'
            for h in ('www.openslr.org', 'us.openslr.org', 'openslr.elda.org', 'openslr.magicdatatech.com')]
    if not fetch('LibriSpeech train-clean-100 (6.3 GB)', urls, tgz, 6.0e9): return False
    log('  unpacking LibriSpeech ...')
    r = subprocess.run(['tar', '-xzf', tgz, '-C', CONTENT])
    ok = r.returncode == 0 and len(glob.glob(f'{base}/*/*/*.flac')) >= 28000
    if ok: os.remove(tgz)
    else:
        log('  !! LibriSpeech archive looks damaged; deleting it so a re-run downloads it again')
        if os.path.exists(tgz): os.remove(tgz)
    return ok

def get_rirs():
    base = f'{CONTENT}/RIRS_NOISES/simulated_rirs'
    n = len(glob.glob(f'{base}/**/*.wav', recursive=True)) if os.path.isdir(base) else 0
    if n >= 10000 or (SMOKE and n > 0): return True
    z = f'{CONTENT}/rirs.zip'
    urls = [f'https://{h}/resources/28/rirs_noises.zip'
            for h in ('www.openslr.org', 'us.openslr.org', 'openslr.elda.org', 'openslr.magicdatatech.com')]
    if not fetch('OpenSLR-28 rooms (1.3 GB)', urls, z, 1.0e9): return False
    log('  unpacking simulated rooms ...')
    r = subprocess.run(['unzip', '-q', '-o', z, '-d', CONTENT])
    ok = r.returncode == 0 and len(glob.glob(f'{base}/**/*.wav', recursive=True)) >= 10000
    if not ok:
        log('  !! rooms archive looks damaged; deleting it so a re-run downloads it again')
        if os.path.exists(z): os.remove(z)
    return ok

# =============================================================================
#  B. load encoders, measured rooms, thresholds
# =============================================================================
INFO = {'started': datetime.datetime.now().isoformat(timespec='seconds'), 'smoke': SMOKE,
        'python': platform.python_version(), 'torch': torch.__version__, 'device': dev,
        'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None,
        'filterbank': FB_SOURCE, 'K': K_ITERS, 'B': B_EOT, 'passes': PASSES}
log('B. Loading encoders, rooms and thresholds from Drive')
modelA, INFO['modelA'] = load_net(SpeakerNet, ['modelA*.pt'])
modelB, INFO['modelB'] = load_net(SpeakerNetB, ['modelB_independent*.pt', 'modelB*.pt'])
md = torch_load(f'{REC}/measured_rirs.pt')
ROOMS = list(md['rooms'])
RIR_MEAS = {r: md['rirs'][r].float().cpu() for r in ROOMS}
TAU = dict(TAU_PAPER)
if os.path.exists(f'{RES}/thresholds_measured.pt'):
    th = torch_load(f'{RES}/thresholds_measured.pt')
    TAU = {'A': float(th['TAU_A']), 'B': float(th['TAU_B'])}
INFO['tau'] = TAU; INFO['rooms'] = ROOMS
RT_MEAS = {r: rt60(RIR_MEAS[r]) for r in ROOMS}
log(f'  rooms: ' + ', '.join(f'{r} (RT60 {RT_MEAS[r]:.2f} s)' for r in ROOMS))
log(f'  thresholds: tau_A {TAU["A"]:.3f}   tau_B {TAU["B"]:.3f}')

# =============================================================================
#  A. downloads and the main-run clips
# =============================================================================
log('A. Downloads')
HAVE_SIM = get_rirs()

def held_from_drive():
    d = {}
    for p in sorted(glob.glob(f'{R}/training_data/heldout_subset/*.flac')):
        d.setdefault(os.path.basename(p).split('_')[0], []).append(p)
    return {k: v for k, v in d.items() if len(v) >= 3}

held = held_from_drive(); HELD_SOURCE = 'Drive training_data/heldout_subset (same files as the main run)'
if len(held) < N_CLIPS:
    log('  the held-out clips are not all on Drive: downloading LibriSpeech to rebuild the split')
    HAVE_LS = get_librispeech()
    LS_FILES = sorted(glob.glob(f'{CONTENT}/LibriSpeech/train-clean-100/*/*/*.flac')) if HAVE_LS else []
    _spks = sorted({p.split('/')[-3] for p in LS_FILES}); random.Random(0).shuffle(_spks)
    HELD_SPK = sorted(_spks[int(0.8 * len(_spks)):])
    d, hs = {}, set(HELD_SPK)
    for p in LS_FILES:
        s = p.split('/')[-3]
        if s in hs: d.setdefault(s, []).append(p)
    held = {k: v for k, v in sorted(d.items()) if len(v) >= 3}
    HELD_SOURCE = 'LibriSpeech held-out split (heldout_subset not found on Drive)'
log(f'  simulated rooms: {"ready" if HAVE_SIM else "NOT available"}')
SPK_RUN = list(held)[:N_CLIPS]
CLIPS = []
for si, s in enumerate(SPK_RUN):                     # same choice of files as the main run
    ps = random.Random(si).sample(held[s], 2)
    CLIPS.append(dict(si=si, spk=s, src_file=ps[0], enr_file=ps[1]))
INFO['held_source'] = HELD_SOURCE; INFO['clips'] = CLIPS
log(f'  clips: {len(CLIPS)} speakers from {HELD_SOURCE}')
if not CLIPS: log('  !! no held-out speakers found: the main loop will be skipped')
if not HAVE_SIM: log('  !! simulated rooms missing: the simulated conditions cannot run (check the download above)')

# =============================================================================
#  C. simulated rooms (copied unchanged from the overnight run v2)
# =============================================================================
log('C. Simulated rooms (as in the overnight run v2)')
SIM_RAW, SIM_PROC, SIM_SCAN = [], [], []        # lists of (file, response)
if HAVE_SIM:
    try:
        allf = sorted(glob.glob(f'{CONTENT}/RIRS_NOISES/simulated_rirs/**/*.wav', recursive=True))
        raw = []                                   # 16_sim_vs_measured.py, exactly
        for f in allf[:400]:
            h = load16k(f); hs = h[:SR // 2]
            if hs.abs().max() > 1e-6: raw.append((f, hs / hs.abs().max(), h))
        random.Random(1).shuffle(raw)
        SIM_RAW = [(f, hs) for f, hs, _ in raw[:200]]
        SIM_PROC = [(f, sim_process(h)) for f, _, h in raw[:200]]
        def size_of(f):
            m = re.search(r'simulated_rirs/([^/]+)/', f); return m.group(1) if m else '?'
        def room_of(f):
            m = re.search(r'simulated_rirs/[^/]+/([^/]+)/', f); return m.group(1) if m else '?'
        comp = {}
        for f, _ in SIM_RAW: comp.setdefault(size_of(f), set()).add(room_of(f))
        INFO['paper_sim_pool'] = {k: {'n_rirs': sum(size_of(f) == k for f, _ in SIM_RAW),
                                      'n_rooms': len(v), 'rooms': sorted(v)} for k, v in comp.items()}
        INFO['paper_sim_pool_rt60'] = [rt60(h) for _, h in SIM_PROC]
        log('  the paper\'s 200 simulated responses came from: ' +
            '; '.join(f'{k}: {v["n_rirs"]} responses in {v["n_rooms"]} rooms' for k, v in INFO['paper_sim_pool'].items()))
        by_size = {}
        for f in allf: by_size.setdefault(size_of(f), []).append(f)
        rs = random.Random(2026)
        for k in sorted(by_size):
            pick = by_size[k] if len(by_size[k]) <= SIM_SCAN_PER_SIZE else rs.sample(by_size[k], SIM_SCAN_PER_SIZE)
            for f in sorted(pick):
                hp = sim_process(load16k(f)); t = rt60(hp)
                if np.isfinite(t): SIM_SCAN.append((f, hp, t, size_of(f)))
        log(f'  scanned {len(SIM_SCAN)} simulated responses for RT60 matching '
            f'(RT60 {min(t for *_, t, _ in SIM_SCAN):.2f}-{max(t for *_, t, _ in SIM_SCAN):.2f} s)' if SIM_SCAN else '  !! no usable simulated responses')
    except Exception:
        log('  !! simulated-room preparation failed:\n' + traceback.format_exc())
        SIM_RAW, SIM_PROC, SIM_SCAN = [], [], []

SIM3_CHOICES = {}
def sim3_for(seed, held_room):
    """3 simulated rooms, one per training room, matched in RT60 (random among the 5 nearest)."""
    key = f'{seed}|{held_room}'
    if key in SIM3_CHOICES: return SIM3_CHOICES[key]
    rng = random.Random(crc('sim3|' + key)); used, out = set(), []
    for r in [q for q in ROOMS if q != held_room]:
        order = sorted((i for i in range(len(SIM_SCAN)) if i not in used),
                       key=lambda i: abs(SIM_SCAN[i][2] - RT_MEAS[r]))[:5]
        i = rng.choice(order); used.add(i)
        out.append(dict(i=i, file=SIM_SCAN[i][0], size=SIM_SCAN[i][3], rt60=SIM_SCAN[i][2],
                        target_room=r, target_rt60=RT_MEAS[r]))
    SIM3_CHOICES[key] = out
    return out

def precompute_sim3():
    if not SIM_SCAN: return
    for _, seed in PASSES:
        for hr in ROOMS: sim3_for(seed, hr)

def save_sim3():
    p = f'{OUT}/sim3_choices.json'
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(SIM3_CHOICES)
    if old: json.dump(old, open(p, 'w'), indent=1)

precompute_sim3(); save_sim3()
json.dump(INFO, open(f'{OUT}/setup_info.json', 'w'), indent=1, default=str)

# =============================================================================
#  D. the response shared by every measurement (loudspeaker + microphone)
# =============================================================================
log('D. Estimating the shared loudspeaker + microphone response (training rooms only)')
NFFT_DEV = 1 << 15
FREQS = np.fft.rfftfreq(NFFT_DEV, 1 / SR)

def mag_db(h):
    return 20 * np.log10(np.abs(np.fft.rfft(np.asarray(h, dtype=np.float64), NFFT_DEV)) + 1e-9)

def smooth_db(L, frac=DEV_SMOOTH_FRAC):
    """fractional-octave smoothing of a magnitude response in dB"""
    lo = np.searchsorted(FREQS, FREQS * 2 ** (-1 / (2 * frac)), 'left')
    hi = np.searchsorted(FREQS, FREQS * 2 ** (1 / (2 * frac)), 'right')
    hi = np.maximum(hi, lo + 1)
    cs = np.concatenate([[0.0], np.cumsum(L)])
    return (cs[hi] - cs[lo]) / (hi - lo)

def minphase(Ldb, taps=DEV_TAPS):
    """minimum-phase filter with the given one-sided magnitude (dB), by the real cepstrum"""
    n = NFFT_DEV
    logmag = Ldb / 20 * np.log(10)
    full = np.concatenate([logmag, logmag[-2:0:-1]])
    c = np.fft.ifft(full).real
    w = np.zeros(n); w[0] = 1; w[1:n // 2] = 2; w[n // 2] = 1
    h = np.fft.ifft(np.exp(np.fft.fft(c * w))).real
    tail = float((h[taps:] ** 2).sum() / max((h ** 2).sum(), 1e-30))
    h = h[:taps].copy(); fl = taps // 8
    h[-fl:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fl)))
    return h, tail

_NCONV = 1 << (SR + DEV_TAPS - 1).bit_length()
def with_device(h, hdev):
    """a room response with the device response added (FFT convolution), peak = 1, 1 s long"""
    y = np.fft.irfft(np.fft.rfft(np.asarray(h, dtype=np.float64), _NCONV) * np.fft.rfft(hdev, _NCONV), _NCONV)[:SR]
    return torch.from_numpy((y / max(np.abs(y).max(), 1e-12)).astype(np.float32))

def band_dist(a_db, b_db):
    """mean absolute difference (dB) of two smoothed responses inside 250 Hz-7 kHz, levels aligned"""
    m = (FREQS >= F_LO) & (FREQS <= F_HI)
    a, b = a_db[m] - a_db[m].mean(), b_db[m] - b_db[m].mean()
    return float(np.abs(a - b).mean())

# Only the SHAPE inside 250 Hz-7 kHz is taken, scaled so that the filter keeps the in-band energy
# (power mean 0 dB); outside the band it is left at 0 dB, with the same band weights as the room
# processing. So it moves energy between frequencies inside the band and adds none: otherwise the
# reverberant energy of the measured rooms would be counted again as an in-band boost.
def pmean_db(L): return 10 * np.log10(np.mean(10 ** (L[INB] / 10)))     # in-band power mean, dB
BW = 0.5 * (1 + np.tanh((FREQS - F_LO) / (0.3 * F_LO))) * 0.5 * (1 - np.tanh((FREQS - F_HI) / (0.1 * F_HI)))
INB = (FREQS >= F_LO) & (FREQS <= F_HI)
SM = {r: smooth_db(mag_db(RIR_MEAS[r].numpy())) for r in ROOMS}
DEVICE, DEV_CURVE, DEV_INFO = {}, {}, {}
for hr in ROOMS:
    train = [r for r in ROOMS if r != hr]
    L = np.mean([SM[r] for r in train], axis=0)            # held-out room never used
    L = (L - pmean_db(L)) * BW
    hdev, tail = minphase(L)
    DEVICE[hr], DEV_CURVE[hr] = hdev, L
    DEV_INFO[hr] = dict(train=train, filter_tail_energy=tail,
                        heldout_vs_device_dB=band_dist(SM[hr], L),
                        train_vs_device_dB=float(np.mean([band_dist(SM[r], L) for r in train])))
    log(f'  held out {hr}: estimate from {", ".join(train)}; it differs from the held-out room by '
        f'{DEV_INFO[hr]["heldout_vs_device_dB"]:.1f} dB on average (250 Hz-7 kHz); filter tail {tail:.1e}')
INFO['device'] = DEV_INFO

if SIM_SCAN:                    # check: does the device response bring simulated rooms closer to the measured ones?
    dd = []
    for hr in ROOMS:
        for x in sim3_for(PASSES[0][1], hr):
            hs = SIM_SCAN[x['i']][1].numpy(); r = x['target_room']
            dd.append((band_dist(SM[r], smooth_db(mag_db(hs))),
                       band_dist(SM[r], smooth_db(mag_db(with_device(hs, DEVICE[hr]).numpy())))))
    INFO['device_check'] = {'sim3_vs_measured_dB_without_device': float(np.mean([a for a, _ in dd])),
                            'sim3_vs_measured_dB_with_device': float(np.mean([b for _, b in dd]))}
    log(f'  RT60-matched simulated rooms differ from their measured room by '
        f'{INFO["device_check"]["sim3_vs_measured_dB_without_device"]:.1f} dB without the device response and '
        f'{INFO["device_check"]["sim3_vs_measured_dB_with_device"]:.1f} dB with it (250 Hz-7 kHz)')

try:                            # a figure of the estimate, one panel per held-out room
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(f'{OUT}/analysis', exist_ok=True)
    m = (FREQS >= 100) & (FREQS <= 8000)
    fig, axs = plt.subplots(1, len(ROOMS), figsize=(3.1 * len(ROOMS), 2.9), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, hr in zip(axs, ROOMS):
        for r in DEV_INFO[hr]['train']:
            ax.semilogx(FREQS[m], SM[r][m] - pmean_db(SM[r]), lw=0.8, color='#a3a3a3')
        ax.semilogx(FREQS[m], DEV_CURVE[hr][m], lw=2.0, color='#2E6FA7', label='device filter (3 training rooms)')
        ax.semilogx(FREQS[m], SM[hr][m] - pmean_db(SM[hr]), lw=1.0, ls='--', color='#D55E00', label='held-out room')
        ax.axvspan(100, F_LO, color='#eeeeee', zorder=0); ax.axvspan(F_HI, 8000, color='#eeeeee', zorder=0)
        ax.set_title(f'held out: {hr}', fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel('frequency (Hz)', fontsize=8)
    axs[0].set_ylabel('dB, 1/3 octave (in-band power mean = 0)', fontsize=8); axs[0].legend(fontsize=6.5, frameon=False, loc='lower center')
    fig.suptitle('Shared loudspeaker + microphone response, estimated without the held-out room (grey: training rooms)', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f'{OUT}/analysis/fig_device_response.png', dpi=200); fig.savefig(f'{OUT}/analysis/fig_device_response.pdf')
    plt.close(fig)
except Exception:
    log('  (device figure failed, continuing)\n' + traceback.format_exc())
json.dump(INFO, open(f'{OUT}/setup_info.json', 'w'), indent=1, default=str)

# =============================================================================
#  E. main loop
# =============================================================================
ROWS = f'{OUT}/results_rows.csv'
DONE = f'{OUT}/done_units.txt'
FIELDS = ['eps', 'seed', 'clip', 'spk', 'fold', 'cond', 'snr_matched', 'snr_db', 'cosB', 'cosA',
          'succB', 'succA', 'pesq', 'stoi', 'estoi', 'prot_sec', 'src_file', 'enr_file']

def done_units():
    return set(open(DONE).read().split()) if os.path.exists(DONE) else set()

def write_rows(rows, key):
    new = not os.path.exists(ROWS)
    with open(ROWS, 'a', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new: wr.writeheader()
        for r in rows: wr.writerow({k: r.get(k, '') for k in FIELDS})
        f.flush(); os.fsync(f.fileno())
    with open(DONE, 'a') as f:
        f.write(key + '\n'); f.flush(); os.fsync(f.fileno())

POOLS = {}
def pool_for(name, make):
    """a channel pool, built once; `make` returns the list of responses"""
    if name not in POOLS: POOLS[name] = Pool(make(), CHUNK)
    return POOLS[name]

def conditions_for(seed, hr):
    """per held-out room: name -> (pool, rng name, rng room). The rng names make every
    condition draw exactly the same channel indices as in the overnight run; the
    device versions reuse the draws of the condition they extend."""
    train = [r for r in ROOMS if r != hr]
    c = {'meas_full': (pool_for(f'meas|{hr}', lambda: [RIR_MEAS[r] for r in train]), 'meas_full', hr)}
    if SIM_SCAN:
        ch = sim3_for(seed, hr)
        c['sim3_rt'] = (pool_for(f'sim3|{seed}|{hr}', lambda: [SIM_SCAN[x['i']][1] for x in ch]), 'sim3_rt', hr)
        c['sim3_rt_dev'] = (pool_for(f'sim3dev|{seed}|{hr}',
                                     lambda: [with_device(SIM_SCAN[x['i']][1], DEVICE[hr]) for x in ch]), 'sim3_rt', hr)
    if SIM_PROC:
        c['sim_proc_200_dev'] = (pool_for(f'simprocdev|{hr}',
                                          lambda: [with_device(h, DEVICE[hr]) for _, h in SIM_PROC]), 'sim_proc_200', '')
    return c

def run_unit(eps, seed, cl):
    src, enr = fit(load16k(cl['src_file'])), fit(load16k(cl['enr_file']))
    si = cl['si']; rows = []
    def rng(cond, hr=''): return random.Random(crc(f'{eps}|{seed}|{si}|{cond}|{hr}'))   # as in the overnight run
    def timed(pool, rc, rh):
        t = time.time(); w = protect(src, eps, pool, rng(rc, rh))
        return w, time.time() - t
    shared = {}
    if SIM_PROC: shared['sim_proc_200'] = timed(pool_for('sim_proc', lambda: [h for _, h in SIM_PROC]), 'sim_proc_200', '')
    q = {}                                          # quality/intelligibility of each protected clip (digital)
    def quality(w):
        st, est = stoi_pair(src, w)
        return dict(snr_db=snr_db(src, w - src), pesq=pesq_wb(src, w), stoi=st, estoi=est)
    for k, (w, _) in shared.items(): q[k] = quality(w)
    for hr in ROOMS:
        train = [r for r in ROOMS if r != hr]
        EB = nrm(torch.stack([emb(modelB, apply_rir(enr, RIR_MEAS[r])) for r in train] + [emb(modelB, enr)]).mean(0))
        EA = nrm(torch.stack([emb(modelA, apply_rir(enr, RIR_MEAS[r])) for r in train] + [emb(modelA, enr)]).mean(0))
        prot = dict(shared)
        for name, (pool, rc, rh) in conditions_for(seed, hr).items():
            prot[name] = timed(pool, rc, rh)
            q[(name, hr)] = quality(prot[name][0])
        base = dict(eps=eps, seed=seed, clip=si, spk=cl['spk'], fold=hr,
                    src_file=os.path.basename(cl['src_file']), enr_file=os.path.basename(cl['enr_file']))
        names = list(prot)
        def judge(ws):
            Y = torch.stack([apply_rir(w, RIR_MEAS[hr]) for w in ws])
            cb = (emb_batch(modelB, Y) @ EB.to(dev)).tolist(); ca = (emb_batch(modelA, Y) @ EA.to(dev)).tolist()
            return cb, ca
        cb, ca = judge([src] + [prot[n][0] for n in names])
        rows.append(dict(base, cond='none', snr_matched=0, snr_db='', cosB=cb[0], cosA=ca[0],
                         succB=int(cb[0] > TAU['B']), succA=int(ca[0] > TAU['A'])))
        for i, n in enumerate(names):
            qq = q[n] if n in q else q[(n, hr)]
            rows.append(dict(base, cond=n, snr_matched=0, cosB=cb[i + 1], cosA=ca[i + 1],
                             succB=int(cb[i + 1] > TAU['B']), succA=int(ca[i + 1] > TAU['A']),
                             prot_sec=round(prot[n][1], 2), **qq))
        # equal-SNR evaluation: every perturbation of this trial scaled down to the quietest one's SNR
        snrs = {n: snr_db(src, prot[n][0] - src) for n in names}
        target = max(snrs.values()); ws = []
        for n in names:
            dlt = prot[n][0] - src
            ws.append((src + dlt * 10 ** ((snrs[n] - target) / 20)).clamp(-1, 1))
        cb2, ca2 = judge(ws)
        for i, n in enumerate(names):
            rows.append(dict(base, cond=n, snr_matched=1, snr_db=snr_db(src, ws[i] - src), cosB=cb2[i], cosA=ca2[i],
                             succB=int(cb2[i] > TAU['B']), succA=int(ca2[i] > TAU['A'])))
    return rows

def main_loop():
    if not CLIPS: return
    done = done_units(); fails = 0
    total = len(PASSES) * len(CLIPS); n0 = len([1 for e, s in PASSES for c in CLIPS if f'{e}|{s}|{c["si"]}' in done])
    log(f'E. Main loop: {total} units ({len(PASSES)} passes x {len(CLIPS)} speakers), {n0} already done')
    t_units = []
    for eps, seed in PASSES:
        for cl in CLIPS:
            key = f'{eps}|{seed}|{cl["si"]}'
            if key in done: continue
            if hours() > TIME_BUDGET_H:
                log(f'  time budget of {TIME_BUDGET_H} h reached: stopping the loop and analysing what is done'); return
            t = time.time()
            try:
                rows = run_unit(eps, seed, cl); write_rows(rows, key); done.add(key); fails = 0
            except KeyboardInterrupt:
                log('  stopped by you: analysing what is done'); return
            except Exception:
                fails += 1
                log(f'  !! unit {key} failed ({fails} in a row):\n' + traceback.format_exc())
                if dev == 'cuda': torch.cuda.empty_cache()
                if fails >= 3: log('  three failures in a row: stopping the loop'); return
                continue
            t_units.append(time.time() - t)
            left = total - len([1 for e, s in PASSES for c in CLIPS if f'{e}|{s}|{c["si"]}' in done])
            eta = np.mean(t_units[-10:]) * left / 3600
            log(f'  eps {eps} seed {seed} speaker {cl["si"] + 1:2d}/{len(CLIPS)}  {t_units[-1] / 60:4.1f} min'
                f'   elapsed {hours():4.1f} h   remaining ~{eta:4.1f} h for all passes')
        try: analyse(quiet=True)                   # a report after every pass, in case Colab stops
        except Exception: log('  (interim analysis failed, continuing)\n' + traceback.format_exc())
        log(f'  pass eps {eps} seed {seed} finished; report updated')

# =============================================================================
#  F. analysis
# =============================================================================
LABEL = {
    'none': 'Unprotected',
    'meas_full': 'Measured, 3 rooms (paper method)',
    'sim3_rt': 'Simulated, 3 rooms, RT60-matched',
    'sim3_rt_dev': 'Simulated, 3 rooms, RT60-matched + device response',
    'sim_proc_200': 'Simulated, same 200, processed',
    'sim_proc_200_dev': 'Simulated, same 200, processed + device response',
}
SHORT = {'none': 'Unprotected', 'meas_full': 'Measured', 'sim3_rt': 'Sim. 3, RT60-matched',
         'sim3_rt_dev': 'Sim. 3, RT60-matched + device', 'sim_proc_200': 'Sim. 200, processed',
         'sim_proc_200_dev': 'Sim. 200, processed + device'}
ORDER = ['none', 'sim_proc_200', 'sim_proc_200_dev', 'sim3_rt', 'sim3_rt_dev', 'meas_full']
PAIRS = [('sim3_rt_dev', 'meas_full', 'KEY: simulated 3 rooms RT60-matched + device vs measured'),
         ('sim3_rt', 'meas_full', 'Reference (as in the paper): simulated 3 rooms RT60-matched vs measured'),
         ('sim_proc_200_dev', 'meas_full', 'Simulated 200 processed + device vs measured'),
         ('sim_proc_200', 'meas_full', 'Reference (as in the paper): simulated 200 processed vs measured'),
         ('sim3_rt', 'sim3_rt_dev', 'Effect of adding the device response (3 RT60-matched rooms)'),
         ('sim_proc_200', 'sim_proc_200_dev', 'Effect of adding the device response (200 rooms)')]

def analyse(quiet=False):
    import pandas as pd
    from scipy import stats
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not os.path.exists(ROWS):
        log('F. Analysis: no results yet'); return
    df = pd.read_csv(ROWS)
    for c in ['snr_db', 'cosB', 'cosA', 'succB', 'succA', 'pesq', 'stoi', 'estoi', 'prot_sec']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    A = f'{OUT}/analysis'; os.makedirs(A, exist_ok=True)
    info = json.load(open(f'{OUT}/setup_info.json')) if os.path.exists(f'{OUT}/setup_info.json') else {}
    tcrit = lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else float('nan')
    out = {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'by_eps': {}}

    def summary(d, cond):                                # as in the overnight analysis
        x = d[d.cond == cond]
        if x.empty: return None
        pt = x.groupby(['clip', 'fold'])[['cosB', 'cosA', 'succB', 'snr_db', 'pesq', 'stoi', 'estoi']].mean()
        ps = x.groupby('seed')[['cosB', 'succB']].mean()
        n, ns = len(pt), x.seed.nunique()
        sd_tr = pt.cosB.std(ddof=1) if n > 1 else float('nan')
        return dict(cond=cond, n_trials=n, n_seeds=ns, cosB=pt.cosB.mean(),
                    cosB_sd_seeds=ps.cosB.std(ddof=1) if ns > 1 else float('nan'),
                    cosB_ci95=tcrit(n) * sd_tr / math.sqrt(n) if n > 1 else float('nan'),
                    succB=100 * ps.succB.mean(), succB_sd_seeds=100 * ps.succB.std(ddof=1) if ns > 1 else float('nan'),
                    cosA=pt.cosA.mean(), snr=pt.snr_db.mean(), pesq=pt.pesq.mean(), stoi=pt.stoi.mean(),
                    estoi=pt.estoi.mean(), seed_means=ps.cosB.round(4).tolist())

    def contrast(d, other, base):                        # as in the overnight analysis
        a = d[d.cond == other].groupby(['seed', 'clip', 'fold']).cosB.mean()
        b = d[d.cond == base].groupby(['seed', 'clip', 'fold']).cosB.mean()
        j = pd.concat([a.rename('o'), b.rename('b')], axis=1).dropna()
        if len(j) < 2: return None
        j['diff'] = j.o - j.b                       # > 0: `base` gives the LOWER similarity (stronger)
        per_trial = j.groupby(level=['clip', 'fold'])['diff'].mean()
        n = len(per_trial); m = per_trial.mean(); sd = per_trial.std(ddof=1)
        ci = tcrit(n) * sd / math.sqrt(n) if n > 1 else float('nan')
        try: p_t = float(stats.ttest_1samp(per_trial, 0.0).pvalue)
        except Exception: p_t = float('nan')
        try: p_w = float(stats.wilcoxon(per_trial).pvalue)
        except Exception: p_w = float('nan')
        rng = np.random.default_rng(0); v = per_trial.values
        boot = rng.choice(v, size=(10000, len(v)), replace=True).mean(1)
        per_room = per_trial.groupby(level='fold').mean()
        per_seed = j.groupby(level='seed')['diff'].mean()
        per_spk = per_trial.groupby(level='clip').mean()
        try: p_spk = float(stats.ttest_1samp(per_spk, 0.0).pvalue) if len(per_spk) > 1 else float('nan')
        except Exception: p_spk = float('nan')
        return dict(other=other, base=base, n=n, mean=m, ci_lo=m - ci, ci_hi=m + ci,
                    boot_lo=float(np.percentile(boot, 2.5)), boot_hi=float(np.percentile(boot, 97.5)),
                    dz=m / sd if sd > 0 else float('nan'), p_t=p_t, p_wilcoxon=p_w, p_speakers=p_spk, n_speakers=len(per_spk),
                    rooms_base_better=int((per_room > 0).sum()), n_rooms=len(per_room),
                    per_room={k: round(float(x), 4) for k, x in per_room.items()},
                    seeds_base_better=int((per_seed > 0).sum()), n_seeds=len(per_seed),
                    per_seed={int(k): round(float(x), 4) for k, x in per_seed.items()})

    def holm(cs):
        ok = [c for c in cs if c and np.isfinite(c['p_t'])]
        ok.sort(key=lambda c: c['p_t']); m = len(ok); run = 0.0
        for i, c in enumerate(ok):
            run = max(run, min(1.0, (m - i) * c['p_t'])); c['p_holm'] = run
        for c in cs:
            if c and 'p_holm' not in c: c['p_holm'] = float('nan')

    def per_trial(d, cond):
        return d[d.cond == cond].groupby(['seed', 'clip', 'fold']).cosB.mean().groupby(level=['clip', 'fold']).mean()

    def share(d, sim, simdev, base='meas_full'):
        """share of the measured-over-simulated gap removed by adding the device response, with bootstrap CI"""
        j = pd.concat([per_trial(d, sim).rename('s'), per_trial(d, simdev).rename('sd'),
                       per_trial(d, base).rename('m')], axis=1).dropna()
        if len(j) < 3: return None
        g, gd = (j.s - j.m).values, (j.sd - j.m).values
        if abs(g.mean()) < 1e-6: return None
        rng = np.random.default_rng(1); idx = rng.integers(0, len(g), size=(10000, len(g)))
        gm, gdm = g[idx].mean(1), gd[idx].mean(1); ok = np.abs(gm) > 1e-6
        bs = 1 - gdm[ok] / gm[ok]
        return dict(share=float(1 - gd.mean() / g.mean()), lo=float(np.percentile(bs, 2.5)),
                    hi=float(np.percentile(bs, 97.5)), gap_without=float(g.mean()), gap_with=float(gd.mean()), n=len(g))

    f3 = lambda x: '—' if x is None or not np.isfinite(x) else f'{x:.3f}'
    f1 = lambda x: '—' if x is None or not np.isfinite(x) else f'{x:.1f}'
    f2 = lambda x: '—' if x is None or not np.isfinite(x) else f'{x:.2f}'
    fp = lambda x: '—' if x is None or not np.isfinite(x) else (f'{x:.1e}' if x < 1e-3 else f'{x:.3f}')
    pc = lambda x: '—' if x is None or not np.isfinite(x) else f'{100 * x:.0f}%'
    tables = []
    for eps in sorted(df.eps.unique(), reverse=True):
        de = df[df.eps == eps]; out['by_eps'][str(eps)] = {}
        for matched in (0, 1):
            d = de[de.snr_matched == matched]
            if matched == 1: d = pd.concat([de[(de.snr_matched == 0) & (de.cond == 'none')], d])
            conds = [c for c in ORDER if c in set(d.cond)]
            S = [s for s in (summary(d, c) for c in conds) if s]
            C = [contrast(d, o, b) for o, b, _ in PAIRS if o in conds and b in conds]
            C = [c for c in C if c]; holm(C)
            SH = {'sim3_rt': share(d, 'sim3_rt', 'sim3_rt_dev') if {'sim3_rt', 'sim3_rt_dev', 'meas_full'} <= set(conds) else None,
                  'sim_proc_200': share(d, 'sim_proc_200', 'sim_proc_200_dev') if {'sim_proc_200', 'sim_proc_200_dev', 'meas_full'} <= set(conds) else None}
            tag = 'equal_eps' if matched == 0 else 'equal_snr'
            out['by_eps'][str(eps)][tag] = dict(summary=S, contrasts=C, share_explained=SH)
            title = f'eps = {eps}, ' + ('equal budget (same eps)' if matched == 0 else 'equal SNR (each perturbation of a trial scaled to the quietest of the five)')
            t = [f'## {title}', '',
                 '| Condition | trials | runs | cos_B mean | ± SD across runs | 95% CI | attacker succeeds % (± SD runs) | cos_A | SNR dB | PESQ | STOI | ESTOI |',
                 '|---|---|---|---|---|---|---|---|---|---|---|---|']
            for s in S:
                t.append(f'| {LABEL.get(s["cond"], s["cond"])} | {s["n_trials"]} | {s["n_seeds"]} | {f3(s["cosB"])} | '
                         f'{f3(s["cosB_sd_seeds"])} | ±{f3(s["cosB_ci95"])} | {f1(s["succB"])} (±{f1(s["succB_sd_seeds"])}) | '
                         f'{f3(s["cosA"])} | {f1(s["snr"])} | {f2(s["pesq"])} | {f2(s["stoi"])} | {f2(s["estoi"])} |')
            t += ['', 'Paired differences in cos_B (first minus second; **positive = the second protects more**), runs averaged '
                  'within each speaker x room trial:', '',
                  '| First | Second | Δ cos_B | 95% CI | bootstrap CI | d_z | p (t) | p (Holm) | p (Wilcoxon) | p (speakers only) | rooms where second better | runs where second better |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
            for c in C:
                t.append(f'| {LABEL.get(c["other"], c["other"])} | {LABEL.get(c["base"], c["base"])} | {f3(c["mean"])} | '
                         f'[{f3(c["ci_lo"])}, {f3(c["ci_hi"])}] | [{f3(c["boot_lo"])}, {f3(c["boot_hi"])}] | {f2(c["dz"])} | '
                         f'{fp(c["p_t"])} | {fp(c["p_holm"])} | {fp(c["p_wilcoxon"])} | {fp(c["p_speakers"])} (n={c["n_speakers"]}) | '
                         f'{c["rooms_base_better"]}/{c["n_rooms"]} | {c["seeds_base_better"]}/{c["n_seeds"]} |')
            for k, sh in SH.items():
                if sh: t.append(f'\nShare of the measured advantage over {LABEL[k].lower()} removed by adding the device response: '
                                f'**{pc(sh["share"])}** (95% bootstrap CI {pc(sh["lo"])} to {pc(sh["hi"])}; gap {f3(sh["gap_without"])} '
                                f'without, {f3(sh["gap_with"])} with the device response).')
            tables += t + ['']

    # ---------------- verdicts, by the rule fixed before the run ----------------
    def get(eps, tag, other, base):
        C = out['by_eps'].get(str(eps), {}).get(tag, {}).get('contrasts', [])
        C = [c for c in C if c['other'] == other and c['base'] == base]
        return C[0] if C else None
    verdict = ['## What the results say (rule fixed before the run)', '']
    for eps in sorted(df.eps.unique(), reverse=True):
        for sim, simdev, nm in (('sim3_rt', 'sim3_rt_dev', '3 RT60-matched simulated rooms'),
                                ('sim_proc_200', 'sim_proc_200_dev', '200 processed simulated rooms')):
            cs = [get(eps, tag, simdev, 'meas_full') for tag in ('equal_eps', 'equal_snr')]
            if None in cs: continue
            key = '**KEY TEST** ' if sim == 'sim3_rt' else ''
            nums = '; '.join(f'{lab}: Δ = {f3(c["mean"])} [95% CI {f3(c["ci_lo"])}, {f3(c["ci_hi"])}], speaker-level p = '
                             f'{fp(c["p_speakers"])}, measured better in {c["rooms_base_better"]}/{c["n_rooms"]} rooms and '
                             f'{c["seeds_base_better"]}/{c["n_seeds"]} runs'
                             for lab, c in zip(('equal budget', 'equal SNR'), cs))
            if all(c['ci_lo'] > 0 for c in cs):
                v = (f'measured rooms still protect better than {nm} WITH the device response added. The advantage is not '
                     f'explained by the loudspeaker and microphone alone.')
            elif all(c['ci_hi'] < 0 for c in cs):
                v = f'{nm} WITH the device response protect better than measured rooms.'
            else:
                v = (f'once the device response is added, measured rooms and {nm} are not distinguishable. The measured '
                     f'advantage is explained by the shared loudspeaker and microphone response.')
            verdict.append(f'- eps {eps}. {key}With {nm}: {v} ({nums}.)')
            for tag, lab in (('equal_eps', 'equal budget'), ('equal_snr', 'equal SNR')):
                sh = out['by_eps'][str(eps)][tag]['share_explained'].get(sim)
                ref = get(eps, tag, sim, 'meas_full'); eff = get(eps, tag, sim, simdev)
                if sh and ref and eff:
                    verdict.append(f'  - {lab}: without the device response the gap is {f3(ref["mean"])} '
                                   f'[{f3(ref["ci_lo"])}, {f3(ref["ci_hi"])}]; adding it changes the simulated rooms by '
                                   f'{f3(eff["mean"])} [{f3(eff["ci_lo"])}, {f3(eff["ci_hi"])}] (positive = better protection) '
                                   f'and removes {pc(sh["share"])} of the gap (95% CI {pc(sh["lo"])} to {pc(sh["hi"])}).')
    verdict += ['', 'Rule: a statement that one set of rooms protects better is made only if the 95% interval excludes 0 '
                'at BOTH equal budget and equal SNR.', '']

    # ---------------- checks ----------------
    chk = ['## Checks', '']
    dv = info.get('device', {})
    for hr, x in dv.items():
        chk.append(f'- Held out {hr}: device response estimated from {", ".join(x.get("train", []))}; it differs from the '
                   f'held-out room by {x.get("heldout_vs_device_dB", float("nan")):.1f} dB on average (250 Hz-7 kHz).')
    dc = info.get('device_check')
    if dc:
        chk.append(f'- RT60-matched simulated rooms differ from their measured room by {dc["sim3_vs_measured_dB_without_device"]:.1f} dB '
                   f'without and {dc["sim3_vs_measured_dB_with_device"]:.1f} dB with the device response (250 Hz-7 kHz): the '
                   f'filter does what it is meant to do if the second number is clearly smaller.')
    try:
        if os.path.exists(MAIN_ROWS):
            m = pd.read_csv(MAIN_ROWS)
            for c in ['cosB']: m[c] = pd.to_numeric(m[c], errors='coerce')
            keep = ['meas_full', 'sim3_rt', 'sim_proc_200']
            x = df[(df.snr_matched == 0) & df.cond.isin(keep)]
            y = m[(m.snr_matched == 0) & m.cond.isin(keep)]
            j = x.merge(y, on=['eps', 'seed', 'clip', 'fold', 'cond'], suffixes=('', '_main'))
            if len(j):
                same_spk = bool((j.spk.astype(str) == j.spk_main.astype(str)).all())
                parts = []
                for c in keep:
                    jj = j[j.cond == c]
                    if len(jj): parts.append(f'{SHORT[c]}: {len(jj)} trials, mean |difference| {(jj.cosB - jj.cosB_main).abs().mean():.4f}, '
                                             f'largest {(jj.cosB - jj.cosB_main).abs().max():.4f}')
                chk.append('- Reproduction of the overnight run (same speakers, rooms, runs and random draws; equal budget): '
                           + '; '.join(parts) + f'. Same speakers in every matched row: {same_spk}. Differences of a few '
                           'thousandths come from GPU arithmetic, not from the method.')
            else:
                chk.append('- Reproduction check: no matching rows in the overnight results.')
        else:
            chk.append('- Reproduction check skipped: overnight_v2/results_rows.csv not found.')
    except Exception:
        chk.append('- Reproduction check failed: ' + traceback.format_exc().splitlines()[-1])
    chk.append('')

    # ---------------- figure: paired differences ----------------
    try:
        rowsF = [(o, b) for o, b, _ in PAIRS]
        epss = sorted(df.eps.unique(), reverse=True)
        fig, axs = plt.subplots(1, len(epss), figsize=(4.2 * len(epss) + 2.6, 0.42 * len(rowsF) + 1.4), sharey=True)
        axs = np.atleast_1d(axs); ys = np.arange(len(rowsF))[::-1]
        for ax, eps in zip(axs, epss):
            for y, (o, b) in zip(ys, rowsF):
                for tag, fill, dy in (('equal_eps', True, -0.15), ('equal_snr', False, 0.15)):
                    c = get(eps, tag, o, b)
                    if not c: continue
                    col = '#2E6FA7' if (o, b) == ('sim3_rt_dev', 'meas_full') else '#222222'
                    ax.plot([c['ci_lo'], c['ci_hi']], [y + dy, y + dy], color=col, lw=1.2)
                    ax.plot([c['mean']], [y + dy], 'o', ms=4.5, mfc=col if fill else 'white', mec=col)
            ax.axvline(0, color='#777777', lw=0.8, ls='--'); ax.set_title(f'eps = {eps}', fontsize=9)
            ax.grid(axis='x', alpha=0.3)
            for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
            ax.tick_params(axis='y', length=0)
        axs[0].set_yticks(ys); axs[0].set_yticklabels([f'{SHORT[o]}  vs  {SHORT[b]}' for o, b in rowsF], fontsize=8)
        fig.supxlabel('Δ cos_B = first minus second, 95% CI (filled: equal budget, open: equal SNR); > 0: the second protects more', fontsize=8)
        fig.tight_layout(); fig.savefig(f'{A}/fig_device_contrasts.png', dpi=200, bbox_inches='tight')
        fig.savefig(f'{A}/fig_device_contrasts.pdf', bbox_inches='tight'); plt.close(fig)
    except Exception:
        log('  (contrast figure failed; tables are still written)\n' + traceback.format_exc())

    head = ['# Device control run: results', '',
            f'Generated {out["generated"]} from `device_control/results_rows.csv` ({len(df)} rows). '
            'Every number below is computed from that file.', '']
    open(f'{A}/REPORT.md', 'w').write('\n'.join(head + verdict + chk + tables))
    json.dump(out, open(f'{A}/summary.json', 'w'), indent=1, default=float)
    if not quiet:
        log('\n'.join(verdict))
        log(f'F. Analysis written to {A}')

# =============================================================================
if True:                                         # runs when the cell runs
    try:
        main_loop()
    except Exception:
        log('!! main loop stopped:\n' + traceback.format_exc())
    finally:
        try: save_sim3()
        except Exception: pass
        try: analyse()
        except Exception: log('!! analysis failed:\n' + traceback.format_exc())
        log(f'ALL DONE in {hours():.1f} h. Open {OUT}/analysis/REPORT.md and send me device_control.zip')
        try:                                         # one zip of everything to send back
            z = f'{RES}/device_control.zip'
            with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in glob.glob(f'{OUT}/**/*', recursive=True):
                    if os.path.isfile(p) and '/cache/' not in p: zf.write(p, os.path.relpath(p, RES))
            log(f'  everything to send: {z}')
        except Exception:
            log('  (zip failed; send the device_control folder instead)')
