# =============================================================================
#  PHC-351 · Hiding a Voice · Bhavishay (24324008)
#  OVERNIGHT RUN v2: the extra experiments asked for in the review of 26 Oct
#
#  HOW TO RUN (Google Colab)
#    1. Runtime -> Change runtime type -> T4 GPU -> Save.
#    2. Run this cell. Allow Google Drive access when the pop-up asks.
#    3. Leave the tab open (laptop on charger, sleep turned off).
#    If Colab disconnects, run the cell again: everything already finished is
#    skipped and the run continues where it stopped.
#
#  WHAT IT DOES (in this order)
#    A. Setup: Drive, pesq + pystoi, GPU check, downloads LibriSpeech
#       train-clean-100 and the OpenSLR-28 simulated rooms (as file 16 did).
#    B. Loads YOUR encoders A and B, YOUR four measured rooms and YOUR
#       thresholds from Drive. Nothing is retrained.
#    C. Rebuilds the measured rooms from the sweep recordings with each
#       processing step switched off in turn (the ablations), and checks
#       that the full processing reproduces the saved rooms.
#    D. Prepares the simulated rooms three ways (as in the paper; processed
#       exactly like the measured rooms; 3 rooms matched in RT60).
#    E. Makes the audio for listening test v2 (12 speakers) -> Drive.
#    F. Main loop: the 20 speakers of the main run x 4 held-out rooms, every
#       condition, several random seeds (eps 0.004 first, then 0.002, then
#       extra seeds while time allows). Every result row goes to Drive.
#    G. Analysis: mean +- SD across seeds, 95% CIs, paired tests with Holm
#       correction, per-room results, LaTeX tables, figures, REPORT.md.
#
#  OUTPUT: MyDrive/PHC351_24324008_VoiceProtection/testing_data/results/overnight_v2/
#  Nothing here invents or edits numbers: every number in the report is
#  computed from results_rows.csv, which is written as the runs finish.
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
OUT = f'{RES}/overnight_v2'

EPS_MAIN = 0.004                     # the paper's operating point
N_CLIPS = 20                         # the main run's 20 held-out speakers
K_ITERS = 300                        # PGD iterations, as in the paper
B_EOT = 4                            # channels per iteration, as in the paper
PASSES = [(0.004, 0), (0.004, 1), (0.004, 2),        # in priority order
          (0.002, 0), (0.002, 1), (0.002, 2),
          (0.004, 3), (0.004, 4)]
TIME_BUDGET_H = float(os.environ.get('PHC_HOURS', '10.0'))
SIM_SCAN_PER_SIZE = 1000             # simulated rooms scanned per size for RT60 matching
LT_N_SPK, LT_EPS = 12, 0.004         # listening test v2
TAU_PAPER = {'A': 0.734, 'B': 0.793}
IMPOSTOR_MEAN_B = 0.700

if SMOKE:
    N_CLIPS, K_ITERS = 2, 3
    PASSES = [(0.004, 0), (0.004, 1)]
    SIM_SCAN_PER_SIZE, LT_N_SPK = 6, 4

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
log('=' * 70); log(f'OVERNIGHT RUN v2 started {datetime.datetime.now():%Y-%m-%d %H:%M}  (smoke={SMOKE})')
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
#  A. downloads (after loading, so a missing model stops the run in seconds)
# =============================================================================
log('A. Downloads')
HAVE_LS = get_librispeech()
HAVE_SIM = get_rirs()
log(f'  LibriSpeech: {"ready" if HAVE_LS else "NOT available"}   simulated rooms: {"ready" if HAVE_SIM else "NOT available"}')

# ------------- the speaker split (02_dataset.py) and the main-run clips ------
LS_FILES = sorted(glob.glob(f'{CONTENT}/LibriSpeech/train-clean-100/*/*/*.flac')) if HAVE_LS else []
_spks = sorted({p.split('/')[-3] for p in LS_FILES}); random.Random(0).shuffle(_spks)
_ntr = int(0.8 * len(_spks))
HELD_SPK = sorted(_spks[_ntr:])

def held_from_drive():
    d = {}
    for p in sorted(glob.glob(f'{R}/training_data/heldout_subset/*.flac')):
        d.setdefault(os.path.basename(p).split('_')[0], []).append(p)
    return {k: v for k, v in d.items() if len(v) >= 3}

held = held_from_drive(); HELD_SOURCE = 'Drive training_data/heldout_subset (same files as the main run)'
if len(held) < N_CLIPS:
    d, hs = {}, set(HELD_SPK)
    for p in LS_FILES:
        s = p.split('/')[-3]
        if s in hs: d.setdefault(s, []).append(p)
    held = {k: v for k, v in sorted(d.items()) if len(v) >= 3}
    HELD_SOURCE = 'LibriSpeech held-out split (heldout_subset not found on Drive)'
SPK_RUN = list(held)[:N_CLIPS]
CLIPS = []
for si, s in enumerate(SPK_RUN):                     # same choice of files as 09_main_run.py
    ps = random.Random(si).sample(held[s], 2)
    CLIPS.append(dict(si=si, spk=s, src_file=ps[0], enr_file=ps[1]))
INFO['held_source'] = HELD_SOURCE; INFO['clips'] = CLIPS
log(f'  clips: {len(CLIPS)} speakers from {HELD_SOURCE}')
if not CLIPS: log('  !! no held-out speakers found: the main loop will be skipped')

# =============================================================================
#  C. measured rooms rebuilt from the sweeps, with each step switched off
# =============================================================================
log('C. Rebuilding the measured rooms from the sweep recordings (ablations)')
VARIANTS = {}           # variant -> {room: response}
ABL_OK, ABL_BASE = False, 'meas_full'
try:
    have_sw = all(os.path.exists(f'{REC}/sweep_{r}.wav') for r in ROOMS) and os.path.exists(f'{REC}/sweep_reference.wav')
    if not have_sw:
        log('  !! sweep recordings not all found in testing_data/recordings: ablations skipped')
    else:
        ref_pad = load16k(f'{REC}/sweep_reference.wav')
        ref_bare = ref_pad[SR: SR + 10 * SR]
        recs = {r: load16k(f'{REC}/sweep_{r}.wav') for r in ROOMS}
        best = None
        for nm, ref in (('padded reference', ref_pad), ('bare sweep', ref_bare)):
            full = {}
            for r in ROOMS:
                tr, _ = trim_to_sweep(recs[r], ref)
                full[r] = post_process(deconv_raw(tr, ref, True, True), True, True)
            score = min(best_xcorr(full[r], RIR_MEAS[r]) for r in ROOMS)
            log(f'  full processing with {nm}: similarity to the saved rooms (worst room) = {score:.4f}')
            if best is None or score > best[0]: best = (score, nm, ref)
        score, nm, ref = best
        INFO['reproduction'] = {'reference': nm, 'worst_room_similarity': score}
        spec = {'abl_full':       dict(freq_reg=True,  bandpass=True,  prenorm=True,  blend=True),
                'abl_no_freqreg': dict(freq_reg=False, bandpass=True,  prenorm=True,  blend=True),
                'abl_no_blend':   dict(freq_reg=True,  bandpass=True,  prenorm=True,  blend=False),
                'abl_no_prenorm': dict(freq_reg=True,  bandpass=True,  prenorm=False, blend=True),
                'abl_naive':      dict(freq_reg=False, bandpass=False, prenorm=True,  blend=False)}
        for v, o in spec.items():
            VARIANTS[v] = {}
            for r in ROOMS:
                tr, _ = trim_to_sweep(recs[r], ref)
                VARIANTS[v][r] = post_process(deconv_raw(tr, ref, o['freq_reg'], o['bandpass']), o['prenorm'], o['blend'])
        ABL_OK = True
        if score >= 0.99:
            ABL_BASE = 'meas_full'; del VARIANTS['abl_full']
            log('  reproduced the saved rooms: ablations are compared with the paper method (meas_full)')
        else:
            ABL_BASE = 'abl_full'
            log('  !! not an exact reproduction: ablations are compared with the rebuilt full version (abl_full)')
        INFO['ablation_rt60'] = {v: {r: rt60(VARIANTS[v][r]) for r in ROOMS} for v in VARIANTS}
except Exception:
    log('  !! ablation rebuild failed, ablations skipped:\n' + traceback.format_exc())
    VARIANTS, ABL_OK = {}, False
INFO['ablation_base'] = ABL_BASE

# =============================================================================
#  D. simulated rooms
# =============================================================================
log('D. Simulated rooms')
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

json.dump(INFO, open(f'{OUT}/setup_info.json', 'w'), indent=1, default=str)

# =============================================================================
#  E. listening test v2 stimuli
# =============================================================================
def median_f0(x):
    x = x.numpy().astype(np.float64); fr, hp = 640, 320
    rms = np.array([x[i:i + fr].std() for i in range(0, len(x) - fr, hp)])
    if len(rms) == 0: return float('nan')
    thr = 0.5 * np.median(rms); f0s = []
    for j, i in enumerate(range(0, len(x) - fr, hp)):
        if rms[j] < thr: continue
        seg = x[i:i + fr] - x[i:i + fr].mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(seg, 2 * fr)) ** 2)[:fr]
        lo, hi = SR // 400, SR // 60
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[lag] / ac[0] > 0.3: f0s.append(SR / lag)
    return float(np.median(f0s)) if f0s else float('nan')

def clean_json(o):
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean_json(v) for v in o]
    return o

def fade(x, ms=20):
    n = int(SR * ms / 1000); x = x.clone()
    r = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, n)))
    x[:n] *= r; x[-n:] *= r.flip(0)
    return x

def build_listening_test():
    LTD = f'{OUT}/listening_test_v2'; AUD = f'{LTD}/audio'
    if os.path.exists(f'{LTD}/listening_test_v2_stimuli.zip'):
        log('E. Listening test v2: already made, skipped'); return
    if not HAVE_LS:
        log('E. Listening test v2: needs LibriSpeech transcripts, which did not download: skipped'); return
    log('E. Listening test v2: choosing speakers and making the audio')
    os.makedirs(AUD, exist_ok=True)
    LSD = f'{CONTENT}/LibriSpeech'
    sex = {}
    if os.path.exists(f'{LSD}/SPEAKERS.TXT'):
        for line in open(f'{LSD}/SPEAKERS.TXT', encoding='utf-8', errors='ignore'):
            if line.startswith(';'): continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2 and parts[1] in ('F', 'M'): sex[parts[0]] = parts[1]
    cand = []
    for s in HELD_SPK:
        texts = {}
        for t in glob.glob(f'{LSD}/train-clean-100/{s}/*/*.trans.txt'):
            for line in open(t):
                k, _, txt = line.strip().partition(' ')
                texts[k] = txt
        files = sorted(glob.glob(f'{LSD}/train-clean-100/{s}/*/*.flac'))
        dur = {}
        for p in files:
            try: i = sf.info(p); dur[p] = i.frames / i.samplerate
            except Exception: pass
        uid = lambda p: os.path.basename(p)[:-5]
        intel = [p for p in files if p in dur and 2.5 <= dur[p] <= 6.0 and uid(p) in texts
                 and 6 <= len(texts[uid(p)].split()) <= 18]
        if not intel: continue
        intel = sorted(intel, key=lambda p: (abs(dur[p] - 4.0), p))[0]
        longs = [p for p in files if p in dur and dur[p] >= 4.2 and p != intel]
        if len(longs) < 2: continue
        cand.append(dict(libri=s, gender=sex.get(s, '?'), ref=longs[0], test=longs[1],
                         intel=intel, transcript=texts[uid(intel)]))
    rng = random.Random(7)
    fem = [c for c in cand if c['gender'] == 'F']; mal = [c for c in cand if c['gender'] == 'M']
    rest = [c for c in cand if c['gender'] not in ('F', 'M')]
    rng.shuffle(fem); rng.shuffle(mal); rng.shuffle(rest)
    half = LT_N_SPK // 2
    chosen = fem[:half] + mal[:half]
    pool_left = fem[half:] + mal[half:] + rest
    while len(chosen) < LT_N_SPK and pool_left: chosen.append(pool_left.pop(0))
    calib = pool_left.pop(0) if pool_left else None
    if len(chosen) < 4:
        log('  !! fewer than 4 suitable speakers: listening test skipped'); return
    pool4 = list(RIR_MEAS.values())
    seg = lambda p: load16k(p)[int(0.2 * SR): int(0.2 * SR) + int(3.5 * SR)]
    lvl = lambda x: x / x.std().clamp_min(1e-6) * 0.05
    audio, metrics, spk_meta = {}, [], []
    for n, c in enumerate(chosen):
        S = f'S{n + 1:02d}'
        ref, test, itl = lvl(seg(c['ref'])), lvl(seg(c['test'])), lvl(load16k(c['intel']))
        w_test = protect(test, LT_EPS, Pool(pool4, test.shape[0]), random.Random(crc('lt-test' + S)))
        w_itl = protect(itl, LT_EPS, Pool(pool4, itl.shape[0]), random.Random(crc('lt-intel' + S)))
        for nm, src, w in (('test', test, w_test), ('int', itl, w_itl)):
            st, est = stoi_pair(src, w)
            metrics.append(dict(item=f'{S}_{nm}', speaker=S, snr_db=snr_db(src, w - src), pesq_wb=pesq_wb(src, w),
                                stoi=st, estoi=est))
        audio[f'{S}_ref'] = ref; audio[f'{S}_test_clean'] = test; audio[f'{S}_test_prot'] = w_test
        audio[f'{S}_int_clean'] = itl; audio[f'{S}_int_prot'] = w_itl
        f0 = median_f0(torch.cat([ref, test]))
        spk_meta.append(dict(id=S, libri=c['libri'], gender=c['gender'], f0=round(f0, 1) if np.isfinite(f0) else None,
                             transcript=c['transcript'], src=dict(ref=os.path.basename(c['ref']),
                             test=os.path.basename(c['test']), intel=os.path.basename(c['intel']))))
        log(f'  {S} ({c["gender"]}) done   SNR {metrics[-2]["snr_db"]:.1f} dB  PESQ {metrics[-2]["pesq_wb"]:.2f}')
    # partner for "different speaker" trials: same gender, closest pitch
    def f0v(m): return m['f0'] if m['f0'] is not None else 150.0
    for m in spk_meta:
        others = [o for o in spk_meta if o['id'] != m['id'] and (o['gender'] == m['gender'] or m['gender'] == '?')]
        if not others: others = [o for o in spk_meta if o['id'] != m['id']]
        m['partner'] = min(others, key=lambda o: abs(f0v(o) - f0v(m)))['id']
    # attention checks: opposite gender, largest pitch difference
    att, used = [], set()
    for m in sorted(spk_meta, key=lambda m: -f0v(m)):
        if len(att) >= 3: break
        opp = [o for o in spk_meta if o['gender'] != m['gender'] and o['id'] not in used and m['id'] not in used]
        if not opp: continue
        o = max(opp, key=lambda o: abs(f0v(o) - f0v(m)))
        att.append((m['id'], o['id'])); used |= {m['id'], o['id']}
    while len(att) < 3:
        a, b = spk_meta[len(att) % len(spk_meta)]['id'], spk_meta[(len(att) + len(spk_meta) // 2) % len(spk_meta)]['id']
        att.append((a, b))
    # quality anchors: low-pass 3.5 kHz + noise at 10 dB SNR
    from scipy.signal import butter, sosfiltfilt
    sos = butter(8, 3500, fs=SR, output='sos')
    for j, m in enumerate([spk_meta[1 % len(spk_meta)], spk_meta[6 % len(spk_meta)]]):
        x = audio[f'{m["id"]}_test_clean'].numpy().astype(np.float64)
        lp = sosfiltfilt(sos, x); g = np.random.default_rng(j).standard_normal(len(lp))
        a = lp + g * lp.std() / 10 ** (10 / 20)
        audio[f'anchor_{j + 1}'] = lvl(torch.from_numpy(a.astype(np.float32)))
    if calib is not None:
        audio['calib_ref'] = lvl(seg(calib['ref'])); audio['calib_test'] = lvl(seg(calib['test']))
        audio['calib_int'] = lvl(load16k(calib['intel'])); calib_text = calib['transcript']
    else:
        audio['calib_ref'] = audio['S01_ref']; audio['calib_test'] = audio['S01_test_clean']
        audio['calib_int'] = audio['S01_int_clean']; calib_text = spk_meta[0]['transcript']
    peak = max(float(v.abs().max()) for v in audio.values())
    g = min(1.0, 0.97 / max(peak, 1e-9))
    for k in audio: save_wav(f'{AUD}/{k}.wav', fade(audio[k] * g))
    A = lambda k: f'audio/{k}.wav'
    ident = []
    for m in spk_meta:
        S, P = m['id'], m['partner']
        ident += [dict(id=f'{S}_clean_same', ref=A(f'{S}_ref'), test=A(f'{S}_test_clean'), truth='same', cond='clean_same', spk=S),
                  dict(id=f'{S}_prot_same', ref=A(f'{S}_ref'), test=A(f'{S}_test_prot'), truth='same', cond='prot_same', spk=S),
                  dict(id=f'{S}_diff', ref=A(f'{S}_ref'), test=A(f'{P}_test_clean'), truth='different', cond='diff', spk=S, other=P)]
    for j, (a, b) in enumerate(att):
        ident.append(dict(id=f'attention_{j + 1}', ref=A(f'{a}_ref'), test=A(f'{b}_test_clean'), truth='different',
                          cond='attention', spk=a, other=b))
    quality = []
    for m in spk_meta:
        quality += [dict(id=f'{m["id"]}_q_clean', file=A(f'{m["id"]}_test_clean'), cond='clean', spk=m['id']),
                    dict(id=f'{m["id"]}_q_prot', file=A(f'{m["id"]}_test_prot'), cond='prot', spk=m['id'])]
    quality += [dict(id='anchor_1', file=A('anchor_1'), cond='anchor', spk=None),
                dict(id='anchor_2', file=A('anchor_2'), cond='anchor', spk=None)]
    intel = [dict(spk=m['id'], clean=A(f'{m["id"]}_int_clean'), prot=A(f'{m["id"]}_int_prot'),
                  transcript=m['transcript']) for m in spk_meta]
    manifest = dict(version='v2', created=datetime.datetime.now().isoformat(timespec='seconds'),
                    eps=LT_EPS, protection='EOT over all four measured rooms, K=%d, B=%d, level 0.05 RMS' % (K_ITERS, B_EOT),
                    sr=SR, speakers=spk_meta, identity_trials=ident, quality_items=quality,
                    intelligibility_items=intel,
                    calibration=dict(ref=A('calib_ref'), test=A('calib_test'), intel=A('calib_int'), transcript=calib_text),
                    objective=metrics)
    json.dump(clean_json(manifest), open(f'{LTD}/manifest.json', 'w'), indent=1, allow_nan=False)
    with open(f'{LTD}/stimuli_metrics.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(metrics[0])); wr.writeheader(); wr.writerows(metrics)
    tmpz = f'{CONTENT}/lt_stimuli.zip'
    with zipfile.ZipFile(tmpz, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(f'{LTD}/manifest.json', 'manifest.json')
        for p in sorted(glob.glob(f'{AUD}/*.wav')): z.write(p, f'audio/{os.path.basename(p)}')
    shutil.copy(tmpz, f'{LTD}/listening_test_v2_stimuli.zip')
    log(f'  saved {len(audio)} audio files + manifest -> {LTD}/listening_test_v2_stimuli.zip')

try:
    build_listening_test()
except Exception:
    log('  !! listening-test audio failed (main experiments continue):\n' + traceback.format_exc())

# =============================================================================
#  F. main loop
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
def pool_for(name, rirs):
    if name not in POOLS: POOLS[name] = Pool(rirs, CHUNK)
    return POOLS[name]

def conditions_for(seed, hr):
    """Every protected condition for held-out room hr: name -> (pool, outnorm)."""
    train = [r for r in ROOMS if r != hr]
    c = {'meas_full': (pool_for(f'meas|{hr}', [RIR_MEAS[r] for r in train]), True)}
    if SIM_SCAN:
        ch = sim3_for(seed, hr)
        c['sim3_rt'] = (pool_for(f'sim3|{seed}|{hr}', [SIM_SCAN[x['i']][1] for x in ch]), True)
    if ABL_OK:
        for v in VARIANTS:
            c[v] = (pool_for(f'{v}|{hr}', [VARIANTS[v][r] for r in train]), True)
        c['abl_no_outnorm'] = (c['meas_full'][0], False)
    return c

def run_unit(eps, seed, cl):
    src, enr = fit(load16k(cl['src_file'])), fit(load16k(cl['enr_file']))
    si = cl['si']; rows = []
    def rng(cond, hr=''): return random.Random(crc(f'{eps}|{seed}|{si}|{cond}|{hr}'))
    def timed(cond, pool, hr='', outnorm=True):
        t = time.time(); w = protect(src, eps, pool, rng(cond, hr), outnorm=outnorm)
        return w, time.time() - t
    shared = {}
    pc = f'{OUT}/cache/pgd_eps{eps}_clip{si}.pt'
    if os.path.exists(pc): shared['pgd'] = (torch_load(pc), float('nan'))
    else:
        shared['pgd'] = timed('pgd', None); torch.save(shared['pgd'][0], pc)
    if SIM_RAW: shared['sim_raw_200'] = timed('sim_raw_200', pool_for('sim_raw', [h for _, h in SIM_RAW]))
    if SIM_PROC: shared['sim_proc_200'] = timed('sim_proc_200', pool_for('sim_proc', [h for _, h in SIM_PROC]))
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
        for name, (pool, on) in conditions_for(seed, hr).items():
            prot[name] = timed(name, pool, hr, on)
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
        # equal-SNR evaluation: every perturbation scaled down to the quietest one's SNR
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
    log(f'F. Main loop: {total} units ({len(PASSES)} passes x {len(CLIPS)} speakers), {n0} already done')
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
    json.dump(SIM3_CHOICES, open(f'{OUT}/sim3_choices.json', 'w'), indent=1)

# =============================================================================
#  G. analysis
# =============================================================================
LABEL = {
    'none': 'Unprotected',
    'pgd': 'PGD, no channel',
    'sim_raw_200': 'Simulated, as in the paper (200, raw, 0.5 s)',
    'sim_proc_200': 'Simulated, same 200, processed like measured',
    'sim3_rt': 'Simulated, 3 rooms, RT60-matched, processed',
    'meas_full': 'Measured, 3 rooms (paper method)',
    'abl_full': 'Measured, rebuilt from sweeps',
    'abl_no_freqreg': 'Ablation: no frequency-dependent regularisation',
    'abl_no_blend': 'Ablation: no out-of-band blend',
    'abl_no_prenorm': 'Ablation: no normalisation before the blend',
    'abl_no_outnorm': 'Ablation: no output-level normalisation',
    'abl_naive': 'Ablation: no processing at all',
}
SHORT = {'none': 'Unprotected', 'pgd': 'PGD (no channel)', 'sim_raw_200': 'Sim., as in paper',
         'sim_proc_200': 'Sim. 200, processed', 'sim3_rt': 'Sim. 3, RT60-matched', 'meas_full': 'Measured (paper)',
         'abl_full': 'Measured, rebuilt', 'abl_no_freqreg': 'No freq.-dep. reg.', 'abl_no_blend': 'No blend',
         'abl_no_prenorm': 'No pre-blend norm.', 'abl_no_outnorm': 'No output norm.', 'abl_naive': 'No processing'}
ORDER = ['none', 'pgd', 'sim_raw_200', 'sim_proc_200', 'sim3_rt', 'meas_full', 'abl_full',
         'abl_no_freqreg', 'abl_no_blend', 'abl_no_prenorm', 'abl_no_outnorm', 'abl_naive']

def analyse(quiet=False):
    import pandas as pd
    from scipy import stats
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not os.path.exists(ROWS):
        log('G. Analysis: no results yet'); return
    df = pd.read_csv(ROWS)
    for c in ['snr_db', 'cosB', 'cosA', 'succB', 'succA', 'pesq', 'stoi', 'estoi', 'prot_sec']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    A = f'{OUT}/analysis'; os.makedirs(A, exist_ok=True)
    info = json.load(open(f'{OUT}/setup_info.json')) if os.path.exists(f'{OUT}/setup_info.json') else {}
    abl_base = info.get('ablation_base', 'meas_full')
    tauB = info.get('tau', TAU_PAPER)['B'] if info else TAU_PAPER['B']
    tcrit = lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else float('nan')
    out = {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'by_eps': {}}
    md_lines = ['# Overnight run v2: results', '',
                f'Generated {out["generated"]} from `results_rows.csv` ({len(df)} rows). '
                'Every number below is computed from that file.', '']

    def summary(d, cond):
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

    def contrast(d, other, base):
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

    f3 = lambda x: '—' if x is None or not np.isfinite(x) else f'{x:.3f}'
    f1 = lambda x: '—' if x is None or not np.isfinite(x) else f'{x:.1f}'
    f2 = lambda x: '—' if x is None or not np.isfinite(x) else f'{x:.2f}'
    fp = lambda x: '—' if x is None or not np.isfinite(x) else (f'{x:.1e}' if x < 1e-3 else f'{x:.3f}')
    tex = ['% Tables generated by phc351_overnight.py from results_rows.csv. Paste into Overleaf.', '']

    for eps in sorted(df.eps.unique(), reverse=True):
        de = df[df.eps == eps]; out['by_eps'][str(eps)] = {}
        for matched in (0, 1):
            d = de[de.snr_matched == matched]
            if matched == 1: d = pd.concat([de[(de.snr_matched == 0) & (de.cond == 'none')], d])
            conds = [c for c in ORDER if c in set(d.cond)]
            S = [summary(d, c) for c in conds]; S = [s for s in S if s]
            contrasts = [contrast(d, o, 'meas_full') for o in ['sim3_rt', 'sim_proc_200', 'sim_raw_200', 'pgd'] if o in conds]
            contrasts += [contrast(d, 'sim_raw_200', 'sim_proc_200') if {'sim_raw_200', 'sim_proc_200'} <= set(conds) else None,
                          contrast(d, 'sim_proc_200', 'sim3_rt') if {'sim_proc_200', 'sim3_rt'} <= set(conds) else None]
            contrasts += [contrast(d, v, abl_base) for v in ORDER if v.startswith('abl_') and v != 'abl_full' and v in conds]
            contrasts = [c for c in contrasts if c]; holm(contrasts)
            tag = 'equal_eps' if matched == 0 else 'equal_snr'
            out['by_eps'][str(eps)][tag] = dict(summary=S, contrasts=contrasts)
            title = (f'eps = {eps}, ' + ('equal budget (same eps)' if matched == 0 else
                                         'equal SNR (each perturbation scaled to the quietest one)'))
            md_lines += [f'## {title}', '',
                         '| Condition | trials | seeds | cos_B mean | ± SD across seeds | 95% CI | attacker succeeds % (± SD seeds) | cos_A | SNR dB | PESQ | STOI | ESTOI |',
                         '|---|---|---|---|---|---|---|---|---|---|---|---|']
            for s in S:
                md_lines.append(f'| {LABEL.get(s["cond"], s["cond"])} | {s["n_trials"]} | {s["n_seeds"]} | {f3(s["cosB"])} | '
                                f'{f3(s["cosB_sd_seeds"])} | ±{f3(s["cosB_ci95"])} | {f1(s["succB"])} (±{f1(s["succB_sd_seeds"])}) | '
                                f'{f3(s["cosA"])} | {f1(s["snr"])} | {f2(s["pesq"])} | {f2(s["stoi"])} | {f2(s["estoi"])} |')
            md_lines += ['', 'Paired differences in cos_B (other condition minus base; **positive = base protects more**), '
                         'averaged over seeds within each speaker x room trial:', '',
                         '| Other | Base | Δ cos_B | 95% CI | bootstrap CI | d_z | p (t) | p (Holm) | p (Wilcoxon) | p (speakers only) | rooms where base better | seeds where base better |',
                         '|---|---|---|---|---|---|---|---|---|---|---|---|']
            for c in contrasts:
                md_lines.append(f'| {LABEL.get(c["other"], c["other"])} | {LABEL.get(c["base"], c["base"])} | {f3(c["mean"])} | '
                                f'[{f3(c["ci_lo"])}, {f3(c["ci_hi"])}] | [{f3(c["boot_lo"])}, {f3(c["boot_hi"])}] | {f2(c["dz"])} | '
                                f'{fp(c["p_t"])} | {fp(c["p_holm"])} | {fp(c["p_wilcoxon"])} | {fp(c["p_speakers"])} (n={c["n_speakers"]}) | {c["rooms_base_better"]}/{c["n_rooms"]} | '
                                f'{c["seeds_base_better"]}/{c["n_seeds"]} |')
            md_lines.append('')
            # LaTeX
            tex += [f'% ---- eps = {eps}, {tag} ----', r'\begin{table}[t]', r'\centering',
                    r'\caption{' + ('Matched comparison' if matched == 0 else 'Matched comparison at equal SNR') +
                    f', $\\eps={eps}$, mean over held-out speakers $\\times$ rooms; $\\pm$ is the SD across '
                    r'independent optimisation runs (seeds).}', r'\footnotesize',
                    r'\begin{tabular}{lccc}', r'\toprule', r'Training channels & $\cosB$ & Attacker succeeds & SNR (dB)\\', r'\midrule']
            for s in S:
                sd = f'\\pm{s["cosB_sd_seeds"]:.3f}' if np.isfinite(s['cosB_sd_seeds']) else ''
                sdp = f'\\pm{s["succB_sd_seeds"]:.0f}' if np.isfinite(s['succB_sd_seeds']) else ''
                tex.append(f'{LABEL.get(s["cond"], s["cond"])} & ${s["cosB"]:.3f}{sd}$ & ${s["succB"]:.0f}{sdp}\\%$ & {f1(s["snr"])}\\\\')
            tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']

        # figures for this eps (equal budget): dot = mean, whisker = +- SD across seeds
        try:
            d = de[de.snr_matched == 0]
            S = out['by_eps'][str(eps)]['equal_eps']['summary']
            if S:
                BLUE, GREY, INK, MUTED, GRID = '#2a78d6', '#8f8e89', '#0b0b0b', '#52514e', '#e6e5e1'
                plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                                     'xtick.color': MUTED, 'ytick.color': INK})
                def style(ax):
                    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
                    ax.tick_params(axis='y', length=0); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
                def dots(ax, rows, key='cosB', sdkey='cosB_sd_seeds', seeds='seed_means'):
                    ys = np.arange(len(rows))[::-1]
                    for y, s in zip(ys, rows):
                        col = BLUE if s['cond'] == 'meas_full' else GREY
                        for v in s.get(seeds, []) or []: ax.plot(v, y, 'o', ms=3, color=col, alpha=0.35, zorder=2)
                        sd = s.get(sdkey, float('nan'))
                        if np.isfinite(sd): ax.errorbar(s[key], y, xerr=sd, fmt='none', ecolor=col, elinewidth=1.6, capsize=3, zorder=3)
                        if np.isfinite(s[key]): ax.plot(s[key], y, 'o', ms=7, color=col, zorder=4)
                    ax.set_yticks(ys); ax.set_yticklabels([SHORT.get(s['cond'], s['cond']) for s in rows])
                    ax.set_ylim(-0.8, len(rows) - 0.2)
                vals = [s['cosB'] for s in S] + [v for s in S for v in s['seed_means']] + [tauB, IMPOSTOR_MEAN_B]
                lo, hi = min(vals) - 0.03, max(vals) + 0.03
                fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(S) + 1.4))
                ax.axvline(tauB, color=INK, lw=1, ls='--', zorder=1)
                ax.axvline(IMPOSTOR_MEAN_B, color=MUTED, lw=1, ls=':', zorder=1)
                dots(ax, S); ax.set_xlim(lo, hi); style(ax)
                ax.text(tauB, len(S) - 0.25, ' τ_B (threshold)', color=INK, fontsize=7.5, va='bottom', ha='left', clip_on=False)
                ax.text(IMPOSTOR_MEAN_B, len(S) - 0.25, 'impostor mean ', color=MUTED, fontsize=7.5, va='bottom', ha='right', clip_on=False)
                ax.set_xlabel("cos_B to the victim's enrolment (lower = stronger protection)")
                ax.set_title(f'eps = {eps}: dot = mean over speakers x held-out rooms, whisker = ± SD across seeds, '
                             'small dots = each seed', fontsize=8, color=INK, loc='left', pad=14)
                fig.savefig(f'{A}/fig_comparison_eps{eps}.png', dpi=200, bbox_inches='tight')
                fig.savefig(f'{A}/fig_comparison_eps{eps}.pdf', bbox_inches='tight'); plt.close(fig)
                # forest plot of the paired differences
                for tag in ('equal_eps', 'equal_snr'):
                    C = out['by_eps'][str(eps)].get(tag, {}).get('contrasts', [])
                    if not C: continue
                    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(C) + 1.3))
                    ys = np.arange(len(C))[::-1]
                    ax.axvline(0, color=MUTED, lw=1, zorder=1)
                    for y, c in zip(ys, C):
                        col = BLUE if c['other'] == 'sim3_rt' and c['base'] == 'meas_full' else GREY
                        ax.plot([c['ci_lo'], c['ci_hi']], [y, y], color=col, lw=1.6, zorder=2)
                        ax.plot(c['mean'], y, 'o', ms=7, color=col, zorder=3)
                    ax.set_yticks(ys)
                    ax.set_yticklabels([f'{SHORT.get(c["other"], c["other"])}  vs  {SHORT.get(c["base"], c["base"])}' for c in C], fontsize=7.5)
                    style(ax); ax.set_ylim(-0.8, len(C) - 0.2)
                    ax.set_xlabel('Δ cos_B = first minus second, with 95% CI   (> 0: the second protects more)')
                    ax.set_title(f'eps = {eps}, ' + ('equal budget' if tag == 'equal_eps' else 'equal SNR') +
                                 ': paired differences over speakers x rooms (seeds averaged)', fontsize=8, loc='left', color=INK)
                    fig.savefig(f'{A}/fig_contrasts_eps{eps}_{tag}.png', dpi=200, bbox_inches='tight')
                    fig.savefig(f'{A}/fig_contrasts_eps{eps}_{tag}.pdf', bbox_inches='tight'); plt.close(fig)
                # one small panel per held-out room
                key = [c for c in ['pgd', 'sim_raw_200', 'sim_proc_200', 'sim3_rt', 'meas_full'] if c in set(d.cond)]
                rooms = [r for r in (info.get('rooms') or sorted(set(d.fold))) if r in set(d.fold)]
                fig, axs = plt.subplots(1, len(rooms), figsize=(1.9 * len(rooms) + 1.6, 0.36 * len(key) + 1.5), sharey=True)
                axs = np.atleast_1d(axs); allv = []
                for ax, room in zip(axs, rooms):
                    dr = d[d.fold == room]; rows = []
                    for c in key:
                        ps = dr[dr.cond == c].groupby('seed').cosB.mean()
                        rows.append(dict(cond=c, cosB=ps.mean(), cosB_sd_seeds=ps.std(ddof=1) if len(ps) > 1 else float('nan'),
                                         seed_means=ps.tolist())); allv += ps.tolist()
                    ax.axvline(tauB, color=INK, lw=0.8, ls='--', zorder=1)
                    dots(ax, rows); style(ax); ax.set_title(room, fontsize=8, color=INK)
                if allv:
                    for ax in axs: ax.set_xlim(min(allv + [tauB]) - 0.03, max(allv + [tauB]) + 0.03)
                fig.supxlabel('cos_B per held-out room (dot = mean, whisker = ± SD across seeds; dashed = τ_B)', fontsize=8)
                fig.savefig(f'{A}/fig_per_room_eps{eps}.png', dpi=200, bbox_inches='tight')
                fig.savefig(f'{A}/fig_per_room_eps{eps}.pdf', bbox_inches='tight'); plt.close(fig)
                # identity, quality and intelligibility, reported separately
                P = [s for s in S if s['cond'] != 'none']
                fig, axs = plt.subplots(1, 3, figsize=(9.4, 0.34 * len(P) + 1.4), sharey=True)
                for ax, k, lab in zip(axs, ['cosB', 'pesq', 'estoi'],
                                      ['Identity: cos_B (lower = better hidden)', 'Quality: PESQ (higher = better)',
                                       'Intelligibility: ESTOI (higher = better)']):
                    rows = [dict(cond=s['cond'], v=s[k]) for s in P]
                    dots(ax, rows, key='v', sdkey='_none', seeds='_none'); style(ax)
                    ax.set_title(lab, fontsize=8, loc='left', color=INK)
                fig.savefig(f'{A}/fig_identity_quality_intelligibility_eps{eps}.png', dpi=200, bbox_inches='tight')
                fig.savefig(f'{A}/fig_identity_quality_intelligibility_eps{eps}.pdf', bbox_inches='tight'); plt.close(fig)
        except Exception:
            log('  (a figure failed; tables are still written)\n' + traceback.format_exc())

    # plain-language verdicts, decided by rules written before the run
    verdict = ['## What the results say (rules fixed before the run)', '']
    def rule(eps, tag, other, base, what):
        C = [c for c in out['by_eps'].get(str(eps), {}).get(tag, {}).get('contrasts', []) if c['other'] == other and c['base'] == base]
        if not C: return None
        c = C[0]; s = f'{f3(c["mean"])} [95% CI {f3(c["ci_lo"])}, {f3(c["ci_hi"])}]'
        if c['ci_lo'] > 0: v = f'{what}: **{LABEL[base]}** gives lower similarity than **{LABEL[other]}** ({s}); the difference is clearly above zero.'
        elif c['ci_hi'] < 0: v = f'{what}: **{LABEL[other]}** gives lower similarity than **{LABEL[base]}** ({s}); the paper\'s claim is reversed here.'
        else: v = f'{what}: no clear difference between **{LABEL[base]}** and **{LABEL[other]}** ({s}); the interval includes zero.'
        return v + f' Base better in {c["rooms_base_better"]}/{c["n_rooms"]} rooms and {c["seeds_base_better"]}/{c["n_seeds"]} seeds.'
    for eps in sorted(df.eps.unique(), reverse=True):
        for tag, nm in (('equal_eps', 'equal budget'), ('equal_snr', 'equal SNR')):
            for other, base, what in [('sim3_rt', 'meas_full', 'KEY TEST (measured vs simulated, matched)'),
                                      ('sim_raw_200', 'meas_full', 'Replication of the paper\'s original comparison'),
                                      ('sim_raw_200', 'sim_proc_200', 'Does processing alone change the simulated result'),
                                      ('sim_proc_200', 'sim3_rt', 'Does the number of simulated rooms matter')]:
                v = rule(eps, tag, other, base, what)
                if v: verdict.append(f'- eps {eps}, {nm}. {v}')
    verdict += ['', 'If the key test shows no clear difference, the paper must stop saying that measured channels beat simulated '
                'ones, and say instead that the earlier advantage did not survive matching. If it shows a clear difference, the '
                'claim can stay, reported with this matched comparison and its error bars.', '']
    info_md = ['## Set-up facts', '',
               f'- Main-run speakers taken from: {info.get("held_source", "?")}',
               f'- Device: {info.get("gpu") or info.get("device")}; torch {info.get("torch")}; filterbank: {info.get("filterbank")}',
               f'- Thresholds used: τ_A {info.get("tau", {}).get("A", "?")}, τ_B {info.get("tau", {}).get("B", "?")}',
               f'- Rebuilding the rooms from the sweeps: {info.get("reproduction", "not run")}',
               f'- Ablations compared against: {LABEL.get(abl_base, abl_base)}']
    if info.get('paper_sim_pool'):
        info_md.append('- The 200 simulated responses used in the paper\'s comparison came from: ' +
                       '; '.join(f'{k}: {v["n_rirs"]} responses from {v["n_rooms"]} virtual rooms ({", ".join(v["rooms"][:8])})'
                                 for k, v in info['paper_sim_pool'].items()))
        rts = [x for x in info.get('paper_sim_pool_rt60', []) if x is not None and np.isfinite(x)]
        if rts: info_md.append(f'- Their RT60 after processing: median {np.median(rts):.2f} s (range {min(rts):.2f}-{max(rts):.2f} s)')
    if os.path.exists(f'{OUT}/sim3_choices.json'):
        ch = json.load(open(f'{OUT}/sim3_choices.json'))
        info_md.append(f'- RT60-matched simulated rooms: {sum(len(v) for v in ch.values())} picks; see sim3_choices.json')
    md = md_lines[:4] + verdict + info_md + [''] + md_lines[4:]
    open(f'{A}/REPORT.md', 'w').write('\n'.join(md))
    open(f'{A}/tables.tex', 'w').write('\n'.join(tex))
    json.dump(out, open(f'{A}/summary.json', 'w'), indent=1, default=float)
    if not quiet:
        log('\n'.join(verdict))
        log(f'G. Analysis written to {A}')

# =============================================================================
if True:                                         # runs when the cell runs
    try:
        main_loop()
    except Exception:
        log('!! main loop stopped:\n' + traceback.format_exc())
    finally:
        try: json.dump(SIM3_CHOICES, open(f'{OUT}/sim3_choices.json', 'w'), indent=1)
        except Exception: pass
        try: analyse()
        except Exception: log('!! analysis failed:\n' + traceback.format_exc())
        log(f'ALL DONE in {hours():.1f} h. Open {OUT}/analysis/REPORT.md')
