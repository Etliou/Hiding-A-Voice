# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 02_dataset.py: Log-mel front end, the speaker-disjoint split (200 / 51 speakers, seed 0), 3-second crops, the data loader.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import torch, torchaudio, glob, random
from torch.utils.data import Dataset, DataLoader

SR, N_FFT, HOP, WIN, N_MELS = 16000, 512, 160, 400, 80
CHUNK = 3 * SR                                   # 48000 samples

_window = torch.hann_window(WIN)
_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FFT//2+1, f_min=0.0, f_max=SR/2,
        n_mels=N_MELS, sample_rate=SR, norm=None, mel_scale="htk")

def logmel(wav):
    """wav: (samples,) -> (80, frames)"""
    X = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                   window=_window.to(wav.device), center=True, return_complex=True)
    return torch.log(_fb.T.to(wav.device) @ (X.abs()**2) + 1e-6)

# ---------- collect files and split BY SPEAKER ----------
files = sorted(glob.glob('/content/LibriSpeech/train-clean-100/*/*/*.flac'))
pairs = [(p, p.split('/')[-3]) for p in files]

speakers = sorted({s for _, s in pairs})
random.Random(0).shuffle(speakers)
n_train = int(0.8 * len(speakers))
train_spk, held_spk = set(speakers[:n_train]), set(speakers[n_train:])
spk2idx = {s: i for i, s in enumerate(sorted(train_spk))}

print(f"speakers: {len(speakers)} -> train {len(train_spk)}, held-out {len(held_spk)}")

class SpeakerSet(Dataset):
    def __init__(self, pairs, keep, labelled=True):
        self.items = [(p, s) for p, s in pairs if s in keep]
        self.labelled = labelled
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, spk = self.items[i]
        wav, _ = torchaudio.load(path)
        wav = wav.mean(0)
        if wav.shape[0] < CHUNK:                       # pad short files
            wav = torch.nn.functional.pad(wav, (0, CHUNK - wav.shape[0]))
        else:                                          # random 3s crop
            start = random.randint(0, wav.shape[0] - CHUNK)
            wav = wav[start:start + CHUNK]
        label = spk2idx[spk] if self.labelled else -1
        return logmel(wav), label

train_ds = SpeakerSet(pairs, train_spk)
held_ds  = SpeakerSet(pairs, held_spk, labelled=False)
print("train clips:", len(train_ds), "| held-out clips:", len(held_ds))

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,
                      num_workers=2, drop_last=True)

xb, yb = next(iter(train_dl))
print("batch features:", tuple(xb.shape), "| labels:", tuple(yb.shape),
      "| label range:", yb.min().item(), "-", yb.max().item())
