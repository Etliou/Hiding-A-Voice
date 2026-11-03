# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 01_features.py: First log-mel feature test on one LibriSpeech file (with a plot).
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import torch, torchaudio, glob, matplotlib.pyplot as plt

# ---- settings (your pipeline constants) ----
SR, N_FFT, HOP, WIN, N_MELS = 16000, 512, 160, 400, 80

# ---- 1. load one file ----
path = glob.glob('/content/LibriSpeech/train-clean-100/*/*/*.flac')[0]
wav, sr = torchaudio.load(path)          # wav: (channels, samples)
wav = wav.mean(0)                        # mono: (samples,)
print("file:", path.split('/')[-1], "| sr:", sr, "| samples:", wav.shape[0],
      "| seconds:", round(wav.shape[0]/sr, 2))

# ---- 2. STFT: frames -> complex spectrum ----
window = torch.hann_window(WIN)
X = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
               window=window, center=True, return_complex=True)
print("STFT:", tuple(X.shape), "= (bins, frames)")

# ---- 3. magnitude -> power ----
power = X.abs() ** 2                     # phase discarded here
print("power:", tuple(power.shape))

# ---- 4. mel filterbank: 257 bins -> 80 bands ----
fb = torchaudio.functional.melscale_fbanks(
        n_freqs=N_FFT // 2 + 1, f_min=0.0, f_max=SR / 2,
        n_mels=N_MELS, sample_rate=SR, norm=None, mel_scale="htk")
print("filterbank:", tuple(fb.shape), "= (bins, bands)")
mel = fb.T @ power                       # (80, frames)

# ---- 5. log ----
logmel = torch.log(mel + 1e-6)
print("log-mel:", tuple(logmel.shape),
      "| min/max:", round(logmel.min().item(),2), round(logmel.max().item(),2))

# ---- 6. look at it ----
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(wav.numpy()); ax[0].set_title("waveform")
ax[1].imshow(logmel.numpy(), origin='lower', aspect='auto', cmap='magma')
ax[1].set_title("log-mel"); ax[1].set_xlabel("frame"); ax[1].set_ylabel("mel band")
plt.tight_layout(); plt.show()
