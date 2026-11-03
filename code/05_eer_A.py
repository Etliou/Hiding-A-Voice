# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 05_eer_A.py: Equal error rate of encoder A on the 51 held-out speakers (clean audio): 0.92%, tau 0.463.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

import torch, random, numpy as np

model.eval()

# build embeddings for held-out speakers
by_spk = {}
for p, s in pairs:
    if s in held_spk:
        by_spk.setdefault(s, []).append(p)
by_spk = {s: v[:10] for s, v in by_spk.items() if len(v) >= 10}   # 10 clips each
print("held-out speakers used:", len(by_spk))

@torch.no_grad()
def embed(path):
    wav, _ = torchaudio.load(path); wav = wav.mean(0)
    if wav.shape[0] < CHUNK:
        wav = torch.nn.functional.pad(wav, (0, CHUNK - wav.shape[0]))
    wav = wav[:CHUNK]
    e, _ = model(logmel(wav).unsqueeze(0).to(dev))
    return torch.nn.functional.normalize(e, dim=1)[0].cpu()

emb = {s: [embed(p) for p in v] for s, v in by_spk.items()}
print("embeddings done")

# same-speaker and different-speaker pairs
rng = random.Random(0)
same, diff = [], []
spk_list = list(emb)
for s in spk_list:
    for i in range(len(emb[s])):
        for j in range(i+1, len(emb[s])):
            same.append(float(emb[s][i] @ emb[s][j]))
for _ in range(len(same)):
    a, b = rng.sample(spk_list, 2)
    diff.append(float(rng.choice(emb[a]) @ rng.choice(emb[b])))

same, diff = np.array(same), np.array(diff)
print(f"pairs: {len(same)} same, {len(diff)} different")
print(f"mean similarity -> same {same.mean():.3f} | different {diff.mean():.3f}")

# EER
ths = np.linspace(-1, 1, 2001)
far = [(diff > t).mean() for t in ths]      # strangers accepted
frr = [(same <= t).mean() for t in ths]     # genuine rejected
i = int(np.argmin(np.abs(np.array(far) - np.array(frr))))
print(f"EER = {100*(far[i]+frr[i])/2:.2f}%  at threshold {ths[i]:.3f}")
