# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 15_transfer_tricks.py: The transfer tricks that did not help: masking budget, targeted objective, momentum, input diversity.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

def masking_budget(x, floor=0.12, w=400):
    """Per-sample perturbation budget that follows the speech envelope."""
    e = torch.nn.functional.avg_pool1d((x**2).view(1,1,-1), w, w//2, padding=w//2).sqrt()
    e = torch.nn.functional.interpolate(e, size=x.shape[0], mode='linear',
                                        align_corners=False)[0,0]
    e = (e / e.max().clamp_min(1e-9)).clamp(0, 1)
    return floor + (1 - floor) * e          # in [floor, 1]

def protect_pro(wav, models, e_tgt, eps=0.002, K=400, eot=4, rir_pool=None,
                mu=0.9, lam_t=1.0, mask=True):
    wav = wav.to(dev); alpha = eps/12
    pool = [h.to(dev) for h in rir_pool] if rir_pool else None
    e0 = [emb(M, wav) for M in models]
    et = [t.to(dev) for t in e_tgt]
    m  = masking_budget(wav).to(dev) if mask else torch.ones_like(wav)
    d, gacc = torch.zeros_like(wav, requires_grad=True), torch.zeros_like(wav)
    for _ in range(K):
        x = (wav + d).clamp(-1, 1); tot = 0
        for _b in range(max(1, eot)):
            xb = apply_rir(x, pool[random.randrange(len(pool))]) if pool else x
            xb = xb * (0.8 + 0.4*random.random())              # random gain
            xb = torch.roll(xb, random.randint(-80, 80))       # +-5 ms shift
            xb = xb + torch.randn_like(xb)*xb.std()*0.01       # 40 dB noise
            for M, a0, at in zip(models, e0, et):
                ee = torch.nn.functional.normalize(
                        M(logmel(xb).unsqueeze(0))[0], dim=1)[0]
                tot = tot + (ee*a0).sum() - lam_t*(ee*at).sum()   # away from self, toward target
        g, = torch.autograd.grad(tot/(max(1,eot)*len(models)), d)
        gacc = mu*gacc + g/g.abs().mean().clamp_min(1e-12)        # momentum
        d = torch.max(torch.min(d - alpha*gacc.sign(), eps*m), -eps*m
                      ).detach().requires_grad_(True)
    return (wav + d).clamp(-1, 1).detach().cpu(), d.detach().cpu()

# target = the held-out speaker least similar to the source
best, bc = None, 2.0
for s, ps in held.items():
    c = cos(emb(model, fit(load16k(ps[0]))), emb(model, src))
    if c < bc: bc, best = c, ps[0]
tgt   = fit(load16k(best))
e_tgt = [emb(M, tgt) for M in [model, modelC]]
