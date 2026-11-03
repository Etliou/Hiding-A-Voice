# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 11_eps_curve.py: protect_meas and the epsilon curve over the four held-out rooms (encoder A, and A + C).
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

from pesq import pesq as pesq_fn

def protect_meas(wav, eps, models, K=300, eot=4, rir_pool=None):
    """plain measured-channel EOT: untargeted, no momentum, no diversity, no mask"""
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

def pq(ref, deg):
    try: return pesq_fn(SR, ref.numpy(), deg.numpy(), 'wb')
    except Exception: return float('nan')

nrm = lambda v: torch.nn.functional.normalize(v, dim=0)
torch.manual_seed(0); random.seed(0)
EPSL = [0.002, 0.005, 0.010, 0.020]
rows = []
for nm, ms in [('A', [model]), ('A+C', [model, modelC])]:
    for eps in EPSL:
        cB, cA, sn, pe = [], [], [], []
        for hr in ready:
            pool  = [RIR_MEAS[r] for r in ready if r != hr]
            train = [r for r in ready if r != hr]
            E = {k: nrm(torch.stack([emb(M, apply_rir(src, RIR_MEAS[r])) for r in train]
                                    + [emb(M, src)]).mean(0))
                 for k, M in [('A',model),('B',modelB)]}
            w = protect_meas(src, eps, ms, rir_pool=pool)
            y = apply_rir(w, RIR_MEAS[hr])
            cB.append(cos(E['B'], emb(modelB, y))); cA.append(cos(E['A'], emb(model, y)))
            sn.append(snr_db(src, w - src)); pe.append(pq(src, w))
        rows.append((nm, eps, np.mean(sn), np.nanmean(pe), np.mean(cB), np.mean(cA)))
