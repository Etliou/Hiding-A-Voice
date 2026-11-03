# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 08_thresholds.py: Decision thresholds measured under the test conditions (fix for mistake C2): tau_A 0.734, tau_B 0.793.
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

# ===== CELL 3a — the threshold that actually applies =====
rng = random.Random(1)
spk_list = [s for s in held if len(held[s]) >= 3][:30]
out = {}
for jn, J in [('A', model), ('B', modelB)]:
    gen, imp = [], []
    for hr in ready:
        train = [r for r in ready if r != hr]; ht = RIR_MEAS[hr]
        for s in spk_list:
            ps  = rng.sample(held[s], 2)
            enr = fit(load16k(ps[0]))
            E = nrm(torch.stack([emb(J, apply_rir(enr, RIR_MEAS[r])) for r in train]
                                + [emb(J, enr)]).mean(0))
            gen.append(cos(E, emb(J, apply_rir(fit(load16k(ps[1])), ht))))
            s2 = rng.choice([x for x in spk_list if x != s])
            imp.append(cos(E, emb(J, apply_rir(fit(load16k(rng.choice(held[s2]))), ht))))
    gen, imp = np.array(gen), np.array(imp)
    ths = np.linspace(-1, 1, 2000)
    far = np.array([(imp > t).mean() for t in ths]); frr = np.array([(gen <= t).mean() for t in ths])
    i = int(np.argmin(np.abs(far - frr))); out[jn] = float(ths[i])
    print(f'judge {jn}: EER {100*(far[i]+frr[i])/2:5.2f}%   tau {ths[i]:.3f}')
    print(f'   genuine  {gen.mean():.3f} +- {gen.std():.3f}'
          f'    impostor {imp.mean():.3f} +- {imp.std():.3f}')
TAU_A, TAU_B = out['A'], out['B']
