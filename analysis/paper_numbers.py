# =============================================================================
#  PHC-351 · Hiding a Voice From Speaker Encoders · paper_numbers.py  (v2)
#
#  ONE documented analysis pipeline. Every number in the paper that is computed
#  from the optimisation trials comes from this script:
#    Section VI (equal-SNR targets), Sections VII-B to VII-E (matched comparison,
#    colouring control, ablations, runs), Section VII-G (operating point),
#    Tables III, IV and VI, corrections C9 and C10, and the "resolved
#    discrepancy" paragraph. It also runs the flat-channel check of C10.
#
#  USE   python paper_numbers.py OVERNIGHT_ROWS.csv DEVICE_ROWS.csv [out.json]
#        OVERNIGHT_ROWS = testing_data/results/overnight_v2/results_rows.csv
#        DEVICE_ROWS    = testing_data/results/device_control/results_rows.csv
#        needs: numpy, pandas, scipy, torch (torch only for the C10 check)
#
#  CONVENTIONS (the same everywhere)
#  - unit of analysis: a speaker x held-out-room trial (20 speakers x 4 rooms =
#    80), averaged over the optimisation runs (seeds) first;
#  - a comparison is the paired difference in cos_B, first minus second
#    (> 0: the second protects more), over the 80 trials: mean, t-based 95% CI,
#    a trial bootstrap CI (for the check that it agrees with the t interval),
#    paired t and Wilcoxon tests, Cohen's d_z;
#  - the 4 trials of one speaker are not independent, so every comparison is
#    also given at the SPEAKER level: (i) a t-test over the 20 speaker means and
#    (ii) a speaker-cluster bootstrap that resamples speakers with all their
#    trials (10 000 resamples, percentile 95% CI);
#  - Holm correction within a family of comparisons;
#  - equal budget = snr_matched 0; equal SNR = snr_matched 1;
#  - numbers are printed rounded half up, as in the paper.
#  Nothing is typed in by hand: every number comes from the two files (C10
#  comes from a copy of the measurement processing applied to a flat channel).
# =============================================================================
import sys, json, math
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
from scipy import stats

OVN, DEV = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else 'paper_numbers.json'
B = 10000

def load(p):
    d = pd.read_csv(p)
    for c in ['snr_db', 'cosB', 'cosA', 'succB', 'succA', 'pesq', 'stoi', 'estoi']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d

def trials(d, eps, cond, matched, col='cosB'):
    """per-trial values (runs averaged). 'none' exists only at equal budget."""
    m = 0 if cond == 'none' else matched
    x = d[(d.eps == eps) & (d.cond == cond) & (d.snr_matched == m)]
    return x.groupby(['clip', 'fold'])[col].mean()

def per_seed(d, eps, cond, matched, col='cosB'):
    m = 0 if cond == 'none' else matched
    x = d[(d.eps == eps) & (d.cond == cond) & (d.snr_matched == m)]
    return x.groupby('seed')[col].mean()

def compare(d, eps, first, second, matched, col='cosB'):
    a, b = trials(d, eps, first, matched, col), trials(d, eps, second, matched, col)
    j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    diff = (j.a - j.b)
    n = len(diff); m = diff.mean(); sd = diff.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    rng = np.random.default_rng(0)
    tboot = diff.values[rng.integers(0, n, size=(B, n))].mean(1)
    spk = diff.groupby(level='clip').mean()
    rng = np.random.default_rng(0)
    boot = spk.values[rng.integers(0, len(spk), size=(B, len(spk)))].mean(1)
    room = diff.groupby(level='fold').mean()
    sa, sb = per_seed(d, eps, first, matched, col), per_seed(d, eps, second, matched, col)
    seeds = (sa - sb).dropna()
    return dict(first=first, second=second, eps=eps, matched=matched, col=col, n=n, mean=m, half=h, ci=[m - h, m + h],
                ci_boot=[float(np.percentile(tboot, 2.5)), float(np.percentile(tboot, 97.5))],
                t=float(stats.ttest_1samp(diff, 0).statistic), p=float(stats.ttest_1samp(diff, 0).pvalue),
                p_wilcoxon=float(stats.wilcoxon(diff).pvalue) if (diff != 0).any() else float('nan'),
                dz=m / sd if sd > 0 else float('nan'),
                n_spk=len(spk), t_spk=float(stats.ttest_1samp(spk, 0).statistic), p_spk=float(stats.ttest_1samp(spk, 0).pvalue),
                ci_cluster=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                spk_same_sign=int((np.sign(spk) == np.sign(m)).sum()),
                per_room={k: float(v) for k, v in room.items()},
                per_seed={int(k): float(v) for k, v in seeds.items()},
                seeds_same_sign=int((np.sign(seeds) == np.sign(m)).sum()), n_seeds=len(seeds))

def holm(rs):
    rs = sorted(rs, key=lambda r: r['p']); m = len(rs); run = 0.0
    for i, r in enumerate(rs):
        run = max(run, min(1.0, (m - i) * r['p'])); r['p_holm'] = run

def summ(d, eps, cond, matched):
    """one row of a results table: means over the 80 runs-averaged trials; SD and range across runs"""
    m = 0 if cond == 'none' else matched
    x = d[(d.eps == eps) & (d.cond == cond) & (d.snr_matched == m)]
    pt = x.groupby(['clip', 'fold'])[['cosB', 'cosA', 'succB', 'snr_db', 'pesq', 'stoi', 'estoi']].mean()
    ps = x.groupby('seed')[['cosB', 'succB']].mean()
    n = len(pt); tq = stats.t.ppf(0.975, n - 1)
    return dict(cosB=pt.cosB.mean(), cosB_half=tq * pt.cosB.std(ddof=1) / math.sqrt(n),
                cosA=pt.cosA.mean(), cosA_half=tq * pt.cosA.std(ddof=1) / math.sqrt(n),
                succ=100 * pt.succB.mean(), snr=pt.snr_db.mean(), pesq=pt.pesq.mean(), stoi=pt.stoi.mean(),
                estoi=pt.estoi.mean(), n=n, runs=int(x.seed.nunique()),
                sd_runs_cosB=float(ps.cosB.std(ddof=1)) if len(ps) > 1 else 0.0,
                sd_runs_succ=float(100 * ps.succB.std(ddof=1)) if len(ps) > 1 else 0.0,
                range_runs_cosB=float(ps.cosB.max() - ps.cosB.min()), range_runs_succ=float(100 * (ps.succB.max() - ps.succB.min())))

def rnd(x, d):
    """round half up, after snapping away binary noise (as in the paper)"""
    if x is None or (isinstance(x, float) and not math.isfinite(x)): return 'nan'
    q = Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = format(q, 'f'); return '0.' + '0' * d if s in ('-0', '-0.' + '0' * d) and d else s
def fp(p): return f'{p:.1e}' if p < 1e-3 else rnd(p, 3)

res = {'conventions': 'per-trial runs-averaged; paired over 80 trials; speaker-level t and speaker-cluster bootstrap'}
ov, dv = load(OVN), load(DEV)
res['rows'] = {'overnight': len(ov), 'device': len(dv)}
P = []                                   # (section, label, printed value) in paper order
def put(sec, label, val): P.append((sec, label, val))

# =============================================================================
#  1. overnight run: every comparison of Fig. 3 (and more), both eps, both evaluations
# =============================================================================
FIG3 = [('sim3_rt', 'meas_full'), ('sim_proc_200', 'meas_full'), ('sim_raw_200', 'meas_full'), ('pgd', 'meas_full'),
        ('sim_raw_200', 'sim_proc_200'), ('sim3_rt', 'sim_proc_200'),
        ('abl_no_freqreg', 'meas_full'), ('abl_no_blend', 'meas_full'), ('abl_no_prenorm', 'meas_full'),
        ('abl_no_outnorm', 'meas_full'), ('abl_naive', 'meas_full')]
EXTRA = [('abl_naive', 'sim_proc_200'), ('abl_naive', 'sim_raw_200'), ('abl_naive', 'sim3_rt')]
fam = {}
for eps in (0.004, 0.002):
    for matched in (0, 1):
        rs = [compare(ov, eps, a, b, matched) for a, b in FIG3]
        holm(rs)
        rs += [compare(ov, eps, a, b, matched) for a, b in EXTRA]
        fam[f'{eps}|{matched}'] = rs
res['overnight_comparisons'] = fam
def C(eps, matched, a, b, fams=fam):
    for r in fams[f'{eps}|{matched}']:
        if r['first'] == a and r['second'] == b: return r
    raise KeyError((eps, matched, a, b))

COND = ['none', 'pgd', 'sim_raw_200', 'sim_proc_200', 'sim3_rt', 'meas_full',
        'abl_no_freqreg', 'abl_no_blend', 'abl_no_prenorm', 'abl_no_outnorm', 'abl_naive']
TAB = {f'{eps}|{m}': {c: summ(ov, eps, c, m) for c in COND} for eps in (0.004, 0.002) for m in (0, 1)}
res['table_III'] = TAB

# the speaker of C9: PGD perturbation stayed zero (SNR far above any real perturbation)
zc = sorted(set(ov[(ov.cond == 'pgd') & (ov.snr_matched == 0) & (ov.snr_db > 100)]['clip']))
zspk = sorted(set(ov[ov['clip'].isin(zc)]['spk']))
ov_wo = ov[~ov['clip'].isin(zc)]

# ---- Section VI: equal-SNR targets
for eps in (0.004, 0.002):
    tg = trials(ov, eps, 'meas_full', 1, 'snr_db')
    put('VI Setup', f'equal-SNR target, median over trials, eps {eps} (dB)', rnd(tg.median(), 1))
res['eqsnr_target_median'] = {str(e): float(trials(ov, e, 'meas_full', 1, 'snr_db').median()) for e in (0.004, 0.002)}
put('VI Setup', 'protected conditions in the overnight run', str(len(COND) - 1))

# ---- Table III
LAB = {'none': 'Unprotected', 'pgd': 'PGD, no channel', 'sim_raw_200': 'Sim., 200 raw, 0.5 s',
       'sim_proc_200': 'Sim., same 200, processed', 'sim3_rt': 'Sim., 3 rooms, RT60-matched',
       'meas_full': 'Measured, 3 rooms', 'abl_no_freqreg': 'no freq.-dep. regularisation',
       'abl_no_blend': 'no out-of-band blend', 'abl_no_prenorm': 'no normalisation before blend',
       'abl_no_outnorm': 'no output-level normalisation', 'abl_naive': 'no processing at all'}
for c in COND:
    a, b, cc, dd = TAB['0.004|0'][c], TAB['0.004|1'][c], TAB['0.002|0'][c], TAB['0.002|1'][c]
    cells = [rnd(a['cosB'], 3), rnd(a['succ'], 1), '--' if c == 'none' else rnd(a['snr'], 1),
             '--' if c == 'none' else rnd(a['pesq'], 2), '--' if c == 'none' else rnd(a['estoi'], 2),
             rnd(b['cosB'], 3), rnd(b['succ'], 1), rnd(cc['cosB'], 3), rnd(cc['succ'], 1), rnd(dd['cosB'], 3), rnd(dd['succ'], 1)]
    put('Table III', LAB[c], ' & '.join(cells))
allrows = [r for k, t in TAB.items() for r in t.values()]
put('Table III', 'largest SD across runs: cos_B', rnd(max(r['sd_runs_cosB'] for r in allrows), 4))
put('Table III', 'largest SD across runs: success (points)', rnd(max(r['sd_runs_succ'] for r in allrows), 1))
put('VII-E Runs', 'largest range across runs: cos_B', rnd(max(r['range_runs_cosB'] for r in allrows), 3))
put('VII-E Runs', 'largest range across runs: success (points)', rnd(max(r['range_runs_succ'] for r in allrows), 1))
w = summ(ov_wo, 0.004, 'pgd', 0)
put('Table III / C9', f'C9 speaker (clip {zc}, LibriSpeech id {zspk}): PGD at eps 0.004 without it: SNR, PESQ, ESTOI',
    f"{rnd(w['snr'], 1)} dB, {rnd(w['pesq'], 2)}, {rnd(w['estoi'], 2)}")
put('C9', 'PGD mean SNR at eps 0.004 with / without that speaker',
    f"{rnd(TAB['0.004|0']['pgd']['snr'], 1)} / {rnd(w['snr'], 1)} dB")

# ---- Fig. 3 caption
cis = [(r['first'], r['second'], e, m, r['ci'][0], r['ci'][1]) for k, rs in fam.items() for r in rs[:len(FIG3)]
       for e, m in [tuple(k.split('|'))]]
put('Fig. 3', 'every t interval of Fig. 3 excludes zero', str(all(lo > 0 or hi < 0 for *_, lo, hi in cis)))
r = C(0.002, 1, 'sim_proc_200', 'meas_full')
put('Fig. 3', 'closest interval: sim200 - meas, eps 0.002, equal SNR: lower limit, speaker p',
    f"{rnd(r['ci'][0], 4)}, p={rnd(r['p_spk'], 2)}")

# ---- VII-B
def ci_txt(r, d=3): return f"{rnd(r['mean'], d)} ({rnd(r['ci'][0], d)}--{rnd(r['ci'][1], d)})"
def cl_txt(r, d=3): return f"{rnd(r['ci_cluster'][0], d)} to {rnd(r['ci_cluster'][1], d)}"
k4b, k4s = C(0.004, 0, 'sim3_rt', 'meas_full'), C(0.004, 1, 'sim3_rt', 'meas_full')
put('VII-B', 'key: sim3 - meas, eps 0.004, equal budget', ci_txt(k4b) + f"; speaker-cluster {cl_txt(k4b)}")
put('VII-B', 'key: sim3 - meas, eps 0.004, equal SNR', ci_txt(k4s) + f"; speaker-cluster {cl_txt(k4s)}")
pr = list(k4b['per_room'].values()); ps_ = list(k4b['per_seed'].values())
put('VII-B', 'key, eps 0.004 budget: per-room range; per-run range', f"{rnd(min(pr), 3)}--{rnd(max(pr), 3)}; {rnd(min(ps_), 3)}--{rnd(max(ps_), 3)}")
put('VII-B', 'key, eps 0.004 budget: speaker-level p (n=20)', f"{k4b['p_spk']:.1e}")
for m in (0, 1):
    r = C(0.002, m, 'sim3_rt', 'meas_full')
    put('VII-B', f"key: sim3 - meas, eps 0.002, {'equal SNR' if m else 'equal budget'}", ci_txt(r) + f"; speaker-cluster {cl_txt(r)}")
r = compare(ov_wo, 0.004, 'sim3_rt', 'meas_full', 1)
put('VII-B', 'key, eps 0.004, equal SNR, without the C9 speaker', ci_txt(r))
put('VII-B', 'sim3 - sim200 (eps 0.004, budget): Delta', rnd(C(0.004, 0, 'sim3_rt', 'sim_proc_200')['mean'], 3))
for e in (0.004, 0.002):
    for m in (0, 1):
        r = C(e, m, 'sim_proc_200', 'meas_full')
        put('VII-B', f"sim200 - meas, eps {e}, {'equal SNR' if m else 'equal budget'}",
            ci_txt(r, 4 if e == 0.002 and m == 1 else 3) + f"; speaker p={rnd(r['p_spk'], 3)}; speaker-cluster {cl_txt(r)}")
for e in (0.004, 0.002):
    for m in (0, 1):
        g, s = C(e, m, 'sim3_rt', 'meas_full')['mean'], C(e, m, 'sim_proc_200', 'meas_full')['mean']
        put('VII-B', f"share of the key gap closed by sim200, eps {e}, {'SNR' if m else 'budget'}", f"{100 * (1 - s / g):.0f}%")
put('VII-B', 'processing the 200 (sim_raw - sim_proc, eps 0.004, budget)', ci_txt(C(0.004, 0, 'sim_raw_200', 'sim_proc_200')))

# ---- VII-D ablations
for a in ('abl_no_prenorm', 'abl_no_freqreg', 'abl_no_blend', 'abl_no_outnorm', 'abl_naive'):
    r = C(0.004, 0, a, 'meas_full')
    put('VII-D', f'{a} - meas, eps 0.004, budget', f"{rnd(r['mean'], 3)}; speaker p={rnd(r['p_spk'], 3)}; speaker-cluster {cl_txt(r)}")
put('VII-D', 'abl_no_outnorm - meas, eps 0.004, equal SNR', rnd(C(0.004, 1, 'abl_no_outnorm', 'meas_full')['mean'], 3))
abl_ok = all((C(e, m, a, 'meas_full')['ci'][0] > 0) for e in (0.004, 0.002) for m in (0, 1)
             for a in ('abl_no_freqreg', 'abl_no_blend', 'abl_no_prenorm', 'abl_no_outnorm', 'abl_naive'))
put('VII-D', 'every ablation interval excludes zero (both eps, both evaluations)', str(abl_ok))
for b in ('sim_proc_200', 'sim_raw_200'):
    r = C(0.004, 0, 'abl_naive', b)
    put('VII-D', f'no processing - {b}, eps 0.004, budget', f"{rnd(r['mean'], 3)}; speaker p={fp(r['p_spk'])}")
r = C(0.004, 0, 'abl_naive', 'sim3_rt')
put('VII-D', 'no processing - sim3, eps 0.004, budget', ci_txt(r) + f"; speaker p={rnd(r['p_spk'], 2)}")

# ---- VII-E runs and quality
same = all(r['seeds_same_sign'] == r['n_seeds'] for rs in fam.values() for r in rs[:len(FIG3)])
put('VII-E Runs', 'every Fig. 3 comparison has the same sign in every run', str(same))
CH = ['sim_raw_200', 'sim_proc_200', 'sim3_rt', 'meas_full', 'abl_no_freqreg', 'abl_no_blend', 'abl_no_prenorm', 'abl_no_outnorm', 'abl_naive']
t4 = TAB['0.004|0']
put('VII-E Runs', 'channel-trained conditions (9), eps 0.004: PESQ range; ESTOI range; cos_B range',
    f"{rnd(min(t4[c]['pesq'] for c in CH), 2)}--{rnd(max(t4[c]['pesq'] for c in CH), 2)}; "
    f"{rnd(min(t4[c]['estoi'] for c in CH), 2)}--{rnd(max(t4[c]['estoi'] for c in CH), 2)}; "
    f"{rnd(min(t4[c]['cosB'] for c in CH), 3)}--{rnd(max(t4[c]['cosB'] for c in CH), 3)}")
put('VII-E Runs', 'correlation of cos_B and PESQ across the 9 conditions',
    rnd(np.corrcoef([t4[c]['cosB'] for c in CH], [t4[c]['pesq'] for c in CH])[0, 1], 2))
r = compare(ov, 0.004, 'meas_full', 'sim3_rt', 0, 'pesq')
put('VII-E Runs', 'PESQ measured - sim3 (eps 0.004, budget)', f"{rnd(r['mean'], 2)}; speaker p={fp(r['p_spk'])}")
res['pesq_meas_vs_sim3'] = r

# ---- VII-G operating point (Table VI)
n0, m0 = TAB['0.004|0']['none'], TAB['0.004|0']['meas_full']
put('VII-G / Table VI', 'unprotected: success; cos_B; cos_A',
    f"{rnd(n0['succ'], 1)}%; {rnd(n0['cosB'], 3)} +- {rnd(n0['cosB_half'], 3)}; {rnd(n0['cosA'], 3)} +- {rnd(n0['cosA_half'], 3)}")
put('VII-G / Table VI', 'protected: success +- SD runs; cos_B; cos_A',
    f"{rnd(m0['succ'], 1)} +- {rnd(m0['sd_runs_succ'], 1)}%; {rnd(m0['cosB'], 3)} +- {rnd(m0['cosB_half'], 3)}; {rnd(m0['cosA'], 3)} +- {rnd(m0['cosA_half'], 3)}")
put('VII-G / Table VI', 'protected: SNR; PESQ; STOI / ESTOI',
    f"{rnd(m0['snr'], 1)} dB; {rnd(m0['pesq'], 2)}; {rnd(m0['stoi'], 2)} / {rnd(m0['estoi'], 2)}")
op = compare(ov, 0.004, 'none', 'meas_full', 0)
res['operating_point'] = op
put('VII-G / Table VI', 'paired difference unprotected - protected',
    f"{rnd(op['mean'], 3)} +- {rnd(op['half'], 3)}, t({op['n'] - 1})={op['t']:.2f}, p={op['p']:.1e}, d_z={rnd(op['dz'], 2)}; "
    f"speaker t({op['n_spk'] - 1})={op['t_spk']:.2f}, p={op['p_spk']:.1e}, {op['spk_same_sign']}/{op['n_spk']} speakers move down")

# ---- resolved discrepancy
t2 = TAB['0.002|0']
put('VIII Resolved discrepancy', 'eps 0.002: PESQ measured / sim 200 raw; SNR measured / sim 200 raw',
    f"{rnd(t2['meas_full']['pesq'], 2)} / {rnd(t2['sim_raw_200']['pesq'], 2)}; {rnd(t2['meas_full']['snr'], 1)} / {rnd(t2['sim_raw_200']['snr'], 1)} dB")

# =============================================================================
#  2. colouring-control run (Section VII-C, Table IV)
# =============================================================================
DCONDS = ['meas_full', 'sim3_rt', 'sim3_rt_dev', 'sim_proc_200', 'sim_proc_200_dev']
tab = {}
for eps in (0.004, 0.002):
    for matched in (0, 1):
        rows = {c: summ(dv, eps, c, matched) for c in DCONDS}
        rs = [compare(dv, eps, o, 'meas_full', matched) for o in ('sim3_rt', 'sim3_rt_dev', 'sim_proc_200', 'sim_proc_200_dev')]
        rs += [compare(dv, eps, 'sim3_rt', 'sim3_rt_dev', matched), compare(dv, eps, 'sim_proc_200', 'sim_proc_200_dev', matched)]
        holm(rs)
        share = {}
        for s in ('sim3_rt', 'sim_proc_200'):
            g = (trials(dv, eps, s, matched) - trials(dv, eps, 'meas_full', matched))
            gd = (trials(dv, eps, s + '_dev', matched) - trials(dv, eps, 'meas_full', matched))
            gs, gds = g.groupby(level='clip').mean().values, gd.groupby(level='clip').mean().values
            rng = np.random.default_rng(1); idx = rng.integers(0, len(gs), size=(B, len(gs)))
            bs = 1 - gds[idx].mean(1) / gs[idx].mean(1)
            share[s] = dict(share=float(1 - gd.mean() / g.mean()), ci_cluster=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                            gap_without=float(g.mean()), gap_with=float(gd.mean()))
        tab[f'{eps}|{matched}'] = dict(rows=rows, comparisons=rs, share_removed=share)
res['colouring_control'] = tab
DFAM = {k: v['comparisons'] for k, v in tab.items()}
DL = {'meas_full': 'Measured, 3 rooms', 'sim3_rt': 'Sim., 3, RT60-matched', 'sim3_rt_dev': '  + shared colouring',
      'sim_proc_200': 'Sim., 200, processed', 'sim_proc_200_dev': '  + shared colouring'}
for c in DCONDS:
    a, b = tab['0.004|0']['rows'][c], tab['0.002|0']['rows'][c]
    da = '--' if c == 'meas_full' else rnd(C(0.004, 0, c, 'meas_full', DFAM)['mean'], 3)
    db = '--' if c == 'meas_full' else rnd(C(0.002, 0, c, 'meas_full', DFAM)['mean'], 3)
    put('Table IV', DL[c], f"{rnd(a['cosB'], 3)} & {rnd(a['succ'], 1)} & {da} & {rnd(b['cosB'], 3)} & {rnd(b['succ'], 1)} & {db}")
bmax = max(max(abs(r['ci_boot'][0] - r['ci'][0]), abs(r['ci_boot'][1] - r['ci'][1]))
           for f_ in (fam, DFAM) for rs in f_.values() for r in rs)
put('VI Statistics', 'largest |trial-bootstrap CI - t CI| over all cos_B comparisons of both runs', rnd(bmax, 4))
put('Table IV', 'runs per eps', str(tab['0.004|0']['rows']['meas_full']['runs']) + ' / ' + str(tab['0.002|0']['rows']['meas_full']['runs']))
for e in (0.004, 0.002):
    for m in (0, 1):
        ev = 'equal SNR' if m else 'equal budget'
        for o in ('sim3_rt_dev', 'sim_proc_200_dev', 'sim3_rt', 'sim_proc_200'):
            r = C(e, m, o, 'meas_full', DFAM)
            rooms = sum(1 for v in r['per_room'].values() if np.sign(v) == np.sign(r['mean']))
            put('VII-C', f'{o} - meas, eps {e}, {ev}',
                ci_txt(r) + f"; speaker p={fp(r['p_spk'])}; speaker-cluster {cl_txt(r)}; same sign in {rooms}/4 rooms, {r['seeds_same_sign']}/{r['n_seeds']} runs")
        sh = tab[f'{e}|{m}']['share_removed']['sim3_rt']
        put('VII-C', f'share of the sim3 gap removed by the colouring, eps {e}, {ev}',
            f"{rnd(100 * sh['share'], 0)}% (speaker-cluster {rnd(100 * sh['ci_cluster'][0], 0)}--{rnd(100 * sh['ci_cluster'][1], 0)}%; "
            f"to one decimal {rnd(100 * sh['share'], 1)} [{rnd(100 * sh['ci_cluster'][0], 1)}, {rnd(100 * sh['ci_cluster'][1], 1)}]); "
            f"gap {rnd(sh['gap_without'], 3)} -> {rnd(sh['gap_with'], 3)}")
res['colouring_quality'] = {}
for col, d_ in (('succB', 1), ('pesq', 3), ('estoi', 3), ('snr_db', 3)):
    r = compare(dv, 0.004, 'sim_proc_200_dev', 'meas_full', 0, col); res['colouring_quality'][col] = r
    k = 100 if col == 'succB' else 1
    put('VII-C', f'sim200 + colouring - measured, eps 0.004, budget: {col}',
        f"{rnd(k * r['mean'], d_)} [{rnd(k * r['ci'][0], d_)}, {rnd(k * r['ci'][1], d_)}], p={fp(r['p'])}; speaker p={fp(r['p_spk'])}")
r = compare(dv, 0.002, 'sim_proc_200_dev', 'meas_full', 0, 'pesq'); res['colouring_quality']['pesq_0.002'] = r
put('VII-C', 'sim200 + colouring - measured, eps 0.002, budget: pesq', f"{rnd(r['mean'], 3)}, p={fp(r['p'])}")
for c in ('meas_full', 'sim_proc_200_dev', 'sim3_rt_dev'):
    q = tab['0.004|0']['rows'][c]
    put('VII-C', f'{c}, eps 0.004, budget: PESQ, ESTOI, SNR', f"{rnd(q['pesq'], 2)}, {rnd(q['estoi'], 2)}, {rnd(q['snr'], 1)} dB")
j = dv[(dv.snr_matched == 0) & dv.cond.isin(['meas_full', 'sim3_rt', 'sim_proc_200'])].merge(
    ov[(ov.snr_matched == 0)], on=['eps', 'seed', 'clip', 'fold', 'cond'], suffixes=('', '_o'))
res['reproduction'] = dict(n=len(j), mean_abs=float((j.cosB - j.cosB_o).abs().mean()), max_abs=float((j.cosB - j.cosB_o).abs().max()),
                           mean_signed=float((j.cosB - j.cosB_o).mean()))
put('VII-C', 'reproduction of 3 overnight conditions (same seeds/draws): trials compared; mean |diff|; largest',
    f"{len(j)}; {rnd(res['reproduction']['mean_abs'], 3)}; {rnd(res['reproduction']['max_abs'], 3)}")

# =============================================================================
#  3. C10: what the measurement processing does to a perfectly flat channel
# =============================================================================
SR, F_LO, F_HI = 16000, 250., 7000.
import torch
def bandweights(n_bins):                              # copied from the measurement code (t462 / 07_rooms.py / overnight v2)
    f = torch.linspace(0, SR / 2, n_bins)
    return 0.5 * (1 + torch.tanh((f - F_LO) / (0.3 * F_LO))) * 0.5 * (1 - torch.tanh((f - F_HI) / (0.1 * F_HI)))
def trim_to_sweep(rec, ref, pad=SR):
    n = 1 << (rec.shape[0] + ref.shape[0] - 1).bit_length()
    c = torch.fft.irfft(torch.fft.rfft(rec, n) * torch.fft.rfft(ref.flip(0), n), n)
    lag = int(c[:rec.shape[0]].argmax()) - (ref.shape[0] - 1)
    a = max(0, lag - pad); b = min(rec.shape[0], max(0, lag) + ref.shape[0] + 2 * pad)
    return rec[a:b], lag
def deconv_raw(rec, ref, lam=1e-2):
    n = 1 << (max(rec.shape[0], ref.shape[0]) * 2 - 1).bit_length()
    Y = torch.fft.rfft(rec, n); X = torch.fft.rfft(ref, n); bp = bandweights(X.shape[0])
    reg = lam * (X.abs() ** 2).max() * (1 + 1e4 * (1 - bp))
    return torch.fft.irfft(Y * X.conj() / (X.abs() ** 2 + reg) * bp, n)
def post_process(h_long, tail=SR):
    k = int(h_long[:SR * 5].abs().argmax()); h = h_long[max(0, k - 64): max(0, k - 64) + tail]
    h = torch.nn.functional.pad(h, (0, max(0, tail - h.shape[0]))); h = h / h.abs().max().clamp_min(1e-9)
    Hc = torch.fft.rfft(h, tail); bc = bandweights(Hc.shape[0]); d = torch.zeros(tail); d[64] = 1.
    h = torch.fft.irfft(Hc * bc + torch.fft.rfft(d, tail) * (1 - bc), tail)
    return h / h.abs().max().clamp_min(1e-9)
T, f1, f2 = 10.0, 20.0, 8000.0                       # the reference sweep of the measurements (t413)
t = torch.linspace(0, T, int(SR * T)); K = 2 * math.pi * f1 * T / math.log(f2 / f1)
ref = torch.cat([torch.zeros(SR), torch.sin(K * (torch.exp(t / T * math.log(f2 / f1)) - 1)) * 0.8, torch.zeros(2 * SR)])
rec = torch.cat([torch.zeros(int(0.7 * SR)), ref, torch.zeros(SR)])          # a perfectly flat loudspeaker + microphone, no room
tr, _ = trim_to_sweep(rec, ref); h = post_process(deconv_raw(tr, ref))
NF = 1 << 15; fr = np.fft.rfftfreq(NF, 1 / SR)
L = 20 * np.log10(np.abs(np.fft.rfft(h.numpy().astype(np.float64), NF)) + 1e-12)
def third_oct(fc):
    m = (fr >= fc * 2 ** (-1 / 6)) & (fr <= fc * 2 ** (1 / 6)); return float(10 * np.log10(np.mean(10 ** (L[m] / 10))))
flat = {f'{f}': third_oct(f) - third_oct(1000) for f in (350, 500, 2000, 4500, 5500, 6500)}
X = torch.fft.rfft(ref, 1 << (ref.shape[0] * 2 - 1).bit_length()); Pw = (X.abs() ** 2).numpy(); fx = np.linspace(0, SR / 2, len(Pw))
res['flat_channel_after_processing_dB_re_1kHz_third_octave'] = flat
res['sweep_power_ratio_peak_to_5kHz'] = float(Pw.max() / np.mean(Pw[(fx > 4900) & (fx < 5100)]))
put('C10', 'flat channel after the processing, 1/3-octave level re 1 kHz (dB) at 350 / 500 / 2000 / 4500 / 5500 / 6500 Hz',
    ' / '.join(rnd(flat[k], 1) for k in ('350', '500', '2000', '4500', '5500', '6500')))
put('C10', 'sweep power: peak / 5 kHz', rnd(res['sweep_power_ratio_peak_to_5kHz'], 0))

json.dump(res, open(OUT, 'w'), indent=1, default=float)

# =============================================================================
#  readable output
# =============================================================================
print(f"ROWS: overnight {res['rows']['overnight']}, colouring control {res['rows']['device']}\n")
print('=' * 100 + '\nPAPER VALUES (rounded as printed in the paper)\n' + '=' * 100)
sec = None
for s, lab, v in P:
    if s != sec: print(f'\n[{s}]'); sec = s
    print(f'  {lab}: {v}')
def line(r):
    return (f"{r['first']} - {r['second']} (eps {r['eps']}, {'equal SNR' if r['matched'] else 'equal budget'}{'' if r['col'] == 'cosB' else ', ' + r['col']}): "
            f"{r['mean']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}], t({r['n'] - 1})={r['t']:.2f}, p={fp(r['p'])}, d_z={r['dz']:.2f}; "
            f"speaker t({r['n_spk'] - 1})={r['t_spk']:.2f}, p={fp(r['p_spk'])}, cluster CI [{r['ci_cluster'][0]:.4f}, {r['ci_cluster'][1]:.4f}], "
            f"speakers same sign {r['spk_same_sign']}/{r['n_spk']}, runs same sign {r['seeds_same_sign']}/{r['n_seeds']}"
            + (f", Holm p={fp(r['p_holm'])}" if 'p_holm' in r else ''))
print('\n' + '=' * 100 + '\nALL COMPARISONS IN FULL\n' + '=' * 100)
for k, rs in fam.items():
    print(f'\nOVERNIGHT eps|equal-SNR = {k}'); [print('  ' + line(r)) for r in rs]
for k, v in tab.items():
    print(f'\nCOLOURING CONTROL eps|equal-SNR = {k}')
    for c, r in v['rows'].items(): print(f"  {c:18s} cosB {r['cosB']:.4f} succ {r['succ']:.2f} SNR {r['snr']:.2f} PESQ {r['pesq']:.3f} ESTOI {r['estoi']:.3f} runs {r['runs']} n {r['n']}")
    for r in v['comparisons']: print('  ' + line(r))
print('\nOPERATING POINT ' + line(op))
