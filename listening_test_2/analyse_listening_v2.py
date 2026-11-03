# =============================================================================
#  PHC-351 · Hiding a Voice · Bhavishay (24324008)
#  analyse_listening_v2.py: analysis of listening test 2, with the plan fixed
#  BEFORE any data was collected (written 26 Oct 2026).
#
#  USE
#    python analyse_listening_v2.py --manifest manifest.json \
#        --codes codes.txt --password "XXXX-XXXX-XXXX"   (answer codes pasted anywhere;
#                                                         the password unlocks keybox.json)
#        [--responses listening_test_2_responses.json]   (Download in the page's Results tab)
#        [--out lt2_analysis]
#
#  ANALYSIS PLAN, fixed before data collection (short version, about 5 minutes per listener)
#    Each listener hears 4 of the 12 speakers (2 women, 2 men; one of 3 fixed
#      groups, chosen at random), so all 12 speakers are covered across
#      listeners: 3 tone checks, 4 sentences (2 clean, 2 protected),
#      14 same/different pairs (4 speakers x clean-same, protected-same,
#      different + 2 attention pairs) and 9 quality ratings (4 clean,
#      4 protected, 1 anchor).
#    Unit of analysis: the listener. Every listener hears every condition.
#    Included: finished sessions of this manifest, not preview.
#    Not data: test runs (name starts with "test"; the researcher trying the page).
#    Excluded (decided now): (1) a repeat: when the same name finishes more than
#      once, only the earliest finished session counts; (2) answered "same
#      person" on both attention pairs (clearly different speakers).
#    Primary outcomes, each a paired difference protected minus clean:
#      P1 identity:  share of "same person" answers on true same-speaker pairs
#      P2 intelligibility: share of words typed correctly
#      P3 quality:   mean opinion score (1-5)
#    Tests: two-sided Wilcoxon signed-rank; mean difference with a 95%
#      bootstrap CI (10 000 resamples of listeners); Holm correction over P1-P3.
#    Equivalence (TOST, paired t) with margins fixed now: P1 ±0.10,
#      P2 ±0.05, P3 ±0.30 MOS. "Equivalent" is written only when p_TOST < 0.05.
#    Secondary: d' (same-speaker vs different-speaker pairs) for clean and for
#      protected tests; per-speaker and per-sentence tables.
#    Sensitivity: (a) only listeners who passed the headphone check (3 of 3),
#      (b) only self-reported headphones or earphones, (c) without listeners
#      reporting a hearing difficulty.
#    Wording rule: the report never says "imperceptible".
# =============================================================================
import argparse, json, math, os, re, sys, zlib, base64
import numpy as np

# ---------------- word scoring: identical rules to the test page --------------
NUMW = {str(i): w for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen '
                                         'fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
def words(s):
    s = (s or '').lower().replace('‘', "'").replace('’', "'").replace('`', "'")
    s = re.sub(r"[^a-z0-9' ]+", ' ', s)
    return [NUMW.get(w, w.replace("'", '')) for w in s.split() if w]

def lev(a, b):
    if abs(len(a) - len(b)) > 1: return 2
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]

def word_eq(a, b): return a == b or (len(a) >= 5 and len(b) >= 5 and lev(a, b) <= 1)

def words_correct(ref, typed):
    r, t = words(ref), words(typed)
    L = [[0] * (len(t) + 1) for _ in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(t) + 1):
            L[i][j] = L[i - 1][j - 1] + 1 if word_eq(r[i - 1], t[j - 1]) else max(L[i - 1][j], L[i][j - 1])
    return L[len(r)][len(t)], len(r)

# ---------------- answer codes ------------------------------------------------
def _b64(s): return base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))

def read_codes(text, jwk):
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    priv = ec.derive_private_key(int.from_bytes(_b64(jwk['d']), 'big'), ec.SECP256R1()) if jwk else None
    out, bad = [], 0
    for c in dict.fromkeys(re.findall(r'LT3\.([A-Za-z0-9_-]+)\.end', re.sub(r'\s+', '', text))):
        try:
            raw = _b64(c); flags = raw[1]
            if raw[0] != 3: raise ValueError('version')
            if flags & 2:
                if priv is None: raise ValueError('locked code needs --key')
                eph = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw[2:67])
                key = HKDF(algorithm=hashes.SHA256(), length=32, salt=b'', info=b'lt3').derive(priv.exchange(ec.ECDH(), eph))
                body = AESGCM(key).decrypt(raw[67:79], raw[79:], None)
            else:
                body = raw[2:-4]
                if zlib.crc32(body) & 0xFFFFFFFF != int.from_bytes(raw[-4:], 'big'): raise ValueError('damaged')
            if flags & 1: body = zlib.decompress(body, -15)
            o = json.loads(body.decode('utf-8')); o['source'] = 'code'; out.append(o)
        except Exception:
            bad += 1
    return out, bad

def key_from_password(password, keybox_path):
    """Unlock the researcher key exactly as the page does (PBKDF2-SHA256, then AES-GCM)."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    txt = open(keybox_path, encoding='utf-8').read()
    m = re.search(r'KEYBOX\s*=\s*(\{[^}]*\})', txt)                # keybox.json, or the page itself
    kb = json.loads(m.group(1) if m else txt)
    norm = re.sub(r'[^A-Z0-9]', '', password.upper()).encode()      # case, spaces and dashes do not matter
    k = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=_b64(kb['salt']), iterations=int(kb['iter'])).derive(norm)
    try:
        return json.loads(AESGCM(k).decrypt(_b64(kb['iv']), _b64(kb['ct']), None))
    except Exception:
        sys.exit('Wrong password: the key could not be unlocked.')

def is_test(s): return bool(s.get('test')) or bool(re.match(r'test\b', (s.get('name') or '').strip(), re.I))

# ---------------- statistics --------------------------------------------------
def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2: return (float('nan'), float('nan'))
    m = np.random.default_rng(seed).choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def wilcoxon_p(x):
    from scipy import stats
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2 or np.all(x == 0): return float('nan')
    try: return float(stats.wilcoxon(x).pvalue)
    except Exception: return float('nan')

def tost_p(x, margin):
    from scipy import stats
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 3: return float('nan')
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    if se == 0: return 0.0 if abs(m) < margin else 1.0
    p1 = 1 - stats.t.cdf((m + margin) / se, n - 1)       # H0: mean <= -margin
    p2 = stats.t.cdf((m - margin) / se, n - 1)           # H0: mean >= +margin
    return float(max(p1, p2))

def holm(ps):
    idx = [i for i, p in enumerate(ps) if np.isfinite(p)]
    order = sorted(idx, key=lambda i: ps[i]); out = [float('nan')] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * ps[i])); out[i] = run
    return out

def zq(p):
    from scipy import stats
    return float(stats.norm.ppf(p))

# ---------------- main analysis -----------------------------------------------
def analyse(man, sessions, out_dir):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    mv = str(man.get('created') or man.get('version'))
    nI, nD, nQ = len(man['intelligibility_items']), len(man['identity_trials']), len(man['quality_items'])
    by, tests = {}, set()
    for s in sessions:
        if s.get('demo') or str(s.get('mv')) != mv: continue
        if is_test(s): tests.add(s['sid']); continue
        if s['sid'] not in by or (s.get('t1') and not by[s['sid']].get('t1')): by[s['sid']] = s
    ses = list(by.values())
    def expected(s): return s.get('n') or [nI, nD, nQ]
    complete = [s for s in ses if s.get('t1') and [len(s.get('I', [])), len(s.get('D', [])), len(s.get('Q', []))] == list(expected(s))]
    rows, excluded = [], {'unfinished': len(ses) - len(complete), 'repeat': 0, 'attention': 0, 'test_runs': len(tests)}
    first = {}
    for s in sorted(complete, key=lambda s: s.get('t1') or 0):         # earliest finished session per name
        nm = ' '.join((s.get('name') or '').lower().split())
        if nm: first.setdefault(nm, s['sid'])
    for s in complete:
        same = {'clean_same': [], 'prot_same': [], 'diff': [], 'attention': []}
        for t, ans, *_ in s['D']:
            same[man['identity_trials'][t]['cond']].append(ans)
        wc = {0: [0, 0], 1: [0, 0]}
        for i, prot, text, *_ in s['I']:
            c, n = words_correct(man['intelligibility_items'][i]['transcript'], text); wc[prot][0] += c; wc[prot][1] += n
        q = {'clean': [], 'prot': [], 'anchor': []}
        for qi, rating, *_ in s['Q']:
            q[man['quality_items'][qi]['cond']].append(rating)
        att_wrong = sum(same['attention'])
        a = s.get('a', {}) or {}
        hp = s.get('hp', {}) or {}; n_hp = len(hp.get('a', [])) or 6
        r = dict(sid=s['sid'], name=s.get('name') or s.get('en', ''), design='short' if s.get('n') else 'long', group=s.get('g', ''), device=a.get('dev', ''), place=a.get('env', ''), english=a.get('eng', ''),
                 hearing=a.get('hear', ''), list=s.get('L', ''), headphone=hp.get('score', float('nan')), headphone_n=n_hp,
                 headphone_pass=int(hp.get('score', -1) >= (5 if n_hp == 6 else n_hp)),
                 attention_wrong=att_wrong, minutes=round((s['t1'] - s['t0']) / 60000, 1),
                 same_clean=np.mean(same['clean_same']), same_prot=np.mean(same['prot_same']), same_diff=np.mean(same['diff']),
                 words_clean=wc[0][0] / wc[0][1] if wc[0][1] else float('nan'),
                 words_prot=wc[1][0] / wc[1][1] if wc[1][1] else float('nan'),
                 mos_clean=np.mean(q['clean']), mos_prot=np.mean(q['prot']), mos_anchor=np.mean(q['anchor']) if q['anchor'] else float('nan'),
                 n_same_clean=len(same['clean_same']), n_same_prot=len(same['prot_same']), n_diff=len(same['diff']))
        k = lambda v, n: (v * n + 0.5) / (n + 1)                   # log-linear correction
        fa = k(r['same_diff'], r['n_diff'])
        r['dprime_clean'] = zq(k(r['same_clean'], r['n_same_clean'])) - zq(fa)
        r['dprime_prot'] = zq(k(r['same_prot'], r['n_same_prot'])) - zq(fa)
        nm = ' '.join((s.get('name') or '').lower().split())
        r['excluded'] = 'repeat' if nm and first.get(nm) != s['sid'] else ('attention' if att_wrong >= 2 else '')
        if r['excluded']: excluded[r['excluded']] += 1
        rows.append(r)
    with open(f'{out_dir}/per_listener.csv', 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    inc = [r for r in rows if not r['excluded']]

    def block(sub, title):
        lines = [f'### {title}  (n = {len(sub)} listeners)', '']
        if len(sub) < 3:
            return lines + ['Too few listeners for statistics.', ''], None
        spec = [('P1 identity: said "same person" on true pairs', 'same_prot', 'same_clean', 0.10, '{:.0%}'),
                ('P2 intelligibility: words typed correctly', 'words_prot', 'words_clean', 0.05, '{:.0%}'),
                ('P3 quality: mean opinion score', 'mos_prot', 'mos_clean', 0.30, '{:.2f}')]
        res = []
        for name, kp, kc, margin, fmt in spec:
            c = np.array([r[kc] for r in sub], float); p = np.array([r[kp] for r in sub], float)
            d = p - c; d = d[np.isfinite(d)]
            lo, hi = boot_ci(d)
            res.append(dict(name=name, clean=float(np.nanmean(c)), prot=float(np.nanmean(p)), diff=float(d.mean()),
                            lo=lo, hi=hi, p=wilcoxon_p(d), tost=tost_p(d, margin), margin=margin, fmt=fmt, n=len(d)))
        for r_, ph in zip(res, holm([r_['p'] for r_ in res])): r_['p_holm'] = ph
        lines += ['| Outcome | Clean | Protected | Difference (protected − clean) | 95% CI | p (Wilcoxon) | p (Holm) | Equivalence margin | p (TOST) |',
                  '|---|---|---|---|---|---|---|---|---|']
        fp = lambda x: '—' if not np.isfinite(x) else (f'{x:.1e}' if x < 1e-3 else f'{x:.3f}')
        for r_ in res:
            f = r_['fmt']; ds = '{:+.3f}'.format(r_['diff'])
            lines.append(f'| {r_["name"]} | {f.format(r_["clean"])} | {f.format(r_["prot"])} | {ds} | [{r_["lo"]:+.3f}, {r_["hi"]:+.3f}] | '
                         f'{fp(r_["p"])} | {fp(r_["p_holm"])} | ±{r_["margin"]} | {fp(r_["tost"])} |')
        lines.append('')
        for r_ in res:
            if np.isfinite(r_['p_holm']) and r_['p_holm'] < 0.05:
                v = f'protected differs from clean by {r_["diff"]:+.3f} (95% CI [{r_["lo"]:+.3f}, {r_["hi"]:+.3f}]).'
            elif np.isfinite(r_['tost']) and r_['tost'] < 0.05:
                v = f'equivalent within ±{r_["margin"]} (TOST p = {fp(r_["tost"])}); difference {r_["diff"]:+.3f}.'
            else:
                v = (f'no detectable difference, and equivalence within ±{r_["margin"]} is not shown either '
                     f'(95% CI [{r_["lo"]:+.3f}, {r_["hi"]:+.3f}]).')
            lines.append(f'- **{r_["name"].split(":")[0]}** ({r_["name"].split(":")[1].strip()}): {v}')
        dc = np.array([r['dprime_clean'] for r in sub]); dpp = np.array([r['dprime_prot'] for r in sub])
        lines += ['', f"Secondary: d' clean {dc.mean():.2f}, d' protected {dpp.mean():.2f}, "
                      f"paired difference {np.mean(dpp - dc):+.2f}, Wilcoxon p = {fp(wilcoxon_p(dpp - dc))}. "
                      f"Different-speaker pairs answered \"same\": {np.mean([r['same_diff'] for r in sub]):.0%}. "
                      f"Quality anchor MOS: {np.nanmean([r['mos_anchor'] for r in sub]):.2f}.", '']
        return lines, res

    L = ['# Listening test 2: results', '',
         f'Manifest {mv}. Sessions found {len(ses)} (plus {len(tests)} test run{"" if len(tests) == 1 else "s"}, not data); finished {len(complete)}; '
         f'excluded as a repeat of the same name {excluded["repeat"]}; excluded by the attention rule '
         f'{excluded["attention"]}; analysed **{len(inc)}**.', '',
         'Plan fixed before data collection (see the header of analyse_listening_v2.py). The report never claims that '
         'the protection is imperceptible; it reports differences with confidence intervals and equivalence only within '
         'the pre-set margins.', '']
    main_lines, main_res = block(inc, 'Primary analysis')
    L += main_lines
    L += ['## Sensitivity analyses', '']
    L += block([r for r in inc if r['headphone_pass']], '(a) passed the headphone check (all tone sets correct)')[0]
    L += block([r for r in inc if r['device'] in ('oh', 'ie')], '(b) self-reported headphones or earphones')[0]
    L += block([r for r in inc if r['hearing'] != 'yes'], '(c) without reported hearing difficulty')[0]
    dev = {}
    for r in inc: dev[r['device']] = dev.get(r['device'], 0) + 1
    hp_pass = np.mean([r['headphone_pass'] for r in inc]) if inc else float('nan')
    L += ['## Listeners', '', f'Devices: {dev}. Passed headphone check: {hp_pass:.0%}. '
          f'Median duration {np.median([r["minutes"] for r in inc]) if inc else float("nan"):.1f} min.', '']
    # per item
    items = {}
    for s in complete:
        if s['sid'] not in {r['sid'] for r in inc}: continue
        for t, ans, *_ in s['D']:
            tr = man['identity_trials'][t]; items.setdefault(('identity', tr['spk'], tr['cond']), []).append(ans)
        for i, prot, text, *_ in s['I']:
            it = man['intelligibility_items'][i]; c, n = words_correct(it['transcript'], text)
            items.setdefault(('intelligibility', it['spk'], 'prot' if prot else 'clean'), []).append(c / n if n else float('nan'))
        for qi, rating, *_ in s['Q']:
            qq = man['quality_items'][qi]; items.setdefault(('quality', qq['spk'] or qq['id'], qq['cond']), []).append(rating)
    with open(f'{out_dir}/per_item.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['measure', 'speaker', 'condition', 'n', 'mean'])
        for (m, spk, cond), v in sorted(items.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            w.writerow([m, spk, cond, len(v), round(float(np.nanmean(v)), 4)])
    open(f'{out_dir}/REPORT.md', 'w').write('\n'.join(L))
    json.dump(dict(manifest=mv, n_sessions=len(ses), n_complete=len(complete), excluded=excluded, n_analysed=len(inc),
                   primary=main_res), open(f'{out_dir}/summary.json', 'w'), indent=1, default=float)
    # figure: paired listener values, mean and 95% CI
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        if len(inc) >= 3:
            BLUE, GREY, INK = '#2a78d6', '#b5b4ae', '#0b0b0b'
            fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.2))
            for ax, (kc, kp, lab) in zip(axs, [('same_clean', 'same_prot', 'Said "same person" (true pairs)'),
                                               ('words_clean', 'words_prot', 'Words typed correctly'),
                                               ('mos_clean', 'mos_prot', 'Quality (MOS, 1-5)')]):
                c = np.array([r[kc] for r in inc]); p = np.array([r[kp] for r in inc])
                for a_, b_ in zip(c, p): ax.plot([0, 1], [a_, b_], color=GREY, lw=0.7, alpha=0.6, zorder=1)
                for x, v in ((0, c), (1, p)):
                    m = np.nanmean(v); lo, hi = boot_ci(v[np.isfinite(v)])
                    ax.plot([x, x], [lo, hi], color=BLUE, lw=2, zorder=3); ax.plot(x, m, 'o', ms=8, color=BLUE, zorder=4)
                ax.set_xticks([0, 1]); ax.set_xticklabels(['Clean', 'Protected']); ax.set_xlim(-0.4, 1.4)
                ax.set_title(lab, fontsize=9, loc='left', color=INK)
                for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
            fig.suptitle(f'Listening test 2, n = {len(inc)} listeners: grey = one listener, blue = mean with 95% CI',
                         fontsize=9, x=0.01, ha='left')
            fig.tight_layout(); fig.savefig(f'{out_dir}/fig_listening_v2.png', dpi=200, bbox_inches='tight')
            fig.savefig(f'{out_dir}/fig_listening_v2.pdf', bbox_inches='tight'); plt.close(fig)
    except Exception as e:
        print('figure failed:', e)
    print('\n'.join(L))
    return rows

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--responses', default=None)
    ap.add_argument('--codes', default=None)
    ap.add_argument('--key', default=None, help='researcher_PRIVATE.jwk.json (instead of --password)')
    ap.add_argument('--password', default=None, help='researcher password; unlocks the key in --keybox')
    ap.add_argument('--keybox', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keybox.json'),
                    help='keybox.json, or the test page .html (both hold the locked key)')
    ap.add_argument('--out', default='lt2_analysis')
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    sessions = []
    if a.responses:
        d = json.load(open(a.responses)); sessions += d['sessions'] if isinstance(d, dict) else d
    if a.codes:
        jwk = json.load(open(a.key)) if a.key else (key_from_password(a.password, a.keybox) if a.password else None)
        got, bad = read_codes(open(a.codes).read(), jwk); sessions += got
        print(f'codes read: {len(got)}, unreadable: {bad}')
    if not sessions: sys.exit('No sessions given: pass --responses and/or --codes.')
    analyse(man, sessions, a.out)
