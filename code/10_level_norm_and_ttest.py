# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 10_level_norm_and_ttest.py: Paired t-test on the 80 trials (t = 18.92), and the level-normalised clip used in the rooms and the listening test (fix for C7).
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

from scipy import stats
d = A['cleanB'] - A['protB']
t, p = stats.ttest_rel(A['cleanB'], A['protB'])
print(f'paired t = {t:.2f}, p = {p:.2e}, mean drop {d.mean():.3f} +- {ci(d):.3f}')
print(f"Cohen's d = {d.mean()/d.std(ddof=1):.2f}")

# ===== next cell =====
src_n = src / src.std() * 0.05                      # standard level
w_op  = protect_meas(src_n, 0.004, [model], rir_pool=list(RIR_MEAS.values()))
torchaudio.save(f'{R}/demo_best_config/C_protected_eps004.wav', w_op.unsqueeze(0), SR)
torchaudio.save(f'{R}/demo_best_config/C_original_norm.wav', src_n.unsqueeze(0), SR)
print(f'SNR {snr_db(src_n, w_op-src_n):.1f} dB   PESQ {pq(src_n, w_op):.2f}')
