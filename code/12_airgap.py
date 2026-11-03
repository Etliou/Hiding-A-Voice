# PHC-351 · Hiding a Voice · Bhavishay (24324008)
# 12_airgap.py: The physical air-gap result: the clips played aloud in 4 rooms and re-recorded on the phone (100% -> 75%).
# The code below is the project's Colab notebook cell(s) for this stage, as run (Colab, T4 GPU), in run order.

md = torch.load(f'{REC}/measured_rirs.pt'); RIR_MEAS, ready = md['rirs'], md['rooms']
th = torch.load(f'{RES}/thresholds_measured.pt'); TAU_A, TAU_B = th['TAU_A'], th['TAU_B']
src_n = fit(load16k(f'{R}/demo_best_config/C_original_norm.wav'))
w_op  = fit(load16k(f'{R}/demo_best_config/C_protected_eps004.wav'))

have = {os.path.basename(q)[:-4] for q in glob.glob(f'{REC}/*.wav')}
air = [r for r in ready if f'clean2_{r}' in have and f'prot2_{r}' in have]

print(f'\n{"room":15s}{"clean":>9s}{"prot":>9s}{"drop":>8s}   judge / enrolment')
rows = []
for r in air:
    tr = [q for q in air if q != r]
    rc = align_to(load16k(f'{REC}/clean2_{r}.wav'), src_n)
    rp = align_to(load16k(f'{REC}/prot2_{r}.wav'),  w_op)
    for jn, J in [('B', modelB), ('A', model)]:
        # (i) enrolment from measured RIRs of the other rooms
        Ed = nrm(torch.stack([emb(J, apply_rir(src_n, RIR_MEAS[q])) for q in tr]
                             + [emb(J, src_n)]).mean(0))
        # (ii) enrolment from the actual clean recordings in the other rooms
        Ep = nrm(torch.stack([emb(J, align_to(load16k(f'{REC}/clean2_{q}.wav'), src_n))
                              for q in tr]).mean(0))
        for en, E in [('simulated', Ed), ('recorded', Ep)]:
            c1, c2 = cos(E, emb(J, rc)), cos(E, emb(J, rp))
            rows.append(dict(room=r, judge=jn, enrol=en, clean=c1, prot=c2))
            print(f'{r:15s}{c1:9.3f}{c2:9.3f}{c1-c2:8.3f}   judge {jn} / {en}')
json.dump(rows, open(f'{RES}/airgap_final.json','w'))
b = [x for x in rows if x['judge']=='B' and x['enrol']=='recorded']
print(f'\nJUDGE B, recorded enrolment  (tau_B = {TAU_B:.3f})')
print(f'  clean     {np.mean([x["clean"] for x in b]):.3f}'
      f'   attacker succeeds {100*np.mean([x["clean"]>TAU_B for x in b]):.0f}%')
print(f'  protected {np.mean([x["prot"]  for x in b]):.3f}'
      f'   attacker succeeds {100*np.mean([x["prot"] >TAU_B for x in b]):.0f}%')
