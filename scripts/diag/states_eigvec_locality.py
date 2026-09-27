"""POST-HOC diagnostic: are the slow eigenvectors localised on rare microstates?"""
import sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import states as S
from vieb.states import micro, operator as op
sel = S._read("select.json"); tune, fit_d, rep = S._splits_std(sel)
z = np.load("work/states/corpus.npz"); lab = z["lab_rep"]
occ = np.bincount(lab[lab >= 0], minlength=micro.N_MICRO)
print("microstate visits: min", occ.min(), "p1", np.percentile(occ, 1), "p10", np.percentile(occ, 10), "median", np.median(occ), "n<100", (occ < 100).sum(), flush=True)
for tau in (1, 16, 64):
    p, keep = op.transition_matrix(micro.counts(lab, rep["bounds"], tau, micro.N_MICRO))
    w, v = np.linalg.eig(p.T if False else p)
    o = np.argsort(-np.abs(w))[:5]
    for i in o[1:4]:
        vec = np.abs(v[:, i].real); vec /= vec.sum()
        top = np.argsort(-vec)[:5]; kk = np.flatnonzero(keep)
        print(f"tau {tau} |l|={abs(w[i]):.3f} top-5 mass {vec[top].sum():.2f} visits of top states {occ[kk[top]].tolist()}", flush=True)
    chi = op.pcca(op.reversibilise(p), 2); st = op.assign(lab, keep, chi, threshold=0.8)
    print(f"tau {tau} PCCA 2: share state0 {np.mean(st==0):.4f} state1 {np.mean(st==1):.4f} abstain {np.mean(st<0):.4f}", flush=True)
