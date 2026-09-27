"""POST-HOC diagnostic, not a registered analysis: why gate 3 failed."""
import sys, json, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import states as S
from vieb.tok import surrogate as sr
from vieb.states import embed, micro, operator as op
from vieb.io import spine
sel = S._read("select.json"); tune, fit_d, rep = S._splits_std(sel)
k = sel["k_star"]; out = {}
def labels_for(train, d):
    rows = embed.sample_rows(train["X"], train["valid"], train["bounds"], k, S.FIT_ROWS, np.random.default_rng(0))
    m = micro.fit(rows, k, seed=0); return S._labels(m, d)
def spectrum(lab, d, taus=(1, 4, 16, 64)):
    r = {}
    for tau in taus:
        p, keep = op.transition_matrix(micro.counts(lab, d["bounds"], tau, micro.N_MICRO))
        ts = op.timescales(p, tau, k=4); r[tau] = [round(float(x), 1) for x in ts]
    return r
lab = labels_for(tune, rep); out["corpus"] = spectrum(lab, rep); print("corpus", out["corpus"], flush=True)
rng = np.random.default_rng(1)
w = {**rep, "X": np.asarray(sr.generate(rep["X"], rep["bounds"], "white", rng)["x"], np.float32)}
wt = {**tune, "X": np.asarray(sr.generate(tune["X"], tune["bounds"], "white", rng)["x"], np.float32)}
out["white"] = spectrum(labels_for(wt, w), w); print("white", out["white"], flush=True)
# plant, recovery by tau
orig = sel["tau_star"]; out["plant"] = {}
for tau in (1, 4, 16, 64):
    sel["tau_star"] = tau
    r = S._plant(rep, sel); out["plant"][tau] = round(r["best_balanced_accuracy"], 3); print("plant tau", tau, out["plant"][tau], flush=True)
json.dump(out, open("results/states_diag.json", "w"))
