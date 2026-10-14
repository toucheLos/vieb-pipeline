"""Paper 1, Table S1: every number in the draft, read from committed bytes.

    python3 paper1/sources.py            # -> paper1/provenance.csv, paper1/values.json

Each entry names a repo, a path, and a JSON pointer (or an atlas record). The
value is read with ``git show HEAD:<path>`` after asserting that the path is
clean in the working tree, so only committed bytes can enter the paper. The
atlas status is checked against the entry's allowed set: LIVE by default;
RETRACTED or SUPERSEDED only where the paper discusses that retraction, and
PROVISIONAL only where the text says so.

No value is computed here. Two transforms exist and both are recorded:
``len`` (the number of items in a stored list) and ``regex`` (a number quoted
inside a stored string).
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = {r: os.path.join(HOME, r) for r in
         ("recur", "vieb-pipeline", "shapeflow", "reversibility-tests")}
ATLAS = os.path.join(HOME, "vieb-atlas", "index.json")


@dataclass
class Src:
    key: str
    repo: str
    path: str
    ptr: str
    fmt: str
    unit: str = ""
    atlas: str | None = None
    allow: tuple[str, ...] = ("LIVE",)
    lo: str | None = None
    hi: str | None = None
    so: Any = None                 # scored_object: pointer or literal dict
    neff: Any = None               # n_effective: pointer or literal
    neff_unit: str = ""
    transform: str | None = None   # None | "len" | "regex:<pattern>"
    note: str = ""
    extra: dict = field(default_factory=dict)


_CACHE: dict[tuple[str, str], tuple[Any, str]] = {}


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", REPOS[repo], *args], check=True,
                          capture_output=True, text=True).stdout


def committed(repo: str, path: str) -> tuple[Any, str]:
    """The JSON at HEAD and the last commit that touched it; the path must be clean."""
    k = (repo, path)
    if k not in _CACHE:
        dirty = _git(repo, "status", "--porcelain", "--", path).strip()
        if dirty:
            raise SystemExit(f"{repo}/{path} is not clean: {dirty}")
        commit = _git(repo, "log", "-1", "--format=%h", "--", path).strip()
        if not commit:
            raise SystemExit(f"{repo}/{path} is not committed")
        _CACHE[k] = (json.loads(_git(repo, "show", f"HEAD:{path}")), commit)
    return _CACHE[k]


def committed_text(repo: str, path: str) -> tuple[str, str]:
    """A source file's committed text (for parameters stated in code)."""
    dirty = _git(repo, "status", "--porcelain", "--", path).strip()
    if dirty:
        raise SystemExit(f"{repo}/{path} is not clean: {dirty}")
    return _git(repo, "show", f"HEAD:{path}"), _git(repo, "log", "-1", "--format=%h", "--",
                                                     path).strip()


def pointer(obj: Any, ptr: str) -> Any:
    x = obj
    for part in [p for p in ptr.split("/") if p]:
        x = x[int(part)] if isinstance(x, list) else x[part]
    return x


def _atlas() -> dict[str, dict]:
    with open(ATLAS, encoding="utf-8") as fh:
        return {r["id"]: r for r in json.load(fh)["records"]}


def resolve(s: Src, atlas: dict[str, dict]) -> dict[str, Any]:
    if s.repo == "atlas":
        rec = atlas[s.path]
        v = pointer(rec, s.ptr)
        commit, path = rec.get("commit") or "", f"vieb-atlas/index.json#{s.path}"
        status = rec["status"]
    elif s.ptr.startswith("re:"):
        text, commit = committed_text(s.repo, s.path)
        m = re.search(s.ptr[3:], text)
        if not m:
            raise SystemExit(f"{s.key}: pattern not found in {s.repo}/{s.path}")
        v = float(m.group(1)) if "." in m.group(1) or "e" in m.group(1) else int(m.group(1))
        path = f"{s.repo}/{s.path}"
        status = "not indexed"
    else:
        obj, commit = committed(s.repo, s.path)
        v = pointer(obj, s.ptr)
        path = f"{s.repo}/{s.path}"
        status = atlas[s.atlas]["status"] if s.atlas else "not indexed"
        if s.atlas is None:
            pass
    if s.transform == "len":
        v = len(v)
    elif s.transform and s.transform.startswith("regex:"):
        m = re.search(s.transform[6:], str(v))
        if not m:
            raise SystemExit(f"{s.key}: pattern not found in {path}{s.ptr}")
        v = float(m.group(1))
    if s.atlas and status not in s.allow and status != "not indexed":
        raise SystemExit(f"{s.key}: atlas {s.atlas} is {status}, allowed {s.allow}")
    if s.repo == "atlas" and status not in s.allow:
        raise SystemExit(f"{s.key}: atlas {s.path} is {status}, allowed {s.allow}")
    out = {"key": s.key, "value": v, "fmt": s.fmt, "unit": s.unit, "repo": s.repo,
           "path": path, "pointer": s.ptr, "commit": commit,
           "atlas_id": s.atlas or (s.path if s.repo == "atlas" else ""),
           "atlas_status": status, "transform": s.transform or "", "note": s.note,
           "lo": "", "hi": "", "scored_object": "", "n_effective": "",
           "n_effective_unit": s.neff_unit}
    if s.repo != "atlas" and not s.ptr.startswith("re:"):
        obj, _c = committed(s.repo, s.path)
        if s.lo:
            out["lo"], out["hi"] = pointer(obj, s.lo), pointer(obj, s.hi or "")
        if isinstance(s.so, str) and s.so.startswith("/"):
            out["scored_object"] = json.dumps(pointer(obj, s.so), sort_keys=True)
        elif s.so is not None:
            out["scored_object"] = json.dumps(s.so, sort_keys=True)
        if isinstance(s.neff, str) and s.neff.startswith("/"):
            out["n_effective"] = pointer(obj, s.neff)
        elif s.neff is not None:
            out["n_effective"] = s.neff
    return out


# --------------------------------------------------------------------------
# The entries
# --------------------------------------------------------------------------

def _entries() -> list[Src]:
    E: list[Src] = []
    a = E.append
    R, V, SF, RT = "recur", "vieb-pipeline", "shapeflow", "reversibility-tests"

    # Dataset (shapeflow ingest / splits; vieb-pipeline QC)
    for k, f in (("n_recordings", "int"), ("n_frames", "int"), ("fps", "f0"),
                 ("n_kept_keypoints", "int")):
        a(Src(f"data.{k}", SF, "results/ingest.json", f"/anchor/{k}", f, atlas="shapeflow.results/ingest"))
    for sp in ("tune", "fit", "report"):
        a(Src(f"data.animals_{sp}", SF, "results/splits.json", f"/n_animals_per_split/{sp}", "int",
              "animals", atlas="shapeflow.results/splits"))
        a(Src(f"data.recordings_{sp}", SF, "results/splits.json", f"/n_recordings_per_split/{sp}", "int",
              "recordings", atlas="shapeflow.results/splits"))
    a(Src("data.bone_violation", V, "results/bones.json", "/reads/skull_raw/detail/headline_rate", "pct1",
          "fraction of frames", atlas="vieb-pipeline.results/bones/reads/skull_raw",
          allow=("LIVE",), so="/reads/skull_raw/scored_object", neff="/reads/skull_raw/n_effective",
          neff_unit="animals"))
    a(Src("data.bone_eps", V, "results/bones.json", "/reads/skull_raw/detail/headline_eps", "pct0",
          "relative stretch beyond the reference length"))
    a(Src("data.wall_inner", V, "results/concentration.json", "/edge_profile/0/rate", "pct1",
          "violation rate, innermost distance decile", note="edge_profile is not indexed"))
    a(Src("data.wall_outer", V, "results/concentration.json", "/edge_profile/9/rate", "pct1",
          "violation rate, outermost distance decile"))
    a(Src("data.sigma_bar", V, "results/plant.json", "/sigma_bar_bl", "f4", "body lengths",
          note="count-weighted skull-bone jitter (scripts/plant.py sigma_bar)"))

    # Finding 1: dwell (the retracted read, discussed as the retraction)
    for arm, nm in (("corpus_stored", "corpus"), ("ou", "ou"), ("ar", "ar"), ("white", "white")):
        base = f"/read/detail/{arm}"
        for k, f in (("n_reject", "int"), ("n_animals", "int"), ("frac_reject", "pct0"),
                     ("var_over_mean_sq", "f2"), ("median_dwell_frames", "f0"), ("n_labels", "int")):
            a(Src(f"dwell.{nm}.{k}", R, "results/dwell_moseq.json", f"{base}/{k}", f,
                  atlas="recur.results/dwell_moseq/read", allow=("RETRACTED",),
                  so="/read/scored_object", neff="/read/n_effective", neff_unit="animals",
                  note="retracted as evidence of discreteness: the surrogates reject too"))
    a(Src("dwell.kappa", R, "results/dwell_moseq.json", "/read/reason", "sci", "",
          atlas="recur.results/dwell_moseq/read", allow=("RETRACTED",),
          transform=r"regex:kappa\s*=\s*([0-9.e+]+)"))

    # Finding 2: grammar (PROVISIONAL; say so)
    PROV = ("PROVISIONAL",)
    for arm, path in (("corpus", "results/grammar_moseq.json"),
                      ("white", "results/grammar_moseq_surrogate_white.json"),
                      ("ou", "results/grammar_moseq_surrogate_ou.json"),
                      ("ar", "results/grammar_moseq_surrogate_ar.json"),
                      ("observed", "results/grammar_moseq_surrogate_observed.json")):
        at = ("recur.results/grammar_moseq/read" if arm == "corpus"
              else f"recur.results/grammar_moseq_surrogate_{arm}/read")
        for k in ("n_significant", "n_tested"):
            a(Src(f"grammar.{arm}.{k}", R, path, f"/summary/{k}", "int", "n-grams", atlas=at,
                  allow=PROV, so="/read/scored_object", neff="/read/n_effective",
                  neff_unit="animals", note="PROVISIONAL pending the four-gate restricted re-run"))
    a(Src("grammar.fdr_q", R, "results/grammar_moseq.json", "/summary/fdr_q", "f2", "",
          atlas="recur.results/grammar_moseq/read", allow=PROV))
    for k in ("n_significant", "n_tested"):
        a(Src(f"grammar.superseded.{k}", R, "results/grammar_moseq_contaminated.json",
              f"/summary/{k}", "int", "n-grams", note="superseded table (claim.progress.grammar_1553)"))
    a(Src("grammar.white_effect", "atlas", "claim.methods.grammar_white_effect", "/value", "f3",
          "median log ratio, surviving n-grams"))

    # Finding 3: preprocessing -> recurrence (q1)
    for nm, cell in (("wiener", 13), ("unfiltered", 11)):
        at = f"recur.q1.{nm}__w0.4__base__amplitude__d192__vs_ar"
        for null in (("ar", "phase") if nm == "unfiltered" else ("ar", "phase", "ou", "white")):
            p = f"/cells/{cell}/vs/{null}/delta"
            a(Src(f"q1.{nm}.{null}", R, "results/q1.json", f"{p}/point", "spct3", "excess rate",
                  atlas=at if null == "ar" else None, lo=f"{p}/lo", hi=f"{p}/hi",
                  neff=f"{p}/n_animals", neff_unit="animals",
                  so={"filt": nm, "window_s": 0.4, "dim": 192, "arm": "base",
                      "norm": "amplitude", "null": null}))
    a(Src("q1.window_s", R, "results/q1.json", "/cells/13/window_s", "f1", "s"))
    a(Src("param.ci_level", R, "results/q1.json", "/cells/13/vs/ar/delta/alpha", "pct0", "",
          note="rendered as the confidence level: see render fmt 'conf'"))
    a(Src("param.null_q", V, "scripts/dynamics_floor.py",
          r"re:corpus_above_null_p95.: float\(\s*np\.mean\(cv > np\.quantile\(nv, ([0-9.]+)\)",
          "pct0", "quantile of the noise null"))
    a(Src("q1.n_boot", R, "results/q1.json", "/cells/13/vs/ar/delta/n_boot", "int", "replicates"))
    a(Src("q1.dim", R, "results/q1.json", "/cells/13/dim", "int", "dimensions"))
    a(Src("q1.n_queries", R, "results/q1.json", "/cells/13/n_queries", "int", "windows"))

    # Finding 5: irreversibility (reversibility-tests)
    ST = "results/surrogate_test.json"
    a(Src("irr.white_02", RT, ST, "/guard_detail/ring/white_noise_sigma/0.2", "f2", "nats",
          atlas="reversibility-tests.results/surrogate_test/ring_guard"))
    a(Src("irr.white_03", RT, ST, "/guard_detail/ring/white_noise_sigma/0.3", "f3", "nats"))
    a(Src("irr.ring_true_02", RT, ST, "/guard_detail/ring/ring_true_state_sigma/0.2", "f2", "nats"))
    a(Src("irr.ring_obs_02", RT, ST, "/ring_sigma_obs/0.2", "f2", "nats"))
    a(Src("irr.ring_sur_02", RT, ST, "/ring_sigma_surrogate/0.2", "f2", "nats"))
    a(Src("irr.db_floor_03", RT, ST, "/guard_detail/ring/detailed_balance_floor/mean", "f3", "nats"))
    a(Src("irr.ring_dwell", RT, ST, "/guard_detail/ring/ring/mean_dwell_frames", "f1", "frames"))
    a(Src("irr.guard", RT, ST, "/ring_guard", "x", "", atlas="reversibility-tests.results/surrogate_test/ring_guard"))
    for i, K in enumerate((1, 2, 4, 12, 24)):
        a(Src(f"irr.embed_K{K}", RT, ST, f"/embedding_artifact_sweep/sweep/{i}/sigma/0.2", "f2", "nats",
              extra={"embed_k_ptr": f"/embedding_artifact_sweep/sweep/{i}/embed_k"}))

    # Finding 6: identity, direct probes
    for arm, path in (("ego", "results/ego_bodylen.json"), ("egoraw", "results/ego_raw_bodylen.json")):
        for kind in ("session", "animal"):
            for k, f in (("leak_nats", "f2"), ("ceiling_nats", "f2"), ("accuracy", "pct1"),
                         ("chance_accuracy", "pct3")):
                a(Src(f"id.{arm}.{kind}.{k}", V, path, f"/reads/identity_leak_{kind}/detail/{k}", f,
                      "nats" if "nats" in k else "accuracy",
                      atlas=f"vieb-pipeline.{path[:-5]}/reads/identity_leak_{kind}",
                      so=f"/reads/identity_leak_{kind}/scored_object",
                      neff=f"/reads/identity_leak_{kind}/n_effective",
                      neff_unit="recordings" if kind == "session" else "animals"))
    for kind in ("recording", "animal"):
        for k, f in (("leak_nats", "f2"), ("ceiling_nats", "f2")):
            a(Src(f"id.g3.{kind}.{k}", SF, "results/g3_identity.json", f"/leak/{kind}/{k}", f, "nats",
                  atlas="shapeflow.results/g3_identity", so={"arm": "g3_identity", "gamma": 0,
                                                              "split": "report"}))
        a(Src(f"id.g3gate.{kind}.leak_nats", SF, "results/gates.json",
              f"/G3_identity/detail/{kind}/leak_nats", "f2", "nats", atlas="shapeflow.results/gates",
              so={"arm": "g3_identity", "gamma": 0.3, "split": "report"}))
    for k, f in (("leak_nats", "f2"), ("ceiling_nats", "f2"), ("fraction_of_ceiling", "pct1")):
        a(Src(f"id.moseq.animal.{k}", R, "results/audit_moseq.json",
              f"/reads/identity_leak_animal/detail/{k}", f, "nats" if "nats" in k else "",
              atlas="recur.results/audit_moseq/reads/identity_leak_animal",
              so="/reads/identity_leak_animal/scored_object",
              neff="/reads/identity_leak_animal/n_effective", neff_unit="animals"))
    for k in ("accuracy", "chance_accuracy"):
        a(Src(f"id.moseq.session.{k}", R, "results/audit_moseq.json",
              f"/reads/identity_leak_session/detail/{k}", "pct3", "accuracy",
              atlas="recur.results/audit_moseq/reads/identity_leak_session",
              note="its 0.0-nat leak is a clamp (refused by the later guard, ebf7acc)"))

    # Finding 7: tracker artifacts
    TR = "results/triage_moseq.json"
    for k in ("artifact", "occlusion_linked", "clean"):
        a(Src(f"tri.mass.{k}", R, TR, f"/read/detail/frame_mass/{k}", "pct2", "fraction of frames",
              atlas="recur.results/triage_moseq/read", so="/read/scored_object",
              neff="/read/n_effective", neff_unit="animals"))
    for k in ("artifact", "occlusion_linked", "clean"):
        a(Src(f"tri.n.{k}", R, TR, f"/read/detail/categories/{k}", "int", "syllables",
              atlas="recur.results/triage_moseq/read", transform="len"))
    a(Src("tri.artifact_min", R, TR, "/read/detail/artifact_min", "f1", "x enrichment"))
    a(Src("tri.base_rate", R, TR, "/read/detail/base_rate", "pct2", "flagged frames"))
    a(Src("tri.min_stratum", R, TR, "/read/detail/min_stratum", "int", "frames"))
    a(Src("tri.superseded", "atlas", "claim.inventory.contamination_176_live", "/value", "pct1",
          "fraction of frames", allow=("SUPERSEDED",),
          note="unstratified figure, superseded by claim.inventory.contamination_282"))

    # Supplements
    MO = "results/montage_moseq.json"
    for lab in ("0", "1"):
        for k, f in (("frame_mass", "pct1"), ("event_mass", "pct1"), ("n_runs", "int")):
            a(Src(f"mass.{lab}.{k}", R, MO, f"/selection/detail/{k}/{lab}", f, "",
                  atlas="recur.results/montage_moseq/selection", allow=("PROVISIONAL",),
                  note="PROVISIONAL pending the four-gate restricted re-run"))
    SEP = "results/seg_validate_separability_rawonly.json"
    for k in ("white", "phase", "var5", "ou"):
        a(Src(f"sep.{k}", V, SEP, f"/results/{k}/auc", "f3", "AUC",
              atlas=f"vieb-pipeline.results/seg_validate_separability_rawonly/reads/{k}",
              so=f"/reads/{k}/scored_object", neff=f"/reads/{k}/n_effective", neff_unit="animals"))
    for k in ("white", "phase", "var5", "ou"):
        a(Src(f"seprough.{k}", V, "results/seg_validate_separability.json", f"/results/{k}/auc", "f3",
              "AUC", atlas=f"vieb-pipeline.results/seg_validate_separability/reads/{k}",
              so=f"/reads/{k}/scored_object", neff=f"/reads/{k}/n_effective", neff_unit="animals",
              note="with the per-channel log mean-squared first difference added"))
    PL = "results/plant.json"
    for o in (1, 2):
        for amp in (1, 2, 4, 8, 16, 32):
            k = f"corpus|o{o}|a{amp}"
            a(Src(f"plant.o{o}.a{amp}", V, PL, f"/reads/{k}/detail/recall/point", "pct1", "recall",
                  atlas=f"vieb-pipeline.results/plant/reads/{k}", lo=f"/reads/{k}/detail/recall/lo",
                  hi=f"/reads/{k}/detail/recall/hi", so=f"/reads/{k}/scored_object",
                  neff=f"/reads/{k}/n_effective", neff_unit="animals"))
    a(Src("plant.chance_o2_a16", V, PL, "/reads/corpus|o2|a16/detail/chance/point", "pct1", "recall"))
    a(Src("plant.n_instances", V, PL, "/reads/corpus|o2|a16/detail/n_instances", "int", "planted breaks"))
    DF = "results/dynamics_floor.json"
    for d_ in ("pole_radius", "pole_frequency_hz", "band_slow", "band_mid", "band_marginal", "band_fast"):
        a(Src(f"dyn.{d_}", V, DF, f"/descriptor_separation/{d_}/corpus_above_null_p95", "pct1",
              "windows above the noise null's p95", neff="/n_animals", neff_unit="animals"))
    a(Src("dyn.f_c", V, "results/inherited/sf_calibration.json", "/f_c_hz", "f2", "Hz",
          atlas="vieb-pipeline.results/inherited/sf_calibration"))
    # Irreversibility design (stored with the run)
    a(Src("irr.n_regions", RT, ST, "/observed_meta/partition/n_regions_used", "int", "regions"))
    a(Src("irr.reduced_dim", RT, ST, "/observed_meta/reduced_dim", "int", "components"))
    for i in range(4):
        a(Src(f"irr.tau{i}", RT, ST, f"/guard_detail/ring/per_tau/{i}/tau_s", "f1", "s"))
    a(Src("irr.db_tau", RT, ST, "/guard_detail/ring/detailed_balance_floor/tau_s", "f1", "s"))
    for i, K in enumerate((1, 2, 4, 12, 24)):
        a(Src(f"irr.K{K}", RT, ST, f"/embedding_artifact_sweep/sweep/{i}/embed_k", "int", "frames"))
    # Recurrence machinery control: white noise against a second draw of white
    a(Src("q1.white_vs_white", R, "results/q1.json", "/cells/13/white_vs_white/delta/point", "spct4",
          "excess rate", lo="/cells/13/white_vs_white/delta/lo", hi="/cells/13/white_vs_white/delta/hi",
          neff="/cells/13/white_vs_white/delta/n_animals", neff_unit="animals"))
    # Descriptor bands (vieb/seg/descriptors.py BANDS)
    for nm, rx in (("slow_lo", r'\("slow", ([0-9.]+),'), ("slow_hi", r'\("slow", [0-9.]+, ([0-9.]+)\)'),
                   ("mid_hi", r'\("mid", [0-9.]+, ([0-9.]+)\)'),
                   ("marginal_hi", r'\("marginal", [0-9.]+, ([0-9.]+)\)'),
                   ("fast_hi", r'\("fast", [0-9.]+, ([0-9.]+)\)')):
        a(Src(f"param.band_{nm}", V, "vieb/seg/descriptors.py", "re:" + rx, "f1", "Hz"))

    # Method parameters stated in committed code
    a(Src("param.dwell_alpha", R, "scripts/dwell_moseq.py", r"re:\(ps < ([0-9.]+)\)\.sum\(\)", "f2",
          "per-animal test level"))
    a(Src("param.var_order", R, "recur/recurrence/surrogate.py",
          r"re:def ar_surrogate\(pose, rng, \*, p: int = (\d+)\)", "int", "lags",
          note="VAR order of the AR surrogate"))
    a(Src("param.plant_amp", V, PL, "/reads/corpus|o2|a16/detail/amp_sigma", "f0", "sigma-bar"))
    a(Src("param.min_count", R, "recur/seq/motif.py", r"re:MIN_COUNT\s*=\s*(\d+)", "int", "occurrences"))
    a(Src("param.leak_clamp", R, "recur/audit/leak.py", r"re:CLAMP_ACCURACY_RATIO\s*=\s*([0-9.]+)", "f1",
          "x chance"))
    return E


def build() -> list[dict[str, Any]]:
    atlas = _atlas()
    rows = [resolve(s, atlas) for s in _entries()]
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)), "duplicate keys"
    return rows


def main() -> int:
    rows = build()
    cols = ["key", "value", "lo", "hi", "unit", "scored_object", "n_effective",
            "n_effective_unit", "repo", "path", "pointer", "commit", "atlas_id",
            "atlas_status", "transform", "note"]
    with open(os.path.join(HERE, "provenance.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "values.json"), "w", encoding="utf-8") as fh:
        json.dump({r["key"]: r for r in rows}, fh, indent=1, default=str)
    print(f"{len(rows)} entries -> paper1/provenance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
