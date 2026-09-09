r"""The only module in this repo that reads `~/recur` or `~/shapeflow` **data**.

## Hash the artifacts, import the utilities

This repo does two different things with recur and they must not be confused.

It **imports** recur's utility layer -- `Read`, `boot`, `splits`, `anchors`,
`util`, `labels`, `kendall`, `geom/*`, `qc/swap`. Those are code. Forking them
would give the project two verdict vocabularies and two copies of a split that is
"fixed once and never resampled", which is the exact failure `recur.splits`
exists to prevent.

It **content-hashes** every artifact it reads. Those are data, and data is what
can change underneath a result. recur and shapeflow are both live repos with
working trees; a stage re-run under this repo's feet would change the numbers
without changing anything a reader could see. That is the characteristic failure
this project keeps paying for -- a statistic correctly computed on an object that
was not the object of interest.

**Inheriting is not trusting**, and the distinction is between the two kinds of
inheritance rather than a licence to skip either check.

## What is hashed, and what is only fingerprinted

Small frozen JSONs are **content-hashed** in full: they carry every threshold a
downstream number depends on. The npz directories hold thousands of files and
gigabytes, so they are fingerprinted by a hash over the sorted
`(basename, size)` listing -- which catches a file added, removed, truncated or
regenerated at a different length -- plus a real SHA-256 of a **seeded sample**,
which catches a rewrite that happens to preserve length. Neither is `mtime`,
because a `touch` is not a change and a copy is not a corruption.

The fingerprint is deliberately weaker than the JSON hash and this docstring says
so, so that nobody later reads "verified" and believes more than was checked.

Ported from `recur/io/spine.py`, which does the same job one repo upstream.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any

import numpy as np
from recur.read import Read
from recur.util import read_json, write_json

#: Overridable so tests can point at a fixture tree rather than the real one.
RECUR = os.environ.get("VIEB_RECUR", "/home/tul26194/recur")
SHAPEFLOW = os.environ.get("VIEB_SHAPEFLOW", "/home/tul26194/shapeflow")
EXBIAS = os.environ.get("VIEB_EXBIAS", "/home/tul26194/exbias")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Small, frozen, load-bearing: content-hashed in full. `(repo, relative path)`.
CONSUMED_JSON: dict[str, tuple[str, str]] = {
    # shapeflow: the corpus itself, and every cleaning threshold.
    "sf_ingest": ("shapeflow", "results/ingest.json"),
    "sf_splits": ("shapeflow", "results/splits.json"),
    "sf_gauge": ("shapeflow", "results/gauge.json"),
    "sf_calibration": ("shapeflow", "results/calibration.json"),
    "sf_clean": ("shapeflow", "results/clean.json"),
    "sf_clean_thresholds": ("shapeflow", "results/clean_thresholds.json"),
    # recur: the provenance chain back to shapeflow, and the swap correction.
    # NOT results/inherited.json -- recur keeps its frozen provenance inside the
    # package dir, at recur/results/, which is a different directory from the
    # repo-level results/ that holds everything else.
    "recur_inherited": ("recur", "recur/results/inherited.json"),
    "recur_swap": ("recur", "results/swap.json"),
    "recur_bilateral_unreliable": ("recur", "results/bilateral_unreliable.json"),
}

#: Large, per-recording: fingerprinted, never fully hashed.
CONSUMED_DIRS: dict[str, tuple[str, str]] = {
    "sf_clean": ("shapeflow", "work/clean"),
    "sf_repr": ("shapeflow", "work/repr"),
    "recur_swap": ("recur", "work/swap"),
    "exbias_segments": ("exbias", "segments_v2"),
}

#: The genuinely raw keypoints, before any cleaning. Phase A needs this and
#: `pose_unfiltered` is NOT it -- that array already carries the gap policy's
#: interpolants on 1.34% of keypoint-frames.
RAW_POSE = ("shapeflow", "work/raw_pose.npz")

N_SAMPLED = 8
SAMPLE_SEED = 0
FROZEN_PATH = os.path.join(REPO, "results", "inherited.json")

ROOTS = {"recur": RECUR, "shapeflow": SHAPEFLOW, "exbias": EXBIAS}
_CACHE: dict[str, Any] = {}


def path_in(repo: str, *parts: str) -> str:
    """A path inside one of the consumed repos. The only place roots are joined."""
    if repo not in ROOTS:
        raise KeyError(f"{repo!r} is not a consumed repo; one of {sorted(ROOTS)}")
    return os.path.join(ROOTS[repo], *parts)


def sf(*parts: str) -> str:
    """A path inside shapeflow. Kept for parity with recur's spine."""
    return path_in("shapeflow", *parts)


def sha256_file(path: str, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def git_sha(repo: str) -> str:
    """HEAD of a consumed repo, with a dirty marker.

    A dirty tree is **recorded rather than refused**: shapeflow's own
    `ingest.json` carries `"git_sha": "HEAD-dirty"`, so demanding cleanliness
    would refuse the artifacts this repo is built on. What matters is that the
    state is written down.
    """
    root = ROOTS[repo]
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        sha = out.stdout.strip() or "unknown"
        st = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                            capture_output=True, text=True, timeout=120)
        return sha + ("-dirty" if st.stdout.strip() else "")
    except Exception as exc:                       # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"


def _dir_fingerprint(repo: str, rel: str) -> dict:
    d = path_in(repo, rel)
    names = sorted(n for n in os.listdir(d) if n.endswith(".npz"))
    sizes = [os.path.getsize(os.path.join(d, n)) for n in names]
    listing = "\n".join(f"{n}\t{s}" for n, s in zip(names, sizes))
    rng = np.random.default_rng(SAMPLE_SEED)
    take = sorted(rng.choice(len(names), size=min(N_SAMPLED, len(names)),
                             replace=False).tolist()) if names else []
    sampled = {names[i]: sha256_file(os.path.join(d, names[i])) for i in take}
    return {
        "repo": repo, "path": rel, "n_files": len(names),
        "total_bytes": int(sum(sizes)),
        "listing_sha256": _sha256_text(listing),
        "sampled_sha256": sampled, "sample_seed": SAMPLE_SEED,
        "note": ("listing_sha256 covers (basename, size) for every file; "
                 "sampled_sha256 is a real content hash of a seeded sample. "
                 "Neither is a full content hash -- see spine.__doc__"),
    }


def provenance() -> dict:
    """Everything this repo consumes, as it stands right now."""
    doc: dict = {"roots": dict(ROOTS), "git_sha": {}, "json": {}, "dirs": {}}
    for repo in sorted(ROOTS):
        doc["git_sha"][repo] = git_sha(repo)
    for name, (repo, rel) in sorted(CONSUMED_JSON.items()):
        p = path_in(repo, rel)
        doc["json"][name] = {"repo": repo, "path": rel,
                             "bytes": os.path.getsize(p),
                             "sha256": sha256_file(p)}
    for name, (repo, rel) in sorted(CONSUMED_DIRS.items()):
        doc["dirs"][name] = _dir_fingerprint(repo, rel)
    rp = path_in(*RAW_POSE)
    doc["raw_pose"] = {"path": RAW_POSE[1], "repo": RAW_POSE[0],
                       "bytes": os.path.getsize(rp)}
    doc["digest"] = digest_of(doc)
    return doc


#: Fields recorded in the provenance but deliberately EXCLUDED from the digest.
#: See `digest_of`.
_NOT_HASHED = ("digest", "git_sha", "roots")


def digest_of(doc: dict) -> str:
    """A short hash of the consumed DATA, and only the data.

    `git_sha` and `roots` are recorded in the provenance and excluded from the
    digest, which diverges from `recur/io/spine.py` on purpose. Upstream folds
    the git sha in, so an unrelated commit to recur -- one touching no artifact
    this repo reads -- changes the digest that every shard here stamps. Two
    shards computed on byte-identical data would then carry different
    provenance labels, and a reader comparing them has to work out that the
    difference means nothing.

    The digest answers "was this computed on the same data"; the git sha answers
    "which commit produced that data". Both are worth recording. Only the first
    belongs in the label.
    """
    body = {k: v for k, v in doc.items() if k not in _NOT_HASHED}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def freeze(path: str = FROZEN_PATH, *, overwrite: bool = False) -> dict:
    """Record the consumed set once. Refuses to clobber.

    The inherited set is frozen once and then only checked; rewriting it would
    make every stamped shard refer to a provenance that no longer exists.
    """
    if os.path.exists(path) and not overwrite:
        raise SystemExit(
            f"{path} already exists. The inherited set is frozen once and then "
            f"only checked -- rewriting it would make every shard that stamped "
            f"the old digest refer to a provenance that no longer exists. Pass "
            f"overwrite=True only to re-freeze against a new upstream.")
    doc = provenance()
    write_json(doc, path)
    # Copy the consumed JSONs in, so this repo's results are self-describing: a
    # reader does not need recur or shapeflow on disk to know which thresholds a
    # number was computed under.
    side = os.path.join(os.path.dirname(path), "inherited")
    for name, (repo, rel) in CONSUMED_JSON.items():
        write_json(read_json(path_in(repo, rel)),
                   os.path.join(side, f"{name}.json"))
    return doc


def frozen(path: str = FROZEN_PATH) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run `scripts/freeze.py` first")
    return read_json(path)


def digest(path: str = FROZEN_PATH) -> str:
    """The frozen digest, cached. Every shard asks for it."""
    if "digest" not in _CACHE:
        _CACHE["digest"] = str(frozen(path)["digest"])
    return str(_CACHE["digest"])


def json_(name: str) -> dict:
    """A consumed JSON, preferring this repo's frozen copy.

    After `freeze`, this repo reads the thresholds it recorded rather than
    whatever upstream currently holds, and `check()` is what notices if the two
    have diverged.
    """
    local = os.path.join(REPO, "results", "inherited", f"{name}.json")
    if os.path.exists(local):
        return read_json(local)
    repo, rel = CONSUMED_JSON[name]
    return read_json(path_in(repo, rel))


def check(path: str = FROZEN_PATH) -> Read:
    """Have the inherited artifacts moved since they were frozen?

    Returns a `Read` rather than a bool, and `n_effective` counts **artifacts
    compared, not animals** -- this read is about provenance, not about the
    corpus, and saying so in the field is cheaper than a reader assuming
    otherwise.
    """
    want = frozen(path)
    live = provenance()
    drift: list[str] = []
    for name in sorted(CONSUMED_JSON):
        a = want["json"].get(name, {}).get("sha256")
        b = live["json"].get(name, {}).get("sha256")
        if a != b:
            drift.append(f"json {name}: {str(a)[:12]} -> {str(b)[:12]}")
    for name in sorted(CONSUMED_DIRS):
        wa, wb = want["dirs"].get(name, {}), live["dirs"].get(name, {})
        if wa.get("listing_sha256") != wb.get("listing_sha256"):
            drift.append(f"dir {name}: listing changed "
                         f"({wa.get('n_files')} -> {wb.get('n_files')} files)")
        for f, h in (wa.get("sampled_sha256") or {}).items():
            if (wb.get("sampled_sha256") or {}).get(f) != h:
                drift.append(f"dir {name}: sampled file {f} rewritten")
    n_eff = len(CONSUMED_JSON) + len(CONSUMED_DIRS)
    obj = {"dataset": "luna", "arm": "inherited", "split": "all",
           "roots": ROOTS}
    detail = {"drift": drift[:6], "n_drift": len(drift),
              "frozen_digest": want.get("digest"),
              "live_digest": live.get("digest"),
              "git_sha": live.get("git_sha")}
    if drift:
        more = f" (+{len(drift) - 6} more)" if len(drift) > 6 else ""
        return Read(
            "FAIL", obj,
            f"the inherited artifacts have moved under this repo since they "
            f"were frozen: {len(drift)} difference(s) -- "
            + "; ".join(drift[:6]) + more +
            f". Every number this repo has written was computed against digest "
            f"{want.get('digest')}, and upstream now hashes to "
            f"{live.get('digest')}",
            n_effective=n_eff, detail=detail)
    return Read(
        "PASS", obj,
        f"all {len(CONSUMED_JSON)} consumed JSONs match their frozen SHA-256, "
        f"and all {len(CONSUMED_DIRS)} npz directories match their "
        f"(basename, size) listing and their {N_SAMPLED}-file seeded content "
        f"sample. That is a fingerprint and not a full content hash -- a "
        f"rewrite that preserves every filename, every size and all "
        f"{N_SAMPLED} sampled files would pass. Digest {want.get('digest')}",
        n_effective=n_eff, detail=detail)


# --------------------------------------------------------------------------
# Accessors -- delegated, not duplicated
# --------------------------------------------------------------------------
#
# Reading `work/clean/<rid>.npz` into a dict is a utility, and recur already has
# it. This repo owns the *hashing* of that file, not the `np.load` around it, so
# these delegate. The one exception is `raw_pose`, which recur has no per-
# recording accessor for and Phase A cannot do without.

def _recur_spine():                                       # noqa: ANN202
    from recur.io import spine as rs
    # recur's spine resolves shapeflow through its OWN env var. Point it at the
    # same root this repo froze, or the two would silently read different trees.
    rs.SHAPEFLOW = SHAPEFLOW
    return rs


def recording_ids() -> list:
    """Canonical corpus order, from the stored id array.

    Not `sorted(os.listdir(...))`: an order that depends on the environment is
    an order that will eventually disagree with a shard index written on another
    node.
    """
    return list(_recur_spine().recording_ids())


def fps() -> float:
    return float(_recur_spine().fps())


def clean(recording_id: str) -> dict:
    """`work/clean/<rid>.npz` -- pose, pose_butterworth, pose_unfiltered, conf,
    missing, interpolated, low_confidence, bone_flagged, fps."""
    return dict(_recur_spine().clean(recording_id))


def representation(recording_id: str) -> dict:
    """`work/repr/<rid>.npz` -- x, y, usable, mu, log_s, theta, phi, speed, omega."""
    return dict(_recur_spine().representation(recording_id))


def raw_pose(recording_id: str | None = None) -> dict:
    r"""The genuinely raw keypoints and DLC confidences, before any cleaning.

    **`pose_unfiltered` is not this.** That array is the gap policy's OUTPUT:
    ~1.34% of its keypoint-frames are linear interpolants. Measuring the gap
    policy against it would report exactly zero, which is the one number that
    cannot be right.

    With no argument, returns the whole corpus dict (1.88 GB). With a recording
    id, slices that recording out using the stored `bounds`.
    """
    key = "_raw_pose"
    if key not in _CACHE:
        _CACHE[key] = np.load(path_in(*RAW_POSE), allow_pickle=False)
    z = _CACHE[key]
    if recording_id is None:
        return {k: z[k] for k in z.files}
    ids = [str(v) for v in z["recording_ids"]]
    try:
        r = ids.index(str(recording_id))
    except ValueError:
        raise KeyError(f"{recording_id!r} is not in raw_pose.npz") from None
    lo, hi = int(z["bounds"][r]), int(z["bounds"][r + 1])
    return {"pose": z["pose"][lo:hi], "conf": z["conf"][lo:hi],
            "recording_id": str(recording_id), "bounds": (lo, hi),
            "fps": float(z["fps"])}
