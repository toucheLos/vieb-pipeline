"""The result-JSON header, stamped with THIS repository's commit.

`recur.anchors.header` records `env.git_sha` from `recur.anchors.env_record`,
which reads the HEAD of `~/recur` -- the repository the helper lives in, not the
one whose code produced the result. Every vieb-pipeline JSON written before this
module carried recur's commit as its provenance. `header` here is anchors'
header, unchanged, except that `env.git_sha` becomes this repository's HEAD (with
the same `-dirty` marker convention as `spine.git_sha`) and recur's is kept
beside it as `env.recur_git_sha`, because the helper code is part of the run too.

Earlier JSONs are not rewritten: their `git_sha` is what was recorded at the
time, and it is read as recur's commit.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pipeline_sha() -> str:
    """HEAD of vieb-pipeline, `-dirty` when the tree has uncommitted changes."""
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        sha = out.stdout.strip() or "unknown"
        st = subprocess.run(["git", "-C", REPO, "status", "--porcelain",
                             "--untracked-files=no"],
                            capture_output=True, text=True, timeout=120)
        return sha + ("-dirty" if st.stdout.strip() else "")
    except Exception as exc:                       # noqa: BLE001
        return f"unknown ({type(exc).__name__})"


def header(anchor: Any, **kwargs: Any) -> dict[str, Any]:
    """`recur.anchors.header`, with this repository's commit as `env.git_sha`."""
    from recur import anchors

    head: dict[str, Any] = anchors.header(anchor, **kwargs)
    env = dict(head.get("env") or {})
    env["recur_git_sha"] = env.get("git_sha")
    env["git_sha"] = pipeline_sha()
    head["env"] = env
    return head
