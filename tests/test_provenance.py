"""`vieb.provenance.header`: this repository's commit, recur's beside it."""
from __future__ import annotations

import subprocess

from recur import anchors

from vieb import provenance


def test_git_sha_is_this_repositorys_head_and_recurs_is_kept():
    h = provenance.header(anchors.LUNA, unverified="a test", stage="t")
    head = subprocess.run(["git", "-C", provenance.REPO, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert h["env"]["git_sha"].split("-")[0] == head
    ref = anchors.header(anchors.LUNA, unverified="a test", stage="t")
    assert h["env"]["recur_git_sha"] == ref["env"]["git_sha"]
    assert h["env"]["git_sha"] != ref["env"]["git_sha"]


def test_everything_else_is_anchors_header_unchanged():
    h = provenance.header(anchors.LUNA, unverified="a test", stage="t", x=1)
    ref = anchors.header(anchors.LUNA, unverified="a test", stage="t", x=1)
    for k in ref:
        if k != "env":
            assert h[k] == ref[k]
    assert h["x"] == 1
