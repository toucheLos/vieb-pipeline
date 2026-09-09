"""Freeze the inherited artifact set, once, and check it thereafter.

    python3 scripts/freeze.py            # check (the default -- safe to re-run)
    python3 scripts/freeze.py --freeze   # write results/inherited.json, once

Every job's preamble runs the check. `freeze` refuses to clobber: the inherited
set is recorded once and then only verified, because rewriting it would make
every shard that stamped the old digest refer to a provenance that no longer
exists.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.util import log                                        # noqa: E402
from vieb.io import spine                                         # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="re-freeze against a new upstream; refuses without it")
    p.add_argument("--path", default=spine.FROZEN_PATH)
    a = p.parse_args(argv)
    if a.freeze:
        doc = spine.freeze(a.path, overwrite=a.overwrite)
        log(f"froze {len(doc['json'])} JSONs and {len(doc['dirs'])} directories "
            f"at digest {doc['digest']}")
        for repo, sha in sorted(doc["git_sha"].items()):
            log(f"  {repo:11s} {sha}")
        return 0
    rd = spine.check(a.path)
    log(rd.line())
    return 0 if rd.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
