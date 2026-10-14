"""Re-check every number in paper1/draft.md against S1, by script.

    python3 paper1/check.py      # exit 1 on any number that did not come from S1

1. `draft.md` must equal a fresh render of `draft.src.md` from `values.json`,
   so no number was typed into the rendered file by hand.
2. Every digit run in `draft.md` must lie inside a rendered S1 span, or be one
   of the structural references listed in ALLOWED (figure, table, section and
   finding numbers, citation years), or sit in the References section.
3. Every S1 value carries a commit; `sources.py` already refused dirty files and
   disallowed atlas statuses when values.json was built.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
#: Structural numbers: they label parts of the paper, they are not results.
ALLOWED = [
    re.compile(r"(?:Figure|Fig\.|Table|Box|Supplementary (?:Figure|Table|Note)|Finding|Rule|"
               r"Section|§)\s*S?\d+(?:[a-z]|[–-]\d+)?"),
    re.compile(r"\bS\d+\b"),
    re.compile(r"^#{1,6} .*$", re.M),                     # headings
    re.compile(r"\((?:[A-Z][^()]*?,\s*)?(?:19|20)\d\d[a-z]?\)"),   # (Author, 2025)
    re.compile(r"et al\.,? (?:19|20)\d\d"),
    re.compile(r"^\s*\d+\.\s", re.M),                     # ordered-list markers
    re.compile(r"\[\d+(?:[,–-]\s*\d+)*\]"),               # numeric citations
    re.compile(r"`[^`\n]+`"),                            # code spans: file names, not results
]


def main() -> int:
    with open(os.path.join(HERE, "draft.src.md"), encoding="utf-8") as fh:
        src = fh.read()
    fresh, spans = render.render_text(src, render.values())
    with open(os.path.join(HERE, "draft.md"), encoding="utf-8") as fh:
        text = fh.read()
    if fresh != text:
        print("FAIL: draft.md differs from a fresh render of draft.src.md (hand edit?)")
        return 1
    refs = text.find("\n## References")
    covered = [False] * len(text)
    for a, b, _t in spans:
        for i in range(a, b):
            covered[i] = True
    for rx in ALLOWED:
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                covered[i] = True
    bad = []
    for m in NUMBER.finditer(text):
        if refs >= 0 and m.start() > refs:
            continue
        if not all(covered[i] for i in range(m.start(), m.end())):
            line = text.count("\n", 0, m.start()) + 1
            ctx = text[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")
            bad.append((line, m.group(), ctx))
    vals = render.values()
    no_commit = [k for k, v in vals.items() if not v.get("commit")]
    for line, num, ctx in bad:
        print(f"line {line}: {num!r} not from S1 -- ...{ctx}...")
    if no_commit:
        print(f"S1 entries without a commit: {no_commit}")
    print(f"{len(spans)} rendered values; {len(bad)} unsourced numbers; "
          f"{len(no_commit)} entries without commit")
    return 1 if (bad or no_commit) else 0


if __name__ == "__main__":
    raise SystemExit(main())
