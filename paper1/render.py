"""Render paper1/draft.src.md into paper1/draft.md, filling every number from S1.

    python3 paper1/render.py

Tokens: ``{{key}}`` (the entry's own format), ``{{key|fmt}}``, ``{{key.ci}}``
(``[lo, hi]``), ``{{key.lo}}``, ``{{key.hi}}``, ``{{key.n}}`` (n_effective).
Rendered spans are written to ``paper1/draft.spans.json`` for `check.py`.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = re.compile(r"\{\{([A-Za-z0-9_.|]+)\}\}")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt(v: Any, f: str) -> str:
    if f == "x":
        return str(v)
    x = float(v)
    if f == "int":
        return f"{int(round(x)):,}"
    if f.startswith("f"):
        return f"{x:.{int(f[1:])}f}"
    if f.startswith("pct"):
        return f"{100 * x:.{int(f[3:])}f}%"
    if f.startswith("spct"):
        return f"{100 * x:+.{int(f[4:])}f}%".replace("-", "−")
    if f == "conf":                 # an alpha shown as its confidence level
        return f"{100 * (1 - x):.0f}%"
    if f == "ord":                  # a quantile shown as an ordinal percentile
        n = int(round(100 * x))
        return f"{n}th"
    if f == "sci":
        e = int(round(__import__("math").log10(x)))
        assert abs(x - 10 ** e) < 1e-9 * x, x
        return "10" + str(e).translate(SUP)
    raise ValueError(f)


def values() -> dict[str, dict]:
    with open(os.path.join(HERE, "values.json"), encoding="utf-8") as fh:
        out: dict[str, dict] = json.load(fh)
    return out


def render_text(src: str, vals: dict[str, dict]) -> tuple[str, list[tuple[int, int, str]]]:
    out, spans, pos = [], [], 0
    n = 0
    for m in TOKEN.finditer(src):
        out.append(src[pos:m.start()])
        n += len(src[pos:m.start()])
        tok = m.group(1)
        key, _, f_over = tok.partition("|")
        field = None
        for suf in (".ci", ".lo", ".hi", ".n"):
            if key.endswith(suf):
                key, field = key[: -len(suf)], suf[1:]
        if key not in vals:
            raise SystemExit(f"unknown key {key!r} in token {{{{{tok}}}}}")
        e = vals[key]
        f = f_over or e["fmt"]
        if field is None:
            s = fmt(e["value"], f)
        elif field == "ci":
            s = f"[{fmt(e['lo'], f)}, {fmt(e['hi'], f)}]"
        elif field in ("lo", "hi"):
            s = fmt(e[field], f)
        else:
            s = fmt(e["n_effective"], "int")
        out.append(s)
        spans.append((n, n + len(s), tok))
        n += len(s)
        pos = m.end()
    out.append(src[pos:])
    return "".join(out), spans


def main() -> int:
    with open(os.path.join(HERE, "draft.src.md"), encoding="utf-8") as fh:
        src = fh.read()
    text, spans = render_text(src, values())
    with open(os.path.join(HERE, "draft.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(HERE, "draft.spans.json"), "w", encoding="utf-8") as fh:
        json.dump(spans, fh)
    print(f"draft.md: {len(spans)} values rendered from S1, {len(text.split()):,} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
