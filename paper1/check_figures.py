"""Every number printed on a figure, outside its axis ticks, is an S1 value.

    python3 paper1/check_figures.py    # exit 1 otherwise

Reads the SVGs (text kept as text, `svg.fonttype = none`). Tick labels live in
groups whose id starts with `xtick`/`ytick`; they are axis scales, not results.
Every other number must appear among the strings `make_figures.py` rendered
from S1 for that figure (`figures/annotated.json`).
"""
from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402

NUM = re.compile(r"[+−-]?\d[\d,]*(?:\.\d+)?%?")
NS = "{http://www.w3.org/2000/svg}"


def texts(node, in_tick=False):
    gid = node.get("id", "")
    tick = in_tick or gid.startswith(("xtick", "ytick"))
    if node.tag == NS + "text":
        yield "".join(node.itertext()), tick
    for ch in node:
        yield from texts(ch, tick)


def main() -> int:
    with open(os.path.join(HERE, "figures", "annotated.json"), encoding="utf-8") as fh:
        ann = json.load(fh)
    vals = render.values()
    bad = 0
    for fig in sorted(f[:-4] for f in os.listdir(os.path.join(HERE, "figures")) if f.endswith(".svg")):
        key = fig.split("_")[0]
        allowed = {render.fmt(vals[k]["value"], f) for k, f in
                   ((a[0], a[1]) if isinstance(a, list) else (a, vals[a]["fmt"]) for a in ann.get(key, []))}
        allowed_nums = {n for a in allowed for n in NUM.findall(a)}
        root = ET.parse(os.path.join(HERE, "figures", f"{fig}.svg")).getroot()
        for s, tick in texts(root):
            if tick:
                continue
            for n in NUM.findall(s):
                if n not in allowed_nums:
                    print(f"{fig}: {n!r} in {s!r} is not an S1 annotation")
                    bad += 1
    print(f"{bad} unsourced numbers on figures")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
