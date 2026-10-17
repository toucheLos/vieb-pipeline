"""STEP 4 / Part B: the session protocol, as the file names state it.

    python3 scripts/protocol_record.py      # -> results/step4/protocol.json

Reads every recording id (the spine) and every file in the raw-video directory,
parses them with `recur.labels`, and records:
- the design as named: phases, days, contexts and the explicit No_Shock tag;
- the per-animal schedule and the animals that deviate from the majority;
- files in the raw-video directory that are not corpus videos, by class;
- whether the video containers carry a usable recording time. They do not:
  `creation_time` is checked on a seeded sample of videos against the date in
  the name, and so is the file mtime.

Nothing about shock is inferred, apart from the Day-0 shock implied by the Day-1
No_Shock tag, which is recorded as implied.
"""
from __future__ import annotations

import collections
import datetime as dt
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab                            # noqa: E402
from recur.render import video as vid                               # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.tok import config                                         # noqa: E402

OUT = os.path.join(config.REPO, "results", "step4", "protocol.json")
N_PROBE = 100
SEED = 0


def lab_sheet_summary(path: str) -> dict:
    """Aggregate counts from the lab's Animal_IDs.xlsx (first sheet), and its join.

    Only aggregates are recorded: the per-animal rows are the lab's data and are
    not copied into this repository. Read with the standard library (the
    environment has no openpyxl).
    """
    import hashlib
    import xml.etree.ElementTree as ET
    import zipfile
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        ss = ["".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        names = [e.get("name") for e in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", ns)]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).find("m:sheetData", ns):
            vals = []
            for c in row:
                v = c.find("m:v", ns)
                vals.append((ss[int(v.text)] if c.get("t") == "s" else v.text) if v is not None else "")
            rows.append(vals)
    hdr, body = rows[0], [r for r in rows[1:] if r and r[0]]
    ids = {str(r[0]).strip().removesuffix(".0") for r in body}
    tags = {lab.animal_tag(i) for i in spine.recording_ids()}
    out: dict = {"path": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                 "sheets": names, "sheet_used": names[0], "n_rows": len(body),
                 "n_ids_in_corpus": len(ids & tags), "n_corpus_animals_missing": len(tags - ids)}
    for j, h in enumerate(hdr[1:5], 1):
        c = collections.Counter((r[j] if j < len(r) else "").strip() for r in body)
        out[h] = dict(c.most_common())
    out["note"] = ("aggregates only; values as written in the sheet (spelling variants such as "
                   "Veh/Vehicle and BL6x129/Bl6x129 are not merged); no drug timing or dose and "
                   "nothing about shock in the sheet")
    return out


def creation_time(path: str) -> str | None:
    r = subprocess.run([vid.FFPROBE, "-v", "quiet", "-show_entries", "format_tags=creation_time",
                        "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    s = r.stdout.strip()
    return s or None


def main() -> int:
    ids = spine.recording_ids()
    rows = lab.parse_all(ids)
    sched = lab.schedule(rows)
    maj = lab.majority_schedule(rows)
    dev = lab.deviants(rows)

    # Design as named
    phases: dict = {}
    for r in rows:
        phases.setdefault(r["protocol"], set()).add(r["day"])
    contexts_by_day: dict = {}
    for r in rows:
        contexts_by_day.setdefault(r["day"], set()).add(r["context"])
    same_date = collections.Counter()
    by_animal_day: dict = {}
    for r in rows:
        by_animal_day.setdefault((r["animal_tag"], r["day"]), set()).add(r["date"])
    for (a, d), dates in by_animal_day.items():
        if d >= 3:
            same_date["one_date" if len(dates) == 1 else "two_dates"] += 1

    # Raw-video directory: what is and is not a corpus video
    files = sorted(os.listdir(vid.VIDEO_ROOT))
    ids_norm = {lab.normalize_recording_id(i) for i in ids}
    classes = collections.Counter()
    nonmatching = []
    for f in files:
        try:
            lab.parse(f)
            ok = lab.normalize_recording_id(f) in ids_norm
        except ValueError:
            ok = False
        ext = f.rsplit(".", 1)[-1] if "." in f else ""
        kind = ("corpus_video" if f.endswith(".mp4") and ok else
                "dlc_output" if "DLC_" in f and ok else "other")
        classes[kind] += 1
        if kind == "other":
            nonmatching.append(f)

    # Container timestamps against the name's date
    rng = random.Random(SEED)
    sample = rng.sample(sorted(ids), N_PROBE)
    probe = []
    for rid in sample:
        p = vid.video_path(rid)
        ct = creation_time(p)
        name_date = dt.datetime.strptime(lab.parse(rid)["date"], "%Y%m%d").date()
        ct_date = dt.datetime.fromisoformat(ct.replace("Z", "+00:00")).date() if ct else None
        mt_date = dt.datetime.fromtimestamp(os.stat(p).st_mtime).date()
        probe.append({"recording_id": rid, "name_date": str(name_date),
                      "creation_time": ct, "creation_matches_name_date": ct_date == name_date,
                      "mtime_date": str(mt_date), "mtime_matches_name_date": mt_date == name_date})
    n_ct = sum(1 for p in probe if p["creation_time"])
    n_ct_match = sum(1 for p in probe if p["creation_matches_name_date"])
    n_mt_match = sum(1 for p in probe if p["mtime_matches_name_date"])

    write_json({**provenance.header(anchors.LUNA, stage="protocol_record",
                                    unverified="source: the file-naming schema only"),
                "source": "file-naming schema: <date>_Box_<box>_<phase>_Day_<day>_(<context>)_<animal>",
                "n_recordings": len(rows), "n_animals_tag": len(sched),
                "n_animals_key": len({r["animal"] for r in rows}),
                "phases": {k: sorted(v) for k, v in sorted(phases.items())},
                "phase_day_min": {k: min(v) for k, v in phases.items()},
                "phase_day_max": {k: max(v) for k, v in phases.items()},
                "contexts_by_day": {str(k): sorted(v) for k, v in sorted(contexts_by_day.items())},
                "design_table": lab.design_table(rows),
                "no_shock_tagged": {"n": sum(r["no_shock_tagged"] for r in rows),
                                    "days": sorted({r["day"] for r in rows if r["no_shock_tagged"]})},
                "day0_shock": "IMPLIED by the Day-1 No_Shock tag; not stated in any name",
                "cfd_same_date": dict(same_date),
                "majority_schedule": [list(k) for k in maj],
                "n_majority": sum(1 for s in sched.values() if s == maj),
                "deviants": {a: {"missing": [list(m) for m in v[0]], "extra": [list(e) for e in v[1]]}
                             for a, v in dev.items()},
                "raw_video_dir": vid.VIDEO_ROOT, "raw_video_dir_classes": dict(classes),
                "raw_video_dir_nonmatching": nonmatching,
                "timestamp_probe": {"n": len(probe), "seed": SEED,
                                    "n_with_creation_time": n_ct,
                                    "n_creation_matches_name_date": n_ct_match,
                                    "n_mtime_matches_name_date": n_mt_match,
                                    "reliability": ("creation_time and mtime are copy/re-encode times, "
                                                    "not recording times, wherever they disagree with "
                                                    "the name's date; within-day order is not recoverable"
                                                    if n_ct_match + n_mt_match < 2 * len(probe) else
                                                    "matches the name date on the sample"),
                                    "rows": probe},
                "lab_animal_sheet": lab_sheet_summary(os.path.join(vid.VIDEO_ROOT, "Animal_IDs.xlsx")),
                "not_in_the_names": ["whether Context A sessions on Days 3-7 deliver shock",
                                     "shock count and timing (s into session)",
                                     "whether the within-day order of A and B was counterbalanced",
                                     "the within-day order itself",
                                     "what CFC and CFD stand for"]},
               OUT)
    log(f"{len(rows)} recordings, {len(sched)} animals; {len(dev)} deviants; files {dict(classes)}; "
        f"creation_time matches name date {n_ct_match}/{len(probe)}, mtime {n_mt_match}/{len(probe)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
