# Task A — seed and hash provenance audit

Prompted by F2, which found the injection benchmark's per-recording corruption
seed built from `abs(hash((SEED, tag, rid))) % 2**32`. **Python salts `hash()` on
`str` per interpreter**, so `SEED = 0` controlled nothing and the
pre-registration's claim that the benchmark was seeded was false.

Audited as a class, not an instance, because the programme's central provenance
claim — every number traces to a shard hash — fails outright if shard identity
is keyed on `hash()` of a string.

## Verdict

**The shard-hash architecture is real.** It was never built on `hash()`. The
animal bootstrap and the Q1 surrogate generators are clean. One load-bearing site
was found in this repository and fixed; one was found in `recur` and is recorded
rather than changed, for a stated reason.

**No halt condition was triggered.**

## Method

`grep -rn 'hash('` across `~/vieb-pipeline` and `~/recur`, plus
`PYTHONHASHSEED`, and every `default_rng(` expression. Each candidate classified
**by demonstration** — the expression run in three separate interpreters, values
compared — not by reading.

The premise itself is pinned by a test
(`test_python_hash_really_is_salted`): if `PYTHONHASHSEED` is ever set in this
environment, the audit's basis disappears and the test says so rather than
passing vacuously.

## Every site

| site | expression | 3 subprocesses | class | action |
|---|---|---|---|---|
| `scripts/bones.py:143` | `abs(hash(args.animal)) % 2**31` | `1112872326`, `1698936080`, `64086395` — **differ** | **salted, load-bearing** | replaced with `seeds.stable_seed` |
| `scripts/injection.py` (Phase F) | `abs(hash((SEED, tag, rid)))` | differ (found in F2) | **salted, load-bearing** | replaced, now via the shared helper |
| `recur/scripts/motif_clips.py:193` | `hash(gram) % 10_000` | `7250`, `7623`, `4283` — **differ** | **salted, load-bearing** (filenames only) | **recorded, not changed** — see below |
| `recur/null/microstate.py:89` `input_hash` | `hashlib.sha256` over array bytes | stable by construction | not salted | none |
| `vieb/io/spine.py` — the shard digest | `sha256_file`, `_sha256_text`, `SAMPLE_SEED = 0` | stable by construction | not salted | none |
| `recur/boot.py` — animal bootstrap | `default_rng(int(seed))` | integer seed | not salted | none |
| `recur/null/microstate.py:118` — microstate null | `default_rng(int(seed))` | integer seed | not salted | none |
| `scripts/q1.py:179,228` — Q1 surrogates | `default_rng(args.seed)` | integer seed | not salted | none |

## The three named paths

**Animal bootstrap — clean.** `recur/boot.py` takes `seed: int = SEED` and calls
`np.random.default_rng(int(seed))`. Every interval in F2, and every animal
bootstrap in this programme, is reproducible.

**Shard identity — clean, and this is the important one.** `vieb/io/spine.py`
derives the inherited digest from `hashlib.sha256` over file contents
(`sha256_file`) and over the sorted listing (`_sha256_text`), with the sampling
RNG seeded by the integer constant `SAMPLE_SEED = 0`. There is no `hash()` in
the digest path. **"Hash the artifacts, import the utilities" was implemented as
stated**, and digest `198eb14ff258c7f6` means what it has always claimed.

**Q1 surrogates — clean.** `scripts/q1.py` and `recur/null/microstate.py` seed
from integers passed on the command line. `input_hash`, which names the fitted
partition a null used, is sha256 over the centroid/scale/column bytes — written
that way deliberately, and its own docstring says why: "a control whose
reproducibility depends on a seed being remembered is not a frozen control."

## Blast radius of the two salted sites

### `scripts/bones.py` — which recordings entered the shuffled-keypoint ceiling

The seed selects `ceiling_for`, the per-animal subsample of recordings on which
the shuffled-keypoint ceiling is computed (`config.CEILING_PER_ANIMAL`). It does
**not** touch the observed violation rates, the reference lengths, the ε sweep,
or any other read in `bones.json`.

Affected artifact: `results/bones.json` → `reads.shuffled_ceiling`, currently
PASS at "2.1341% observed against a 79.5914% ceiling, a margin of **37.3×**".

**The number is not reproducible; the conclusion is not in doubt.** A 37.3×
margin does not turn on which subsample of recordings was shuffled. Recorded
here rather than re-run: re-running `bones` is a corpus job, the verdict cannot
move at that margin, and this prompt's guard says a parameter that looks wrong
gets recorded, not changed to improve an outcome. The ceiling figure should be
read as "≈80% under a non-reproducible subsample" until `bones` is next re-run
for another reason, at which point it becomes reproducible for free.

### `recur/scripts/motif_clips.py` — blinded clip filenames

`blind.blind_id("ngram", hash(gram) % 10_000, "n_animals")` names a tiled motif
video. It is salted, and it does reach a committed artifact — the filename.

**Deliberately not changed.** The identity is not lost: the manifest records
`gram` explicitly beside every entry, so the mapping from filename to n-gram is
recoverable from the artifact itself. Changing the expression would give future
runs different filenames from the 1,711 clips already published and already
referenced by the Atlas manifest — breaking working links to fix a label that is
not ambiguous. The cost exceeds the benefit and the reasoning is recorded so the
next reader does not have to re-derive it.

## What was changed

`vieb/seeds.py` — one helper, `stable_seed(*parts, modulus)`, blake2b over the
parts joined by `\x00`. The separator matters: without it `("a","b")` and
`("ab",)` collide, and two recordings would silently share a corruption layout.

Both live sites now route through it. `tests/test_seeds.py` holds a `SITES`
registry and asserts **in three fresh interpreters** that each agrees with
itself. These must be subprocess tests — within one interpreter a salted hash is
perfectly stable, which is precisely why the original bug survived review.

## Why this was easy to miss, recorded so it is missed less next time

`hash()` on an **int** or a **float** is stable by construction — `hash(7) == 7`
— so a numeric key is genuinely safe and reads identically to a string key at a
glance. The failure needs a string, a fresh process, and someone comparing two
runs. None of those is present in a code review or in a single-session test.
