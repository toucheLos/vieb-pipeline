r"""Seeds that are the same in every process. One helper, because it was not.

Phase F seeded each recording's injected corruption with

    np.random.default_rng(abs(hash((SEED, tag, rid))) % (2 ** 32))

and **Python salts `hash()` on `str` per interpreter** unless `PYTHONHASHSEED`
is set. So `SEED = 0` controlled nothing: every run drew a different corruption
layout, and the pre-registration's claim that the benchmark was seeded was false.
Three subprocesses given identical inputs return three different values.

The bug is a *class*, not an instance. Any seed, shard key, cache key or artifact
identity derived from `hash()` of a string was never reproducible, and this
programme's central provenance claim is that every number traces to a shard hash.
`SEED_AUDIT.md` records the sweep of every such site.

What is safe, and why `hash()` looked safe:

* `hash()` on **ints** and **floats** is stable by construction -- `hash(7) == 7`
  -- so a numeric key is fine and reads identically to a string key at a glance.
  That is what makes this easy to miss in review.
* `hashlib` digests are stable across processes, machines and Python versions.
  `recur`'s `null/microstate.py:input_hash` already used sha256 for exactly this
  reason, and `vieb/io/spine.py` uses sha256 throughout -- so the shard-hash
  architecture was never built on `hash()`.

blake2b rather than sha256 only because it is faster and the digest is truncated
to 8 bytes anyway; neither is a cryptographic requirement here, and a seed does
not need collision resistance so much as it needs to be the same tomorrow.
"""
from __future__ import annotations

import hashlib

#: Seeds are consumed by `np.random.default_rng`, which accepts any non-negative
#: int but is conventionally given something that fits a 32-bit word here.
_MODULUS = 2 ** 32


def stable_seed(*parts: object, modulus: int = _MODULUS) -> int:
    """A reproducible seed from any parts, joined by a separator that cannot
    appear in the encoded form of a part.

    Parts are rendered with `str()` and joined with `\\x00`, so
    `stable_seed("a", "b")` and `stable_seed("ab")` differ -- a plain
    concatenation would collide them and quietly give two different runs the same
    corruption layout.
    """
    raw = "\x00".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") % int(modulus)
