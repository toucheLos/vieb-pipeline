"""STABILISE 7: saved masks round-trip; the animal view reads the |a series."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))


def test_packed_masks_round_trip():
    import stabilise7 as s7
    rng = np.random.default_rng(0)
    crops = {3 * k: (5 * k, 7 * k, rng.random((11 + k, 13 + k)) > 0.5)
             for k in range(5)}
    p = s7._packed_masks({"key_crops": crops})
    for i, t in enumerate(p["mask_t"]):
        x0, y0, h, w = p["mask_box"][i]
        bits = p["mask_bits"][p["mask_offsets"][i]:p["mask_offsets"][i + 1]]
        m = np.unpackbits(bits)[:h * w].reshape(h, w).astype(bool)
        assert (x0, y0) == crops[int(t)][:2]
        assert np.array_equal(m, crops[int(t)][2])


def test_animal_view_replaces_plain_keys_only_where_an_animal_twin_exists():
    import stabilise7 as s7
    z = {"SM6__head__0.6": 1, "SM6__head__0.6__a": 2, "K__head__0.6": 3}
    v = s7._animal_view(z)
    assert v["SM6__head__0.6"] == 2 and v["K__head__0.6"] == 3
    assert z["SM6__head__0.6"] == 1
