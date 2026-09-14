"""Every seed derived from a string must be the same in every process.

Phase F's corruption layout was drawn from `abs(hash((SEED, tag, rid)))`, and
Python salts `hash()` on `str` per interpreter, so `SEED = 0` controlled nothing.
The tests here are deliberately **subprocess** tests: an in-process check cannot
see the bug, because within one interpreter a salted hash is perfectly stable.
That is exactly why it survived review.
"""
import subprocess
import sys

import pytest

from vieb import seeds

#: Every call site that derives a seed from a string. Adding one without adding
#: it here is what this list exists to prevent.
SITES = {
    "injection.corruption_layout":
        "import importlib.util as u; "
        "sp=u.spec_from_file_location('inj','scripts/injection.py'); "
        "m=u.module_from_spec(sp); sp.loader.exec_module(m); "
        "print(m._seed_for('109','rec_a'))",
    "bones.shuffled_ceiling_selection":
        "from vieb import seeds; "
        "print(seeds.stable_seed('109', modulus=2**31))",
}


def _in_subprocess(code: str) -> str:
    full = ("import sys; sys.path.insert(0,'.'); "
            "sys.path.insert(1,'/home/tul26194/recur'); " + code)
    r = subprocess.run([sys.executable, "-c", full], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip()


class TestTheHelper:

    def test_it_is_deterministic_in_process(self):
        assert seeds.stable_seed("a", "b") == seeds.stable_seed("a", "b")

    def test_it_survives_a_fresh_interpreter(self):
        want = seeds.stable_seed("a", "b")
        got = _in_subprocess("from vieb import seeds; "
                             "print(seeds.stable_seed('a','b'))")
        assert int(got) == want

    def test_the_separator_prevents_a_concatenation_collision(self):
        """Without a separator, ('a','b') and ('ab',) would seed identically and
        two different recordings would get the same corruption layout."""
        assert seeds.stable_seed("a", "b") != seeds.stable_seed("ab")

    def test_different_parts_give_different_seeds(self):
        assert seeds.stable_seed(0, "109", "r1") != seeds.stable_seed(0, "109", "r2")
        assert seeds.stable_seed(0, "109", "r1") != seeds.stable_seed(0, "110", "r1")

    def test_it_respects_the_modulus(self):
        for _ in range(50):
            assert 0 <= seeds.stable_seed("x", modulus=2 ** 31) < 2 ** 31

    def test_python_hash_really_is_salted(self):
        """The premise. If this ever fails the audit's whole basis is wrong."""
        got = {_in_subprocess("print(abs(hash('109')) % (2**31))")
               for _ in range(3)}
        assert len(got) > 1, ("hash() was stable across processes -- "
                              "PYTHONHASHSEED is probably set, and this test "
                              "cannot verify what it exists to verify")


@pytest.mark.parametrize("name", sorted(SITES))
def test_every_seeding_site_is_stable_across_processes(name):
    """BLOCKING. Three fresh interpreters must agree."""
    got = {_in_subprocess(SITES[name]) for _ in range(3)}
    assert len(got) == 1, f"{name} differs across processes: {got}"
