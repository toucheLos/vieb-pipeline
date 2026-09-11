"""The bakeoff verdict, and the intervals it is not allowed to ignore.

The first version of `bakeoff_read` compared point estimates. On this corpus that
crowned one arm over another on a 0.029 px difference in mean displacement and
declared both to beat the incumbent on violations -- while the animal-bootstrap
intervals for violations overlapped almost entirely.

Every interval in this project is an animal bootstrap for a reason. A gate that
then ranks on point estimates launders one into a verdict, which is the single
most repeated failure in this project's history. These tests are that rule.
"""
import numpy as np
import pytest

from vieb.clean import score

OBJ = {"dataset": "luna", "arm": "cleaning", "split": "report"}


def ci(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


def arm(viol, disp, hf, *, w=0.02):
    """One arm's three intervals, each `w` wide either side of the point."""
    return {"violation_rate": ci(viol, viol - w, viol + w),
            "distortion_mean_px": ci(disp, disp - w, disp + w),
            "hf_retained": ci(hf, hf - w, hf + w)}


class TestSeparates:

    def test_clear_separation_lower_is_better(self):
        a, b = ci(0.10, 0.09, 0.11), ci(0.20, 0.19, 0.21)
        assert score.separates(a, b, lower_is_better=True) == 1
        assert score.separates(b, a, lower_is_better=True) == -1

    def test_clear_separation_higher_is_better(self):
        a, b = ci(0.60, 0.58, 0.62), ci(0.20, 0.18, 0.22)
        assert score.separates(a, b, lower_is_better=False) == 1
        assert score.separates(b, a, lower_is_better=False) == -1

    def test_overlap_is_zero_not_a_tie_broken_on_the_point(self):
        """The whole point. These differ by 0.01 on the point estimate and
        overlap almost entirely; that is not a winner."""
        a, b = ci(0.159, 0.143, 0.176), ci(0.167, 0.149, 0.187)
        assert score.separates(a, b, lower_is_better=True) == 0

    def test_touching_intervals_do_not_separate(self):
        a, b = ci(0.1, 0.0, 0.2), ci(0.3, 0.2, 0.4)
        assert score.separates(a, b, lower_is_better=True) == 0

    def test_a_non_finite_bound_never_separates(self):
        a, b = ci(0.1, float("nan"), 0.2), ci(0.3, 0.25, 0.4)
        assert score.separates(a, b, lower_is_better=True) == 0


class TestTheRead:

    def rows(self, names):
        return [{"arm": n, "violation_rate": 0.0, "distortion_mean_px": 0.0,
                 "hf_retained": 0.0, "violation_reduction": 0.0} for n in names]

    def test_an_arm_better_on_one_axis_and_worse_on_none_dominates(self):
        iv = {"wiener": arm(0.017, 1.55, 0.22),
              "median": arm(0.016, 1.50, 0.32)}   # hf clearly higher
        rd = score.bakeoff_read(self.rows(iv), iv, scored_object=OBJ,
                                n_effective=89)
        assert rd.verdict == "PASS"
        assert "median" in rd.detail["dominating"]

    def test_an_arm_better_on_one_axis_and_worse_on_another_does_not(self):
        iv = {"wiener": arm(0.017, 1.55, 0.22),
              "smoother": arm(0.017, 3.00, 0.40)}  # better hf, far worse disp
        rd = score.bakeoff_read(self.rows(iv), iv, scored_object=OBJ,
                                n_effective=89)
        assert rd.detail["dominating"] == []
        assert rd.verdict == "NOT_A_RESULT"

    def test_overlapping_everywhere_refuses(self):
        iv = {"wiener": arm(0.017, 1.55, 0.22, w=0.5),
              "other": arm(0.016, 1.50, 0.27, w=0.5)}
        rd = score.bakeoff_read(self.rows(iv), iv, scored_object=OBJ,
                                n_effective=89)
        assert rd.verdict == "NOT_A_RESULT"
        assert "ordering is not a finding" in rd.reason

    def test_it_names_the_axes_that_separate_for_no_arm(self):
        """The corpus result: violations do not separate for anything, so the
        violation ranking is a ranking of point estimates."""
        iv = {"wiener": arm(0.017, 1.55, 0.22),
              "median": arm(0.0169, 1.50, 0.32)}   # violations overlap
        rd = score.bakeoff_read(self.rows(iv), iv, scored_object=OBJ,
                                n_effective=89)
        assert "violation_rate" in rd.detail["axes_that_separate_for_no_arm"]
        assert "violation_rate" in rd.reason

    def test_the_reason_never_lists_an_axis_as_both(self):
        """An earlier version unioned the separating axes across arms and printed
        the union as though it belonged to the named arm, producing a sentence
        that listed displacement as separating and as not separating at once."""
        iv = {"wiener": arm(0.017, 1.55, 0.22),
              "a": arm(0.017, 0.20, 0.58),        # separates on disp and hf
              "b": arm(0.017, 1.54, 0.32)}        # separates on hf only
        rd = score.bakeoff_read(self.rows(iv), iv, scored_object=OBJ,
                                n_effective=89)
        ever = set(rd.detail["axes_that_separate_for_some_arm"])
        never = set(rd.detail["axes_that_separate_for_no_arm"])
        assert not (ever & never)

    def test_a_missing_incumbent_is_inconclusive(self):
        iv = {"median": arm(0.016, 1.50, 0.32)}
        rd = score.bakeoff_read(self.rows(iv), iv, scored_object=OBJ,
                                n_effective=89)
        assert rd.verdict == "INCONCLUSIVE"

    def test_raw_is_never_counted_as_a_challenger(self):
        """`raw` retains 100% of the high-frequency power by definition; it must
        not win an axis for not being a filter."""
        iv = {"wiener": arm(0.017, 1.55, 0.22),
              "raw": arm(0.0179, 0.0, 1.0)}
        rd = score.bakeoff_read(self.rows(iv), iv, scored_object=OBJ,
                                n_effective=89)
        assert "raw" not in rd.detail["dominating"]
