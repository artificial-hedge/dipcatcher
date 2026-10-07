"""Probe for gsadf date_stamp PSY consolidation: the docstring promises
runs separated by a single unflagged observation are merged before the
min_len filter — the old code split on any gap."""

from __future__ import annotations

import numpy as np

from quant_fund.models.gsadf_bubble import date_stamp


def test_single_gap_runs_merge_then_survive_min_len() -> None:
    # Episodes [0,1] and [3,4] are separated by one gap at index 2.
    # Consolidated: run [0..4] length 5 >= min_len. Unmerged: two runs of 2,
    # both dropped -> the fix must keep them.
    bsadf = np.array([10.0, 10.0, 0.0, 10.0, 10.0])
    flags = date_stamp(bsadf, cv=5.0, min_len=3)
    assert flags.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_two_gap_runs_stay_split() -> None:
    # A two-observation gap is not a mergeable separation: both runs drop.
    bsadf = np.array([10.0, 10.0, 0.0, 0.0, 10.0, 10.0])
    flags = date_stamp(bsadf, cv=5.0, min_len=3)
    assert flags.tolist() == [0.0] * 6


def test_chained_single_gaps_merge_fully() -> None:
    bsadf = np.array([10.0, 0.0, 10.0, 0.0, 10.0])
    flags = date_stamp(bsadf, cv=5.0, min_len=5)
    assert flags.tolist() == [1.0] * 5


def test_nan_entries_never_flagged() -> None:
    bsadf = np.array([10.0, np.nan, 10.0, 10.0, 10.0])
    flags = date_stamp(bsadf, cv=5.0, min_len=1)
    assert flags[1] == 0.0
