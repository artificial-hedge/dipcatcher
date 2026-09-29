"""Wave 25: CPCV combinatorial_purged_cv extremes (complements test_validation / bench_cpcv_audit)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import comb

import numpy as np
import pytest

from quant_fund.research.benches import bench_cpcv_audit
from quant_fund.validation.cpcv import (
    assert_fold_embargo,
    combinatorial_purged_cv,
    cpcv_folds_verified,
    cpcv_n_paths,
    cpcv_n_splits,
    cpcv_path_assignments,
    fold_embargo_report,
    rank_configs,
    stitch_group_paths,
)
from quant_fund.validation.walk_forward import Fold, assert_no_label_overlap, session_index


def _times(n: int, *, start: datetime | None = None) -> list[datetime]:
    t0 = start or datetime(2020, 1, 2, tzinfo=UTC)
    return [t0 + timedelta(days=i) for i in range(n)]


@pytest.mark.parametrize(
    "n_groups,n_test_groups",
    [
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 3),
        (0, 1),
        (-1, 1),
        (4, -1),
    ],
)
def test_cpcv_invalid_group_counts_raise(n_groups: int, n_test_groups: int) -> None:
    with pytest.raises(ValueError, match="invalid CPCV group counts"):
        combinatorial_purged_cv(_times(60), n_groups, n_test_groups, horizon_bars=1, embargo_bars=0)


def test_cpcv_negative_horizon_or_embargo_raise() -> None:
    times = _times(40)
    with pytest.raises(ValueError, match="horizon_bars"):
        combinatorial_purged_cv(times, 4, 1, horizon_bars=-1, embargo_bars=0)
    with pytest.raises(ValueError, match="embargo_bars"):
        combinatorial_purged_cv(times, 4, 1, horizon_bars=0, embargo_bars=-2)


def test_cpcv_empty_dates_yield_no_folds() -> None:
    assert combinatorial_purged_cv([], 6, 2, horizon_bars=2, embargo_bars=2) == []


def test_cpcv_too_few_unique_dates_raise() -> None:
    # n_groups=6 with only 5 unique dates cannot form non-empty groups
    with pytest.raises(ValueError, match="at least n_groups"):
        combinatorial_purged_cv(_times(5), 6, 2, horizon_bars=1, embargo_bars=0)


def test_cpcv_fold_count_matches_combinations() -> None:
    times = _times(60)
    for n_groups, n_test in ((6, 2), (5, 2), (4, 1), (8, 3)):
        folds = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=2, embargo_bars=1)
        assert len(folds) == comb(n_groups, n_test)


def test_cpcv_train_test_disjoint_and_no_label_overlap() -> None:
    times = _times(120)
    horizon, embargo = 3, 2
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=horizon, embargo_bars=embargo
    )
    idx = session_index(times)
    assert len(folds) == comb(6, 2)
    for f in folds:
        assert f.train_times, "purged train must be non-empty"
        assert f.test_times
        assert set(f.train_times).isdisjoint(set(f.test_times))
        assert_no_label_overlap(f, horizon, idx)


def test_cpcv_purge_embargo_geometry_adjacent_test_groups() -> None:
    """n=60, 6 groups of 10; test groups 0+1 → train starts at index 22 after h=3,e=2."""
    times = _times(60)
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=3, embargo_bars=2
    )
    idx = session_index(times)
    target_test = set(times[0:20])
    matches = [f for f in folds if set(f.test_times) == target_test]
    assert len(matches) == 1
    train_idx = [idx[t] for t in matches[0].train_times]
    # Embargo after group1 (hi=19) drops 20,21; purge span does not drop later train.
    assert train_idx == list(range(22, 60))
    assert_no_label_overlap(matches[0], 3, idx)


def test_cpcv_embargo_clears_sessions_before_later_test_block() -> None:
    """Non-contiguous test groups: embargo before a later test group drops prior train."""
    times = _times(60)
    # groups of 10; pick groups 0 and 3 → test [0:10) U [30:40)
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=0, embargo_bars=2
    )
    idx = session_index(times)
    target = set(times[0:10]) | set(times[30:40])
    matches = [f for f in folds if set(f.test_times) == target]
    assert len(matches) == 1
    train_idx = set(idx[t] for t in matches[0].train_times)
    # Embargo after g0 drops 10,11; before g3 drops 28,29; after g3 drops 40,41
    assert 10 not in train_idx and 11 not in train_idx
    assert 28 not in train_idx and 29 not in train_idx
    assert 40 not in train_idx and 41 not in train_idx
    # Complement groups 1,2,4,5 without embargo zones remain
    assert 12 in train_idx and 27 in train_idx and 42 in train_idx


def test_cpcv_dedupes_duplicate_timestamps() -> None:
    base = _times(60)
    duped = base + base[10:30]
    folds = combinatorial_purged_cv(
        duped, n_groups=6, n_test_groups=2, horizon_bars=1, embargo_bars=1
    )
    assert len(folds) == comb(6, 2)
    idx = session_index(base)
    for f in folds:
        assert len(f.train_times) == len(set(f.train_times))
        assert len(f.test_times) == len(set(f.test_times))
        assert_no_label_overlap(f, 1, idx)


def test_bench_cpcv_audit_integrity_flags() -> None:
    """Research audit is validation_integrity_only — no live P&L / Sharpe claim."""
    out = bench_cpcv_audit(n_dates=120, n_groups=6, n_test_groups=2, horizon_bars=2, embargo_bars=2)
    assert out["purge_embargo_valid"] is True
    assert out["n_folds"] == out["expected_folds"] == float(comb(6, 2))
    assert out["claim"] == "validation_integrity_only"
    assert out["date_level"] is True
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav", "return"}
    assert forbidden.isdisjoint(out.keys())


def test_cpcv_aggressive_purge_may_yield_fewer_folds_than_combinations() -> None:
    """Huge horizon/embargo can wipe purged train → fold count < C(n_groups, n_test).

    Honest residual path: fewer (even zero) folds, never a silent claim of full
    combinatorial coverage. Complements empty-group / bad n_test ValueErrors.
    """
    times = _times(24)
    n_groups, n_test = 6, 2
    expected = comb(n_groups, n_test)
    full = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=0, embargo_bars=0)
    assert len(full) == expected

    reduced = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=10, embargo_bars=10)
    assert 0 < len(reduced) < expected

    wiped = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=20, embargo_bars=20)
    assert wiped == []


# ---------------------------------------------------------------------------
# Wave 28: bar-level embargo ASSERTION (not just a disjointness annotation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon,embargo", [(0, 0), (1, 1), (3, 2), (5, 4), (0, 3), (2, 0)])
def test_fold_embargo_report_passes_for_real_cpcv_folds(horizon: int, embargo: int) -> None:
    """Every geometry the splitter produces must satisfy its own bar-level guarantee."""
    times = _times(120)
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=horizon, embargo_bars=embargo
    )
    report = fold_embargo_report(folds, times, horizon_bars=horizon, embargo_bars=embargo)
    assert report["n_folds"] == len(folds) == comb(6, 2)
    # 6 groups of 20 -> the 5 adjacent test pairs form ONE contiguous block each,
    # the 10 non-adjacent pairs form two. 5*1 + 10*2 = 25 blocks inspected.
    assert report["n_blocks"] == 25
    assert report["train_in_test"] == 0
    assert report["label_reaches_test"] == 0
    assert report["embargo_after_violation"] == 0
    assert report["embargo_before_violation"] == 0
    assert report["n_violations"] == 0
    assert report["ok"] is True
    assert report["claim"] == "validation_integrity_only"
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav", "return"}
    assert forbidden.isdisjoint(report.keys())


def test_cpcv_folds_verified_returns_checked_report() -> None:
    times = _times(90)
    folds, report = cpcv_folds_verified(times, 6, 2, horizon_bars=2, embargo_bars=2)
    assert len(folds) == comb(6, 2)
    assert report["ok"] is True
    assert report["n_violations"] == 0


def test_fold_embargo_report_detects_a_hollowed_out_embargo() -> None:
    """A fold that ignores the embargo must be CAUGHT, not merely annotated.

    This is the test that fails if purge/embargo enforcement ever regresses:
    the fold is disjoint by construction (so a set-disjointness check passes)
    yet trains on the three bars immediately after the test block.
    """
    times = _times(60)
    horizon, embargo = 0, 3
    folds = combinatorial_purged_cv(times, 6, 2, horizon_bars=horizon, embargo_bars=embargo)
    target = set(times[0:20])
    honest = next(f for f in folds if set(f.test_times) == target)
    # Rebuild the same fold but let the 3 embargoed sessions after the block back in.
    leaked_train = sorted(set(honest.train_times) | set(times[20:23]))
    hollow = [Fold(train_times=leaked_train, val_times=[], test_times=list(honest.test_times))]
    report = fold_embargo_report(hollow, times, horizon_bars=horizon, embargo_bars=embargo)
    assert report["ok"] is False
    assert report["embargo_after_violation"] == 3
    assert report["n_violations"] == 3
    with pytest.raises(AssertionError, match="CPCV fold integrity violated"):
        assert_fold_embargo(hollow, times, horizon_bars=horizon, embargo_bars=embargo)


def test_fold_embargo_report_detects_label_leakage_across_the_purge() -> None:
    """A train session whose label window reaches into the block is a violation."""
    times = _times(60)
    # test block is sessions 20..29; train at session 18 with horizon 3 reaches 21.
    report = fold_embargo_report(
        [
            Fold(
                train_times=[times[18], times[40], times[50]],
                val_times=[],
                test_times=list(times[20:30]),
            )
        ],
        times,
        horizon_bars=3,
        embargo_bars=0,
    )
    assert report["ok"] is False
    assert report["label_reaches_test"] == 1
    assert report["n_violations"] == 1


def test_fold_embargo_report_detects_train_inside_test_block() -> None:
    times = _times(40)
    report = fold_embargo_report(
        [Fold(train_times=[times[5]], val_times=[], test_times=list(times[4:8]))],
        times,
        horizon_bars=0,
        embargo_bars=0,
    )
    assert report["ok"] is False
    assert report["train_in_test"] == 1


def test_fold_embargo_report_detects_missing_pre_block_embargo() -> None:
    times = _times(40)
    # test block 10..19, embargo 2 -> sessions 8 and 9 must not be train
    report = fold_embargo_report(
        [
            Fold(
                train_times=[times[8], times[9], times[30]],
                val_times=[],
                test_times=list(times[10:20]),
            )
        ],
        times,
        horizon_bars=0,
        embargo_bars=2,
    )
    assert report["ok"] is False
    assert report["embargo_before_violation"] == 2


def test_fold_embargo_report_rejects_negative_windows() -> None:
    times = _times(20)
    folds = combinatorial_purged_cv(times, 4, 1, horizon_bars=0, embargo_bars=0)
    with pytest.raises(ValueError, match="horizon_bars"):
        fold_embargo_report(folds, times, horizon_bars=-1, embargo_bars=0)
    with pytest.raises(ValueError, match="embargo_bars"):
        fold_embargo_report(folds, times, horizon_bars=0, embargo_bars=-5)


def test_fold_embargo_report_fails_closed_on_unknown_timestamp() -> None:
    times = _times(20)
    stray = datetime(2031, 5, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match="not in times"):
        fold_embargo_report(
            [Fold(train_times=[times[0]], val_times=[], test_times=[stray])],
            times,
            horizon_bars=1,
            embargo_bars=1,
        )


def test_fold_embargo_report_fails_closed_on_empty_test_block() -> None:
    times = _times(20)
    with pytest.raises(ValueError, match="empty test block"):
        fold_embargo_report(
            [Fold(train_times=[times[0]], val_times=[], test_times=[])],
            times,
            horizon_bars=1,
            embargo_bars=1,
        )


def test_fold_embargo_report_ok_requires_something_inspected() -> None:
    """`ok` is False for zero folds: 'nothing inspected' is not 'nothing wrong'."""
    report = fold_embargo_report([], _times(20), horizon_bars=1, embargo_bars=1)
    assert report["n_folds"] == 0
    assert report["n_blocks"] == 0
    assert report["n_violations"] == 0
    assert report["ok"] is False


# ---------------------------------------------------------------------------
# Wave 28: worst-path ranking (opt-in conservative selection criterion)
# ---------------------------------------------------------------------------


def test_rank_configs_worst_path_demotes_a_one_lucky_path_config() -> None:
    """Mean selection picks the config that wins on ONE path; worst-path does not.

    ``lucky`` is spectacular on path 0 and useless elsewhere; ``steady`` is
    mediocre but identical on every path. Ranking on the mean reproduces the
    in-sample-best anti-pattern; ranking on the worst path picks ``steady``.
    """
    scores = {
        "lucky": np.array([0.90, 0.01, 0.01, 0.01]),
        "steady": np.array([0.20, 0.20, 0.20, 0.20]),
    }

    by_mean = rank_configs(scores, criterion="mean")
    assert by_mean.selected == "lucky"
    assert by_mean.order == ("lucky", "steady")
    assert by_mean.detail["lucky"]["mean"] == pytest.approx(0.2325)

    by_worst = rank_configs(scores, criterion="worst_path")
    assert by_worst.selected == "steady"
    assert by_worst.order == ("steady", "lucky")
    assert by_worst.detail["lucky"]["worst_path"] == pytest.approx(0.01)
    assert by_worst.detail["steady"]["worst_path"] == pytest.approx(0.20)
    assert by_worst.detail["lucky"]["path_dispersion"] == pytest.approx(0.89)
    assert by_worst.detail["steady"]["path_dispersion"] == pytest.approx(0.0)


def test_rank_configs_default_is_mean_so_no_caller_changes_silently() -> None:
    scores = {"a": np.array([1.0, 0.0]), "b": np.array([0.4, 0.4])}
    assert rank_configs(scores).criterion == "mean"
    assert rank_configs(scores).selected == "a"
    assert rank_configs(scores, criterion="worst_path").selected == "b"


def test_rank_configs_reduces_a_path_panel_by_mean_per_path() -> None:
    """(n_paths, n_periods) input is reduced to one score per path first."""
    panel = np.array([[0.10, 0.30], [0.02, 0.02]])  # path means: 0.20, 0.02
    ranking = rank_configs({"a": panel}, criterion="worst_path")
    assert ranking.detail["a"]["mean"] == pytest.approx(0.11)
    assert ranking.detail["a"]["worst_path"] == pytest.approx(0.02)
    assert ranking.detail["a"]["best_path"] == pytest.approx(0.20)
    assert ranking.detail["a"]["n_paths"] == 2.0


def test_rank_configs_ties_break_deterministically_on_name() -> None:
    same = np.array([0.3, 0.3])
    ranking = rank_configs({"zeta": same, "alpha": same, "mid": same}, criterion="worst_path")
    assert ranking.order == ("alpha", "mid", "zeta")
    assert ranking.selected == "alpha"


def test_rank_configs_is_research_only_and_carries_no_ratio_keys() -> None:
    ranking = rank_configs({"a": np.array([0.1])}, criterion="worst_path")
    assert ranking.claim == "research_diagnostic_only"
    assert ranking.research_only is True
    blob = str(ranking.detail).lower()
    for token in ("sharpe", "sortino", "calmar", "pnl", "nav"):
        assert token not in blob
    assert set(ranking.detail["a"]) == {
        "mean",
        "worst_path",
        "best_path",
        "path_dispersion",
        "n_paths",
    }


def test_rank_configs_rejects_unknown_criterion() -> None:
    with pytest.raises(ValueError, match="criterion must be one of"):
        rank_configs({"a": np.array([0.1])}, criterion="best_path")  # type: ignore[arg-type]


def test_rank_configs_rejects_mismatched_path_counts() -> None:
    """Silently dropping a config would understate the selection multiplicity."""
    with pytest.raises(ValueError, match="same reconstructed paths"):
        rank_configs({"a": np.array([0.1, 0.2]), "b": np.array([0.1, 0.2, 0.3])})


@pytest.mark.parametrize(
    "scores",
    [
        {},
        {"a": np.array([])},
        {"a": np.array([0.1, np.nan])},
        {"a": np.array([0.1, np.inf])},
        {"a": np.zeros((0, 3))},
        {"a": np.zeros((3, 0))},
        {"a": np.array([[[0.1]]])},
        {"": np.array([0.1])},
        {"a": np.array([0.1]), "": np.array([0.2])},
    ],
)
def test_rank_configs_fails_closed_on_degenerate_input(scores: dict) -> None:
    with pytest.raises(ValueError):
        rank_configs(scores, criterion="worst_path")


def test_worst_path_ranking_ranks_a_pure_noise_grid_conservatively() -> None:
    """On pure noise the mean-best config is a coin flip; worst-path is not fooled.

    30 configs x 6 paths of N(0,1) proper scores. The worst-path winner is by
    construction never worse on its worst split than the mean winner is on
    theirs, and both rankings cover the full grid (no config silently dropped).
    """
    rng = np.random.default_rng(11)
    scores = {f"c{i:02d}": rng.normal(size=6) for i in range(30)}
    by_mean = rank_configs(scores, criterion="mean")
    by_worst = rank_configs(scores, criterion="worst_path")
    assert by_mean.selected is not None and by_worst.selected is not None
    assert (
        by_worst.detail[by_worst.selected]["worst_path"]
        >= by_worst.detail[by_mean.selected]["worst_path"]
    )
    assert set(by_mean.order) == set(scores)
    assert set(by_worst.order) == set(scores)


def test_stitched_cpcv_paths_feed_the_worst_path_criterion_end_to_end() -> None:
    """Real CPCV stitching -> ranking, with no manual reduction in between.

    ``lucky`` scores 0.55 on group 0 in every split except split 0, where it
    collapses to -4.25. Exactly one of the five reconstructed paths routes
    group 0 through split 0, so ``lucky`` has four good paths and one disaster:
    its MEAN (0.14) beats ``steady`` (0.10) while its WORST PATH (-0.50) is far
    below it. That is precisely the selection the worst-path criterion exists
    to refuse.
    """
    n_groups, n_test = 6, 2
    n_splits = cpcv_n_splits(n_groups, n_test)
    n_paths = cpcv_n_paths(n_groups, n_test)
    assert (n_splits, n_paths) == (15, 5)
    assignments = cpcv_path_assignments(n_groups, n_test)
    assert assignments.shape == (n_paths, n_groups)

    steady = np.full((n_splits, n_groups), 0.10)
    lucky = np.full((n_splits, n_groups), 0.25)
    lucky[:, 0] = 0.55
    lucky[0, 0] = -4.25  # split 0 == combination (0, 1)

    stitched_steady = stitch_group_paths(steady, assignments)
    stitched_lucky = stitch_group_paths(lucky, assignments)
    assert stitched_steady.shape == stitched_lucky.shape == (n_paths, n_groups)

    ranking_input = {
        "steady": stitched_steady.mean(axis=1),
        "lucky": stitched_lucky.mean(axis=1),
    }
    assert sorted(np.round(ranking_input["lucky"], 4).tolist()) == [
        -0.5,
        0.3,
        0.3,
        0.3,
        0.3,
    ]
    assert ranking_input["steady"].tolist() == pytest.approx([0.1] * n_paths)

    by_mean = rank_configs(ranking_input, criterion="mean")
    assert by_mean.selected == "lucky"
    assert by_mean.detail["lucky"]["mean"] == pytest.approx(0.14)

    by_worst = rank_configs(ranking_input, criterion="worst_path")
    assert by_worst.selected == "steady"
    assert by_worst.detail["lucky"]["worst_path"] == pytest.approx(-0.5)
    assert by_worst.detail["lucky"]["path_dispersion"] == pytest.approx(0.8)
