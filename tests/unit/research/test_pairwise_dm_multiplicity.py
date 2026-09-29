"""Multiplicity control for pairwise forecast comparison (SOTA-05 F-08).

SYNTHETIC, seeded, deterministic — correctness evidence about test *size* and
estimator behaviour, never market evidence. Nothing here is a live-trading or
P&L claim.

Motivation, measured with exactly this test's generator (eight IDENTICAL models,
every H0 true by construction, n = 120..150 squared-normal losses, 300
experiments, 28 pairs each = 8400 tests, seeds 50_000..50_299):

    raw per-test rejection rate = 0.0673  (nominal 0.05; mild HAC size drift)
    raw  FWER = 0.617   <- a 62% chance of a spurious "A beats B" finding
    BH   FWER = 0.087
    Holm FWER = 0.080
    Bonf FWER = 0.080

The assertions below are written against stream-robust bounds (raw FWER > 0.40,
corrected < raw/3 and < 0.20) rather than these exact digits, so the test pins
the order-of-magnitude fix and survives an RNG-stream change.

This is honesty-contract-adjacent: pairwise DM matrices are exactly what gets
written into a receipt, so an unadjusted matrix would seal multiplicity
artefacts as reproducible findings. Reproducibility is not validity.

:func:`pairwise_diebold_mariano` is deliberately left with its existing
signature, defaults and row keys — these tests pin that non-change.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.inference import (
    benjamini_hochberg,
    bh_adjusted_pvalues,
    bonferroni_adjusted_pvalues,
    holm_adjusted_pvalues,
    pairwise_diebold_mariano,
    pairwise_diebold_mariano_corrected,
)

ALPHA = 0.05
_N_MODELS = 8
_REPS = 300
_FWER_SEED = 50_000


def _identical_model_losses(rep: int) -> dict[str, np.ndarray]:
    """Eight models with the same loss law, so every pairwise H0 is true."""
    rng = np.random.default_rng(_FWER_SEED + rep)
    n = int(rng.integers(120, 151))
    return {f"m{i}": rng.normal(0.0, 1.0, size=n) ** 2 for i in range(_N_MODELS)}


# --- adjusted p-value primitives --------------------------------------------


def test_bh_adjusted_pvalues_match_the_existing_reject_mask() -> None:
    """``reject_k <=> adjusted_k <= alpha`` — the same decision as
    :func:`benjamini_hochberg`, so a receipt can quote either form."""
    rng = np.random.default_rng(7)
    for m in (1, 2, 3, 8, 28, 190):
        p = rng.random(m) ** 2
        adjusted = bh_adjusted_pvalues(p)
        reject_bh, _cutoff = benjamini_hochberg(p, ALPHA)
        np.testing.assert_array_equal(reject_bh, adjusted <= ALPHA)


def test_bh_adjusted_pvalues_are_monotone_and_bounded() -> None:
    """Suffix-minimum enforcement: adjusted values preserve the raw ranking."""
    p = np.array([0.001, 0.04, 0.03, 0.9, 1.0, 0.0])
    adjusted = bh_adjusted_pvalues(p)
    assert np.all(adjusted >= 0.0) and np.all(adjusted <= 1.0)
    assert np.all(adjusted >= p - 1e-12)  # never more significant than raw
    # The raw order 0.0 < 0.001 < 0.03 < 0.04 < 0.9 <= 1.0 is preserved.
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    # Hand-checkable cells: m=6, so p_(1)=0 -> 0; p_(2)=0.001 -> 6*0.001/2=0.003.
    assert adjusted[5] == pytest.approx(0.0)
    assert adjusted[0] == pytest.approx(0.003)
    # 0.03 and 0.04 both step up to 0.06 (the suffix-minimum ties them).
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)


def test_holm_is_never_less_conservative_than_bh() -> None:
    """Holm controls FWER, BH controls FDR, so Holm adjusted >= BH adjusted."""
    rng = np.random.default_rng(11)
    p = rng.random(50) ** 3
    holm = holm_adjusted_pvalues(p)
    bh = bh_adjusted_pvalues(p)
    assert np.all(holm >= bh - 1e-12)
    assert np.all(holm <= 1.0 + 1e-12)


def test_bonferroni_is_a_pure_scaling_with_no_ranking() -> None:
    p = np.array([0.001, 0.04, 0.03, 0.9, 1.0, 0.0])
    adjusted = bonferroni_adjusted_pvalues(p)
    np.testing.assert_allclose(adjusted, np.minimum(1.0, p * p.size), atol=1e-15)
    # Unlike BH/Holm, Bonferroni does NOT enforce monotonicity in rank.
    assert adjusted[1] > adjusted[2]


def test_adjusters_pass_nan_through_and_exclude_it_from_the_denominator() -> None:
    """A non-finite p-value is neither evidence for nor against H0."""
    p = np.array([0.01, np.nan, 0.4, np.nan])
    for fn in (bh_adjusted_pvalues, holm_adjusted_pvalues, bonferroni_adjusted_pvalues):
        adjusted = fn(p)
        assert math.isnan(adjusted[1]) and math.isnan(adjusted[3])
        assert np.all(np.isfinite(adjusted[[0, 2]]))
        # Multiplier uses m_finite = 2, not m = 4.
        assert adjusted[0] == pytest.approx(0.01 * 2.0)


def test_adjusters_handle_empty_and_all_nan() -> None:
    for fn in (bh_adjusted_pvalues, holm_adjusted_pvalues, bonferroni_adjusted_pvalues):
        assert fn(np.array([])).size == 0
        out = fn(np.array([np.nan, np.nan]))
        assert out.size == 2 and np.all(np.isnan(out))


# --- the corrected comparison API -------------------------------------------


def test_corrected_api_cuts_fwer_under_a_global_null() -> None:
    """The central claim: eight identical models must stop producing findings.

    Raw FWER is ~0.62 (module docstring, same seed stream); every correction
    must bring it to ~nominal (~0.08). Tolerances are set well outside the
    Monte-Carlo s.e. of a ~0.08 rate over 300 reps (~0.016), so the test is
    about the order-of-magnitude fix, not a specific RNG stream.
    """
    raw_any = 0
    corrected_any = {"bh": 0, "holm": 0, "bonferroni": 0}
    raw_tests = 0
    raw_rejects = 0
    for rep in range(_REPS):
        losses = _identical_model_losses(rep)
        result = pairwise_diebold_mariano_corrected(losses, alpha=ALPHA, method="bh")
        raw_ps = np.array([float(row["p_value"]) for row in result.rows], dtype=float)
        raw_tests += raw_ps.size
        raw_rejects += int(np.sum(raw_ps <= ALPHA))
        raw_any += int(bool(np.any(raw_ps <= ALPHA)))
        corrected_any["bh"] += int(result.n_rejected_corrected > 0)
        for method in ("holm", "bonferroni"):
            other = pairwise_diebold_mariano_corrected(losses, alpha=ALPHA, method=method)
            corrected_any[method] += int(other.n_rejected_corrected > 0)

    raw_fwer = raw_any / _REPS
    assert raw_fwer > 0.40, raw_fwer  # the defect is real, and is pinned
    assert raw_rejects / raw_tests == pytest.approx(ALPHA, abs=0.03)
    for method, count in corrected_any.items():
        fwer = count / _REPS
        assert fwer < raw_fwer / 3.0, f"{method}: {fwer} vs raw {raw_fwer}"
        assert fwer < 0.20, f"{method}: {fwer}"


def test_corrected_api_keeps_power_when_differences_are_real() -> None:
    """Multiplicity control must not erase genuine skill gaps."""
    rng = np.random.default_rng(3)
    n = 200
    losses = {
        "best": rng.normal(0.2, 0.05, size=n) ** 2,
        "mid": rng.normal(0.5, 0.05, size=n) ** 2,
        "worst": rng.normal(1.0, 0.05, size=n) ** 2,
    }
    result = pairwise_diebold_mariano_corrected(losses, alpha=ALPHA, method="holm")
    assert result.n_pairs == 3
    assert result.n_rejected_corrected == 3
    for row in result.rows:
        assert row["reject_corrected"] is True
        assert row["reject_raw"] is True
        # preferred_corrected must agree with the raw sign-based preference.
        assert row["preferred_corrected"] == row["preferred"]


def test_preferred_corrected_is_inconclusive_without_a_rejection() -> None:
    """The corrected preference cannot contradict the corrected decision."""
    rng = np.random.default_rng(5)
    n = 150
    losses = {f"m{i}": rng.normal(0.0, 1.0, size=n) ** 2 for i in range(4)}
    result = pairwise_diebold_mariano_corrected(losses, alpha=1e-12, method="bh")
    for row in result.rows:
        assert row["reject_corrected"] is False
        assert row["preferred_corrected"] == "inconclusive"
        # The raw fields still report what the uncorrected test saw.
        assert isinstance(row["preferred"], str)


def test_short_series_stay_inconclusive_and_leave_the_denominator() -> None:
    """n < 5 pairs keep the raw NaN contract and are not counted as tests."""
    rng = np.random.default_rng(13)
    losses = {
        "tiny": rng.normal(size=4),
        "ok": rng.normal(size=4),
        "ok2": rng.normal(size=4),
    }
    result = pairwise_diebold_mariano_corrected(losses, alpha=ALPHA, method="bh")
    assert result.n_pairs == 3
    for row in result.rows:
        assert math.isnan(float(row["p_value"]))
        assert math.isnan(float(row["p_value_adjusted"]))
        assert row["reject_raw"] is False
        assert row["reject_corrected"] is False
        assert row["preferred_corrected"] == "inconclusive"
    assert result.n_rejected_raw == 0


def test_stamp_fields_record_that_an_adjustment_happened() -> None:
    """A receipt must be able to show the method and alpha, not just numbers."""
    rng = np.random.default_rng(17)
    n = 120
    losses = {f"m{i}": rng.normal(0.0, 1.0, size=n) ** 2 for i in range(5)}
    result = pairwise_diebold_mariano_corrected(losses, alpha=0.10, method="holm")
    assert result.method == "holm"
    assert result.alpha == pytest.approx(0.10)
    assert result.n_pairs == 10
    payload = result.as_dict()
    assert payload["method"] == "holm"
    assert payload["alpha"] == pytest.approx(0.10)
    assert payload["n_pairs"] == 10
    assert payload["n_rejected_raw"] == result.n_rejected_raw
    assert len(payload["rows"]) == 10


def test_corrected_rows_are_a_superset_of_the_raw_rows() -> None:
    """The correction ADDS columns; it must not alter or drop the raw evidence."""
    rng = np.random.default_rng(19)
    n = 140
    losses = {f"m{i}": rng.normal(0.0, 1.0, size=n) ** 2 for i in range(4)}
    raw = pairwise_diebold_mariano(losses)
    result = pairwise_diebold_mariano_corrected(losses, alpha=ALPHA, method="bh")
    assert len(raw) == len(result.rows)
    added = {"p_value_adjusted", "reject_raw", "reject_corrected", "preferred_corrected"}
    for raw_row, corr_row in zip(raw, result.rows, strict=True):
        for key, value in raw_row.items():
            assert corr_row[key] == value
        assert set(corr_row) - set(raw_row) == added


# --- the existing API is NOT changed ---------------------------------------


def test_pairwise_diebold_mariano_row_keys_are_unchanged() -> None:
    """No multiplicity columns leak into the pre-existing function's output.

    Pinned because receipts already reference these exact keys and the honesty
    contract makes sealed evidence immutable.
    """
    rng = np.random.default_rng(0)
    good = rng.normal(0.1, 0.05, size=80)
    bad = rng.normal(0.5, 0.05, size=80)
    rows = pairwise_diebold_mariano({"good": good, "bad": bad})
    assert sorted(rows[0]) == [
        "a",
        "b",
        "mean_loss_diff",
        "n",
        "p_value",
        "preferred",
        "statistic",
    ]
    assert rows[0]["preferred"] == "good"
    assert float(rows[0]["p_value"]) < ALPHA


def test_pairwise_diebold_mariano_still_fails_closed_on_mismatched_lengths() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 150)
    with pytest.raises(ValueError, match="must align"):
        pairwise_diebold_mariano({"a": a, "b": b})
    with pytest.raises(ValueError, match="must align"):
        pairwise_diebold_mariano_corrected({"a": a, "b": b})


def test_corrected_api_rejects_an_unknown_method_or_bad_alpha() -> None:
    rng = np.random.default_rng(23)
    n = 60
    losses = {"a": rng.normal(size=n) ** 2, "b": rng.normal(size=n) ** 2}
    with pytest.raises(ValueError, match="method must be one of"):
        pairwise_diebold_mariano_corrected(losses, method="sidak")
    for bad_alpha in (0.0, -0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match="alpha must be finite"):
            pairwise_diebold_mariano_corrected(losses, alpha=bad_alpha)
    # alpha = 1.0 is the closed upper bound and must be accepted.
    assert pairwise_diebold_mariano_corrected(losses, alpha=1.0).n_pairs == 1
