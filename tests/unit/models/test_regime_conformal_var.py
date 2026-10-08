"""SYNTHETIC correctness tests for regime-weighted conformal VaR.

All streams are seeded synthetic fixtures (two-regime Gaussian via the repo's
own ``quant_fund.stress.regimes.simulate_gaussian_hmm``); nothing here is
market evidence and nothing is a live-trading claim. Lab scores are coverage
and width only.
"""

from pathlib import Path

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile, cqr_scores, expand_interval
from quant_fund.models.regime_conformal_var import (
    RegimeWeightedConformalVaR,
    bench_regime_weighted_conformal_var,
    onehot_regimes,
    regime_coverage_report,
    weighted_quantile_exact,
)


def _separated_scores() -> tuple[np.ndarray, np.ndarray]:
    """Regime-0 scores 1..50, regime-1 scores 101..150 (stochastically ordered)."""
    scores = np.concatenate([np.arange(1.0, 51.0), np.arange(101.0, 151.0)])
    labels = np.concatenate([np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)])
    return scores, labels


# ------------------------------------------------ weighted_quantile_exact


def test_weighted_quantile_exact_uniform_matches_inverted_cdf() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        got = weighted_quantile_exact(x, np.ones_like(x), q)
        expected = float(np.quantile(x, q, method="inverted_cdf"))
        assert got == pytest.approx(expected, abs=1e-12)


def test_weighted_quantile_exact_hand_case() -> None:
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, 0.0, 0.5])
    # Cumulative weight fractions: 1.0 -> 0.5, 2.0 -> 0.5, 3.0 -> 1.0.
    assert weighted_quantile_exact(values, weights, 0.5) == pytest.approx(1.0)
    assert weighted_quantile_exact(values, weights, 0.51) == pytest.approx(3.0)
    assert weighted_quantile_exact(values, weights, 1.0) == pytest.approx(3.0)
    heavy_low = weighted_quantile_exact(values, np.array([9.0, 0.5, 0.5]), 0.85)
    assert heavy_low == pytest.approx(1.0)
    assert weighted_quantile_exact(values, np.array([9.0, 0.5, 0.5]), 0.95) == pytest.approx(2.0)


def test_weighted_quantile_exact_fail_closed() -> None:
    v = np.array([1.0, 2.0, 3.0])
    w = np.ones(3)
    with pytest.raises(ValueError):
        weighted_quantile_exact(v, np.ones(2), 0.5)  # length mismatch
    with pytest.raises(ValueError):
        weighted_quantile_exact(np.array([]), np.array([]), 0.5)  # empty
    with pytest.raises(ValueError):
        weighted_quantile_exact(v, w, -0.1)  # q < 0
    with pytest.raises(ValueError):
        weighted_quantile_exact(v, w, 1.5)  # q > 1
    with pytest.raises(ValueError):
        weighted_quantile_exact(v, np.array([1.0, -1.0, 1.0]), 0.5)  # negative weight
    with pytest.raises(ValueError):
        weighted_quantile_exact(v, np.zeros(3), 0.5)  # zero total weight
    with pytest.raises(ValueError):
        weighted_quantile_exact(np.array([1.0, np.nan, 3.0]), w, 0.5)  # non-finite value
    with pytest.raises(ValueError):
        weighted_quantile_exact(v, np.array([1.0, np.inf, 1.0]), 0.5)  # non-finite weight


def test_onehot_regimes_helper() -> None:
    m = onehot_regimes(np.array([0, 2, 1]), 3)
    assert m.shape == (3, 3)
    assert np.all(m.sum(axis=1) == 1.0)
    assert m[1, 2] == 1.0
    with pytest.raises(ValueError):
        onehot_regimes(np.array([0, 3]), 3)  # label >= n_states
    with pytest.raises(ValueError):
        onehot_regimes(np.array([0.5, 1.0]), 2)  # non-integer labels


# --------------------------------------------- exact unweighted reduction


def test_uniform_posterior_reduces_to_unweighted_closed_form() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=200)
    labels = rng.integers(0, 2, size=200).astype(np.int64)
    alpha = 0.10
    expected = conformal_quantile(scores, alpha)
    model = RegimeWeightedConformalVaR(alpha=alpha, n_states=2).calibrate(scores, labels)
    # w_i = sum_k M[i,k] * (1/K) = 1/K for every row -> uniform weights ->
    # exactly the unweighted split-conformal order statistic.
    assert model.quantile_at(np.array([0.5, 0.5])) == pytest.approx(expected, abs=1e-9)
    assert model.quantile_at(np.array([1.0, 1.0])) == pytest.approx(expected, abs=1e-9)
    # Same reduction with SOFT calibration posteriors (rows already sum to 1).
    soft = np.column_stack([rng.uniform(0.1, 0.9, size=200)])
    soft = np.column_stack([soft[:, 0], 1.0 - soft[:, 0]])
    model_soft = RegimeWeightedConformalVaR(alpha=alpha, n_states=2).calibrate(scores, soft)
    assert model_soft.quantile_at(np.array([0.5, 0.5])) == pytest.approx(expected, abs=1e-9)


def test_label_shift_correction_at_calibration_marginal_reduces() -> None:
    rng = np.random.default_rng(5)
    scores = rng.normal(size=150)
    labels = rng.integers(0, 2, size=150).astype(np.int64)
    alpha = 0.10
    model = RegimeWeightedConformalVaR(
        alpha=alpha, n_states=2, label_shift_correction=True
    ).calibrate(scores, labels)
    marginal = model.regime_marginal_
    assert marginal is not None
    # current == pi_cal -> ratio weights are exactly 1 -> unweighted reduction.
    q = model.quantile_at(marginal)
    assert q == pytest.approx(conformal_quantile(scores, alpha), abs=1e-9)
    w = model.weights(marginal)
    assert w == pytest.approx(np.ones_like(w), abs=1e-9)


def test_hard_onehot_selects_regime_scores_closed_form() -> None:
    scores, labels = _separated_scores()
    alpha = 0.10
    model = RegimeWeightedConformalVaR(alpha=alpha, n_states=2).calibrate(scores, labels)
    # One-hot posterior masks all other-regime rows; the weighted quantile of
    # the surviving uniform subset is exactly the subset's conformal quantile.
    q1 = model.quantile_at(1)
    q0 = model.quantile_at(0)
    assert q1 == pytest.approx(conformal_quantile(scores[labels == 1], alpha), abs=1e-9)
    assert q0 == pytest.approx(conformal_quantile(scores[labels == 0], alpha), abs=1e-9)
    assert q1 > q0 + 50.0


def test_soft_posterior_monotone_in_regime_mass() -> None:
    scores, labels = _separated_scores()
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(scores, labels)
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    qhats = np.array([model.quantile_at(np.array([1.0 - p, p])) for p in grid])
    assert np.all(np.diff(qhats) >= -1e-9)
    assert qhats[0] < 60.0 < qhats[-1]  # regimes are well separated
    # The uniform midpoint sits strictly between the one-hot extremes.
    assert qhats[0] < qhats[2] < qhats[-1]


# ------------------------------------------------------------ degradation


def test_ess_gate_falls_back_to_global_quantile() -> None:
    rng = np.random.default_rng(7)
    scores = rng.normal(size=100)
    labels = np.zeros(100, dtype=np.int64)
    labels[:5] = 1  # regime 1 has only 5 calibration rows -> ESS 5 < 12
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(scores, labels)
    assert model.quantile_at(1) == pytest.approx(model.global_qhat, abs=1e-12)
    assert model.global_qhat == pytest.approx(conformal_quantile(scores, 0.10), abs=1e-12)
    loose = RegimeWeightedConformalVaR(alpha=0.10, n_states=2, min_ess=1.0).calibrate(
        scores, labels
    )
    assert loose.quantile_at(1) == pytest.approx(
        conformal_quantile(scores[labels == 1], 0.10), abs=1e-9
    )
    ungated = RegimeWeightedConformalVaR(alpha=0.10, n_states=2, min_ess=None).calibrate(
        scores, labels
    )
    assert ungated.quantile_at(1) == pytest.approx(loose.quantile_at(1), abs=1e-12)


def test_unsupported_regime_falls_back_to_global() -> None:
    rng = np.random.default_rng(9)
    scores = rng.normal(size=120)
    labels = rng.integers(0, 2, size=120).astype(np.int64)  # state 2 never seen
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=3).calibrate(scores, labels)
    assert model.quantile_at(np.array([0.0, 0.0, 1.0])) == pytest.approx(
        model.global_qhat, abs=1e-12
    )
    assert np.all(model.weights(np.array([0.0, 0.0, 1.0])) == 0.0)


# ------------------------------------------------------ predict / VaR API


def test_predict_dispatch_and_predict_var_bounds() -> None:
    scores, labels = _separated_scores()
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(scores, labels)
    # Scalar hard label -> float; batch of labels -> array; (m, K) -> array.
    assert model.predict(1) == pytest.approx(model.quantile_at(1), abs=1e-12)
    batch_labels = np.array([0, 1, 1, 0])
    q_rows = model.predict(batch_labels)
    assert isinstance(q_rows, np.ndarray)
    assert q_rows == pytest.approx(
        np.array(
            [model.quantile_at(0), model.quantile_at(1), model.quantile_at(1), model.quantile_at(0)]
        ),
        abs=1e-12,
    )
    q_soft = model.predict_rows(onehot_regimes(batch_labels, 2))
    assert q_soft == pytest.approx(q_rows, abs=1e-12)
    # VaR bounds: lower = center - qhat, upper = center + qhat.
    center = np.zeros(4)
    lo = model.predict_var(center, batch_labels, side="lower")
    hi = model.predict_var(center, batch_labels, side="upper")
    assert np.all(lo < 0.0) and np.all(hi > 0.0)
    assert hi == pytest.approx(-lo, abs=1e-12)
    # Broadcast a single posterior across rows.
    lo_b = model.predict_var(center, np.array([0.2, 0.8]), side="lower")
    assert lo_b == pytest.approx(np.full(4, -model.quantile_at(np.array([0.2, 0.8]))), abs=1e-12)
    with pytest.raises(ValueError):
        model.predict_var(center, batch_labels, side="both")
    with pytest.raises(ValueError):
        model.predict_var(center, np.array([0, 1, 1]), side="lower")  # misaligned


def test_regime_coverage_report_counts() -> None:
    n = 12
    y = np.zeros(n)
    y[0] = 5.0  # one miss in regime 0
    lo = np.full(n, -1.0)
    hi = np.full(n, 1.0)
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    report = regime_coverage_report(y, lo, hi, labels)
    assert report["unconditional_coverage"] == pytest.approx(11.0 / 12.0, abs=1e-12)
    assert report["per_regime_coverage"]["0"] == pytest.approx(5.0 / 6.0, abs=1e-12)
    assert report["per_regime_coverage"]["1"] == pytest.approx(1.0, abs=1e-12)
    assert report["per_regime_n"]["0"] == 6.0
    assert report["mean_width"] == pytest.approx(2.0, abs=1e-12)
    # Soft posteriors stratify by argmax -> identical report.
    soft = onehot_regimes(labels, 2)
    soft_report = regime_coverage_report(y, lo, hi, soft)
    assert soft_report["per_regime_coverage"] == report["per_regime_coverage"]


def test_regime_coverage_report_fail_closed() -> None:
    y = np.zeros(6)
    lo = np.full(6, -1.0)
    hi = np.full(6, 1.0)
    with pytest.raises(ValueError):
        regime_coverage_report(y, lo, hi, np.zeros(5, dtype=np.int64))  # length mismatch
    with pytest.raises(ValueError):
        regime_coverage_report(y, lo, hi, np.full(6, 0.5))  # non-integer labels
    with pytest.raises(ValueError):
        regime_coverage_report(np.array([]), np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError):
        regime_coverage_report(y, lo, hi, np.zeros((6, 2)))  # zero-mass posterior rows


# ----------------------------------------------------------- constructor


def test_metadata_shape() -> None:
    scores, labels = _separated_scores()
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(scores, labels)
    meta = model.metadata()
    assert meta.family == "conformal"
    assert meta.name == "regime_weighted_conformal_var"
    assert meta.version == "v1"
    assert meta.extra["alpha"] == pytest.approx(0.10)
    assert meta.extra["n_states"] == 2
    assert meta.extra["label_shift_correction"] is False


def test_save_load_roundtrip(tmp_path: Path) -> None:
    scores, labels = _separated_scores()
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(scores, labels)
    path = tmp_path / "rwcv.joblib"
    model.save(path)
    loaded = RegimeWeightedConformalVaR.load(path)
    assert loaded.quantile_at(1) == pytest.approx(model.quantile_at(1), abs=1e-12)


def test_constructor_fail_closed() -> None:
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.0)
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=1.0)
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10, n_states=0)
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10, min_ess=-1.0)
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10, min_ess=0.0)
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10, weight_cap=0.0)


def test_calibrate_fail_closed() -> None:
    rng = np.random.default_rng(13)
    scores = rng.normal(size=50)
    labels = rng.integers(0, 2, size=50).astype(np.int64)
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10).calibrate(scores, labels[:40])  # misaligned
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10).calibrate(np.array([]), np.array([], dtype=int))
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10).calibrate(
            np.array([1.0, np.nan]), np.array([0, 1])
        )  # non-finite scores
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10).calibrate(scores, np.full(50, 0.5))  # float labels
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10).calibrate(scores, labels - 1)  # negative label
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(
            scores, labels + 5
        )  # label >= n_states
    bad_post = np.column_stack([np.full(50, -1.0), np.full(50, 2.0)])
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10).calibrate(scores, bad_post)  # negative mass
    with pytest.raises(ValueError):
        RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(
            scores, onehot_regimes(labels, 3)
        )  # wrong K


def test_predict_before_calibrate_raises() -> None:
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2)
    with pytest.raises(RuntimeError):
        model.quantile_at(0)
    with pytest.raises(RuntimeError):
        model.predict(np.array([0, 1]))
    with pytest.raises(RuntimeError):
        model.predict_var(np.zeros(2), np.array([0, 1]))


def test_predict_rejects_invalid_current_regime() -> None:
    scores, labels = _separated_scores()
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(scores, labels)
    with pytest.raises(ValueError):
        model.quantile_at(2)  # label out of range
    with pytest.raises(ValueError):
        model.quantile_at(-1)
    with pytest.raises(ValueError):
        model.quantile_at(np.array([0.5, 0.5, 0.0]))  # wrong K
    with pytest.raises(ValueError):
        model.quantile_at(np.array([-1.0, 2.0]))  # negative posterior mass
    with pytest.raises(ValueError):
        model.quantile_at(np.array([0.0, 0.0]))  # zero posterior mass
    with pytest.raises(ValueError):
        model.predict_rows(np.zeros((3, 3)))  # wrong K in batch


# --------------------------------------------------- SYNTHETIC bench (HMM)


def test_bench_unconditional_coverage_near_nominal() -> None:
    alpha = 0.10
    row = bench_regime_weighted_conformal_var(seed=11, alpha=alpha)
    assert row["synthetic_dgp"] == "fixture"
    assert row["synthetic_claim"] == "research_metric_only"
    assert row["synthetic_regimes"] == "oracle_true_states_synthetic"
    assert all("sharpe" not in str(key).lower() for key in row)
    # Unconditional coverage approximately nominal at 1 - alpha.
    assert float(row["synthetic_coverage"]) == pytest.approx(1.0 - alpha, abs=0.05)
    assert float(row["synthetic_n"]) > 0


def test_bench_highvol_conditional_coverage_beats_unweighted() -> None:
    for seed in (11, 12, 13):
        row = bench_regime_weighted_conformal_var(seed=seed, alpha=0.10)
        gain = float(row["synthetic_highvol_coverage_gain"])
        assert gain > 0.05, f"seed {seed}: high-vol coverage gain {gain:.4f} too small"
        assert float(row["synthetic_highvol_coverage"]) >= 0.82
        assert float(row["synthetic_unweighted_highvol_coverage"]) < float(
            row["synthetic_highvol_coverage"]
        )
        # Regime adaptivity: wider than the pooled interval under stress,
        # narrower in the calm regime (oracle-regime synthetic check).
        assert float(row["synthetic_highvol_mean_width"]) > float(
            row["synthetic_unweighted_mean_width"]
        )
        assert float(row["synthetic_lowvol_mean_width"]) < float(
            row["synthetic_unweighted_mean_width"]
        )


def test_bench_deterministic_and_seed_sensitive() -> None:
    a = bench_regime_weighted_conformal_var(seed=23)
    b = bench_regime_weighted_conformal_var(seed=23)
    c = bench_regime_weighted_conformal_var(seed=24)
    assert a == b  # same seed -> bit-identical research row
    assert (
        a["synthetic_coverage"] != c["synthetic_coverage"]
        or a["synthetic_mean_width"] != c["synthetic_mean_width"]
    )


def test_bench_label_shift_correction_also_covers() -> None:
    row = bench_regime_weighted_conformal_var(seed=11, label_shift_correction=True)
    assert float(row["synthetic_coverage"]) == pytest.approx(0.90, abs=0.05)
    assert float(row["synthetic_highvol_coverage_gain"]) > 0.05


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError):
        bench_regime_weighted_conformal_var(n_cal=0)
    with pytest.raises(ValueError):
        bench_regime_weighted_conformal_var(alpha=0.0)


def test_end_to_end_interval_pipeline_matches_module_docs() -> None:
    """Calibrate -> predict -> expand_interval on a seeded synthetic stream."""
    from quant_fund.models.regime_conformal_var import _synthetic_regime_stream

    y_c, s_c, y_t, s_t = _synthetic_regime_stream(400, 300, 42)
    band = 0.25
    lo_c, hi_c = np.full(400, -band), np.full(400, band)
    lo_t, hi_t = np.full(300, -band), np.full(300, band)
    model = RegimeWeightedConformalVaR(alpha=0.10, n_states=2).calibrate(
        cqr_scores(y_c, lo_c, hi_c), s_c
    )
    wlo, whi = expand_interval(lo_t, hi_t, model.predict_rows(onehot_regimes(s_t, 2)))
    report = regime_coverage_report(y_t, wlo, whi, s_t)
    assert 0.80 <= float(report["unconditional_coverage"]) <= 0.98
    high_cov = float(report["per_regime_coverage"]["1"])
    low_cov = float(report["per_regime_coverage"]["0"])
    assert high_cov >= 0.75
    assert low_cov >= 0.75
    # High-vol intervals are wider than low-vol intervals (regime adaptivity).
    assert float(report["per_regime_mean_width"]["1"]) > float(report["per_regime_mean_width"]["0"])
