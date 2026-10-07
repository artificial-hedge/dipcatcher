"""Wave 19: distribution model extremes — empty/short/nonfinite/ordering/fail-closed.

Complements wrappee_reselect / wrappee_cache (family selection + train-fit reuse);
this module covers Empirical / Gaussian / Scaled* / LinearQuantile edge behavior only.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    LinearQuantileDistribution,
    ScaledEmpiricalDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
)

TAUS = [0.05, 0.50, 0.95]
LO_HI = [0.10, 0.90]


def _assert_ordered(q: np.ndarray) -> None:
    assert q.ndim == 2
    assert np.all(np.isfinite(q))
    # lo <= ... <= hi across tau columns
    assert np.all(np.diff(q, axis=1) >= -1e-12)


def test_empirical_empty_y_zeros_and_ordered() -> None:
    m = EmpiricalDistribution(TAUS).fit(np.zeros((0, 1)), np.array([]))
    assert m.q_ is not None
    assert np.allclose(m.q_, 0.0)
    q = m.predict(np.zeros((3, 1)))
    assert q.shape == (3, 3)
    _assert_ordered(q)


def test_empirical_nonfinite_dropped() -> None:
    y = np.array([np.nan, 1.0, np.inf, -2.0, 3.0])
    m = EmpiricalDistribution(LO_HI).fit(np.zeros((5, 1)), y)
    # quantiles of {-2, 1, 3}
    assert m.q_ is not None
    assert m.q_[0] <= m.q_[1]
    q = m.predict(np.zeros((2, 1)))
    _assert_ordered(q)


def test_empirical_unfitted_predict_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        EmpiricalDistribution(LO_HI).predict(np.zeros((2, 1)))


def test_gaussian_empty_and_short_y() -> None:
    empty = GaussianDistribution(TAUS).fit(np.zeros((0, 1)), np.array([]))
    assert empty.mu == 0.0
    assert empty.sig == pytest.approx(0.01)
    qe = empty.predict(np.zeros((2, 1)))
    _assert_ordered(qe)

    short = GaussianDistribution(TAUS).fit(np.zeros((1, 1)), np.array([0.05]))
    assert short.mu == pytest.approx(0.05)
    assert short.sig == pytest.approx(0.01)
    _assert_ordered(short.predict(np.zeros((1, 1))))


def test_gaussian_nonfinite_drop_and_ordered() -> None:
    y = np.array([0.01, np.nan, -0.02, np.inf, 0.03, -np.inf])
    m = GaussianDistribution(TAUS).fit(np.zeros((6, 1)), y)
    assert np.isfinite(m.mu) and np.isfinite(m.sig) and m.sig > 0
    q = m.predict(np.zeros((4, 1)))
    _assert_ordered(q)
    # Explicit lo<=hi on first/last tau
    assert np.all(q[:, 0] <= q[:, -1])


def test_gaussian_unfitted_predict_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        GaussianDistribution(LO_HI).predict(np.zeros((2, 1)))


def test_scaled_empirical_standardized_residuals_are_ordered() -> None:
    y = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    scale = np.ones_like(y)
    model = ScaledEmpiricalDistribution(TAUS).fit(y, scale)
    q = model.predict(np.array([0.5, 2.0]))
    _assert_ordered(q)
    assert q[1, 0] < q[0, 0]
    assert model.metadata().name == "scaled_empirical"


def test_scaled_empirical_unfitted_predict_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        ScaledEmpiricalDistribution(LO_HI).predict(np.ones(2))


def test_scaled_gaussian_empty_short_mismatch() -> None:
    empty = ScaledGaussianDistribution(LO_HI).fit(np.array([]), np.array([]))
    assert empty.mu == 0.0
    assert empty.z_sig == pytest.approx(1.0)
    qe = empty.predict(np.array([0.02, 0.04]))
    assert qe.shape == (2, 2)
    _assert_ordered(qe)

    short = ScaledGaussianDistribution(LO_HI).fit(np.array([0.01]), np.array([0.02]))
    assert short.z_sig == pytest.approx(1.0)
    _assert_ordered(short.predict(np.array([0.03])))

    with pytest.raises(ValueError, match="same length"):
        ScaledGaussianDistribution(LO_HI).fit(np.array([1.0, 2.0]), np.array([0.1]))


def test_scaled_gaussian_nonfinite_drop_ordered() -> None:
    y = np.array([0.01, np.nan, -0.02, 0.03, np.inf])
    sc = np.array([0.02, 0.02, np.nan, 0.05, 0.02])
    m = ScaledGaussianDistribution(TAUS).fit(y, sc)
    assert np.isfinite(m.mu) and np.isfinite(m.z_sig) and m.z_sig > 0
    q = m.predict(np.array([0.01, 0.05, 0.10]))
    _assert_ordered(q)
    # Larger scale → wider band
    width = q[:, -1] - q[:, 0]
    assert width[0] < width[1] < width[2]


def test_scaled_gaussian_unfitted_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        ScaledGaussianDistribution(LO_HI).predict(np.array([0.02]))


def test_scaled_student_t_empty_short_ordered() -> None:
    empty = ScaledStudentTDistribution(LO_HI).fit(np.array([]), np.array([]))
    assert empty.mu == 0.0
    assert empty.z_sig == pytest.approx(1.0)
    assert empty.nu == pytest.approx(8.0)
    qe = empty.predict(np.array([0.02]))
    _assert_ordered(qe)

    rng = np.random.default_rng(0)
    y = rng.standard_t(4.0, size=5) * 0.02
    sc = np.full(5, 0.02)
    short = ScaledStudentTDistribution(TAUS).fit(y, sc)
    assert 3.0 <= short.nu <= 30.0
    _assert_ordered(short.predict(np.array([0.01, 0.03])))


def test_scaled_student_t_nonfinite_and_unfitted() -> None:
    y = np.array([0.0, np.nan, 0.01, -0.02, np.inf, 0.03, -0.01, 0.02])
    sc = np.array([0.02, 0.02, np.nan, 0.02, 0.02, 0.03, 0.02, 0.02])
    m = ScaledStudentTDistribution(LO_HI).fit(y, sc)
    assert np.isfinite(m.mu) and np.isfinite(m.z_sig) and m.z_sig > 0
    q = m.predict(np.array([0.02, 0.04]))
    _assert_ordered(q)
    assert np.all(q[:, 0] <= q[:, 1])
    with pytest.raises(RuntimeError, match="not been fitted"):
        ScaledStudentTDistribution(LO_HI).predict(np.array([0.02]))
    with pytest.raises(ValueError, match="same length"):
        ScaledStudentTDistribution(LO_HI).fit(np.ones(3), np.ones(2))


def test_linear_quantile_empty_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least one finite"):
        LinearQuantileDistribution(LO_HI).fit(np.zeros((0, 2)), np.array([]))
    # all-nonfinite → empty after _finite
    with pytest.raises(ValueError, match="at least one finite"):
        LinearQuantileDistribution(LO_HI).fit(
            np.array([[np.nan, 1.0], [1.0, np.inf]]),
            np.array([np.nan, np.inf]),
        )


def test_linear_quantile_short_nonfinite_ordered() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 2))
    y = 0.1 * x[:, 0] - 0.05 * x[:, 1] + rng.normal(size=40) * 0.01
    y[3] = np.nan
    x[5, 0] = np.inf
    m = LinearQuantileDistribution(TAUS).fit(x, y)
    q = m.predict(rng.normal(size=(8, 2)))
    assert q.shape == (8, 3)
    # Linear QR can cross; document honesty — we only require finite predictions here
    assert np.all(np.isfinite(q))
    # On a well-conditioned design, median should sit between extremes often
    mid_between = np.mean((q[:, 0] <= q[:, 1]) & (q[:, 1] <= q[:, 2]))
    assert mid_between >= 0.5


def test_linear_quantile_unfitted_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        LinearQuantileDistribution(LO_HI).predict(np.zeros((2, 2)))


def test_quantile_ordering_happy_path_all_families() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 0.02, size=200)
    x = np.zeros((200, 1))
    sc = np.full(200, 0.02)
    for m in (
        EmpiricalDistribution(TAUS).fit(x, y),
        GaussianDistribution(TAUS).fit(x, y),
    ):
        _assert_ordered(m.predict(np.zeros((5, 1))))
    for m in (
        ScaledGaussianDistribution(TAUS).fit(y, sc),
        ScaledStudentTDistribution(TAUS).fit(y, sc),
    ):
        _assert_ordered(m.predict(np.array([0.01, 0.02, 0.05])))


# --- TreeQuantileDistribution (Wave 20) ---


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


requires_lightgbm = pytest.mark.skipif(
    not _lightgbm_available(),
    reason="LightGBM not installed — TreeQuantile backend unavailable",
)


@requires_lightgbm
def test_tree_quantile_empty_fail_closed() -> None:
    from quant_fund.models.distribution import TreeQuantileDistribution

    with pytest.raises(ValueError, match="at least one finite"):
        TreeQuantileDistribution(LO_HI).fit(np.zeros((0, 2)), np.array([]))
    with pytest.raises(ValueError, match="at least one finite"):
        TreeQuantileDistribution(LO_HI).fit(
            np.array([[np.nan, 1.0], [1.0, np.inf]]),
            np.array([np.nan, np.inf]),
        )


@requires_lightgbm
def test_tree_quantile_unfitted_fail_closed() -> None:
    from quant_fund.models.distribution import TreeQuantileDistribution

    with pytest.raises(RuntimeError, match="not been fitted"):
        TreeQuantileDistribution(LO_HI).predict(np.zeros((2, 2)))


@requires_lightgbm
def test_tree_quantile_short_ordered_lo_le_hi() -> None:
    from quant_fund.models.distribution import TreeQuantileDistribution

    rng = np.random.default_rng(20)
    # Enough rows for LightGBM quantile leaves; planted monotone signal
    n = 80
    x = rng.normal(size=(n, 2))
    y = 0.15 * x[:, 0] - 0.08 * x[:, 1] + rng.normal(size=n) * 0.02
    y[2] = np.nan
    x[4, 1] = np.inf
    m = TreeQuantileDistribution(TAUS, backend="lightgbm", seed=20).fit(x, y)
    q = m.predict(rng.normal(size=(12, 2)))
    assert q.shape == (12, 3)
    assert np.all(np.isfinite(q))
    # Tree QR can cross; rearrange=True enforces lo<=...<=hi
    qr = m.predict(rng.normal(size=(12, 2)), rearrange=True)
    _assert_ordered(qr)
    assert np.all(qr[:, 0] <= qr[:, -1])


@requires_lightgbm
def test_tree_quantile_metadata_backend() -> None:
    from quant_fund.models.distribution import TreeQuantileDistribution

    rng = np.random.default_rng(21)
    x = rng.normal(size=(40, 1))
    y = 0.1 * x[:, 0] + rng.normal(size=40) * 0.01
    m = TreeQuantileDistribution(LO_HI, backend="lightgbm", seed=21).fit(x, y)
    meta = m.metadata()
    assert meta.family == "distribution"
    assert "lightgbm" in meta.name


# ============================================================================
# Honesty/determinism audit probes
# ============================================================================


def test_scaled_gaussian_drops_nonpositive_scale_rows() -> None:
    """scale <= 0 (incl. -inf) is invalid — the row must be DROPPED, not
    clamped to 1e-8 (which made z_i = y_i / 1e-8 an enormous outlier that
    dominated mu/z_sig)."""
    y = np.array([0.05, -0.03, 0.02, 0.07, -0.01, 0.04])
    scale = np.array([1.0, 1.0, -np.inf, 1.0, 0.0, -2.0])  # rows 2,4,5 invalid
    m = ScaledGaussianDistribution(LO_HI).fit(y, scale)
    # Surviving rows: 0,1,3 → z = y/scale on those rows.
    yy = np.array([0.05, -0.03, 0.07])
    want_mu = float(np.mean(yy))
    assert m.mu == pytest.approx(want_mu, abs=1e-10)
    z = (yy - want_mu) / np.array([1.0, 1.0, 1.0])
    assert m.z_sig == pytest.approx(float(np.std(z, ddof=1)), abs=1e-10)


def test_scaled_gaussian_scale_nan_rows_dropped() -> None:
    y = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    scale = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
    m = ScaledGaussianDistribution(LO_HI).fit(y, scale)
    yy = np.array([0.01, 0.03, 0.04, 0.05])
    assert m.mu == pytest.approx(float(np.mean(yy)), abs=1e-10)


def test_tree_quantile_lightgbm_single_threaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LGBMRegressor must pin n_jobs=1/num_threads=1 like every sibling —
    an unpinned booster makes 'seeded' fits nondeterministic on shared hosts."""
    import sys
    import types

    captured: list[dict] = []

    class _FakeLGBM:
        def __init__(self, **kw: object) -> None:
            captured.append(dict(kw))

        def fit(self, x: np.ndarray, y: np.ndarray) -> _FakeLGBM:
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    fake = types.SimpleNamespace(LGBMRegressor=_FakeLGBM)
    monkeypatch.setitem(sys.modules, "lightgbm", fake)
    from quant_fund.models.distribution import TreeQuantileDistribution

    rng = np.random.default_rng(32)
    x = rng.normal(size=(40, 2))
    y = 0.1 * x[:, 0] + rng.normal(size=40) * 0.01
    TreeQuantileDistribution(LO_HI, backend="lightgbm", seed=32).fit(x, y)
    assert len(captured) == len(LO_HI)
    for kw in captured:
        assert kw["n_jobs"] == 1
        assert kw["num_threads"] == 1
        assert kw["random_state"] == 32
