"""Tests for metrics/surrogate_nonlinear.py — SYNTHETIC correctness only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.surrogate_nonlinear import (
    aaft_surrogate,
    bds_statistic,
    bench_surrogate_nonlinear,
    iaaft_surrogate,
    keenan_test,
    nonlinear_statistic,
    surrogate_test,
    synth_ar,
    synth_garch,
    synth_tent,
    tsay_test,
)


class TestSurrogates:
    def test_aaft_preserves_distribution(self) -> None:
        x = synth_ar(400, seed=0)
        s = aaft_surrogate(x, seed=0)
        assert s.shape == x.shape
        # same sorted values → identical marginal distribution
        assert np.allclose(np.sort(s), np.sort(x))

    def test_aaft_preserves_spectrum_roughly(self) -> None:
        x = synth_ar(400, phi=0.8, seed=1)
        s = aaft_surrogate(x, seed=2)
        c = np.corrcoef(np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(s)) ** 2)[0, 1]
        assert c > 0.7

    def test_iaaft_closer_than_aaft_on_spectrum(self) -> None:
        x = synth_ar(400, phi=0.9, seed=2)
        psd = np.abs(np.fft.rfft(x)) ** 2
        c_a = np.corrcoef(psd, np.abs(np.fft.rfft(aaft_surrogate(x, seed=3))) ** 2)[0, 1]
        c_i = np.corrcoef(psd, np.abs(np.fft.rfft(iaaft_surrogate(x, n_iter=60, seed=3))) ** 2)[
            0, 1
        ]
        assert c_i >= c_a - 0.05

    def test_determinism(self) -> None:
        x = synth_ar(200, seed=3)
        assert np.array_equal(aaft_surrogate(x, seed=1), aaft_surrogate(x, seed=1))
        assert np.array_equal(
            iaaft_surrogate(x, n_iter=10, seed=1), iaaft_surrogate(x, n_iter=10, seed=1)
        )


class TestSurrogateTest:
    def test_p_in_unit_interval(self) -> None:
        x = synth_ar(200, seed=4)
        p = surrogate_test(x, "tser_rev", n_surr=19, seed=0)
        assert 0.0 <= p <= 1.0

    def test_tent_rejected(self) -> None:
        x = synth_tent(300, seed=5)
        p = surrogate_test(x, "tser_rev", n_surr=19, seed=0)
        assert p < 0.3

    def test_callable_stat(self) -> None:
        x = synth_tent(200, seed=6)
        p = surrogate_test(x, lambda z: float(np.std(z)), n_surr=19, seed=0)
        assert 0.0 <= p <= 1.0

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            surrogate_test(np.ones(10))
        with pytest.raises(ValueError):
            surrogate_test(synth_ar(200, seed=0), "bogus", n_surr=19)
        with pytest.raises(ValueError):
            surrogate_test(synth_ar(200, seed=0), "tser_rev", n_surr=3)


class TestBDS:
    def test_iid_no_reject_mostly(self) -> None:
        rng = np.random.default_rng(0)
        rejects = sum(bds_statistic(rng.standard_normal(400), m=3)[1] < 0.05 for _ in range(8))
        assert rejects <= 4  # generous bound on a small panel

    def test_tent_rejects(self) -> None:
        _z, p = bds_statistic(synth_tent(400, seed=1), m=3)
        assert p < 0.05

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            bds_statistic(np.ones(400), m=3)
        with pytest.raises(ValueError):
            bds_statistic(synth_ar(200, seed=0), m=1)
        with pytest.raises(ValueError):
            bds_statistic(synth_ar(200, seed=0), m=3, eps_frac=0.0)


class TestKeenanTsay:
    @staticmethod
    def _bilinear(n: int = 500, seed: int = 0) -> np.ndarray:
        # x_t = 0.4 e_{t-1} x_{t-2} + e_t — the classic Keenan/Tsay target
        rng = np.random.default_rng(seed)
        e = rng.standard_normal(n)
        x = np.zeros(n)
        for t in range(2, n):
            x[t] = 0.4 * e[t - 1] * x[t - 2] + e[t]
        return x

    def test_keenan_rejects_bilinear(self) -> None:
        _f, p = keenan_test(self._bilinear(500, seed=7), m=4)
        assert p < 0.1

    def test_tsay_rejects_bilinear(self) -> None:
        _f, p = tsay_test(self._bilinear(500, seed=7), m=3)
        assert p < 0.1

    def test_keenan_mostly_calm_on_ar(self) -> None:
        rejects = sum(keenan_test(synth_ar(400, seed=s), m=3)[1] < 0.05 for s in range(6))
        assert rejects <= 3

    def test_tsay_returns_p(self) -> None:
        g = synth_garch(400, seed=8)
        f, p = tsay_test(g, m=3)
        assert np.isfinite(f) and 0.0 <= p <= 1.0


class TestStatistic:
    def test_kinds(self) -> None:
        x = synth_ar(200, seed=9)
        for k in ("tser_rev", "skewness", "bds"):
            assert np.isfinite(nonlinear_statistic(x, k))
        with pytest.raises(ValueError):
            nonlinear_statistic(x, "bogus")


class TestBench:
    def test_keys_finite(self) -> None:
        blob = bench_surrogate_nonlinear()
        assert len(blob) >= 8
        assert all(np.isfinite(v) for v in blob.values())
        for key in blob:
            assert key.startswith("synthetic_")

    def test_bench_quality(self) -> None:
        blob = bench_surrogate_nonlinear()
        assert blob["synthetic_bds_power_tent"] > 0.8
        assert blob["synthetic_bds_size_ar"] < 0.4
        assert blob["synthetic_surr_p_tent"] < 0.15
        assert blob["synthetic_surr_p_ar"] > 0.1
        assert blob["synthetic_aaft_psd_corr"] > 0.8
        assert blob["synthetic_determinism"] == 1.0
