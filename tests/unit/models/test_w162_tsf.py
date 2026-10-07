"""Wave-162 TS-foundation canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._tsf_synth import naive_quantiles, pinball, ts_series
from quant_fund.models.chronos_lite import bench_chronos_lite
from quant_fund.models.lagllama_lite import bench_lagllama_lite
from quant_fund.models.moirai_lite import bench_moirai_lite
from quant_fund.models.moment_lite import bench_moment_lite
from quant_fund.models.timer_lite import bench_timer_lite
from quant_fund.models.timesfm_lite import bench_timesfm_lite


class TestTSFSynth:
    def test_series(self) -> None:
        s = ts_series(seed=3, n=48)
        assert s.shape == (48,)

    def test_pinball(self) -> None:
        y = np.array([1.0])
        qs = np.array([[0.5, 1.0, 1.5]])
        assert pinball(y, qs, np.array([0.25, 0.5, 0.75])) >= 0

    def test_naive(self) -> None:
        s = ts_series(seed=5, n=48)
        qs = naive_quantiles(s[:24], np.array([0.5]), period=24)
        assert qs.shape == (24, 1)


class TestChronos:
    def test_bench(self) -> None:
        out = bench_chronos_lite(seed=5, n_train=8, iters=10)
        assert np.isfinite(out["synthetic_chronos_pinball"])


class TestTFM:
    def test_bench(self) -> None:
        out = bench_timesfm_lite(seed=7, n_train=8, iters=10)
        assert np.isfinite(out["synthetic_tfm_pinball"])


class TestMoirai:
    def test_bench(self) -> None:
        out = bench_moirai_lite(seed=9, n_train=8, iters=10)
        assert np.isfinite(out["synthetic_moirai_pinball"])


class TestLL:
    def test_bench(self) -> None:
        out = bench_lagllama_lite(seed=11, n_train=8, iters=10)
        assert np.isfinite(out["synthetic_ll_pinball"])


class TestTimer:
    def test_bench(self) -> None:
        out = bench_timer_lite(seed=13, n_train=8, iters=10)
        assert np.isfinite(out["synthetic_timer_pinball"])


class TestMoment:
    def test_bench(self) -> None:
        out = bench_moment_lite(seed=15, n_train=8, iters=10)
        assert np.isfinite(out["synthetic_moment_pinball"])


class TestNaiveHonesty:
    def test_no_season_scaling(self) -> None:
        hist = np.arange(1.0, 97.0)  # len 96 = 4 full periods
        qs = naive_quantiles(hist, np.array([0.5]), period=24)
        assert qs.shape == (24, 1)
        expect = hist[-24:] + np.quantile(np.diff(hist[-24:]), 0.5)
        assert np.allclose(qs[:, 0], expect)
