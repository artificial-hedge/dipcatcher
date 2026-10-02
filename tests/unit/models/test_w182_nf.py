"""Wave-182 normalizing-flow canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._nf_synth import pinwheel
from quant_fund.models.glow_flow import bench_glow_flow
from quant_fund.models.iaf_flow import bench_iaf_flow
from quant_fund.models.maf_flow import bench_maf_flow
from quant_fund.models.neural_spline_flow import bench_neural_spline_flow
from quant_fund.models.planar_flow import bench_planar_flow
from quant_fund.models.real_nvp import bench_real_nvp


class TestPinwheel:
    def test_shape(self) -> None:
        X = pinwheel(seed=3, n=200)
        assert X.shape == (200, 2)
        assert np.isfinite(X).all()


class TestRealNVP:
    def test_bench(self) -> None:
        out = bench_real_nvp(seed=3, iters=80)
        assert out["synthetic_realnvp_nll"] < out["synthetic_gauss_nll"]


class TestGlow:
    def test_bench(self) -> None:
        out = bench_glow_flow(seed=5, iters=80)
        assert np.isfinite(out["synthetic_glow_nll"])


class TestNSF:
    def test_bench(self) -> None:
        out = bench_neural_spline_flow(seed=7, iters=80)
        assert np.isfinite(out["synthetic_nsf_nll"])


class TestMAF:
    def test_bench(self) -> None:
        out = bench_maf_flow(seed=9, iters=80)
        assert np.isfinite(out["synthetic_maf_nll"])


class TestPlanar:
    def test_bench(self) -> None:
        out = bench_planar_flow(seed=11, iters=80)
        assert np.isfinite(out["synthetic_planar_nll"])


class TestIAF:
    def test_bench(self) -> None:
        out = bench_iaf_flow(seed=13, iters=80)
        assert np.isfinite(out["synthetic_iaf_nll"])
