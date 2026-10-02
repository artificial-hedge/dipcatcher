"""Wave-155 causal-DL canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._causal_synth import synth_iv, synth_observational
from quant_fund.models.causal_rep import bench_causal_rep
from quant_fund.models.cevae_latent import bench_cevae_latent
from quant_fund.models.deep_iv import bench_deep_iv
from quant_fund.models.dragonnet_dr import bench_dragonnet_dr
from quant_fund.models.policy_value import bench_policy_value
from quant_fund.models.tarnet_ite import bench_tarnet_ite


class TestFixture:
    def test_obs(self) -> None:
        x, t, y, tau, e = synth_observational(0, 100)
        assert x.shape == (100, 5) and set(np.unique(t)) == {0, 1}

    def test_iv(self) -> None:
        z, t, y, be = synth_iv(0, 100)
        assert z.shape == (100,) and be[0] == 1.5


class TestTarnet:
    def test_bench(self) -> None:
        out = bench_tarnet_ite(seed=3, n=160, iters=20)
        assert out["synthetic_tarnet_pehe"] >= 0


class TestDrag:
    def test_bench(self) -> None:
        out = bench_dragonnet_dr(seed=5, n=160, iters=20)
        assert out["synthetic_drag_dr_err"] >= 0


class TestDIV:
    def test_bench(self) -> None:
        out = bench_deep_iv(seed=7, n=160, iters=30)
        assert out["synthetic_div_iv_err"] >= 0


class TestCEVAE:
    def test_bench(self) -> None:
        out = bench_cevae_latent(seed=9, n=160, iters=20)
        assert out["synthetic_cevae_pehe"] >= 0


class TestCRep:
    def test_bench(self) -> None:
        out = bench_causal_rep(seed=11, n=160, iters=20)
        assert out["synthetic_crep_pehe"] >= 0


class TestPV:
    def test_bench(self) -> None:
        out = bench_policy_value(seed=13, n=400)
        assert out["synthetic_pv_dr_err"] >= 0
