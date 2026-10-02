"""Wave-180 PDMP/exotic-sampling canon tests."""

from __future__ import annotations

from quant_fund.models.boomerang_sampler import _boomerang
from quant_fund.models.bouncy_particle import _bps
from quant_fund.models.elliptical_slice import _ess
from quant_fund.models.kinetic_langevin import _klmc
from quant_fund.models.riemannian_mala import _rmala
from quant_fund.models.zigzag_sampler import _zigzag


class TestBPS:
    def test_runs(self) -> None:
        smp = _bps(seed=3, horizon=60.0)
        assert smp.shape[1] == 4 and len(smp) > 10


class TestZigZag:
    def test_runs(self) -> None:
        smp = _zigzag(seed=5, horizon=100.0)
        assert smp.shape[1] == 4 and len(smp) > 10


class TestBoomerang:
    def test_runs(self) -> None:
        smp = _boomerang(seed=7, horizon=40.0)
        assert smp.shape[1] == 4 and len(smp) > 10


class TestKLMC:
    def test_runs(self) -> None:
        smp = _klmc(seed=9, n=500)
        assert smp.shape == (500, 4)


class TestESS:
    def test_runs(self) -> None:
        smp = _ess(seed=11, n=400)
        assert smp.shape == (400, 4)


class TestRMALA:
    def test_runs(self) -> None:
        smp = _rmala(seed=13, n=500)
        assert smp.shape == (500, 4)
