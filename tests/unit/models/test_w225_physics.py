"""Wave-225 physics-simulation canon tests."""

from __future__ import annotations

from quant_fund.models.barnes_hut import bench_barnes_hut
from quant_fund.models.fem_truss import bench_fem_truss
from quant_fund.models.nbody_leapfrog import bench_nbody_leapfrog
from quant_fund.models.rigid_collision import bench_rigid_collision
from quant_fund.models.sph_fluid import bench_sph_fluid
from quant_fund.models.verlet_cloth import bench_verlet_cloth

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestNBody:
    def test_bench(self) -> None:
        out = bench_nbody_leapfrog()
        _clean(out)
        assert out["synthetic_leapfrog_better"] == 1.0
        assert out["synthetic_momentum_err"] < 1e-10
        assert out["synthetic_orbit_stable"] == 1.0


class TestBarnesHut:
    def test_bench(self) -> None:
        out = bench_barnes_hut()
        _clean(out)
        assert out["synthetic_monotone"] == 1.0
        assert out["synthetic_small_theta_ok"] == 1.0


class TestSPH:
    def test_bench(self) -> None:
        out = bench_sph_fluid()
        _clean(out)
        assert out["synthetic_kernel_unity"] == 1.0
        assert out["synthetic_uniform_density"] == 1.0
        assert out["synthetic_grad_uniform"] == 1.0


class TestCollision:
    def test_bench(self) -> None:
        out = bench_rigid_collision()
        _clean(out)
        assert out["synthetic_momentum"] == 1.0
        assert out["synthetic_restitution"] == 1.0
        assert out["synthetic_cradle_swap"] == 1.0


class TestCloth:
    def test_bench(self) -> None:
        out = bench_verlet_cloth()
        _clean(out)
        assert out["synthetic_len_ok"] == 1.0
        assert out["synthetic_beats_free"] == 1.0


class TestFEM:
    def test_bench(self) -> None:
        out = bench_fem_truss()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_stretch"] == 1.0
