from quant_fund.models.diffusion_approx import (
    bench_diffusion_approx,
)
from quant_fund.models.fluid_limit import bench_fluid_limit
from quant_fund.models.halfin_whitt import bench_halfin_whitt
from quant_fund.models.heavy_traffic import bench_heavy_traffic
from quant_fund.models.kingman_bound import (
    bench_kingman_bound,
)
from quant_fund.models.qed_regime import bench_qed_regime


def test_fluid_limit():
    assert bench_fluid_limit()["synthetic_fluid_limit"] == 1.0


def test_heavy_traffic():
    assert bench_heavy_traffic()["synthetic_heavy_traffic"] == 1.0


def test_diffusion_approx():
    assert bench_diffusion_approx()["synthetic_diffusion_approx"] == 1.0


def test_kingman_bound():
    assert bench_kingman_bound()["synthetic_kingman_bound"] == 1.0


def test_halfin_whitt():
    assert bench_halfin_whitt()["synthetic_halfin_whitt"] == 1.0


def test_qed_regime():
    assert bench_qed_regime()["synthetic_qed_regime"] == 1.0
