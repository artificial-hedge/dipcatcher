"""Wave-998 singularity/blow-up canon tests."""

from __future__ import annotations

from quant_fund.models.fujita_exponent import bench_fujita_exponent
from quant_fund.models.matched_asymptotic_pde import bench_matched_asymptotic_pde
from quant_fund.models.regularity_critical import bench_regularity_critical
from quant_fund.models.self_similar_blowup import bench_self_similar_blowup
from quant_fund.models.semilinear_heat import bench_semilinear_heat
from quant_fund.models.singularity_formation import bench_singularity_formation


def test_semilinear_heat():
    assert bench_semilinear_heat()["synthetic_semilinear_heat"] == 1.0


def test_fujita_exponent():
    assert bench_fujita_exponent()["synthetic_fujita_exponent"] == 1.0


def test_singularity_formation():
    assert bench_singularity_formation()["synthetic_singularity_formation"] == 1.0


def test_matched_asymptotic_pde():
    assert bench_matched_asymptotic_pde()["synthetic_matched_asymptotic_pde"] == 1.0


def test_self_similar_blowup():
    assert bench_self_similar_blowup()["synthetic_self_similar_blowup"] == 1.0


def test_regularity_critical():
    assert bench_regularity_critical()["synthetic_regularity_critical"] == 1.0
