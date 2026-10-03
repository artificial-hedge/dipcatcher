"""Wave-977 Bochner/vector-valued canon tests."""

from __future__ import annotations

from quant_fund.models.bochner_integral import bench_bochner_integral
from quant_fund.models.bochner_meas import bench_bochner_meas
from quant_fund.models.lusin_rep import bench_lusin_rep
from quant_fund.models.norm_integrable import bench_norm_integrable
from quant_fund.models.pettis_weak import bench_pettis_weak
from quant_fund.models.radon_nikodym_prop import bench_radon_nikodym_prop


def test_bochner_integral():
    assert bench_bochner_integral()["synthetic_bochner_integral"] == 1.0


def test_lusin_rep():
    assert bench_lusin_rep()["synthetic_lusin_rep"] == 1.0


def test_radon_nikodym_prop():
    assert bench_radon_nikodym_prop()["synthetic_radon_nikodym_prop"] == 1.0


def test_bochner_meas():
    assert bench_bochner_meas()["synthetic_bochner_meas"] == 1.0


def test_norm_integrable():
    assert bench_norm_integrable()["synthetic_norm_integrable"] == 1.0


def test_pettis_weak():
    assert bench_pettis_weak()["synthetic_pettis_weak"] == 1.0
