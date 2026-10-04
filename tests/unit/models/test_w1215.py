"""Wave-1215 cardio-surgery canon tests."""

from __future__ import annotations

from quant_fund.models.adult_congenital import bench_adult_congenital
from quant_fund.models.cardiac_surgery import bench_cardiac_surgery
from quant_fund.models.structural_heart import bench_structural_heart
from quant_fund.models.thoracic_surgery import bench_thoracic_surgery
from quant_fund.models.transplant_cardiology import bench_transplant_cardiology
from quant_fund.models.vascular_surgery import bench_vascular_surgery


def test_vascular_surgery():
    assert bench_vascular_surgery()["synthetic_vascular_surgery"] == 1.0


def test_cardiac_surgery():
    assert bench_cardiac_surgery()["synthetic_cardiac_surgery"] == 1.0


def test_thoracic_surgery():
    assert bench_thoracic_surgery()["synthetic_thoracic_surgery"] == 1.0


def test_transplant_cardiology():
    assert bench_transplant_cardiology()["synthetic_transplant_cardiology"] == 1.0


def test_structural_heart():
    assert bench_structural_heart()["synthetic_structural_heart"] == 1.0


def test_adult_congenital():
    assert bench_adult_congenital()["synthetic_adult_congenital"] == 1.0
