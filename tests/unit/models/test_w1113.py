"""Wave-1113 medicine-4 canon tests."""

from __future__ import annotations

from quant_fund.models.anesthesiology import bench_anesthesiology
from quant_fund.models.emergency_medicine import bench_emergency_medicine
from quant_fund.models.family_medicine import bench_family_medicine
from quant_fund.models.obstetrics_gynecology import bench_obstetrics_gynecology
from quant_fund.models.pediatrics import bench_pediatrics
from quant_fund.models.surgery import bench_surgery


def test_surgery():
    assert bench_surgery()["synthetic_surgery"] == 1.0


def test_anesthesiology():
    assert bench_anesthesiology()["synthetic_anesthesiology"] == 1.0


def test_obstetrics_gynecology():
    assert bench_obstetrics_gynecology()["synthetic_obstetrics_gynecology"] == 1.0


def test_pediatrics():
    assert bench_pediatrics()["synthetic_pediatrics"] == 1.0


def test_emergency_medicine():
    assert bench_emergency_medicine()["synthetic_emergency_medicine"] == 1.0


def test_family_medicine():
    assert bench_family_medicine()["synthetic_family_medicine"] == 1.0
