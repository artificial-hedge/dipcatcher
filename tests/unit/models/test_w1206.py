"""Wave-1206 transplantation canon tests."""

from __future__ import annotations

from quant_fund.models.immunosuppression import bench_immunosuppression
from quant_fund.models.organ_donation import bench_organ_donation
from quant_fund.models.regenerative_medicine import bench_regenerative_medicine
from quant_fund.models.stem_cell_therapy import bench_stem_cell_therapy
from quant_fund.models.transplantation_medicine import bench_transplantation_medicine
from quant_fund.models.xenotransplantation import bench_xenotransplantation


def test_transplantation_medicine():
    assert bench_transplantation_medicine()["synthetic_transplantation_medicine"] == 1.0


def test_organ_donation():
    assert bench_organ_donation()["synthetic_organ_donation"] == 1.0


def test_immunosuppression():
    assert bench_immunosuppression()["synthetic_immunosuppression"] == 1.0


def test_xenotransplantation():
    assert bench_xenotransplantation()["synthetic_xenotransplantation"] == 1.0


def test_stem_cell_therapy():
    assert bench_stem_cell_therapy()["synthetic_stem_cell_therapy"] == 1.0


def test_regenerative_medicine():
    assert bench_regenerative_medicine()["synthetic_regenerative_medicine"] == 1.0
