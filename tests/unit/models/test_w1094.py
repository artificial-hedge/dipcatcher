"""Wave-1094 medicine-2 canon tests."""

from __future__ import annotations

from quant_fund.models.dermatology import bench_dermatology
from quant_fund.models.neurology import bench_neurology
from quant_fund.models.oncology import bench_oncology
from quant_fund.models.orthopedics import bench_orthopedics
from quant_fund.models.psychiatry import bench_psychiatry
from quant_fund.models.radiology import bench_radiology


def test_oncology():
    assert bench_oncology()["synthetic_oncology"] == 1.0


def test_neurology():
    assert bench_neurology()["synthetic_neurology"] == 1.0


def test_dermatology():
    assert bench_dermatology()["synthetic_dermatology"] == 1.0


def test_orthopedics():
    assert bench_orthopedics()["synthetic_orthopedics"] == 1.0


def test_psychiatry():
    assert bench_psychiatry()["synthetic_psychiatry"] == 1.0


def test_radiology():
    assert bench_radiology()["synthetic_radiology"] == 1.0
