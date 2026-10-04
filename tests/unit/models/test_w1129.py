"""Wave-1129 philosophy-5 canon tests."""

from __future__ import annotations

from quant_fund.models.african_philosophy import bench_african_philosophy
from quant_fund.models.bioethics import bench_bioethics
from quant_fund.models.environmental_philosophy import bench_environmental_philosophy
from quant_fund.models.feminist_philosophy import bench_feminist_philosophy
from quant_fund.models.philosophy_of_education import bench_philosophy_of_education
from quant_fund.models.philosophy_of_medicine import bench_philosophy_of_medicine


def test_bioethics():
    assert bench_bioethics()["synthetic_bioethics"] == 1.0


def test_philosophy_of_education():
    assert bench_philosophy_of_education()["synthetic_philosophy_of_education"] == 1.0


def test_feminist_philosophy():
    assert bench_feminist_philosophy()["synthetic_feminist_philosophy"] == 1.0


def test_african_philosophy():
    assert bench_african_philosophy()["synthetic_african_philosophy"] == 1.0


def test_environmental_philosophy():
    assert bench_environmental_philosophy()["synthetic_environmental_philosophy"] == 1.0


def test_philosophy_of_medicine():
    assert bench_philosophy_of_medicine()["synthetic_philosophy_of_medicine"] == 1.0
