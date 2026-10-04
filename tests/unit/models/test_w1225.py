"""Wave-1225 surgery canon tests."""

from __future__ import annotations

from quant_fund.models.colorectal_surgery import bench_colorectal_surgery
from quant_fund.models.general_surgery_studies import bench_general_surgery_studies
from quant_fund.models.hepatobiliary_surgery import bench_hepatobiliary_surgery
from quant_fund.models.minimally_invasive_surgery import bench_minimally_invasive_surgery
from quant_fund.models.surgical_oncology_studies import bench_surgical_oncology_studies
from quant_fund.models.trauma_surgery import bench_trauma_surgery


def test_general_surgery_studies():
    assert bench_general_surgery_studies()["synthetic_general_surgery_studies"] == 1.0


def test_trauma_surgery():
    assert bench_trauma_surgery()["synthetic_trauma_surgery"] == 1.0


def test_colorectal_surgery():
    assert bench_colorectal_surgery()["synthetic_colorectal_surgery"] == 1.0


def test_hepatobiliary_surgery():
    assert bench_hepatobiliary_surgery()["synthetic_hepatobiliary_surgery"] == 1.0


def test_surgical_oncology_studies():
    assert bench_surgical_oncology_studies()["synthetic_surgical_oncology_studies"] == 1.0


def test_minimally_invasive_surgery():
    assert bench_minimally_invasive_surgery()["synthetic_minimally_invasive_surgery"] == 1.0
