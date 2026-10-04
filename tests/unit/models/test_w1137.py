"""Wave-1137 education-4 canon tests."""

from __future__ import annotations

from quant_fund.models.adult_education import bench_adult_education
from quant_fund.models.bilingual_education import bench_bilingual_education
from quant_fund.models.early_childhood_education import bench_early_childhood_education
from quant_fund.models.educational_leadership import bench_educational_leadership
from quant_fund.models.gifted_education import bench_gifted_education
from quant_fund.models.instructional_design import bench_instructional_design


def test_early_childhood_education():
    assert bench_early_childhood_education()["synthetic_early_childhood_education"] == 1.0


def test_bilingual_education():
    assert bench_bilingual_education()["synthetic_bilingual_education"] == 1.0


def test_gifted_education():
    assert bench_gifted_education()["synthetic_gifted_education"] == 1.0


def test_adult_education():
    assert bench_adult_education()["synthetic_adult_education"] == 1.0


def test_instructional_design():
    assert bench_instructional_design()["synthetic_instructional_design"] == 1.0


def test_educational_leadership():
    assert bench_educational_leadership()["synthetic_educational_leadership"] == 1.0
