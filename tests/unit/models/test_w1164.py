"""Wave-1164 communication canon tests."""

from __future__ import annotations

from quant_fund.models.communication_studies_2 import bench_communication_studies_2
from quant_fund.models.education_5 import bench_education_5
from quant_fund.models.information_science_2 import bench_information_science_2
from quant_fund.models.journalism_2 import bench_journalism_2
from quant_fund.models.library_science_2 import bench_library_science_2
from quant_fund.models.media_studies_2 import bench_media_studies_2


def test_education_5():
    assert bench_education_5()["synthetic_education_5"] == 1.0


def test_communication_studies_2():
    assert bench_communication_studies_2()["synthetic_communication_studies_2"] == 1.0


def test_media_studies_2():
    assert bench_media_studies_2()["synthetic_media_studies_2"] == 1.0


def test_journalism_2():
    assert bench_journalism_2()["synthetic_journalism_2"] == 1.0


def test_library_science_2():
    assert bench_library_science_2()["synthetic_library_science_2"] == 1.0


def test_information_science_2():
    assert bench_information_science_2()["synthetic_information_science_2"] == 1.0
