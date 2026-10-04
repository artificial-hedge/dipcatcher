"""Wave-1173 media canon tests."""

from __future__ import annotations

from quant_fund.models.communication_3 import bench_communication_3
from quant_fund.models.digital_media_2 import bench_digital_media_2
from quant_fund.models.information_science_3 import bench_information_science_3
from quant_fund.models.journalism_3 import bench_journalism_3
from quant_fund.models.media_studies_3 import bench_media_studies_3
from quant_fund.models.rhetoric_2 import bench_rhetoric_2


def test_communication_3():
    assert bench_communication_3()["synthetic_communication_3"] == 1.0


def test_journalism_3():
    assert bench_journalism_3()["synthetic_journalism_3"] == 1.0


def test_media_studies_3():
    assert bench_media_studies_3()["synthetic_media_studies_3"] == 1.0


def test_rhetoric_2():
    assert bench_rhetoric_2()["synthetic_rhetoric_2"] == 1.0


def test_information_science_3():
    assert bench_information_science_3()["synthetic_information_science_3"] == 1.0


def test_digital_media_2():
    assert bench_digital_media_2()["synthetic_digital_media_2"] == 1.0
