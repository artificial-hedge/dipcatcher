"""Wave-1135 medicine-5 canon tests."""

from __future__ import annotations

from quant_fund.models.ophthalmology import bench_ophthalmology
from quant_fund.models.otolaryngology import bench_otolaryngology
from quant_fund.models.palliative_medicine import bench_palliative_medicine
from quant_fund.models.rehabilitation_medicine import bench_rehabilitation_medicine
from quant_fund.models.sports_medicine import bench_sports_medicine
from quant_fund.models.urology import bench_urology


def test_urology():
    assert bench_urology()["synthetic_urology"] == 1.0


def test_ophthalmology():
    assert bench_ophthalmology()["synthetic_ophthalmology"] == 1.0


def test_otolaryngology():
    assert bench_otolaryngology()["synthetic_otolaryngology"] == 1.0


def test_palliative_medicine():
    assert bench_palliative_medicine()["synthetic_palliative_medicine"] == 1.0


def test_sports_medicine():
    assert bench_sports_medicine()["synthetic_sports_medicine"] == 1.0


def test_rehabilitation_medicine():
    assert bench_rehabilitation_medicine()["synthetic_rehabilitation_medicine"] == 1.0
