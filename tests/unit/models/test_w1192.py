"""Wave-1192 integrative-medicine canon tests."""

from __future__ import annotations

from quant_fund.models.acupuncture_studies import bench_acupuncture_studies
from quant_fund.models.chiropractic_studies import bench_chiropractic_studies
from quant_fund.models.herbal_medicine import bench_herbal_medicine
from quant_fund.models.homeopathy import bench_homeopathy
from quant_fund.models.naturopathy import bench_naturopathy
from quant_fund.models.osteopathy_studies import bench_osteopathy_studies


def test_acupuncture_studies():
    assert bench_acupuncture_studies()["synthetic_acupuncture_studies"] == 1.0


def test_chiropractic_studies():
    assert bench_chiropractic_studies()["synthetic_chiropractic_studies"] == 1.0


def test_naturopathy():
    assert bench_naturopathy()["synthetic_naturopathy"] == 1.0


def test_homeopathy():
    assert bench_homeopathy()["synthetic_homeopathy"] == 1.0


def test_herbal_medicine():
    assert bench_herbal_medicine()["synthetic_herbal_medicine"] == 1.0


def test_osteopathy_studies():
    assert bench_osteopathy_studies()["synthetic_osteopathy_studies"] == 1.0
