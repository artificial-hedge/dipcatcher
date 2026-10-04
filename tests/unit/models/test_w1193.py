"""Wave-1193 allied-health-2 canon tests."""

from __future__ import annotations

from quant_fund.models.audiology_studies import bench_audiology_studies
from quant_fund.models.clinical_psychology_2 import bench_clinical_psychology_2
from quant_fund.models.midwifery_studies import bench_midwifery_studies
from quant_fund.models.opticianry import bench_opticianry
from quant_fund.models.orthoptics import bench_orthoptics
from quant_fund.models.prosthetics_orthotics import bench_prosthetics_orthotics


def test_midwifery_studies():
    assert bench_midwifery_studies()["synthetic_midwifery_studies"] == 1.0


def test_orthoptics():
    assert bench_orthoptics()["synthetic_orthoptics"] == 1.0


def test_audiology_studies():
    assert bench_audiology_studies()["synthetic_audiology_studies"] == 1.0


def test_opticianry():
    assert bench_opticianry()["synthetic_opticianry"] == 1.0


def test_prosthetics_orthotics():
    assert bench_prosthetics_orthotics()["synthetic_prosthetics_orthotics"] == 1.0


def test_clinical_psychology_2():
    assert bench_clinical_psychology_2()["synthetic_clinical_psychology_2"] == 1.0
