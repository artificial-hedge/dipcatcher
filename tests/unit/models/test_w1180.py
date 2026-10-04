"""Wave-1180 allied-health canon tests."""

from __future__ import annotations

from quant_fund.models.allied_health import bench_allied_health
from quant_fund.models.midwifery import bench_midwifery
from quant_fund.models.nursing_studies import bench_nursing_studies
from quant_fund.models.occupational_science import bench_occupational_science
from quant_fund.models.paramedicine import bench_paramedicine
from quant_fund.models.speech_pathology import bench_speech_pathology


def test_nursing_studies():
    assert bench_nursing_studies()["synthetic_nursing_studies"] == 1.0


def test_allied_health():
    assert bench_allied_health()["synthetic_allied_health"] == 1.0


def test_midwifery():
    assert bench_midwifery()["synthetic_midwifery"] == 1.0


def test_paramedicine():
    assert bench_paramedicine()["synthetic_paramedicine"] == 1.0


def test_occupational_science():
    assert bench_occupational_science()["synthetic_occupational_science"] == 1.0


def test_speech_pathology():
    assert bench_speech_pathology()["synthetic_speech_pathology"] == 1.0
