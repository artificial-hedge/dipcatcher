"""Wave-1049 dentistry canon tests."""

from __future__ import annotations

from quant_fund.models.dental_anatomy import bench_dental_anatomy
from quant_fund.models.endodontics import bench_endodontics
from quant_fund.models.oral_pathology import bench_oral_pathology
from quant_fund.models.orthodontics import bench_orthodontics
from quant_fund.models.periodontology import bench_periodontology
from quant_fund.models.prosthodontics import bench_prosthodontics


def test_dental_anatomy():
    assert bench_dental_anatomy()["synthetic_dental_anatomy"] == 1.0


def test_oral_pathology():
    assert bench_oral_pathology()["synthetic_oral_pathology"] == 1.0


def test_periodontology():
    assert bench_periodontology()["synthetic_periodontology"] == 1.0


def test_endodontics():
    assert bench_endodontics()["synthetic_endodontics"] == 1.0


def test_orthodontics():
    assert bench_orthodontics()["synthetic_orthodontics"] == 1.0


def test_prosthodontics():
    assert bench_prosthodontics()["synthetic_prosthodontics"] == 1.0
