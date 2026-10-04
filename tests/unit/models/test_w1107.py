"""Wave-1107 medicine-3 canon tests."""

from __future__ import annotations

from quant_fund.models.endocrinology import bench_endocrinology
from quant_fund.models.gastroenterology import bench_gastroenterology
from quant_fund.models.hematology import bench_hematology
from quant_fund.models.infectious_diseases import bench_infectious_diseases
from quant_fund.models.nephrology import bench_nephrology
from quant_fund.models.pulmonology import bench_pulmonology


def test_gastroenterology():
    assert bench_gastroenterology()["synthetic_gastroenterology"] == 1.0


def test_endocrinology():
    assert bench_endocrinology()["synthetic_endocrinology"] == 1.0


def test_hematology():
    assert bench_hematology()["synthetic_hematology"] == 1.0


def test_pulmonology():
    assert bench_pulmonology()["synthetic_pulmonology"] == 1.0


def test_nephrology():
    assert bench_nephrology()["synthetic_nephrology"] == 1.0


def test_infectious_diseases():
    assert bench_infectious_diseases()["synthetic_infectious_diseases"] == 1.0
