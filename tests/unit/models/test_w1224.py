"""Wave-1224 obgyn canon tests."""

from __future__ import annotations

from quant_fund.models.fetal_medicine import bench_fetal_medicine
from quant_fund.models.gynecologic_oncology import bench_gynecologic_oncology
from quant_fund.models.gynecology_studies import bench_gynecology_studies
from quant_fund.models.maternal_fetal_medicine import bench_maternal_fetal_medicine
from quant_fund.models.obstetrics_studies import bench_obstetrics_studies
from quant_fund.models.reproductive_endocrinology import bench_reproductive_endocrinology


def test_obstetrics_studies():
    assert bench_obstetrics_studies()["synthetic_obstetrics_studies"] == 1.0


def test_gynecology_studies():
    assert bench_gynecology_studies()["synthetic_gynecology_studies"] == 1.0


def test_maternal_fetal_medicine():
    assert bench_maternal_fetal_medicine()["synthetic_maternal_fetal_medicine"] == 1.0


def test_reproductive_endocrinology():
    assert bench_reproductive_endocrinology()["synthetic_reproductive_endocrinology"] == 1.0


def test_gynecologic_oncology():
    assert bench_gynecologic_oncology()["synthetic_gynecologic_oncology"] == 1.0


def test_fetal_medicine():
    assert bench_fetal_medicine()["synthetic_fetal_medicine"] == 1.0
