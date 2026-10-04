"""Wave-1064 social-work/policy canon tests."""

from __future__ import annotations

from quant_fund.models.disability_studies import bench_disability_studies
from quant_fund.models.ethnic_studies import bench_ethnic_studies
from quant_fund.models.gender_studies import bench_gender_studies
from quant_fund.models.public_policy import bench_public_policy
from quant_fund.models.social_work import bench_social_work
from quant_fund.models.urban_studies import bench_urban_studies


def test_social_work():
    assert bench_social_work()["synthetic_social_work"] == 1.0


def test_public_policy():
    assert bench_public_policy()["synthetic_public_policy"] == 1.0


def test_urban_studies():
    assert bench_urban_studies()["synthetic_urban_studies"] == 1.0


def test_gender_studies():
    assert bench_gender_studies()["synthetic_gender_studies"] == 1.0


def test_ethnic_studies():
    assert bench_ethnic_studies()["synthetic_ethnic_studies"] == 1.0


def test_disability_studies():
    assert bench_disability_studies()["synthetic_disability_studies"] == 1.0
