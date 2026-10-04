"""Wave-1169 social-policy canon tests."""

from __future__ import annotations

from quant_fund.models.disability_studies_2 import bench_disability_studies_2
from quant_fund.models.ethnic_studies_2 import bench_ethnic_studies_2
from quant_fund.models.gender_studies_2 import bench_gender_studies_2
from quant_fund.models.public_policy_2 import bench_public_policy_2
from quant_fund.models.social_work_2 import bench_social_work_2
from quant_fund.models.urban_studies_2 import bench_urban_studies_2


def test_social_work_2():
    assert bench_social_work_2()["synthetic_social_work_2"] == 1.0


def test_public_policy_2():
    assert bench_public_policy_2()["synthetic_public_policy_2"] == 1.0


def test_urban_studies_2():
    assert bench_urban_studies_2()["synthetic_urban_studies_2"] == 1.0


def test_gender_studies_2():
    assert bench_gender_studies_2()["synthetic_gender_studies_2"] == 1.0


def test_ethnic_studies_2():
    assert bench_ethnic_studies_2()["synthetic_ethnic_studies_2"] == 1.0


def test_disability_studies_2():
    assert bench_disability_studies_2()["synthetic_disability_studies_2"] == 1.0
