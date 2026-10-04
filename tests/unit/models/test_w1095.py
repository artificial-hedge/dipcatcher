"""Wave-1095 sociology-2 canon tests."""

from __future__ import annotations

from quant_fund.models.deviance_studies import bench_deviance_studies
from quant_fund.models.family_sociology import bench_family_sociology
from quant_fund.models.medical_sociology import bench_medical_sociology
from quant_fund.models.organization_theory import bench_organization_theory
from quant_fund.models.rural_sociology import bench_rural_sociology
from quant_fund.models.social_movements import bench_social_movements


def test_medical_sociology():
    assert bench_medical_sociology()["synthetic_medical_sociology"] == 1.0


def test_deviance_studies():
    assert bench_deviance_studies()["synthetic_deviance_studies"] == 1.0


def test_family_sociology():
    assert bench_family_sociology()["synthetic_family_sociology"] == 1.0


def test_organization_theory():
    assert bench_organization_theory()["synthetic_organization_theory"] == 1.0


def test_social_movements():
    assert bench_social_movements()["synthetic_social_movements"] == 1.0


def test_rural_sociology():
    assert bench_rural_sociology()["synthetic_rural_sociology"] == 1.0
