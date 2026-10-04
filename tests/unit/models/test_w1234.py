"""Wave-1234 womens-health canon tests."""

from __future__ import annotations

from quant_fund.models.breast_medicine import bench_breast_medicine
from quant_fund.models.contraception_studies import bench_contraception_studies
from quant_fund.models.infertility_studies import bench_infertility_studies
from quant_fund.models.menopause_medicine import bench_menopause_medicine
from quant_fund.models.pelvic_health_studies import bench_pelvic_health_studies
from quant_fund.models.urogynecology_studies import bench_urogynecology_studies


def test_menopause_medicine():
    assert bench_menopause_medicine()["synthetic_menopause_medicine"] == 1.0


def test_urogynecology_studies():
    assert bench_urogynecology_studies()["synthetic_urogynecology_studies"] == 1.0


def test_breast_medicine():
    assert bench_breast_medicine()["synthetic_breast_medicine"] == 1.0


def test_infertility_studies():
    assert bench_infertility_studies()["synthetic_infertility_studies"] == 1.0


def test_contraception_studies():
    assert bench_contraception_studies()["synthetic_contraception_studies"] == 1.0


def test_pelvic_health_studies():
    assert bench_pelvic_health_studies()["synthetic_pelvic_health_studies"] == 1.0
