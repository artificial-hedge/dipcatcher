"""Wave-1236 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.chronic_pain_studies import bench_chronic_pain_studies
from quant_fund.models.fibromyalgia_studies import bench_fibromyalgia_studies
from quant_fund.models.headache_studies import bench_headache_studies
from quant_fund.models.interventional_pain_studies import bench_interventional_pain_studies
from quant_fund.models.neuropathic_pain_studies import bench_neuropathic_pain_studies
from quant_fund.models.opioid_stewardship_studies import bench_opioid_stewardship_studies


def test_chronic_pain_studies() -> None:
    assert bench_chronic_pain_studies()["synthetic_chronic_pain_studies"] == 1.0


def test_fibromyalgia_studies() -> None:
    assert bench_fibromyalgia_studies()["synthetic_fibromyalgia_studies"] == 1.0


def test_headache_studies() -> None:
    assert bench_headache_studies()["synthetic_headache_studies"] == 1.0


def test_neuropathic_pain_studies() -> None:
    assert bench_neuropathic_pain_studies()["synthetic_neuropathic_pain_studies"] == 1.0


def test_opioid_stewardship_studies() -> None:
    assert bench_opioid_stewardship_studies()["synthetic_opioid_stewardship_studies"] == 1.0


def test_interventional_pain_studies() -> None:
    assert bench_interventional_pain_studies()["synthetic_interventional_pain_studies"] == 1.0
