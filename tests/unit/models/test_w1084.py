"""Wave-1084 comparative-literature canon tests."""

from __future__ import annotations

from quant_fund.models.comparative_literature import bench_comparative_literature
from quant_fund.models.critical_theory import bench_critical_theory
from quant_fund.models.literary_theory import bench_literary_theory
from quant_fund.models.postcolonial_studies import bench_postcolonial_studies
from quant_fund.models.translation_studies import bench_translation_studies
from quant_fund.models.world_literature import bench_world_literature


def test_comparative_literature():
    assert bench_comparative_literature()["synthetic_comparative_literature"] == 1.0


def test_literary_theory():
    assert bench_literary_theory()["synthetic_literary_theory"] == 1.0


def test_postcolonial_studies():
    assert bench_postcolonial_studies()["synthetic_postcolonial_studies"] == 1.0


def test_world_literature():
    assert bench_world_literature()["synthetic_world_literature"] == 1.0


def test_translation_studies():
    assert bench_translation_studies()["synthetic_translation_studies"] == 1.0


def test_critical_theory():
    assert bench_critical_theory()["synthetic_critical_theory"] == 1.0
