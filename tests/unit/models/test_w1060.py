"""Wave-1060 education canon tests."""

from __future__ import annotations

from quant_fund.models.assessment_theory import bench_assessment_theory
from quant_fund.models.curriculum_design import bench_curriculum_design
from quant_fund.models.educational_psychology import bench_educational_psychology
from quant_fund.models.educational_technology import bench_educational_technology
from quant_fund.models.learning_sciences import bench_learning_sciences
from quant_fund.models.pedagogy import bench_pedagogy


def test_curriculum_design():
    assert bench_curriculum_design()["synthetic_curriculum_design"] == 1.0


def test_pedagogy():
    assert bench_pedagogy()["synthetic_pedagogy"] == 1.0


def test_educational_psychology():
    assert bench_educational_psychology()["synthetic_educational_psychology"] == 1.0


def test_assessment_theory():
    assert bench_assessment_theory()["synthetic_assessment_theory"] == 1.0


def test_learning_sciences():
    assert bench_learning_sciences()["synthetic_learning_sciences"] == 1.0


def test_educational_technology():
    assert bench_educational_technology()["synthetic_educational_technology"] == 1.0
