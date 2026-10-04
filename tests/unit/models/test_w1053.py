"""Wave-1053 psychology canon tests."""

from __future__ import annotations

from quant_fund.models.behavioral_neuroscience import bench_behavioral_neuroscience
from quant_fund.models.clinical_psychology import bench_clinical_psychology
from quant_fund.models.cognitive_psychology import bench_cognitive_psychology
from quant_fund.models.developmental_psychology import bench_developmental_psychology
from quant_fund.models.psychometrics import bench_psychometrics
from quant_fund.models.social_psychology import bench_social_psychology


def test_cognitive_psychology():
    assert bench_cognitive_psychology()["synthetic_cognitive_psychology"] == 1.0


def test_psychometrics():
    assert bench_psychometrics()["synthetic_psychometrics"] == 1.0


def test_behavioral_neuroscience():
    assert bench_behavioral_neuroscience()["synthetic_behavioral_neuroscience"] == 1.0


def test_social_psychology():
    assert bench_social_psychology()["synthetic_social_psychology"] == 1.0


def test_developmental_psychology():
    assert bench_developmental_psychology()["synthetic_developmental_psychology"] == 1.0


def test_clinical_psychology():
    assert bench_clinical_psychology()["synthetic_clinical_psychology"] == 1.0
