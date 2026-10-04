"""Wave-1057 linguistics canon tests."""

from __future__ import annotations

from quant_fund.models.morphology import bench_morphology
from quant_fund.models.phonetics import bench_phonetics
from quant_fund.models.phonology import bench_phonology
from quant_fund.models.pragmatics import bench_pragmatics
from quant_fund.models.semantics import bench_semantics
from quant_fund.models.syntax_theory import bench_syntax_theory


def test_phonetics():
    assert bench_phonetics()["synthetic_phonetics"] == 1.0


def test_phonology():
    assert bench_phonology()["synthetic_phonology"] == 1.0


def test_morphology():
    assert bench_morphology()["synthetic_morphology"] == 1.0


def test_syntax_theory():
    assert bench_syntax_theory()["synthetic_syntax_theory"] == 1.0


def test_semantics():
    assert bench_semantics()["synthetic_semantics"] == 1.0


def test_pragmatics():
    assert bench_pragmatics()["synthetic_pragmatics"] == 1.0
