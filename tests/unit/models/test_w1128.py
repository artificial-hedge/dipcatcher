"""Wave-1128 linguistics-6 canon tests."""

from __future__ import annotations

from quant_fund.models.computational_stylistics import bench_computational_stylistics
from quant_fund.models.corpus_phonology import bench_corpus_phonology
from quant_fund.models.language_documentation import bench_language_documentation
from quant_fund.models.lexical_semantics import bench_lexical_semantics
from quant_fund.models.stylistics import bench_stylistics
from quant_fund.models.translation_technology import bench_translation_technology


def test_lexical_semantics():
    assert bench_lexical_semantics()["synthetic_lexical_semantics"] == 1.0


def test_computational_stylistics():
    assert bench_computational_stylistics()["synthetic_computational_stylistics"] == 1.0


def test_stylistics():
    assert bench_stylistics()["synthetic_stylistics"] == 1.0


def test_corpus_phonology():
    assert bench_corpus_phonology()["synthetic_corpus_phonology"] == 1.0


def test_language_documentation():
    assert bench_language_documentation()["synthetic_language_documentation"] == 1.0


def test_translation_technology():
    assert bench_translation_technology()["synthetic_translation_technology"] == 1.0
