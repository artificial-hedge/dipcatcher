"""Wave-1118 linguistics-4 canon tests."""

from __future__ import annotations

from quant_fund.models.field_linguistics import bench_field_linguistics
from quant_fund.models.language_acquisition import bench_language_acquisition
from quant_fund.models.linguistic_typology import bench_linguistic_typology
from quant_fund.models.sign_linguistics import bench_sign_linguistics
from quant_fund.models.theoretical_linguistics import bench_theoretical_linguistics
from quant_fund.models.translation_theory import bench_translation_theory


def test_theoretical_linguistics():
    assert bench_theoretical_linguistics()["synthetic_theoretical_linguistics"] == 1.0


def test_field_linguistics():
    assert bench_field_linguistics()["synthetic_field_linguistics"] == 1.0


def test_translation_theory():
    assert bench_translation_theory()["synthetic_translation_theory"] == 1.0


def test_sign_linguistics():
    assert bench_sign_linguistics()["synthetic_sign_linguistics"] == 1.0


def test_linguistic_typology():
    assert bench_linguistic_typology()["synthetic_linguistic_typology"] == 1.0


def test_language_acquisition():
    assert bench_language_acquisition()["synthetic_language_acquisition"] == 1.0
