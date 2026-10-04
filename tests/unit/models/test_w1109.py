"""Wave-1109 linguistics-3 canon tests."""

from __future__ import annotations

from quant_fund.models.anthropological_linguistics import bench_anthropological_linguistics
from quant_fund.models.applied_linguistics import bench_applied_linguistics
from quant_fund.models.discourse_analysis import bench_discourse_analysis
from quant_fund.models.evolutionary_linguistics import bench_evolutionary_linguistics
from quant_fund.models.forensic_linguistics import bench_forensic_linguistics
from quant_fund.models.neurolinguistics import bench_neurolinguistics


def test_applied_linguistics():
    assert bench_applied_linguistics()["synthetic_applied_linguistics"] == 1.0


def test_anthropological_linguistics():
    assert bench_anthropological_linguistics()["synthetic_anthropological_linguistics"] == 1.0


def test_neurolinguistics():
    assert bench_neurolinguistics()["synthetic_neurolinguistics"] == 1.0


def test_evolutionary_linguistics():
    assert bench_evolutionary_linguistics()["synthetic_evolutionary_linguistics"] == 1.0


def test_forensic_linguistics():
    assert bench_forensic_linguistics()["synthetic_forensic_linguistics"] == 1.0


def test_discourse_analysis():
    assert bench_discourse_analysis()["synthetic_discourse_analysis"] == 1.0
