"""Wave-1038 medicine canon tests."""

from __future__ import annotations

from quant_fund.models.cardiology import bench_cardiology
from quant_fund.models.human_physiology import bench_human_physiology
from quant_fund.models.immunology import bench_immunology
from quant_fund.models.neuroscience_med import bench_neuroscience_med
from quant_fund.models.pathology import bench_pathology
from quant_fund.models.pharmacokinetics import bench_pharmacokinetics


def test_human_physiology():
    assert bench_human_physiology()["synthetic_human_physiology"] == 1.0


def test_pharmacokinetics():
    assert bench_pharmacokinetics()["synthetic_pharmacokinetics"] == 1.0


def test_immunology():
    assert bench_immunology()["synthetic_immunology"] == 1.0


def test_pathology():
    assert bench_pathology()["synthetic_pathology"] == 1.0


def test_neuroscience_med():
    assert bench_neuroscience_med()["synthetic_neuroscience_med"] == 1.0


def test_cardiology():
    assert bench_cardiology()["synthetic_cardiology"] == 1.0
