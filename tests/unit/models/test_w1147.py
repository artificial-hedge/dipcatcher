"""Wave-1147 biomedical-science canon tests."""

from __future__ import annotations

from quant_fund.models.anatomy import bench_anatomy
from quant_fund.models.cardiology_2 import bench_cardiology_2
from quant_fund.models.endocrinology_2 import bench_endocrinology_2
from quant_fund.models.immunology_2 import bench_immunology_2
from quant_fund.models.neuroscience_2 import bench_neuroscience_2
from quant_fund.models.physiology_2 import bench_physiology_2


def test_anatomy():
    assert bench_anatomy()["synthetic_anatomy"] == 1.0


def test_physiology_2():
    assert bench_physiology_2()["synthetic_physiology_2"] == 1.0


def test_endocrinology_2():
    assert bench_endocrinology_2()["synthetic_endocrinology_2"] == 1.0


def test_neuroscience_2():
    assert bench_neuroscience_2()["synthetic_neuroscience_2"] == 1.0


def test_cardiology_2():
    assert bench_cardiology_2()["synthetic_cardiology_2"] == 1.0


def test_immunology_2():
    assert bench_immunology_2()["synthetic_immunology_2"] == 1.0
