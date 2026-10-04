"""Wave-980 approximation-theory-2 canon tests."""

from __future__ import annotations

from quant_fund.models.de_boor_stable import bench_de_boor_stable
from quant_fund.models.faber_schauder import bench_faber_schauder
from quant_fund.models.haar_system import bench_haar_system
from quant_fund.models.korovkin_thm import bench_korovkin_thm
from quant_fund.models.walsh_series import bench_walsh_series
from quant_fund.models.whitney_ext import bench_whitney_ext


def test_walsh_series():
    assert bench_walsh_series()["synthetic_walsh_series"] == 1.0


def test_haar_system():
    assert bench_haar_system()["synthetic_haar_system"] == 1.0


def test_faber_schauder():
    assert bench_faber_schauder()["synthetic_faber_schauder"] == 1.0


def test_de_boor_stable():
    assert bench_de_boor_stable()["synthetic_de_boor_stable"] == 1.0


def test_whitney_ext():
    assert bench_whitney_ext()["synthetic_whitney_ext"] == 1.0


def test_korovkin_thm():
    assert bench_korovkin_thm()["synthetic_korovkin_thm"] == 1.0
