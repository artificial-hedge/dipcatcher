"""Wave-968 KK-theory canon tests."""

from __future__ import annotations

from quant_fund.models.baaj_julg import bench_baaj_julg
from quant_fund.models.cuntz_picture import bench_cuntz_picture
from quant_fund.models.ext_functor import bench_ext_functor
from quant_fund.models.kasparov_prod import bench_kasparov_prod
from quant_fund.models.kk_duality import bench_kk_duality
from quant_fund.models.kk_theory import bench_kk_theory


def test_kk_theory():
    assert bench_kk_theory()["synthetic_kk_theory"] == 1.0


def test_kasparov_prod():
    assert bench_kasparov_prod()["synthetic_kasparov_prod"] == 1.0


def test_ext_functor():
    assert bench_ext_functor()["synthetic_ext_functor"] == 1.0


def test_baaj_julg():
    assert bench_baaj_julg()["synthetic_baaj_julg"] == 1.0


def test_cuntz_picture():
    assert bench_cuntz_picture()["synthetic_cuntz_picture"] == 1.0


def test_kk_duality():
    assert bench_kk_duality()["synthetic_kk_duality"] == 1.0
