"""Wave-1021 epidemiology canon tests."""

from __future__ import annotations

from quant_fund.models.branching_epidemic import bench_branching_epidemic
from quant_fund.models.herd_immunity import bench_herd_immunity
from quant_fund.models.r0_estimation import bench_r0_estimation
from quant_fund.models.seir_epidemic import bench_seir_epidemic
from quant_fund.models.sir_epidemic import bench_sir_epidemic
from quant_fund.models.sis_epidemic import bench_sis_epidemic


def test_sir_epidemic():
    assert bench_sir_epidemic()["synthetic_sir_epidemic"] == 1.0


def test_sis_epidemic():
    assert bench_sis_epidemic()["synthetic_sis_epidemic"] == 1.0


def test_seir_epidemic():
    assert bench_seir_epidemic()["synthetic_seir_epidemic"] == 1.0


def test_r0_estimation():
    assert bench_r0_estimation()["synthetic_r0_estimation"] == 1.0


def test_herd_immunity():
    assert bench_herd_immunity()["synthetic_herd_immunity"] == 1.0


def test_branching_epidemic():
    assert bench_branching_epidemic()["synthetic_branching_epidemic"] == 1.0
