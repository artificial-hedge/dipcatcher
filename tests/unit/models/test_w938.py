"""Wave-938 fixed-point canon tests."""

from __future__ import annotations

from quant_fund.models.averaged_operator import bench_averaged_operator
from quant_fund.models.cocoercive import bench_cocoercive
from quant_fund.models.fejer_monotone import bench_fejer_monotone
from quant_fund.models.firmly_nonexpansive import bench_firmly_nonexpansive
from quant_fund.models.monotone_inclusion import bench_monotone_inclusion
from quant_fund.models.quasinonexpansive import bench_quasinonexpansive


def test_fejer_monotone():
    assert bench_fejer_monotone()["synthetic_fejer_monotone"] == 1.0


def test_firmly_nonexpansive():
    assert bench_firmly_nonexpansive()["synthetic_firmly_nonexpansive"] == 1.0


def test_averaged_operator():
    assert bench_averaged_operator()["synthetic_averaged_operator"] == 1.0


def test_cocoercive():
    assert bench_cocoercive()["synthetic_cocoercive"] == 1.0


def test_quasinonexpansive():
    assert bench_quasinonexpansive()["synthetic_quasinonexpansive"] == 1.0


def test_monotone_inclusion():
    assert bench_monotone_inclusion()["synthetic_monotone_inclusion"] == 1.0
