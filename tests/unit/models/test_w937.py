"""Wave-937 operator-splitting canon tests."""

from __future__ import annotations

from quant_fund.models.chambolle_pock import bench_chambolle_pock
from quant_fund.models.davis_yin import bench_davis_yin
from quant_fund.models.douglas_rachford import bench_douglas_rachford
from quant_fund.models.forward_backward import bench_forward_backward
from quant_fund.models.peaceman_rachford import bench_peaceman_rachford
from quant_fund.models.tseng_split import bench_tseng_split


def test_douglas_rachford():
    assert bench_douglas_rachford()["synthetic_douglas_rachford"] == 1.0


def test_peaceman_rachford():
    assert bench_peaceman_rachford()["synthetic_peaceman_rachford"] == 1.0


def test_tseng_split():
    assert bench_tseng_split()["synthetic_tseng_split"] == 1.0


def test_forward_backward():
    assert bench_forward_backward()["synthetic_forward_backward"] == 1.0


def test_chambolle_pock():
    assert bench_chambolle_pock()["synthetic_chambolle_pock"] == 1.0


def test_davis_yin():
    assert bench_davis_yin()["synthetic_davis_yin"] == 1.0
