"""Wave-1209 behavioral-health canon tests."""

from __future__ import annotations

from quant_fund.models.addiction_medicine import bench_addiction_medicine
from quant_fund.models.community_psychiatry import bench_community_psychiatry
from quant_fund.models.consultation_liaison import bench_consultation_liaison
from quant_fund.models.eating_disorders import bench_eating_disorders
from quant_fund.models.psychosomatic_medicine import bench_psychosomatic_medicine
from quant_fund.models.sleep_disorders import bench_sleep_disorders


def test_addiction_medicine():
    assert bench_addiction_medicine()["synthetic_addiction_medicine"] == 1.0


def test_eating_disorders():
    assert bench_eating_disorders()["synthetic_eating_disorders"] == 1.0


def test_sleep_disorders():
    assert bench_sleep_disorders()["synthetic_sleep_disorders"] == 1.0


def test_psychosomatic_medicine():
    assert bench_psychosomatic_medicine()["synthetic_psychosomatic_medicine"] == 1.0


def test_consultation_liaison():
    assert bench_consultation_liaison()["synthetic_consultation_liaison"] == 1.0


def test_community_psychiatry():
    assert bench_community_psychiatry()["synthetic_community_psychiatry"] == 1.0
