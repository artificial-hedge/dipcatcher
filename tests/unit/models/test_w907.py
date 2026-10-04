"""Wave-907 union-find + priority-queue canon tests."""

from __future__ import annotations

from quant_fund.models.dsu_rollback import bench_dsu_rollback
from quant_fund.models.interval_heap import bench_interval_heap
from quant_fund.models.potential_dsu import bench_potential_dsu
from quant_fund.models.union_find import bench_union_find
from quant_fund.models.van_emde_boas import bench_van_emde_boas
from quant_fund.models.weak_heap import bench_weak_heap


def test_union_find():
    assert bench_union_find()["synthetic_union_find"] == 1.0


def test_dsu_rollback():
    assert bench_dsu_rollback()["synthetic_dsu_rollback"] == 1.0


def test_potential_dsu():
    assert bench_potential_dsu()["synthetic_potential_dsu"] == 1.0


def test_van_emde_boas():
    assert bench_van_emde_boas()["synthetic_van_emde_boas"] == 1.0


def test_interval_heap():
    assert bench_interval_heap()["synthetic_interval_heap"] == 1.0


def test_weak_heap():
    assert bench_weak_heap()["synthetic_weak_heap"] == 1.0
