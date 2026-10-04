"""Wave-1208 therapy-modalities canon tests."""

from __future__ import annotations

from quant_fund.models.child_adolescent_therapy import bench_child_adolescent_therapy
from quant_fund.models.couples_therapy import bench_couples_therapy
from quant_fund.models.family_therapy import bench_family_therapy
from quant_fund.models.group_therapy import bench_group_therapy
from quant_fund.models.marriage_family_therapy import bench_marriage_family_therapy
from quant_fund.models.trauma_therapy import bench_trauma_therapy


def test_marriage_family_therapy():
    assert bench_marriage_family_therapy()["synthetic_marriage_family_therapy"] == 1.0


def test_group_therapy():
    assert bench_group_therapy()["synthetic_group_therapy"] == 1.0


def test_couples_therapy():
    assert bench_couples_therapy()["synthetic_couples_therapy"] == 1.0


def test_family_therapy():
    assert bench_family_therapy()["synthetic_family_therapy"] == 1.0


def test_child_adolescent_therapy():
    assert bench_child_adolescent_therapy()["synthetic_child_adolescent_therapy"] == 1.0


def test_trauma_therapy():
    assert bench_trauma_therapy()["synthetic_trauma_therapy"] == 1.0
