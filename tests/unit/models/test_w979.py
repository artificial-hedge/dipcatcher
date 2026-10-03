"""Wave-979 ergodic-2 canon tests."""

from __future__ import annotations

from quant_fund.models.disjointness_dyn import bench_disjointness_dyn
from quant_fund.models.horocycle_flow import bench_horocycle_flow
from quant_fund.models.ratner_thm import bench_ratner_thm
from quant_fund.models.unipotent_ergodic import bench_unipotent_ergodic
from quant_fund.models.van_der_corput import bench_van_der_corput
from quant_fund.models.weyl_equidist import bench_weyl_equidist


def test_weyl_equidist():
    assert bench_weyl_equidist()["synthetic_weyl_equidist"] == 1.0


def test_van_der_corput():
    assert bench_van_der_corput()["synthetic_van_der_corput"] == 1.0


def test_horocycle_flow():
    assert bench_horocycle_flow()["synthetic_horocycle_flow"] == 1.0


def test_unipotent_ergodic():
    assert bench_unipotent_ergodic()["synthetic_unipotent_ergodic"] == 1.0


def test_ratner_thm():
    assert bench_ratner_thm()["synthetic_ratner_thm"] == 1.0


def test_disjointness_dyn():
    assert bench_disjointness_dyn()["synthetic_disjointness_dyn"] == 1.0
