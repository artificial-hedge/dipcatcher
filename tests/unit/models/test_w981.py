"""Wave-981 convex-geometry-2 canon tests."""

from __future__ import annotations

from quant_fund.models.grothendieck_const import bench_grothendieck_const
from quant_fund.models.john_ellipsoid import bench_john_ellipsoid
from quant_fund.models.kadison_singer import bench_kadison_singer
from quant_fund.models.loewner_ellipsoid import bench_loewner_ellipsoid
from quant_fund.models.milman_isotropic import bench_milman_isotropic
from quant_fund.models.milman_rev_thm import bench_milman_rev_thm


def test_john_ellipsoid():
    assert bench_john_ellipsoid()["synthetic_john_ellipsoid"] == 1.0


def test_loewner_ellipsoid():
    assert bench_loewner_ellipsoid()["synthetic_loewner_ellipsoid"] == 1.0


def test_milman_rev_thm():
    assert bench_milman_rev_thm()["synthetic_milman_rev_thm"] == 1.0


def test_grothendieck_const():
    assert bench_grothendieck_const()["synthetic_grothendieck_const"] == 1.0


def test_kadison_singer():
    assert bench_kadison_singer()["synthetic_kadison_singer"] == 1.0


def test_milman_isotropic():
    assert bench_milman_isotropic()["synthetic_milman_isotropic"] == 1.0
