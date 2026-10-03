"""Wave-988 spectral-geometry canon tests."""

from __future__ import annotations

from quant_fund.models.cheeger_ineq import bench_cheeger_ineq
from quant_fund.models.heat_invariants import bench_heat_invariants
from quant_fund.models.isospectral import bench_isospectral
from quant_fund.models.nodal_domain import bench_nodal_domain
from quant_fund.models.spectral_geometry import bench_spectral_geometry
from quant_fund.models.weyl_law import bench_weyl_law


def test_spectral_geometry():
    assert bench_spectral_geometry()["synthetic_spectral_geometry"] == 1.0


def test_heat_invariants():
    assert bench_heat_invariants()["synthetic_heat_invariants"] == 1.0


def test_weyl_law():
    assert bench_weyl_law()["synthetic_weyl_law"] == 1.0


def test_isospectral():
    assert bench_isospectral()["synthetic_isospectral"] == 1.0


def test_cheeger_ineq():
    assert bench_cheeger_ineq()["synthetic_cheeger_ineq"] == 1.0


def test_nodal_domain():
    assert bench_nodal_domain()["synthetic_nodal_domain"] == 1.0
