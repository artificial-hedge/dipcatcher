"""Wave-964 unbounded-operator canon tests."""

from __future__ import annotations

from quant_fund.models.adjoint_unbounded import bench_adjoint_unbounded
from quant_fund.models.closed_operator import bench_closed_operator
from quant_fund.models.domain_dense import bench_domain_dense
from quant_fund.models.resolvent_op import bench_resolvent_op
from quant_fund.models.spectral_measure import bench_spectral_measure
from quant_fund.models.unbounded_operator import bench_unbounded_operator


def test_unbounded_operator():
    assert bench_unbounded_operator()["synthetic_unbounded_operator"] == 1.0


def test_closed_operator():
    assert bench_closed_operator()["synthetic_closed_operator"] == 1.0


def test_domain_dense():
    assert bench_domain_dense()["synthetic_domain_dense"] == 1.0


def test_adjoint_unbounded():
    assert bench_adjoint_unbounded()["synthetic_adjoint_unbounded"] == 1.0


def test_resolvent_op():
    assert bench_resolvent_op()["synthetic_resolvent_op"] == 1.0


def test_spectral_measure():
    assert bench_spectral_measure()["synthetic_spectral_measure"] == 1.0
