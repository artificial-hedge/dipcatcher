"""Wave-942 primal-dual canon tests."""

from __future__ import annotations

from quant_fund.models.backward_forward import bench_backward_forward
from quant_fund.models.ishikawa_iter import bench_ishikawa_iter
from quant_fund.models.malitsky_golden import bench_malitsky_golden
from quant_fund.models.mann_iter import bench_mann_iter
from quant_fund.models.primal_dual_hybrid import bench_primal_dual_hybrid
from quant_fund.models.vu_condat import bench_vu_condat


def test_primal_dual_hybrid():
    assert bench_primal_dual_hybrid()["synthetic_primal_dual_hybrid"] == 1.0


def test_vu_condat():
    assert bench_vu_condat()["synthetic_vu_condat"] == 1.0


def test_backward_forward():
    assert bench_backward_forward()["synthetic_backward_forward"] == 1.0


def test_malitsky_golden():
    assert bench_malitsky_golden()["synthetic_malitsky_golden"] == 1.0


def test_mann_iter():
    assert bench_mann_iter()["synthetic_mann_iter"] == 1.0


def test_ishikawa_iter():
    assert bench_ishikawa_iter()["synthetic_ishikawa_iter"] == 1.0
