"""Stack-invariance lane: column order cannot leak into the stacker."""

from __future__ import annotations

import numpy as np

from quant_fund.fusion.engine import cross_fitted_ridge_stack
from quant_fund.fusion.stack_invariance import _folds, _panel, stack_invariance_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_permutation_equivariance_directly() -> None:
    x, y = _panel(seed=5)
    folds = _folds(x.shape[0])
    ref = cross_fitted_ridge_stack(x, y, folds)
    perm = np.random.default_rng(0).permutation(x.shape[1])
    got = cross_fitted_ridge_stack(x[:, perm], y, folds)
    assert np.allclose(got.weights[np.argsort(perm)], ref.weights, atol=1e-9)
    assert np.allclose(got.oof_predictions, ref.oof_predictions, atol=1e-9, equal_nan=True)


def test_bench_all_checks_pass() -> None:
    r = stack_invariance_bench()
    assert r["claim"]["ok"] is True


def test_bench_seals_and_verifies() -> None:
    receipt = stack_invariance_bench(seed=1)
    result = verify_receipt_payload(receipt, "stack_invariance_test.json")
    assert result["valid"], result.get("errors")
