import numpy as np
import pytest

from quant_fund.flowbars.bars import dollar_bar_ids, synth_tape
from quant_fund.flowbars.imbalance import tick_rule_signs
from quant_fund.flowbars.ofi_impact import (
    aggregate_ofi,
    ar_signed_flow,
    kyle_lambda_ofi,
    ofi_events,
)

pytestmark = pytest.mark.synthetic


def test_ofi_events_signed_sizes() -> None:
    signs = np.array([1, -1, 1], dtype=np.int64)
    sizes = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(ofi_events(signs, sizes), [2.0, -3.0, 4.0])


def test_aggregate_ofi_matches_bar_groups() -> None:
    events = np.array([1.0, 2.0, -3.0, 4.0])
    ids = np.array([0, 0, 1, 1])
    np.testing.assert_allclose(aggregate_ofi(events, ids), [3.0, 1.0])


def test_kyle_lambda_recovers_impact() -> None:
    rng = np.random.default_rng(10)
    n = 4000
    lam_true = 2e-4
    ofi = rng.standard_normal(n) * 100.0
    noise = rng.standard_normal(n) * 0.05
    dp = lam_true * ofi + noise
    out = kyle_lambda_ofi(dp, ofi, n_perm=300, seed=0)
    assert out["lambda"] == pytest.approx(lam_true, rel=0.15)
    assert out["r2"] > 0.1
    assert out["p_perm"] < 0.05


def test_kyle_lambda_permutation_calibrates_null() -> None:
    rng = np.random.default_rng(11)
    n = 4000
    ofi = rng.standard_normal(n) * 100.0
    dp = rng.standard_normal(n) * 0.05  # independent of flow
    out = kyle_lambda_ofi(dp, ofi, n_perm=300, seed=0)
    assert out["p_perm"] > 0.05


def test_ar_signed_flow_detects_autocorrelation() -> None:
    rng = np.random.default_rng(12)
    n = 3000
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    out = ar_signed_flow(x, p=3)
    assert out["f_stat"] > 10.0
    assert out["p"] < 0.05
    assert out["r2"] > 0.2


def test_ar_signed_flow_white_noise_not_predictable() -> None:
    rng = np.random.default_rng(13)
    out = ar_signed_flow(rng.standard_normal(3000), p=3)
    assert out["p"] > 0.01


def test_end_to_end_tape() -> None:
    tape = synth_tape(3000, seed=14)
    signs = tick_rule_signs(tape["price"])
    ids = dollar_bar_ids(tape["dollar"], 2000.0)
    events = ofi_events(signs, tape["size"])
    per_bar = aggregate_ofi(events, ids)
    ends = np.append(np.flatnonzero(np.diff(ids)), len(ids) - 1)
    closes = tape["price"][ends]
    dp = np.diff(closes)
    out = kyle_lambda_ofi(dp, per_bar[1:], n_perm=200, seed=0)
    assert np.isfinite(out["lambda"])
