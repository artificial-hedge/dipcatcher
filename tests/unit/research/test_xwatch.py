"""CrossWatch — anytime-valid cross-head lead audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant_fund.research.xwatch import (
    CrossWatch,
    write_xwatch_receipt,
    xwatch_contract_errors,
    xwatch_report,
)


def _iid_stream(n: int, seed: int) -> list[float]:
    """Sign-balanced iid diffs — the honest null for both heads."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n).tolist()


def _lead_stream(
    n: int, seed: int, lag: int = 1, flip: bool = False
) -> tuple[list[float], list[float]]:
    """A leads B by ``lag`` origins: ``dB_t == sign(dA_{t-lag})`` plus tiny noise."""
    rng = np.random.default_rng(seed)
    a = rng.choice([-1.0, 1.0], size=n)
    b = np.empty(n)
    for t in range(n):
        src = a[t - lag] if t >= lag else rng.choice([-1.0, 1.0])
        b[t] = (-src if flip else src) + rng.uniform(-0.01, 0.01)
    return a.tolist(), b.tolist()


def test_null_family_and_pooled_rates() -> None:
    """iid signs: per-family any-alarm AND pooled alarm each <= alpha + 3 sigma."""
    alpha, n_runs, n = 0.10, 200, 300
    family = 0
    pooled = 0
    for seed in range(n_runs):
        watch = CrossWatch(n_lags=3, alpha=alpha, lam=0.5)
        for a_i, b_i in zip(_iid_stream(n, seed), _iid_stream(n, 10_000 + seed), strict=True):
            watch.update(a_i, b_i)
        final = watch.states[-1]
        family += int(final.any_lag_alarmed)
        pooled += int(final.pooled_alarmed)
    slack = 3.0 * np.sqrt(alpha * (1 - alpha) / n_runs)
    assert family / n_runs <= alpha + slack
    assert pooled / n_runs <= alpha + slack


def test_injected_lead_alarms_lag1_fast() -> None:
    """A leads B by one origin — the (1, +) process must alarm early."""
    a, b = _lead_stream(400, 7, lag=1)
    watch = CrossWatch(n_lags=3, alpha=0.05, lam=0.5)
    for a_i, b_i in zip(a, b, strict=True):
        watch.update(a_i, b_i)
    assert (1, 1.0) in watch.alarm_origin
    # Every lag-1 product is +1: e_{1,+} multiplies by 1.5 each paired step
    # and reaches 2*3/0.05 = 120 in ~12 pairs — well inside this bound.
    assert watch.alarm_origin[(1, 1.0)] <= 30
    assert (1, -1.0) not in watch.alarm_origin


def test_anti_lead_fires_neg_direction() -> None:
    """B copies -A one origin later — the (1, -) process alarms."""
    a, b = _lead_stream(400, 11, lag=1, flip=True)
    watch = CrossWatch(n_lags=3, alpha=0.05, lam=0.5)
    for a_i, b_i in zip(a, b, strict=True):
        watch.update(a_i, b_i)
    assert (1, -1.0) in watch.alarm_origin
    assert (1, 1.0) not in watch.alarm_origin


def test_lag2_lead_alarms_lag2() -> None:
    """A two-origin lead lands on the lag-2 process, not lag-1."""
    a, b = _lead_stream(600, 13, lag=2)
    watch = CrossWatch(n_lags=3, alpha=0.05, lam=0.5)
    for a_i, b_i in zip(a, b, strict=True):
        watch.update(a_i, b_i)
    assert (2, 1.0) in watch.alarm_origin
    assert (1, 1.0) not in watch.alarm_origin


def test_ties_freeze_factors_and_window() -> None:
    """A zero sign is no observation: identity factor AND no window advance.

    Slot-occupying semantics would pair B[3] with the zero A[2] and shift
    the whole window; freeze semantics drops the step so B[3] still meets
    the previous *valid* A. The two give different e-values — pinned here.
    """
    a = [1.0, -1.0, 0.0, -1.0, -1.0]
    b = [-1.0, -1.0, 1.0, -1.0, -1.0]
    watch = CrossWatch(n_lags=1, alpha=0.05, lam=0.5)
    for a_i, b_i in zip(a, b, strict=True):
        watch.update(a_i, b_i)
    assert watch.n_seen == 5
    assert watch.n_paired == 4
    assert watch.states[2].frozen
    # Paired products over the valid subsequence: -1, +1, +1.
    assert watch.lag_evalues[(1, 1.0)] == pytest.approx(0.5 * 1.5 * 1.5)
    assert watch.lag_evalues[(1, -1.0)] == pytest.approx(1.5 * 0.5 * 0.5)


def test_frozen_step_is_pure_identity() -> None:
    watch = CrossWatch(n_lags=2, lam=0.5)
    watch.update(1.0, -1.0)
    watch.update(-1.0, -1.0)
    before = dict(watch.lag_evalues)
    watch.update(0.0, 1.0)
    watch.update(1.0, 0.0)
    assert watch.lag_evalues == before
    assert watch.states[-1].frozen and watch.states[-2].frozen
    assert watch.n_paired == 2 and watch.n_seen == 4


def test_alarms_are_permanent() -> None:
    """Once latched an alarm never clears — 'ever crossed' is the claim."""
    a, b = _lead_stream(60, 3, lag=1)
    watch = CrossWatch(n_lags=3, alpha=0.05, lam=0.5)
    for a_i, b_i in zip(a, b, strict=True):
        watch.update(a_i, b_i)
    origin = watch.alarm_origin[(1, 1.0)]
    assert all(s.any_lag_alarmed for s in watch.states[origin:] if s.origin >= origin)
    # Feed noise that pulls e back down — the alarm must stay latched.
    for a_i, b_i in zip(_iid_stream(200, 99), _iid_stream(200, 199), strict=True):
        watch.update(a_i, b_i)
    assert watch.alarm_origin[(1, 1.0)] == origin
    assert watch.states[-1].any_lag_alarmed
    assert (1, 1.0) in [tuple(p) for p in watch.states[-1].alarmed_pairs]


def test_causality_prefix_replay() -> None:
    """Prefix states are identical under a different future."""
    a, b = _lead_stream(300, 17, lag=1)
    watch = CrossWatch(n_lags=3, alpha=0.05, lam=0.5)
    for a_i, b_i in zip(a, b, strict=True):
        watch.update(a_i, b_i)
    k = 120
    watch2 = CrossWatch(n_lags=3, alpha=0.05, lam=0.5)
    for a_i, b_i in zip(a[:k], b[:k], strict=True):
        watch2.update(a_i, b_i)
    assert [s.pooled_evalue for s in watch2.states] == [s.pooled_evalue for s in watch.states[:k]]
    assert watch2.alarm_origin == {p: o for p, o in watch.alarm_origin.items() if o < k}


def test_fail_closed() -> None:
    watch = CrossWatch(n_lags=2)
    with pytest.raises(ValueError, match="finite"):
        watch.update(float("nan"), 1.0)
    with pytest.raises(ValueError, match="finite"):
        watch.update(1.0, float("inf"))
    with pytest.raises(ValueError, match="n_lags"):
        CrossWatch(n_lags=0)
    with pytest.raises(ValueError, match="alpha"):
        CrossWatch(alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        CrossWatch(alpha=0.0)
    with pytest.raises(ValueError, match="lam"):
        CrossWatch(lam=0.0)
    with pytest.raises(ValueError, match="lam"):
        CrossWatch(lam=1.0)
    a = _iid_stream(50, 5)
    with pytest.raises(ValueError, match="nonempty"):
        xwatch_report([], [])
    with pytest.raises(ValueError, match="equal length"):
        xwatch_report(a, a[:-1])
    with pytest.raises(ValueError, match="finite"):
        xwatch_report([1.0, float("nan"), -1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="nonempty"):
        xwatch_report(a, _iid_stream(50, 6), data_label="  ")
    with pytest.raises(ValueError, match="no paired"):
        xwatch_report([0.0] * 10, [1.0] * 10)


def test_report_shape_and_flags() -> None:
    a, b = _lead_stream(300, 23, lag=1)
    receipt = xwatch_report(a, b, n_lags=3, alpha=0.05, lam=0.5, data_label="SYNTHETIC")
    assert receipt["kind"] == "xwatch.v1"
    assert receipt["research_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["n_origins"] == 300
    assert receipt["n_paired"] == 300
    assert set(receipt["per_lag"]) == {1, 2, 3}
    assert all(
        e["pos"] > 0 and e["neg"] > 0 and isinstance(e["alarmed"], bool)
        for e in receipt["per_lag"].values()
    )
    assert "lag1_pos" in receipt["alarmed_pairs"]
    assert receipt["any_lag_alarmed"]
    assert "anytime_valid" in receipt["evidence"]
    assert xwatch_contract_errors(receipt) == []


def test_receipt_seal_verify_and_tamper(tmp_path: Path) -> None:
    a, b = _lead_stream(200, 29, lag=1)
    receipt = xwatch_report(a, b, data_label="SYNTHETIC")
    path = write_xwatch_receipt(receipt, tmp_path)
    assert path.name.startswith("xwatch_") and path.suffix == ".json"

    from quant_fund.research.receipt_v2 import verify_receipt_file

    assert verify_receipt_file(path)["valid"]

    bad = json.loads(path.read_text())
    bad["pooled_evalue"] = 1e9
    path.write_text(json.dumps(bad))
    assert not verify_receipt_file(path)["valid"]


def test_receipt_writer_fail_closed(tmp_path: Path) -> None:
    a, b = _lead_stream(120, 31, lag=1)
    receipt = xwatch_report(a, b, data_label="SYNTHETIC")
    bad = dict(receipt)
    bad["live_pnl_claim"] = True
    with pytest.raises(ValueError, match="contract"):
        write_xwatch_receipt(bad, tmp_path)
    bad2 = dict(receipt)
    bad2["per_lag"] = {1: {"pos": -1.0, "neg": 1.0, "alarmed": False}}
    with pytest.raises(ValueError, match="contract"):
        write_xwatch_receipt(bad2, tmp_path)


def test_contract_catches_inconsistent_flags() -> None:
    a, b = _lead_stream(200, 37, lag=1)
    receipt = xwatch_report(a, b, data_label="SYNTHETIC")
    bad = json.loads(json.dumps(receipt))
    bad["per_lag"]["2"]["alarmed"] = True  # no lag-2 pair fired — must flag
    assert xwatch_contract_errors(bad)
    bad2 = json.loads(json.dumps(receipt))
    bad2["alarmed_pairs"].append("lag9_pos")  # outside [1, n_lags]
    assert xwatch_contract_errors(bad2)
    bad3 = json.loads(json.dumps(receipt))
    bad3["pooled_evalue"] = -1.0
    assert xwatch_contract_errors(bad3)
    bad4 = json.loads(json.dumps(receipt))
    bad4["data_label"] = ""
    assert xwatch_contract_errors(bad4)


def test_xwatch_receipt_v2_round_trip(tmp_path: Path) -> None:
    """receipt_version=2 seals the same xwatch body in the envelope."""
    from quant_fund.research.receipt_v2 import verify_receipt_file

    a, b = _lead_stream(300, 11, lag=1)
    receipt = xwatch_report(a, b, n_lags=3, data_label="SYNTHETIC")
    path = write_xwatch_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "xwatch.v1"
    assert payload["payload"]["alarmed_pairs"] == receipt["alarmed_pairs"]
    assert verify_receipt_file(path)["valid"] is True
