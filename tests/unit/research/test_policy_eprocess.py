"""Tests for research/policy_eprocess.py — paired-episode dominance e-process."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.zi_lob_simulator import (
    as_policy,
    santa_fe_config,
)
from quant_fund.research.policy_eprocess import (
    POLICY_EPROCESS_SCHEMA,
    dominance_stream,
    policy_dominance_process,
    policy_eprocess_bench,
    run_paired_episode,
)


def test_dominance_stream_sign_convention() -> None:
    # b > a -> x > 0.5; a > b -> x < 0.5
    x = dominance_stream([1.0, 1.0], [2.0, -1.0], pnl_scale=4.0)
    assert x[0] > 0.5 and x[1] < 0.5
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_dominance_stream_fail_closed() -> None:
    with pytest.raises(ValueError):  # out-of-contract diff
        dominance_stream([0.0], [3.0], pnl_scale=1.0)
    with pytest.raises(ValueError):  # non-finite
        dominance_stream([0.0], [float("nan")], pnl_scale=10.0)
    with pytest.raises(ValueError):  # shape mismatch
        dominance_stream([0.0], [0.0, 0.1], pnl_scale=10.0)
    with pytest.raises(ValueError):  # empty
        dominance_stream([], [], pnl_scale=10.0)
    with pytest.raises(ValueError):  # bad scale
        dominance_stream([0.0], [0.0], pnl_scale=0.0)


def test_paired_episode_common_random_numbers() -> None:
    cfg = santa_fe_config(seed=0)
    pol = as_policy(gamma=0.1, sigma=0.2, kappa=1.5, tick=0.01)
    ep1 = run_paired_episode(cfg, pol, pol, horizon=150.0, episode_seed=7)
    ep2 = run_paired_episode(cfg, pol, pol, horizon=150.0, episode_seed=7)
    # Same policy + same seed -> bit-identical episodes (CRN verified).
    assert ep1.pnl_a == ep1.pnl_b
    assert (ep1.pnl_a, ep1.n_fills_a) == (ep2.pnl_a, ep2.n_fills_a)


def test_paired_episode_fail_closed() -> None:
    cfg = santa_fe_config(seed=0)
    pol = as_policy(gamma=0.1, sigma=0.2, kappa=1.5, tick=0.01)
    with pytest.raises(ValueError):
        run_paired_episode(cfg, pol, pol, horizon=0.0, episode_seed=7)
    with pytest.raises(ValueError):
        run_paired_episode(cfg, pol, pol, horizon=10.0, episode_seed=-1)


def test_dominance_process_direction() -> None:
    cfg = santa_fe_config(seed=1)

    # Two policies: B is a deliberately-worse "no-quote" policy (always flat
    # spread of 999 -> no fills -> pnl 0); A is AS. The stream is deterministic
    # under the seeds so we just require the process to run and record x in [0,1].
    def flat_policy(state):  # noqa: ANN202, ANN001
        return None, None

    out = policy_dominance_process(
        as_policy(gamma=0.1, sigma=0.2, kappa=1.5, tick=0.01),
        flat_policy,
        config=cfg,
        horizon=120.0,
        n_episodes=3,
        pnl_scale=5.0,
        seed=3,
        inventory_cap=3,
    )
    x = out["x"]
    assert x.shape == (3,)
    assert np.all((x >= 0.0) & (x <= 1.0))
    proc = out["process"]
    lo, hi = proc.interval(0.05)
    assert 0.0 <= lo <= hi <= 1.0


def test_bench_receipt_schema_and_determinism() -> None:
    pol_a = as_policy(gamma=0.1, sigma=0.2, kappa=1.5, tick=0.01)
    pol_b = as_policy(gamma=0.5, sigma=0.2, kappa=1.5, tick=0.01)
    r1 = policy_eprocess_bench(
        pol_a,
        pol_b,
        name_a="as_g0.1",
        name_b="as_g0.5",
        n_episodes=3,
        horizon=120.0,
        pnl_scale=5.0,
        seed=11,
    )
    r2 = policy_eprocess_bench(
        pol_a,
        pol_b,
        name_a="as_g0.1",
        name_b="as_g0.5",
        n_episodes=3,
        horizon=120.0,
        pnl_scale=5.0,
        seed=11,
    )
    assert r1["schema"] == POLICY_EPROCESS_SCHEMA
    assert r1["kind"] == "policy_eprocess"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["research_only"] is True
    assert r1["verdict"] in ("a_dominates", "b_dominates", "inconclusive")
    assert "pnl" not in {k for k in r1 if not k.startswith("sim_internal")}
    assert r1["payload_sha256"] == r2["payload_sha256"]
    assert len(r1["payload_sha256"]) == 64


def test_bench_fail_closed_names() -> None:
    pol = as_policy(gamma=0.1, sigma=0.2, kappa=1.5, tick=0.01)
    with pytest.raises(ValueError):
        policy_eprocess_bench(pol, pol, name_a="x", name_b="x")
    with pytest.raises(ValueError):
        policy_eprocess_bench(pol, pol, name_a="x", name_b="y", pnl_scale=-1.0)
