"""Adversarial probes for the microstructure flow lane (wave audit).

Every probe is SYNTHETIC and seeded — correctness evidence only. The suite
pins the lane's audit fixes: zero-fill market events must not fabricate
sign entries, episode seeds must actually reseed the engine, forward-vol
windows must not overlap the predictor window, and the propagator cost
curve must match the schedule quadratic form.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.exec_cost_split import exec_episode
from quant_fund.microstructure.flow_memory import sign_sequence
from quant_fund.microstructure.intraday_exec import sim_intraday_exec
from quant_fund.microstructure.propagator_impact import (
    expected_cost_curve,
    fit_kernel_ls,
)
from quant_fund.microstructure.sign_autocorr_real import lobster_signs
from quant_fund.microstructure.trade_decomp import sim_trade_decomp
from quant_fund.microstructure.vpin import _vpin_stats
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)

# ---------------------------------------------------------------------------
# flow_memory.sign_sequence — zero-fill market events


def test_sign_sequence_skips_zero_fill_market_events(monkeypatch):
    """A "market" step that fills nothing must not re-emit the prior sign.

    Before the fix the sequence read ``sim.trades[-1]`` unconditionally, so
    every no-fill market event duplicated the previous fill's aggressor —
    fabricating sign memory that never happened.
    """
    cfg = ZILobConfig(seed=3)
    expected: list[float] = []
    calls = {"n": 0}
    real_step = ZILobSimulator.step

    def patched(self: ZILobSimulator) -> str:
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            # phantom market event: clock advances, no fill is appended
            self._t += 1.0  # noqa: SLF001 — probing internals on purpose
            return "market"
        n0 = len(self.trades)
        kind = real_step(self)
        if kind == "market" and len(self.trades) > n0:
            expected.append(1.0 if self.trades[n0].aggressor == "buy" else -1.0)
        return kind

    monkeypatch.setattr(ZILobSimulator, "step", patched)
    out = sign_sequence(config=cfg, horizon=60.0)
    assert calls["n"] > 10, "phantom events never fired — probe is vacuous"
    assert out.tolist() == expected


# ---------------------------------------------------------------------------
# exec_cost_split.exec_episode — episode_seed must reseed the engine


def test_exec_episode_seed_makes_episodes_independent() -> None:
    """Same-side episodes with different seeds must take different paths.

    ``episode_seed`` used to be swallowed (``_ = episode_seed``), so every
    same-side episode replayed the identical ZI stream and ``n_episodes``
    overstated independent evidence.
    """
    cfg = ZILobConfig(seed=7)
    kw = {"parent_size": 20, "n_children": 4, "events_per_child": 20, "side": "buy"}
    a = exec_episode(cfg, None, episode_seed=101, **kw)
    b = exec_episode(cfg, None, episode_seed=102, **kw)
    c = exec_episode(cfg, None, episode_seed=101, **kw)
    assert a is not None and b is not None and c is not None
    assert a == c  # deterministic in episode_seed
    assert a["shortfall_ticks"] != b["shortfall_ticks"]


# ---------------------------------------------------------------------------
# vpin._vpin_stats — forward vol must start strictly after the VPIN window


def test_vpin_forward_vol_does_not_overlap_predictor_window() -> None:
    """A mid jump placed strictly after a window must land on the right j.

    Crafted stream: bucket imbalance decreases in bucket index (vpin[j] =
    29-2j), and a single mid step sits between starts[4] and starts[5]
    (window=2 → it is the forward move for j=2, where vpin is just above
    its mean). The overlapped indexing would attribute the jump to j=4 —
    where vpin is below its mean — flipping the measured sign.
    """
    n_buckets, per_bucket, window = 8, 30, 2
    signs: list[float] = []
    for k in range(n_buckets):
        n_buy = per_bucket - k  # imbalance 30-2k, decreasing in k
        signs.extend([1.0] * n_buy + [-1.0] * k)
    mid = np.ones(len(signs))
    mid[per_bucket * 5 :] = 10.0  # one permanent jump at starts[5]
    out = _vpin_stats(np.asarray(signs), mid, n_buckets=n_buckets, window=window)
    assert out.get("ok") is not False
    corr = out["vpin_fwd_vol_corr"]
    assert corr is not None and math.isfinite(corr)
    assert corr > 0.0, "post-window move must pair with the correct predictor"


# ---------------------------------------------------------------------------
# propagator_impact — cost curve is the schedule quadratic form


def test_expected_cost_curve_matches_schedule_quadratic_form() -> None:
    """cost(t) = sum_{i,j} (1/t)^2 G(|i-j|), verified by brute force.

    The diagonal has ``t`` pairs contributing ``t*G(0)`` — the old code used
    ``G(0)`` once and underweighted the instantaneous component by a factor
    of t.
    """
    g = np.array([1.0, 0.5, 0.25])
    got = expected_cost_curve(g, horizon=5)
    expected = []
    for t in range(1, 6):
        w = 1.0 / t
        c = 0.0
        for i in range(t):
            for j in range(t):
                lag = abs(i - j)
                if lag < g.size:
                    c += w * w * g[lag]
        expected.append(c)
    assert np.allclose(got, np.asarray(expected), rtol=1e-12)


def test_fit_kernel_ls_rejects_degenerate_ridge() -> None:
    """A non-positive ridge leaves a singular gram — fail closed."""
    rng = np.random.default_rng(0)
    flow = rng.choice([-1.0, 1.0], size=400)
    resp = np.convolve(flow, [1.0, 0.5, 0.25])[:400]
    with pytest.raises(ValueError, match="ridge"):
        fit_kernel_ls(flow, resp, n_lags=4, ridge=0.0)
    with pytest.raises(ValueError, match="ridge"):
        fit_kernel_ls(flow, resp, n_lags=4, ridge=float("nan"))


# ---------------------------------------------------------------------------
# sign_autocorr_real — hidden executions count, like sibling sign modules


def test_lobster_signs_counts_hidden_executions(tmp_path: Path) -> None:
    """EXECUTION_HIDDEN (type 5) carries an aggressor sign too.

    Sibling sign modules count {4, 5}; dropping hidden execs measured a
    different (thinner) stream than the sibling lanes.
    """
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        # time, event_type, order_id, size, price, direction
        w.writerow([34200.0, 1, 1, 100, 1000000, 1])  # submission
        w.writerow([34200.1, 4, 1, 50, 1000000, 1])  # visible execution
        w.writerow([34200.2, 5, 2, 30, 1000100, -1])  # hidden execution
    out = lobster_signs(tmp_path, ticker="AMZN")
    assert out["n_execs"] == 2


# ---------------------------------------------------------------------------
# zi_lob_simulator.open_order_sides — public live-registry view


def test_open_order_sides_reflects_live_book() -> None:
    """The snapshot contains only 'buy'/'sell' and tracks resting orders."""
    cfg = santa_fe_config(seed=5)
    sim = ZILobSimulator(cfg)
    for _ in range(200):
        sim.step()
    sides = sim.open_order_sides()
    assert sides
    assert set(sides) <= {"buy", "sell"}
    bid = sim.best_bid
    assert bid is not None
    oid = sim.submit_limit_order("buy", bid - 20 * cfg.tick, tag="probe")
    assert oid in sim._orders  # noqa: SLF001 — probing internals on purpose
    after = sim.open_order_sides()
    assert len(after) == len(sides) + 1
    assert after.count("buy") == sides.count("buy") + 1


# ---------------------------------------------------------------------------
# Note-label honesty — the rescaled-time notes describe a MEAN rate


def test_rescaled_time_notes_say_mean_not_median() -> None:
    """The rescaling uses horizon/sim.t — an event-RATE mean, not a median."""
    out = sim_trade_decomp(None, seed=0, horizon=2000)
    assert "mean event rate" in out["note"]
    assert "median" not in out["note"]
    out2 = sim_intraday_exec(None, seed=0, horizon=2000)
    assert "mean event rate" in out2["note"]
    assert "median" not in out2["note"]


# ---------------------------------------------------------------------------
# Honesty / fail-closed surface


def test_benches_fail_closed_without_lobster_tape(tmp_path: Path) -> None:
    """Real-tape benches refuse to run on a missing tape — no silent synth."""
    from quant_fund.microstructure.markout import markout_bench
    from quant_fund.microstructure.sign_predict import sign_predict_bench

    with pytest.raises((FileNotFoundError, OSError)):
        markout_bench(tmp_path, ticker="AMZN", sim_seconds=200.0)
    with pytest.raises((FileNotFoundError, OSError)):
        sign_predict_bench(tmp_path, ticker="AMZN")
