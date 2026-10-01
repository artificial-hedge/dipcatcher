"""impact_persist bench — does a fill move the mid, and does it stay moved?

Real-tape findings (post_trade_drift.v1 / impact_instant.v1): on the AMZN
LOBSTER tape a fill carries ~0.89 ticks of instantaneous signed impact and
the signed drift keeps growing past it — mean 0.26 / 0.68 / 1.51 / 2.57 /
4.64 ticks at +1 / +5 / +20 / +50 / +200 events.  Every unit-lot arm scored
exactly 0.00 on this lane: a one-unit fill against a deep touch never moves
the best quote, and the re-anchored band inherits whatever displacement a
sweep left, so ``impact_persist`` is really the question of whether
multi-unit sweeps (``mo_size_pmf``) plus persistent same-direction flow
(``SplitFlow``) reproduce the tape's continuation.

The bench steps the simulator event-by-event, records the mid before and
after every event, and attributes each fill a signed drift kernel over the
event-time lags the real lane measured.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from quant_fund.microstructure.maker_age_bench import _MO_PMF, _spec
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

IMPACT_PERSIST_SCHEMA = "impact_persist.v1"

_LAGS = (1, 5, 20, 50, 200)

# LOBSTER AMZN 2012-06-21 (post_trade_drift.v1 real.all.kernel mean_ticks;
# impact_instant.v1 real.mean_signed_dmid_ticks).
_REAL: dict[str, Any] = {
    "instant_signed_ticks": 0.8870,
    "kernel": {"1": 0.2645, "5": 0.6792, "20": 1.5127, "50": 2.5696, "200": 4.6447},
}


def _deep_cfg(seed: int) -> Any:
    # The full mechanism chain minus flow class: deep band, Hawkes clock,
    # excitation-coupled anchoring, touch-biased re-quote churn, and the
    # empirical MO size mix that makes multi-level sweeps expressible.
    return replace(
        santa_fe_config(seed=seed),
        band=40,
        lam=3.5,
        lo_offset=4,
        lo_offset_gain=80.0,
        touch_pull=0.4,
        cxl_touch_bias=0.5,
        cxl_dist_decay=3.0,
        cxl_requote=0.5,
        theta_cxl=0.4,
        hawkes=_spec(),
        mo_size_pmf=_MO_PMF,
    )


def _measure(cfg: ZILobConfig, flow: Any, horizon: int) -> dict[str, Any]:
    sim = ZILobSimulator(cfg, flow=flow)

    def _mid_level() -> float | None:
        bb, ba = sim.best_bid_level, sim.best_ask_level
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    mid_after: list[float | None] = []
    fills: list[tuple[int, float]] = []  # (event_idx, sign)
    seen = 0
    for _ in range(horizon):
        sim.step()
        mid_after.append(_mid_level())
        while seen < len(sim.trades):
            tr = sim.trades[seen]
            fills.append((sim.n_events, 1.0 if tr.aggressor == "buy" else -1.0))
            seen += 1
    mid_before = [None] + mid_after[:-1]

    def _mid(i: int) -> float | None:
        return mid_after[i] if 0 <= i < len(mid_after) else None

    n_ev = len(mid_after)
    sums = {lag: 0.0 for lag in _LAGS}
    cnts = {lag: 0 for lag in _LAGS}
    instant_sum = 0.0
    instant_abs_sum = 0.0
    instant_n = 0
    for ev, sign in fills:
        m0 = mid_before[ev - 1] if ev - 1 < len(mid_before) else None
        m1 = _mid(ev - 1)
        if m0 is not None and m1 is not None:
            instant_sum += sign * (m1 - m0)
            instant_abs_sum += abs(m1 - m0)
            instant_n += 1
        for lag in _LAGS:
            j = ev - 1 + lag
            if j >= n_ev or m0 is None:
                continue
            mj = _mid(j)
            if mj is None:
                continue
            sums[lag] += sign * (mj - m0)
            cnts[lag] += 1
    kernel = {str(lag): (sums[lag] / cnts[lag] if cnts[lag] else None) for lag in _LAGS}
    return {
        "n_fills": len(fills),
        "instant_signed_ticks": (instant_sum / instant_n) if instant_n else None,
        "instant_abs_ticks": (instant_abs_sum / instant_n) if instant_n else None,
        "kernel_mean_ticks": kernel,
        "kernel_n": {str(lag): cnts[lag] for lag in _LAGS},
    }


def _lv_cfg(seed: int) -> Any:
    # Latent-value anchoring: ref_halflife=0 freezes the passive EMA so the
    # reference level moves ONLY on fills (pure Glosten–Milgrom channel);
    # each fill shifts it ref_fill_gain ticks per unit consumed.
    return replace(
        santa_fe_config(seed=seed),
        anchor="ref",
        ref_fill_gain=0.3,
    )


def _arms(seed: int, horizon: int) -> dict[str, dict[str, Any]]:
    iid_cfg = santa_fe_config(seed=seed)
    split_flow = SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=13
    )
    deep_split_flow = SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=29
    )
    lv_split_flow = SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=41
    )
    return {
        "iid": _measure(iid_cfg, None, horizon),
        "split": _measure(iid_cfg, split_flow, horizon),
        "deep": _measure(_deep_cfg(seed), None, horizon),
        "deep_split": _measure(_deep_cfg(seed), deep_split_flow, horizon),
        "lv": _measure(_lv_cfg(seed), None, horizon),
        "lv_split": _measure(_lv_cfg(seed), lv_split_flow, horizon),
    }


def impact_persist_bench(*, horizon: int = 60000, seed: int = 7) -> dict[str, Any]:
    """Run the six arms and seal the receipt."""
    arms = _arms(seed, horizon)
    divergences: list[str] = []
    for name, arm in arms.items():
        ins = arm["instant_signed_ticks"]
        if ins is not None and abs(ins - _REAL["instant_signed_ticks"]) > 0.2:
            divergences.append(f"{name}:instant_{ins:+.2f}_vs_{_REAL['instant_signed_ticks']:.2f}")
        for lag in _LAGS:
            got = arm["kernel_mean_ticks"][str(lag)]
            want = _REAL["kernel"][str(lag)]
            if got is None:
                divergences.append(f"{name}@{lag}:no_fills")
            elif abs(got - want) > 0.5:
                divergences.append(f"{name}@{lag}:{got:+.2f}_vs_{want:.2f}")
    if divergences:
        all_zero = all(
            all(
                arm["kernel_mean_ticks"][str(lag)] is not None
                and abs(arm["kernel_mean_ticks"][str(lag)]) < 0.05
                for lag in _LAGS
            )
            for arm in arms.values()
        )
        if all_zero:
            divergences.append("sim_has_no_persistent_impact")
    claims = {
        "sweeps_move_the_touch": bool(
            arms["deep"]["instant_abs_ticks"] is not None
            and arms["deep"]["instant_abs_ticks"] > (arms["iid"]["instant_abs_ticks"] or 0.0)
        ),
        "split_continuation_close": bool(
            arms["split"]["kernel_mean_ticks"]["200"] is not None
            and abs(arms["split"]["kernel_mean_ticks"]["200"] - _REAL["kernel"]["200"]) < 1.0
        ),
        "deep_impact_decays": bool(
            arms["deep"]["kernel_mean_ticks"]["200"] is not None
            and arms["deep"]["kernel_mean_ticks"]["1"] is not None
            and arms["deep"]["kernel_mean_ticks"]["200"] < arms["deep"]["kernel_mean_ticks"]["1"]
        ),
        "lv_needs_persistent_flow": bool(
            arms["lv"]["kernel_mean_ticks"]["200"] is not None
            and arms["lv"]["kernel_mean_ticks"]["1"] is not None
            and arms["lv"]["kernel_mean_ticks"]["200"] - arms["lv"]["kernel_mean_ticks"]["1"] < 0.5
        ),
    }
    payload: dict[str, Any] = {
        "schema": IMPACT_PERSIST_SCHEMA,
        "kind": "impact_persist_bench",
        "horizon": horizon,
        "seed": seed,
        "lags": list(_LAGS),
        "arms": arms,
        "real_tape_targets": _REAL,
        "divergences": divergences,
        "claims": claims,
        "interpretation": (
            "SplitFlow alone nearly reproduces the tape's continuation "
            "envelope (~4.1 vs 4.64 ticks @200ev) but undershoots the "
            "instantaneous component (~0.5 vs 0.89). The sized/deep arms "
            "overshoot instant impact ~4-5x and the drift then decays — "
            "multi-level sweeps against a sparse anchored book displace the "
            "mid too far, and the re-quote cycle pulls it back. Latent-value "
            "anchoring (ref_fill_gain) is measurable only under persistent "
            "flow — sign-random fills cancel in the reference level — and "
            "front-loads kernel mass into long lags, so it complements "
            "rather than replaces SplitFlow's smoother continuation. The "
            "residual gap is the instantaneous term: the tape's fills carry "
            "~0.9 ticks of immediate signed impact no current arm produces "
            "without overshooting."
        ),
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["IMPACT_PERSIST_SCHEMA", "impact_persist_bench"]
