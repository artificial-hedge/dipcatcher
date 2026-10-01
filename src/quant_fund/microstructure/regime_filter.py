"""Online regime filter: track the true Markov flow state from the tape.

``MarkovRegimeFlow`` hides the regime that drives MO intensity/direction.
A trading policy only sees the tape — inter-arrival times and signs.
This lane implements the classical Hamilton two-state filter over the
MO event stream:

    belief_i(t) = P(state_t = i | tape up to t)

Prediction uses the per-MO-clock transition matrix (stay_probs),
correction uses the event likelihood under each state:

    f_i(dt, sign) = mu_i * exp(-mu_i * dt) * p_i^sign * (1 - p_i)^(1-sign)

with mu_i = mu * intensity_mult_i and p_i = p_buy_i. The state model is
exactly the simulator's own contract, so the filter is *well-specified*:
the bench measures how much tape it needs to localize the regime.

- ``RegimeFilter`` — predict+update per MO event; posterior, MAP state,
  filtered intensity E[mu | tape].
- ``filtered_session`` — run the sim under a two-state flow, record the
  true regime at each MO and the filter's posterior.
- ``regime_filter_bench`` — sealed ``regime_filter.v1``: filter vs
  climatology on log-loss + Brier, median detection delay (events from
  a true transition to posterior crossing 0.5), transition-conditional
  accuracy.

SYNTHETIC only.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

REGIME_FILTER_SCHEMA = "regime_filter.v1"


def _prob(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0 or v >= 1.0:
        raise ValueError(f"{name} must be in (0,1), got {x!r}")
    return v


@dataclass(frozen=True)
class FilterConfig:
    """Two-state observation/transition model (the sim's own contract)."""

    mu: float  # base MO rate
    intensity: tuple[float, float]
    p_buy: tuple[float, float]
    stay: tuple[float, float]
    mu_floor: float = 1e-9


def _check_config(c: FilterConfig) -> FilterConfig:
    if not math.isfinite(c.mu) or c.mu <= 0.0:
        raise ValueError(f"mu must be positive and finite, got {c.mu!r}")
    for x in c.intensity:
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError(f"intensity entries must be positive, got {x!r}")
    for x in c.p_buy:
        _prob(x, "p_buy")
    for x in c.stay:
        _prob(x, "stay")
    return c


class RegimeFilter:
    """Hamilton filter over the MO event clock."""

    def __init__(self, config: FilterConfig, prior: tuple[float, float] = (0.5, 0.5)) -> None:
        _check_config(config)
        p0 = _prob(prior[0], "prior[0]")
        p1 = _prob(prior[1], "prior[1]")
        if abs(p0 + p1 - 1.0) > 1e-9:
            raise ValueError("prior must sum to 1")
        self._cfg = config
        self._b = np.asarray([p0, p1], dtype=np.float64)
        self._last_t: float | None = None

    @property
    def belief(self) -> NDArray[np.float64]:
        return self._b.copy()

    @property
    def map_state(self) -> int:
        return int(np.argmax(self._b))

    def update_neutral(self, t: float) -> NDArray[np.float64]:
        """Update on an MO whose side was not observed (no trade)."""
        return self._update(t, None)

    def update(self, t: float, is_buy: bool) -> NDArray[np.float64]:
        return self._update(t, is_buy)

    def _update(self, t: float, is_buy: bool | None) -> NDArray[np.float64]:
        """Posterior after an MO at time ``t`` with aggressor ``is_buy``."""
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"t must be non-negative and finite, got {t!r}")
        c = self._cfg
        # prediction: transition over one MO clock tick
        stay = np.asarray(c.stay)
        pred = np.asarray(
            [
                self._b[0] * stay[0] + self._b[1] * (1.0 - stay[1]),
                self._b[0] * (1.0 - stay[0]) + self._b[1] * stay[1],
            ]
        )
        dt = 0.0 if self._last_t is None else t - self._last_t
        if dt < 0.0:
            raise ValueError("non-monotone event times")
        mus = c.mu * np.asarray(c.intensity)
        ps = np.asarray(c.p_buy)
        # Poisson-arrival likelihood * sign likelihood (dt=0 at the first
        # event carries no rate information — use rate only)
        rate_lik = mus if self._last_t is None else mus * np.exp(-mus * dt)
        if is_buy is None:
            sign_lik = np.ones(2)
        else:
            sign_lik = ps if is_buy else (1.0 - ps)
        post = pred * rate_lik * sign_lik
        tot = float(post.sum())
        if not math.isfinite(tot) or tot <= 0.0:
            raise RuntimeError("degenerate posterior (all-zero likelihood)")
        self._b = post / tot
        self._last_t = t
        return self._b.copy()


def filtered_session(
    config: ZILobConfig,
    *,
    states: tuple[RegimeState, RegimeState],
    stay_probs: tuple[float, float],
    seed: int,
    horizon: float,
) -> dict[str, Any]:
    """Run the sim; record true state + filter posterior per MO."""
    _pos = float(horizon)
    if not math.isfinite(_pos) or _pos <= 0.0:
        raise ValueError(f"horizon must be positive and finite, got {horizon!r}")
    flow = MarkovRegimeFlow(states, stay_probs, seed=seed)
    filt = RegimeFilter(
        FilterConfig(
            mu=config.mu,
            intensity=(states[0].intensity_mult, states[1].intensity_mult),
            p_buy=(states[0].p_buy, states[1].p_buy),
            stay=stay_probs,
        )
    )
    cfg = dataclasses.replace(config, seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    true_states: list[int] = []
    posteriors: list[float] = []  # P(state=1) at each MO
    times: list[float] = []
    while sim.t < horizon:
        # the state that will generate this event is read BEFORE step()
        # (the flow advances inside step() after each MO)
        cur = flow.state_index
        n_trades_before = len(sim.trades)
        if sim.step() == "market":
            # a no-op MO (nothing consumed) leaves sim.trades unchanged;
            # infer side only when a fresh trade landed
            if len(sim.trades) > n_trades_before:
                is_buy = sim.trades[-1].aggressor == "buy"
            else:
                is_buy = None
            true_states.append(cur)
            if is_buy is None:
                # sign unobserved: rate-only update via neutral sign
                b = filt.update_neutral(sim.t)
            else:
                b = filt.update(sim.t, is_buy)
            posteriors.append(float(b[1]))
            times.append(sim.t)
    return {
        "times": np.asarray(times),
        "true_states": np.asarray(true_states, dtype=np.int64),
        "posteriors": np.asarray(posteriors),
        # transition records are (n_mo_1based, new_state_idx)
        "transitions": [m for m, _ in flow.transitions],
        "n_mo": len(times),
    }


def _logloss(truth: NDArray[np.int64], p: NDArray[np.float64], eps: float = 1e-9) -> float:
    pp = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(truth * np.log(pp) + (1 - truth) * np.log(1 - pp)))


def _brier(truth: NDArray[np.int64], p: NDArray[np.float64]) -> float:
    return float(np.mean((p - truth) ** 2))


def detection_delays(
    true_states: NDArray[np.int64],
    posteriors: NDArray[np.float64],
    transitions: list[int],
) -> list[int]:
    """Events from each true transition until the MAP state matches."""
    delays: list[int] = []
    for ti in transitions:
        if ti >= len(true_states):
            continue
        target = int(true_states[ti])
        d = None
        for k in range(ti, len(true_states)):
            if int(posteriors[k] >= 0.5) == target:
                d = k - ti
                break
        if d is not None:
            delays.append(d)
    return delays


def regime_filter_bench(
    *,
    horizon: float = 1500.0,
    seed: int = 0,
    states: tuple[RegimeState, RegimeState] = (
        RegimeState("calm", 1.0, 0.5),
        RegimeState("stress", 2.4, 0.62),
    ),
    stay_probs: tuple[float, float] = (0.99, 0.95),
) -> dict[str, Any]:
    """Sealed ``regime_filter.v1`` receipt."""
    from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

    _pos = float(horizon)
    if not math.isfinite(_pos) or _pos <= 0.0:
        raise ValueError(f"horizon must be positive and finite, got {horizon!r}")
    if len(states) != 2:
        raise ValueError("states must be a two-state flow")
    for x in stay_probs:
        _prob(x, "stay")
    cfg = santa_fe_config(seed=seed)
    out = filtered_session(cfg, states=states, stay_probs=stay_probs, seed=seed, horizon=horizon)
    if out["n_mo"] < 30:
        raise RuntimeError(f"too few MOs ({out['n_mo']})")
    truth = out["true_states"]
    post = out["posteriors"]
    base_rate = float(truth.mean())
    clim = np.full_like(post, base_rate)
    acc = float(np.mean((post >= 0.5).astype(np.int64) == truth))
    delays = detection_delays(truth, post, out["transitions"])
    payload: dict[str, Any] = {
        "schema": REGIME_FILTER_SCHEMA,
        "kind": "regime_filter",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "disclaimer": (
            "Hamilton filter tracking the true Markov flow regime on the "
            "synthetic ZI-LOB; never market evidence."
        ),
        "n_mo": int(out["n_mo"]),
        "n_transitions": len(out["transitions"]),
        "filter_logloss": _logloss(truth, post),
        "climatology_logloss": _logloss(truth, clim),
        "filter_brier": _brier(truth, post),
        "climatology_brier": _brier(truth, clim),
        "map_accuracy": acc,
        "stress_recall": (
            float(np.mean(post[truth == 1] >= 0.5)) if np.any(truth == 1) else float("nan")
        ),
        "calm_specificity": (
            float(np.mean(post[truth == 0] < 0.5)) if np.any(truth == 0) else float("nan")
        ),
        "median_detection_delay_events": (float(np.median(delays)) if delays else float("nan")),
        "mean_detection_delay_events": (float(np.mean(delays)) if delays else float("nan")),
        "stay_probs": [float(x) for x in stay_probs],
    }
    payload["payload_sha256"] = hash_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


__all__ = [
    "REGIME_FILTER_SCHEMA",
    "FilterConfig",
    "RegimeFilter",
    "detection_delays",
    "filtered_session",
    "regime_filter_bench",
]
