"""event_granger — do quote-side events predict executions?

Binned counting processes per LOBSTER event type (10ms bins):
SUBMISSION / CANCEL_PARTIAL / DELETE / EXECUTION. Lagged
cross-correlations between every (type_a → type_b) pair answer: does
a cancel burst predict imminent executions? Do submissions chase
executions? The book-thinning signature of sweeps — depth pulled
right before it gets hit — shows up as cancel_rate leading exec_rate.

Sim arms: the sim has no information coupling between order flow
phases — event types are drawn from fixed mixture weights — so all
cross-correlations should sit near zero (the honest null).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    HALT,
    SUBMISSION,
    parse_messages,
)
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_BIN_S = 0.01
_MAX_LAG_BINS = 50  # 0.5s
_EVENT_NAMES = {
    SUBMISSION: "submit",
    CANCEL_PARTIAL: "cancel_partial",
    DELETE: "delete",
    EXECUTION: "exec",
    EXECUTION_HIDDEN: "exec_hidden",
    HALT: "halt",
}
_KINDS = ("submit", "cancel_partial", "delete", "exec")


def _binned_counts(times: np.ndarray, types: np.ndarray, bin_s: float) -> dict[str, np.ndarray]:
    t0 = times[0]
    n_bins = int((times[-1] - t0) / bin_s) + 1
    out = {}
    for et, name in _EVENT_NAMES.items():
        c = np.zeros(n_bins)
        idx = ((times[types == et] - t0) / bin_s).astype(int)
        np.add.at(c, idx, 1)
        out[name] = c
    return out


def _xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """corr(a[t], b[t+lag]) for lag in 0..max_lag (b leads)."""
    a = a - a.mean()
    b = b - b.mean()
    na = a.size
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return np.zeros(max_lag + 1)
    out = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        out[lag] = (a[: na - lag] * b[lag:]).sum() / denom
    return out


def _pairwise(counts: dict[str, np.ndarray]) -> dict[str, Any]:
    """Peak cross-correlation and its lag for each (a→b) pair.

    ``exec`` folds in hidden-liquidity prints: a type-5 fill is still a
    market-order arrival, and leaving it out of the MO channel undercounts
    the very stream the Granger lags are meant to describe.
    """
    exec_all = counts["exec"] + counts.get("exec_hidden", np.zeros_like(counts["exec"]))
    merged = {**counts, "exec": exec_all}
    pairs: dict[str, Any] = {}
    for a in _KINDS:
        for b in _KINDS:
            if a == b:
                continue
            xc = _xcorr(merged[a], merged[b], _MAX_LAG_BINS)
            peak = int(np.argmax(np.abs(xc)))
            pos = int(np.argmax(np.abs(xc[1:]))) + 1  # best strictly-positive lag
            pairs[f"{a}->{b}"] = {
                "peak_corr": float(xc[peak]),
                "peak_lag_bins": peak,
                "peak_lag_s": peak * _BIN_S,
                "lag0_corr": float(xc[0]),
                "peak_corr_lead": float(xc[pos]),
                "peak_lag_lead_s": pos * _BIN_S,
            }
    return pairs


def lobster_event_granger(msg_path: Path) -> dict[str, Any]:
    times: list[float] = []
    types: list[int] = []
    for ev in parse_messages(msg_path):
        times.append(ev.time_s)
        types.append(ev.event_type)
    counts = _binned_counts(np.asarray(times), np.asarray(types), _BIN_S)
    out = _pairwise(counts)
    out["n_events"] = len(times)
    out["ok"] = len(times) > 0
    return out


def sim_event_granger(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    times: list[float] = []
    kinds: list[str] = []
    for _ in range(horizon):
        times.append(sim.t)
        kinds.append(sim.step())
    ta = np.asarray(times)
    ka = np.asarray(kinds)
    n_bins = int((ta[-1] - ta[0]) / _BIN_S) + 1

    def _bins(kind: str) -> np.ndarray:
        c = np.zeros(n_bins)
        idx = ((ta[ka == kind] - ta[0]) / _BIN_S).astype(int)
        np.add.at(c, idx, 1)
        return c

    # map the sim's event vocabulary onto the LOBSTER pair names:
    # limit→submit, market→exec, cancel→delete (partial has no analog)
    counts = {
        "submit": _bins("limit"),
        "exec": _bins("market"),
        "exec_hidden": np.zeros(n_bins),  # sim has no hidden-fill channel
        "delete": _bins("cancel"),
        "cancel_partial": np.zeros(n_bins),
    }
    out = _pairwise(counts)
    out["n_events"] = len(times)
    out["ok"] = True
    return out


def event_granger_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    if not msg:
        raise FileNotFoundError(f"no LOBSTER message CSV under {tape_dir}")
    real = lobster_event_granger(msg[0])
    arms = {
        "iid": sim_event_granger(seed=seed),
        "regime": sim_event_granger(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_event_granger(
            flow=SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        for name, arm in arms.items():
            if not arm.get("ok"):
                continue
            for pair, vals in real.items():
                if not isinstance(vals, dict) or "peak_corr" not in vals:
                    continue
                s = arm.get(pair, {}).get("peak_corr")
                if s is not None and abs(vals["peak_corr"] - s) > 0.15:
                    divergences.append(f"{name}:{pair}_{s:.2f}_vs_{vals['peak_corr']:.2f}")
    payload: dict[str, Any] = {
        "kind": "event_granger",
        "schema": "event_granger.v1",
        "ticker": ticker,
        "bin_s": _BIN_S,
        "max_lag_s": _MAX_LAG_BINS * _BIN_S,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "event_type_cross_correlation_measured",
        "interpretation": (
            "Per-pair cross-correlation of 10ms event counts, "
            "corr(a[t], b[t+lag]): 'a->b' = a leading b. lag0_corr is "
            "same-bin co-movement (coupling, not prediction); "
            "peak_corr_lead / peak_lag_lead_s is the strongest strictly "
            "positive-lag signal — the true lead. Book thinning shows as "
            "cancel/delete→exec peaks; post-exec retreat as exec→delete."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
