"""split_flow_bench — does metaorder splitting reproduce flow persistence?

SYNTHETIC / research-diagnostic only. Measures the sign-autocorrelation
decay curve of the ZI-LOB trade stream under three flow drivers — iid,
two-state Markov modulation, metaorder splitting — and seals the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.split_flow import SplitFlow, sign_autocorr_curve
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SPLIT_FLOW_SCHEMA = "split_flow.v1"

_LAGS = (1, 2, 4, 8, 16, 32, 64)


def _collect_signs(cfg: ZILobConfig, flow: Any, horizon: int) -> NDArray[np.float64]:
    sim = ZILobSimulator(cfg, flow=flow)
    signs: list[float] = []
    for _ in range(horizon):
        sim.step()
        while len(sim.trades) > len(signs):
            tr = sim.trades[len(signs)]
            signs.append(1.0 if tr.aggressor == "buy" else -1.0)
    return np.asarray(signs)


def _arm(name: str, cfg: ZILobConfig, flow: Any, horizon: int) -> dict[str, Any]:
    signs = _collect_signs(cfg, flow, horizon)
    return {
        "name": name,
        "n_trades": int(signs.size),
        "autocorr": sign_autocorr_curve(signs, _LAGS),
        "mo_fraction": float(signs.size / horizon),
    }


def split_flow_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    cfg = ZILobConfig(seed=seed)
    arms = [
        _arm("iid", cfg, None, horizon),
        _arm(
            "regime",
            cfg,
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("trend", 2.2, 0.78)),
                stay_probs=(0.97, 0.94),
                seed=11,
            ),
            horizon,
        ),
        _arm(
            "split",
            cfg,
            # Realistic-split parameterization: ~10% of MOs open a parent,
            # parents are 10-600 unit children with Pareto tail 1.2, and
            # run at 3x background MO intensity (aggressive slicing).
            SplitFlow(
                p_start=0.10,
                size_tail=1.2,
                k_min=10,
                k_max=600,
                intensity_mult=3.0,
                seed=13,
            ),
            horizon,
        ),
    ]
    split = arms[2]
    lag1 = split["autocorr"]["lag1"]
    payload: dict[str, Any] = {
        "schema": SPLIT_FLOW_SCHEMA,
        "kind": "split_flow_bench",
        "horizon": horizon,
        "lags": list(_LAGS),
        "arms": arms,
        "reference_real_tape_sign_lag1": 0.721,
        "claims": {
            "split_produces_persistence": bool(lag1 is not None and lag1 > 0.2),
        },
        "interpretation": (
            "Metaorder splitting (same-sign child runs with heavy-tailed "
            "parent size) is the mechanism that lifts sign autocorrelation "
            "toward the real-tape 0.72; Markov intensity modulation alone "
            "cannot (it only tilts p_buy slowly). The split arm's "
            "parameters were chosen to match the real-tape lag-1 — this "
            "is a calibration, not an independent prediction"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
