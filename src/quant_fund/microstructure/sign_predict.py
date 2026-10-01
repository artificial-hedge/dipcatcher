"""sign_predict — conditional continuation probability of the exec-sign stream.

After a run of k consecutive same-sign fills, P(next fill keeps the sign)
measures how predictable order flow actually is. iid flow → 0.5 at every k;
metaorder slicing → elevated probability growing with k (more children
remaining); real tape — measured here.

This is the operational version of the Lillo–Farmer autocorrelation: not
just *that* signs correlate but *how much* the next sign is knowable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import parse_messages
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

EXEC_TYPES = frozenset({4, 5})
K_MAX = 15


def _continuation(signs: list[int]) -> dict[str, Any]:
    """P(next = current sign | run of exactly/at-least k same signs)."""
    s = np.asarray(signs, dtype=int)
    n = s.size
    if n < 50:
        return {"ok": False, "reason": "too_few_signs"}
    # run length ending at i (number of consecutive equal signs up to i)
    runlen = np.ones(n, dtype=int)
    for i in range(1, n):
        runlen[i] = runlen[i - 1] + 1 if s[i] == s[i - 1] else 1
    cont: dict[str, Any] = {}
    for k in range(1, K_MAX + 1):
        # positions i where the run ending at i has length ≥ k and i+1 exists
        idx = np.where((runlen >= k) & (np.arange(n) < n - 1))[0]
        if idx.size < 10:
            continue
        p = float(np.mean(s[idx + 1] == s[idx]))
        cont[str(k)] = {"n": int(idx.size), "p_continue": round(p, 4)}
    return {"n_signs": n, "continuation": cont}


def lobster_sign_predict(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    signs = [-ev.direction for ev in parse_messages(msg) if ev.event_type in EXEC_TYPES]
    return _continuation(signs)


def sim_sign_predict(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 30000,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    n_before = 0
    signs: list[int] = []
    for _ in range(horizon):
        sim.step()
        signs.extend(1 if tr.aggressor == "buy" else -1 for tr in sim.trades[n_before:])
        n_before = len(sim.trades)
    return _continuation(signs)


def sign_predict_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Continuation-probability comparison real vs sim. Sealed."""
    real = lobster_sign_predict(tape_dir, ticker)
    arms = {
        "iid": sim_sign_predict(seed=seed),
        "regime": sim_sign_predict(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_sign_predict(
            flow=SplitFlow(
                p_start=0.10,
                size_tail=1.2,
                k_min=10,
                k_max=600,
                intensity_mult=3.0,
                seed=seed + 2,
            ),
            seed=seed + 2,
        ),
    }
    divergences = []
    r5 = real.get("continuation", {}).get("5", {}).get("p_continue")
    for name, arm in arms.items():
        a5 = arm.get("continuation", {}).get("5", {}).get("p_continue")
        if r5 and a5 is not None and abs(a5 - r5) > 0.1:
            divergences.append(f"{name}_p5_{a5}_vs_{r5}")
    payload: dict[str, Any] = {
        "kind": "sign_predict",
        "schema": "sign_predict.v1",
        "ticker": ticker,
        "k_max": K_MAX,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "sign_continuation_measured",
        "interpretation": (
            "p_continue(k)=0.5 at all k under iid flow; the real curve's "
            "height and k-profile trace metaorder slicing depth."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
