"""Anytime-valid serial-independence audit for PIT streams.

Every other sequential lane in the suite tests a *marginal* property —
coverage rate, calibration, tail depth, level drift. None of them see a
*dynamically* misspecified forecaster whose PITs are marginally uniform
but serially dependent (AR structure in the probability transforms — the
signature of a model that has the right marginals but the wrong
dynamics).

Construction (sign-product e-process; Ville + union bound):

- PITs ``u_t in (0,1)`` map to signs ``s_t = sign(u_t - 1/2)`` in
  ``{-1, +1}``. Under the null — correct dynamic spec, so PITs are iid
  and median-symmetric — ``s_t`` is iid Rademacher: ``E[s_t | F_{t-1}]
  = 0`` *exactly*, not just bounded.
- For lag ``k`` and direction ``d in {+,-}`` the product ``w_t =
  s_t * s_{t-k}`` satisfies ``E[w_t | F_{t-1}] = 0`` under H0, so
  ``e_t = 1 + d * lam * w_t`` with fixed ``lam in (0,1)`` is a
  nonnegative test martingale: ``E[e_t | F_{t-1}] = 1``. Positive
  autocorrelation grows the ``+`` process, negative grows the ``-``.
- ``K`` lags × 2 directions gives ``2K`` processes; a per-process alarm
  needs ``E >= 2K / alpha`` (Bonferroni across the family — the
  ``any_lag_alarmed`` claim holds at level ``alpha``), while the pooled
  statistic ``(1/2K) * sum(E)`` — an e-value in its own right — alarms
  at ``1/alpha`` as a *separate* ``pooled_alarmed`` claim also at level
  ``alpha``. They are reported as distinct claims: merging them into one
  flag would silently double the joint level to ``2*alpha``.

Design choices:

- Sign bets are distribution-free: no normality assumption on the PIT
  transform, only median-symmetry + independence — the widest null a
  serial-correlation claim can honestly test without parametric bets.
- Ties: ``u_t == 0.5`` exactly contributes ``s_t = 0`` — a degenerate
  factor ``e_t = 1`` that neither gains nor loses evidence.
- Fail closed: ``u`` outside ``(0,1)`` or non-finite raises; an empty
  stream raises on ``update``; unknown lag/direction in accessors raise.

Receipt: ``serial_watch.v1`` — sealed by ``write_serial_receipt``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERIAL_WATCH_SCHEMA = "serial_watch.v1"


def _sign(u: float) -> float:
    if not math.isfinite(u) or not (0.0 < u < 1.0):
        raise ValueError(f"PIT value outside (0,1): {u!r}")
    x = u - 0.5
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


@dataclass(frozen=True)
class SerialState:
    """Snapshot after one PIT observation."""

    origin: int
    pooled_evalue: float
    any_lag_alarmed: bool
    pooled_alarmed: bool
    alarmed_lags: tuple[int, ...]
    newly_alarmed: tuple[int, ...]


@dataclass
class SerialWatch:
    """Per-lag sign-product e-processes over a PIT stream.

    ``update(u)`` appends one PIT in ``(0,1)``. Alarm per (lag, sign of
    autocorrelation) is permanent — "ever crossed" is the event Ville
    controls.
    """

    n_lags: int = 5
    alpha: float = 0.05
    lam: float = 0.5
    _s: list[float] = field(default_factory=list, init=False)
    _e: dict[tuple[int, float], float] = field(default_factory=dict, init=False)
    _alarm_origin: dict[tuple[int, float], int] = field(default_factory=dict, init=False)
    _states: list[SerialState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lam must lie in (0, 1)")
        for k in range(1, self.n_lags + 1):
            for d in (1.0, -1.0):
                self._e[(k, d)] = 1.0

    def update(self, u: float) -> SerialState:
        """Append one PIT observation; return the new state."""
        s_t = _sign(float(u))
        t = len(self._s)
        newly: list[int] = []
        for k in range(1, self.n_lags + 1):
            if t < k:
                continue
            w = s_t * self._s[t - k]
            for d in (1.0, -1.0):
                e_t = 1.0 + d * self.lam * w
                self._e[(k, d)] *= e_t
                if (k, d) not in self._alarm_origin and self._e[
                    (k, d)
                ] >= 2 * self.n_lags / self.alpha:
                    self._alarm_origin[(k, d)] = t
                    newly.append(k)
        self._s.append(s_t)
        pooled = sum(self._e.values()) / (2 * self.n_lags)
        alarmed_lags = tuple(sorted({k for (k, _d) in self._alarm_origin}))
        state = SerialState(
            origin=t,
            pooled_evalue=float(pooled),
            any_lag_alarmed=bool(alarmed_lags),
            pooled_alarmed=pooled >= 1.0 / self.alpha,
            alarmed_lags=alarmed_lags,
            newly_alarmed=tuple(sorted(set(newly))),
        )
        self._states.append(state)
        return state

    @property
    def lag_evalues(self) -> dict[tuple[int, float], float]:
        return dict(self._e)

    @property
    def alarm_origin(self) -> dict[tuple[int, float], int]:
        """First alarming origin per (lag, direction)."""
        return dict(self._alarm_origin)

    @property
    def states(self) -> list[SerialState]:
        return list(self._states)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def serial_report(
    pits: Any,
    *,
    n_lags: int = 5,
    alpha: float = 0.05,
    lam: float = 0.5,
    data_label: str = "UNKNOWN",
) -> dict[str, Any]:
    """Receipt-shaped serial-independence verdict over a PIT stream.

    ``pits`` is a sequence of values in ``(0,1)`` — e.g. Rosenblatt
    transforms ``F_t(y_t)`` from a density forecaster. Fails closed on
    empty/out-of-range/non-finite input. ``data_label`` stamps the
    receipt's provenance (bare arrays carry none — default UNKNOWN).
    """
    if not isinstance(data_label, str) or not data_label.strip():
        raise ValueError("data_label must be a nonempty string")
    import numpy as np

    u = np.asarray(list(pits), dtype=np.float64).reshape(-1)
    if u.size == 0:
        raise ValueError("PIT stream must be nonempty")
    if not np.isfinite(u).all():
        raise ValueError("PIT stream contains non-finite values")

    watch = SerialWatch(n_lags=n_lags, alpha=alpha, lam=lam)
    for x in u.tolist():
        watch.update(float(x))
    final = watch.states[-1]

    per_lag = {
        k: {
            "pos": watch.lag_evalues[(k, 1.0)],
            "neg": watch.lag_evalues[(k, -1.0)],
            "alarmed": k in final.alarmed_lags,
        }
        for k in range(1, n_lags + 1)
    }
    return {
        "kind": SERIAL_WATCH_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": data_label,
        "alpha": alpha,
        "lam": lam,
        "n_lags": n_lags,
        "n_origins": int(u.size),
        "per_lag": per_lag,
        "alarmed_lags": list(final.alarmed_lags),
        "alarm_origins": {
            f"lag{k}_{'pos' if d > 0 else 'neg'}": int(o)
            for (k, d), o in watch.alarm_origin.items()
        },
        "pooled_evalue": float(final.pooled_evalue),
        "any_lag_alarmed": bool(final.any_lag_alarmed),
        "pooled_alarmed": bool(final.pooled_alarmed),
        "evidence": [
            "ville_inequality",
            "rademacher_sign_products",
            "union_bound_2k_processes",
            "bonferroni_per_lag",
            "anytime_valid",
        ],
    }


def write_serial_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a serial_watch receipt and write ``serial_watch_<hash>.json``.

    Filename digest = sha256 of the canonical payload, embedded as
    ``receipt_sha256`` (fleet_eval seal convention). Atomic, fail-closed
    on a malformed receipt. ``receipt_version=2`` wraps the same body in
    the unified ``receipt.v2`` envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    if (
        receipt.get("kind") != SERIAL_WATCH_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("per_lag"), Mapping)
        or not isinstance(receipt.get("any_lag_alarmed"), bool)
        or not isinstance(receipt.get("pooled_alarmed"), bool)
    ):
        raise ValueError("serial_watch receipt violates its contract")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass",
                params={
                    "alpha": receipt.get("alpha"),
                    "lam": receipt.get("lam"),
                    "n_lags": receipt.get("n_lags"),
                    "n_origins": receipt.get("n_origins"),
                },
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"serial_watch_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "SERIAL_WATCH_SCHEMA",
    "SerialState",
    "SerialWatch",
    "serial_report",
    "write_serial_receipt",
]
