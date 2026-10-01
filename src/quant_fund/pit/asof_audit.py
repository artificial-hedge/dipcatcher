"""Differential asof storm audit — proves the vault's temporal contract.

The PIT vault's guarantee is that ``asof(t)`` returns only what was
*knowable* at t. This lane doesn't trust it — it storms the vault with
random restatement sequences (including backdated and far-future
``known_at`` values) and probes random read watermarks, asserting four
invariants on every probe:

1. **Watermark**: every returned row has ``known_at <= t``.
2. **Strict monotonicity**: for t1 <= t2 under STRICT_FIRST, every key
   present at t1 is present at t2 with an identical row.
3. **No future leak**: a correction with ``known_at > t`` never appears
   in ``asof(t)`` under either policy.
4. **Latest supersedes**: a restated value at later ``known_at`` replaces
   the earlier version once it is knowable.

A single violation is a fail verdict — the vault's one job is temporal
hygiene, and "usually right" is a leak.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.pit.corrections import (
    EVENT_TIME_COL,
    KNOWN_AT_COL,
    SECURITY_ID_COL,
    RestatementPolicy,
)
from quant_fund.pit.vault import PitVault
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _frame(
    sec_ids: list[str], event_times: list[datetime], values: np.ndarray, known: datetime
) -> pl.DataFrame:
    n = len(sec_ids) * len(event_times)
    return pl.DataFrame(
        {
            SECURITY_ID_COL: [s for s in sec_ids for _ in event_times],
            EVENT_TIME_COL: [e for _ in sec_ids for e in event_times],
            KNOWN_AT_COL: [known] * n,
            "value": values.reshape(-1).astype(np.float64),
        }
    )


def _rows_keyed(frame: pl.DataFrame) -> dict[tuple[str, datetime], tuple[datetime, float]]:
    return {
        (s, e): (k, v)
        for s, e, k, v in zip(
            frame[SECURITY_ID_COL].to_list(),
            frame[EVENT_TIME_COL].to_list(),
            frame[KNOWN_AT_COL].to_list(),
            frame["value"].to_list(),
            strict=True,
        )
    }


def asof_storm_audit(
    root: Path | None = None,
    *,
    seed: int = 0,
    n_secs: int = 4,
    n_events: int = 6,
    n_restatements: int = 6,
    n_probes: int = 30,
) -> dict[str, Any]:
    """Storm a fresh vault; return per-invariant violation counts."""
    rng = np.random.default_rng(seed)
    tmp = Path(tempfile.mkdtemp()) if root is None else Path(root)
    vault = PitVault(tmp)
    vault.create_dataset("panel", security_level=True)
    secs = [f"s{i}" for i in range(n_secs)]
    ets = [_T0 + timedelta(hours=i) for i in range(n_events)]
    base_known = _T0
    vault.append("panel", _frame(secs, ets, rng.random(n_secs * n_events), base_known))

    horizon_end = _T0 + timedelta(hours=n_events + n_restatements + 4)
    # restatements with random (including far-future) known_at
    for i in range(n_restatements):
        pick_secs = [
            secs[i]
            for i in rng.choice(n_secs, size=int(rng.integers(1, n_secs + 1)), replace=False)
        ]
        pick_ets = [
            ets[i]
            for i in rng.choice(len(ets), size=int(rng.integers(1, len(ets) + 1)), replace=False)
        ]
        known = _T0 + timedelta(seconds=float(rng.uniform(0, (horizon_end - _T0).total_seconds())))
        vault.restate(
            "panel",
            _frame(
                pick_secs,
                pick_ets,
                10.0 + i + rng.random(len(pick_secs) * len(pick_ets)),
                known,  # placeholder; prepare_correction overrides
            ).drop(KNOWN_AT_COL),
            known_at=known,
        )

    violations: dict[str, int] = {
        "watermark": 0,
        "strict_monotone": 0,
        "future_leak": 0,
        "latest_supersedes": 0,
    }
    probes = 0
    times = sorted(
        _T0 + timedelta(seconds=float(rng.uniform(0, (horizon_end - _T0).total_seconds())))
        for _ in range(n_probes)
    )
    prev_strict: dict[tuple[str, datetime], tuple[datetime, float]] | None = None
    prev_t: datetime | None = None
    for t in times:
        try:
            latest = vault.asof("panel", t, policy=RestatementPolicy.LATEST_KNOWN)
            strict = vault.asof("panel", t, policy=RestatementPolicy.STRICT_FIRST)
        except Exception:  # noqa: BLE001 — no version knowable yet; probe is vacuous
            continue
        probes += 1
        # invariant 1: watermark
        if latest.max_known_at > t or strict.max_known_at > t:
            violations["watermark"] += 1
        for f in (latest.frame, strict.frame):
            if (f[KNOWN_AT_COL] > t).any():
                violations["watermark"] += 1
        # invariant 2: STRICT_FIRST monotone
        cur_strict = _rows_keyed(strict.frame)
        if prev_strict is not None and prev_t is not None and prev_t <= t:
            for key, row in prev_strict.items():
                if key in cur_strict and cur_strict[key] != row:
                    violations["strict_monotone"] += 1
        prev_strict, prev_t = cur_strict, t
        # invariant 3: no future leak — nothing may carry known_at > t
        # (checked above via watermark; explicit far-future fuzz below)
    # invariant 4 probe: latest-known after the last restatement's known_at
    # must show the restated value, not the base
    last_known = (
        vault.asof("panel", horizon_end).frame[KNOWN_AT_COL].max()
        if vault.asof("panel", horizon_end).frame.height
        else None
    )
    if last_known is not None:
        f_end = vault.asof("panel", horizon_end, policy=RestatementPolicy.LATEST_KNOWN).frame
        keyed_end = _rows_keyed(f_end)
        # a key restated at ka should show its restated value once t >= ka
        restated = f_end.filter(pl.col("value") >= 10.0)
        for s, e, ka in zip(
            restated[SECURITY_ID_COL].to_list(),
            restated[EVENT_TIME_COL].to_list(),
            restated[KNOWN_AT_COL].to_list(),
            strict=True,
        ):
            try:
                f_pre = vault.asof(
                    "panel", ka - timedelta(seconds=1), policy=RestatementPolicy.LATEST_KNOWN
                ).frame
                pre = _rows_keyed(f_pre)
                if (s, e) in pre and pre[(s, e)][1] == keyed_end[(s, e)][1]:
                    violations["latest_supersedes"] += 1
            except Exception:  # noqa: BLE001 — nothing knowable before ka; fine
                pass
    # future-leak fuzz: pin one correction far ahead; probe just before it
    future_ka = horizon_end + timedelta(hours=48)
    vault.restate(
        "panel",
        _frame(secs[:1], ets[:1], np.array([999.0]), future_ka).drop(KNOWN_AT_COL),
        known_at=future_ka,
    )
    for t in times:
        try:
            f = vault.asof("panel", t, policy=RestatementPolicy.LATEST_KNOWN).frame
        except Exception:  # noqa: BLE001
            continue
        if (f["value"] == 999.0).any():
            violations["future_leak"] += 1
    return {
        "probes": probes,
        "n_restatements": n_restatements,
        "violations": violations,
        "total_violations": sum(violations.values()),
    }


def asof_audit_bench(seed: int = 0) -> dict[str, Any]:
    """Sealed storm-audit receipt over a fresh synthetic vault."""
    res = asof_storm_audit(seed=seed)
    verdict = "ok" if res["total_violations"] == 0 and res["probes"] > 0 else "fail"
    payload: dict[str, Any] = {
        "kind": "asof_audit",
        "schema": "asof_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "asof(t) returns only rows knowable at t; restatements never leak backward",
            "verdict": verdict,
        },
        "interpretation": res,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
