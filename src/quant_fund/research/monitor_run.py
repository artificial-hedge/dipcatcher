"""Fleet monitor runner — all anytime-valid monitors over one tournament.

``verdict_run`` shows *which head wins*; ``monitor_fleet`` answers a
different question — *is anything wrong with the heads*, streaming every
per-origin observation through the anytime-valid monitor family:

- ``CoverageEProcess`` — interval breach-rate vs nominal (coverage_watch)
- ``TailDepthEProcess`` — nested-quantile deep-share (tail_watch)
- ``CalibrationEProcess`` — PIT uniformity (calibration_eprocess)
- ``ConformalMartingale`` — PIT exchangeability (conformal_monitor)
- ``EProcessDriftAlarm`` — head-vs-fleet-median loss drift (drift_alarm)

Each monitor is lazy-imported and probed per (shard, head) cell; a lane
whose module is not merged yet contributes ``None`` columns (recorded as
``lanes_available[...] = false`` on the receipt) instead of being
silently skipped — the receipt records which lanes ran.

Drift pairing: every head's loss diff is taken against the fleet median
at the same origin (computed across the cell's heads), so a head that
degrades relative to its peers alarms even when absolute loss rises
market-wide.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.metrics.scoring import pinball_loss, pit_values
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    HeadFactory,
    ShardGenerator,
    _atomic_write_text,
)
from quant_fund.research.verdict_run import predict_eval_matrix
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

MONITOR_RUN_SCHEMA = "monitor_run.v1"

_MONITOR_LANES = ("coverage", "tail", "calibration", "conformal", "drift")


def _lazy(name: str) -> Any:
    """Import a monitor lane module; ``None`` when the lane is not merged."""
    try:
        import importlib

        return importlib.import_module(f"quant_fund.research.{name}")
    except ImportError:
        return None


def _central_pair(tau_arr: np.ndarray, level: float) -> tuple[int, int] | None:
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    i = np.flatnonzero(np.isclose(tau_arr, lo, atol=1e-9))
    j = np.flatnonzero(np.isclose(tau_arr, hi, atol=1e-9))
    if i.size == 0 or j.size == 0:
        return None
    return int(i[0]), int(j[0])


def monitor_fleet(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.05,
    level: float = 0.9,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run all five monitor lanes over every (shard, head) cell.

    Returns (row frame, ``monitor_run.v1`` receipt). Each cell row carries
    one column per lane's alarm flag; a lane that cannot be imported shows
    ``None`` rather than an error or a fabricated pass, and the receipt's
    ``lanes_available`` map records the miss.
    """
    from quant_fund.research.verdict_run import resolved_names

    tau_arr = np.asarray(taus, dtype=float)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0,1), got {level}")
    if tau_arr.ndim != 1 or tau_arr.size < 2 or np.any(np.diff(tau_arr) <= 0):
        raise ValueError("taus must be a strictly increasing vector of length >= 2")
    if not factories:
        raise ValueError("factories is empty")

    coverage_mod = _lazy("coverage_watch")
    tail_mod = _lazy("tail_watch")
    calib_mod = _lazy("calibration_eprocess")
    conformal_mod = _lazy("conformal_monitor")
    drift_mod = _lazy("drift_alarm")
    emerge_mod = _lazy("emerge")
    lanes_available = {
        "coverage": coverage_mod is not None,
        "tail": tail_mod is not None,
        "calibration": calib_mod is not None,
        "conformal": conformal_mod is not None,
        "drift": drift_mod is not None,
        "emerge": emerge_mod is not None,
    }

    cov_pair = _central_pair(tau_arr, level)
    # tail cell: the two deepest adjacent taus
    tail_lo, tail_hi = float(tau_arr[0]), float(tau_arr[1])
    tail_p0 = tail_lo / tail_hi

    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        from quant_fund.research.fleet_eval import resolve_shard_generators

        resolved = resolve_shard_generators(None)
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        from quant_fund.research.fleet_eval import resolve_shard_generators

        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("no shard generators resolved")

    n_shard = n_train + n_eval
    rows: list[dict[str, Any]] = []
    shard_names: list[str] = []
    shard_labels: dict[str, str] = {}
    shard_digests: dict[str, dict[str, str]] = {}

    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard = generator(n_shard, int(seed) + shard_index)
        shard_names.append(shard_name)
        shard_digests[shard_name] = {
            "x_sha256": hash_bytes(np.asarray(shard.x, dtype=float).tobytes()),
            "y_sha256": hash_bytes(np.asarray(shard.y, dtype=float).tobytes()),
        }
        label = str(shard.config.get("data_label") or "").strip()
        if not label:
            label = "UNKNOWN"
        shard_labels[shard_name] = label
        y_eval = np.asarray(shard.y[n_train : n_train + n_eval], dtype=float)

        # pass 1: collect per-origin streams per head
        cell: dict[str, dict[str, Any]] = {}
        for name in sorted(factories):
            try:
                model = factories[name]()
                model.fit(shard.x[:n_train], shard.y[:n_train])
                q = predict_eval_matrix(shard, model, n_train, n_eval, tau_arr)
                loss_rows = np.stack(
                    [pinball_loss(y_eval, q[:, j], float(t)) for j, t in enumerate(tau_arr)],
                    axis=1,
                )
                cell[name] = {
                    "loss": np.asarray(loss_rows).mean(axis=1),
                    "pits": np.asarray(pit_values(y_eval, q, tau_arr), dtype=float),
                    "q": q,
                    "status": "ok",
                    "error": None,
                }
            except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError) as exc:
                # Narrowed from `except Exception` (quality ratchet): head fit/predict
                # faults are solver/numeric; exotic errors propagate. Recorded per cell.
                cell[name] = {
                    "status": "error",
                    "error": str(exc),
                }

        # fleet median loss per origin — the drift lane's pairing target
        ok_losses = (
            np.stack([v["loss"] for v in cell.values() if v["status"] == "ok"], axis=0)
            if any(v["status"] == "ok" for v in cell.values())
            else None
        )
        fleet_median = np.median(ok_losses, axis=0) if ok_losses is not None else None

        # pass 2: drive the monitors
        for name in sorted(cell):
            v = cell[name]
            row: dict[str, Any] = {
                "shard": shard_name,
                "head": name,
                "status": v["status"],
                "error": v["error"],
                "n_eval": n_eval,
                "coverage_alarmed": None,
                "tail_alarmed": None,
                "calibration_evalue": None,
                "calibration_alarmed": None,
                "conformal_alarmed": None,
                "drift_alarmed": None,
            }
            if v["status"] != "ok":
                rows.append(row)
                continue
            q = v["q"]
            evals: list[float] = []

            if coverage_mod is not None and cov_pair is not None:
                ep = coverage_mod.CoverageEProcess(alpha=alpha, p0=1.0 - level)
                lo_q, hi_q = q[:, cov_pair[0]], q[:, cov_pair[1]]
                for i in range(n_eval):
                    ep.update(bool(y_eval[i] < lo_q[i] or y_eval[i] > hi_q[i]))
                row["coverage_alarmed"] = ep.alarmed
                row["coverage_breach_rate"] = ep.breach_rate
                evals.append(ep.evalue)
            elif coverage_mod is not None:
                row["coverage_alarmed"] = None  # no central pair on this tau grid

            if tail_mod is not None:
                j_lo = int(np.flatnonzero(np.isclose(tau_arr, tail_lo))[0])
                j_hi = int(np.flatnonzero(np.isclose(tau_arr, tail_hi))[0])
                tep = tail_mod.TailDepthEProcess(alpha=alpha, p0=tail_p0)
                for i in range(n_eval):
                    tep.update(bool(y_eval[i] < q[i, j_hi]), bool(y_eval[i] < q[i, j_lo]))
                row["tail_alarmed"] = tep.alarmed
                evals.append(tep.evalue)

            if calib_mod is not None:
                cep = calib_mod.CalibrationEProcess(alpha=alpha)
                for u in v["pits"]:
                    cep.update(float(u))
                row["calibration_evalue"] = cep.wealth
                row["calibration_alarmed"] = cep.alarmed
                evals.append(cep.wealth)

            if conformal_mod is not None:
                cm = conformal_mod.ConformalMartingale(alpha=alpha)
                last_m = 1.0
                for u in v["pits"]:
                    _, last_m = cm.update(float(u))
                row["conformal_alarmed"] = cm.alarmed
                row["conformal_final_m"] = last_m
                evals.append(last_m)

            if drift_mod is not None and fleet_median is not None:
                dep = drift_mod.EProcessDriftAlarm(alpha=alpha)
                diffs = np.asarray(v["loss"]) - fleet_median
                last_e = 1.0
                for d in diffs:
                    step = dep.update(float(d))
                    last_e = float(step.statistic)
                row["drift_alarmed"] = dep.alarmed
                row["drift_alarm_index"] = dep.alarm_index
                row["drift_evalue"] = last_e
                evals.append(last_e)

            if emerge_mod is not None and evals:
                pooled = emerge_mod.emerge_mean(evals)
                # arithmetic mean is a valid e-value under arbitrary
                # dependence between the lanes (Vovk & Wang 2021)
                row["pooled_evalue"] = pooled
                row["pooled_alarmed"] = bool(pooled >= 1.0 / alpha)
                row["n_lanes_evidence"] = len(evals)

            rows.append(row)

    frame = pl.DataFrame(rows)
    distinct_labels = set(shard_labels.values())
    if len(distinct_labels) > 1:
        raise ValueError(
            "shards carry mixed data_label values "
            f"{sorted(distinct_labels)}; run mixed corpora as separate receipts"
        )
    data_label = distinct_labels.pop() if distinct_labels else "UNKNOWN"
    receipt: dict[str, Any] = {
        "schema": MONITOR_RUN_SCHEMA,
        "kind": "monitor_run",
        "level": "research",
        "data_label": data_label,
        "research_only": True,
        "live_pnl_claim": False,
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "dataset_sha256": hash_bytes(canonical_json_bytes({"shards": shard_digests})),
        "meta": {"code_revision": git_revision()},
        "params": {
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": seed,
            "alpha": alpha,
            "level": level,
            "tail_cell": [tail_lo, tail_hi],
            "heads": sorted(map(str, factories)),
            "shards": shard_names if shards is None else resolved_names(shards),
            "data_labels": shard_labels,
        },
        "lanes_available": lanes_available,
        "n_rows": frame.height,
        # Every alarm column must feed the count — a head whose only alarm
        # is calibration must still read as an alarm row (the v2 envelope
        # verdict below maps n_alarm_rows>0 to "fail").
        "n_alarm_rows": int(
            frame.filter(
                (pl.col("coverage_alarmed") == True)  # noqa: E712
                | (pl.col("tail_alarmed") == True)  # noqa: E712
                | (pl.col("calibration_alarmed") == True)  # noqa: E712
                | (pl.col("conformal_alarmed") == True)  # noqa: E712
                | (pl.col("drift_alarmed") == True)  # noqa: E712
            ).height
        ),
        "evidence": [
            "anytime_valid_monitor_lanes",
            "per_shard_head_independence",
            "fleet_median_drift_pairing",
            "lazy_lane_resolution",
            "proper_score_only",
        ],
    }
    return frame, receipt


def write_monitor_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a monitor_run receipt and write ``monitor_run_<hash>.json``.

    Filename digest = canonical ``receipt_sha256``. Atomic, fail-closed on
    a malformed receipt. ``receipt_version=2`` wraps the same body in the
    unified ``receipt.v2`` envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if (
        receipt.get("schema") != MONITOR_RUN_SCHEMA
        or receipt.get("kind") != "monitor_run"
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("inputs_sha256"), str)
        or not isinstance(receipt.get("dataset_sha256"), str)
        or not isinstance(receipt.get("params"), Mapping)
    ):
        raise ValueError("monitor_run receipt violates its contract")
    if receipt_version == 1:
        body = {key: value for key, value in receipt.items() if key != "meta"}
        canonical = json.loads(canonical_json_bytes(body))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="fail" if receipt.get("n_alarm_rows") else "pass",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"monitor_run_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = ["MONITOR_RUN_SCHEMA", "monitor_fleet", "write_monitor_receipt"]
