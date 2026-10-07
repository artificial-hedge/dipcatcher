"""Verdict runner — fleet tournament → per-origin streams → honest verdict.

``run_distribution_fleet`` reports aggregates (mean pinball per tau,
coverage, PIT-KS). The composite verdict in ``honest_verdict`` needs the
full per-origin loss stream — the object the tournament is actually
deciding on. ``verdict_streams`` replays the same tournament (same
shard seeds, same fit/predict contract, same lagged-head convention)
but keeps every per-origin proper-score and PIT value; ``run_verdict``
then feeds those streams to ``honest_verdict`` and wraps the result in
a ``honest_verdict_run.v1`` receipt.

The stream pairing is what makes the composite lanes meaningful: all
heads see the same shard order and the same eval slice, so the t-th
entry of every stream is the same forecast origin — the pairing the
winner's-curse and drift lanes assume.
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
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

VERDICT_RUN_SCHEMA = "honest_verdict_run.v1"


def predict_eval_matrix(
    shard: Any, model: Any, n_train: int, n_eval: int, tau_arr: np.ndarray
) -> np.ndarray:
    """Fit-time → eval-quantile matrix, enforcing the fleet predict contract.

    Returns the (n_eval, n_taus) quantile matrix; raises on shape,
    finiteness, or crossing violations — same contract as ``fleet_eval``'s
    ``_score_row``, including the ``fleet_lagged_predict`` convention.
    """
    if getattr(model, "fleet_lagged_predict", False):
        lag_x = shard.y[n_train - 1 : n_train + n_eval - 1].reshape(-1, 1)
        q = np.asarray(model.predict(lag_x), dtype=float)
    else:
        q = np.asarray(model.predict(shard.x[n_train : n_train + n_eval]), dtype=float)
    if q.ndim != 2 or q.shape[0] != n_eval or q.shape[1] != tau_arr.size:
        raise ValueError(f"predict returned shape {q.shape}; expected ({n_eval}, {tau_arr.size})")
    if not np.isfinite(q).all():
        raise ValueError("predict returned non-finite quantiles")
    if np.any(np.diff(q, axis=1) < 0.0):
        raise ValueError("predict returned crossing quantiles")
    return q


def verdict_streams(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pl.DataFrame]:
    """Replay the tournament; return (scores, pits, status_frame).

    ``scores[head]`` is the per-origin mean-pinball stream (shard-major
    concat, identical ordering for every head); ``pits[head]`` the PIT
    stream. Heads that fail to fit/predict are excluded from both maps
    and recorded in the status frame as ``error`` — a failed head can't
    launder a verdict by going missing.
    """
    tau_arr = np.asarray(taus, dtype=float)
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = resolve_shard_generators(None)
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("no shard generators resolved")
    if not factories:
        raise ValueError("factories is empty")

    losses: dict[str, list[np.ndarray]] = {name: [] for name in factories}
    pits: dict[str, list[np.ndarray]] = {name: [] for name in factories}
    status_rows: list[dict[str, Any]] = []
    n_shard = n_train + n_eval

    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard = generator(n_shard, int(seed) + shard_index)
        y_eval = np.asarray(shard.y[n_train : n_train + n_eval], dtype=float)
        for name in sorted(factories):
            status = "ok"
            err: str | None = None
            try:
                model = factories[name]()
                model.fit(shard.x[:n_train], shard.y[:n_train])
                q = predict_eval_matrix(shard, model, n_train, n_eval, tau_arr)
            except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError) as exc:
                # Narrowed from `except Exception` (quality ratchet): head fit/predict
                # faults are solver/numeric; exotic errors propagate. Recorded in status rows.
                status = "error"
                err = str(exc)
                q = None
            if q is not None:
                # per-origin mean pinball across taus; per-origin PIT
                loss_rows = np.stack(
                    [pinball_loss(y_eval, q[:, j], float(t)) for j, t in enumerate(tau_arr)],
                    axis=1,
                )
                losses[name].append(np.asarray(loss_rows).mean(axis=1))
                pits[name].append(np.asarray(pit_values(y_eval, q, tau_arr), dtype=float))
            status_rows.append(
                {
                    "shard": shard_name,
                    "head": name,
                    "status": status,
                    "error": err,
                    "n_eval": n_eval,
                }
            )

    scores: dict[str, np.ndarray] = {}
    pit_map: dict[str, np.ndarray] = {}
    for name, parts in losses.items():
        if parts:
            scores[name] = np.concatenate(parts)
            pit_map[name] = np.concatenate(pits[name])
    return scores, pit_map, pl.DataFrame(status_rows)


def run_verdict(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    alpha: float = 0.05,
    n_boot: int = 2000,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Streams → honest_verdict → sealed-shape receipt dict.

    Heads that failed are excluded from the verdict but recorded — the
    receipt's ``excluded_heads`` makes the exclusion visible, so a broken
    challenger cannot silently drop out of the claim.
    """
    from quant_fund.research.honest_verdict import honest_verdict

    scores, pits, status = verdict_streams(
        factories, shards, n_train=n_train, n_eval=n_eval, seed=seed, taus=taus
    )
    if not scores:
        raise ValueError("no head produced a stream — verdict impossible")
    excluded = sorted(set(map(str, factories)) - set(scores))

    # stamp the receipt with the shards' own data_label — generators are
    # pure, so re-instantiating each resolved shard is the same draw
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = resolve_shard_generators(None)
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    n_shard = n_train + n_eval
    shard_labels = {
        str(name): str(gen(n_shard, int(seed) + i).config.get("data_label") or "UNKNOWN")
        for i, (name, gen) in enumerate(resolved.items())
    }
    distinct = set(shard_labels.values())
    if len(distinct) > 1:
        raise ValueError(
            "shards carry mixed data_label values "
            f"{sorted(distinct)}; run mixed corpora as separate receipts"
        )
    data_label = distinct.pop() if distinct else "UNKNOWN"

    verdict = honest_verdict(
        scores, pits=pits, alpha=alpha, seed=seed, n_boot=n_boot, data_label=data_label
    )
    # keep the verdict's own schema/kind (contract-checked); the run context
    # rides under `run` so verify-receipt dispatch is unaffected
    verdict["run"] = {
        "schema": VERDICT_RUN_SCHEMA,
        "status_sha256": hash_bytes(status.write_csv().encode("utf-8")),
        "params": {
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": seed,
            "alpha": alpha,
            "n_boot": n_boot,
            "heads": sorted(map(str, scores)),
            "shards": resolved_names(shards),
            "data_labels": shard_labels,
        },
        "excluded_heads": excluded,
    }
    verdict["meta"] = {"code_revision": git_revision()}
    return verdict, status


_ENVELOPE_VERDICTS = {
    # honest_verdict emits "confirmed" (never "supported") — an unmapped
    # verdict silently degrades to "blocked" below, which would stamp a
    # confirmed claim as blocked on the envelope.
    "confirmed": "pass",
    "supported": "pass",
    "supported_with_caveats": "pass",
    "not_supported": "fail",
    "inconclusive": "blocked",
}


def write_verdict_receipt(
    verdict: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a verdict report and write ``receipts/honest_verdict_<hash>.json``.

    Same convention as ``write_fleet_receipt``: filename hash = canonical
    ``receipt_sha256``, atomic publish, existing identical file is a no-op
    and a diverging one refuses to overwrite. ``receipt_version=2`` wraps
    the same body in the unified ``receipt.v2`` envelope instead.
    """
    if receipt_version == 1:
        payload = seal_receipt({key: value for key, value in verdict.items() if key != "meta"})
    elif receipt_version == 2:
        run_obj = verdict.get("run")
        run: Mapping[str, Any] = run_obj if isinstance(run_obj, Mapping) else {}
        params_obj = run.get("params")
        params = params_obj if isinstance(params_obj, Mapping) else {}
        dataset = {"status_sha256": run["status_sha256"]} if "status_sha256" in run else None
        payload = seal_receipt(
            wrap_receipt_v2(
                verdict,
                code_files=(Path(__file__),),
                verdict=_ENVELOPE_VERDICTS.get(str(verdict.get("verdict")), "blocked"),
                dataset=dataset,
                params=params,
            )
        )
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"honest_verdict_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def resolved_names(
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None,
) -> list[str]:
    if shards is None:
        return sorted(resolve_shard_generators(None))
    if isinstance(shards, Mapping):
        return sorted(map(str, shards))
    return sorted(str(s).strip() for s in shards if str(s).strip())


__all__ = [
    "VERDICT_RUN_SCHEMA",
    "resolved_names",
    "run_verdict",
    "verdict_streams",
    "write_verdict_receipt",
]
