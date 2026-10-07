"""Fleet race — sequential head elimination over ordered eval chunks.

The batch fleet scores every (head, shard) cell and reports means; this
lane asks the sharper question *during* the eval slice: at which chunk
is each head provably out of contention / provably dominant. Statistical
selection procedure (racing — Birattari et al. F-race lineage); the
anytime-valid claims come from the e-process, the halving backstop is
explicitly heuristic.

Per shard, each head fits on the leading ``n_train`` slice and predicts
the trailing ``n_eval`` slice once; the eval rows are then sliced into
``n_chunks`` consecutive blocks in time order. Per (shard, head) two
e-processes run on chunk mean-pinball streams against the current
cumulative-loss incumbent:

- ``promote`` process (challenger=head): crosses ``1/alpha`` → head
  provably better than the incumbent; locks as shard winner.
- ``demote`` process (challenger=incumbent, incumbent=head): crosses
  ``1/alpha`` → head provably *worse*; eliminated with anytime-valid
  evidence at that chunk (an e-process can dip and recover, so small
  ``E`` alone never eliminates — only a crossing of the reverse test
  does).

Survivors accumulate the next chunk; eliminated heads freeze (their
verdict cannot be rewritten by later data — prefix invariance). If no
head promotes, the shard winner is the survivor with the lowest final
mean pinball, labeled ``"final_mean"`` — an honest non-sequential
verdict, distinguished from ``"anytime"`` promotions in the receipt.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.research.evalues import LossEProcess
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    SHARD_GENERATORS,
    HeadFactory,
    ShardGenerator,
    SyntheticShard,
    _atomic_write_text,
    resolve_shard_generators,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

FLEET_RACE_SCHEMA = "fleet_race.v1"

_MIN_CHUNKS = 4


@dataclass
class _Lane:
    """Per-(shard, head) race state."""

    name: str
    chunk_losses: list[float]
    promote: LossEProcess
    demote: LossEProcess
    eliminated_at: int | None = None
    promoted_at: int | None = None
    error: str | None = None


def _chunk_pinball(y: Array, q: Array, taus: Array) -> Array:
    """Mean pinball per eval row, then per chunk — (n_chunks,) stream."""
    from quant_fund.metrics.scoring import pinball_loss

    per_row = np.mean(
        np.stack(
            [pinball_loss(y, q[:, j], float(t)) for j, t in enumerate(taus)],
            axis=0,
        ),
        axis=0,
    )
    return per_row


def _predict_head(
    factory: HeadFactory, x_tr: Array, y_tr: Array, x_ev: Array, n_taus: int
) -> Array | str:
    try:
        head = factory()
        head.fit(x_tr, y_tr)
        q = np.asarray(head.predict(x_ev), dtype=np.float64)
    except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError) as exc:
        # Narrowed from `except Exception` (quality ratchet): head fit/predict faults
        # are solver/numeric; exotic errors propagate. Error rows are visible, never silent.
        return f"{type(exc).__name__}: {exc}"
    if (
        q.ndim != 2
        or q.shape[0] != x_ev.shape[0]
        or q.shape[1] != n_taus
        or not np.isfinite(q).all()
        or bool(np.any(np.diff(q, axis=1) < 0.0))
    ):
        return f"predict returned invalid quantile frame {q.shape!r}"
    return q


def fleet_race(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    n_train: int = 256,
    n_eval: int = 128,
    n_chunks: int = 8,
    alpha: float = 0.05,
    taus: Sequence[float] = DEFAULT_TAUS,
    seed: int = 0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run the sequential elimination race. Returns (frame, receipt payload)."""
    if not factories:
        raise ValueError("fleet_race requires a nonempty head mapping")
    if n_train < 16 or n_eval < _MIN_CHUNKS:
        raise ValueError("n_train >= 16 and n_eval >= n_chunks required")
    if n_chunks < _MIN_CHUNKS or n_eval % n_chunks != 0:
        raise ValueError(f"n_chunks >= {_MIN_CHUNKS} and must divide n_eval")
    tau_arr = np.asarray(list(taus), dtype=float)
    if tau_arr.size == 0 or not np.isfinite(tau_arr).all():
        raise ValueError("taus must be finite and nonempty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = SHARD_GENERATORS
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("fleet_race requires at least one shard")

    chunk = n_eval // n_chunks
    rows: list[dict[str, Any]] = []
    shard_meta: dict[str, Any] = {}
    shard_winners: dict[str, str] = {}
    # per-head per-shard stopping-time e-values for cross-shard pooling
    head_evidence: dict[str, list[tuple[str, float, float]]] = {}

    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard_seed = int(seed) + shard_index
        shard = generator(n_train + n_eval, shard_seed)
        if not isinstance(shard, SyntheticShard) or not np.isfinite(shard.y).all():
            raise ValueError(f"shard {shard_name!r} did not produce a finite SyntheticShard")
        if shard.y.shape[0] < n_train + n_eval:
            raise ValueError(
                f"shard {shard_name!r} produced {shard.y.shape[0]} rows; "
                f"the race needs n_train + n_eval = {n_train + n_eval}"
            )
        x = np.asarray(shard.x, dtype=np.float64)
        y = np.asarray(shard.y, dtype=np.float64).reshape(-1)
        x_tr, y_tr = x[:n_train], y[:n_train]
        x_ev, y_ev = x[n_train:], y[n_train:]
        shard_meta[shard_name] = {
            "n": int(y.size),
            "seed": shard_seed,
            "x_sha256": hash_bytes(x.tobytes()),
            "y_sha256": hash_bytes(y.tobytes()),
            "data_label": str(shard.config.get("data_label") or "UNKNOWN"),
        }

        lanes: list[_Lane] = []
        for model_name, factory in factories.items():
            lane = _Lane(
                name=model_name,
                chunk_losses=[],
                promote=LossEProcess(alpha=alpha),
                demote=LossEProcess(alpha=alpha),
            )
            pred = _predict_head(factory, x_tr, y_tr, x_ev, int(tau_arr.size))
            if isinstance(pred, str):
                lane.error = pred
                lanes.append(lane)
                rows.append(_race_row(shard_name, lane, n_chunks, status="error"))
                continue
            per_row = _chunk_pinball(y_ev, pred, tau_arr)
            lane.chunk_losses = [
                float(per_row[k * chunk : (k + 1) * chunk].mean()) for k in range(n_chunks)
            ]
            lanes.append(lane)

        # Incumbent is fixed at chunk 0 so every (challenger, incumbent)
        # stream stays coherent — a re-elected incumbent would restart or
        # contaminate each head's e-process history.
        clean_lanes = [lane for lane in lanes if lane.error is None]
        incumbent_lane = (
            min(clean_lanes, key=lambda lane: lane.chunk_losses[0]) if clean_lanes else None
        )

        if incumbent_lane is not None:
            for k in range(n_chunks):
                b = incumbent_lane.chunk_losses[k]
                for lane in clean_lanes:
                    if lane.name == incumbent_lane.name or lane.eliminated_at is not None:
                        continue
                    c = lane.chunk_losses[k]
                    lane.promote.update(c, b)
                    lane.demote.update(b, c)
                    if lane.promoted_at is None and lane.promote.promotion_origin is not None:
                        lane.promoted_at = lane.promote.promotion_origin
                    if lane.eliminated_at is None and lane.demote.promotion_origin is not None:
                        lane.eliminated_at = lane.demote.promotion_origin

        # A head that promoted and was later eliminated carries
        # contradictory evidence — provably better AND provably worse than
        # the incumbent — so it stays out of the winner pool (its row still
        # reports both stopping times).
        promoted = [
            lane
            for lane in clean_lanes
            if lane.promoted_at is not None and lane.eliminated_at is None
        ]
        contenders = [lane for lane in clean_lanes if lane.eliminated_at is None]
        winner_lane = min(
            promoted if promoted else (contenders or clean_lanes),
            key=lambda lane: sum(lane.chunk_losses),
            default=None,
        )
        verdict = "none"
        if winner_lane is not None:
            verdict = "anytime" if winner_lane.promoted_at is not None else "final_mean"
            shard_winners[shard_name] = winner_lane.name

        for lane in lanes:
            if lane.error is None:
                rows.append(_race_row(shard_name, lane, n_chunks, status="ok"))
            row = rows[-1]
            if winner_lane is not None and lane.name == winner_lane.name:
                row["shard_winner"] = True
                row["verdict"] = verdict

        for lane in clean_lanes:
            ev = head_evidence.setdefault(lane.name, [])
            promote_states = lane.promote.states
            demote_states = lane.demote.states
            ev.append(
                (
                    shard_name,
                    float(promote_states[-1].evalue) if promote_states else 1.0,
                    float(demote_states[-1].evalue) if demote_states else 1.0,
                )
            )

    frame = pl.DataFrame(
        rows,
        schema={
            "shard": pl.String,
            "model": pl.String,
            "status": pl.String,
            "error": pl.String,
            "n_chunks": pl.Int64,
            "mean_pinball": pl.Float64,
            "promoted_at": pl.Int64,
            "eliminated_at": pl.Int64,
            "final_evalue": pl.Float64,
            "demote_evalue": pl.Float64,
            "shard_winner": pl.Boolean,
            "verdict": pl.String,
        },
        orient="row",
    )

    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "shards": {
                    name: {
                        "x_sha256": meta["x_sha256"],
                        "y_sha256": meta["y_sha256"],
                        "seed": meta["seed"],
                        "n": meta["n"],
                    }
                    for name, meta in shard_meta.items()
                },
                "models": sorted(str(k) for k in factories),
                "n_train": n_train,
                "n_eval": n_eval,
                "n_chunks": n_chunks,
                "alpha": alpha,
                "taus": [float(t) for t in tau_arr],
                "seed": int(seed),
            }
        )
    )
    distinct_labels = {m["data_label"] for m in shard_meta.values()}
    if len(distinct_labels) > 1:
        raise ValueError(
            "shards carry mixed data_label values "
            f"{sorted(distinct_labels)}; run mixed corpora as separate receipts"
        )
    receipt: dict[str, Any] = {
        "kind": FLEET_RACE_SCHEMA,
        "schema": "fleet_race.v1",
        "data_label": distinct_labels.pop() if distinct_labels else "UNKNOWN",
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at_commit": git_revision(),
        "inputs_sha256": inputs_sha256,
        "params": {
            "n_train": n_train,
            "n_eval": n_eval,
            "n_chunks": n_chunks,
            "alpha": alpha,
            "taus": [float(t) for t in tau_arr],
            "seed": int(seed),
        },
        "n_shards": len(shard_meta),
        "n_models": len(factories),
        "shard_meta": shard_meta,
        "shard_winners": shard_winners,
        "global_evidence": {
            head: {
                "promote_evalue_product": float(np.prod([p for _, p, _ in entries])),
                "demote_evalue_product": float(np.prod([d for _, _, d in entries])),
                "n_shards": len(entries),
                # product of e-values is an e-value (Shafer merging) —
                # a global claim without cross-shard independence
                "global_promotion": bool(np.prod([p for _, p, _ in entries]) >= 1.0 / alpha),
            }
            for head, entries in head_evidence.items()
        },
        "evidence": [
            "anytime_valid_promotion",
            "anytime_valid_elimination",
            "successive_chunk_stream",
            "proper_scores_only",
        ],
    }
    return frame, receipt


def write_race_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a fleet_race receipt and write ``fleet_race_<hash>.json``.

    Filename digest = ``inputs_sha256`` (v1) or the canonical
    ``receipt_sha256`` (v2). Atomic, fail-closed on a malformed receipt.
    ``receipt_version=2`` wraps the same body in the unified ``receipt.v2``
    envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if (
        receipt.get("kind") != FLEET_RACE_SCHEMA
        or receipt.get("schema") != FLEET_RACE_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("inputs_sha256"), str)
    ):
        raise ValueError("fleet_race receipt violates its contract")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
        name_digest = str(receipt["inputs_sha256"])[:16]
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                # verdict follows the evidence: a race with no shard winner
                # recorded no usable head (every lane errored) — not a pass.
                verdict="pass" if receipt.get("shard_winners") else "fail",
            )
        )
        name_digest = str(payload["receipt_sha256"])[:16]
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"fleet_race_{name_digest}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _race_row(shard: str, lane: _Lane, n_chunks: int, *, status: str) -> dict[str, Any]:
    return {
        "shard": shard,
        "model": lane.name,
        "status": status,
        "error": lane.error,
        "n_chunks": n_chunks,
        "mean_pinball": (float(np.mean(lane.chunk_losses)) if lane.chunk_losses else None),
        "promoted_at": lane.promoted_at,
        "eliminated_at": lane.eliminated_at,
        "final_evalue": (lane.promote.states[-1].evalue if lane.promote.states else None),
        "demote_evalue": (lane.demote.states[-1].evalue if lane.demote.states else None),
        "shard_winner": False,
        "verdict": None,
    }
