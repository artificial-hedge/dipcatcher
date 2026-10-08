"""Anytime-valid model confidence set — sequential elimination of fleet heads.

The batch MCS (Hansen–Lunde–Nason) answers "which heads survive" after a
fixed sample. This lane maintains a survivor set whose coverage — the
probability it still contains a best head — holds *uniformly over time*
under arbitrary stopping, at level ``alpha``.

Construction (e-value elimination; Ville's inequality + union bound):

- For each ordered pair ``(j, h)``, a ``LossEProcess`` tracks evidence
  that ``j`` strictly beats ``h`` (challenger=j, incumbent=h).
- Head ``h`` is eliminated permanently the first origin at which
  ``max_j e_{j -> h} >= (K - 1) / alpha``. Under "h is optimal" every
  ``e_{j -> h}`` is a nonnegative supermartingale, so
  ``P(h ever eliminated | h optimal) <= (K-1) * alpha/(K-1) = alpha``.
  The survivor set therefore contains an optimal head with probability
  ``>= 1 - alpha`` at *every* origin, however long the fleet runs.

Design choices:

- Pairwise processes, not "each head vs the current leader": the leader
  moves between origins, which breaks predictable betting. Pairwise
  e-processes never read the future.
- Elimination is permanent — "ever crossed" is exactly the event the
  union bound controls; a later dip below threshold cannot resurrect.
- The pair process is instantiated with ``alpha/(K-1)`` so its built-in
  promotion flag trips at precisely the elimination threshold.
- Pair bets are *sign* bets (``LossEProcess``): elimination certifies
  median dominance of the loss difference, not mean dominance — the
  distribution-free guarantee that keeps Ville valid under skewed
  loss streams. Mean dominance claims belong to ``loss_cs``.
- Fail closed: non-finite loss, unknown head, or a missing head raises.

Receipt: ``mcs_seq.v1`` — sealed by callers via ``receipt_v2``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.research.evalues import LossEProcess
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

MCS_SEQ_SCHEMA = "mcs_seq.v1"


@dataclass(frozen=True)
class MCSState:
    """Snapshot of the confidence set after one origin."""

    origin: int
    survivors: tuple[str, ...]
    n_eliminated: int
    newly_eliminated: tuple[str, ...]
    champion: str | None


@dataclass
class AnytimeMCS:
    """Sequential MCS over named heads fed per-origin proper losses.

    ``update(losses)`` appends one origin: ``losses`` maps head → loss
    (lower = better) and must contain every registered head.
    """

    heads: tuple[str, ...]
    alpha: float = 0.05
    lam: float = 0.5
    init_scale: float = 1e-3
    _pairs: dict[tuple[str, str], LossEProcess] = field(default_factory=dict, init=False)
    _eliminated: dict[str, int] = field(default_factory=dict, init=False)
    _states: list[MCSState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.heads) < 2:
            raise ValueError("MCS needs at least two heads")
        if len(set(self.heads)) != len(self.heads):
            raise ValueError("head names must be unique")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        pair_alpha = self.alpha / (len(self.heads) - 1)
        self._pairs = {
            (j, h): LossEProcess(lam=self.lam, alpha=pair_alpha, init_scale=self.init_scale)
            for j, h in permutations(self.heads, 2)
        }

    def update(self, losses: dict[str, float]) -> MCSState:
        """Append one origin of losses for all heads; return the new state."""
        if set(losses) != set(self.heads):
            raise ValueError(
                "losses must cover exactly the registered heads "
                f"{sorted(self.heads)}; got {sorted(losses)}"
            )
        for name, v in losses.items():
            if not np.isfinite(float(v)):
                raise ValueError(f"loss for {name!r} is non-finite: {v}")

        # Fix iteration order so every pair sees origins in the same order.
        for j in self.heads:
            for h in self.heads:
                if j == h:
                    continue
                self._pairs[(j, h)].update(losses[j], losses[h])

        newly: list[str] = []
        for h in self.heads:
            if h in self._eliminated:
                continue
            if any(self._pairs[(j, h)].states[-1].promoted for j in self.heads if j != h):
                self._eliminated[h] = len(self._states)
                newly.append(h)

        survivors = tuple(h for h in self.heads if h not in self._eliminated)
        state = MCSState(
            origin=len(self._states),
            survivors=survivors,
            n_eliminated=len(self._eliminated),
            newly_eliminated=tuple(newly),
            champion=survivors[0] if len(survivors) == 1 else None,
        )
        self._states.append(state)
        return state

    @property
    def survivors(self) -> tuple[str, ...]:
        return tuple(h for h in self.heads if h not in self._eliminated)

    @property
    def eliminated(self) -> dict[str, int]:
        return dict(self._eliminated)

    @property
    def states(self) -> list[MCSState]:
        return list(self._states)


def mcs_report(
    loss_streams: dict[str, Any],
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
    init_scale: float = 1e-3,
    data_label: str = "UNKNOWN",
) -> dict[str, Any]:
    """Receipt-shaped sequential-MCS verdict over per-origin loss streams.

    ``loss_streams`` maps head → equal-length per-origin proper losses
    (lower = better). Fails closed on empty/mismatched/non-finite input.
    ``data_label`` stamps the receipt's provenance — callers routing real
    tape through shard configs should pass that declared label; bare
    arrays have no provenance, hence the UNKNOWN default.
    """
    if not isinstance(data_label, str) or not data_label.strip():
        raise ValueError("data_label must be a nonempty string")
    heads = sorted(loss_streams)
    if len(heads) < 2:
        raise ValueError("need at least two heads")
    arrays: dict[str, np.ndarray] = {}
    n_obs: int | None = None
    for h in heads:
        a = np.asarray(loss_streams[h], dtype=np.float64).ravel()
        if a.size == 0 or not np.isfinite(a).all():
            raise ValueError(f"stream {h!r} empty or non-finite")
        if n_obs is None:
            n_obs = int(a.size)
        elif a.size != n_obs:
            raise ValueError("all heads must observe the same number of losses")
        arrays[h] = a
    if not (n_obs is not None):
        raise ValueError("n_obs is not None")

    mcs = AnytimeMCS(tuple(heads), alpha=alpha, lam=lam, init_scale=init_scale)
    for t in range(n_obs):
        mcs.update({h: float(arrays[h][t]) for h in heads})

    final = mcs.states[-1]
    return {
        "kind": "mcs_seq.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": data_label,
        "alpha": alpha,
        "lam": lam,
        "n_heads": len(heads),
        "n_origins": n_obs,
        "survivors": list(final.survivors),
        "eliminated": {h: int(o) for h, o in mcs.eliminated.items()},
        "n_eliminated": len(mcs.eliminated),
        "champion": final.champion,
        "coverage_guarantee": "P(set contains an optimal head at every origin) >= 1 - alpha",
        "evidence": [
            "ville_inequality",
            "pairwise_supermartingales",
            "union_bound_k_minus_1",
            "permanent_elimination",
            "anytime_valid",
        ],
    }


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


def write_mcs_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal an mcs_seq receipt and write ``receipts/mcs_seq_<hash>.json``.

    Filename digest = sha256 of the canonical payload, embedded as
    ``receipt_sha256`` (fleet_eval seal convention). Atomic, fail-closed
    on a malformed receipt. ``receipt_version=2`` wraps the same body in
    the unified ``receipt.v2`` envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if (
        receipt.get("kind") != "mcs_seq.v1"
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("survivors"), list)
        or not isinstance(receipt.get("eliminated"), dict)
    ):
        raise ValueError("mcs_seq receipt violates its contract")
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
                    "n_heads": receipt.get("n_heads"),
                    "n_origins": receipt.get("n_origins"),
                },
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"mcs_seq_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_loss_streams(path: Path | str) -> dict[str, list[float]]:
    """Load per-origin loss streams: JSON ``{head: [losses]}`` or a
    parquet/csv frame whose numeric columns are the heads."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or not data:
            raise ValueError("loss-stream JSON must be a nonempty {head: [losses]} map")
        return {str(h): [float(x) for x in v] for h, v in data.items()}
    if p.suffix in (".parquet", ".csv"):
        import polars as pl

        frame = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
        import polars.selectors as cs

        cols = frame.select(cs.numeric()).columns
        if len(cols) < 2:
            raise ValueError("need >=2 numeric loss-stream columns")
        return {c: frame.get_column(c).to_list() for c in cols}
    raise ValueError(f"unsupported loss-stream format: {p.suffix}")
