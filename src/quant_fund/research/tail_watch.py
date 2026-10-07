"""Anytime-valid tail-depth audit — is the tail as deep as claimed?

``coverage_watch`` audits the *rate* of breaches of the central band; it
cannot see whether outcomes that DO breach land where the reported tail
says they should. A forecaster can be exactly coverage-correct at 90%
while its tail is far too thin — the failure mode that matters most for
risk claims.

``tail_watch`` tests **nested-quantile consistency**, which is exact and
shape-free. Take the two deepest reported taus, e.g. 0.05 and 0.10. Among
outcomes that breach the 0.10 quantile, the fraction that also breach the
0.05 quantile is, under *any* correctly-shaped conditional tail,

    P(y < q_0.05 | y < q_0.10) = 0.05 / 0.10 = 1/2,

because a breach of q(tau_hi) means u ~ Uniform(0, tau_hi) under the
reported distribution, and the deeper breach is u < tau_lo — no tail
model, interpolation, or extrapolation assumption enters. A head whose
tail is THINNER than reported pushes disproportionate mass below the
deep quantile (deep share > tau_lo/tau_hi — understates risk); a head
whose tail is FATTER over-concentrates breaches inside the shallow cell
(deep share < ratio — overstates risk). Both directions are dishonest
and both are detected.

The statistic is the exact Bernoulli likelihood-ratio bet used by
``coverage_watch``, applied to the conditional stream: on each eval row,

    f_i = LR(b_i)^{c_i},   b_i = 1[y_i < q_lo], c_i = 1[y_i < q_hi],

so non-breach rows contribute the factor 1 and each breach row is an
exact e-value (E[LR] = 1 exactly under the conditional Bernoulli null).
Mixing over a small grid of alternative deep-shares keeps validity by
convexity while detecting both failure directions. P(alarm ever |
correct tail) <= alpha.

``audit_tail_depth`` runs one process per (head, shard) over the fleet-
eval shards and seals a ``tail_audit.v1`` receipt. Fail closed
throughout: non-finite or crossed quantile rows are inconclusive and
never compound wealth.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    HeadFactory,
    ShardGenerator,
    _atomic_write_text,
    resolve_shard_generators,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

TAIL_AUDIT_SCHEMA = "tail_audit.v1"
# deep-share alternatives relative to p0 = tau_lo/tau_hi
DEFAULT_ALT_GRID: tuple[float, ...] = (0.5, 0.75, 1.35, 1.9)


def _strict_breach(flag: object) -> int:
    if isinstance(flag, (bool, np.bool_)):
        return int(flag)
    if isinstance(flag, (int, np.integer)) and flag in (0, 1):
        return int(flag)
    raise ValueError(f"breach must be bool or a 0/1 int, got {flag!r}")


@dataclass
class TailDepthEProcess:
    """LR-mixture e-process on the conditional deep-breach stream.

    ``p0`` is the nominal deep-share tau_lo/tau_hi; ``alt_grid`` holds
    multiplicative alternatives on that share, each with a private LR
    wealth. The process wealth is the uniform mixture — an e-value under
    H0 by convexity, exact at every step.
    """

    alpha: float = 0.05
    p0: float = 0.5
    alt_grid: Sequence[float] = DEFAULT_ALT_GRID

    _wealths: list[float] = field(default_factory=list)
    _n: int = 0
    _n_outer: int = 0
    _n_deep: int = 0
    _alarm_at: int | None = None

    def __post_init__(self) -> None:
        if not (np.isfinite(self.alpha) and 0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (np.isfinite(self.p0) and 0.0 < self.p0 < 1.0):
            raise ValueError("p0 must be in (0, 1)")
        if not self.alt_grid:
            raise ValueError("alt_grid must be nonempty")
        self._wealths = [1.0] * len(self.alt_grid)

    @staticmethod
    def _lr(deep: bool, p1: float, p0: float) -> float:
        # exact Bernoulli LR: E[f] = 1 under b ~ Bern(p0)
        num = p1 if deep else 1.0 - p1
        den = p0 if deep else 1.0 - p0
        if den <= 0.0:
            return 1.0
        return min(num / den, 1e300)

    def update(self, outer_breach: bool, deep_breach: bool) -> float:
        """Fold one row: ``outer_breach`` = y < q_hi, ``deep_breach`` =
        y < q_lo. Non-outer rows contribute the factor 1.

        Strict flags: ``None`` is a missing observation — folding it as a
        non-outer row would deflate the deep-share estimate.
        """
        outer = _strict_breach(outer_breach)
        deep = _strict_breach(deep_breach)
        if deep and not outer:
            # deep ⊆ outer by construction (q_lo < q_hi); a deep breach on
            # a non-outer row is impossible input — likely swapped args —
            # and must fail closed rather than drop an event silently.
            raise ValueError("deep_breach cannot be set on a non-outer row")
        self._n += 1
        if not outer:
            return self.evalue
        self._n_outer += 1
        self._n_deep += deep
        eps = 1e-9
        for j, ratio in enumerate(self.alt_grid):
            p1 = min(1.0 - eps, max(eps, self.p0 * float(ratio)))
            self._wealths[j] = min(self._wealths[j] * self._lr(bool(deep), p1, self.p0), 1e300)
        if self._alarm_at is None and self.evalue >= 1.0 / self.alpha:
            self._alarm_at = self._n - 1
        return self.evalue

    @property
    def evalue(self) -> float:
        return float(np.mean(self._wealths))

    @property
    def alarmed(self) -> bool:
        return self._alarm_at is not None

    @property
    def deep_share(self) -> float:
        if self._n_outer == 0:
            return float("nan")
        return self._n_deep / self._n_outer

    @property
    def n_eval(self) -> int:
        return self._n

    @property
    def n_outer(self) -> int:
        return self._n_outer

    @property
    def n_deep(self) -> int:
        return self._n_deep


def audit_tail_depth(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    taus: Sequence[float] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    alpha: float = 0.05,
    alt_grid: Sequence[float] = DEFAULT_ALT_GRID,
    data_label: str | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Audit nested-quantile consistency per (head, shard).

    The cell is the deepest adjacent tau pair on the grid. Fails closed
    if the grid has fewer than two taus. ``data_label`` stamps the
    receipt's provenance; when None it is derived from the shard configs
    (all-SYNTHETIC → SYNTHETIC, mixed → MIXED, unlabeled → UNKNOWN).
    """
    tau_arr = np.asarray(taus if taus is not None else DEFAULT_TAUS, dtype=float)
    if tau_arr.size < 2:
        raise ValueError("taus must contain at least the two deepest levels")
    order = np.argsort(tau_arr)
    i_lo, i_hi = int(order[0]), int(order[1])
    tau_lo, tau_hi = float(tau_arr[i_lo]), float(tau_arr[i_hi])
    p0 = tau_lo / tau_hi

    if n_train <= 0 or n_eval <= 0:
        raise ValueError("n_train and n_eval must be positive")
    if not factories:
        raise ValueError("factories is empty")
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = resolve_shard_generators(None)
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("no shard generators resolved")
    rows: list[dict[str, object]] = []
    shard_labels: set[str] = set()
    n_shard = n_train + n_eval
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard = generator(n_shard, int(seed) + shard_index)
        shard_labels.add(str(shard.config.get("data_label") or "UNKNOWN"))
        y_eval = np.asarray(shard.y[n_train : n_train + n_eval], dtype=float)
        for name in sorted(factories):
            factory = factories[name]
            err: str | None = None
            try:
                model = factory()
                model.fit(shard.x[:n_train], shard.y[:n_train])
                if getattr(model, "fleet_lagged_predict", False):
                    lag_x = shard.y[n_train - 1 : n_train + n_eval - 1].reshape(-1, 1)
                    q = np.asarray(model.predict(lag_x), dtype=float)
                else:
                    q = np.asarray(model.predict(shard.x[n_train : n_train + n_eval]), dtype=float)
            except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError) as exc:
                # Narrowed from `except Exception` (quality ratchet): head fit/predict
                # faults are solver/numeric; exotic errors propagate. Recorded as error rows.
                q = None
                err = str(exc)
            if q is None or q.ndim != 2 or q.shape[1] != tau_arr.shape[0]:
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "cell": f"[{tau_lo}, {tau_hi})",
                        "status": "error",
                        "error": err,
                        "n_eval": 0,
                        "n_outer": 0,
                        "n_deep": 0,
                        "deep_share": float("nan"),
                        "final_evalue": float("nan"),
                        "tail_alarm": False,
                        "alarm_origin": float("nan"),
                    }
                )
                continue
            ep = TailDepthEProcess(alpha=alpha, p0=p0, alt_grid=alt_grid)
            n_malformed = 0
            alarm_origin = float("nan")
            for i in range(min(n_eval, q.shape[0], y_eval.shape[0])):
                lo, hi, y = float(q[i, i_lo]), float(q[i, i_hi]), float(y_eval[i])
                if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(y)) or hi <= lo:
                    n_malformed += 1  # inconclusive: never compounds wealth
                    continue
                ep.update(y < hi, y < lo)
                if ep.alarmed and not np.isfinite(alarm_origin):
                    alarm_origin = float(i + 1)
            rows.append(
                {
                    "shard": shard_name,
                    "head": name,
                    "cell": f"[{tau_lo}, {tau_hi})",
                    "status": "ok" if ep.n_outer else "inconclusive",
                    "error": err,
                    "n_eval": ep.n_eval,
                    "n_outer": ep.n_outer,
                    "n_deep": ep.n_deep,
                    "n_malformed": n_malformed,
                    "deep_share": ep.deep_share,
                    "final_evalue": ep.evalue,
                    "tail_alarm": ep.alarmed,
                    "alarm_origin": alarm_origin,
                }
            )
    if data_label is None:
        if shard_labels == {"SYNTHETIC"}:
            data_label = "SYNTHETIC"
        elif len(shard_labels) > 1:
            data_label = "MIXED"
        else:
            data_label = next(iter(shard_labels), "UNKNOWN")
    frame = pl.DataFrame(rows)
    receipt: dict[str, object] = {
        "schema": TAIL_AUDIT_SCHEMA,
        "kind": "tail_audit",
        "data_label": data_label,
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "cell": [tau_lo, tau_hi],
            "p0_deep_share": p0,
            "alpha": alpha,
            "alt_grid": list(alt_grid),
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": seed,
        },
        "evidence": [
            "nested_quantile_consistency",
            "bernoulli_lr_bet",
            "exact_evalue_under_h0",
            "shape_free_tail_depth",
            "alternative_mixture_two_sided",
        ],
        "claims": [
            {
                "text": (
                    "anytime-valid verdict on tail depth: conditional deep-share "
                    "of breaches matches tau_lo/tau_hi under any correct tail"
                ),
                "kind": "theoretical",
            }
        ],
    }
    return frame, receipt


def write_tail_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a tail_audit receipt and write ``tail_watch_<hash>.json``.

    Filename digest = sha256 of the canonical payload, embedded as
    ``receipt_sha256`` (fleet_eval seal convention). Atomic, fail-closed
    on a malformed receipt. ``receipt_version=2`` wraps the same body in
    the unified ``receipt.v2`` envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.hashing import canonical_json_bytes

    if (
        receipt.get("schema") != TAIL_AUDIT_SCHEMA
        or receipt.get("kind") != "tail_audit"
        or not _is_sha256_str(receipt.get("inputs_sha256"))
        or not isinstance(receipt.get("params"), Mapping)
        or not isinstance(receipt.get("evidence"), list)
        or not isinstance(receipt.get("claims"), list)
    ):
        raise ValueError("tail_audit receipt violates its contract")
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
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"tail_watch_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _is_sha256_str(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


__all__ = [
    "DEFAULT_ALT_GRID",
    "TAIL_AUDIT_SCHEMA",
    "TailDepthEProcess",
    "audit_tail_depth",
    "write_tail_receipt",
]
