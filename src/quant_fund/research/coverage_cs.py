"""Time-uniform confidence sequence on interval coverage.

``coverage_watch`` answers "is the band's breach rate still nominal?" —
an e-test at the nominal point. This module answers the stronger
question an evidence ledger actually needs: "at *every* time t, which
breach rates are consistent with what we've seen?" — a **confidence
sequence** on the true breach probability.

Construction: for each candidate breach rate `p` on a fixed grid, keep
the Bernoulli likelihood-ratio e-process of ``coverage_watch`` (mixture
over multiplicative alternatives). `e_t(p)` is anytime-valid under the
point null `H0: rate = p`, so inverting it gives a time-uniform CS at
level alpha:

``CS_t = { p in grid : e_t(p) < 1/alpha }``   and   P(forall t: rate ∈ CS_t) ≥ 1 − α

This is the Waudby-Smith & Ramdas inversion; it is the honest upgrade of
the static "coverage = 89.7%" number — the interval carries its own
sequential validity, narrows as evidence accumulates, and never widens
again on a compatible stream.

``audit_coverage_cs`` runs the construction per head per level over the
fleet shards and seals a ``coverage_cs.v1`` receipt whose headline fields
are the final CS and whether the nominal rate `1 - level` is inside it —
the coverage claim's direct, bounded answer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    HeadFactory,
    ShardGenerator,
    _central_interval_index,
    resolve_shard_generators,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

COVERAGE_CS_SCHEMA = "coverage_cs.v1"
DEFAULT_P_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.01, 0.991, 0.01), 2))
DEFAULT_ALT_GRID: tuple[float, ...] = (0.5, 0.7, 1.3, 2.0)


def _strict_breach(flag: object) -> int:
    if isinstance(flag, (bool, np.bool_)):
        return int(flag)
    if isinstance(flag, (int, np.integer)) and flag in (0, 1):
        return int(flag)
    raise ValueError(f"breach must be bool or a 0/1 int, got {flag!r}")


@dataclass
class CoverageCS:
    """Bernoulli CS on a breach-rate stream, inverted LR mixture.

    ``p0_grid`` are the tested rates; ``alt_grid`` multiplicative
    alternatives per tested rate. All bookkeeping is in log space.
    """

    alpha: float = 0.05
    p0_grid: Sequence[float] = DEFAULT_P_GRID
    alt_grid: Sequence[float] = DEFAULT_ALT_GRID
    n_eval: int = 0
    n_breach: int = 0
    _log_w: np.ndarray | None = None  # [n_p0, n_alt] log-wealth per (p0, alt)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if not self.p0_grid:
            raise ValueError("p0_grid is empty")
        if not self.alt_grid or any(g <= 0.0 for g in self.alt_grid):
            raise ValueError("alt_grid must be non-empty positive ratios")
        p0 = np.asarray(self.p0_grid, dtype=float)
        if np.any((p0 <= 0.0) | (p0 >= 1.0)):
            raise ValueError("p0_grid entries must lie in (0,1)")
        self._log_w = np.zeros((p0.size, len(self.alt_grid)), dtype=float)

    def update(self, breach: bool) -> tuple[float, float]:
        """Fold one breach indicator; return the current (lo, hi) CS.

        Strict flag: ``None`` is a missing observation, not a non-breach —
        silently folding it would deflate the measured breach rate, so
        anything outside {False, True, 0, 1} raises.
        """
        if not (self._log_w is not None):
            raise ValueError("self._log_w is not None")
        b = float(_strict_breach(breach))
        p0 = np.asarray(self.p0_grid, dtype=float)
        for j, g in enumerate(self.alt_grid):
            p1 = np.minimum(0.999999, g * p0)
            # log LR = b*log(p1/p0) + (1-b)*log((1-p1)/(1-p0))
            self._log_w[:, j] += b * (np.log(p1) - np.log(p0)) + (1.0 - b) * (
                np.log1p(-p1) - np.log1p(-p0)
            )
        self.n_eval += 1
        self.n_breach += int(_strict_breach(breach))
        return self.interval()

    def _log_e(self) -> np.ndarray:
        """Log of the uniform mixture e-value per tested p0."""
        if not (self._log_w is not None):
            raise ValueError("self._log_w is not None")
        lw = self._log_w
        m = lw.max(axis=1)
        out: np.ndarray = m + np.log(np.exp(lw - m[:, None]).mean(axis=1))
        return out

    def interval(self) -> tuple[float, float]:
        """(lo, hi) bounds of the time-uniform CS on the breach rate."""
        p0 = np.asarray(self.p0_grid, dtype=float)
        inside = self._log_e() < np.log(1.0 / self.alpha)
        if not inside.any():
            # every tested rate is rejected — the honest empty CS
            return (float("nan"), float("nan"))
        idx = np.flatnonzero(inside)
        return (float(p0[idx[0]]), float(p0[idx[-1]]))

    def evalue_at(self, p: float) -> float:
        """E-value for the point null `rate == p` (0 when p is off-grid)."""
        p0 = np.asarray(self.p0_grid, dtype=float)
        i = int(np.argmin(np.abs(p0 - p)))
        if abs(float(p0[i]) - p) > 1e-9:
            return float("nan")
        return float(np.exp(min(700.0, self._log_e()[i])))


def audit_coverage_cs(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    levels: Sequence[float] = (0.8, 0.9),
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.05,
    p0_grid: Sequence[float] = DEFAULT_P_GRID,
    alt_grid: Sequence[float] = DEFAULT_ALT_GRID,
    data_label: str | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Per (head, level): stream breaches into a CoverageCS.

    Reported per row: the final CS bounds, whether the nominal rate
    `1 - level` sits inside the CS (the coverage claim's bounded answer),
    the first index at which nominal left the CS (coverage-cs origin), and
    the empirical breach rate for context. A malformed (non-finite or
    crossed) interval row is inconclusive — never a breach.
    ``data_label`` stamps the receipt's provenance; when None it is
    derived from the shard configs (all-SYNTHETIC → SYNTHETIC, mixed →
    MIXED, unlabeled → UNKNOWN).
    """
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
    tau_arr = np.asarray(taus, dtype=float)
    level_index: dict[float, tuple[int, int] | None] = {
        lv: _central_interval_index(tau_arr, lv) for lv in levels
    }
    if all(idx is None for idx in level_index.values()):
        raise ValueError("no requested level is expressible in the tau grid")

    rows: list[dict] = []
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
            for level, idx in level_index.items():
                if idx is None or q is None or q.ndim != 2 or q.shape[1] != tau_arr.shape[0]:
                    rows.append(
                        {
                            "shard": shard_name,
                            "head": name,
                            "level": level,
                            "status": "level_unsupported" if q is not None else "error",
                            "error": err,
                            "n_eval": 0,
                            "breach_rate": float("nan"),
                            "cs_low": float("nan"),
                            "cs_high": float("nan"),
                            "nominal_inside": False,
                            "nominal_exit_origin": float("nan"),
                        }
                    )
                    continue
                lo_i, hi_i = idx
                cs = CoverageCS(alpha=alpha, p0_grid=p0_grid, alt_grid=alt_grid)
                nominal = 1.0 - level
                exit_origin = float("nan")
                lo = hi = float("nan")
                for i in range(min(n_eval, q.shape[0], y_eval.shape[0])):
                    lo_q, hi_q = float(q[i, lo_i]), float(q[i, hi_i])
                    if not (np.isfinite(lo_q) and np.isfinite(hi_q)) or hi_q < lo_q:
                        continue
                    lo, hi = cs.update(bool(y_eval[i] < lo_q or y_eval[i] > hi_q))
                    inside = np.isfinite(lo) and np.isfinite(hi) and lo <= nominal <= hi
                    if not inside and not np.isfinite(exit_origin):
                        exit_origin = float(i + 1)
                nominal_inside = np.isfinite(lo) and np.isfinite(hi) and lo <= nominal <= hi
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "level": level,
                        "status": "ok" if cs.n_eval else "inconclusive",
                        "error": err,
                        "n_eval": cs.n_eval,
                        "breach_rate": cs.n_eval and cs.n_breach / cs.n_eval or float("nan"),
                        "cs_low": lo,
                        "cs_high": hi,
                        "nominal_inside": nominal_inside,
                        "nominal_exit_origin": exit_origin,
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
        "schema": COVERAGE_CS_SCHEMA,
        "kind": "coverage_cs",
        "data_label": data_label,
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "levels": list(levels),
            "alpha": alpha,
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": seed,
        },
        "evidence": [
            "time_uniform_confidence_sequence",
            "bernoulli_lr_inversion",
            "waudby_smith_ramdas",
            "two_sided_interval_not_point_test",
        ],
        "claims": [
            {
                "text": (
                    "time-uniform (lo, hi) on each head's true breach rate; "
                    "nominal_inside is the coverage claim's bounded answer"
                ),
                "kind": "theoretical",
            }
        ],
    }
    return frame, receipt


__all__ = [
    "COVERAGE_CS_SCHEMA",
    "CoverageCS",
    "audit_coverage_cs",
]
