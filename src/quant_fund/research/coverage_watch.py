"""Anytime-valid interval-coverage audit — is the 90% band really 90%?

The nominal-coverage claim is the headline honesty claim of a quantile
forecaster ("our 90% band covers 90%"), and it is the one this repo most
often reports as a static number. ``coverage_watch`` turns it into a
**sequential e-process**: each evaluation row contributes a breach
indicator `b_i = 1[y_i ∉ [lo, hi]]`, a Bernoulli stream under the null
`P(breach) = 1 - level`, and wealth accumulates by the exact
**likelihood-ratio bet**

``f_i(b_i) = LR(b_i) = p1^b_i (1-p1)^{1-b_i} / p0^b_i (1-p0)^{1-b_i}``

which is an *exact* e-value under H0 for a Bernoulli stream —
E[f]=1 exactly, no boundedness or plug-in approximation needed. Mixing
over a small grid of alternative breach rates keeps validity (convex
combination of e-values) while detecting *both* undercoverage (band too
narrow — claims more than it delivers) and overcoverage (band too wide —
its width is inflated). Both failures break honesty: a too-wide band is
a vacuous claim dressed as rigor.

This complements ``calibration_eprocess`` (PIT uniformity — *every*
distributional degree of freedom at once) with a directed, interpretable
test on the specific coverage level a report headlined.

``audit_interval_coverage`` runs the audit per head per coverage level
over the fleet-eval shards and seals a ``coverage_audit.v1`` receipt.
Fail closed throughout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

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

COVERAGE_AUDIT_SCHEMA = "coverage_audit.v1"
DEFAULT_ALT_GRID: tuple[float, ...] = (0.5, 0.7, 1.3, 2.0)  # ratios to p0


def _strict_breach(flag: object) -> int:
    if isinstance(flag, (bool, np.bool_)):
        return int(flag)
    if isinstance(flag, (int, np.integer)) and flag in (0, 1):
        return int(flag)
    raise ValueError(f"breach must be bool or a 0/1 int, got {flag!r}")


@dataclass
class CoverageEProcess:
    """LR-mixture e-process on a Bernoulli breach stream.

    ``p0`` is the nominal breach rate (1 - level). ``alt_grid`` holds
    multiplicative alternatives (0.5 = half the nominal breach rate,
    2.0 = double); each gets a private LR wealth; the process wealth is
    the uniform mixture — an e-value under H0 by convexity.
    """

    alpha: float = 0.05
    p0: float = 0.10
    alt_grid: Sequence[float] = DEFAULT_ALT_GRID
    _wealths: list[float] = field(init=False)
    n_breach: int = 0
    n_eval: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if not (0.0 < self.p0 < 1.0):
            raise ValueError(f"p0 must be in (0,1), got {self.p0}")
        if not self.alt_grid or any(g <= 0.0 for g in self.alt_grid):
            raise ValueError("alt_grid must be a non-empty list of positive ratios")
        self._wealths = [1.0] * len(self.alt_grid)

    @staticmethod
    def _lr(b: int, p1: float, p0: float) -> float:
        # Bernoulli likelihood ratio p1^b (1-p1)^(1-b) / p0^b (1-p0)^(1-b)
        num = p1 if b else 1.0 - p1
        den = p0 if b else 1.0 - p0
        return num / den

    def update(self, breach: bool) -> float:
        """Fold one breach indicator. Returns the running e-value.

        Strict flag: ``None`` is a missing observation, not a non-breach —
        silently folding it would deflate the measured breach rate, so
        anything outside {False, True, 0, 1} raises.
        """
        b = _strict_breach(breach)
        n = self.n_eval + 1
        for i, g in enumerate(self.alt_grid):
            p1 = min(0.999999, g * self.p0)  # alt rate, kept < 1
            lr = self._lr(b, p1, self.p0)
            self._wealths[i] = min(self._wealths[i] * lr, 1e300)
        self.n_eval = n
        self.n_breach += b
        return self.evalue

    @property
    def evalue(self) -> float:
        return float(sum(self._wealths) / len(self._wealths))

    @property
    def alarmed(self) -> bool:
        return self.evalue >= 1.0 / self.alpha

    @property
    def breach_rate(self) -> float:
        return self.n_breach / self.n_eval if self.n_eval else float("nan")


def audit_interval_coverage(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    levels: Sequence[float] = (0.8, 0.9),
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.05,
    alt_grid: Sequence[float] = DEFAULT_ALT_GRID,
    data_label: str | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Score each head's central intervals on every shard, sequentially.

    For each (head, level): find the central [lo, hi] quantile pair in the
    head's tau grid, stream the eval breaches through a fresh
    ``CoverageEProcess``, and report the final e-value, the alarm flag,
    the empirical breach rate, and the alarm origin. Fails closed on a
    head whose tau grid can't express the level (returns
    ``level_unsupported`` instead of guessing a wider/narrower pair).
    ``data_label`` stamps the receipt's provenance; when None it is derived
    from the shard configs (all-SYNTHETIC → SYNTHETIC, mixed → MIXED,
    unlabeled → UNKNOWN).
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
            except Exception as exc:
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
                            "n_breach": 0,
                            "breach_rate": float("nan"),
                            "final_evalue": float("nan"),
                            "coverage_alarm": False,
                            "alarm_origin": float("nan"),
                        }
                    )
                    continue
                lo_i, hi_i = idx
                ep = CoverageEProcess(alpha=alpha, p0=1.0 - level, alt_grid=alt_grid)
                alarm_origin = float("nan")
                for i in range(min(n_eval, q.shape[0], y_eval.shape[0])):
                    lo, hi = float(q[i, lo_i]), float(q[i, hi_i])
                    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
                        continue  # malformed row: inconclusive, never a breach
                    ep.update(bool(y_eval[i] < lo or y_eval[i] > hi))
                    if ep.alarmed and not np.isfinite(alarm_origin):
                        alarm_origin = float(i + 1)
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "level": level,
                        "status": "ok" if ep.n_eval else "inconclusive",
                        "error": err,
                        "n_eval": ep.n_eval,
                        "n_breach": ep.n_breach,
                        "breach_rate": ep.breach_rate,
                        "final_evalue": ep.evalue,
                        "coverage_alarm": ep.alarmed,
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
        "schema": COVERAGE_AUDIT_SCHEMA,
        "data_label": data_label,
        "kind": "coverage_audit",
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "levels": list(levels),
            "alpha": alpha,
            "alt_grid": list(alt_grid),
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": seed,
        },
        "evidence": [
            "bernoulli_lr_bet",
            "exact_evalue_under_h0",
            "alternative_mixture_two_sided",
            "nominal_coverage_test",
        ],
        "claims": [
            {
                "text": (
                    "anytime-valid verdict on nominal interval coverage; "
                    "lr mixture over breach-rate alternatives"
                ),
                "kind": "theoretical",
            }
        ],
    }
    return frame, receipt


__all__ = [
    "COVERAGE_AUDIT_SCHEMA",
    "DEFAULT_ALT_GRID",
    "CoverageEProcess",
    "audit_interval_coverage",
]
