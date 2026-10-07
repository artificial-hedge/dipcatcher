"""Hedged-wallet coverage floor — a time-uniform lower bound on coverage.

``coverage_watch`` answers "is the breach rate still nominal?" (a point
e-test) and ``coverage_cs`` inverts a Bernoulli likelihood-ratio mixture into
a two-sided CS on the breach rate. This module answers the operational
question directly: **"what is the worst coverage consistent with the stream
so far, valid at every t?"** — a one-sided lower confidence bound on the
coverage probability (equivalently, an upper bound on the breach rate).

Construction (wallets/switching): for each candidate breach rate ``p0`` on a
fixed grid, maintain ``K`` betting *wallets* with static fractions
``lambda_k(p0) = frac_k / bound(p0)``; wallet ``k`` plays the e-factor
``1 + lambda_k (p0 - b_t)`` for the one-sided composite null
``H0: rate >= p0`` (nonneg because ``lambda_k (1 - p0) <= 1``; sub-valid
because ``E[1 + lambda (p0 - b)] <= 1`` whenever the true rate is at least
``p0``). The wallet *mixture* ``E(p0) = (1/K) sum_k prod_t f_t`` is an
e-process (this is the degenerate case of λ-switching/GRAPA — wallets are
predictable-by-construction). Rejecting ``rate >= p0`` at threshold
``1/alpha`` rules out every ``p0`` too *small* — so the largest grid point
still alive bounds the breach rate from above, and ``1 - hi`` is the
coverage floor. A symmetric wallet bank betting ``rate <= p0`` yields the
breach lower bound; the two-sided interval uses ``alpha/2`` per side
(Bonferroni) so the headline floor can stay at level ``alpha``.

``audit_coverage_floor`` runs the construction per head per level over the
fleet shards and seals a ``coverage_floor.v1`` receipt whose headline field
is the final one-sided floor and whether it dips below the nominal coverage
``level`` (the claim a monitor actually acts on).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.coverage_cs import _strict_breach
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    HeadFactory,
    ShardGenerator,
    _central_interval_index,
    resolve_shard_generators,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

COVERAGE_FLOOR_SCHEMA = "coverage_floor.v1"
DEFAULT_P_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.01, 0.991, 0.01), 2))
#: Static wallet fractions of the per-side admissible lambda bound; the
#: coarse near-saturated grid is what makes the wallet mixture tight on
#: both fast-drifting and near-null streams.
DEFAULT_WALLET_FRACS: tuple[float, ...] = (0.05, 0.2, 0.5, 0.8, 0.95)


@dataclass
class CoverageFloor:
    """Wallet-mixture e-process bound on a breach-rate stream.

    ``p0_grid`` are the tested breach rates; ``wallet_fracs`` are each
    wallet's fraction of the per-side admissible lambda. All bookkeeping is
    in log space. ``floor()`` is the one-sided ``1-alpha`` lower bound on
    coverage; ``interval()`` returns the two-sided ``1-alpha`` coverage CS
    (``alpha/2`` per side).
    """

    alpha: float = 0.05
    p0_grid: Sequence[float] = DEFAULT_P_GRID
    wallet_fracs: Sequence[float] = DEFAULT_WALLET_FRACS
    n_eval: int = 0
    n_breach: int = 0
    _log_w_hi: np.ndarray | None = None  # [n_p0, n_wallets] H0: rate >= p0
    _log_w_lo: np.ndarray | None = None  # [n_p0, n_wallets] H0: rate <= p0

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if not self.p0_grid:
            raise ValueError("p0_grid is empty")
        if not self.wallet_fracs:
            raise ValueError("wallet_fracs is empty")
        fracs = np.asarray(self.wallet_fracs, dtype=float)
        if np.any((fracs <= 0.0) | (fracs >= 1.0)) or not np.all(np.isfinite(fracs)):
            raise ValueError("wallet_fracs must lie in (0,1)")
        p0 = np.asarray(self.p0_grid, dtype=float)
        if np.any((p0 <= 0.0) | (p0 >= 1.0)):
            raise ValueError("p0_grid entries must lie in (0,1)")
        n_p0, n_w = p0.size, fracs.size
        self._log_w_hi = np.zeros((n_p0, n_w), dtype=float)
        self._log_w_lo = np.zeros((n_p0, n_w), dtype=float)

    def update(self, breach: bool) -> float:
        """Fold one breach indicator; return the current coverage floor.

        Strict flag: ``None`` is a missing observation, not a non-breach —
        silently folding it would deflate the measured breach rate, so
        anything outside {False, True, 0, 1} raises.
        """
        if not (self._log_w_hi is not None and self._log_w_lo is not None):
            raise ValueError("self._log_w_hi is not None and self._log_w_lo is not None")
        b = float(_strict_breach(breach))
        p0 = np.asarray(self.p0_grid, dtype=float)
        fracs = np.asarray(self.wallet_fracs, dtype=float)
        # Wallet lambdas: per-side admissible bound scaled by the wallet
        # fraction — lam_hi(p0) <= 1/(1-p0), lam_lo(p0) <= 1/p0.
        lam_hi = (fracs[None, :] * 0.99) / (1.0 - p0[:, None])  # [n_p0, n_w]
        lam_lo = (fracs[None, :] * 0.99) / p0[:, None]
        # H0: rate >= p0 → factor 1 + lam_hi (p0 - b) is an e-variable.
        self._log_w_hi += np.log1p(lam_hi * (p0[:, None] - b))
        # H0: rate <= p0 → factor 1 + lam_lo (b - p0) is an e-variable.
        self._log_w_lo += np.log1p(lam_lo * (b - p0[:, None]))
        self.n_eval += 1
        self.n_breach += int(b)
        return self.floor()

    @staticmethod
    def _log_mix(log_w: np.ndarray) -> np.ndarray:
        """Log of the uniform wallet-mixture e-value per tested p0."""
        m = log_w.max(axis=1)
        out: np.ndarray = m + np.log(np.exp(log_w - m[:, None]).mean(axis=1))
        return out

    def _breach_bounds(self, alpha: float) -> tuple[float, float]:
        """Two-sided CS on the breach rate at level ``alpha``."""
        if not (self._log_w_hi is not None and self._log_w_lo is not None):
            raise ValueError("self._log_w_hi is not None and self._log_w_lo is not None")
        p0 = np.asarray(self.p0_grid, dtype=float)
        thr = math.log(1.0 / alpha)
        alive_hi = self._log_mix(self._log_w_hi) < thr  # rate >= p0 unrejected
        alive_lo = self._log_mix(self._log_w_lo) < thr  # rate <= p0 unrejected
        inside = alive_hi & alive_lo
        if not inside.any():
            return (float("nan"), float("nan"))
        idx = np.flatnonzero(inside)
        return (float(p0[idx[0]]), float(p0[idx[-1]]))

    def floor(self) -> float:
        """One-sided ``1-alpha`` lower bound on the true coverage rate."""
        hi_breach = self._hi_breach_bound(self.alpha)
        return float("nan") if not math.isfinite(hi_breach) else 1.0 - hi_breach

    def _hi_breach_bound(self, alpha: float) -> float:
        """Upper bound on the breach rate at one-sided level ``alpha``."""
        if not (self._log_w_hi is not None):
            raise ValueError("self._log_w_hi is not None")
        p0 = np.asarray(self.p0_grid, dtype=float)
        thr = math.log(1.0 / alpha)
        rejected = self._log_mix(self._log_w_hi) >= thr
        if not rejected.any():
            return float(p0[-1])
        if bool(rejected.all()):
            # every "rate >= p0" rejected — the bound lies below the grid
            # bottom; report the grid min so the answer stays bounded
            return float(p0[0])
        # Nulls are nested: rejecting "rate >= p0" rules out all p0' > p0
        # (the same e-value is valid for every subset), so the smallest
        # rejected grid point is the level-alpha upper bound.
        return float(p0[np.flatnonzero(rejected)[0]])

    def _lo_breach_bound(self, alpha: float) -> float:
        """Lower bound on the breach rate at one-sided level ``alpha``."""
        if not (self._log_w_lo is not None):
            raise ValueError("self._log_w_lo is not None")
        p0 = np.asarray(self.p0_grid, dtype=float)
        thr = math.log(1.0 / alpha)
        rejected = self._log_mix(self._log_w_lo) >= thr
        if not rejected.any():
            return float(p0[0])
        if bool(rejected.all()):
            # every "rate <= p0" rejected — the bound lies above the grid top
            return float(p0[-1])
        return float(p0[np.flatnonzero(rejected)[-1]])

    def interval(self) -> tuple[float, float]:
        """Two-sided ``1-alpha`` CS on the coverage rate (alpha/2 per side)."""
        lo_b, hi_b = self._breach_bounds(self.alpha / 2.0)
        if not (math.isfinite(lo_b) and math.isfinite(hi_b)):
            return (float("nan"), float("nan"))
        return (1.0 - hi_b, 1.0 - lo_b)

    def breach_rate(self) -> float:
        """Empirical breach rate so far (nan before any update)."""
        if self.n_eval == 0:
            return float("nan")
        return self.n_breach / self.n_eval

    def evalue_above(self, p: float) -> float:
        """E-value for the composite null `rate >= p` (hi-side wallets)."""
        if not (self._log_w_hi is not None):
            raise ValueError("self._log_w_hi is not None")
        p0 = np.asarray(self.p0_grid, dtype=float)
        i = int(np.argmin(np.abs(p0 - p)))
        if abs(float(p0[i]) - p) > 1e-9:
            return float("nan")
        return float(np.exp(min(700.0, self._log_mix(self._log_w_hi)[i])))

    def evalue_below(self, p: float) -> float:
        """E-value for the composite null `rate <= p` (lo-side wallets)."""
        if not (self._log_w_lo is not None):
            raise ValueError("self._log_w_lo is not None")
        p0 = np.asarray(self.p0_grid, dtype=float)
        i = int(np.argmin(np.abs(p0 - p)))
        if abs(float(p0[i]) - p) > 1e-9:
            return float("nan")
        return float(np.exp(min(700.0, self._log_mix(self._log_w_lo)[i])))

    def alarm(self, nominal_coverage: float) -> bool:
        """True when 'coverage >= nominal_coverage' is rejected at level
        ``alpha`` — equivalently the composite null `breach rate <=
        1 - nominal` is rejected by the lo-side wallet e-process."""
        if not (0.0 < nominal_coverage < 1.0):
            raise ValueError(f"nominal_coverage must be in (0,1), got {nominal_coverage}")
        e = self.evalue_below(1.0 - nominal_coverage)
        return math.isfinite(e) and e >= 1.0 / self.alpha


def coverage_floor_bench(
    *,
    p_true: Sequence[float] = (0.05, 0.10, 0.20),
    n_stream: int = 512,
    alpha: float = 0.05,
    n_seeds: int = 32,
    seed: int = 0,
    p0_grid: Sequence[float] = DEFAULT_P_GRID,
    wallet_fracs: Sequence[float] = DEFAULT_WALLET_FRACS,
) -> dict[str, Any]:
    """Seeded validity/tightness drill over Bernoulli streams.

    For each true breach rate ``p_true`` and seed, stream ``n_stream``
    iid Bernoulli(p_true) indicators through a fresh ``CoverageFloor`` and
    record (a) whether the floor ever exceeded the true coverage ``1-p_true``
    — a one-sided miscoverage event at level ``alpha`` — and (b) the final
    slack ``floor - (1 - p_true)`` (negative = conservative). SYNTHETIC
    receipt dict; determinism given ``seed``.
    """
    if n_stream <= 0 or n_seeds <= 0:
        raise ValueError("n_stream and n_seeds must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for p in p_true:
        if not (0.0 < p < 1.0):
            raise ValueError(f"p_true entries must be in (0,1), got {p}")
        n_over = 0
        slacks: list[float] = []
        for _s in range(n_seeds):
            draws = rng.random(n_stream) < p
            fl = CoverageFloor(alpha=alpha, p0_grid=p0_grid, wallet_fracs=wallet_fracs)
            over = False
            last_floor = float("nan")
            for b in draws:
                last_floor = fl.update(bool(b))
                if math.isfinite(last_floor) and last_floor > 1.0 - p + 1e-12:
                    over = True
                    break
            n_over += int(over)
            if math.isfinite(last_floor):
                slacks.append(last_floor - (1.0 - p))
        rows.append(
            {
                "p_true": float(p),
                "n_seeds": n_seeds,
                "floor_exceeded_true": n_over,
                "floor_exceed_rate": n_over / n_seeds,
                "median_slack": float(np.median(slacks)) if slacks else float("nan"),
            }
        )
    frame = pl.DataFrame(rows)
    receipt: dict[str, Any] = {
        "schema": COVERAGE_FLOOR_SCHEMA,
        "kind": "coverage_floor",
        "data_label": "SYNTHETIC",
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "p_true": list(p_true),
            "n_stream": n_stream,
            "alpha": alpha,
            "n_seeds": n_seeds,
            "seed": seed,
        },
        "evidence": [
            "wallet_mixture_e_process",
            "time_uniform_one_sided_floor",
            "grapa_lambda_switching_special_case",
        ],
        "claims": [
            {
                "text": (
                    "coverage floor is a time-uniform 1-alpha lower bound on "
                    "the true coverage rate; floor_exceed_rate bounds the "
                    "one-sided miscoverage probability"
                ),
                "kind": "theoretical",
            }
        ],
    }
    out: dict[str, Any] = {"frame": frame, "receipt": receipt}
    return out


def audit_coverage_floor(
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
    wallet_fracs: Sequence[float] = DEFAULT_WALLET_FRACS,
    data_label: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Per (head, level): stream breaches into a CoverageFloor.

    Reported per row: the final one-sided floor on coverage, the two-sided
    CS, whether the floor sits below the nominal ``level`` (the coverage
    claim's floor answer), the first index at which nominal dropped below
    the floor's complement, and the empirical breach rate for context.
    Malformed interval rows are inconclusive — never a breach.
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

    rows: list[dict[str, Any]] = []
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
                            "coverage_floor": float("nan"),
                            "cs_low": float("nan"),
                            "cs_high": float("nan"),
                            "floor_below_nominal": False,
                            "floor_breach_origin": float("nan"),
                        }
                    )
                    continue
                lo_i, hi_i = idx
                cf = CoverageFloor(alpha=alpha, p0_grid=p0_grid, wallet_fracs=wallet_fracs)
                nominal_cov = level
                breach_origin = float("nan")
                floor_v = float("nan")
                lo_c = hi_c = float("nan")
                for i in range(min(n_eval, q.shape[0], y_eval.shape[0])):
                    lo_q, hi_q = float(q[i, lo_i]), float(q[i, hi_i])
                    if not (np.isfinite(lo_q) and np.isfinite(hi_q)) or hi_q < lo_q:
                        continue
                    floor_v = cf.update(bool(y_eval[i] < lo_q or y_eval[i] > hi_q))
                    lo_c, hi_c = cf.interval()
                    if cf.alarm(nominal_cov) and not math.isfinite(breach_origin):
                        breach_origin = float(i + 1)
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "level": level,
                        "status": "ok" if cf.n_eval else "inconclusive",
                        "error": err,
                        "n_eval": cf.n_eval,
                        "breach_rate": cf.breach_rate(),
                        "coverage_floor": floor_v,
                        "cs_low": lo_c,
                        "cs_high": hi_c,
                        "floor_below_nominal": cf.alarm(nominal_cov),
                        "floor_breach_origin": breach_origin,
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
    receipt: dict[str, Any] = {
        "schema": COVERAGE_FLOOR_SCHEMA,
        "kind": "coverage_floor",
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
            "wallet_mixture_e_process",
            "time_uniform_coverage_floor",
            "one_sided_lower_bound",
        ],
        "claims": [
            {
                "text": (
                    "coverage_floor is a time-uniform 1-alpha lower bound on "
                    "each head's true coverage rate; floor_below_nominal is "
                    "the coverage claim's floor answer"
                ),
                "kind": "theoretical",
            }
        ],
    }
    return frame, receipt


__all__ = [
    "COVERAGE_FLOOR_SCHEMA",
    "DEFAULT_WALLET_FRACS",
    "CoverageFloor",
    "audit_coverage_floor",
    "coverage_floor_bench",
]
