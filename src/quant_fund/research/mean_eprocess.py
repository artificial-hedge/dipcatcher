"""Bounded-mean e-process — time-uniform inference on the mean of a [0,1]
stream.

``loss_cs`` gives a CS on *loss differences* under a bounded-gap
assumption; ``coverage_floor`` gives a one-sided bound on a Bernoulli
rate. This module is the general bounded-mean primitive: for any stream
``x_t in [0,1]`` it maintains wallet-mixture e-processes for the two
composite one-sided nulls ``H0: E[X] <= mu0`` and ``H0: E[X] >= mu0``,
yielding a time-uniform confidence sequence on the mean plus one-sided
bounds — the Waudby-Smith & Ramdas bounded-mean construction in wallet
(GRAPA special-case) form.

E-factor: ``1 + lambda (x_t - mu0)`` is an e-variable under
``E[X] <= mu0`` for ``0 < lambda <= 1/mu0`` (nonneg on ``x in [0,1]``,
``E[1 + lambda(X - mu0)] <= 1``). The mirror factor
``1 - lambda' (x_t - mu0)`` with ``0 < lambda' <= 1/(1 - mu0)`` tests
``E[X] >= mu0``. Each side runs a static wallet mixture over lambda
fractions — predictable by construction, so the mixture is an e-process.

Nested-null bounds: rejecting ``E[X] <= mu0`` rules out every smaller
``mu0`` (the same e-value is valid for the subset null), so the *largest
rejected* grid point is the level-alpha lower bound on the mean; the
mirror gives the upper bound.

``audit_mean_eprocess`` applies it to fleet dominance: the bounded stream
``x_t = loss_head / (loss_head + loss_baseline)`` (``0/0 -> 0.5``) has
mean below 0.5 exactly when the head beats the baseline — an
anytime-valid dominance statement, complementary to ``loss_cs`` which
needs a bounded per-pair gap.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.scoring import pinball_loss
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    HeadFactory,
    ShardGenerator,
    resolve_shard_generators,
)
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

MEAN_EPROCESS_SCHEMA = "mean_eprocess.v1"
DEFAULT_MU_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.01, 0.991, 0.01), 2))
DEFAULT_WALLET_FRACS: tuple[float, ...] = (0.05, 0.2, 0.5, 0.8, 0.95)


def _strict_unit(x: float) -> float:
    """Strict [0,1] stream element — missing/non-finite folds would deflate
    the running mean, so anything outside the closed unit interval raises."""
    if x is None or isinstance(x, bool):
        raise ValueError(f"stream value must lie in [0,1], got {x!r}")
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stream value must lie in [0,1], got {x!r}") from exc
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"stream value must lie in [0,1], got {x!r}")
    return v


@dataclass
class MeanEProcess:
    """Wallet-mixture e-processes for a bounded stream's mean.

    ``mu0_grid`` are the tested means; ``wallet_fracs`` scale the per-side
    admissible lambda. ``lower_bound()`` / ``upper_bound()`` are the
    one-sided ``1-alpha`` bounds on the running mean; ``interval()`` is
    the two-sided ``1-alpha`` CS (``alpha/2`` per side, Bonferroni).
    """

    alpha: float = 0.05
    mu0_grid: Sequence[float] = DEFAULT_MU_GRID
    wallet_fracs: Sequence[float] = DEFAULT_WALLET_FRACS
    n: int = 0
    running_mean: float = float("nan")
    _log_w_le: np.ndarray | None = None  # [n_mu0, n_wallets] H0: mean <= mu0
    _log_w_ge: np.ndarray | None = None  # [n_mu0, n_wallets] H0: mean >= mu0

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if not self.mu0_grid:
            raise ValueError("mu0_grid is empty")
        if not self.wallet_fracs:
            raise ValueError("wallet_fracs is empty")
        fracs = np.asarray(self.wallet_fracs, dtype=float)
        if np.any((fracs <= 0.0) | (fracs >= 1.0)) or not np.all(np.isfinite(fracs)):
            raise ValueError("wallet_fracs must lie in (0,1)")
        mu0 = np.asarray(self.mu0_grid, dtype=float)
        if np.any((mu0 <= 0.0) | (mu0 >= 1.0)):
            raise ValueError("mu0_grid entries must lie in (0,1)")
        n_mu0, n_w = mu0.size, fracs.size
        self._log_w_le = np.zeros((n_mu0, n_w), dtype=float)
        self._log_w_ge = np.zeros((n_mu0, n_w), dtype=float)

    def update(self, x: float) -> tuple[float, float]:
        """Fold one bounded observation; return the two-sided CS."""
        assert self._log_w_le is not None and self._log_w_ge is not None
        v = _strict_unit(x)
        mu0 = np.asarray(self.mu0_grid, dtype=float)
        fracs = np.asarray(self.wallet_fracs, dtype=float)
        # Admissible lambdas: lam_le <= 1/mu0 keeps 1 + lam(x - mu0) >= 0 at
        # x = 0; lam_ge <= 1/(1 - mu0) keeps it nonneg at x = 1.
        lam_le = (fracs[None, :] * 0.99) / mu0[:, None]  # [n_mu0, n_w]
        lam_ge = (fracs[None, :] * 0.99) / (1.0 - mu0[:, None])
        # H0: mean <= mu0 → 1 + lam_le (x - mu0) is an e-variable.
        self._log_w_le += np.log1p(lam_le * (v - mu0[:, None]))
        # H0: mean >= mu0 → 1 + lam_ge (mu0 - x) is an e-variable.
        self._log_w_ge += np.log1p(lam_ge * (mu0[:, None] - v))
        self.n += 1
        self.running_mean = (
            v if self.n == 1 else (self.running_mean + (v - self.running_mean) / self.n)
        )
        return self.interval()

    @staticmethod
    def _log_mix(log_w: np.ndarray) -> np.ndarray:
        """Log of the uniform wallet-mixture e-value per tested mu0."""
        m = log_w.max(axis=1)
        out: np.ndarray = m + np.log(np.exp(log_w - m[:, None]).mean(axis=1))
        return out

    def lower_bound(self, alpha: float | None = None) -> float:
        """One-sided level-``alpha`` lower bound on the running mean.

        Nested nulls: rejecting ``mean <= mu0`` rules out all mu0' < mu0,
        so the largest rejected grid point is the bound. Nothing rejected
        reports the grid floor; everything rejected reports the grid top.
        """
        assert self._log_w_le is not None
        a = self.alpha if alpha is None else float(alpha)
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {a}")
        mu0 = np.asarray(self.mu0_grid, dtype=float)
        thr = math.log(1.0 / a)
        rejected = self._log_mix(self._log_w_le) >= thr
        if not rejected.any():
            return float(mu0[0])
        return float(mu0[np.flatnonzero(rejected)[-1]])

    def upper_bound(self, alpha: float | None = None) -> float:
        """One-sided level-``alpha`` upper bound on the running mean."""
        assert self._log_w_ge is not None
        a = self.alpha if alpha is None else float(alpha)
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {a}")
        mu0 = np.asarray(self.mu0_grid, dtype=float)
        thr = math.log(1.0 / a)
        rejected = self._log_mix(self._log_w_ge) >= thr
        if not rejected.any():
            return float(mu0[-1])
        return float(mu0[np.flatnonzero(rejected)[0]])

    def interval(self, alpha: float | None = None) -> tuple[float, float]:
        """Two-sided ``1-alpha`` CS on the mean (alpha/2 per side)."""
        a = self.alpha if alpha is None else float(alpha)
        lo = self.lower_bound(a / 2.0)
        hi = self.upper_bound(a / 2.0)
        if lo > hi:
            return (float("nan"), float("nan"))
        return (lo, hi)

    def evalue_le(self, mu0: float) -> float:
        """E-value for the composite null `mean <= mu0` (nan off-grid)."""
        assert self._log_w_le is not None
        g = np.asarray(self.mu0_grid, dtype=float)
        i = int(np.argmin(np.abs(g - mu0)))
        if abs(float(g[i]) - mu0) > 1e-9:
            return float("nan")
        return float(np.exp(min(700.0, self._log_mix(self._log_w_le)[i])))

    def evalue_ge(self, mu0: float) -> float:
        """E-value for the composite null `mean >= mu0` (nan off-grid)."""
        assert self._log_w_ge is not None
        g = np.asarray(self.mu0_grid, dtype=float)
        i = int(np.argmin(np.abs(g - mu0)))
        if abs(float(g[i]) - mu0) > 1e-9:
            return float("nan")
        return float(np.exp(min(700.0, self._log_mix(self._log_w_ge)[i])))

    def alarm_below(self, floor: float) -> bool:
        """True when `mean >= floor` is rejected at level ``alpha`` — the
        mean has time-uniformly fallen below the promised floor."""
        if not (0.0 < floor < 1.0):
            raise ValueError(f"floor must be in (0,1), got {floor}")
        e = self.evalue_ge(floor)
        return math.isfinite(e) and e >= 1.0 / self.alpha

    def alarm_above(self, ceiling: float) -> bool:
        """True when `mean <= ceiling` is rejected at level ``alpha``."""
        if not (0.0 < ceiling < 1.0):
            raise ValueError(f"ceiling must be in (0,1), got {ceiling}")
        e = self.evalue_le(ceiling)
        return math.isfinite(e) and e >= 1.0 / self.alpha


def share_stream(
    head_loss: Sequence[float] | NDArray[np.float64],
    baseline_loss: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bounded dominance stream ``x_t = l_h / (l_h + l_b)`` with 0/0 -> 0.5.

    ``mean < 0.5`` iff the head's average loss is below the baseline's —
    the anytime-valid dominance primitive consumed by
    ``audit_mean_eprocess``. Both inputs must be finite and nonnegative.
    """
    lh = np.asarray(head_loss, dtype=float)
    lb = np.asarray(baseline_loss, dtype=float)
    if lh.shape != lb.shape:
        raise ValueError("loss streams must have equal length")
    if not (np.all(np.isfinite(lh)) and np.all(np.isfinite(lb))):
        raise ValueError("loss streams must be finite")
    if np.any(lh < 0.0) or np.any(lb < 0.0):
        raise ValueError("losses must be nonnegative")
    denom = lh + lb
    out = np.full(lh.shape, 0.5)
    np.divide(lh, denom, out=out, where=denom > 0.0)
    return out


def mean_eprocess_bench(
    *,
    means: Sequence[float] = (0.3, 0.5, 0.7),
    n_stream: int = 512,
    alpha: float = 0.05,
    n_seeds: int = 32,
    seed: int = 0,
    mu0_grid: Sequence[float] = DEFAULT_MU_GRID,
    wallet_fracs: Sequence[float] = DEFAULT_WALLET_FRACS,
) -> dict[str, Any]:
    """Seeded validity drill over Bernoulli(mu) streams.

    For each true mean and seed, stream ``n_stream`` iid Bernoulli(mu)
    draws and record (a) whether the two-sided CS ever excluded the true
    mean — a time-uniform miscoverage event at level ``alpha`` — and (b)
    the final CS width (the shrink-rate diagnostic). SYNTHETIC receipt
    dict; deterministic given ``seed``.
    """
    if n_stream <= 0 or n_seeds <= 0:
        raise ValueError("n_stream and n_seeds must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for mu in means:
        if not (0.0 < mu < 1.0):
            raise ValueError(f"means entries must be in (0,1), got {mu}")
        n_bad = 0
        widths: list[float] = []
        for _s in range(n_seeds):
            draws = (rng.random(n_stream) < mu).astype(float)
            ep = MeanEProcess(alpha=alpha, mu0_grid=mu0_grid, wallet_fracs=wallet_fracs)
            bad = False
            last = (float("nan"), float("nan"))
            for x in draws:
                last = ep.update(float(x))
                if math.isfinite(last[0]) and (last[0] > mu + 1e-12 or last[1] < mu - 1e-12):
                    bad = True
                    break
            n_bad += int(bad)
            if math.isfinite(last[0]):
                widths.append(last[1] - last[0])
        rows.append(
            {
                "mu_true": float(mu),
                "n_seeds": n_seeds,
                "cs_excluded_true": n_bad,
                "exclusion_rate": n_bad / n_seeds,
                "median_width": float(np.median(widths)) if widths else float("nan"),
            }
        )
    frame = pl.DataFrame(rows)
    receipt: dict[str, Any] = {
        "schema": MEAN_EPROCESS_SCHEMA,
        "kind": "mean_eprocess",
        "data_label": "SYNTHETIC",
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "means": list(means),
            "n_stream": n_stream,
            "alpha": alpha,
            "n_seeds": n_seeds,
            "seed": seed,
        },
        "evidence": [
            "wallet_mixture_e_process",
            "bounded_mean_confidence_sequence",
            "waudby_smith_ramdas_construction",
        ],
        "claims": [
            {
                "text": (
                    "interval() is a time-uniform 1-alpha CS on the mean of a "
                    "bounded stream; exclusion_rate bounds the anytime "
                    "miscoverage probability"
                ),
                "kind": "theoretical",
            }
        ],
    }
    return {"frame": frame, "receipt": receipt}


def audit_mean_eprocess(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    baseline: str | None = None,
    taus: Sequence[float] = DEFAULT_TAUS,
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    alpha: float = 0.05,
    mu0_grid: Sequence[float] = DEFAULT_MU_GRID,
    wallet_fracs: Sequence[float] = DEFAULT_WALLET_FRACS,
    data_label: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Per (head, tau): stream the dominance share ``x_t = l_head / (l_head
    + l_base)`` through a MeanEProcess.

    The baseline is the alphabetically-first head when ``baseline`` is
    None. Reported per row: final CS, whether the CS excludes 0.5 and on
    which side (``dominates``/``dominated``/``inconclusive``), the first
    exclusion index, and the running share mean. ``data_label`` stamps
    provenance; when None it is derived from shard configs.
    """
    if n_train <= 0 or n_eval <= 0:
        raise ValueError("n_train and n_eval must be positive")
    if not factories or len(factories) < 2:
        raise ValueError("need at least two head factories")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    tau_arr = np.asarray(taus, dtype=float)
    if np.any((tau_arr <= 0.0) | (tau_arr >= 1.0)):
        raise ValueError("taus must lie in (0,1)")
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = resolve_shard_generators(None)
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("no shard generators resolved")
    names = sorted(factories)
    base_name = baseline if baseline is not None else names[0]
    if base_name not in factories:
        raise ValueError(f"baseline {base_name!r} not in factories")

    rows: list[dict[str, Any]] = []
    shard_labels: set[str] = set()
    n_shard = n_train + n_eval
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard = generator(n_shard, int(seed) + shard_index)
        shard_labels.add(str(shard.config.get("data_label") or "UNKNOWN"))
        y_eval = np.asarray(shard.y[n_train : n_train + n_eval], dtype=float)
        preds: dict[str, np.ndarray | None] = {}
        errs: dict[str, str | None] = {}
        for name in names:
            err: str | None = None
            try:
                model = factories[name]()
                model.fit(shard.x[:n_train], shard.y[:n_train])
                if getattr(model, "fleet_lagged_predict", False):
                    lag_x = shard.y[n_train - 1 : n_train + n_eval - 1].reshape(-1, 1)
                    q = np.asarray(model.predict(lag_x), dtype=float)
                else:
                    q = np.asarray(model.predict(shard.x[n_train : n_train + n_eval]), dtype=float)
                if q.ndim != 2 or q.shape[1] != tau_arr.shape[0]:
                    raise ValueError("prediction tau-width mismatch")
            except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError) as exc:
                q = None
                err = str(exc)
            preds[name] = q
            errs[name] = err
        q_base = preds[base_name]
        for name in names:
            if name == base_name:
                continue
            q = preds[name]
            if q is None or q_base is None:
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "baseline": base_name,
                        "tau": float("nan"),
                        "status": "error",
                        "error": errs[name] or errs[base_name],
                        "n_eval": 0,
                        "share_mean": float("nan"),
                        "cs_low": float("nan"),
                        "cs_high": float("nan"),
                        "verdict": "inconclusive",
                        "first_exclusion": float("nan"),
                    }
                )
                continue
            for t_idx, tau in enumerate(tau_arr):
                n_rows = min(n_eval, q.shape[0], y_eval.shape[0])
                lh = np.asarray(
                    pinball_loss(y_eval[:n_rows], q[:n_rows, t_idx], float(tau)),
                    dtype=float,
                )
                lb = np.asarray(
                    pinball_loss(y_eval[:n_rows], q_base[:n_rows, t_idx], float(tau)),
                    dtype=float,
                )
                xs = share_stream(lh, lb)
                ep = MeanEProcess(alpha=alpha, mu0_grid=mu0_grid, wallet_fracs=wallet_fracs)
                lo = hi = float("nan")
                first_excl = float("nan")
                for i, x in enumerate(xs):
                    lo, hi = ep.update(float(x))
                    if (
                        math.isfinite(lo)
                        and (lo > 0.5 or hi < 0.5)
                        and not math.isfinite(first_excl)
                    ):
                        first_excl = float(i + 1)
                if not math.isfinite(lo):
                    verdict = "inconclusive"
                elif hi < 0.5:
                    verdict = "dominates"
                elif lo > 0.5:
                    verdict = "dominated"
                else:
                    verdict = "inconclusive"
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "baseline": base_name,
                        "tau": float(tau),
                        "status": "ok" if ep.n else "inconclusive",
                        "error": errs[name],
                        "n_eval": ep.n,
                        "share_mean": ep.running_mean,
                        "cs_low": lo,
                        "cs_high": hi,
                        "verdict": verdict,
                        "first_exclusion": first_excl,
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
        "schema": MEAN_EPROCESS_SCHEMA,
        "kind": "mean_eprocess_audit",
        "data_label": data_label,
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "baseline": base_name,
            "taus": list(taus),
            "alpha": alpha,
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": seed,
        },
        "evidence": [
            "wallet_mixture_e_process",
            "bounded_share_stream",
            "anytime_valid_dominance",
        ],
        "claims": [
            {
                "text": (
                    "verdict='dominates' is an anytime-valid claim that the "
                    "head's mean dominance share is below 0.5 against the "
                    "baseline at the stated tau"
                ),
                "kind": "statistical",
            }
        ],
    }
    return frame, receipt


__all__ = [
    "DEFAULT_MU_GRID",
    "DEFAULT_WALLET_FRACS",
    "MEAN_EPROCESS_SCHEMA",
    "MeanEProcess",
    "audit_mean_eprocess",
    "mean_eprocess_bench",
    "share_stream",
]
