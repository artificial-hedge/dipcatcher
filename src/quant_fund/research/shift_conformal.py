"""Shift-adaptive conformal coverage — weighted split-conformal quantiles
plus an e-process monitor on the *weighted* breach stream.

Split conformal's coverage guarantee needs exchangeability. Under
covariate shift the calibration set must be importance-weighted
(Tibshirani et al. 2019): ``weighted_quantile`` computes the level-`q`
quantile of the weighted conformity-score distribution, and the target
marginal coverage holds when the weights are the true likelihood ratio
``w = dP_target / dP_source``.

The guarantee is only as good as the weights — so coverage under shift
must be *monitored*, not assumed. ``WeightedCoverageEProcess`` is the
monitor: for stream pairs ``(w_t, b_t)`` with ``b_t in {0,1}`` a breach
flag and ``w_t in [0, w_max]`` a clipped weight, define
``d_t = w_t (b_t - alpha) / w_max in [-alpha, 1-alpha]``. Under the null
``E[w * b] <= alpha * E[w]`` (weighted coverage is at least nominal under
the target distribution), ``E[d_t] <= 0`` — so
``prod_t (1 + lambda d_t)`` with ``0 < lambda <= 1/alpha`` is an
e-process. The clip bound ``w_max`` is part of the honest contract:
clipped weights trade the exact marginal guarantee for a bounded,
monitorable stream.

``audit_shift_conformal`` runs the full loop per head per level over the
fleet shards: fit, predict intervals, apply a caller-supplied
``weight_fn(x_row)`` (uniform by default — the no-shift control), build
the weighted-conformal interval from the cal split, stream eval breaches
into the monitor, and report the final e-value / alarm / ESS.
SYNTHETIC-labeled; never a headline metric.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

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

SHIFT_CONFORMAL_SCHEMA = "shift_conformal.v1"
DEFAULT_LAMBDA_FRACS: tuple[float, ...] = (0.05, 0.2, 0.5, 0.8, 0.95)


def _strict_weight(w: float, w_max: float) -> float:
    """Clipped likelihood-ratio weight: finite, nonnegative, <= w_max.

    ``w_max`` is a declared bound of the analysis — inputs above it are
    errors, not silent clips, so a misspecified bound can't launder an
    unbounded weight into a "bounded" stream.
    """
    try:
        v = float(w)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weight must be a real number, got {w!r}") from exc
    if not math.isfinite(v) or v < 0.0 or v > w_max:
        raise ValueError(f"weight must lie in [0, {w_max}], got {w!r}")
    return v


def ess(weights: Sequence[float] | NDArray[np.float64]) -> float:
    """Effective sample size ``(sum w)^2 / sum w^2`` — the standard
    importance-weight degeneracy diagnostic (nan on empty input)."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return float("nan")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and nonnegative")
    s2 = float(np.square(w).sum())
    if s2 == 0.0:
        return 0.0
    return float(np.square(w.sum()) / s2)


def weighted_quantile(
    scores: Sequence[float] | NDArray[np.float64],
    weights: Sequence[float] | NDArray[np.float64],
    level: float,
) -> float:
    """Weighted empirical quantile at ``level`` — the weighted split-
    conformal calibration quantile (Tibshirani et al. 2019).

    Returns the smallest score whose cumulative weight share reaches
    ``level``. ``nan`` on empty input or all-zero weights.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must lie in (0,1), got {level}")
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if s.shape != w.shape:
        raise ValueError("scores and weights must have equal length")
    if s.size == 0:
        return float("nan")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
        raise ValueError("scores and weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be nonnegative")
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(w[order]) / total
    i = int(np.searchsorted(cum, level, side="left"))
    i = min(i, s.size - 1)
    return float(s[order[i]])


class WeightedCoverageEProcess:
    """E-process for the weighted-coverage null.

    ``H0: E[w * breach] <= alpha * E[w]`` — under exact likelihood-ratio
    weights this is precisely "target-distribution coverage >= 1-alpha".
    Per-step factor ``1 + lam_k * w (b - alpha) / w_max``; ``lam_k`` are
    wallet fractions of the admissible bound ``1/alpha`` (the factor stays
    nonneg because ``w(b - alpha)/w_max >= -alpha``).
    """

    def __init__(
        self,
        *,
        alpha: float,
        w_max: float,
        lambda_fracs: Sequence[float] = DEFAULT_LAMBDA_FRACS,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if not (math.isfinite(w_max) and w_max > 0.0):
            raise ValueError(f"w_max must be positive and finite, got {w_max}")
        fracs = np.asarray(lambda_fracs, dtype=float)
        if fracs.size == 0:
            raise ValueError("lambda_fracs is empty")
        if np.any((fracs <= 0.0) | (fracs >= 1.0)) or not np.all(np.isfinite(fracs)):
            raise ValueError("lambda_fracs must lie in (0,1)")
        self.alpha = float(alpha)
        self.w_max = float(w_max)
        self._lam = fracs * 0.99 / self.alpha  # [K] wallet lambdas
        self._log_w = np.zeros(fracs.size, dtype=float)
        self.n = 0
        self.n_wbreach = 0.0
        self.sum_w = 0.0

    def update(self, weight: float, breach: bool) -> float:
        """Fold one (weight, breach) pair; return the current e-value."""
        w = _strict_weight(weight, self.w_max)
        b = float(_strict_breach(breach))
        d = w * (b - self.alpha) / self.w_max
        self._log_w += np.log1p(self._lam * d)
        self.n += 1
        self.n_wbreach += w * b
        self.sum_w += w
        return self.evalue()

    def evalue(self) -> float:
        """Uniform wallet-mixture e-value (log-sum-exp over wallets)."""
        m = float(self._log_w.max())
        return float(np.exp(min(700.0, m + math.log(float(np.exp(self._log_w - m).mean())))))

    def weighted_breach_rate(self) -> float:
        """``sum(w * b) / sum(w)`` — the weighted empirical breach rate
        (nan before any update; positive-weight mass required)."""
        if self.sum_w <= 0.0:
            return float("nan")
        return self.n_wbreach / self.sum_w

    def alarm(self) -> bool:
        """True when the weighted-coverage null is rejected at ``alpha``."""
        return self.evalue() >= 1.0 / self.alpha


def make_logistic_tilt(beta: float, *, col: int = 0) -> Callable[[np.ndarray], float]:
    """Covariate-shift weight oracle ``w(x) = 2 / (1 + exp(beta * x[col]))``
    — a deterministic mean-1-ish tilt for SYNTHETIC drills: positive beta
    downweights large-x rows. Caller-declared; the function is data-blind.
    """
    b = float(beta)
    if not math.isfinite(b):
        raise ValueError(f"beta must be finite, got {beta}")
    j = int(col)
    if j < 0:
        raise ValueError("col must be nonnegative")

    def _w(x_row: np.ndarray) -> float:
        z = b * float(np.asarray(x_row, dtype=float)[j])
        return float(2.0 / (1.0 + math.exp(min(60.0, max(-60.0, z)))))

    return _w


def uniform_weight(x_row: np.ndarray) -> float:
    """The no-shift control weight oracle."""
    del x_row
    return 1.0


def shift_conformal_bench(
    *,
    mu_shift: float = 2.0,
    n_stream: int = 512,
    alpha: float = 0.10,
    w_max: float = 4.0,
    n_seeds: int = 32,
    seed: int = 0,
    lambda_fracs: Sequence[float] = DEFAULT_LAMBDA_FRACS,
) -> dict[str, Any]:
    """Seeded validity/power drill.

    Control arm: ``w_t`` iid mean-1 (clipped), breach iid Bernoulli(alpha)
    — the null holds; alarm rate must stay under ``1/w_max``-scale chance
    (pinned at <= 1/n_seeds slack). Treated arm: breach rate inflated to
    ``min(1, mu_shift * alpha)`` — the alarm must fire. SYNTHETIC receipt;
    deterministic given ``seed``.
    """
    if n_stream <= 0 or n_seeds <= 0:
        raise ValueError("n_stream and n_seeds must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if not (math.isfinite(w_max) and w_max > 0.0):
        raise ValueError(f"w_max must be positive and finite, got {w_max}")
    if not (math.isfinite(mu_shift) and mu_shift >= 1.0):
        raise ValueError(f"mu_shift must be >= 1, got {mu_shift}")
    rng = np.random.default_rng(seed)
    p_shift = min(1.0, mu_shift * alpha)
    rows: list[dict[str, Any]] = []
    for arm, p in (("control", alpha), ("shifted", p_shift)):
        n_alarm = 0
        evals: list[float] = []
        for _s in range(n_seeds):
            ep = WeightedCoverageEProcess(alpha=alpha, w_max=w_max, lambda_fracs=lambda_fracs)
            # mean-1 weights: lognormal(0, 0.5) clipped to w_max
            ws = np.minimum(rng.lognormal(0.0, 0.5, n_stream), w_max)
            bs = rng.random(n_stream) < p
            fired = False
            for w, b in zip(ws, bs, strict=True):
                e = ep.update(float(w), bool(b))
                if e >= 1.0 / alpha:
                    fired = True
                    break
            n_alarm += int(fired)
            evals.append(e)
        rows.append(
            {
                "arm": arm,
                "breach_rate": p,
                "n_seeds": n_seeds,
                "alarm_count": n_alarm,
                "alarm_rate": n_alarm / n_seeds,
                "median_final_evalue": float(np.median(evals)),
            }
        )
    frame = pl.DataFrame(rows)
    receipt: dict[str, Any] = {
        "schema": SHIFT_CONFORMAL_SCHEMA,
        "kind": "shift_conformal",
        "data_label": "SYNTHETIC",
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "mu_shift": mu_shift,
            "n_stream": n_stream,
            "alpha": alpha,
            "w_max": w_max,
            "n_seeds": n_seeds,
            "seed": seed,
        },
        "evidence": [
            "wallet_mixture_e_process",
            "weighted_coverage_null",
            "weighted_split_conformal",
        ],
        "claims": [
            {
                "text": (
                    "the control arm's alarm rate bounds the probability of "
                    "ever rejecting when weighted coverage is nominal; the "
                    "shifted arm demonstrates detection power"
                ),
                "kind": "statistical",
            }
        ],
    }
    return {"frame": frame, "receipt": receipt}


def audit_shift_conformal(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    *,
    levels: Sequence[float] = (0.8, 0.9),
    n_train: int = 512,
    n_eval: int = 256,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.05,
    w_max: float = 4.0,
    weight_fn: Callable[[np.ndarray], float] = uniform_weight,
    lambda_fracs: Sequence[float] = DEFAULT_LAMBDA_FRACS,
    data_label: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Per (head, level): weighted-conformal interval + monitored eval.

    Calibration conformity = ``max(lo - y, y - hi)`` on a held-out
    quarter of the train split; the eval interval widens the head's own
    band by the weighted conformal quantile minus the head's implied
    half-width... For heads already emitting intervals we keep the
    conformity residual ``res = max(lo - y, y - hi, 0)`` — the margin by
    which the band must grow — and apply ``band + q_{1-alpha}`` widening
    to the eval rows. Each eval row then streams ``(w, breach)`` into a
    ``WeightedCoverageEProcess``; rows report ESS, weighted breach rate,
    e-value, alarm, and the first alarm index. ``weight_fn`` receives the
    eval feature row (uniform is the no-shift control); every weight is
    validated against the declared ``w_max``. ``data_label`` is derived
    from shard configs when None.
    """
    if n_train <= 64 or n_eval <= 0:
        raise ValueError("n_train must exceed 64 and n_eval must be positive")
    if not factories:
        raise ValueError("factories is empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if not (math.isfinite(w_max) and w_max > 0.0):
        raise ValueError(f"w_max must be positive and finite, got {w_max}")
    if not callable(weight_fn):
        raise TypeError("weight_fn must be callable")
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

    n_cal = max(32, n_train // 4)
    rows: list[dict[str, Any]] = []
    shard_labels: set[str] = set()
    n_shard = n_train + n_eval
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard = generator(n_shard, int(seed) + shard_index)
        shard_labels.add(str(shard.config.get("data_label") or "UNKNOWN"))
        y_cal = np.asarray(shard.y[n_train - n_cal : n_train], dtype=float)
        x_cal = np.asarray(shard.x[n_train - n_cal : n_train], dtype=float)
        y_eval = np.asarray(shard.y[n_train : n_train + n_eval], dtype=float)
        x_eval = np.asarray(shard.x[n_train : n_train + n_eval], dtype=float)
        w_eval = np.asarray([float(weight_fn(row)) for row in x_eval], dtype=float)
        w_cal = np.asarray([float(weight_fn(row)) for row in x_cal], dtype=float)
        for name in sorted(factories):
            factory = factories[name]
            err: str | None = None
            q_cal = q_eval = None
            try:
                model = factory()
                model.fit(shard.x[: n_train - n_cal], shard.y[: n_train - n_cal])
                x_full = np.asarray(shard.x, dtype=float)
                if getattr(model, "fleet_lagged_predict", False):
                    lag = np.asarray(shard.y, dtype=float)
                    q_all = np.asarray(model.predict(lag[:-1].reshape(-1, 1)), dtype=float)
                else:
                    q_all = np.asarray(model.predict(x_full), dtype=float)
                if q_all.ndim != 2 or q_all.shape[1] != tau_arr.shape[0]:
                    raise ValueError("prediction tau-width mismatch")
                q_cal = q_all[n_train - n_cal : n_train]
                q_eval = q_all[n_train : n_train + n_eval]
            except (
                ValueError,
                TypeError,
                RuntimeError,
                ArithmeticError,
                KeyError,
            ) as exc:
                err = str(exc)
            for level, idx in level_index.items():
                if idx is None or q_eval is None or q_cal is None:
                    rows.append(
                        {
                            "shard": shard_name,
                            "head": name,
                            "level": level,
                            "status": "level_unsupported" if q_eval is not None else "error",
                            "error": err,
                            "n_eval": 0,
                            "ess": float("nan"),
                            "weighted_breach_rate": float("nan"),
                            "conformal_evalue": float("nan"),
                            "alarm": False,
                            "first_alarm": float("nan"),
                        }
                    )
                    continue
                lo_i, hi_i = idx
                residuals = np.maximum.reduce(
                    [
                        q_cal[:, lo_i] - y_cal,
                        y_cal - q_cal[:, hi_i],
                        np.zeros(y_cal.shape[0]),
                    ]
                )
                # residual is the one-sided margin the band must grow by;
                # the level-quantile widened band covers at >= level.
                q_widen = weighted_quantile(residuals, w_cal, level)
                ep = WeightedCoverageEProcess(
                    alpha=1.0 - level, w_max=w_max, lambda_fracs=lambda_fracs
                )
                first_alarm = float("nan")
                last_e = float("nan")
                n_seen = 0
                for i in range(min(n_eval, q_eval.shape[0], y_eval.shape[0])):
                    lo_q = float(q_eval[i, lo_i]) - q_widen
                    hi_q = float(q_eval[i, hi_i]) + q_widen
                    w = _strict_weight(w_eval[i], w_max)
                    breach = bool(y_eval[i] < lo_q or y_eval[i] > hi_q)
                    last_e = ep.update(w, breach)
                    n_seen += 1
                    if ep.alarm() and not math.isfinite(first_alarm):
                        first_alarm = float(n_seen)
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "level": level,
                        "status": "ok" if n_seen else "inconclusive",
                        "error": err,
                        "n_eval": n_seen,
                        "ess": ess(w_eval[: max(n_seen, 1)]),
                        "weighted_breach_rate": ep.weighted_breach_rate(),
                        "conformal_evalue": last_e,
                        "alarm": ep.alarm(),
                        "first_alarm": first_alarm,
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
        "schema": SHIFT_CONFORMAL_SCHEMA,
        "kind": "shift_conformal_audit",
        "data_label": data_label,
        "level": "research",
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "code_revision": git_revision(),
        "params": {
            "levels": list(levels),
            "alpha": alpha,
            "w_max": w_max,
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": seed,
        },
        "evidence": [
            "weighted_split_conformal",
            "clipped_likelihood_ratio_weights",
            "e_process_weighted_coverage_monitor",
        ],
        "claims": [
            {
                "text": (
                    "alarm=True is an anytime-valid rejection of "
                    "weighted-coverage >= level under the declared weight "
                    "oracle and clip bound"
                ),
                "kind": "statistical",
            }
        ],
    }
    return frame, receipt


__all__ = [
    "DEFAULT_LAMBDA_FRACS",
    "SHIFT_CONFORMAL_SCHEMA",
    "WeightedCoverageEProcess",
    "audit_shift_conformal",
    "ess",
    "make_logistic_tilt",
    "shift_conformal_bench",
    "uniform_weight",
    "weighted_quantile",
]
