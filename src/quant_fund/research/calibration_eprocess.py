"""Anytime-valid calibration audit — an e-process over the PIT stream.

The lineage is Vovk (2021, "Testing randomness online") plus the
calibration-testing-by-betting programme (Vovk & Petej 2014, "Venn–Abers
predictors") and the conformal-test-martingale literature the sibling
module ``metrics/conformal_martingale.py`` already implements. The
distinction that matters operationally:

* ``conformal_martingale`` tests **exchangeability of the score stream** —
  it detects *change*, and self-normalizes against its own growing bag, so
  a head that is miscalibrated from the very first evaluation is
  indistinguishable: the bag converges to the head's own (wrong) PIT law
  and the conformal p-values re-uniformize.
* This module tests the **level claim** ``PIT_t ~ Uniform(0, 1)`` directly.
  A persistently too-narrow head produces U-shaped PIT forever, and the
  e-process keeps compounding evidence against calibration — it never
  re-baselines the null.

Construction. Under the calibration null the PIT draws are uniform. Any
predictable density ``f_t : [0,1] -> [0, inf)`` with ``∫ f_t(u) du = 1``
makes ``W_t = ∏ f_i(U_i)`` a nonnegative supermartingale with W_0 = 1;
Ville's inequality then gives ``P(sup_t W_t >= 1/alpha) <= alpha`` — an
anytime-valid miscalibration alarm with zero burn-in. We use four fixed
polynomial bets (bounded, so boundary PIT draws u ∈ {0, 1} cannot explode
the wealth the way power martingales ``κ p^{κ−1}`` do):

* ``loc_hi`` / ``loc_lo`` — ``f(u) = 1 + λ (2u − 1)`` with ``λ = ±1/2``
  (``|λ| ≤ 1`` keeps ``f ≥ 0``; ``E_U[2U−1] = 0`` ⇒ ``∫ f = 1``). Alarm on
  PIT drifted toward one tail — directional bias in the predictive law.
* ``overconf`` / ``underconf`` — ``f(u) = 1 + λ ((2u−1)² − 1/3)`` with
  ``λ = +3/2`` and ``λ = −3/2`` respectively (``λ ∈ [−3/2, 3]`` keeps
  ``f ≥ 0``; ``E_U[(2U−1)²] = 1/3`` ⇒ ``∫ f = 1``). Positive λ bets on
  U-shaped PIT (intervals too narrow); negative λ on hump-shaped PIT
  (intervals too wide).

Wealth is the fixed convex mixture over channels — a convex combination
of e-processes is an e-process, so the mixture keeps the Ville bound
without choosing a channel ex ante. Fixed bets are a deliberate honesty
choice: predictable λ-adaptation would sharpen power but invites silent
hindsight tuning; the constants are printed in the receipt so the audit
is replayable.

Outputs feed ``verify-receipt`` contracts (``calibration_audit.v1``) and
the composite ``honest_verdict`` lane. Proper scores only: the audit reads
PIT, never P&L.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from quant_fund.metrics.scoring import pit_values
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    SHARD_GENERATORS,
    ShardGenerator,
    SyntheticShard,
    resolve_shard_generators,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CALIBRATION_AUDIT_SCHEMA = "calibration_audit.v1"


def _loc_density(lam: float) -> Callable[[float], float]:
    # |λ| ≤ 1 keeps f ≥ 0 on [0, 1]; E_U[2U−1] = 0 ⇒ ∫f = 1.
    if abs(lam) > 1.0:
        raise ValueError("loc |λ| must be <= 1")
    return lambda u: 1.0 + lam * (2.0 * u - 1.0)


def _disp_density(lam: float) -> Callable[[float], float]:
    # λ ∈ [−3/2, 3] keeps f ≥ 0 on [0, 1]; E_U[(2U−1)²] = 1/3 ⇒ ∫f = 1.
    if not (-1.5 <= lam <= 3.0):
        raise ValueError("disp λ must be in [-1.5, 3]")
    return lambda u: 1.0 + lam * ((2.0 * u - 1.0) * (2.0 * u - 1.0) - 1.0 / 3.0)


class _GrapaChannel:
    """Predictable-mixture bet on a centered PIT moment.

    λ_t is fitted from the moment's running mean/variance *before* seeing
    u_t (GRAPA-style plug-in, clipped to ±lam_max), so the bet is
    F_{t-1}-measurable — valid even when the underlying series is serially
    dependent, since only the *PIT* stream enters the history.
    ``moment(u)`` must satisfy E_U[moment]=0 and |moment|<=1.

    The factor ``1 + λ·moment(u)`` is floored at 0 on return, but flooring
    is NOT the safety mechanism — ``max(0, 1+λd) ≥ 1+λd``, so if the floor
    ever bound, E[factor|null] could exceed 1. Validity therefore requires
    ``lam_max · sup_u|moment(u)| ≤ 1``, which ``__init__`` certifies
    numerically on a dense grid (the shipped moments are low-order
    polynomials; callers adding a moment must keep it bounded).
    """

    _CERT_GRID = 10_001

    def __init__(self, moment: Callable[[float], float], lam_max: float) -> None:
        if not (np.isfinite(lam_max) and 0.0 < lam_max < 1.0):
            raise ValueError("lam_max must be in (0, 1)")
        grid = np.linspace(0.0, 1.0, self._CERT_GRID)
        m_max = float(np.max(np.abs([moment(float(u)) for u in grid])))
        if not np.isfinite(m_max):
            raise ValueError("moment must be finite on [0, 1]")
        if lam_max * m_max > 1.0:
            raise ValueError(f"uncertified bet: lam_max({lam_max}) * sup|moment|({m_max:.4f}) > 1")
        self._moment = moment
        self._lam_max = lam_max
        self._sum = 0.0
        self._sum2 = 0.0
        self._n = 0

    def __call__(self, u: float) -> float:
        if self._n > 1:
            mean = self._sum / self._n
            var = max(1e-9, self._sum2 / self._n - mean * mean)
            lam = mean / var
            lam = max(-self._lam_max, min(self._lam_max, lam))
        else:
            lam = 0.0
        d = self._moment(u)
        self._sum += d
        self._sum2 += d * d
        self._n += 1
        return max(0.0, 1.0 + lam * d)


def _loc_moment(u: float) -> float:
    return 2.0 * u - 1.0


def _disp_moment(u: float) -> float:
    d = 2.0 * u - 1.0
    return (d * d - 1.0 / 3.0) / (4.0 / 5.0)  # rescale into [-1/3, ~0.94]


def default_channels() -> dict[str, Callable[[float], float]]:
    """Fixed bets + two adaptive GRAPA channels, mixed at equal weight."""
    return {
        "loc_hi": _loc_density(0.5),
        "loc_lo": _loc_density(-0.5),
        "overconf": _disp_density(1.5),
        "underconf": _disp_density(-1.5),
        # adaptive: lam_max |d| <= 0.9 keeps bets bounded and positive
        "grapa_loc": _GrapaChannel(_loc_moment, lam_max=0.9),
        "grapa_disp": _GrapaChannel(_disp_moment, lam_max=0.9),
    }


@dataclass
class CalibrationEProcess:
    """Fixed-mixture betting e-process against the uniform-PIT null.

    ``alpha`` is the alarm level: ``P(alarm ever | calibrated head) <= alpha``.
    Non-finite or out-of-range PIT values are recorded as ``inconclusive``
    steps (they neither compound wealth nor reset it) — a head emitting NaN
    quantiles must not be able to launder evidence.
    """

    alpha: float = 0.05
    channels: dict[str, Callable[[float], float]] | None = None

    _weights: dict[str, float] = field(default_factory=dict)
    _wealths: dict[str, float] = field(default_factory=dict)
    _n: int = 0
    _n_inconclusive: int = 0
    _alarm_at: int | None = None
    _wealth_path: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.alpha) and 0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        channels = self.channels if self.channels is not None else default_channels()
        if not channels:
            raise ValueError("channels must be nonempty")
        w = 1.0 / len(channels)
        self._wealths = {name: 1.0 for name in channels}
        self._weights = {name: w for name in channels}
        self.channels = channels

    def update(self, u: float) -> float:
        """Fold one PIT draw into the wealth; returns current wealth."""
        if not (np.isfinite(u) and 0.0 <= u <= 1.0):
            self._n_inconclusive += 1
            self._wealth_path.append(self.wealth)
            self._n += 1
            return self.wealth
        if not (self.channels is not None):
            raise ValueError("self.channels is not None")
        for name, f in self.channels.items():
            self._wealths[name] *= f(u)
        self._n += 1
        w = self.wealth
        self._wealth_path.append(w)
        if self._alarm_at is None and w >= 1.0 / self.alpha:
            self._alarm_at = self._n - 1
        return w

    @property
    def wealth(self) -> float:
        return sum(self._weights[k] * v for k, v in self._wealths.items())

    @property
    def alarmed(self) -> bool:
        return self._alarm_at is not None

    @property
    def alarm_origin(self) -> int | None:
        return self._alarm_at

    @property
    def channel_wealths(self) -> dict[str, float]:
        return dict(self._wealths)

    @property
    def n_steps(self) -> int:
        return self._n

    @property
    def n_inconclusive(self) -> int:
        return self._n_inconclusive


def audit_head_calibration(
    factories: Mapping[str, Callable[[], Any]],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.05,
    data_label: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit each head per shard; stream eval-PIT through the e-process.

    Mirrors ``fleet_eval`` conventions: identical (shard, seed) discipline,
    error rows recorded not crashed, honesty stamps on the receipt. The PIT
    stream order is the eval-row order — the same order a ``fleet_race``
    incumbent would face them. Diagnostics (mean PIT) ride along for
    context; the e-process is the inference.
    """
    if not factories:
        raise ValueError("calibration audit requires a nonempty factories mapping")
    if n_train < 1 or n_eval < 1:
        raise ValueError("n_train and n_eval must be positive")
    tau_arr = np.asarray(list(taus), dtype=float)
    if (
        tau_arr.size == 0
        or not np.isfinite(tau_arr).all()
        or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0))
        or np.any(np.diff(tau_arr) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")

    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = SHARD_GENERATORS
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("calibration audit requires at least one shard")

    channel_names = list(default_channels())
    rows: list[dict[str, Any]] = []
    shard_meta: dict[str, Any] = {}
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard_seed = int(seed) + shard_index
        shard: SyntheticShard = generator(n_train + n_eval, shard_seed)
        x = np.asarray(shard.x, dtype=float)
        y = np.asarray(shard.y, dtype=float)
        if x.shape[0] != y.shape[0] or y.shape[0] < n_train + n_eval:
            raise ValueError(f"shard {shard_name!r} produced inconsistent arrays")
        x_tr, x_ev = x[:n_train], x[n_train : n_train + n_eval]
        y_tr, y_ev = y[:n_train], y[n_train : n_train + n_eval]
        shard_meta[shard_name] = {
            "n": int(y.size),
            "seed": shard_seed,
            "x_sha256": hash_bytes(x.tobytes()),
            "y_sha256": hash_bytes(y.tobytes()),
            "data_label": str(shard.config.get("data_label") or "UNKNOWN"),
        }

        for model_name, factory in factories.items():
            row: dict[str, Any] = {
                "shard": shard_name,
                "model": model_name,
                "status": "ok",
                "error": None,
            }
            try:
                head = factory()
                head.fit(x_tr, y_tr)
                q = np.asarray(head.predict(x_ev), dtype=float)
                if q.ndim != 2 or q.shape[0] != x_ev.shape[0] or q.shape[1] != tau_arr.size:
                    raise ValueError(
                        f"predict returned shape {q.shape}; expected ({x_ev.shape[0]}, {tau_arr.size})"
                    )
                pits = pit_values(y_ev, q, tau_arr)
                proc = CalibrationEProcess(alpha=alpha)
                for u in pits.tolist():
                    proc.update(u)
                row.update(
                    {
                        "n_pit": proc.n_steps,
                        "n_inconclusive": proc.n_inconclusive,
                        "final_evalue": proc.wealth,
                        "miscalibrated": proc.alarmed,
                        "alarm_origin": proc.alarm_origin,
                        "pit_mean": float(np.nanmean(pits)),
                    }
                )
                for name in channel_names:
                    row[f"wealth_{name}"] = proc.channel_wealths[name]
            except (ValueError, TypeError, RuntimeError, ArithmeticError, KeyError) as exc:
                # Narrowed from `except Exception` (quality ratchet): head fit/predict
                # faults are solver/numeric; exotic errors propagate. Recorded, not crashed.
                row["status"] = "error"
                row["error"] = str(exc)
                for name in channel_names:
                    row[f"wealth_{name}"] = None
            rows.append(row)

    schema = {
        "shard": pl.String,
        "model": pl.String,
        "status": pl.String,
        "error": pl.String,
        "n_pit": pl.Int64,
        "n_inconclusive": pl.Int64,
        "final_evalue": pl.Float64,
        "miscalibrated": pl.Boolean,
        "alarm_origin": pl.Int64,
        "pit_mean": pl.Float64,
    }
    for name in channel_names:
        schema[f"wealth_{name}"] = pl.Float64
    frame = pl.DataFrame(rows, schema=schema)
    inputs_sha256 = hash_bytes(
        "|".join(
            f"{k}:{v['seed']}:{v['x_sha256']}:{v['y_sha256']}"
            for k, v in sorted(shard_meta.items())
        ).encode()
    )
    # Cross-receipt dataset fingerprint: evaluated-stream digests only —
    # receipts over the same shard content edge in the consistency lattice.
    dataset_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "shards": {
                    name: {"x_sha256": m["x_sha256"], "y_sha256": m["y_sha256"]}
                    for name, m in shard_meta.items()
                }
            }
        )
    )
    if data_label is None:
        distinct = {str(m["data_label"]) for m in shard_meta.values()}
        if distinct == {"SYNTHETIC"}:
            data_label = "SYNTHETIC"
        elif len(distinct) > 1:
            data_label = "MIXED"
        else:
            data_label = next(iter(distinct), "UNKNOWN")
    receipt: dict[str, Any] = {
        "kind": CALIBRATION_AUDIT_SCHEMA,
        "schema": CALIBRATION_AUDIT_SCHEMA,
        "data_label": data_label,
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at_commit": git_revision(),
        "inputs_sha256": inputs_sha256,
        "dataset_sha256": dataset_sha256,
        "params": {
            "n_train": n_train,
            "n_eval": n_eval,
            "alpha": alpha,
            "channels": channel_names,
            "taus": [float(t) for t in tau_arr],
            "seed": int(seed),
        },
        "n_shards": len(shard_meta),
        "n_models": len(factories),
        "evidence": [
            "ville_inequality",
            "nonnegative_test_martingale",
            "fixed_predictable_bets",
            "level_test_not_change_test",
        ],
    }
    return frame, receipt
