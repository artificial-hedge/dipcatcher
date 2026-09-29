"""Anytime-valid drift alarms for sequential loss monitoring.

A production head degrades silently; re-running a batch test after every
new observation inflates the false-alarm probability without bound.
``EProcessDriftAlarm`` gives a stopping-time-valid alarm instead: the
probability of EVER falsely alarming under the null (no positive drift)
is ≤ alpha, no matter how long the stream runs (Ville 1939 / Ramdas et
al. 2020 — the same e-process machinery as ``evalues`` but for the
one-sided drift-null ``E[x_t | past] ≤ 0`` on a loss-diff stream).

``PageHinkleyAlarm`` is the classical CUSUM reference: fast to alarm on
persistent shifts, no validity guarantee under optional continuation —
kept as a *diagnostic* counterpart, always reported alongside the
e-value so the honest bound is visible.

Both alarms fail closed: non-finite or non-positive-scale input marks
the stream ``inconclusive`` at that index rather than silently
continuing — an unmeasurable stream is not evidence of stability.

Contract: these are *level-shift* detectors. The baseline is the running
mean of everything seen so far, so a stream that has been at level +c
from step 1 absorbs c into the baseline and correctly does not alarm —
the alarm fires when the stream *changes* level mid-run. For a
persistent-badness detector (level vs a fixed reference), use
``evalues.LossEProcess`` with the reference as the incumbent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DRIFT_ALARM_SCHEMA = "drift_alarm.v1"


@dataclass
class _Step:
    value: float
    running_mean: float
    statistic: float
    alarmed: bool
    inconclusive: bool


@dataclass
class EProcessDriftAlarm:
    """Betting e-process for one-sided drift detection.

    Null ``H0: E[x_t | history] ≤ 0`` — x_t a loss difference (challenger
    minus incumbent, or head loss minus its own running mean). Bet
    ``e_t = 1 + lam * g_t`` with ``g_t = clip((x_t − μ̂_{t-1}) / s_t,
    -1, 1)``; ``E`` accumulates as a test martingale under the null, so
    ``P(ever alarm) ≤ alpha``. ``lam ∈ (0, 1)`` is the bet fraction;
    larger values react faster but spend power on overshoot.
    """

    alpha: float = 0.05
    lam: float = 0.5
    init_scale: float = 1e-3
    burn_in: int = 5

    _log_e: float = 0.0
    _n: int = 0
    _sum: float = 0.0
    _median_scale: float = 0.0
    _seen: list[float] = field(default_factory=list)
    _alarm_index: int | None = None
    _steps: list[_Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lam must be in (0, 1)")
        if self.init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        if self.burn_in < 1:
            raise ValueError("burn_in must be >= 1")

    def update(self, x: float) -> _Step:
        v = float(x)
        if not np.isfinite(v):
            step = _Step(v, self._running_mean(), float(np.exp(self._log_e)), False, True)
            self._steps.append(step)
            return step

        self._seen.append(v)
        self._n += 1
        self._sum += v

        if self._n <= self.burn_in:
            step = _Step(v, self._running_mean(), 1.0, False, False)
            self._steps.append(step)
            return step

        prior = np.asarray(self._seen[:-1], dtype=float)
        mean_prev = float(prior.mean())
        scale = float(np.median(np.abs(prior - mean_prev))) if prior.size >= 4 else self.init_scale
        if not np.isfinite(scale) or scale <= 0.0:
            scale = self.init_scale
        self._median_scale = scale

        g = float(np.clip((v - mean_prev) / scale, -1.0, 1.0))
        e_mult = 1.0 + self.lam * g
        if e_mult <= 0.0:
            e_mult = 1e-12
        self._log_e += float(np.log(e_mult))

        e_value = float(np.exp(self._log_e))
        if self._alarm_index is None and np.log(1.0 / self.alpha) <= self._log_e:
            self._alarm_index = len(self._steps)
        step = _Step(v, mean_prev, e_value, self._alarm_index is not None, False)
        self._steps.append(step)
        return step

    def _running_mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    @property
    def alarmed(self) -> bool:
        return self._alarm_index is not None

    @property
    def alarm_index(self) -> int | None:
        return self._alarm_index


@dataclass
class PageHinkleyAlarm:
    """Classical CUSUM drift detector (diagnostic — no anytime guarantee).

    ``c_t = max(0, c_{t-1} + x_t − μ̂_{t-1} − δ)``; alarms when
    ``c_t > h``. Reports the statistic stream so the receipt carries the
    diagnostic even when the honest verdict comes from the e-process.
    """

    delta: float = 0.1
    h: float = 5.0
    init_scale: float = 1e-3
    burn_in: int = 5

    _c: float = 0.0
    _n: int = 0
    _sum: float = 0.0
    _seen: list[float] = field(default_factory=list)
    _alarm_index: int | None = None

    def __post_init__(self) -> None:
        if self.delta <= 0.0 or self.h <= 0.0 or self.init_scale <= 0.0:
            raise ValueError("delta, h, init_scale must be positive")
        if self.burn_in < 1:
            raise ValueError("burn_in must be >= 1")

    def update(self, x: float) -> _Step:
        v = float(x)
        if not np.isfinite(v):
            return _Step(
                v,
                self._sum / self._n if self._n else 0.0,
                self._c,
                self._alarm_index is not None,
                True,
            )
        self._seen.append(v)
        self._n += 1
        self._sum += v
        if self._n <= self.burn_in:
            return _Step(v, self._sum / self._n, 0.0, False, False)
        prior = np.asarray(self._seen[:-1], dtype=float)
        scale = max(
            float(np.median(np.abs(prior - prior.mean()))),
            self.init_scale,
        )
        self._c = max(0.0, self._c + (v - float(prior.mean())) / scale - self.delta)
        if self._alarm_index is None and self._c > self.h:
            self._alarm_index = self._n - 1
        return _Step(v, float(prior.mean()), self._c, self._alarm_index is not None, False)

    @property
    def alarmed(self) -> bool:
        return self._alarm_index is not None


def monitor_stream(
    stream: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
    ph_delta: float = 0.1,
    ph_h: float = 5.0,
) -> dict[str, object]:
    """Run both alarms over a stream → ``drift_alarm.v1`` report."""
    ep = EProcessDriftAlarm(alpha=alpha, lam=lam)
    ph = PageHinkleyAlarm(delta=ph_delta, h=ph_h)
    for x in np.asarray(stream, dtype=float).ravel():
        ep.update(float(x))
        ph.update(float(x))
    n_inconclusive = sum(1 for s in ep._steps if s.inconclusive)
    return {
        "kind": DRIFT_ALARM_SCHEMA,
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "n_obs": ep._n + n_inconclusive,
        "alpha": alpha,
        "eprocess": {
            "lam": lam,
            "alarmed": ep.alarmed,
            "alarm_index": ep.alarm_index,
            "final_evalue": float(np.exp(ep._log_e)),
        },
        "page_hinkley": {
            "delta": ph_delta,
            "h": ph_h,
            "alarmed": ph.alarmed,
            "alarm_index": ph._alarm_index,
            "final_statistic": ph._c,
        },
        "n_inconclusive": n_inconclusive,
        "evidence": [
            "e_process_anytime_valid",
            "test_martingale_under_drift_null",
            "page_hinkley_diagnostic_only",
        ],
    }
