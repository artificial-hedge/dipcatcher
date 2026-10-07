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

Both alarms fail closed: non-finite input marks the stream
``inconclusive`` at that index rather than silently continuing — an
unmeasurable stream is not evidence of stability.

Contract: these are *level-shift* detectors. The baseline is the stream's
own history, so a stream that has been at level +c from step 1 absorbs c
into the baseline and correctly does not alarm — the alarm fires when the
stream *changes* mid-run. For a persistent-badness detector (level vs a
fixed reference), use ``evalues.LossEProcess`` with the reference as the
incumbent.
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
    """Conformal-rank betting e-process for one-sided upward drift.

    The earlier clip bet ``e_t = 1 + lam·clip((x_t − μ̂_{t-1})/s_t, ±1)``
    was not a valid e-factor: clipping around a running mean is asymmetric
    under skew (a stable left tail saturates at −1 while the positive side
    stays linear), so a level-stable left-skewed or heavy-tailed stream
    accumulated ``E[e_t | F] > 1`` factors — the meta-audit measured 45%
    false-alarm on left-skewed iid and ~10% on t3/Laplace/heteroskedastic
    at alpha = 0.05. Same bug class as the ``LossEProcess`` clip-bet fix.

    Construction (distribution-free, valid under *any* iid stream):
    ``p_t`` is the smoothed conformal rank of ``x_t`` against its own
    strict history — for an exchangeable stream ``p_t`` is uniform on
    ``(0, 1)`` regardless of tails, skew, or scale. The factor
    ``e_t = 1 + lam_t·(1 − 2 p_t)`` with ``lam_t`` a predictable GRAPA-style
    plug-in clipped to ``[0, lam]`` satisfies ``E[e_t | F_{t-1}] = 1``
    under exchangeability, so ``E_t = ∏ e_i`` is a nonnegative test
    martingale and ``P(ever alarm) ≤ alpha`` by Ville. The null tested is
    "the stream's law is stable" — a mid-run change in level *or* scale
    breaks exchangeability and is alarmed.

    ``init_scale`` is retained for API compatibility — the rank bet needs
    no scale.
    """

    alpha: float = 0.05
    lam: float = 0.5
    init_scale: float = 1e-3
    burn_in: int = 5
    seed: int = 0

    _log_e: float = 0.0
    _n: int = 0
    _sum: float = 0.0
    _seen: list[float] = field(default_factory=list)
    _s_sum: float = 0.0
    _s_sum2: float = 0.0
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
        self._rng = np.random.default_rng(self.seed)

    def _predictable_lam(self, n_seen_scores: int) -> float:
        """GRAPA plug-in on strict s-history, clipped to [0, lam]."""
        if n_seen_scores < 2:
            return 0.0
        mean = self._s_sum / n_seen_scores
        var = max(1e-9, self._s_sum2 / n_seen_scores - mean * mean)
        return float(min(max(mean / var, 0.0), self.lam))

    def update(self, x: float) -> _Step:
        v = float(x)
        if not np.isfinite(v):
            step = _Step(
                v,
                self._running_mean(),
                float(np.exp(min(self._log_e, 700.0))),
                False,
                True,
            )
            self._steps.append(step)
            return step

        if self._n < self.burn_in:
            self._seen.append(v)
            self._n += 1
            self._sum += v
            step = _Step(v, self._running_mean(), 1.0, False, False)
            self._steps.append(step)
            return step

        # lam_t from strict history of the score s = 1 - 2p (predictable).
        lam_t = self._predictable_lam(self._n - self.burn_in)

        # smoothed conformal rank of v among strict history (+ v itself):
        # uniform on (0, 1) under exchangeability of the stream.
        past = np.asarray(self._seen, dtype=float)
        greater = float(np.sum(past > v))
        ties = float(np.sum(past == v))
        u = float(self._rng.uniform())
        p = (greater + u * (ties + 1.0)) / (past.size + 1.0)
        s = 1.0 - 2.0 * p  # in (-1, 1); positive = high rank = upward drift
        e_mult = 1.0 + lam_t * s  # in (0, 1 + lam), strictly positive
        self._log_e += float(np.log(e_mult))
        self._s_sum += s
        self._s_sum2 += s * s

        self._seen.append(v)
        self._n += 1
        self._sum += v

        # capped like evalues.LossEProcess — unbounded exp overflows to inf
        e_value = float(np.exp(min(self._log_e, 700.0)))
        if self._alarm_index is None and np.log(1.0 / self.alpha) <= self._log_e:
            self._alarm_index = len(self._steps)
        step = _Step(v, self._running_mean(), e_value, self._alarm_index is not None, False)
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

    @property
    def log_e(self) -> float:
        """Log of the running e-value (sum of log factors so far)."""
        return self._log_e


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
    data_label: str = "UNKNOWN",
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
        "data_label": data_label,
        "research_only": True,
        "live_pnl_claim": False,
        "n_obs": ep._n + n_inconclusive,
        "alpha": alpha,
        "eprocess": {
            "lam": lam,
            "alarmed": ep.alarmed,
            "alarm_index": ep.alarm_index,
            "final_evalue": float(np.exp(ep.log_e)),
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
