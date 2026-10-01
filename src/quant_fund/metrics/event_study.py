"""Event-study abnormal-return inference (MacKinlay).

Market model r_it = α_i + β_i r_mt + e_it estimated on an
estimation window before the event; abnormal returns in the
event window AR_it = r_it − (α̂_i + β̂_i r_mt) are aggregated into
CAR and tested with the standardized Patell statistic —
separating genuine event response from ordinary co-movement.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure abnormal-return detection on
generated event data — never market evidence.

References:
- MacKinlay, A. C. (1997). Event studies in economics and
  finance. *J. Economic Literature* 35, 13-39 — market model,
  CAR, cross-sectional tests.
- Patell, J. M. (1976). Corporate forecasts of earnings per
  share and stock price behavior. *J. Accounting Research* 14 —
  standardized abnormal-return test.
- Boehmer, E., Musumeci, J., Poulsen, A. B. (1991). Event-study
  methodology under conditions of event-induced variance.
  *J. Financial Economics* 30 — variance-adjusted BMP test.
- Campbell, J. Y., Lo, A. W., MacKinlay, A. C. (1997). *The
  Econometrics of Financial Markets*, ch. 4.

Composition: pure numpy + scipy — per-event OLS market model,
CAR aggregation, Patell z and BMP t; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def event_study(
    r: FloatArray,
    r_mkt: FloatArray,
    event_idx: FloatArray,
    est_window: int = 120,
    evt_window: int = 5,
) -> dict[str, float]:
    """Market-model event study for a panel of events.

    ``r`` is (n_events × T) asset returns aligned so each event
    sits at index ``event_idx[i]`` within its row; ``r_mkt`` is
    the shared market series of length T. Returns mean CAR,
    Patell z, BMP t, and per-event σ-adjusted stats."""
    rr = np.asarray(r, dtype=np.float64)
    rm = np.asarray(r_mkt, dtype=np.float64).ravel()
    ev = np.asarray(event_idx, dtype=np.float64).ravel().astype(int)
    if rr.ndim != 2:
        raise ValueError("r must be (n_events × T)")
    n_ev, t = rr.shape
    if n_ev < 8 or t < 60:
        raise ValueError("n_events>=8, T>=60")
    if rm.size != t:
        raise ValueError("r_mkt must match T")
    if ev.size != n_ev:
        raise ValueError("event_idx must match n_events")
    if not (30 <= est_window <= t - 20):
        raise ValueError("est_window in 30..T-20")
    if not (1 <= evt_window <= 15):
        raise ValueError("evt_window in 1..15")
    if not np.all(np.isfinite(rr)) or not np.all(np.isfinite(rm)):
        raise ValueError("finite inputs required")

    half = evt_window // 2
    cars = []
    sars = []
    used = 0
    for i in range(n_ev):
        e_i = int(ev[i])
        est_lo, est_hi = e_i - est_window - half - 1, e_i - half - 1
        ev_lo, ev_hi = e_i - half, e_i + half + 1
        if est_lo < 0 or ev_hi > t or est_hi <= est_lo:
            continue
        x_est = np.column_stack([np.ones(est_hi - est_lo), rm[est_lo:est_hi]])
        y_est = rr[i, est_lo:est_hi]
        b = np.linalg.lstsq(x_est, y_est, rcond=None)[0]
        resid = y_est - x_est @ b
        s_i = float(np.std(resid))
        if s_i < 1e-12:
            continue
        x_ev = np.column_stack([np.ones(ev_hi - ev_lo), rm[ev_lo:ev_hi]])
        ar = rr[i, ev_lo:ev_hi] - x_ev @ b
        cars.append(float(np.sum(ar)))
        sars.append(float(np.sum(ar)) / (s_i * np.sqrt(ev_hi - ev_lo)))
        used += 1
    if used < 6:
        raise ValueError("fewer than 6 usable events")

    car = np.array(cars)
    sar = np.array(sars)
    mean_car = float(np.mean(car))
    # Patell: z = mean(SAR) · √n (SAR_i ~ N(0,1) under null)
    patell_z = float(np.mean(sar) * np.sqrt(used))
    p_patell = float(2 * (1 - stats.norm.cdf(abs(patell_z))))
    # BMP: standardize by cross-sectional sd of SAR
    s_cross = float(np.std(sar, ddof=1))
    bmp_t = float(np.mean(sar) / (s_cross / np.sqrt(used)))
    p_bmp = float(2 * (1 - stats.t.cdf(abs(bmp_t), used - 1)))

    return {
        "n_events": float(used),
        "mean_car": mean_car,
        "sd_car": float(np.std(car)),
        "mean_sar": float(np.mean(sar)),
        "patell_z": patell_z,
        "p_patell": p_patell,
        "bmp_t": bmp_t,
        "p_bmp": p_bmp,
        "share_pos_car": float(np.mean(car > 0)),
    }


def synth_events(
    n_events: int = 40,
    t: int = 260,
    abnormal: float = 0.015,
    beta: float = 1.1,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """Market-model DGP: r = β·r_m + idio; an abnormal return of
    ``abnormal`` is injected into each event's post window."""
    rng = np.random.default_rng(seed)
    r_m = rng.normal(0.0005, 0.01, t)
    r = np.zeros((n_events, t))
    ev = np.full(n_events, 200)
    half = 2
    for i in range(n_events):
        e = rng.normal(0.0, 0.008, t)
        r[i] = -0.002 + beta * r_m + e
        # decaying abnormal response over the event window
        for k in range(ev[i] - half, ev[i] + half + 1):
            r[i, k] += abnormal * (1.0 - 0.15 * (k - ev[i] + half))
    return {
        "r": r,
        "r_mkt": r_m,
        "event_idx": ev.astype(np.float64),
        "abnormal": np.array([abnormal]),
    }


def bench_event_study(seed: int = 20261231 + 237) -> dict[str, float]:
    """Event-study self-check: injected abnormal returns detected
    by Patell z and BMP t; a zero-shock panel keeps the null.
    All ``synthetic_*``."""
    d = synth_events(abnormal=0.015, seed=seed)
    out = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))
    d0 = synth_events(abnormal=0.0, seed=seed + 1)
    out0 = event_study(np.asarray(d0["r"]), np.asarray(d0["r_mkt"]), np.asarray(d0["event_idx"]))
    out_b = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))

    return {
        "synthetic_mean_car": float(out["mean_car"]),
        "synthetic_car_err": float(abs(float(out["mean_car"]) - 0.015 * 3.4)),
        "synthetic_patell_z": float(out["patell_z"]),
        "synthetic_p_patell": float(out["p_patell"]),
        "synthetic_bmp_t": float(out["bmp_t"]),
        "synthetic_p_bmp": float(out["p_bmp"]),
        "synthetic_p_noshock": float(out0["p_patell"]),
        "synthetic_share_pos": float(out["share_pos_car"]),
        "synthetic_detects": float(
            float(out["p_patell"]) < 0.01
            and float(out["p_bmp"]) < 0.01
            and float(out0["p_patell"]) > 0.05
            and float(out["mean_car"]) > 0.02
        ),
        "synthetic_determinism": float(float(out["mean_car"]) == float(out_b["mean_car"])),
    }
