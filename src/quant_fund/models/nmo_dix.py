"""NMO correction + Dix interval-velocity inversion (seismic canon) (SYNTHETIC).

t^2(x) = t0^2 + x^2 / v_nmo^2 on a CMP gather; Dix converts a stack of
RMS velocities into interval velocities:
  v_int,i^2 = (v_rms,i^2 t_i - v_rms,i-1^2 t_{i-1}) / (t_i - t_{i-1})
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 890


def nmo_curve(t0: float, x: np.ndarray, v: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(np.sqrt(t0 * t0 + x * x / (v * v)))


def nmo_correct(times: np.ndarray, x: np.ndarray, v: float) -> np.ndarray:
    """Return t0 per trace: t0_i = sqrt(t_i^2 - x_i^2/v^2)."""
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(np.sqrt(np.maximum(t * t - x * x / (v * v), 0.0)))


def fit_vnmo(t0_guess: float, x: np.ndarray, t_obs: np.ndarray) -> float:
    """Least-squares NMO velocity from observed moveout times."""
    x = np.asarray(x, dtype=np.float64)
    t2 = np.asarray(t_obs, dtype=np.float64) ** 2
    a = np.column_stack([np.ones_like(x), x * x])
    coef, *_ = np.linalg.lstsq(a, t2, rcond=None)
    slope = float(coef[1])
    return float(1.0 / np.sqrt(max(slope, 1e-12)))


def dix_interval(v_rms: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Dix interval velocities from RMS velocities at two-way times."""
    v = np.asarray(v_rms, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    v2t = v * v * t
    num = np.diff(v2t, prepend=0.0)
    den = np.diff(t, prepend=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(np.maximum(num / np.maximum(den, 1e-12), 0.0))
    return out


def bench_nmo_dix(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # 3 layers: true interval velocities and thicknesses -> RMS series.
    vint = np.array([1500.0, 2200.0, 3000.0])
    thick = np.array([400.0, 800.0, 1000.0])  # one-way thickness (m)
    t_int = 2.0 * np.cumsum(thick) / np.cumsum(vint * np.ones(3))  # not used
    t0s = 2.0 * np.cumsum(thick) / vint  # two-way time to each base
    # true RMS at each reflector
    vrms = np.array(
        [
            np.sqrt(
                np.sum(vint[: i + 1] ** 2 * np.diff(np.concatenate([[0.0], t0s]))[: i + 1]) / t0s[i]
            )
            for i in range(3)
        ]
    )
    x = np.linspace(0.0, 2000.0, 40)
    score = 0.0
    # per-reflector NMO fit accuracy
    vn_est = np.array(
        [
            fit_vnmo(
                t0, x + rng.normal(0, 5, x.size), nmo_curve(t0, x, v) + rng.normal(0, 0.002, x.size)
            )
            for t0, v in zip(t0s, vrms, strict=True)
        ]
    )
    err = float(np.max(np.abs(vn_est - vrms) / vrms))
    score += 1.0 if err < 0.02 else 0.0
    # Dix recovery of interval velocities
    vi = dix_interval(vrms, t0s)
    derr = float(np.max(np.abs(vi - vint) / vint))
    score += 1.0 if derr < 0.02 else 0.0
    # NMO flattening: corrected times residual curvature
    t_obs = nmo_curve(t0s[0], x, vrms[0])
    resid = nmo_correct(t_obs, x, vrms[0]) - t0s[0]
    score += 1.0 if float(np.max(np.abs(resid))) < 1e-9 else 0.0
    # sanity: monotonic RMS with depth
    score += 1.0 if bool(np.all(np.diff(vrms) > 0)) else 0.0
    _ = t_int
    return {"synthetic_nmo_dix": score / 4.0}
