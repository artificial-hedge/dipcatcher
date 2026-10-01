"""Jaeger echo-state reservoir forecasting.

References
----------
- Jaeger, H. (2001). "The Echo State Approach to Analysing and
  Training Recurrent Neural Networks." *GMD Report* 148.
- Jaeger, H. & Haas, H. (2004). "Harnessing Nonlinearity:
  Predicting Chaotic Systems and Saving Energy in Wireless
  Communication." *Science* 304(5667), 78-80.
- Lukosevicius, M. (2012). "A Practical Guide to Applying Echo
  State Networks." *LNCS* 7700, 659-686.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The echo-state property requires the reservoir's spectral
radius below ~1 — we draw ``W`` dense uniform and rescale to
an explicit target ``rho`` so the fading-memory condition
holds by construction (no leakage search). States update
``s_t = (1-alpha) s_{t-1} + alpha * tanh(W_in x_t + W s_{t-1})``
with leak rate ``alpha``; the readout is ridge OLS on
``[1, x_t, s_t]`` which keeps the whole system deterministic
given ``seed`` and keeps training a single solve. We wash out
``washout`` transients before fitting — without the washout
the first-state transient contaminates the readout and the
benchmark's signal-vs-noise separation collapses.
``synth_esn`` drives the reservoir with a NARMA-10 target
(long memory, strong nonlinearity) and a pure iid series; the
bench gates on the readout achieving a materially lower
normalized MSE on NARMA than the iid baseline predicts under
an AR fallback.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 200) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _reservoir(n_res: int, sr: float, seed: int) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    w_in = rng.uniform(-0.5, 0.5, size=(n_res, 1))
    w = rng.uniform(-1.0, 1.0, size=(n_res, n_res))
    eig = np.linalg.eigvals(w)
    r_max = float(np.max(np.abs(eig)))
    if not np.isfinite(r_max) or r_max < 1e-9:
        raise ValueError("degenerate reservoir")
    return w_in, w * (sr / r_max)


def echo_state_esn(
    x: FloatArray,
    n_res: int = 100,
    sr: float = 0.9,
    leak: float = 0.5,
    ridge: float = 1e-6,
    washout: int = 100,
    seed: int = 20261231,
) -> dict[str, float]:
    """Jaeger ESN one-step-ahead ridge readout."""
    v = _as_series(x)
    if n_res < 8 or not 0.0 < sr < 1.5 or not 0.0 < leak <= 1.0:
        raise ValueError("bad reservoir hyperparameters")
    if washout + 40 >= v.size:
        raise ValueError("washout exceeds data")
    n = v.size
    w_in, w = _reservoir(n_res, sr, seed)
    s = np.zeros(n_res)
    states = np.empty((n, n_res))
    xn = (v - v.mean()) / (v.std() + 1e-12)
    for t in range(n):
        pre = w_in[:, 0] * xn[t] + w @ s
        s = (1.0 - leak) * s + leak * np.tanh(pre)
        states[t] = s
    # readout on [1, x_t, s_t] predicts x_{t+1}
    feat = np.column_stack([np.ones(n), xn, states])
    tgt = np.roll(xn, -1)[:-1]
    feat = feat[:-1]
    fit_idx = np.arange(washout, n - 1)
    ft = feat[fit_idx]
    tt = tgt[fit_idx]
    g = ft.T @ ft + ridge * np.eye(feat.shape[1])
    beta = np.linalg.solve(g, ft.T @ tt)
    pred = feat @ beta
    mse = float(np.mean((tgt[washout:] - pred[fit_idx]) ** 2))
    naive = float(np.mean(tgt[washout:] ** 2))
    rho_eff = float(np.max(np.abs(np.linalg.eigvals(w))))
    out: dict[str, float] = {
        "mse": mse,
        "nmse": mse / naive,
        "spectral_radius": rho_eff,
        "n_res": float(n_res),
    }
    return out


def synth_esn(
    seed: int = 20261231 + 337,
    n: int = 2500,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC NARMA-10 driver vs iid noise."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n)
    y = np.zeros(n)
    for t in range(10, n):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * float(np.sum(y[t - 10 : t]))
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )
    iid = rng.standard_normal(n)
    return y, iid


def bench_echo_state(seed: int = 20261231 + 337) -> dict[str, float]:
    y, iid = synth_esn(seed=seed)
    r_sig = echo_state_esn(y, n_res=80, sr=0.9, leak=0.5, washout=200, seed=7)
    r_iid = echo_state_esn(iid, n_res=80, sr=0.9, leak=0.5, washout=200, seed=7)
    gain = 1.0 - r_sig["nmse"]
    ok = r_sig["nmse"] < 0.6 and r_iid["nmse"] > 0.85 and gain > 0.4
    out: dict[str, float] = {
        "synthetic_esn_nmse_narma": r_sig["nmse"],
        "synthetic_esn_nmse_iid": r_iid["nmse"],
        "synthetic_esn_signal_gain": gain,
        "synthetic_esn_spectral_radius": r_sig["spectral_radius"],
        "score": 1.0 if ok else 0.0,
    }
    return out
