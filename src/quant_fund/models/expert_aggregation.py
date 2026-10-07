"""Prediction with expert advice — regret-bounded adaptive forecaster mixing (SYNTHETIC).

Online aggregation of a panel of expert forecasters under proper losses,
with finite-sample regret guarantees: Hedge / exponentially weighted
average (Littlestone & Warmuth 1994; Freund & Schapire 1997), Herbster &
Warmuth (1998) fixed-share tracking for switching experts, and
sleeping/specialist experts for time-varying active sets. The static
combination counterpart lives in ``research`` vincentization / mean-ensemble
machinery; ``models/ensemble`` is Bayesian stacking — this module is the
*regret-bounded online* complement.

Honesty
-------
All bench outputs are ``synthetic_*`` correctness diagnostics on seeded
synthetic expert panels — they verify regret bounds and tracking behaviour,
never market value. Proper losses only (pinball, squared). No
Sharpe/Sortino/Calmar/P&L/NAV ever.

References
----------
- Littlestone, N. & Warmuth, M.K. (1994). The weighted majority algorithm.
  *Information and Computation* 108(2):212–261.
- Freund, Y. & Schapire, R.E. (1997). A decision-theoretic generalization of
  on-line learning and an application to boosting. *Journal of Computer and
  System Sciences* 55(1):119–139. (Hedge algorithm.)
- Herbster, M. & Warmuth, M.K. (1998). Tracking the best expert.
  *Machine Learning* 32:151–178. (Fixed-share.)
- de Rooij, S., van Erven, T., Grünwald, P. & Koolen, W. (2014). Follow the
  leader if you can, hedge if you must. *Journal of Machine Learning
  Research* 15:1281–1316. (Hedge η tuning.)

Composition notes
-----------------
- ``research`` vincentization / ensemble families: static combination; this
  module is the online regret-bounded complement.
- ``models.ensemble``: Bayesian stacking rather than regret-guaranteed
  exponential weighting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FORBIDDEN_KEYS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _check_losses(losses: object, name: str = "losses") -> tuple[FloatArray, int, int]:
    arr = np.asarray(losses, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be a (T, K) array with T, K >= 2")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    if (arr < 0).any():
        raise ValueError(f"{name} must be nonnegative (proper losses)")
    return arr, int(arr.shape[0]), int(arr.shape[1])


def _normalize(v: FloatArray) -> FloatArray:
    s = float(v.sum())
    if s <= 0 or not np.isfinite(s):
        return np.full(v.size, 1.0 / v.size)
    return v / s


def hedge_weights(
    losses: object,
    eta: float | None = None,
) -> dict[str, FloatArray]:
    """Hedge / EWA weight path over experts.

    ``w[t+1,i] ∝ w[t,i] exp(-η l[t,i])`` with ``w[0]`` uniform. For losses
    in ``[0,1]``, regret ≤ ``η·T/8 + ln K/η``; the default
    ``η = sqrt(8 ln K / T)`` balances the two terms (de Rooij et al. 2014).

    Returns ``weights`` (T, K) played before each loss, ``cum_regret`` (T,)
    vs the best fixed expert, and ``algo_loss`` per period.
    """
    arr, n_t, n_k = _check_losses(losses)
    if eta is None:
        eta = float(np.sqrt(8.0 * np.log(n_k) / n_t))
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be a positive float")

    w = np.full(n_k, 1.0 / n_k)
    weights = np.empty((n_t, n_k))
    algo_loss = np.empty(n_t)
    cum_expert = np.zeros(n_k)
    for t in range(n_t):
        weights[t] = w
        l_t = arr[t]
        algo_loss[t] = w @ l_t
        cum_expert += l_t
        w = _normalize(w * np.exp(-eta * l_t))
    cum_regret = np.cumsum(algo_loss) - np.cumsum(arr, axis=0).min(axis=1)
    return {
        "weights": weights,
        "algo_loss": algo_loss,
        "cum_regret": cum_regret,
        "eta": np.array(eta),
        "bound": np.array(eta * n_t / 8.0 + np.log(n_k) / eta),
    }


def fixed_share_weights(
    losses: object,
    eta: float | None = None,
    alpha: float = 0.01,
) -> dict[str, FloatArray]:
    """Herbster-Warmuth fixed-share tracking (uniform-share variant).

    Loss update ``v_i ∝ w_i exp(-η l_i)`` then mixing
    ``w_i ← (1−α) v_i + α/K``: at rate α every expert receives a floor of
    fresh mass, so the algorithm tracks the best *switching* sequence of
    experts rather than the best fixed one.
    """
    arr, n_t, n_k = _check_losses(losses)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if eta is None:
        eta = float(np.sqrt(8.0 * np.log(n_k) / n_t))
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be a positive float")

    w = np.full(n_k, 1.0 / n_k)
    weights = np.empty((n_t, n_k))
    algo_loss = np.empty(n_t)
    for t in range(n_t):
        weights[t] = w
        l_t = arr[t]
        algo_loss[t] = w @ l_t
        v = _normalize(w * np.exp(-eta * l_t))
        w = (1.0 - alpha) * v + alpha / n_k
    # best-switching baseline via short DP over last-break positions is O(T^2 K);
    # report regret vs best fixed expert plus tracking quality separately.
    cum_regret = np.cumsum(algo_loss) - np.cumsum(arr, axis=0).min(axis=1)
    return {
        "weights": weights,
        "algo_loss": algo_loss,
        "cum_regret": cum_regret,
        "eta": np.array(eta),
        "alpha": np.array(alpha),
    }


def specialist_weights(
    losses: object,
    awake_mask: object,
    eta: float | None = None,
) -> dict[str, FloatArray]:
    """Sleeping / specialist experts: only awake experts are played.

    ``awake_mask[t,i]`` nonzero marks expert ``i`` as active at period ``t``;
    inactive experts get zero weight and are not updated. A period with no
    awake expert plays uniform over all experts (no loss-free pass).
    """
    arr, n_t, n_k = _check_losses(losses)
    mask = np.asarray(awake_mask)
    if mask.shape != arr.shape:
        raise ValueError("awake_mask must match losses shape (T, K)")
    awake = mask.astype(bool)
    if eta is None:
        eta = float(np.sqrt(8.0 * np.log(n_k) / n_t))
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be a positive float")

    logw = np.zeros(n_k)
    weights = np.empty((n_t, n_k))
    algo_loss = np.empty(n_t)
    for t in range(n_t):
        a_t = awake[t]
        if not a_t.any():
            a_t = np.ones(n_k, dtype=bool)
        w = np.zeros(n_k)
        sub = logw[a_t] - logw[a_t].max()
        w[a_t] = np.exp(sub) / np.exp(sub).sum()
        weights[t] = w
        l_t = arr[t]
        algo_loss[t] = w @ l_t
        logw[a_t] -= eta * l_t[a_t]
    # best *awake-weighted* expert: cum loss restricted to awake periods
    cum_awake = (arr * awake).sum(axis=0) / np.maximum(awake.sum(axis=0), 1)
    cum_regret = np.cumsum(algo_loss) - np.arange(1, n_t + 1) * cum_awake.min()
    return {
        "weights": weights,
        "algo_loss": algo_loss,
        "cum_regret": cum_regret,
        "eta": np.array(eta),
    }


def expert_fit(
    forecasts: object,
    realized: object,
    loss: str = "squared",
    eta: float | None = None,
    tau: float = 0.5,
) -> dict[str, FloatArray]:
    """Aggregate a forecaster panel under a proper loss.

    ``forecasts`` is (T, K): expert ``k``'s prediction at ``t``; ``realized``
    is (T,). ``loss``: ``'squared'`` (proper for the mean) or ``'pinball'``
    (proper for the ``tau``-quantile, 0 < tau < 1). Pinball losses are
    rescaled to [0, 1] per realized range so η tuning stays valid.
    """
    fc = np.asarray(forecasts, dtype=np.float64)
    y = np.asarray(realized, dtype=np.float64)
    if fc.ndim != 2 or fc.shape[0] < 2 or fc.shape[1] < 2:
        raise ValueError("forecasts must be (T, K) with T, K >= 2")
    if y.shape != (fc.shape[0],) or not np.isfinite(y).all() or not np.isfinite(fc).all():
        raise ValueError("realized must be finite (T,) matching forecasts")
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")

    err = y[:, None] - fc
    if loss == "squared":
        l_mat = err**2
        rng = float(np.ptp(l_mat))
        if rng > 0:
            l_mat = l_mat / rng
    elif loss == "pinball":
        l_mat = np.maximum(tau * err, (tau - 1.0) * err)
        rng = float(np.ptp(l_mat))
        if rng > 0:
            l_mat = l_mat / rng
    else:
        raise ValueError(f"loss must be 'squared' or 'pinball', got {loss!r}")

    out = hedge_weights(l_mat, eta=eta)
    out["loss_matrix"] = l_mat
    return out


def regret_decomp(losses: object, weights: object) -> dict[str, FloatArray]:
    """Per-period regret series and best-fixed-expert identity path."""
    arr, n_t, n_k = _check_losses(losses)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != arr.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be a finite nonnegative (T, K) array")
    row = w.sum(axis=1)
    if (row <= 0).any():
        raise ValueError("each weights row must have positive mass")
    w = w / row[:, None]

    algo_loss = np.einsum("tk,tk->t", w, arr)
    cum_expert = np.cumsum(arr, axis=0)
    best_id = np.argmin(cum_expert, axis=1)
    cum_regret = np.cumsum(algo_loss) - cum_expert[np.arange(n_t), best_id]
    return {
        "algo_loss": algo_loss,
        "cum_regret": cum_regret,
        "best_expert_id": best_id.astype(np.float64),
        "cum_expert_losses": cum_expert,
    }


def synth_experts(
    n_obs: int,
    n_experts: int,
    regime_breaks: int,
    seed: int = 0,
    sep: float = 1.5,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Synthetic expert panel with switching best expert.

    Regime ``r`` (of ``n_experts`` possible) makes expert ``r`` the accurate
    forecaster (small noise) while the rest are biased/noisy. Returns
    ``(realized (T,), forecasts (T,K), truth_regime (T,))`` — the truth is the
    index of the best expert per period.
    """
    if n_obs < 10 or n_experts < 2 or regime_breaks < 1 or regime_breaks > n_obs - 2:
        raise ValueError("need n_obs >= 10, n_experts >= 2, 1 <= breaks < n_obs-1")
    if not np.isfinite(sep) or sep <= 0:
        raise ValueError("sep must be positive")

    rng = np.random.default_rng(seed)
    n_t = int(n_obs)
    n_k = int(n_experts)
    edges = np.linspace(0, n_t, regime_breaks + 1).astype(int)
    regime_seq = rng.choice(n_k, size=regime_breaks, replace=False)
    truth = np.zeros(n_t, dtype=int)
    for i in range(regime_breaks):
        truth[edges[i] : edges[i + 1]] = regime_seq[i]

    signal = np.cumsum(rng.normal(0.0, 1.0, n_t)) * 0.1
    forecasts = np.empty((n_t, n_k))
    for j in range(n_k):
        good = truth == j
        noise = np.where(good, rng.normal(0.0, 0.15, n_t), rng.normal(sep, 0.5, n_t))
        forecasts[:, j] = signal + noise
    realized = signal + rng.normal(0.0, 0.15, n_t)
    return realized, forecasts, truth.astype(np.float64)


def _switch_lag(weights_path: FloatArray, truth: FloatArray) -> float:
    """Mean lag for the argmax weight to reach the new best expert post-break."""
    arg_w = np.argmax(weights_path, axis=1)
    lags: list[float] = []
    breaks = np.nonzero(np.diff(truth))[0]
    for b in breaks:
        tgt = int(truth[b + 1])
        hits = np.nonzero(arg_w[b + 1 :] == tgt)[0]
        lags.append(float(hits[0]) if hits.size else float(truth.size - b))
    return float(np.mean(lags)) if lags else 0.0


def bench_expert_aggregation(seed: int = 0) -> dict[str, float]:
    """SYNTHETIC bench for expert-advice aggregation (correctness only)."""
    y, fc, truth = synth_experts(240, 5, 4, seed=seed)

    # squared-loss panel (forecast error), normalized to [0,1]
    l_mat = (y[:, None] - fc) ** 2
    l_mat /= max(float(np.ptp(l_mat)), 1e-12)

    hedge = hedge_weights(l_mat)
    fs = fixed_share_weights(l_mat, alpha=0.02)

    # specialists: experts awake in a rotating mask
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(l_mat.shape) < 0.6
    mask[:, 0] = True  # always one awake
    spec = specialist_weights(l_mat, mask)

    dec = regret_decomp(l_mat, hedge["weights"])

    # regret ratio: cum regret vs cum best-fixed loss
    best_cum = float(np.cumsum(l_mat, axis=0)[-1].min())
    hedge_ratio = float(hedge["cum_regret"][-1] / max(best_cum, 1e-12))
    fs_ratio = float(fs["cum_regret"][-1] / max(best_cum, 1e-12))
    spec_ratio = float(spec["cum_regret"][-1] / (spec["algo_loss"].size))

    # fixed-share beats static under switching: lower final algo loss
    beats = float(fs["algo_loss"].sum() < hedge["algo_loss"].sum())

    # weight mass on never-best experts (truth coverage)
    best_set = set(np.unique(truth).astype(int))
    never_best = [j for j in range(fc.shape[1]) if j not in best_set]
    bad_mass = float(fs["weights"][-1, never_best].sum()) if never_best else 0.0

    pin = expert_fit(fc, y, loss="pinball", tau=0.9)
    sq = expert_fit(fc, y, loss="squared")

    det = hedge_weights(l_mat)
    determinism = float(np.array_equal(det["weights"], hedge["weights"]))

    bound_ok = float(hedge["cum_regret"][-1] <= hedge["bound"] + 1e-9)

    blob: dict[str, float] = {
        "synthetic_hedge_regret_ratio": hedge_ratio,
        "synthetic_fixedshare_regret_ratio": fs_ratio,
        "synthetic_fixedshare_beats_static": beats,
        "synthetic_switch_detect_lag": _switch_lag(fs["weights"], truth),
        "synthetic_specialist_regret_per_period": spec_ratio,
        "synthetic_bad_expert_mass": bad_mass,
        "synthetic_eta_bound_respected": bound_ok,
        "synthetic_pinball_regret_ratio": float(
            pin["cum_regret"][-1]
            / max(float(np.cumsum(pin["loss_matrix"], axis=0)[-1].min()), 1e-12)
        ),
        "synthetic_squared_regret_ratio": float(
            sq["cum_regret"][-1] / max(float(np.cumsum(sq["loss_matrix"], axis=0)[-1].min()), 1e-12)
        ),
        "synthetic_best_expert_path_span": float(np.unique(dec["best_expert_id"]).size),
        "synthetic_determinism": determinism,
    }
    for k in blob:
        if FORBIDDEN_KEYS.intersection(k.split("_")):
            raise ValueError(f"forbidden bench key {k!r}")
    return blob
