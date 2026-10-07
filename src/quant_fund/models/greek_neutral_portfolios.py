"""Greek-neutral option portfolios: hedging as a training inductive bias.

Research-grade minimal implementation of the learning framework of

- Tan, W.L., Roberts, S. & Zohren, S. (2026), "Taming the Greeks: Option
  Portfolios with Inductive Biases", arXiv:2609.33767 (q-fin.PM; cs.LG;
  q-fin.CP), https://doi.org/10.48550/arXiv.2609.33767, submitted 27 Sep
  2026. End-to-end deep learning for systematic options trading that embeds
  hedging behavior in the training objective: ``L(theta) = J(theta) +
  alpha * ||grad_x V||`` (their eq. (6)) couples a performance-driven loss
  ``J`` with a differentiable risk-sensitivity penalty on the portfolio
  value ``V`` w.r.t. selected risk factors ``x``, enforcing neutrality to
  the chosen risk dimension. The paper instantiates the framework on static
  delta-neutral straddle portfolios with the penalty directed at first-order
  directional (delta) exposure, and evaluates two scale-invariant penalty
  variants: an exposure-normalized penalty (ENP, their eqs. (10)-(11)) and a
  Greek-ratio drift penalty (DP, their eqs. (12)-(13)). Their empirical
  finding on Nasdaq-100 options: appropriately calibrated regularization
  simultaneously improves out-of-sample risk-adjusted performance relative
  to the unregularized baseline while reducing realized directional
  exposure, with performance degrading again once alpha is pushed past a
  variant-specific threshold (their §6.3).

What is implemented here, keyed to the paper:

1. *Static delta-neutral straddle universe* (their §3–§4.1, eqs. (3)–(4)):
   :func:`build_straddle_universe` pairs a call and a put per (strike,
   tenor) with inception weights ``w_C = -Delta_P / (Delta_C - Delta_P)``,
   ``w_P = Delta_C / (Delta_C - Delta_P)`` so each straddle is delta-neutral
   at formation (their eq. (4)); the weights are held fixed until expiry
   (static hedge at initiation only, no subsequent adjustment). Inception
   Greeks come from :func:`quant_fund.models.options.bs_greeks` (Black, F. &
   Scholes, M. 1973, Journal of Political Economy 81(3), 637–654; Merton,
   R.C. 1973, Bell Journal of Economics 4(1), 141–183); pathwise marking
   uses a vectorized BSM twin pinned to ``bs_greeks``/``bs_price`` in tests.
2. *Performance-driven objective* (their §5.2): ``J =
   -sqrt(periods_per_year) * E_Omega[R~] / sqrt(Var_Omega[R~])`` over the
   pooled scaled per-straddle returns ``R~_{i,t+1} = X_{i,t} (sigma*/
   sigma_{i,t}) R_{i,t+1}`` (their eqs. (1)–(2)) — the negative annualized
   Sharpe ratio (Sharpe, W.F. 1994, Journal of Portfolio Management 21(1),
   49–58) of the pooled observations. It is used here strictly as an
   internal training loss and is surfaced only under ``sim_internal_*``
   keys, never as a headline metric (AGENTS.md honesty contract). The
   volatility-target scaling ``sigma*/sigma_{i,t}`` (Moskowitz, T.J., Ooi,
   Y.H. & Pedersen, L.H. 2012, Journal of Financial Economics 104(2),
   228–250) uses their 20-day EWMA of straddle returns when ``vol_target``
   is given, and is off (scales = 1) by default.
3. *Risk-sensitivity penalties* (their §5.3, eqs. (9)–(13)):
   :func:`greek_neutrality_penalty` (differentiable torch) and
   :func:`greek_neutrality_penalty_np` (numpy evaluation twin) implement the
   naive L1 penalty (their eq. (9); degenerate — it admits "signal
   shrinkage", the trivial solution of scaling all positions down), the
   exposure-normalized penalties ENP-L1/L2 (their eqs. (10)–(11), gross
   directional exposure per unit of gross allocation) and the Greek-ratio
   drift penalties DP-L1/L2 (their eqs. (12)–(13), directional exposure per
   unit of realized convexity — a structural risk weight that discourages
   holding contracts whose Delta dominates their Gamma, i.e. that have
   drifted away from the at-the-money region). ENP/DP are invariant to a
   uniform rescaling ``X -> cX`` — the paper's fix to shrinkage; tests pin
   the invariance. The framework is pricing-model agnostic: the exposure
   Greek is selectable (``delta`` — the paper's instantiation — or ``vega``)
   and only enters through the penalty norm.
4. *Training* (their §5.4): :func:`train_greek_neutral_portfolio` minimizes
   ``J + alpha * penalty`` by full-batch Adam (Kingma, D.P. & Ba, J. 2014,
   arXiv:1412.6980). The paper uses minibatch Adam with validation-based
   early stopping; full-batch fixed-epoch training is used here for
   determinism, matching the sibling :mod:`quant_fund.models.deep_hedging`
   convention. Signals are bounded ``X in [-1, 1]`` by a tanh head (their
   admissible-signal range; the bound also rules out leveraged blow-ups).
5. *Realized exposure diagnostics* (their §6.2, eqs. (14)–(15)):
   :func:`evaluate_portfolio` reports the net position-normalized delta
   ``sum_i X_i Delta_i / sum_i |X_i|`` (aggregate tilt; reducible by
   cross-sectional netting) and the gross position-normalized delta
   ``sum_i |X_i Delta_i| / sum_i |X_i|`` (stricter, position-level measure
   with no cancellation), pooled over the path ensemble. Both are invariant
   to a uniform rescaling of all positions.
6. *Penalty-strength sweep* (their §6.3): :func:`penalty_strength_sweep`
   retrains across an ``alpha`` grid and records the paper's central
   trade-off — realized directional exposure falls with ``alpha`` while
   out-of-sample risk-adjusted performance holds or improves up to a
   calibrated ``alpha`` (interior optimum) and degrades once the penalty
   overwhelms the performance objective.

Honesty: everything here runs on SYNTHETIC seeded path ensembles (GBM or
two-state Markov-regime-switch GBM, composed from
:mod:`quant_fund.models.deep_hedging`; Hamilton, J.D. 1989, Econometrica
57(2), 357–384) marked under a flat BSM implied vol. Outputs are
algorithmic-correctness evidence, never market evidence. The Sharpe-like
training objective and mean-return quantities are simulator-internal
training signals reported only under ``sim_internal_*`` keys (never
headlined); the headline-eligible diagnostics are risk-sensitivity (delta
exposure) quantities. No live-trading claims — no broker connectivity
exists in this repo.

Documented simplifications vs the paper: static per-contract signals ``X_i``
(a tanh-bounded parameter per straddle) instead of their LSTM ``X_{i,t} =
f_theta(u_{i,t})`` over time-varying features — the objective and penalty
machinery over the pooled index set Omega is identical; one seeded training
run per alpha instead of multi-seed aggregation; fixed epochs instead of
early stopping; optional vol targeting (default off) instead of their
always-on 15% annualized target; a synthetic BSM-marked universe instead of
OptionMetrics Nasdaq-100 data (their framework is model-agnostic — BSM
Greeks are the practical default, composed from
:mod:`quant_fund.models.options`).

Conventions: torch is the optional ``nn`` extra and is imported lazily via
:func:`_torch` (mirrors ``deep_hedging``), so this module imports cleanly
without torch and every torch entry point raises ``ImportError`` with
install guidance. Numpy core for universe construction, marking, penalties,
objectives, and exposure evaluation. Fail-closed edges: ``ValueError`` on
non-positive or non-finite inputs, empty or mismatched shapes, unknown
penalty variants or exposure Greeks, drift penalties without a denominator
Greek, non-increasing alpha grids, degenerate return variance, zero gross
allocation, and non-positive straddle values. Training is CPU
single-thread full-batch Adam and deterministic given ``seed`` (GPU
determinism is not claimed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.models.deep_hedging import simulate_gbm_paths, simulate_regime_switch_paths
from quant_fund.models.options import bs_greeks

Array = NDArray[np.float64]

__all__ = [
    "GreekNeutralPortfolioResult",
    "PenaltySweepResult",
    "StraddleBookMarks",
    "StraddleUniverse",
    "build_straddle_universe",
    "evaluate_portfolio",
    "greek_neutrality_penalty",
    "greek_neutrality_penalty_np",
    "mark_straddle_book",
    "penalty_strength_sweep",
    "performance_objective",
    "portfolio_returns",
    "scaled_straddle_returns",
    "simulate_path_ensemble",
    "train_greek_neutral_portfolio",
]

# Penalty variants of Tan, Roberts & Zohren (2026), eqs. (9)-(13).
_PENALTY_VARIANTS = ("naive_l1", "enp_l1", "enp_l2", "dp_l1", "dp_l2")
_EXPOSURE_GREEKS = ("delta", "gamma", "vega")
_PATH_ENSEMBLE_KINDS = ("gbm", "regime_switch")
# Numerical floor inside the torch variance sqrt so a degenerate X ~ 0 does
# not NaN the gradient; the numpy twin fails closed instead (var <= 0).
_VAR_FLOOR = 1e-12


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "greek-neutral portfolios need the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


def _check_count(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return int(value)


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}")
    return v


def _check_nonnegative(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite; got {value!r}")
    return v


def _check_finite(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return v


def _as_paths(paths: Array | Sequence[Sequence[float]], *, name: str = "paths") -> Array:
    """Validate and normalize simulated price paths to float64 (n_paths, n_steps+1)."""
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array of shape (n_paths, n_steps+1); got ndim={arr.ndim}"
        )
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one path; got n_paths={arr.shape[0]}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"{name} must have at least two time points (n_steps >= 1); got {arr.shape[1]}"
        )
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} entries must be finite (NaN/inf rejected)")
    if not bool(np.all(arr > 0.0)):
        raise ValueError(f"{name} must be strictly positive (price paths)")
    return arr


def _as_positive_vector(x: Array | Sequence[float], name: str) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(v))) or not bool(np.all(v > 0.0)):
        raise ValueError(f"{name} must be strictly positive and finite")
    return v


def _check_variant(variant: str) -> str:
    if variant not in _PENALTY_VARIANTS:
        raise ValueError(
            f"unknown penalty variant {variant!r}; expected one of {_PENALTY_VARIANTS}"
        )
    return variant


def _check_eps(eps: float) -> float:
    e = float(eps)
    if not math.isfinite(e) or e <= 0.0:
        raise ValueError(f"eps must be positive and finite (numerical stabilizer); got {eps!r}")
    return e


def simulate_path_ensemble(
    n_paths: int,
    n_steps: int,
    *,
    kind: str = "gbm",
    seed: int = 0,
    **params: Any,
) -> Array:
    """Seeded SYNTHETIC underlying-price path ensemble (n_paths, n_steps + 1).

    Thin composition over :mod:`quant_fund.models.deep_hedging`: ``kind='gbm'``
    delegates to :func:`~quant_fund.models.deep_hedging.simulate_gbm_paths`,
    ``kind='regime_switch'`` to
    :func:`~quant_fund.models.deep_hedging.simulate_regime_switch_paths`
    (keyword ``params`` are forwarded: ``s0``, ``mu``, ``sigma``, ``dt``, ...
    — heterogeneous across the two simulators, e.g. ``initial_regime`` is an
    ``int``, the rest ``float``; the callees validate each value).
    SYNTHETIC data: correctness material, never market evidence.
    """
    n = _check_count(n_paths, "n_paths")
    m = _check_count(n_steps, "n_steps")
    s = int(seed)
    if kind == "gbm":
        return simulate_gbm_paths(n, m, seed=s, **params)
    if kind == "regime_switch":
        return simulate_regime_switch_paths(n, m, seed=s, **params)
    raise ValueError(f"unknown path ensemble kind {kind!r}; expected one of {_PATH_ENSEMBLE_KINDS}")


@dataclass(frozen=True)
class StraddleUniverse:
    """Static delta-neutral straddle universe (Tan et al. 2026, eqs. (3)–(4)).

    One straddle per (strike, tenor) pair: long ``call_weights[j]`` calls and
    ``put_weights[j]`` puts of strike ``strikes[j]`` and maturity
    ``maturities[j]``, normalized so the pair is delta-neutral at inception
    under the flat implied vol ``sigma`` at spot ``s0`` (weights sum to 1 and
    are held fixed until expiry — static hedge at initiation only).
    """

    strikes: Array  # (n_opts,)
    maturities: Array  # (n_opts,) years to expiry at inception
    call_weights: Array  # (n_opts,) eq. (4): -Delta_P / (Delta_C - Delta_P)
    put_weights: Array  # (n_opts,) eq. (4): Delta_C / (Delta_C - Delta_P)
    s0: float
    sigma: float  # flat BSM implied vol used for construction and marking
    r: float

    @property
    def n_opts(self) -> int:
        return int(self.strikes.shape[0])


def build_straddle_universe(
    strikes: Array | Sequence[float],
    maturities: Array | Sequence[float],
    *,
    s0: float = 100.0,
    sigma: float = 0.2,
    r: float = 0.0,
) -> StraddleUniverse:
    """Seed a straddle universe over the (strike x tenor) grid.

    For every (K, T) pair the inception call/put weights follow the paper's
    eq. (4) from the BSM inception deltas of
    :func:`quant_fund.models.options.bs_greeks`, so each straddle starts
    delta-neutral (their §3 ATM-straddle selection targets the same
    near-zero-inception-delta structure). SYNTHETIC construction on a flat
    implied vol: correctness material, never market evidence.
    """
    spot = _check_positive(s0, "s0")
    sig = _check_positive(sigma, "sigma")
    rate = _check_finite(r, "r")
    k = _as_positive_vector(strikes, "strikes")
    t = _as_positive_vector(maturities, "maturities")
    ks, ts = np.meshgrid(k, t, indexing="ij")
    flat_k = ks.reshape(-1)
    flat_t = ts.reshape(-1)
    wc = np.empty(flat_k.size, dtype=float)
    wp = np.empty(flat_k.size, dtype=float)
    for j in range(flat_k.size):
        gc = bs_greeks(spot, float(flat_k[j]), float(flat_t[j]), sig, rate, call=True)
        gp = bs_greeks(spot, float(flat_k[j]), float(flat_t[j]), sig, rate, call=False)
        denom = gc["delta"] - gp["delta"]
        if abs(denom) < 1e-12:
            raise ValueError(
                f"degenerate straddle weights: call/put delta spread {denom!r} at "
                f"K={flat_k[j]}, T={flat_t[j]}"
            )
        wc[j] = -gp["delta"] / denom
        wp[j] = gc["delta"] / denom
    if not bool(np.all(np.isfinite(wc))) or not bool(np.all(np.isfinite(wp))):
        raise ValueError("straddle inception weights must be finite")
    return StraddleUniverse(
        strikes=flat_k,
        maturities=flat_t,
        call_weights=wc,
        put_weights=wp,
        s0=spot,
        sigma=sig,
        r=rate,
    )


def _bs_call_put(
    s: Array, k: float, tau: Array, sigma: float, r: float
) -> tuple[Array, Array, Array, Array, Array]:
    """Vectorized BSM twin of options.bs_price/bs_greeks (parity pinned in tests).

    Returns ``(call_price, put_price, call_delta, put_delta, greeks)`` with
    ``greeks = stack([gamma, vega])``, each broadcast to the shape of ``s``;
    requires strictly positive ``tau``.
    """
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * tau) / sq
    d2 = d1 - sq
    pdf1 = np.asarray(norm.pdf(d1), dtype=float)
    df = np.exp(-r * tau)
    cdf1 = np.asarray(norm.cdf(d1), dtype=float)
    cdf2 = np.asarray(norm.cdf(d2), dtype=float)
    c = s * cdf1 - k * df * cdf2
    p = k * df * np.asarray(norm.cdf(-d2), dtype=float) - s * np.asarray(norm.cdf(-d1), dtype=float)
    gamma = pdf1 / (s * sq)
    vega = s * pdf1 * np.sqrt(tau)
    return c, p, cdf1, cdf1 - 1.0, np.stack([gamma, vega])


@dataclass(frozen=True)
class StraddleBookMarks:
    """Pathwise marks of a straddle universe over a simulated ensemble.

    Shapes: ``values`` is ``(n_opts, n_paths, n_steps + 1)`` (straddle mark
    ``V_{i,p,k}`` of their eq. (3) at grid times ``t_k = k * dt``);
    ``returns`` is ``(n_opts, n_paths, n_steps)`` with ``R_{i,p,k+1} =
    (V_{k+1} - V_k) / V_k`` (their eq. (2)); ``delta``/``gamma``/``vega`` are
    straddle-level Greeks ``(n_opts, n_paths, n_steps)`` at the decision
    times ``t_0..t_{n_steps-1}`` (the ``Delta_{i,t}``, ``Gamma_{i,t}`` of
    their eqs. (9)–(15)); ``scales`` is the vol-target factor
    ``sigma*/sigma_{i,t}`` of their eq. (1) (all ones when vol targeting is
    off). Expired contracts (``tau <= 0``) are marked at weighted intrinsic
    value with intrinsic delta and zero gamma/vega. SYNTHETIC marks on a
    flat implied vol: correctness material, never market evidence.
    """

    values: Array
    returns: Array
    delta: Array
    gamma: Array
    vega: Array
    scales: Array
    dt: float
    periods_per_year: float

    @property
    def n_opts(self) -> int:
        return int(self.delta.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        """(n_opts, n_paths, n_steps) of the decision-time tensors."""
        return (int(self.delta.shape[0]), int(self.delta.shape[1]), int(self.delta.shape[2]))


def _ewma_vol_scales(returns: Array, *, dt: float, vol_target: float, ewma_span: int) -> Array:
    """Vol-target scales ``sigma*/sigma_{i,t}`` (their eq. (1), 20-day EWMA).

    ``sigma_{i,p,k}`` is the exponentially weighted moving standard deviation
    of the straddle returns strictly before the interval ``[t_k, t_{k+1}]``
    (no lookahead); at ``k = 0`` no history exists and the scale is 1.0.
    ``sigma*`` is the annualized ``vol_target`` converted to a per-step
    target. Fail-closed on degenerate (non-positive) EWMA variance.
    """
    decay = 2.0 / (float(ewma_span) + 1.0)
    n_steps = returns.shape[2]
    var = np.zeros(returns.shape, dtype=float)
    if n_steps > 1:
        var[..., 1] = returns[..., 0] ** 2
        for k in range(2, n_steps):
            var[..., k] = (1.0 - decay) * var[..., k - 1] + decay * returns[..., k - 1] ** 2
    if n_steps > 1 and not bool(np.all(var[..., 1:] > 0.0)):
        raise ValueError(
            "degenerate EWMA variance while vol targeting: straddle returns must be "
            "non-degenerate along each path (vol_target requires varying marks)"
        )
    per_step = vol_target * math.sqrt(dt)
    scales = np.ones(returns.shape, dtype=float)
    if n_steps > 1:
        scales[..., 1:] = per_step / np.sqrt(var[..., 1:])
    return scales


def mark_straddle_book(
    universe: StraddleUniverse,
    paths: Array | Sequence[Sequence[float]],
    *,
    dt: float = 1.0 / 252.0,
    vol_target: float | None = None,
    ewma_span: int = 20,
    periods_per_year: float = 252.0,
) -> StraddleBookMarks:
    """Mark the straddle universe along a SYNTHETIC path ensemble (numpy core).

    Values ``V_{i,p,k}`` are the static-weight call+put marks of their eq.
    (3) under the flat implied vol ``universe.sigma``; returns follow their
    eq. (2); straddle Greeks at decision times are the weight-combined leg
    Greeks (the paper's dataset Greeks are BSM-twin quantities here — their
    framework is pricing-model agnostic). ``vol_target`` (annualized) turns
    on the ``sigma*/sigma_{i,t}`` scaling of their eq. (1) with an
    ``ewma_span``-step EWMA of straddle returns. Fail-closed on non-positive
    straddle marks (degenerate payoff at expiry), non-finite outputs, and
    invalid parameters.
    """
    if not isinstance(universe, StraddleUniverse):
        raise ValueError("universe must be a StraddleUniverse (see build_straddle_universe)")
    arr = _as_paths(paths)
    step = _check_positive(dt, "dt")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    n_paths, n_marks = arr.shape
    n_steps = n_marks - 1
    vt: float | None = None
    if vol_target is not None:
        vt = _check_positive(vol_target, "vol_target")
        _check_count(ewma_span, "ewma_span")
    t_grid = np.arange(n_marks, dtype=float) * step  # (n_marks,)
    n_opts = universe.n_opts
    values = np.empty((n_opts, n_paths, n_marks), dtype=float)
    delta = np.empty((n_opts, n_paths, n_steps), dtype=float)
    gamma = np.empty((n_opts, n_paths, n_steps), dtype=float)
    vega = np.empty((n_opts, n_paths, n_steps), dtype=float)
    sig = universe.sigma
    rate = universe.r
    for j in range(n_opts):
        k_j = float(universe.strikes[j])
        wc = float(universe.call_weights[j])
        wp = float(universe.put_weights[j])
        tau_marks = universe.maturities[j] - t_grid  # (n_marks,)
        live = tau_marks > 0.0
        v_j = np.empty((n_paths, n_marks), dtype=float)
        if bool(np.any(live)):
            tau_pos = np.maximum(tau_marks[live], 1e-12)
            c, p, _, _, _gv = _bs_call_put(arr[:, live], k_j, tau_pos[None, :], sig, rate)
            v_j[:, live] = wc * c + wp * p
        if bool(np.any(~live)):
            expired = arr[:, ~live]
            v_j[:, ~live] = wc * np.maximum(expired - k_j, 0.0) + wp * np.maximum(
                k_j - expired, 0.0
            )
        values[j] = v_j
        # Decision-time Greeks at t_0..t_{n_steps-1}.
        tau_dec = universe.maturities[j] - t_grid[:-1]
        live_d = tau_dec > 0.0
        d_j = np.zeros((n_paths, n_steps), dtype=float)
        g_j = np.zeros((n_paths, n_steps), dtype=float)
        v_j_greek = np.zeros((n_paths, n_steps), dtype=float)
        if bool(np.any(live_d)):
            s_live = arr[:, :-1][:, live_d]
            tau_pos = np.maximum(tau_dec[live_d], 1e-12)[None, :]
            _, _, dc, dp, gv = _bs_call_put(s_live, k_j, tau_pos, sig, rate)
            d_j[:, live_d] = wc * dc + wp * dp
            g_j[:, live_d] = gv[0]
            v_j_greek[:, live_d] = gv[1]
        if bool(np.any(~live_d)):
            s_exp = arr[:, :-1][:, ~live_d]
            d_j[:, ~live_d] = np.where(s_exp > k_j, wc, np.where(s_exp < k_j, -wp, 0.0))
        delta[j], gamma[j], vega[j] = d_j, g_j, v_j_greek
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("straddle marks must be finite (degenerate inputs rejected)")
    if not bool(np.all(values > 0.0)):
        raise ValueError(
            "straddle marks must be strictly positive (a zero-weighted intrinsic mark at "
            "expiry, S == K exactly, is degenerate — perturb the ensemble or the grid)"
        )
    returns = (values[:, :, 1:] - values[:, :, :-1]) / values[:, :, :-1]
    if not bool(np.all(np.isfinite(returns))):
        raise ValueError("straddle returns must be finite")
    if vt is None:
        scales = np.ones((n_opts, n_paths, n_steps), dtype=float)
    else:
        scales = _ewma_vol_scales(returns, dt=step, vol_target=vt, ewma_span=int(ewma_span))
    return StraddleBookMarks(
        values=values,
        returns=returns,
        delta=delta,
        gamma=gamma,
        vega=vega,
        scales=scales,
        dt=step,
        periods_per_year=ppy,
    )


def _check_weights(weights: Array | Sequence[float], n_opts: int) -> Array:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n_opts:
        raise ValueError(f"weights must have length n_opts={n_opts}; got {w.shape[0]}")
    if not bool(np.all(np.isfinite(w))):
        raise ValueError("weights must be finite (NaN/inf rejected)")
    return w


def _greek_matrix(marks: StraddleBookMarks, name: str) -> Array:
    if name not in _EXPOSURE_GREEKS:
        raise ValueError(f"unknown exposure greek {name!r}; expected one of {_EXPOSURE_GREEKS}")
    return {"delta": marks.delta, "gamma": marks.gamma, "vega": marks.vega}[name]


def _check_marks(marks: StraddleBookMarks, name: str = "marks") -> None:
    if not isinstance(marks, StraddleBookMarks):
        raise ValueError(f"{name} must be a StraddleBookMarks (see mark_straddle_book)")


def scaled_straddle_returns(marks: StraddleBookMarks, weights: Array | Sequence[float]) -> Array:
    """Pooled scaled per-straddle returns ``R~ = X_i * scale * R`` (their eqs. (1)–(2)).

    Returns an ``(n_opts, n_paths, n_steps)`` array — the observations the
    performance objective pools over (their index set Omega).
    """
    _check_marks(marks)
    w = _check_weights(weights, marks.n_opts)
    return w[:, None, None] * marks.scales * marks.returns


def portfolio_returns(marks: StraddleBookMarks, weights: Array | Sequence[float]) -> Array:
    """Portfolio return series ``R^Pi_{p,k+1}`` of their eq. (1) (n_paths, n_steps).

    The cross-sectional mean of the scaled per-straddle returns — a
    P&L-like SYNTHETIC quantity: report only under ``sim_internal_*`` keys.
    """
    return scaled_straddle_returns(marks, weights).mean(axis=0)


def performance_objective(scaled_returns: Array, *, periods_per_year: float = 252.0) -> float:
    """Negative annualized Sharpe of the pooled scaled returns (their §5.2).

    ``J = -sqrt(periods_per_year) * E_Omega[R~] / sqrt(Var_Omega[R~])`` — the
    paper's performance-driven training loss. Sharpe-like by construction, it
    is a simulator-internal training signal here (``sim_internal_*``
    namespacing downstream), never a headline research metric (AGENTS.md
    honesty contract). Fail-closed on fewer than two observations or
    degenerate (zero) variance.
    """
    x = np.asarray(scaled_returns, dtype=float).reshape(-1)
    ppy = _check_positive(periods_per_year, "periods_per_year")
    if x.size < 2:
        raise ValueError(
            f"performance objective needs at least two pooled observations; got {x.size}"
        )
    if not bool(np.all(np.isfinite(x))):
        raise ValueError("scaled returns must be finite (NaN/inf rejected)")
    v = float(x.var())
    if v <= 0.0:
        raise ValueError("degenerate return variance: performance objective undefined")
    return float(-math.sqrt(ppy) * x.mean() / math.sqrt(v))


def _validate_penalty_inputs(
    weights: Array, exposure: Array, variant: str, denominator: Array | None, eps: float
) -> float:
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be a 1-D vector over the option universe; got {weights.shape}"
        )
    if exposure.ndim < 2:
        raise ValueError(
            f"exposure must be broadcastable per contract with ndim >= 2 "
            f"(n_opts, n_paths, n_steps); got ndim={exposure.ndim}"
        )
    if exposure.shape[0] != weights.shape[0]:
        raise ValueError(
            f"exposure leading dim must match n_opts={weights.shape[0]}; got {exposure.shape[0]}"
        )
    if not bool(np.all(np.isfinite(weights))) or not bool(np.all(np.isfinite(exposure))):
        raise ValueError("weights and exposure must be finite (NaN/inf rejected)")
    if variant.startswith("dp"):
        if denominator is None:
            raise ValueError(f"drift penalty {variant!r} requires the denominator Greek (gamma)")
        if denominator.shape != exposure.shape:
            raise ValueError(
                f"denominator shape {denominator.shape} must match exposure shape {exposure.shape}"
            )
        if not bool(np.all(np.isfinite(denominator))):
            raise ValueError("denominator must be finite (NaN/inf rejected)")
    return _check_eps(eps)


def greek_neutrality_penalty_np(
    weights: Array | Sequence[float],
    exposure: Array,
    *,
    variant: str,
    denominator: Array | None = None,
    eps: float = 1e-8,
) -> float:
    """Numpy evaluation twin of :func:`greek_neutrality_penalty` (their eqs. (9)–(13)).

    ``variant='naive_l1'`` (eq. 9): ``sum |X_i g_i|`` — degenerate, admits
    signal shrinkage. ``'enp_l1'`` (eq. 10): ``sum |X_i g_i| / (sum_Omega
    |X_i| + eps)``. ``'enp_l2'`` (eq. 11): ``sum (X_i g_i)^2 / (sum_Omega
    X_i^2 + eps)``. ``'dp_l1'`` (eq. 12): ``sum |X_i g_i| / (sum |X_i
    Gamma_i| + eps)``. ``'dp_l2'`` (eq. 13): ``sum (X_i g_i)^2 / (sum (X_i
    Gamma_i)^2 + eps)``. ``exposure`` is the selected risk-sensitivity Greek
    ``(n_opts, n_paths, n_steps)`` (delta in the paper's instantiation; vega
    admissible under their general norm); ``denominator`` is Gamma for the
    drift variants. ENP/DP are invariant to ``X -> cX`` (c > 0).
    """
    v = _check_variant(variant)
    w = np.asarray(weights, dtype=float)
    g = np.asarray(exposure, dtype=float)
    den = None if denominator is None else np.asarray(denominator, dtype=float)
    e = _validate_penalty_inputs(w, g, v, den, eps)
    shape = (w.shape[0],) + (1,) * (g.ndim - 1)
    contrib = w.reshape(shape) * g
    if v == "naive_l1":
        return float(np.abs(contrib).sum())
    if v == "enp_l1":
        alloc = np.abs(np.broadcast_to(w.reshape(shape), g.shape)).sum()
        return float(np.abs(contrib).sum() / (alloc + e))
    if v == "enp_l2":
        alloc = np.broadcast_to((w * w).reshape(shape), g.shape).sum()
        return float((contrib * contrib).sum() / (alloc + e))
    if not (den is not None):
        raise ValueError("den is not None")  # validated above for dp_*
    den_contrib = w.reshape(shape) * den
    if v == "dp_l1":
        return float(np.abs(contrib).sum() / (np.abs(den_contrib).sum() + e))
    return float((contrib * contrib).sum() / ((den_contrib * den_contrib).sum() + e))


def greek_neutrality_penalty(
    weights: Any,
    exposure: Any,
    *,
    variant: str,
    denominator: Any = None,
    eps: float = 1e-8,
) -> Any:
    """Differentiable risk-sensitivity penalty (Tan et al. 2026, eqs. (9)–(13)).

    Torch mirror of :func:`greek_neutrality_penalty_np` with identical
    formulas and validation semantics: ``weights`` is the signal tensor
    ``(n_opts,)`` (gradient-carrying), ``exposure`` the selected Greek tensor
    ``(n_opts, n_paths, n_steps)`` (constant), and ``denominator`` the Gamma
    tensor required by the drift variants ``'dp_l1'``/``'dp_l2'``. The naive
    variant ``'naive_l1'`` (their eq. (9)) is included as the degenerate
    baseline their scale-invariant ENP/DP variants are designed to fix.
    Returns a scalar tensor; the penalty is differentiable w.r.t.
    ``weights`` so hedging behavior is embedded at the loss level.
    """
    torch = _torch()
    v = _check_variant(variant)
    e = _check_eps(eps)
    for nm, t in (("weights", weights), ("exposure", exposure), ("denominator", denominator)):
        if t is None or torch.is_tensor(t):
            continue
        raise ValueError(f"{nm} must be a torch tensor (numpy twin: greek_neutrality_penalty_np)")
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be a 1-D tensor over the option universe; got {weights.shape}"
        )
    if exposure.ndim < 2:
        raise ValueError(
            f"exposure must have ndim >= 2 (n_opts, n_paths, n_steps); got ndim={exposure.ndim}"
        )
    if int(exposure.shape[0]) != int(weights.shape[0]):
        raise ValueError(
            f"exposure leading dim must match n_opts={int(weights.shape[0])}; "
            f"got {int(exposure.shape[0])}"
        )
    if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(exposure).all()):
        raise ValueError("weights and exposure must be finite (NaN/inf rejected)")
    if v.startswith("dp"):
        if denominator is None:
            raise ValueError(f"drift penalty {v!r} requires the denominator Greek (gamma)")
        if tuple(denominator.shape) != tuple(exposure.shape):
            raise ValueError(
                f"denominator shape {tuple(denominator.shape)} must match exposure shape "
                f"{tuple(exposure.shape)}"
            )
        if not bool(torch.isfinite(denominator).all()):
            raise ValueError("denominator must be finite (NaN/inf rejected)")
    shape = (int(weights.shape[0]),) + (1,) * (int(exposure.ndim) - 1)
    contrib = weights.reshape(shape) * exposure
    if v == "naive_l1":
        return contrib.abs().sum()
    if v == "enp_l1":
        alloc = weights.reshape(shape).abs().expand(exposure.shape).sum()
        return contrib.abs().sum() / (alloc + e)
    if v == "enp_l2":
        w2 = (weights * weights).reshape(shape).expand(exposure.shape).sum()
        return (contrib * contrib).sum() / (w2 + e)
    den_contrib = weights.reshape(shape) * denominator
    if v == "dp_l1":
        return contrib.abs().sum() / (den_contrib.abs().sum() + e)
    return (contrib * contrib).sum() / ((den_contrib * den_contrib).sum() + e)


def _torch_performance_objective(torch: Any, scaled_t: Any, periods_per_year: float) -> Any:
    """Differentiable negative annualized Sharpe of pooled scaled returns (§5.2).

    Torch mirror of :func:`performance_objective`; the variance carries a
    ``_VAR_FLOOR`` numerical floor inside the sqrt so a degenerate ``X ~ 0``
    yields a finite loss/gradient instead of NaN (the numpy twin fails closed
    on zero variance instead).
    """
    flat = scaled_t.reshape(-1)
    mean = flat.mean()
    centered = flat - mean
    var = (centered * centered).mean()
    return -math.sqrt(periods_per_year) * mean / torch.sqrt(var + _VAR_FLOOR)


@dataclass(frozen=True)
class GreekNeutralPortfolioResult:
    """Learned static signal and training diagnostics (see :func:`train_greek_neutral_portfolio`).

    ``weights`` are the bounded signals ``X_i in [-1, 1]`` (their eq. (1)
    ``X_{i,t}``, static across time here); curves record the value before
    each update. ``final_performance`` is the Sharpe-like internal training
    loss component — a ``sim_internal`` quantity, never a headline metric.
    """

    weights: Array  # (n_opts,)
    alpha: float
    variant: str
    exposure_greek: str
    loss_curve: Array  # per-epoch total loss J + alpha * penalty
    performance_curve: Array
    penalty_curve: Array
    final_loss: float
    final_performance: float
    final_penalty: float
    seed: int
    epochs: int


def train_greek_neutral_portfolio(
    marks: StraddleBookMarks,
    *,
    alpha: float,
    variant: str,
    exposure_greek: str = "delta",
    eps: float = 1e-8,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    init_scale: float = 0.5,
) -> GreekNeutralPortfolioResult:
    """Train static straddle signals minimizing ``J + alpha * penalty`` (their eq. (6)).

    Full-batch Adam on the tanh-bounded per-contract signal ``X = tanh(theta)``;
    ``J`` is the pooled negative annualized Sharpe of their §5.2 and the
    penalty is :func:`greek_neutrality_penalty` on the selected exposure
    Greek (``delta`` — the paper's first-order directional instantiation — or
    ``vega``; the drift variants take Gamma as the structural denominator).
    ``alpha`` is their risk-aversion coefficient (``alpha = 0`` recovers the
    unregularized baseline). Deterministic given ``seed`` on CPU with a
    single thread; GPU determinism is not claimed.
    """
    torch = _torch()
    _check_marks(marks)
    a = _check_nonnegative(alpha, "alpha")
    v = _check_variant(variant)
    if exposure_greek not in _EXPOSURE_GREEKS:
        raise ValueError(
            f"unknown exposure greek {exposure_greek!r}; expected one of {_EXPOSURE_GREEKS}"
        )
    e = _check_eps(eps)
    n_epochs = _check_count(epochs, "epochs")
    rate = _check_positive(lr, "lr")
    scale0 = _check_positive(init_scale, "init_scale")
    exposure_np = _greek_matrix(marks, exposure_greek)
    denom_np = marks.gamma if v.startswith("dp") else None

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    scaled_const = torch.as_tensor(marks.scales * marks.returns, dtype=torch.float32)
    exposure_t = torch.as_tensor(exposure_np, dtype=torch.float32)
    denom_t = None if denom_np is None else torch.as_tensor(denom_np, dtype=torch.float32)
    theta = (scale0 * torch.randn(marks.n_opts, dtype=torch.float32)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=rate)
    loss_curve: list[float] = []
    perf_curve: list[float] = []
    pen_curve: list[float] = []
    for _ in range(n_epochs):
        opt.zero_grad(set_to_none=True)
        x = torch.tanh(theta)
        scaled_t = x.reshape(-1, 1, 1) * scaled_const
        perf_t = _torch_performance_objective(torch, scaled_t, marks.periods_per_year)
        pen_t = greek_neutrality_penalty(x, exposure_t, variant=v, denominator=denom_t, eps=e)
        total_t = perf_t + a * pen_t
        loss_curve.append(float(total_t.detach().numpy()))
        perf_curve.append(float(perf_t.detach().numpy()))
        pen_curve.append(float(pen_t.detach().numpy()))
        total_t.backward()
        opt.step()
    with torch.no_grad():
        x = torch.tanh(theta)
        scaled_t = x.reshape(-1, 1, 1) * scaled_const
        final_perf = float(
            _torch_performance_objective(torch, scaled_t, marks.periods_per_year).numpy()
        )
        final_pen = float(
            greek_neutrality_penalty(x, exposure_t, variant=v, denominator=denom_t, eps=e).numpy()
        )
        weights = np.asarray(x.numpy(), dtype=float)
    return GreekNeutralPortfolioResult(
        weights=weights,
        alpha=a,
        variant=v,
        exposure_greek=exposure_greek,
        loss_curve=np.asarray(loss_curve, dtype=float),
        performance_curve=np.asarray(perf_curve, dtype=float),
        penalty_curve=np.asarray(pen_curve, dtype=float),
        final_loss=final_perf + a * final_pen,
        final_performance=final_perf,
        final_penalty=final_pen,
        seed=int(seed),
        epochs=n_epochs,
    )


def evaluate_portfolio(
    marks: StraddleBookMarks,
    weights: Array | Sequence[float],
    *,
    variant: str | None = None,
    exposure_greek: str = "delta",
    eps: float = 1e-8,
) -> dict[str, float]:
    """Realized risk-sensitivity diagnostics of a static signal on an ensemble.

    Headline-eligible keys are the position-normalized delta exposures of
    their §6.2: ``gross_delta_exposure_mean`` (their eq. (15), the stricter
    position-level measure — no cancellation) and
    ``net_delta_exposure_mean`` / ``abs_net_delta_exposure_mean`` (their eq.
    (14), signed aggregate tilt and its per-observation magnitude). Both are
    invariant to a uniform rescaling of the signals. The Sharpe-like
    objective and mean portfolio return are simulator-internal training
    signals and are namespaced ``sim_internal_*`` (never headline metrics,
    AGENTS.md honesty contract). ``variant`` optionally adds the realized
    ``penalty_value``. SYNTHETIC diagnostics, never market evidence; no
    live-trading claims. Fail-closed on zero gross allocation (exposures
    undefined) and degenerate return variance.
    """
    _check_marks(marks)
    w = _check_weights(weights, marks.n_opts)
    gross_alloc = float(np.abs(w).sum())
    if gross_alloc <= 0.0:
        raise ValueError("weights must have positive gross allocation (exposures are normalized)")
    e_net = np.einsum("i,ipk->pk", w, marks.delta) / gross_alloc
    e_gross = np.einsum("i,ipk->pk", np.abs(w), np.abs(marks.delta)) / gross_alloc
    scaled = scaled_straddle_returns(marks, w)
    perf = performance_objective(scaled, periods_per_year=marks.periods_per_year)
    out = {
        "net_delta_exposure_mean": float(e_net.mean()),
        "abs_net_delta_exposure_mean": float(np.abs(e_net).mean()),
        "gross_delta_exposure_mean": float(e_gross.mean()),
        "gross_allocation": gross_alloc,
        "sim_internal_perf_objective": perf,
        "sim_internal_mean_portfolio_return": float(portfolio_returns(marks, w).mean()),
    }
    if variant is not None:
        v = _check_variant(variant)
        exposure = _greek_matrix(marks, exposure_greek)
        denom = marks.gamma if v.startswith("dp") else None
        out["penalty_value"] = greek_neutrality_penalty_np(
            w, exposure, variant=v, denominator=denom, eps=eps
        )
    return out


@dataclass(frozen=True)
class PenaltySweepResult:
    """Alpha-grid retraining sweep and the paper's §6.3 trade-off diagnostics.

    ``gross_delta_exposure_by_alpha`` / ``abs_net_delta_exposure_by_alpha``
    are the realized directional exposures on the evaluation ensemble (their
    eqs. (14)–(15)); ``sim_internal_perf_objective_by_alpha`` is the
    Sharpe-like OOS training-objective value — a simulator-internal signal
    under ``sim_internal_*`` namespacing, never a headline metric.
    ``best_index`` minimizes the objective (most negative = best
    risk-adjusted performance); ``interior_optimum`` is True when the best
    alpha is strictly inside the grid — the paper's central trade-off:
    exposure falls with alpha while performance holds-or-improves up to a
    calibrated alpha and degrades beyond it. SYNTHETIC correctness evidence
    on seeded ensembles, never market evidence.
    """

    variant: str
    exposure_greek: str
    alphas: Array  # (n_alphas,) increasing
    rows: list[dict[str, float]]  # per-alpha: alpha, train diagnostics, eval diagnostics
    weights_by_alpha: Array  # (n_alphas, n_opts)
    gross_delta_exposure_by_alpha: Array
    abs_net_delta_exposure_by_alpha: Array
    net_delta_exposure_by_alpha: Array
    sim_internal_perf_objective_by_alpha: Array
    best_index: int
    interior_optimum: bool
    seed: int
    epochs: int
    metrics: dict[str, float]


def penalty_strength_sweep(
    train_marks: StraddleBookMarks,
    eval_marks: StraddleBookMarks,
    *,
    alphas: Sequence[float],
    variant: str,
    exposure_greek: str = "delta",
    eps: float = 1e-8,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    init_scale: float = 0.5,
) -> PenaltySweepResult:
    """Retrain across an alpha grid; quantify the exposure/performance trade-off.

    Each alpha trains on ``train_marks`` (same seed — determinism over the
    paper's multi-seed aggregation) and is evaluated on the independent
    ``eval_marks`` ensemble (honest train/eval split). The paper's §6.3
    findings to reproduce on seeded SYNTHETIC ensembles: realized
    directional exposure falls (weakly monotonically) in alpha, and the OOS
    risk-adjusted objective has an interior optimum — holding or improving
    over the unregularized baseline (``alphas[0] = 0``) up to a calibrated
    alpha, then degrading as the penalty overwhelms the performance signal.
    Metric keys follow the honesty contract: exposure diagnostics are
    headline-eligible risk-sensitivity quantities (``gnp_*``); the
    Sharpe-like objective values are ``sim_internal_gnp_*`` training
    signals, never headline metrics.
    """
    _check_marks(train_marks, "train_marks")
    _check_marks(eval_marks, "eval_marks")
    v = _check_variant(variant)
    if train_marks.n_opts != eval_marks.n_opts:
        raise ValueError(
            f"train/eval marks must share n_opts; got {train_marks.n_opts} vs {eval_marks.n_opts}"
        )
    a_arr = np.asarray(alphas, dtype=float).reshape(-1)
    if a_arr.size < 3:
        raise ValueError(f"alphas must contain at least three strengths; got {a_arr.size}")
    if not bool(np.all(np.isfinite(a_arr))) or bool(np.any(a_arr < 0.0)):
        raise ValueError("alphas must be finite and non-negative")
    if not bool(np.all(np.diff(a_arr) > 0.0)):
        raise ValueError("alphas must be strictly increasing")

    rows: list[dict[str, float]] = []
    weights_rows: list[Array] = []
    for a in a_arr:
        res = train_greek_neutral_portfolio(
            train_marks,
            alpha=float(a),
            variant=v,
            exposure_greek=exposure_greek,
            eps=eps,
            epochs=epochs,
            lr=lr,
            seed=seed,
            init_scale=init_scale,
        )
        ev = evaluate_portfolio(
            eval_marks, res.weights, variant=v, exposure_greek=exposure_greek, eps=eps
        )
        rows.append(
            {
                "alpha": float(a),
                "train_final_loss": res.final_loss,
                "train_final_performance": res.final_performance,
                "train_final_penalty": res.final_penalty,
                **ev,
            }
        )
        weights_rows.append(res.weights)
    gross_by = np.asarray([r["gross_delta_exposure_mean"] for r in rows], dtype=float)
    absnet_by = np.asarray([r["abs_net_delta_exposure_mean"] for r in rows], dtype=float)
    net_by = np.asarray([r["net_delta_exposure_mean"] for r in rows], dtype=float)
    perf_by = np.asarray([r["sim_internal_perf_objective"] for r in rows], dtype=float)
    best = int(np.argmin(perf_by))
    interior = bool(0 < best < a_arr.size - 1)
    metrics = {
        "gnp_n_alphas": float(a_arr.size),
        "gnp_seed": float(int(seed)),
        "gnp_epochs": float(int(epochs)),
        "gnp_best_index": float(best),
        "gnp_best_alpha": float(a_arr[best]),
        "gnp_interior_optimum": 1.0 if interior else 0.0,
        "gnp_gross_delta_exposure_baseline": float(gross_by[0]),
        "gnp_gross_delta_exposure_best": float(gross_by[best]),
        "gnp_gross_delta_exposure_max_alpha": float(gross_by[-1]),
        "gnp_gross_delta_exposure_reduction": float(gross_by[0] - gross_by[best]),
        "gnp_abs_net_delta_exposure_baseline": float(absnet_by[0]),
        "gnp_abs_net_delta_exposure_best": float(absnet_by[best]),
        "gnp_net_delta_exposure_baseline": float(net_by[0]),
        "gnp_penalty_value_eval_best": float(rows[best]["penalty_value"]),
        "sim_internal_gnp_perf_objective_oos_baseline": float(perf_by[0]),
        "sim_internal_gnp_perf_objective_oos_best": float(perf_by[best]),
        "sim_internal_gnp_perf_objective_oos_max_alpha": float(perf_by[-1]),
        "sim_internal_gnp_perf_objective_gain_best_vs_baseline": float(perf_by[0] - perf_by[best]),
        "sim_internal_gnp_mean_return_oos_best": float(
            rows[best]["sim_internal_mean_portfolio_return"]
        ),
    }
    return PenaltySweepResult(
        variant=v,
        exposure_greek=exposure_greek,
        alphas=a_arr,
        rows=rows,
        weights_by_alpha=np.asarray(weights_rows, dtype=float),
        gross_delta_exposure_by_alpha=gross_by,
        abs_net_delta_exposure_by_alpha=absnet_by,
        net_delta_exposure_by_alpha=net_by,
        sim_internal_perf_objective_by_alpha=perf_by,
        best_index=best,
        interior_optimum=interior,
        seed=int(seed),
        epochs=int(epochs),
        metrics=metrics,
    )
