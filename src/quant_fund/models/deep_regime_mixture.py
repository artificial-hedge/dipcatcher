"""DeRegiME: deep regime mixture-of-experts head for probabilistic forecasting.

Research-grade minimal implementation of the predictive mechanism of:

- Wood, K., Zohren, S. & Roberts, S.J. (2026), "DeRegiME: Deep Regime
  Mixtures for Probabilistic Forecasting under Distribution Shift",
  arXiv:2605.19231 (submitted 19 May 2026, cs.LG),
  https://doi.org/10.48550/arXiv.2605.19231, code:
  https://github.com/kieranjwood/deregime.

The paper's decomposition (their Eq. 1) splits each forecast location
``xi = (x, t, d)`` into a shared conditional mean ``mu_theta(xi)``, a
structured residual ``delta_xi`` (a sparse variational GP under the
regime-mixing kernel of their Eq. 2), and regime-dependent noise
``eps_{r,xi}``: a shared neural gate ``pi_r(xi)`` softly assigns each location
to one of ``R_max`` candidate uncertainty regimes, and conditional on the
regime the noise is Student-t with regime scale ``sigma_r(xi)`` (their Eq. 7:
``sigma_r^2 = (c_d tau_r)^2 + v_res(xi) + sigma_floor^2``) and tail ``nu_r``.
The predictive density (their Eq. 3) marginalises the Gaussian residual
posterior ``q(delta) = N(m_delta, s_delta^2)`` and the regime label; the
expectation over ``delta`` is a Gauss-Hermite quadrature with ``Q = 20``
nodes (their Assumption 4 / Appendix B.6). Gate logits are mapped to the
simplex by a finite deterministic stick-breaking transform (their Eq. 5;
Sethuraman 1994, Ishwaran & James 2001): break fractions
``v_r = sigmoid(gamma_r)``, ``pi_r = v_r prod_{j<r}(1 - v_j)``, last stick
``pi_R = prod_j (1 - v_j)``. The effective regime count ``R_eff`` is the
number of regimes whose average gate mass exceeds ``1e-2`` (their §3,
"Finite gate and effective regimes").

Scope of this module (documented deviations from the paper):

- The regime-mixing kernel ``K_mix(xi, xi') = sum_r pi_r(xi) pi_r(xi')
  K_r(z_r(xi), z_r(xi'))`` (their Eq. 2, RBF base kernels with per-regime
  amplitude ``a_r`` and lengthscale ``ell_r``, their Eq. 6) is implemented in
  numpy (:func:`regime_mix_kernel`) and its positive semi-definiteness —
  their Proposition 1 (direct-sum feature map) and Theorem 2 — is a tested
  invariant, but the full sparse-variational-GP posterior (inducing
  variables + KL term, their Eq. 8) is NOT fitted here. Instead the marginal
  residual posterior ``q(delta_xi) = N(m_delta(xi), s_delta^2(xi))`` — the
  object that actually enters their predictive Eq. 3 — is produced directly
  by learned residual heads on the shared encoder. Training is therefore
  plain NLPD (log-score) maximum likelihood via Adam, not the SVGP ELBO.
- Single channel (``c_d = 1``); horizon-indexed gates and multi-horizon
  direct targets as in the paper; no RevIN (targets are standardised with
  train statistics instead — the same change-of-variables bookkeeping). The
  shared residual-variance correction ``v_res`` of their Eq. 7 is optional
  and defaults OFF: it is an add-on the paper itself ablates away, and with
  learned residual heads standing in for the SVGP a free ``v_res`` absorbs
  the heteroskedasticity that the regime scales ``tau_r`` should carry.
- The paper's headline benchmark compares against DeepAR/GluonTS-style
  dynamic Student-t heads (NLPD -20.3% across ten real datasets). The
  comparison here is a SYNTHETIC lane adaptation against
  :class:`~quant_fund.models.ngboost_lite.NGBoostGaussian` (Duan et al.
  2020, arXiv:1910.03225) on seeded regime-switching heteroskedastic
  streams — correctness evidence for the mechanism, not a reproduction of
  the paper's benchmark table, and never market evidence.

Propriety (their §4 "Proper density", Appendix B.2): with gates on the
simplex, proper Student-t/Gaussian components and a proper Gaussian
``q(delta)``, Tonelli's theorem gives ``int p(y|xi) dy = 1``; this is
asserted by quadrature in the tests, together with the proper-scoring
dominance of the true generator's NLPD over perturbed variants. Tails
(their Proposition 3): the mixture has polynomial tail rate
``Theta(|y|^{-(nu_eff + 1)})`` with ``nu_eff = min_r nu_r``.

Honesty: everything runs on SYNTHETIC simulated streams. Outputs are proper
scores (NLPD / log score) only — no Sharpe/Sortino/P&L headline, no
live-trading claims (AGENTS.md honesty contract).

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``deep_hedging.py``), so this module imports cleanly
without torch and every torch entry point raises ``ImportError`` with
install guidance. The gate map, mixture density, quadrature, kernel and all
scoring are numpy (usable without torch); torch is needed only to *fit* the
encoder/heads. Fail-closed edges: ``ValueError`` on non-finite inputs, shape
mismatches, ``n_regimes < 2``, unknown gate/family, non-positive scales or
lengthscales, gates off the simplex, and non-finite training loss. Training
is CPU single-thread full-batch AdamW with cosine LR annealing and
deterministic given ``seed`` (GPU
determinism is not claimed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.special import expit, gammaln, logsumexp

from quant_fund.models.ngboost_lite import NGBoostGaussian

Array = NDArray[np.float64]

__all__ = [
    "DeRegiMEResult",
    "RegimeMixturePrediction",
    "SyntheticRegimeBenchmark",
    "SyntheticRegimeStream",
    "effective_regime_count",
    "fit_deregime",
    "gate_weights",
    "mixture_log_density",
    "mixture_nlpd",
    "regime_mix_kernel",
    "simulate_regime_stream",
    "stick_break_weights",
    "student_t_logpdf",
    "synthetic_regime_benchmark",
]

#: Gauss-Hermite nodes used in the paper's reported runs (their §4).
GH_NODES_DEFAULT = 20
#: Average-gate-mass threshold defining R_eff (paper §3).
R_EFF_THRESHOLD = 1e-2
#: Observation-noise floor: sigma_floor^2 enters every regime scale (Eq. 7)
#: and bounds the mixture-MLE degenerate spike (a component scale collapsing
#: onto a data point has unbounded density; the floor keeps every component
#: proper with a finite score).
SIGMA_FLOOR_DEFAULT = 0.05

_GATES = ("stick_breaking", "softmax")
_FAMILIES = ("student_t", "gaussian")
_LOG_SCALE_CLIP = (-12.0, 12.0)


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "deep regime mixture needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


def _check_count(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}; got {value!r}")
    return int(value)


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}")
    return v


def _as_2d(x: Array | Sequence[Sequence[float]], *, name: str = "X") -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (n, p); got ndim={arr.ndim}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row; got {arr.shape[0]}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} entries must be finite (NaN/inf rejected)")
    return arr


def _np_softplus(x: Array) -> Array:
    """Numerically stable numpy softplus: log(1 + exp(x))."""
    return np.asarray(np.logaddexp(0.0, np.asarray(x, dtype=float)), dtype=float)


# ---------------------------------------------------------------------------
# Gate math (numpy; paper Eq. 5)
# ---------------------------------------------------------------------------


def stick_break_weights(logits: Array) -> Array:
    """Finite deterministic stick-breaking map (paper Eq. 5).

    Break fractions ``v_r = sigmoid(gamma_r)`` for ``r = 1..L`` map
    ``(.., L)`` logits to ``(.., L + 1)`` simplex weights:
    ``pi_r = v_r prod_{j<r}(1 - v_j)`` and ``pi_{L+1} = prod_j (1 - v_j)``.
    Sethuraman (1994) / Ishwaran & James (2001) truncation: finite logits
    give strictly positive weights (open simplex), so exact zero pruning is
    a limiting event — unused regimes are identified operationally through
    :func:`effective_regime_count` (paper §4, "Identifiability and finite
    sticks").
    """
    a = np.asarray(logits, dtype=float)
    if a.ndim < 1 or a.shape[-1] < 1:
        raise ValueError(f"logits must have a non-empty last axis; got shape {a.shape}")
    if not bool(np.all(np.isfinite(a))):
        raise ValueError("logits must be finite (NaN/inf rejected)")
    v = expit(a)
    log_remainder = np.cumsum(np.log1p(-v), axis=-1)  # log prod_{j<=r}(1 - v_j)
    prefix = np.concatenate(
        [np.zeros(a.shape[:-1] + (1,)), log_remainder[..., :-1]],
        axis=-1,
    )
    body = v * np.exp(prefix)
    tail = np.exp(log_remainder[..., -1:])
    return np.asarray(np.concatenate([body, tail], axis=-1), dtype=float)


def gate_weights(logits: Array, gate: str = "stick_breaking") -> Array:
    """Simplex gate weights from raw logits under the chosen parameterisation.

    ``'stick_breaking'`` (paper default): ``(.., R - 1)`` logits ->
    ``(.., R)`` weights via :func:`stick_break_weights`. ``'softmax'``:
    ``(.., R)`` logits -> ``(.., R)`` weights via a numerically stable
    softmax.
    """
    if gate not in _GATES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {_GATES}")
    a = np.asarray(logits, dtype=float)
    if not bool(np.all(np.isfinite(a))):
        raise ValueError("logits must be finite (NaN/inf rejected)")
    if gate == "stick_breaking":
        return stick_break_weights(a)
    return np.asarray(np.exp(a - logsumexp(a, axis=-1, keepdims=True)), dtype=float)


def effective_regime_count(gates: Array, *, threshold: float = R_EFF_THRESHOLD) -> int:
    """Effective regime count ``R_eff`` (paper §3 diagnostic).

    Number of regimes whose *average* gate mass over all forecast locations
    exceeds ``threshold`` (``1e-2`` in the paper). ``gates`` has shape
    ``(n_locations..., R)`` with non-negative finite entries.
    """
    g = np.asarray(gates, dtype=float)
    if g.ndim < 2:
        raise ValueError(f"gates must be at least 2-D with a trailing regime axis; got {g.shape}")
    if not bool(np.all(np.isfinite(g))) or bool(np.any(g < -1e-12)):
        raise ValueError("gates must be finite and non-negative")
    t = float(threshold)
    if not math.isfinite(t) or not (0.0 < t < 1.0):
        raise ValueError(f"threshold must be in (0, 1); got {threshold!r}")
    mass = g.reshape(-1, g.shape[-1]).mean(axis=0)
    return int(np.count_nonzero(mass > t))


# ---------------------------------------------------------------------------
# Regime-mixing kernel (numpy; paper Eq. 2 / Eq. 6, Proposition 1, Theorem 2)
# ---------------------------------------------------------------------------


def regime_mix_kernel(
    features: Array,
    gates: Array,
    *,
    amplitudes: Array,
    lengthscales: Array,
) -> Array:
    """Regime-mixing kernel Gram matrix ``K_mix`` (paper Eq. 2).

    ``K_mix(xi, xi') = sum_r pi_r(xi) pi_r(xi') K_r(z_r(xi), z_r(xi'))`` with
    RBF base kernels ``K_r(z, z') = a_r^2 exp(-||z - z'||^2 / (2 ell_r^2))``
    on per-regime features (paper Eq. 6). Inputs: ``features`` of shape
    ``(R, n, d)`` holding ``z_r`` per regime, ``gates`` of shape ``(n, R)``,
    per-regime ``amplitudes`` and ``lengthscales`` of shape ``(R,)``.

    By Proposition 1 the map ``Phi(xi) = (+)_r pi_r(xi) phi_r(z_r(xi))`` is
    an explicit direct-sum feature representation, so by Theorem 2 every
    Gram matrix is positive semi-definite for *any* real-valued gate
    function — gates need not lie on the simplex for kernel validity
    (non-negativity is only required for the density interpretation).
    """
    f = np.asarray(features, dtype=float)
    g = np.asarray(gates, dtype=float)
    a = np.asarray(amplitudes, dtype=float).reshape(-1)
    ell = np.asarray(lengthscales, dtype=float).reshape(-1)
    if f.ndim != 3:
        raise ValueError(f"features must be (R, n, d); got ndim={f.ndim}")
    n_r, n, d = f.shape
    if g.shape != (n, n_r):
        raise ValueError(f"gates must be (n, R)=({n}, {n_r}); got {g.shape}")
    if a.shape != (n_r,) or ell.shape != (n_r,):
        raise ValueError(f"amplitudes and lengthscales must both have shape ({n_r},)")
    for arr, nm in ((f, "features"), (g, "gates"), (a, "amplitudes"), (ell, "lengthscales")):
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"{nm} must be finite (NaN/inf rejected)")
    if bool(np.any(a <= 0.0)):
        raise ValueError("amplitudes must be strictly positive")
    if bool(np.any(ell <= 0.0)):
        raise ValueError("lengthscales must be strictly positive")
    k_mix = np.zeros((n, n), dtype=float)
    for r in range(n_r):
        z = f[r]
        sq = (z * z).sum(axis=1)[:, None] + (z * z).sum(axis=1)[None, :] - 2.0 * (z @ z.T)
        np.maximum(sq, 0.0, out=sq)
        k_r = float(a[r]) ** 2 * np.exp(-sq / (2.0 * float(ell[r]) ** 2))
        k_mix += np.outer(g[:, r], g[:, r]) * k_r
    return np.asarray(0.5 * (k_mix + k_mix.T), dtype=float)


# ---------------------------------------------------------------------------
# Predictive density (numpy; paper Eq. 3 + Gauss-Hermite, their §4)
# ---------------------------------------------------------------------------


def student_t_logpdf(y: Array, mu: Array, sigma: Array, nu: Array) -> Array:
    """Elementwise Student-t log density with scale ``sigma`` and tail ``nu``.

    ``log p = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 log(nu pi) - log sigma
    - ((nu+1)/2) log(1 + z^2 / nu)``, ``z = (y - mu) / sigma``. Broadcasts
    over all inputs; ``sigma > 0`` and ``nu > 0`` enforced (fail closed).
    """
    yy = np.asarray(y, dtype=float)
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    n = np.asarray(nu, dtype=float)
    if not (bool(np.all(np.isfinite(yy))) and bool(np.all(np.isfinite(m)))):
        raise ValueError("y and mu must be finite")
    if not bool(np.all(np.isfinite(s))) or bool(np.any(s <= 0.0)):
        raise ValueError("sigma must be strictly positive and finite")
    if not bool(np.all(np.isfinite(n))) or bool(np.any(n <= 0.0)):
        raise ValueError("nu must be strictly positive and finite")
    z = (yy - m) / s
    out = (
        gammaln((n + 1.0) / 2.0)
        - gammaln(n / 2.0)
        - 0.5 * np.log(n * math.pi)
        - np.log(s)
        - ((n + 1.0) / 2.0) * np.log1p(z * z / n)
    )
    return np.asarray(out, dtype=float)


def _gh_nodes(n_nodes: int) -> tuple[Array, Array]:
    """Gauss-Hermite nodes/weights normalised so weights sum to one.

    ``int N(delta; m, s^2) f(delta) d(delta) ~= sum_k w_k f(m + sqrt(2) s x_k)``
    with ``w_k = hermgauss weights / sqrt(pi)`` (paper: Q = 20 nodes, their
    Assumption 4 and Appendix B.6).
    """
    q = _check_count(n_nodes, "n_nodes", minimum=3)
    x, w = hermgauss(q)
    return np.asarray(x, dtype=float), np.asarray(w / math.sqrt(math.pi), dtype=float)


def _check_mixture_shapes(
    y: Array,
    mu: Array,
    delta_mean: Array,
    delta_var: Array,
    scales: Array,
    tails: Array,
    gates: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    yy = np.asarray(y, dtype=float)
    m = np.asarray(mu, dtype=float)
    dm = np.asarray(delta_mean, dtype=float)
    dv = np.asarray(delta_var, dtype=float)
    sc = np.asarray(scales, dtype=float)
    tl = np.asarray(tails, dtype=float).reshape(-1)
    g = np.asarray(gates, dtype=float)
    if yy.ndim != 2:
        raise ValueError(f"y must be 2-D (n, H); got ndim={yy.ndim}")
    n, h = yy.shape
    for arr, nm in ((m, "mu"), (dm, "delta_mean"), (dv, "delta_var")):
        if arr.shape != (n, h):
            raise ValueError(f"{nm} must have shape ({n}, {h}); got {arr.shape}")
    n_reg = g.shape[-1] if g.ndim == 3 else -1
    if g.ndim != 3 or g.shape != (n, h, n_reg):
        raise ValueError(f"gates must be (n, H, R); got {g.shape}")
    if sc.shape != (n, h, n_reg):
        raise ValueError(f"scales must be (n, H, R)={n, h, n_reg}; got {sc.shape}")
    if tl.shape != (n_reg,):
        raise ValueError(f"tails must have shape ({n_reg},); got {tl.shape}")
    for arr, nm in (
        (yy, "y"),
        (m, "mu"),
        (dm, "delta_mean"),
        (dv, "delta_var"),
        (sc, "scales"),
        (tl, "tails"),
        (g, "gates"),
    ):
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"{nm} must be finite (NaN/inf rejected)")
    if bool(np.any(dv < 0.0)):
        raise ValueError("delta_var must be non-negative")
    if bool(np.any(sc <= 0.0)):
        raise ValueError("scales must be strictly positive")
    if bool(np.any(tl <= 0.0)):
        raise ValueError("tails must be strictly positive")
    if bool(np.any(g < -1e-12)):
        raise ValueError("gates must be non-negative (simplex weights)")
    if not bool(np.all(np.abs(g.sum(axis=-1) - 1.0) <= 1e-6)):
        raise ValueError("gates must sum to 1 over the regime axis (simplex)")
    return yy, m, dm, dv, sc, tl, g


def mixture_log_density(
    y: Array,
    mu: Array,
    *,
    delta_mean: Array,
    delta_var: Array,
    scales: Array,
    tails: Array,
    gates: Array,
    family: str = "student_t",
    n_nodes: int = GH_NODES_DEFAULT,
) -> Array:
    """DeRegiME predictive log density per location (paper Eq. 3).

    ``log p(y|xi) = log int q(delta) sum_r pi_r p_r(y; mu + delta, sigma_r,
    nu_r) d(delta)`` where ``q(delta) = N(delta_mean, delta_var)`` is the
    marginal sparse-GP residual posterior (here: learned residual heads, see
    module docstring) and the delta-integral is the Q-node Gauss-Hermite
    quadrature of the paper's Assumption 4. ``family='gaussian'`` replaces
    the Student-t components with regime-scale Gaussians (the paper's
    Gaussian ablation, their Appendix A.5); ``tails`` is then ignored but
    must still be finite and positive.

    Shapes: ``y``/``mu``/``delta_mean``/``delta_var`` are ``(n, H)``;
    ``scales``/``gates`` are ``(n, H, R)``; ``tails`` is ``(R,)``. Gates are
    fail-closed checked to lie on the simplex — with proper components and a
    proper Gaussian ``q(delta)``, Tonelli's theorem (paper Appendix B.2)
    then guarantees the returned density integrates to one in ``y``.
    """
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {_FAMILIES}")
    yy, m, dm, dv, sc, tl, g = _check_mixture_shapes(
        y, mu, delta_mean, delta_var, scales, tails, gates
    )
    x, w = _gh_nodes(n_nodes)
    delta = dm[..., None] + np.sqrt(2.0 * dv)[..., None] * x  # (n, H, Q)
    resid = yy[..., None, None] - m[..., None, None] - delta[..., None]  # (n,H,Q,1)
    z = resid / sc[:, :, None, :]  # (n, H, Q, R)
    if family == "student_t":
        log_comp = (
            gammaln((tl + 1.0) / 2.0)
            - gammaln(tl / 2.0)
            - 0.5 * np.log(tl * math.pi)
            - np.log(sc)[:, :, None, :]
            - ((tl + 1.0) / 2.0) * np.log1p(z * z / tl)
        )
    else:
        log_comp = -0.5 * np.log(2.0 * math.pi) - np.log(sc)[:, :, None, :] - 0.5 * z * z
    log_pi = np.log(np.clip(g, 1e-300, None))[:, :, None, :]  # (n, H, 1, R)
    inner = logsumexp(log_pi + log_comp, axis=-1)  # (n, H, Q)
    total = logsumexp(inner + np.log(w)[None, None, :], axis=-1)  # (n, H)
    return np.asarray(total, dtype=float)


def mixture_nlpd(
    y: Array,
    mu: Array,
    *,
    delta_mean: Array,
    delta_var: Array,
    scales: Array,
    tails: Array,
    gates: Array,
    family: str = "student_t",
    n_nodes: int = GH_NODES_DEFAULT,
) -> float:
    """Mean negative log predictive density (lower is better; proper score).

    NLPD is the log score the paper headlines; it is a proper score, so the
    true generator attains the minimal expected value (AGENTS.md honesty
    contract: proper scores only).
    """
    log_p = mixture_log_density(
        y,
        mu,
        delta_mean=delta_mean,
        delta_var=delta_var,
        scales=scales,
        tails=tails,
        gates=gates,
        family=family,
        n_nodes=n_nodes,
    )
    return float(-np.mean(log_p))


# ---------------------------------------------------------------------------
# SYNTHETIC regime-switching stream (correctness material, never market data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticRegimeStream:
    """Seeded SYNTHETIC regime-switching heteroskedastic forecast problem.

    ``X`` holds causal encoder features (lags and realised-scale summaries of
    the history only — no lookahead), ``Y`` the direct multi-horizon targets
    ``y_{t+1..t+H}``, ``target_regimes`` the planted regime label of each
    target, and ``sigmas`` the planted per-regime noise scales.
    """

    y: Array  # (n_steps,) the full simulated series
    regimes: Array  # (n_steps,) int planted regime per step
    X: Array  # (n_obs, p) causal features
    Y: Array  # (n_obs, H) direct multi-horizon targets
    target_regimes: Array  # (n_obs, H) planted regime per target
    sigmas: Array  # (R_planted,) planted per-regime scales
    n_horizons: int


def simulate_regime_stream(
    n_steps: int,
    *,
    sigmas: Array | Sequence[float],
    n_horizons: int = 2,
    p_stay: float = 0.97,
    tail_nu: float | None = 6.0,
    window: int = 5,
    seed: int = 0,
) -> SyntheticRegimeStream:
    """Simulate a Markov-regime-switch heteroskedastic stream (SYNTHETIC).

    ``y_t = sigma_{r_t} eps_t`` with a symmetric ``R``-state Markov chain
    (stay probability ``p_stay``, uniform off-diagonal mass — Hamilton 1989
    style, as in ``deep_hedging.simulate_regime_switch_paths``) and i.i.d.
    unit-variance noise: standard normal, or Student-t with ``tail_nu > 2``
    rescaled to unit variance when ``tail_nu`` is given. Features per
    observation: ``window`` lagged values plus the rolling std and rolling
    mean-|y| over the same window — the residual-similarity information the
    paper's gate uses to cluster uncertainty regimes. Correctness material
    only, never market evidence.
    """
    sig = np.asarray(sigmas, dtype=float).reshape(-1)
    if sig.size < 2 or not bool(np.all(np.isfinite(sig))) or bool(np.any(sig <= 0.0)):
        raise ValueError(f"sigmas must be >= 2 strictly positive finite values; got {sigmas!r}")
    n = _check_count(n_steps, "n_steps", minimum=2)
    h = _check_count(n_horizons, "n_horizons")
    w = _check_count(window, "window", minimum=2)
    if n < w + h + 10:
        raise ValueError(
            f"n_steps must be >= window + n_horizons + 10 = {w + h + 10}; got {n_steps!r}"
        )
    p = float(p_stay)
    if not math.isfinite(p) or not (0.0 < p < 1.0):
        raise ValueError(f"p_stay must be a probability in (0, 1); got {p_stay!r}")
    nu: float | None = None
    if tail_nu is not None:
        nu = float(tail_nu)
        if not math.isfinite(nu) or nu <= 2.0:
            raise ValueError(f"tail_nu must be > 2 (finite variance) or None; got {tail_nu!r}")
    rng = np.random.default_rng(seed)
    n_reg = int(sig.size)
    y = np.empty(n, dtype=float)
    if nu is None:
        eps = rng.standard_normal(n)
    else:
        eps = rng.standard_t(nu, size=n) * math.sqrt((nu - 2.0) / nu)
    regimes = np.empty(n, dtype=int)
    regimes[0] = 0
    jumps = rng.random(n) >= p  # symmetric chain: leave -> uniform among others
    picks = rng.integers(0, n_reg - 1, size=n)
    for t in range(1, n):
        if jumps[t]:
            other = int(picks[t])
            regimes[t] = other if other < regimes[t - 1] else other + 1
        else:
            regimes[t] = int(regimes[t - 1])
    y = sig[regimes] * eps
    starts = np.arange(w - 1, n - h)
    n_obs = starts.size
    feats = np.empty((n_obs, w + 2), dtype=float)
    for i, t0 in enumerate(starts):
        hist = y[t0 - w + 1 : t0 + 1]
        feats[i, :w] = hist[::-1]
        feats[i, w] = float(np.std(hist))
        feats[i, w + 1] = float(np.mean(np.abs(hist)))
    targets = np.empty((n_obs, h), dtype=float)
    target_regimes = np.empty((n_obs, h), dtype=int)
    for j in range(h):
        targets[:, j] = y[starts + 1 + j]
        target_regimes[:, j] = regimes[starts + 1 + j]
    return SyntheticRegimeStream(
        y=y,
        regimes=regimes,
        X=feats,
        Y=targets,
        target_regimes=target_regimes,
        sigmas=sig,
        n_horizons=h,
    )


# ---------------------------------------------------------------------------
# Torch model (lazy; fitting only — scoring math above is pure numpy)
# ---------------------------------------------------------------------------


class _DeRegiMENet:
    """Encoder + head bundle (plain object so no torch at import time).

    Shared encoder ``h_psi(x)`` with lightweight linear heads as in the
    paper's §3: location head ``mu_theta``, gate-logit head ``gamma_omega``
    (horizon-indexed), residual heads for the marginal posterior
    ``q(delta) = N(m_delta, s_delta^2)`` (learned stand-in for the SVGP
    posterior — see module docstring), optional shared residual-variance
    head ``v_res`` (their Eq. 7; default OFF — their own ablation removes
    it, and with learned residual heads in place of the SVGP a free
    ``v_res`` absorbs the heteroskedasticity that the regime scales should
    carry), and per-regime GP feature head ``z_r`` with amplitude/lengthscale
    parameters for the regime-mixing kernel.
    """

    def __init__(
        self,
        torch: Any,
        *,
        n_features: int,
        hidden: Sequence[int],
        n_horizons: int,
        n_regimes: int,
        gate: str,
        family: str,
        nu_min: float,
        use_vres: bool = False,
    ) -> None:
        self.torch = torch
        self.n_regimes = int(n_regimes)
        self.n_horizons = int(n_horizons)
        self.gate = gate
        self.family = family
        self.nu_min = float(nu_min)
        self.use_vres = bool(use_vres)
        layers: list[Any] = []
        d = int(n_features)
        for h in hidden:
            layers += [torch.nn.Linear(d, int(h)), torch.nn.Tanh()]
            d = int(h)
        self.encoder = torch.nn.Sequential(*layers)
        self.mean_head = torch.nn.Linear(d, self.n_horizons)
        self.n_logits = self.n_regimes - 1 if gate == "stick_breaking" else self.n_regimes
        self.gate_head = torch.nn.Linear(d, self.n_horizons * self.n_logits)
        self.res_head = torch.nn.Linear(d, self.n_horizons)  # delta mean m_delta(xi)
        # Residual posterior variance s_delta^2: one global softplus parameter
        # per horizon (NOT location-dependent). In the paper s_delta^2(xi) is
        # the sparse-GP posterior variance — structured and kernel-smoothed;
        # a free per-location variance head would absorb the heteroskedastic
        # signal that the regime scales tau_r must carry, collapsing the gate.
        self.raw_sdelta = torch.nn.Parameter(torch.zeros(self.n_horizons))
        self.vres_head = torch.nn.Linear(d, self.n_horizons) if self.use_vres else None
        self.feat_head = torch.nn.Linear(d, self.n_regimes * 4)
        # Per-regime noise scale multipliers tau_r; log-space, symmetric
        # spread at init breaks the label symmetry (paper Appendix B.5).
        self.log_tau = torch.nn.Parameter(torch.linspace(-0.5, 0.5, self.n_regimes))
        self.raw_nu = torch.nn.Parameter(torch.zeros(self.n_regimes))
        self.log_amp = torch.nn.Parameter(torch.zeros(self.n_regimes))
        self.log_ell = torch.nn.Parameter(torch.zeros(self.n_regimes))

    def parameters(self) -> list[Any]:
        params: list[Any] = []
        modules: list[Any] = [
            self.encoder,
            self.mean_head,
            self.gate_head,
            self.res_head,
            self.feat_head,
        ]
        if self.vres_head is not None:
            modules.append(self.vres_head)
        for mod in modules:
            params.extend(mod.parameters())
        params += [self.log_tau, self.raw_nu, self.raw_sdelta, self.log_amp, self.log_ell]
        return params

    def head_outputs(self, x: Any) -> dict[str, Any]:
        """Raw head outputs for standardised features ``x`` (all torch)."""
        torch = self.torch
        h = self.encoder(x)
        n = int(h.shape[0])
        delta_var = torch.nn.functional.softplus(self.raw_sdelta).expand(n, self.n_horizons)
        out = {
            "mu": self.mean_head(h),
            "gate_logits": self.gate_head(h).view(n, self.n_horizons, self.n_logits),
            "delta_mean": self.res_head(h),
            "delta_var": delta_var,
            "z": self.feat_head(h).view(n, self.n_regimes, 4),
        }
        if self.vres_head is not None:
            out["vres"] = torch.nn.functional.softplus(self.vres_head(h))
        return out

    def tau(self) -> Any:
        return self.torch.exp(self.log_tau.clamp(*_LOG_SCALE_CLIP))

    def nu(self) -> Any:
        torch = self.torch
        return self.nu_min + torch.nn.functional.softplus(self.raw_nu)


def _torch_gate_log_weights(torch: Any, logits: Any, gate: str) -> Any:
    """Log simplex weights from gate logits (torch mirror of gate_weights)."""
    if gate == "softmax":
        return logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # Stick-breaking in log space: log pi_r = logsigmoid(g_r) + sum_{j<r} logsigmoid(-g_j).
    log_v = torch.nn.functional.logsigmoid(logits)
    log_not_v = torch.nn.functional.logsigmoid(-logits)
    cum = torch.cumsum(log_not_v, dim=-1)
    prefix = cum - log_not_v  # sum over j < r
    body = log_v + prefix
    tail = cum[..., -1:]
    return torch.cat([body, tail], dim=-1)


def _init_regimes(
    torch: Any, net: _DeRegiMENet, ys: Array, n_regimes: int, gate: str, n_horizons: int
) -> None:
    """Symmetry-breaking init for regime scales and gate bias.

    ``tau_r`` starts at quantile-matched noise scales of the standardised
    targets (the k-quantile init style of ``regime_switch.py``; paper
    Appendix B.5 symmetry breaking at initialisation), so regimes are
    specialised before gate logits can saturate — the MoE local minimum that
    otherwise collapses the stick-breaking gate onto regime 1. The
    stick-breaking gate bias is set to ``v_r = 1 / (R - r + 1)``, i.e.
    near-uniform initial gate mass over the truncation; the softmax gate
    starts uniform at zero bias.
    """
    abs_resid = np.abs(np.asarray(ys, dtype=float).ravel())
    qs = np.quantile(abs_resid, (np.arange(n_regimes) + 0.5) / n_regimes)
    scales = np.maximum(qs * math.sqrt(math.pi / 2.0), 1e-3)
    with torch.no_grad():
        net.log_tau.copy_(torch.as_tensor(np.log(scales), dtype=torch.float32))
        if gate == "stick_breaking":
            v = 1.0 / (n_regimes - np.arange(n_regimes - 1, dtype=float))
            gamma = np.log(v / (1.0 - v))
            net.gate_head.bias.copy_(
                torch.as_tensor(np.tile(gamma, n_horizons), dtype=torch.float32)
            )


def _torch_nlpd(
    torch: Any,
    net: _DeRegiMENet,
    out: dict[str, Any],
    y_std_space: Any,
    *,
    nodes_t: Any,
    logw_t: Any,
    sigma_floor: float,
    gate_penalty: float = 0.0,
) -> Any:
    """Differentiable mean NLPD in STANDARDISED target space.

    Torch mirror of :func:`mixture_log_density`: all heads and targets live
    in the standardised coordinate system, so the density here is the
    standardised-space one. The original-space NLPD adds the constant
    change-of-variables term ``mean(log y_std)`` (paper Appendix A.2 in
    spirit: the density transformation under instance normalisation); the
    constant does not affect gradients, and held-out scoring goes through
    the numpy path (:meth:`DeRegiMEResult.predict`) which applies it.

    ``gate_penalty`` adds the weak Dirichlet-style simplex regulariser of the
    paper's Appendix A.3: ``lambda * mean_n sum_r log pi_r(n)`` with
    ``lambda > 0`` mimics a Dirichlet concentration ``alpha < 1`` (negative
    log prior ``propto (1 - alpha) sum_r log pi_r``), which pushes vanishing
    gate mass onto unused regimes and guides stick-breaking pruning. The log
    weights are clamped at ``-30`` so the penalty stays finite.
    """
    tau = net.tau()
    sigma2 = tau.pow(2.0).view(1, 1, -1) + sigma_floor**2
    if "vres" in out:
        sigma2 = sigma2 + out["vres"].unsqueeze(-1)
    sigma = sigma2.clamp_min(1e-24).sqrt()  # (n, H, R) standardised scale
    delta = (
        out["delta_mean"].unsqueeze(-1) + torch.sqrt(2.0 * out["delta_var"]).unsqueeze(-1) * nodes_t
    )  # (n, H, Q)
    log_pi = _torch_gate_log_weights(torch, out["gate_logits"], net.gate)  # (n,H,R)
    z = (
        y_std_space[:, :, None, None] - out["mu"][:, :, None, None] - delta[..., None]
    ) / sigma.unsqueeze(2)  # (n, H, Q, R)
    log_sigma = torch.log(sigma).unsqueeze(2)  # (n, H, 1, R)
    if net.family == "student_t":
        nu = net.nu().view(1, 1, -1)
        log_comp = (
            torch.lgamma((nu + 1.0) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * torch.log(nu * math.pi)
            - log_sigma
            - ((nu + 1.0) / 2.0) * torch.log1p(z * z / nu)
        )
    else:
        log_comp = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * z * z
    inner = torch.logsumexp(log_pi.unsqueeze(2) + log_comp, dim=-1)  # (n, H, Q)
    log_p = torch.logsumexp(inner + logw_t, dim=-1)  # (n, H)
    nlpd = -log_p.mean()
    if gate_penalty > 0.0:
        nlpd = nlpd + gate_penalty * log_pi.clamp_min(-30.0).sum(dim=-1).mean()
    return nlpd


def _prediction_from_raw(
    raw: dict[str, Array],
    *,
    y_stats: tuple[Array, Array],
    log_tau: Array,
    raw_nu: Array,
    n_regimes: int,
    gate: str,
    family: str,
    sigma_floor: float,
    nu_min: float,
    n_nodes: int,
) -> RegimeMixturePrediction:
    """Assemble the original-scale numpy prediction from raw head outputs.

    Raw outputs live in standardised space; the gate simplex map, softplus
    variance links, regime scales (paper Eq. 7) and the change-of-variables
    rescaling are all numpy — torch is not involved beyond the head forward.
    """
    y_mean, y_std = y_stats
    gates = gate_weights(raw["gate_logits"], gate)  # numpy simplex map (Eq. 5)
    tau = np.exp(np.clip(log_tau, *_LOG_SCALE_CLIP))
    vres = raw["vres"] if "vres" in raw else np.zeros_like(raw["mu"])
    sigma2_n = tau[None, None, :] ** 2 + vres[..., None] + sigma_floor**2
    scales = np.sqrt(np.maximum(sigma2_n, 1e-24)) * y_std[None, :, None]
    if family == "student_t":
        tails = nu_min + _np_softplus(raw_nu)
    else:
        tails = np.ones(n_regimes, dtype=float)
    return RegimeMixturePrediction(
        mu=raw["mu"] * y_std[None, :] + y_mean[None, :],
        delta_mean=raw["delta_mean"] * y_std[None, :],
        delta_var=raw["delta_var"] * y_std[None, :] ** 2,
        scales=np.asarray(scales, dtype=float),
        tails=np.asarray(tails, dtype=float),
        gates=np.asarray(gates, dtype=float),
        family=family,
        n_nodes=n_nodes,
    )


def _raw_head_outputs(torch: Any, net: _DeRegiMENet, xs: Array) -> dict[str, Array]:
    """Float64 numpy raw head outputs at standardised features ``xs``."""
    torch.set_num_threads(1)
    x_t = torch.as_tensor(xs, dtype=torch.float32)
    with torch.no_grad():
        out = net.head_outputs(x_t)
        keys = ("mu", "gate_logits", "delta_mean", "delta_var", "vres", "z")
        return {key: np.asarray(out[key].numpy(), dtype=float) for key in keys if key in out}


@dataclass(frozen=True)
class RegimeMixturePrediction:
    """Numpy predictive state of a fitted DeRegiME at forecast locations.

    Original-scale arrays: ``mu``/``delta_mean``/``delta_var`` are ``(n, H)``,
    ``scales``/``gates`` are ``(n, H, R)`` (``scales`` is ``sigma_r(xi)`` of
    paper Eq. 7 including ``v_res`` and the floor), ``tails`` is ``(R,)``
    (``nu_r``; ones for ``family='gaussian'`` — ignored). Gate/mixture/
    quadrature math is numpy, so scoring works without torch.
    """

    mu: Array
    delta_mean: Array
    delta_var: Array
    scales: Array
    tails: Array
    gates: Array
    family: str
    n_nodes: int

    def log_density(self, y: Array) -> Array:
        """Per-location predictive log density (paper Eq. 3) on targets ``y``."""
        return mixture_log_density(
            y,
            self.mu,
            delta_mean=self.delta_mean,
            delta_var=self.delta_var,
            scales=self.scales,
            tails=self.tails,
            gates=self.gates,
            family=self.family,
            n_nodes=self.n_nodes,
        )

    def nlpd(self, y: Array) -> float:
        """Mean NLPD (proper log score; lower is better)."""
        return float(-np.mean(self.log_density(y)))

    def effective_regimes(self, *, threshold: float = R_EFF_THRESHOLD) -> int:
        """Effective regime count ``R_eff`` from these locations' gate mass."""
        return effective_regime_count(self.gates, threshold=threshold)

    def mean_regime_scales(self) -> Array:
        """Average predictive scale ``sigma_r`` per regime over all locations."""
        return np.asarray(self.scales.reshape(-1, self.scales.shape[-1]).mean(axis=0), dtype=float)

    def regime_masses(self) -> Array:
        """Average gate mass per regime over all locations (R_eff input)."""
        return np.asarray(self.gates.reshape(-1, self.gates.shape[-1]).mean(axis=0), dtype=float)


@dataclass(frozen=True)
class DeRegiMEResult:
    """Fitted DeRegiME head: torch net plus numpy parameter snapshot."""

    net: Any = field(repr=False, compare=False)
    x_mean: Array
    x_std: Array
    y_mean: Array
    y_std: Array
    log_tau: Array  # (R,) regime scale multipliers, log space
    raw_nu: Array  # (R,) raw tail parameters (nu = nu_min + softplus(raw))
    n_regimes: int
    n_horizons: int
    gate: str
    family: str
    sigma_floor: float
    nu_min: float
    gate_penalty: float
    n_nodes: int
    seed: int
    epochs: int
    loss_curve: Array  # per-epoch training NLPD, float32 torch trace (before update)
    nlpd_train: float  # final training NLPD via the float64 numpy predict path

    def predict(self, x: Array | Sequence[Sequence[float]]) -> RegimeMixturePrediction:
        """Original-scale predictive state at new locations (eval, no grad).

        The torch pass produces raw head outputs only; the gate simplex map,
        softplus variance links and all density math are numpy.
        """
        xx = _as_2d(x, name="X")
        if xx.shape[1] != self.x_mean.size:
            raise ValueError(
                f"X must have {self.x_mean.size} features to match the fitted "
                f"model; got {xx.shape[1]}"
            )
        torch = _torch()
        raw = _raw_head_outputs(torch, self.net, (xx - self.x_mean) / self.x_std)
        return _prediction_from_raw(
            raw,
            y_stats=(self.y_mean, self.y_std),
            log_tau=self.log_tau,
            raw_nu=self.raw_nu,
            n_regimes=self.n_regimes,
            gate=self.gate,
            family=self.family,
            sigma_floor=self.sigma_floor,
            nu_min=self.nu_min,
            n_nodes=self.n_nodes,
        )

    def nlpd(self, x: Array, y: Array) -> float:
        """Mean held-out NLPD (proper score) on original-scale targets."""
        return self.predict(x).nlpd(np.asarray(y, dtype=float))

    def mix_kernel(self, x: Array | Sequence[Sequence[float]], *, horizon: int = 0) -> Array:
        """Learned regime-mixing kernel Gram matrix ``K_mix`` (paper Eq. 2).

        Builds per-regime deep features ``z_r`` from the expert head, the
        horizon-``horizon`` gate slice ``pi_r`` and the per-regime
        amplitude/lengthscale parameters, then evaluates
        :func:`regime_mix_kernel`. By the paper's Proposition 1 / Theorem 2
        the result is positive semi-definite for any real gate values.
        """
        xx = _as_2d(x, name="X")
        if xx.shape[1] != self.x_mean.size:
            raise ValueError(
                f"X must have {self.x_mean.size} features to match the fitted "
                f"model; got {xx.shape[1]}"
            )
        hh = int(horizon)
        if hh < 0 or hh >= self.n_horizons:
            raise ValueError(f"horizon must be in [0, {self.n_horizons - 1}]; got {horizon!r}")
        torch = _torch()
        raw = _raw_head_outputs(torch, self.net, (xx - self.x_mean) / self.x_std)
        feats = np.transpose(raw["z"], (1, 0, 2))  # (R, n, 4)
        gates = gate_weights(raw["gate_logits"], self.gate)[:, hh, :]  # (n, R)
        with torch.no_grad():
            amps = np.asarray(torch.exp(self.net.log_amp).numpy(), dtype=float)
            ells = np.asarray(torch.exp(self.net.log_ell).numpy(), dtype=float)
        return regime_mix_kernel(feats, gates, amplitudes=amps, lengthscales=ells)

    def effective_regimes(self, x: Array, *, threshold: float = R_EFF_THRESHOLD) -> int:
        """Effective regime count ``R_eff`` on the given locations."""
        return self.predict(x).effective_regimes(threshold=threshold)


def fit_deregime(
    x: Array | Sequence[Sequence[float]],
    y: Array | Sequence[Sequence[float]],
    *,
    n_regimes: int = 4,
    gate: str = "stick_breaking",
    family: str = "student_t",
    hidden: Sequence[int] = (32, 32),
    epochs: int = 400,
    lr: float = 5e-3,
    weight_decay: float = 1e-2,
    n_nodes: int = GH_NODES_DEFAULT,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
    nu_min: float = 2.0,
    gate_penalty: float = 0.0,
    vres: bool = False,
    seed: int = 0,
) -> DeRegiMEResult:
    """Fit the DeRegiME regime-mixture head by AdamW on the NLPD log score.

    Weight decay regularises the encoder/mean/residual heads the way the
    paper's kernel-smoothed sparse-GP residual regularises ``delta``: it
    stops the location heads from memorising training residuals, which is
    the mixture-MLE degenerate-spike failure mode (a component scale
    collapsing onto a fitted point).

    Direct multi-horizon probabilistic regression of ``y`` ``(n, H)`` on
    encoder features ``x`` ``(n, p)`` (paper setup §3; the SVGP residual is
    replaced by learned residual heads — module docstring). Inputs and
    targets are standardised with train statistics; the predictive density is
    transformed back with the per-horizon Jacobian term. ``n_regimes`` is
    the truncation ``R_max``; the stick-breaking gate (their Eq. 5) plus the
    ``R_eff`` diagnostic (their §3) implement effective-regime pruning.
    ``family='gaussian'`` fits the paper's Gaussian likelihood ablation.
    ``gate_penalty`` is the weak Dirichlet-style simplex regulariser of their
    Appendix A.3 guiding stick-breaking pruning (see :func:`_torch_nlpd`);
    it defaults to zero — in this SVGP-free simplification even a weak
    penalty can collapse the gate onto one regime before the experts
    specialise, so pruning is left to the NLPD fit and monitored through
    ``R_eff``. ``vres=True`` re-enables their Eq. 7 shared residual-variance
    head (their ablation removes it; default off — see
    :class:`_DeRegiMENet`).
    ``nlpd_train`` is evaluated through the float64 numpy predict path, so it
    matches :meth:`DeRegiMEResult.nlpd` on the training batch exactly.
    Deterministic given ``seed`` on CPU single-thread. Fail-closed: diverged
    (non-finite) loss raises ``ValueError``.
    """
    torch = _torch()
    xx = _as_2d(x, name="X")
    yy = np.asarray(y, dtype=float)
    if yy.ndim != 2:
        raise ValueError(f"Y must be a 2-D array (n, H); got ndim={yy.ndim}")
    if yy.shape[0] != xx.shape[0]:
        raise ValueError(f"X and Y row mismatch: {xx.shape[0]} vs {yy.shape[0]}")
    if not bool(np.all(np.isfinite(yy))):
        raise ValueError("Y entries must be finite (NaN/inf rejected)")
    if gate not in _GATES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {_GATES}")
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {_FAMILIES}")
    r = _check_count(n_regimes, "n_regimes", minimum=2)
    n_epochs = _check_count(epochs, "epochs")
    _check_count(n_nodes, "n_nodes", minimum=3)
    rate = _check_positive(lr, "lr")
    floor = _check_positive(sigma_floor, "sigma_floor")
    nmin = _check_positive(nu_min, "nu_min")
    pen = float(gate_penalty)
    if not math.isfinite(pen) or pen < 0.0:
        raise ValueError(f"gate_penalty must be finite and >= 0; got {gate_penalty!r}")
    wd = float(weight_decay)
    if not math.isfinite(wd) or wd < 0.0:
        raise ValueError(f"weight_decay must be finite and >= 0; got {weight_decay!r}")
    widths = tuple(int(hh) for hh in hidden)
    if not widths or any(hh < 1 for hh in widths):
        raise ValueError(f"hidden must be a non-empty sequence of positive widths; got {hidden!r}")
    if yy.shape[0] < 2 * r:
        raise ValueError(f"need at least 2 * n_regimes = {2 * r} rows; got {yy.shape[0]}")

    hidden_widths = widths
    n_features = int(xx.shape[1])
    n_horizons = int(yy.shape[1])
    x_mean = xx.mean(axis=0)
    x_std = np.maximum(xx.std(axis=0), 1e-8)
    y_mean = yy.mean(axis=0)
    y_std = np.maximum(yy.std(axis=0), 1e-8)
    xs = (xx - x_mean) / x_std
    ys = (yy - y_mean) / y_std

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    net = _DeRegiMENet(
        torch,
        n_features=n_features,
        hidden=hidden_widths,
        n_horizons=n_horizons,
        n_regimes=r,
        gate=gate,
        family=family,
        nu_min=nmin,
        use_vres=bool(vres),
    )
    _init_regimes(torch, net, ys, r, gate, n_horizons)
    x_t = torch.as_tensor(xs, dtype=torch.float32)
    y_t = torch.as_tensor(ys, dtype=torch.float32)
    gh_x, gh_w = _gh_nodes(n_nodes)
    nodes_t = torch.as_tensor(gh_x, dtype=torch.float32).view(1, 1, -1)
    logw_t = torch.as_tensor(np.log(gh_w), dtype=torch.float32).view(1, 1, -1)
    # Constant change-of-variables term mapping standardised-space NLPD back
    # to the original target scale (mean over horizons of log y_std).
    jacobi = float(np.mean(np.log(y_std)))
    opt = torch.optim.AdamW(net.parameters(), lr=rate, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=rate / 10.0)
    curve: list[float] = []
    for _ in range(n_epochs):
        opt.zero_grad(set_to_none=True)
        out = net.head_outputs(x_t)
        loss = _torch_nlpd(
            torch,
            net,
            out,
            y_t,
            nodes_t=nodes_t,
            logw_t=logw_t,
            sigma_floor=floor,
            gate_penalty=pen,
        )
        val = float(loss.detach().numpy())
        if not math.isfinite(val):
            raise ValueError(f"training diverged at epoch {len(curve)}: NLPD={val!r}")
        curve.append(val + jacobi)
        loss.backward()
        opt.step()
        sched.step()
    log_tau_np = np.asarray(net.log_tau.detach().numpy(), dtype=float)
    raw_nu_np = np.asarray(net.raw_nu.detach().numpy(), dtype=float)
    raw = _raw_head_outputs(torch, net, xs)
    train_pred = _prediction_from_raw(
        raw,
        y_stats=(y_mean, y_std),
        log_tau=log_tau_np,
        raw_nu=raw_nu_np,
        n_regimes=r,
        gate=gate,
        family=family,
        sigma_floor=floor,
        nu_min=nmin,
        n_nodes=int(n_nodes),
    )
    final = train_pred.nlpd(yy)
    if not math.isfinite(final):
        raise ValueError(f"final training NLPD is not finite: {final!r}")
    return DeRegiMEResult(
        net=net,
        x_mean=np.asarray(x_mean, dtype=float),
        x_std=np.asarray(x_std, dtype=float),
        y_mean=np.asarray(y_mean, dtype=float),
        y_std=np.asarray(y_std, dtype=float),
        log_tau=log_tau_np,
        raw_nu=raw_nu_np,
        n_regimes=r,
        n_horizons=n_horizons,
        gate=gate,
        family=family,
        sigma_floor=floor,
        nu_min=nmin,
        gate_penalty=pen,
        n_nodes=int(n_nodes),
        seed=int(seed),
        epochs=n_epochs,
        loss_curve=np.asarray(curve, dtype=float),
        nlpd_train=final,
    )


# ---------------------------------------------------------------------------
# SYNTHETIC benchmark vs NGBoostGaussian (lane adaptation; see module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticRegimeBenchmark:
    """DeRegiME vs NGBoostGaussian on seeded SYNTHETIC regime-switch streams.

    ``metrics`` is a flat float dict with bench-ready keys (``drm_*``).
    SYNTHETIC correctness comparison only — never market evidence, and not
    a reproduction of the paper's real-data benchmark table.
    """

    metrics: dict[str, float]
    deregime: DeRegiMEResult
    prediction: RegimeMixturePrediction
    ngboost_nlpd_by_horizon: tuple[float, ...]


def synthetic_regime_benchmark(
    *,
    sigmas: Sequence[float] = (0.3, 0.8, 1.6),
    n_steps: int = 1500,
    n_eval_steps: int = 2000,
    n_horizons: int = 2,
    p_stay: float = 0.99,
    tail_nu: float | None = 6.0,
    n_regimes: int = 4,
    gate: str = "stick_breaking",
    family: str = "student_t",
    hidden: Sequence[int] = (32, 32),
    epochs: int = 400,
    lr: float = 5e-3,
    weight_decay: float = 1e-2,
    gate_penalty: float = 0.0,
    vres: bool = False,
    sigma_floor: float = 0.2,
    ngboost_estimators: int = 150,
    seed: int = 0,
    eval_seed: int = 101,
    r_eff_threshold: float = R_EFF_THRESHOLD,
) -> SyntheticRegimeBenchmark:
    """Train DeRegiME and NGBoostGaussian on one stream; score both on a second.

    The planted generator has ``len(sigmas)`` volatility regimes with unit-
    variance Student-t noise (``tail_nu``); both models see identical causal
    features (``window=5`` lags plus rolling scale summaries). Protocol
    notes: ``p_stay=0.99`` gives persistent regimes (expected duration ~100
    steps) so the causal window identifies the current regime; ``eval_seed``
    is fixed while ``seed`` varies the training stream, keeping the scoring
    distribution common across seeds; ``sigma_floor=0.2`` (standardised
    units) bounds the mixture-MLE degenerate spike — a component scale
    collapsing toward zero, which the paper's ``sigma_floor`` term (their
    Eq. 7) exists to prevent. NGBoostGaussian (Gaussian likelihood,
    log-score trained) is fit per horizon. DeRegiME's advantage is
    mechanistic: regime-mixture scales plus Student-t tails match the
    generator's heteroskedastic heavy tails, which a single Gaussian per
    location cannot. Monte-Carlo note: NLPD gaps are seed-dependent; the
    paper's own headline is a 20.3% NLPD
    improvement over a dynamic Student-t head averaged over three seeds and
    ten real datasets — here we only claim DeRegiME <= NGBoost within a
    documented tolerance on this SYNTHETIC stream (see the lane test).

    Metric keys: ``drm_nlpd`` (eval, mean over horizons), ``drm_ngboost_nlpd``,
    ``drm_nlpd_gap_vs_ngboost`` (DeRegiME - NGBoost; lower is better),
    ``drm_nlpd_train``, ``drm_effective_regimes``, ``drm_planted_regimes``,
    ``drm_regime_scale_max_rel_err`` (each planted scale matched to its
    nearest recovered active-regime scale — labels are identified only up to
    permutation), ``drm_nu_recovered_min``, ``drm_seed``, ``drm_n_eval``,
    ``drm_n_horizons``.
    """
    sig = np.asarray(sigmas, dtype=float).reshape(-1)
    train = simulate_regime_stream(
        n_steps, sigmas=sig, n_horizons=n_horizons, p_stay=p_stay, tail_nu=tail_nu, seed=seed
    )
    ev = simulate_regime_stream(
        n_eval_steps,
        sigmas=sig,
        n_horizons=n_horizons,
        p_stay=p_stay,
        tail_nu=tail_nu,
        seed=eval_seed,
    )
    fitted = fit_deregime(
        train.X,
        train.Y,
        n_regimes=n_regimes,
        gate=gate,
        family=family,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        gate_penalty=gate_penalty,
        vres=vres,
        sigma_floor=sigma_floor,
        seed=seed,
    )
    pred = fitted.predict(ev.X)
    drm_nlpd = pred.nlpd(ev.Y)
    ng_nlpd: list[float] = []
    for j in range(n_horizons):
        ng = NGBoostGaussian(
            n_estimators=_check_count(ngboost_estimators, "ngboost_estimators"),
            learning_rate=0.05,
            score="logscore",
            seed=seed,
        ).fit(train.X, train.Y[:, j])
        ng_nlpd.append(-ng.log_score(ev.X, ev.Y[:, j]))
    ng_mean = float(np.mean(ng_nlpd))
    masses = pred.regime_masses()
    active = masses > r_eff_threshold
    recovered = np.sort(pred.mean_regime_scales()[active])
    # Nearest-neighbour matching of planted scales to recovered active-regime
    # scales: regime labels are only identified up to permutation (paper
    # Appendix B.5), and the truncated gate may keep more candidates than
    # planted regimes, so each planted sigma is matched to its closest
    # recovered scale rather than by sorted position.
    if recovered.size:
        rel_errs = np.abs(recovered[None, :] - sig[:, None]) / sig[:, None]
        rel_err = float(np.max(rel_errs.min(axis=1)))
    else:
        rel_err = 1.0
    tails = pred.tails
    metrics = {
        "drm_nlpd": float(drm_nlpd),
        "drm_ngboost_nlpd": ng_mean,
        "drm_nlpd_gap_vs_ngboost": float(drm_nlpd) - ng_mean,
        "drm_nlpd_train": fitted.nlpd_train,
        "drm_effective_regimes": float(
            effective_regime_count(pred.gates, threshold=r_eff_threshold)
        ),
        "drm_planted_regimes": float(sig.size),
        "drm_regime_scale_max_rel_err": rel_err,
        "drm_nu_recovered_min": float(np.min(tails)) if family == "student_t" else float("nan"),
        "drm_seed": float(seed),
        "drm_n_eval": float(ev.Y.shape[0]),
        "drm_n_horizons": float(n_horizons),
    }
    return SyntheticRegimeBenchmark(
        metrics=metrics,
        deregime=fitted,
        prediction=pred,
        ngboost_nlpd_by_horizon=tuple(ng_nlpd),
    )
