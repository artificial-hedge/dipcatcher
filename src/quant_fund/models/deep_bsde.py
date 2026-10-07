"""Deep BSDE solver: semilinear PDEs via forward SDE + layered Z-networks.

Research-grade minimal implementation of the *decoupled* deep BSDE method:

- E, W., Han, J. & Jentzen, A. (2017), "Deep learning-based numerical methods
  for high-dimensional parabolic partial differential equations and backward
  stochastic differential equations", Communications in Mathematics and
  Statistics 5(4), 349-380, https://doi.org/10.1007/s40304-017-0117-6,
  arXiv:1706.04702 (the deep BSDE solver: the BSDE is viewed as model-based
  reinforcement learning with the diffusion component Z playing the role of
  the policy; Framework 3.2 discretizes the forward SDE with Euler-Maruyama,
  propagates Y forward from a *learned scalar* theta_0 approximating u(0, xi),
  approximates Z at each time layer with its own MLP, and minimizes the
  terminal-condition MSE loss, eq. (12)/(18)/(26); Lemma 4.3 / Subsection 4.5
  give the multidimensional Burgers-type PDE with an explicit solution used as
  a validation benchmark here).
- Han, J., Jentzen, A. & E, W. (2018), "Solving high-dimensional partial
  differential equations using deep learning", Proceedings of the National
  Academy of Sciences 115(34), 8505-8510,
  https://doi.org/10.1073/pnas.1718942115, arXiv:1707.02568 (the PNAS
  presentation of the same method; benchmarks: nonlinear Black-Scholes,
  Hamilton-Jacobi-Bellman, Allen-Cahn, all in d=100).
- Han, J. & Long, J. (2020), "Convergence of the deep BSDE method for coupled
  FBSDEs", Probability, Uncertainty and Quantitative Risk 5, article 5,
  https://doi.org/10.1186/s41546-020-00047-w, arXiv:1811.01165 (the objective
  (2.4) is the terminal MSE E|g(X_T) - Y_T|^2 -- the same loss, with *no*
  additive running term; Theorem 1 gives the a posteriori bound
  simulation error <= C [h + E|g(X_T^pi) - Y_T^pi|^2], so the minimized
  terminal loss doubles as an error indicator).
- Pardoux, E. & Peng, S. (1990), "Adapted solution of a backward stochastic
  differential equation", Systems & Control Letters 14(1), 55-61,
  https://doi.org/10.1016/0167-6911(90)90082-6 (BSDEs; the representation
  dY = -f(t,X,Y,Z) dt + Z dW, Y_T = g(X_T) behind the nonlinear
  Feynman-Kac formula).
- Black, F. & Scholes, M. (1973), "The pricing of options and corporate
  liabilities", Journal of Political Economy 81(3), 637-654, and Merton, R.C.
  (1973), "Theory of rational option pricing", Bell Journal of Economics 4(1),
  141-183 (the analytic European-call price used to validate the linear
  Black-Scholes BSDE).

The semilinear PDE solved (E-Han-Jentzen 2017, eq. (30)) is

    u_t + (1/2) Tr(sigma sigma^T Hess u) + <mu, grad u>
        + f(t, x, u, sigma^T grad u) = 0,    u(T, x) = g(x),

with the forward SDE dX = mu(t,X) dt + sigma(t,X) dW, X_0 = xi, and the BSDE
dY = -f(t,X,Y,Z) dt + Z dW, Y_T = g(X_T), so that Y_t = u(t, X_t) and
Z_t = sigma^T grad u(t, X_t). The Euler-time-discretized solver learns
theta_0 = Y_0 approx u(0, xi) (a scalar parameter) together with layered
Z-networks (one MLP per time layer, plus a constant Z_0 since X_0 = xi is
deterministic), minimizing the terminal-condition MSE with full-batch Adam on
freshly seeded Brownian batches each epoch.

Validation problems with known solutions:

- :func:`black_scholes_call_problem` -- the linear BSDE f = -r y whose PDE is
  the Black-Scholes PDE; Y_0 must recover the analytic call price.
- :func:`burgers_hopf_problem` -- the multidimensional Burgers-type PDE of
  E-Han-Jentzen Lemma 4.3 with the explicit logistic solution
  u(t,x) = exp(t + kappa sum_i x_i) / (1 + exp(t + kappa sum_i x_i)), so
  u(0, 0) = 1/2 exactly in *every* dimension; the paper's Subsection 4.5
  choice alpha = d^2, kappa = 1/d gives the driver
  f = (y - (2+d)/(2d)) sum_i z_i.
- :func:`allen_cahn_problem` -- the paper's Subsection 4.2 Allen-Cahn benchmark
  (cubic nonlinearity; no closed form -- the paper's d=100 reference value
  u(0,0) ~ 0.052802 was computed by branching diffusion, and nothing here
  claims to reproduce it).
- :func:`validate_dimension_scaling` -- the method's headline: the *same*
  problem family and the same architecture rule (per-layer MLPs of width
  d+10) trains at d=1 and d=10 with no dimension-dependent change; only the
  parameter count grows, polynomially in d.

Honesty: everything here runs on SYNTHETIC simulated Brownian paths solving
PDEs with known analytic or semi-analytic solutions. Outputs are solver
errors (|Y_0 - exact|), terminal-loss values (the Han-Long a posteriori error
indicator -- an indicator with an unknown constant C, not a guarantee), and
parameter counts: algorithmic correctness evidence, never market evidence.
No Sharpe/Sortino/P&L headline; no live-trading claims (AGENTS.md honesty
contract).

Conventions: torch is the optional ``nn`` extra and is imported lazily via
:func:`_torch` (mirrors ``models/deep_hedging.py``), so this module imports
cleanly without torch and every training entry point raises ``ImportError``
with install guidance. The numpy core (Brownian sampling, Euler-Maruyama
forward simulation, BSDE rollout, terminal-loss evaluation) is torch-free.
Problem callables are duck-typed: they receive numpy arrays in the numpy
core and torch tensors during training, and must use only arithmetic,
``.sum(-1)``, and the :func:`ns_exp` / :func:`ns_clip_min` dispatch helpers.
Fail-closed edges: ``ValueError`` on non-positive/non-finite shapes and
parameters, non-broadcastable coefficient outputs, non-finite simulations or
losses. Training is CPU single-thread full-batch Adam in float64 (the paper's
reference code is float64) and deterministic given ``seed`` (GPU determinism
is not claimed). Deviations from the paper's TensorFlow reference code, all
simplifications: no batch normalization inside the Z-networks (plain
Linear+ReLU MLPs), constant learning rate, one fresh seeded full batch per
epoch instead of mini-batches of 64, and no clipping of the terminal delta.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]

__all__ = [
    "BSDEProblem",
    "DeepBSDEResult",
    "allen_cahn_problem",
    "black_scholes_call_price",
    "black_scholes_call_problem",
    "bsde_rollout",
    "bsde_terminal_loss",
    "burgers_hopf_problem",
    "deep_bsde_solve",
    "euler_sde_paths",
    "exact_initial_value",
    "ns_clip_min",
    "ns_exp",
    "sample_brownian_increments",
    "validate_black_scholes_call",
    "validate_burgers_hopf",
    "validate_dimension_scaling",
]


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "the deep BSDE solver needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


def _is_torch(x: Any) -> bool:
    return str(getattr(x, "__module__", "")).startswith("torch")


def ns_exp(x: Any) -> Any:
    """Elementwise exp for numpy arrays *and* torch tensors (autograd-safe)."""
    return _torch().exp(x) if _is_torch(x) else np.exp(x)


def ns_clip_min(x: Any, lo: float) -> Any:
    """Elementwise max(x, lo) for numpy arrays *and* torch tensors."""
    lo_f = float(lo)
    return _torch().clamp(x, min=lo_f) if _is_torch(x) else np.clip(x, lo_f, None)


def _check_count(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an int >= 1; got {value!r}")
    return int(value)


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}")
    return v


def _check_finite(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return v


@dataclass(frozen=True, eq=False)
class BSDEProblem:
    """Decoupled FBSDE / semilinear-PDE specification (see module docstring).

    ``mu(t, x) -> (n, d)``, ``sigma(t, x) -> (n, d, d)`` define the forward
    SDE (numpy-only: the scheme is decoupled, so X never depends on learned
    quantities and needs no gradient). ``f(t, x, y, z) -> (n,)`` is the BSDE
    driver in the PDE convention ``u_t + L u + f(t, x, u, sigma^T grad u) = 0``
    (so the Y recursion subtracts ``f * dt``), and ``g(x) -> (n,)`` the
    terminal condition. ``f`` and ``g`` are called with numpy arrays in the
    torch-free core and with torch tensors during training; write them with
    arithmetic, ``.sum(-1)``, and the ``ns_*`` helpers only. ``exact_u(t, x)``
    (optional) is the known solution used for honest error reporting.
    """

    name: str
    dim: int
    horizon: float
    n_steps: int
    x0: Array
    mu: Callable[[float, Any], Any]
    sigma: Callable[[float, Any], Any]
    f: Callable[[float, Any, Any, Any], Any]
    g: Callable[[Any], Any]
    exact_u: Callable[[float, Any], Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string; got {self.name!r}")
        dim = _check_count(self.dim, "dim")
        horizon = _check_positive(self.horizon, "horizon")
        n_steps = _check_count(self.n_steps, "n_steps")
        x0 = np.asarray(self.x0, dtype=float).reshape(-1)
        if x0.shape[0] != dim:
            raise ValueError(f"x0 must have length dim={dim}; got {x0.shape[0]}")
        if not bool(np.all(np.isfinite(x0))):
            raise ValueError("x0 entries must be finite (NaN/inf rejected)")
        for fn_name in ("mu", "sigma", "f", "g"):
            if not callable(getattr(self, fn_name)):
                raise ValueError(f"{fn_name} must be callable; got {getattr(self, fn_name)!r}")
        if self.exact_u is not None and not callable(self.exact_u):
            raise ValueError(f"exact_u must be callable or None; got {self.exact_u!r}")
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "x0", x0)

    def step_size(self) -> float:
        """Uniform time step dt = T / N of the Euler discretization."""
        return float(self.horizon) / int(self.n_steps)

    def time_grid(self) -> Array:
        """Grid times t_0=0 < t_1 < ... < t_N=T (length n_steps+1)."""
        return np.arange(int(self.n_steps) + 1, dtype=float) * self.step_size()


def sample_brownian_increments(
    n_paths: int,
    dim: int,
    n_steps: int,
    *,
    horizon: float,
    seed: int = 0,
) -> Array:
    """Seeded i.i.d. Brownian increments of shape ``(n_paths, n_steps, dim)``.

    Entry [p, i, j] is N(0, dt) with dt = horizon / n_steps, drawn from
    ``numpy.random.default_rng(seed)``. SYNTHETIC data: correctness material,
    never market evidence.
    """
    n = _check_count(n_paths, "n_paths")
    d = _check_count(dim, "dim")
    m = _check_count(n_steps, "n_steps")
    t = _check_positive(horizon, "horizon")
    rng = np.random.default_rng(seed)
    dt = t / m
    return rng.standard_normal((n, m, d)) * math.sqrt(dt)


def _check_problem(problem: Any) -> BSDEProblem:
    if not isinstance(problem, BSDEProblem):
        raise TypeError(f"problem must be a BSDEProblem; got {type(problem).__name__}")
    return problem


def euler_sde_paths(problem: BSDEProblem, dw: Array) -> Array:
    """Euler-Maruyama simulation of the forward SDE on the problem's grid.

    ``X_{n+1} = X_n + mu(t_n, X_n) dt + sigma(t_n, X_n) dW_n`` with
    ``X_0 = x0`` (E-Han-Jentzen 2017, Framework 3.2 eq. (24) with the
    Euler-Maruyama choice of Upsilon, Appendix 5.4). ``dw`` comes from
    :func:`sample_brownian_increments`. Returns ``(n_paths, n_steps+1, dim)``.
    numpy-only and torch-free: in the decoupled scheme X carries no gradient.
    """
    prob = _check_problem(problem)
    dw_arr = np.asarray(dw, dtype=float)
    n_steps, d = int(prob.n_steps), int(prob.dim)
    if dw_arr.ndim != 3 or dw_arr.shape[1:] != (n_steps, d):
        raise ValueError(
            f"dw must have shape (n_paths, n_steps={n_steps}, dim={d}); got {dw_arr.shape}"
        )
    n_paths = int(dw_arr.shape[0])
    if n_paths < 1:
        raise ValueError(f"dw must contain at least one path; got n_paths={n_paths}")
    if not bool(np.all(np.isfinite(dw_arr))):
        raise ValueError("dw entries must be finite (NaN/inf rejected)")
    dt = prob.step_size()
    ts = prob.time_grid()
    out = np.empty((n_paths, n_steps + 1, d), dtype=float)
    out[:, 0, :] = prob.x0
    x = np.repeat(prob.x0[None, :], n_paths, axis=0)
    for i in range(n_steps):
        t = float(ts[i])
        try:
            mu_v = np.broadcast_to(np.asarray(prob.mu(t, x), dtype=float), (n_paths, d))
            sig_v = np.broadcast_to(np.asarray(prob.sigma(t, x), dtype=float), (n_paths, d, d))
        except ValueError as exc:
            raise ValueError(
                f"mu/sigma outputs at step {i} must broadcast to ({n_paths}, {d}) / "
                f"({n_paths}, {d}, {d})"
            ) from exc
        if not (bool(np.all(np.isfinite(mu_v))) and bool(np.all(np.isfinite(sig_v)))):
            raise ValueError(f"mu/sigma produced non-finite values at step {i}")
        x = x + mu_v * dt + np.einsum("nij,nj->ni", sig_v, dw_arr[:, i, :])
        out[:, i + 1, :] = x
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(
            "forward SDE Euler simulation produced non-finite values; "
            "reduce the step size or check mu/sigma"
        )
    return out


def bsde_rollout(
    problem: BSDEProblem,
    x_paths: Array,
    dw: Array,
    y0: float | Array,
    z_values: Array,
) -> Array:
    """Forward Euler rollout of the BSDE recursion (torch-free).

    ``Y_{n+1} = Y_n - f(t_n, X_n, Y_n, Z_n) dt + <Z_n, dW_n>`` from ``Y_0 =
    y0`` (E-Han-Jentzen 2017, eq. (25); Han & Long 2020, eq. (2.3)).
    ``x_paths`` is ``(n_paths, n_steps+1, dim)`` (from :func:`euler_sde_paths`
    on the same ``dw``), ``z_values`` is ``(n_paths, n_steps, dim)``, and
    ``y0`` is a scalar or ``(n_paths,)``. Returns Y as ``(n_paths,
    n_steps+1)``.
    """
    prob = _check_problem(problem)
    n_steps, d = int(prob.n_steps), int(prob.dim)
    x = np.asarray(x_paths, dtype=float)
    dw_arr = np.asarray(dw, dtype=float)
    z = np.asarray(z_values, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (n_steps + 1, d):
        raise ValueError(
            f"x_paths must have shape (n_paths, n_steps+1={n_steps + 1}, dim={d}); got {x.shape}"
        )
    n_paths = int(x.shape[0])
    if dw_arr.shape != (n_paths, n_steps, d):
        raise ValueError(
            f"dw must have shape ({n_paths}, {n_steps}, {d}) matching x_paths; got {dw_arr.shape}"
        )
    if z.shape != (n_paths, n_steps, d):
        raise ValueError(f"z_values must have shape ({n_paths}, {n_steps}, {d}); got {z.shape}")
    for name, arr in (("x_paths", x), ("dw", dw_arr), ("z_values", z)):
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"{name} entries must be finite (NaN/inf rejected)")
    y_init = np.asarray(y0, dtype=float).reshape(-1)
    if y_init.size == 1:
        y = np.full(n_paths, float(y_init[0]), dtype=float)
    elif y_init.size == n_paths:
        y = y_init.copy()
    else:
        raise ValueError(
            f"y0 must be a scalar or a length n_paths={n_paths} vector; got size {y_init.size}"
        )
    if not bool(np.all(np.isfinite(y))):
        raise ValueError("y0 entries must be finite (NaN/inf rejected)")
    dt = prob.step_size()
    ts = prob.time_grid()
    out = np.empty((n_paths, n_steps + 1), dtype=float)
    out[:, 0] = y
    for i in range(n_steps):
        f_v = np.asarray(prob.f(float(ts[i]), x[:, i, :], y, z[:, i, :]), dtype=float).reshape(-1)
        if f_v.size != n_paths:
            raise ValueError(f"f(t, x, y, z) must return n_paths={n_paths} values; got {f_v.size}")
        y = y - f_v * dt + np.sum(z[:, i, :] * dw_arr[:, i, :], axis=-1)
        out[:, i + 1] = y
    if not bool(np.all(np.isfinite(out))):
        raise ValueError(
            "BSDE rollout produced non-finite values; check f/z magnitudes or the step size"
        )
    return out


def bsde_terminal_loss(
    problem: BSDEProblem,
    x_paths: Array,
    dw: Array,
    y0: float | Array,
    z_values: Array,
) -> float:
    """Terminal-condition MSE objective, evaluated in numpy (torch-free).

    ``mean((Y_N - g(X_N))^2)`` -- the deep BSDE objective of E-Han-Jentzen
    (eq. (12)) and, identically, the objective (2.4) of Han & Long (2020):
    both papers use the terminal MSE alone, with no additive running term.
    By Han & Long Theorem 1 the minimized value is an a posteriori error
    indicator (simulation error <= C [h + loss], C unknown).
    """
    prob = _check_problem(problem)
    y = bsde_rollout(prob, x_paths, dw, y0, z_values)
    x = np.asarray(x_paths, dtype=float)
    g_v = np.asarray(prob.g(x[:, -1, :]), dtype=float).reshape(-1)
    if g_v.size != x.shape[0] or not bool(np.all(np.isfinite(g_v))):
        raise ValueError(
            f"g(x) must return n_paths={x.shape[0]} finite values; got size {g_v.size}"
        )
    return float(np.mean((y[:, -1] - g_v) ** 2))


def exact_initial_value(problem: BSDEProblem) -> float:
    """Known solution value u(0, x0); requires ``problem.exact_u``."""
    prob = _check_problem(problem)
    if prob.exact_u is None:
        raise ValueError(f"problem {prob.name!r} has no exact_u; exact initial value unavailable")
    v = np.asarray(prob.exact_u(0.0, prob.x0[None, :]), dtype=float).reshape(-1)
    if v.size != 1 or not bool(np.isfinite(v[0])):
        raise ValueError(f"exact_u(0, x0) must return one finite value; got {v!r}")
    return float(v[0])


def black_scholes_call_price(
    *,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    horizon: float,
) -> float:
    """Analytic European call price (Black & Scholes 1973; Merton 1973).

    ``C = s0 Phi(d1) - strike exp(-r T) Phi(d2)`` with
    ``d1 = (ln(s0/K) + (r + sigma^2/2) T) / (sigma sqrt(T))``, ``d2 = d1 -
    sigma sqrt(T)``. The validation target for :func:`black_scholes_call_problem`.
    """
    s = _check_positive(s0, "s0")
    k = _check_positive(strike, "strike")
    sig = _check_positive(sigma, "sigma")
    t = _check_positive(horizon, "horizon")
    rate = _check_finite(r, "r")
    sq = sig * math.sqrt(t)
    d1 = (math.log(s / k) + (rate + 0.5 * sig * sig) * t) / sq
    d2 = d1 - sq
    return float(s * norm.cdf(d1) - k * math.exp(-rate * t) * norm.cdf(d2))


def black_scholes_call_problem(
    *,
    s0: float = 1.0,
    strike: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.4,
    horizon: float = 1.0,
    n_steps: int = 20,
) -> BSDEProblem:
    """Linear BSDE whose PDE is the Black-Scholes PDE (validation benchmark).

    Forward GBM ``dX = r X dt + sigma X dW`` from ``X_0 = s0``; driver
    ``f(t,x,y,z) = -r y`` (linear in y, independent of z); terminal condition
    ``g(x) = max(x - strike, 0)``. The associated PDE

        ``u_t + r x u_x + (1/2) sigma^2 x^2 u_xx - r u = 0``

    is the Black-Scholes PDE, so ``u(0, s0)`` equals the analytic call price
    :func:`black_scholes_call_price` -- the recovered ``Y_0`` is compared
    against it. SYNTHETIC correctness target, never market evidence.
    """
    s = _check_positive(s0, "s0")
    k = _check_positive(strike, "strike")
    sig = _check_positive(sigma, "sigma")
    t_mat = _check_positive(horizon, "horizon")
    rate = _check_finite(r, "r")
    eye = np.eye(1)

    def mu(t: float, x: Any) -> Any:
        return rate * x

    def sigma_fn(t: float, x: Any) -> Any:
        return sig * (x[:, None, :] * eye)

    def f(t: float, x: Any, y: Any, z: Any) -> Any:
        return -rate * y

    def g(x: Any) -> Any:
        return ns_clip_min(x[:, 0] - k, 0.0)

    def exact_u(t: float, x: Any) -> Any:
        # numpy-only reporting path: analytic Black-Scholes price at (t, x).
        spot = np.asarray(x, dtype=float).reshape(-1)
        tau = t_mat - float(t)
        if tau <= 0.0:
            return np.maximum(spot - k, 0.0)
        sq = sig * math.sqrt(tau)
        d1 = (np.log(spot / k) + (rate + 0.5 * sig * sig) * tau) / sq
        return spot * norm.cdf(d1) - k * math.exp(-rate * tau) * norm.cdf(d1 - sq)

    return BSDEProblem(
        name="black_scholes_call",
        dim=1,
        horizon=t_mat,
        n_steps=n_steps,
        x0=np.array([s], dtype=float),
        mu=mu,
        sigma=sigma_fn,
        f=f,
        g=g,
        exact_u=exact_u,
    )


def burgers_hopf_problem(
    *,
    dim: int,
    horizon: float = 0.3,
    n_steps: int = 12,
    alpha: float | None = None,
    kappa: float | None = None,
) -> BSDEProblem:
    """Multidimensional Burgers-type PDE with an explicit solution.

    E-Han-Jentzen (2017) Lemma 4.3 / Subsection 4.5 (after Chassagneux): with
    ``sigma = sqrt(alpha) I`` (constant, so the Euler simulation of X is
    exact on the grid), the logistic function

        ``u(t, x) = exp(t + kappa sum_i x_i) / (1 + exp(t + kappa sum_i x_i))``

    solves ``u_t + (alpha/2) Laplacian(u) + f(t, x, u, sqrt(alpha) grad u) = 0``
    with driver ``f(t,x,y,z) = (sqrt(alpha) kappa y - 1/(sqrt(alpha) dim kappa)
    - sqrt(alpha) kappa / 2) sum_i z_i``, terminal condition ``g = u(T, .)``,
    and ``u(0, 0) = 1/2`` exactly in every dimension. Defaults reproduce the
    paper's Subsection 4.5 choice ``alpha = dim^2``, ``kappa = 1/dim``, for
    which the driver reduces to ``f = (y - (2+dim)/(2 dim)) sum_i z_i``.
    """
    d = _check_count(dim, "dim")
    t_mat = _check_positive(horizon, "horizon")
    a = _check_positive(alpha if alpha is not None else float(d) ** 2, "alpha")
    kap = _check_positive(kappa if kappa is not None else 1.0 / d, "kappa")
    s = math.sqrt(a)
    cy = s * kap  # coefficient of y * sum(z)
    cz = (1.0 / (d * kap) + a * kap / 2.0) / s  # constant coefficient of sum(z)
    sig_mat = s * np.eye(d)

    def mu(t: float, x: Any) -> Any:
        return 0.0 * x

    def sigma_fn(t: float, x: Any) -> Any:
        return np.broadcast_to(sig_mat, (np.asarray(x).shape[0], d, d))

    def f(t: float, x: Any, y: Any, z: Any) -> Any:
        return (cy * y - cz) * z.sum(-1)

    def exact_u(t: float, x: Any) -> Any:
        w = ns_exp(float(t) + kap * x.sum(-1))
        return w / (1.0 + w)

    def g(x: Any) -> Any:
        return exact_u(t_mat, x)

    return BSDEProblem(
        name=f"burgers_hopf_d{d}",
        dim=d,
        horizon=t_mat,
        n_steps=n_steps,
        x0=np.zeros(d, dtype=float),
        mu=mu,
        sigma=sigma_fn,
        f=f,
        g=g,
        exact_u=exact_u,
    )


def allen_cahn_problem(
    *,
    dim: int,
    horizon: float = 0.3,
    n_steps: int = 20,
) -> BSDEProblem:
    """The paper's Allen-Cahn benchmark (E-Han-Jentzen 2017, Subsection 4.2).

    ``sigma = sqrt(2) I``, ``mu = 0``, ``xi = 0``, driver ``f(y) = y - y^3``,
    terminal ``g(x) = 1 / (2 + 0.4 ||x||^2)``, i.e. the PDE ``u_t + u - u^3 +
    Laplacian(u) = 0``. There is no closed-form solution: the paper's d=100
    reference value ``u(0,0) ~ 0.052802`` was computed by the branching
    diffusion method, and nothing here reproduces or claims it -- this factory
    exists so the cubic-nonlinearity benchmark can be exercised as a smoke
    problem (``exact_u`` is None).
    """
    d = _check_count(dim, "dim")
    t_mat = _check_positive(horizon, "horizon")
    sig_mat = math.sqrt(2.0) * np.eye(d)

    def mu(t: float, x: Any) -> Any:
        return 0.0 * x

    def sigma_fn(t: float, x: Any) -> Any:
        return np.broadcast_to(sig_mat, (np.asarray(x).shape[0], d, d))

    def f(t: float, x: Any, y: Any, z: Any) -> Any:
        return y - y**3

    def g(x: Any) -> Any:
        return 1.0 / (2.0 + 0.4 * (x**2).sum(-1))

    return BSDEProblem(
        name=f"allen_cahn_d{d}",
        dim=d,
        horizon=t_mat,
        n_steps=n_steps,
        x0=np.zeros(d, dtype=float),
        mu=mu,
        sigma=sigma_fn,
        f=f,
        g=g,
        exact_u=None,
    )


def _derive_seed(seed: int, salt: int) -> int:
    return (int(seed) * 1_000_003 + int(salt)) & 0x7FFFFFFF


def _build_z_net(torch: Any, dim: int, hidden: Sequence[int]) -> Any:
    """One per-time-layer MLP mapping X_n (dim) to Z_n (dim); Linear+ReLU."""
    layers: list[Any] = []
    d = int(dim)
    for h in hidden:
        layers += [
            torch.nn.Linear(d, int(h), dtype=torch.float64),
            torch.nn.ReLU(),
        ]
        d = int(h)
    layers.append(torch.nn.Linear(d, int(dim), dtype=torch.float64))
    return torch.nn.Sequential(*layers)


@dataclass(frozen=True, eq=False)
class DeepBSDEResult:
    """Learned initial value, Z-networks, and honest diagnostics.

    ``y0`` is the learned theta_0 approximating u(0, x0). ``terminal_loss`` is
    the Han-Long (2020) objective (2.4) estimated on a fresh, independently
    seeded evaluation batch of Brownian paths -- the a posteriori error
    indicator of their Theorem 1 (an indicator with unknown constant C, not a
    bound). ``y0_exact`` / ``y0_abs_err`` are set when the problem carries a
    known solution. SYNTHETIC solver diagnostics, never market evidence.
    """

    y0: float
    terminal_loss: float
    train_loss_curve: Array  # per-epoch training terminal loss (value before each update)
    y0_curve: Array  # learned theta_0 after each update (paper's init_history)
    y0_exact: float | None
    y0_abs_err: float | None
    seed: int
    epochs: int
    n_paths: int
    z0: Array  # learned constant Z_0 (X_0 = xi deterministic), shape (dim,)
    eval_x: Array = field(repr=False, compare=False)  # (eval_paths, n_steps+1, dim)
    eval_dw: Array = field(repr=False, compare=False)  # (eval_paths, n_steps, dim)
    z_nets: Any = field(repr=False, compare=False)  # list of n_steps-1 torch MLPs
    problem: BSDEProblem = field(repr=False, compare=False)

    def n_parameters(self) -> int:
        """Total trainable parameters: 1 (theta_0) + dim (Z_0) + Z-networks."""
        net_params = sum(int(p.numel()) for net in self.z_nets for p in net.parameters())
        return int(1 + self.problem.dim + net_params)

    def z_values(self, x_paths: Array) -> Array:
        """Layered Z evaluation on given forward paths (numpy, no grad).

        Returns ``(n_paths, n_steps, dim)``: layer 0 is the learned constant
        ``Z_0``, layers 1..N-1 come from the per-time-layer MLPs at ``X_n``.
        Feed the result to :func:`bsde_terminal_loss` for a torch-free loss
        evaluation consistent with training.
        """
        torch = _torch()
        torch.set_num_threads(1)
        prob = self.problem
        n_steps, d = int(prob.n_steps), int(prob.dim)
        x = np.asarray(x_paths, dtype=float)
        if x.ndim != 3 or x.shape[1:] != (n_steps + 1, d):
            raise ValueError(
                f"x_paths must have shape (n_paths, n_steps+1={n_steps + 1}, dim={d}); "
                f"got {x.shape}"
            )
        if not bool(np.all(np.isfinite(x))):
            raise ValueError("x_paths entries must be finite (NaN/inf rejected)")
        n_paths = int(x.shape[0])
        out = np.empty((n_paths, n_steps, d), dtype=float)
        out[:, 0, :] = self.z0
        xt = torch.as_tensor(x, dtype=torch.float64)
        with torch.no_grad():
            for i in range(1, n_steps):
                net = self.z_nets[i - 1]
                net.eval()
                out[:, i, :] = np.asarray(net(xt[:, i]).numpy(), dtype=float)
        if not bool(np.all(np.isfinite(out))):
            raise ValueError("Z-network evaluation produced non-finite values")
        return out


def deep_bsde_solve(
    problem: BSDEProblem,
    *,
    hidden: Sequence[int] | None = None,
    epochs: int = 400,
    lr: float = 1e-2,
    n_paths: int = 512,
    eval_paths: int = 1024,
    seed: int = 0,
    y0_init: float = 0.0,
) -> DeepBSDEResult:
    """Train the decoupled deep BSDE solver (E-Han-Jentzen 2017; PNAS 2018).

    Architecture (the paper's decoupled scheme): a learned scalar ``theta_0``
    for ``Y_0 = u(0, xi)``; a learned constant ``Z_0`` (since ``X_0 = xi`` is
    deterministic); one MLP per time layer ``n = 1..N-1`` mapping ``X_n`` to
    ``Z_n`` (default two hidden layers of width ``dim + 10``, ReLU, as in the
    paper's Section 4). Each epoch draws a *fresh* seeded Brownian batch
    (``n_paths`` paths), simulates the forward SDE with Euler-Maruyama in
    numpy, rolls Y forward in torch through

        ``Y_{n+1} = Y_n - f(t_n, X_n, Y_n, Z_n) dt + <Z_n, dW_n>``,

    and full-batch Adam minimizes the terminal MSE ``mean((Y_N - g(X_N))^2)``
    -- the loss of E-Han-Jentzen eq. (12) and, identically, Han & Long
    objective (2.4); neither paper adds a running term. Deterministic given
    ``seed`` on CPU with a single thread (GPU determinism is not claimed).
    """
    torch = _torch()
    prob = _check_problem(problem)
    if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) < 1:
        raise ValueError(f"epochs must be an int >= 1; got {epochs!r}")
    n_epochs = int(epochs)
    rate = _check_positive(lr, "lr")
    n_train = _check_count(n_paths, "n_paths")
    n_eval = _check_count(eval_paths, "eval_paths")
    y_init = _check_finite(y0_init, "y0_init")
    d = int(prob.dim)
    n_steps = int(prob.n_steps)
    hidden_widths: tuple[int, ...]
    if hidden is None:
        hidden_widths = (d + 10, d + 10)  # paper: two hidden layers of width d+10
    else:
        hidden_widths = tuple(int(h) for h in hidden)
        if not hidden_widths or any(h < 1 for h in hidden_widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths; got {hidden!r}"
            )

    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        y0_param = torch.tensor([y_init], dtype=torch.float64, requires_grad=True)
        z0_param = torch.nn.Parameter(
            torch.zeros((1, d), dtype=torch.float64).uniform_(-0.1, 0.1)  # paper's Z0 init
        )
        nets = [_build_z_net(torch, d, hidden_widths) for _ in range(max(n_steps - 1, 0))]
    params: list[Any] = [y0_param, z0_param]
    for net in nets:
        params.extend(net.parameters())
    opt = torch.optim.Adam(params, lr=rate)

    dt = prob.step_size()
    ts = prob.time_grid()
    curve: list[float] = []
    y0_curve: list[float] = []
    for epoch in range(n_epochs):
        dw_np = sample_brownian_increments(
            n_train, d, n_steps, horizon=prob.horizon, seed=_derive_seed(seed, epoch + 1)
        )
        x_np = euler_sde_paths(prob, dw_np)
        dw_t = torch.as_tensor(dw_np, dtype=torch.float64)
        x_t = torch.as_tensor(x_np, dtype=torch.float64)
        opt.zero_grad(set_to_none=True)
        y = y0_param.expand(n_train)
        z = z0_param.expand(n_train, d)
        for i in range(n_steps):
            y = y - prob.f(float(ts[i]), x_t[:, i, :], y, z) * dt + (z * dw_t[:, i, :]).sum(-1)
            if i < n_steps - 1:
                net = nets[i]
                net.train()
                z = net(x_t[:, i + 1, :])
        loss_t = ((y - prob.g(x_t[:, n_steps, :])) ** 2).mean()
        loss = float(loss_t.detach().numpy())
        if not math.isfinite(loss):
            raise FloatingPointError(
                f"deep BSDE training loss became non-finite at epoch {epoch}; "
                "reduce lr or check f/g magnitudes"
            )
        curve.append(loss)
        loss_t.backward()
        opt.step()
        y0_curve.append(float(y0_param.detach().numpy()[0]))

    eval_dw_np = sample_brownian_increments(
        n_eval, d, n_steps, horizon=prob.horizon, seed=_derive_seed(seed, 999_999)
    )
    eval_x_np = euler_sde_paths(prob, eval_dw_np)
    dw_t = torch.as_tensor(eval_dw_np, dtype=torch.float64)
    x_t = torch.as_tensor(eval_x_np, dtype=torch.float64)
    with torch.no_grad():
        y = y0_param.expand(n_eval)
        z = z0_param.expand(n_eval, d)
        for i in range(n_steps):
            y = y - prob.f(float(ts[i]), x_t[:, i, :], y, z) * dt + (z * dw_t[:, i, :]).sum(-1)
            if i < n_steps - 1:
                net = nets[i]
                net.eval()
                z = net(x_t[:, i + 1, :])
        terminal_loss = float(((y - prob.g(x_t[:, n_steps, :])) ** 2).mean().numpy())
    if not math.isfinite(terminal_loss):
        raise FloatingPointError("deep BSDE evaluation terminal loss is non-finite")
    y0_final = float(y0_param.detach().numpy()[0])
    y0_exact = exact_initial_value(prob) if prob.exact_u is not None else None
    return DeepBSDEResult(
        y0=y0_final,
        terminal_loss=terminal_loss,
        train_loss_curve=np.asarray(curve, dtype=float),
        y0_curve=np.asarray(y0_curve, dtype=float),
        y0_exact=y0_exact,
        y0_abs_err=None if y0_exact is None else abs(y0_final - y0_exact),
        seed=int(seed),
        epochs=n_epochs,
        n_paths=n_train,
        z0=np.asarray(z0_param.detach().numpy(), dtype=float).reshape(-1),
        eval_x=eval_x_np,
        eval_dw=eval_dw_np,
        z_nets=nets,
        problem=prob,
    )


def validate_black_scholes_call(
    *,
    s0: float = 1.0,
    strike: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.4,
    horizon: float = 1.0,
    n_steps: int = 20,
    hidden: Sequence[int] | None = None,
    epochs: int = 700,
    lr: float = 1e-2,
    n_paths: int = 512,
    eval_paths: int = 1024,
    seed: int = 0,
) -> dict[str, float]:
    """Recovered ``Y_0`` vs the analytic Black-Scholes call price.

    Trains :func:`black_scholes_call_problem` (linear BSDE, ``f = -r y``) and
    reports the solver error against :func:`black_scholes_call_price`. Metric
    keys are scorecard-ready. SYNTHETIC correctness comparison on simulated
    Brownian paths -- never market evidence, no live-trading claims.
    """
    prob = black_scholes_call_problem(
        s0=s0, strike=strike, r=r, sigma=sigma, horizon=horizon, n_steps=n_steps
    )
    analytic = black_scholes_call_price(s0=s0, strike=strike, r=r, sigma=sigma, horizon=horizon)
    res = deep_bsde_solve(
        prob,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        n_paths=n_paths,
        eval_paths=eval_paths,
        seed=seed,
    )
    abs_err = abs(res.y0 - analytic)
    return {
        "bsde_bs_call_y0": float(res.y0),
        "bsde_bs_call_analytic": float(analytic),
        "bsde_bs_call_abs_err": float(abs_err),
        "bsde_bs_call_rel_err": float(abs_err / abs(analytic)) if analytic != 0.0 else float("nan"),
        "bsde_bs_call_terminal_loss": float(res.terminal_loss),
        "bsde_bs_call_train_loss_start": float(res.train_loss_curve[0]),
        "bsde_bs_call_n_params": float(res.n_parameters()),
        "bsde_bs_call_n_steps": float(n_steps),
        "bsde_bs_call_epochs": float(epochs),
    }


def validate_burgers_hopf(
    *,
    dim: int,
    horizon: float = 0.3,
    n_steps: int = 12,
    alpha: float | None = None,
    kappa: float | None = None,
    hidden: Sequence[int] | None = None,
    epochs: int = 500,
    lr: float = 1e-2,
    n_paths: int = 512,
    eval_paths: int = 1024,
    seed: int = 0,
) -> dict[str, float]:
    """Recovered ``Y_0`` vs the explicit Burgers-Hopf solution ``u(0,0)=1/2``.

    E-Han-Jentzen (2017) Lemma 4.3 / Subsection 4.5 benchmark
    (:func:`burgers_hopf_problem`). SYNTHETIC correctness evidence with a
    closed-form target -- never market evidence.
    """
    prob = burgers_hopf_problem(dim=dim, horizon=horizon, n_steps=n_steps, alpha=alpha, kappa=kappa)
    res = deep_bsde_solve(
        prob,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        n_paths=n_paths,
        eval_paths=eval_paths,
        seed=seed,
    )
    d = int(prob.dim)
    exact = exact_initial_value(prob)  # 0.5 exactly
    abs_err = abs(res.y0 - exact)
    return {
        f"bsde_burgers_d{d}_y0": float(res.y0),
        f"bsde_burgers_d{d}_exact": float(exact),
        f"bsde_burgers_d{d}_abs_err": float(abs_err),
        f"bsde_burgers_d{d}_terminal_loss": float(res.terminal_loss),
        f"bsde_burgers_d{d}_train_loss_start": float(res.train_loss_curve[0]),
        f"bsde_burgers_d{d}_n_params": float(res.n_parameters()),
        f"bsde_burgers_d{d}_epochs": float(epochs),
    }


def validate_dimension_scaling(
    *,
    dims: Sequence[int] = (1, 10),
    horizon: float = 0.3,
    n_steps: int = 12,
    hidden: Sequence[int] | None = None,
    epochs: int = 500,
    lr: float = 1e-2,
    n_paths: int = 512,
    eval_paths: int = 1024,
    seed: int = 0,
) -> dict[str, float]:
    """Same Burgers-Hopf family at every dim with one architecture rule.

    The method's headline claim (E-Han-Jentzen 2017; Han-Jentzen-E PNAS 2018):
    the solver is dimension-agnostic -- identical per-layer MLP rule (width
    ``dim + 10``), identical optimizer settings, and each dimension recovers
    ``u(0, 0) = 1/2``. Only the parameter count grows, polynomially in dim.
    SYNTHETIC correctness evidence, never market evidence.
    """
    dim_list = tuple(int(d) for d in dims)
    if len(dim_list) < 2 or any(d < 1 for d in dim_list):
        raise ValueError(f"dims must contain at least two positive ints; got {dims!r}")
    metrics: dict[str, float] = {
        "bsde_dim_scaling_n_dims": float(len(dim_list)),
        "bsde_dim_scaling_n_steps": float(n_steps),
        "bsde_dim_scaling_epochs": float(epochs),
    }
    errs: list[float] = []
    counts: list[float] = []
    for d in dim_list:
        row = validate_burgers_hopf(
            dim=d,
            horizon=horizon,
            n_steps=n_steps,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            n_paths=n_paths,
            eval_paths=eval_paths,
            seed=seed,
        )
        metrics.update(row)
        errs.append(float(row[f"bsde_burgers_d{d}_abs_err"]))
        counts.append(float(row[f"bsde_burgers_d{d}_n_params"]))
    metrics["bsde_dim_scaling_max_abs_err"] = float(max(errs))
    metrics["bsde_dim_scaling_param_ratio"] = float(max(counts) / min(counts))
    return metrics
