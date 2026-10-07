"""G-SLiCE: generative structured linear CDEs for path-space flow matching.

Implements Generative SLiCEs (G-SLiCEs) of Berndt, T., Farjallah, E., Seute,
L., Saqur, R., Walker, B. & Stühmer, J. (2026), "Universal Time Series
Generation with Neural Controlled Differential Equations", arXiv:2605.28507
[cs.LG] — a continuous-time generative model that learns a path-valued flow

    d/ds X^(s) = F_θ(s, X^(s)),     X^(0) ~ μ,     s ∈ [0, 1],

where the vector field F_θ is a causal Structured Linear CDE (SLiCE) acting
along the *physical* time axis of the path and s is *flow time*, injected by
concatenating it as an additional constant channel to the input path (§3).
Corollary 2.1 of the paper shows maximally expressive SLiCE classes are
universal causal time-series generators: they approximate induced path laws
of continuous causal pushforwards in W_infinity (w.r.t. the supremum metric
on path space).

SLiCE backbone (paper §2.1, eqs. 2–3): a Linear Neural CDE with hidden state
h_t ∈ R^{d_h} driven by a causal augmentation ω of the input path,

    h_{t0} = ξ_φ(X_{t0}, c),
    h_{j+1} = Φ_j h_j,   Φ_j = exp( Σ_i A^i Δω^i_j ),   z_t = r_ψ(h_t),

i.e. the exact zero-order-hold discretization of dh = Σ_i A^i h dω^i on
piecewise-linear controls. Each A^i is restricted to a structured family —
here block-diagonal with ``block_size`` b (b = 1 is the diagonal selective
case of Mamba-style models, which is NOT universal per the paper's
Proposition 3 / Cirone et al.; b = d_h is the dense Linear-NCDE case;
1 < b < d_h is the maximally expressive block-diagonal SLiCE). exp of a
block-diagonal matrix is block-diagonal of block exponentials, so the
transition is computed per block and applied to reshaped hidden state.
Residual stacks of ``n_blocks`` such layers (Appendix F: 3–5 residual blocks)
form the field; each block re-drives on the previous block's output path.

Flow matching (paper §3, following the grid version of Kollovieh et al.
2023, and conditional flow matching of Lipman et al. 2023): pairs
(X^(0), X^(1)) are drawn from the mini-batched optimal-transport coupling
q(X^(0), X^(1)) (Tong et al. 2023, "Improving and generalizing flow-based
generative models with minibatch optimal transport", ICML — here exact
Hungarian assignment via scipy linear_sum_assignment on squared Frobenius
distances of flattened paths); with the straight-line interpolant
X^(s) = (1-s) X^(0) + s X^(1) the target velocity is u_s = X^(1) − X^(0) and

    L(θ) = E[ ||F_θ(s, X^(s)) − (X^(1) − X^(0))||² ],

evaluated on the discretized physical-time grid. Sampling integrates the
learned path-ODE from s = 0 to 1 by Euler or midpoint steps.

Prior on path space (paper §3): a Gaussian process, μ = GP(m, k) in the
unconditional case and the GP posterior μ(·|C) = GP(m_post, k_post) fitted
on context C = {(t_i, X_{t_i})} in the conditional case. Kernels here:
"rbf" (squared exponential), "matern32", and "wiener" (k = min(t, t'),
natural for integrated return paths); channels share the kernel and are
independent under the prior (the paper's benchmarks are univariate).

Signature conditioning (repo-lane composition, additive to the paper's
construction): each context prefix is summarized by its truncated
logsignature (``path_signatures.logsignature``) of the time-augmented
observed path, and the resulting feature vector is appended as constant
driving channels AND fed to every block's initialization map ξ. Alongside
the signature vector the field receives the paper's explicit deterministic
context channels (§2.1: "ω may include time, observed values, masks,
lags"): the observed-knot mask and the GP-posterior mean path m(·|C),
so the vector field sees raw context at observed knots rather than only
its signature summary. This stays inside the paper's "deterministic causal
augmentation" clause for ω while reusing the repo's canonical signature
implementation rather than reimplementing it.

Diagnostics are proper scores only (AGENTS.md honesty contract): ensemble
energy score on flattened paths (``metrics.energy_score.energy_score``;
Gneiting & Raftery 2007), marginal band coverage/width over the generated
ensemble, and per-point empirical PIT values consumable by
``metrics.density_forecast`` — never Sharpe/Sortino/P&L.

Honesty: all training/evaluation in this module and its tests runs on
SYNTHETIC simulated path laws (GP priors, regime-switching increments) —
correctness evidence for the algorithm, never market evidence. Generated
paths are distributional objects for research scoring only; no trading or
live-deployment claim is made or implied.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``models/deep_hedging.py``), so the module imports
cleanly without torch and every torch entry point raises ``ImportError``
with install guidance. The numpy layer (GP kernels/prior/posterior,
signature contexts, OT coupling, interpolant, diagnostics, SYNTHETIC law
samplers) is torch-free. Fail-closed edges: ``ValueError`` on degenerate
shapes, non-finite values, non-increasing grids, invalid kernels/hyper-
parameters, missing context where the model was trained with one, and
non-PSD GP covariances (Cholesky with escalating jitter, then raise).
Deterministic: all randomness flows through seeded numpy Generators and
``torch.manual_seed``; CPU single-thread; repeated calls bit-identical
(GPU determinism not claimed).

Composition notes: imports ``logsignature`` from ``models/path_signatures``
(owns signature primitives), ``energy_score`` from ``metrics/energy_score``
(owns the U-statistic ensemble score), and mirrors the lazy-``_torch``
pattern of ``models/deep_hedging``. The spec's ``models/diffpts.py`` does
not exist in the repo; the flow-matching training loop instead follows the
same module's full-batch Adam conventions. Sparse and Walsh–Hadamard
structured families of the paper are not implemented (block-diagonal covers
dense and diagonal as b = d_h and b = 1). At conditional sampling time the
observed knots are pinned to their GP-posterior draws during integration
(imputation-style projection — the paper does not specify this detail); it
keeps generated paths anchored at the observation while the learned field
drives all unobserved positions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import kstest

from quant_fund.metrics.energy_score import energy_score
from quant_fund.models.path_signatures import logsignature

Array = NDArray[np.float64]
RNG = np.random.Generator | int

__all__ = [
    "GP_KERNELS",
    "GSliceConfig",
    "GSliceContext",
    "GSliceModel",
    "bench_gslice",
    "build_context",
    "context_logsignature",
    "evaluate_samples",
    "gp_kernel_matrix",
    "gp_posterior_moments",
    "interpolant",
    "ot_coupling",
    "path_band_coverage",
    "path_energy_score",
    "path_pit_values",
    "sample_gp_paths",
    "sample_gp_posterior_paths",
    "synthetic_gp_paths",
    "synthetic_switching_paths",
    "train_gslice",
]

#: GP prior kernels on the physical-time grid (paper §3 uses GPs as the
#: canonical non-parametric noise law on path space).
GP_KERNELS: tuple[str, str, str] = ("rbf", "matern32", "wiener")

_JITTER_LEVELS = (1e-10, 1e-8, 1e-6, 1e-4)


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "G-SLiCE needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# validation helpers (fail-closed; mirror models/deep_hedging.py conventions)
# ---------------------------------------------------------------------------


def _check_count(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}; got {value!r}")
    return int(value)


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {value!r}")
    return v


def _check_probability(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1]; got {value!r}")
    return v


def _resolve_rng(rng: RNG | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _as_grid(grid: Array | Sequence[float]) -> Array:
    arr = np.asarray(grid, dtype=float).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError(f"grid must have at least 2 points; got {arr.shape[0]}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("grid must be finite (NaN/inf rejected)")
    if not bool(np.all(np.diff(arr) > 0.0)):
        raise ValueError("grid must be strictly increasing")
    return arr


def _as_paths3(paths: Array, *, name: str = "paths") -> Array:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must be a 3-D array of shape (n_paths, n_grid, n_channels); "
            f"got ndim={arr.ndim}"
        )
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one path; got {arr.shape[0]}")
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 grid points; got {arr.shape[1]}")
    if arr.shape[2] < 1:
        raise ValueError(f"{name} must have at least 1 channel; got {arr.shape[2]}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} entries must be finite (NaN/inf rejected)")
    return arr


def _check_kernel(kernel: str) -> str:
    if kernel not in GP_KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {GP_KERNELS}")
    return kernel


def _norm_grid(grid: Array) -> Array:
    """Normalize the physical-time grid to [0, 1] (preserves causality)."""
    span = float(grid[-1] - grid[0])
    return np.asarray((grid - grid[0]) / span, dtype=float)


# ---------------------------------------------------------------------------
# Gaussian-process prior / posterior on path space (paper §3)
# ---------------------------------------------------------------------------


def gp_kernel_matrix(
    grid: Array | Sequence[float],
    *,
    kernel: str = "rbf",
    length_scale: float = 0.3,
    amplitude: float = 1.0,
) -> Array:
    """GP covariance K[i, j] = k(t_i, t_j) on ``grid`` (n_grid, n_grid).

    Kernels: ``rbf`` = a² exp(-r² / 2ℓ²); ``matern32`` = a² (1 + √3 r/ℓ)
    exp(-√3 r/ℓ); ``wiener`` = a² min(t_i - t_0, t_j - t_0) (Brownian motion
    anchored at the grid start — a valid GP covariance for integrated paths;
    first row/column are exactly zero).
    """
    t = _as_grid(grid)
    kind = _check_kernel(kernel)
    ell = _check_positive(length_scale, "length_scale")
    amp = _check_positive(amplitude, "amplitude")
    if kind == "wiener":
        tt = t - t[0]
        return np.asarray(amp * amp * np.minimum.outer(tt, tt), dtype=float)
    dist = np.abs(t[:, None] - t[None, :])
    if kind == "rbf":
        return amp * amp * np.exp(-0.5 * (dist / ell) ** 2)
    scaled = math.sqrt(3.0) * dist / ell
    return amp * amp * (1.0 + scaled) * np.exp(-scaled)


def _cholesky_jittered(cov: Array, *, name: str = "covariance") -> Array:
    """Cholesky of a PSD matrix with escalating diagonal jitter; fail closed."""
    for jitter in _JITTER_LEVELS:
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise ValueError(f"{name} is not positive semidefinite even with jitter")


def sample_gp_paths(
    n_paths: int,
    grid: Array | Sequence[float],
    *,
    kernel: str = "rbf",
    length_scale: float = 0.3,
    amplitude: float = 1.0,
    mean: Array | Sequence[float] | None = None,
    rng: RNG | None = None,
) -> Array:
    """Sample ``n_paths`` paths ~ GP(mean, k) on ``grid`` → (n, n_grid, 1).

    The unconditional path-space prior μ of G-SLiCE (paper §3). Cholesky
    sampling m + L ε with ε ~ N(0, I); degenerate/non-PSD covariances fail
    closed after jitter escalation. Deterministic given ``rng``.
    """
    n = _check_count(n_paths, "n_paths")
    t = _as_grid(grid)
    cov = gp_kernel_matrix(t, kernel=kernel, length_scale=length_scale, amplitude=amplitude)
    m = np.zeros(t.shape[0]) if mean is None else np.asarray(mean, dtype=float).reshape(-1)
    if m.shape[0] != t.shape[0] or not bool(np.all(np.isfinite(m))):
        raise ValueError(f"mean must be finite with length n_grid={t.shape[0]}")
    chol = _cholesky_jittered(cov)
    gen = _resolve_rng(rng)
    eps = gen.standard_normal((n, t.shape[0]))
    return (m[None, :] + eps @ chol.T)[:, :, None]


def gp_posterior_moments(
    grid: Array | Sequence[float],
    obs_times: Array | Sequence[float],
    obs_values: Array,
    *,
    kernel: str = "rbf",
    length_scale: float = 0.3,
    amplitude: float = 1.0,
    noise: float = 1e-4,
) -> tuple[Array, Array]:
    """GP posterior mean (n_grid, d) and covariance (n_grid, n_grid).

    Standard GP regression conditioning (Rasmussen & Williams 2006, §2.2):
    m_post(t) = K(t, T) [K(T, T) + νI]^{-1} y applied per channel; the
    covariance k_post is shared across channels (same input locations).
    ``obs_times`` is a shared observation grid of length m >= 1 lying inside
    the span of ``grid``; ``obs_values`` is (m, d). Fails closed on empty or
    non-finite observations.
    """
    t = _as_grid(grid)
    t_obs = np.asarray(obs_times, dtype=float).reshape(-1)
    y = np.asarray(obs_values, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if t_obs.shape[0] < 1:
        raise ValueError("obs_times must be non-empty")
    if not bool(np.all(np.isfinite(t_obs))):
        raise ValueError("obs_times must be finite")
    if y.ndim != 2 or y.shape[0] != t_obs.shape[0] or y.shape[1] < 1:
        raise ValueError(
            f"obs_values must have shape (n_obs, d) with n_obs={t_obs.shape[0]}; got {y.shape}"
        )
    if not bool(np.all(np.isfinite(y))):
        raise ValueError("obs_values must be finite (NaN/inf rejected)")
    nu = float(noise)
    if not math.isfinite(nu) or nu < 0.0:
        raise ValueError(f"noise must be finite and >= 0; got {noise!r}")
    k = _check_kernel(kernel)
    ell = _check_positive(length_scale, "length_scale")
    amp = _check_positive(amplitude, "amplitude")
    k_tt = gp_kernel_matrix(t, kernel=k, length_scale=ell, amplitude=amp)
    # cross-covariance via the same kernel formulas on mixed grids
    if k == "wiener":
        k_to = amp * amp * np.minimum.outer(t - t[0], t_obs - t[0])
        k_oo = amp * amp * np.minimum.outer(t_obs - t[0], t_obs - t[0])
    else:
        d_to = np.abs(t[:, None] - t_obs[None, :])
        d_oo = np.abs(t_obs[:, None] - t_obs[None, :])
        if k == "rbf":
            k_to = amp * amp * np.exp(-0.5 * (d_to / ell) ** 2)
            k_oo = amp * amp * np.exp(-0.5 * (d_oo / ell) ** 2)
        else:
            r_to = math.sqrt(3.0) * d_to / ell
            r_oo = math.sqrt(3.0) * d_oo / ell
            k_to = amp * amp * (1.0 + r_to) * np.exp(-r_to)
            k_oo = amp * amp * (1.0 + r_oo) * np.exp(-r_oo)
    a = k_oo + nu * np.eye(t_obs.shape[0])
    sol = np.linalg.solve(a, y)  # (m, d)
    mean = k_to @ sol
    w = np.linalg.solve(a, k_to.T)  # (m, n_grid)
    cov = k_tt - k_to @ w
    cov = 0.5 * (cov + cov.T)  # symmetrize against solve roundoff
    return mean, cov


def sample_gp_posterior_paths(
    n_paths: int,
    mean: Array,
    cov: Array,
    *,
    rng: RNG | None = None,
) -> Array:
    """Sample paths from a GP posterior → (n, n_grid, d).

    ``mean`` is (n_grid, d) and ``cov`` the shared (n_grid, n_grid)
    posterior covariance from :func:`gp_posterior_moments`; channel draws
    share the Cholesky factor (independent channels under the prior).
    """
    n = _check_count(n_paths, "n_paths")
    m = np.asarray(mean, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 1:
        raise ValueError(f"mean must have shape (n_grid, d); got {m.shape}")
    c = np.asarray(cov, dtype=float)
    if c.shape != (m.shape[0], m.shape[0]):
        raise ValueError(f"cov must be (n_grid, n_grid); got {c.shape}")
    if not bool(np.all(np.isfinite(m))) or not bool(np.all(np.isfinite(c))):
        raise ValueError("mean/cov must be finite")
    chol = _cholesky_jittered(cov, name="posterior covariance")
    gen = _resolve_rng(rng)
    eps = gen.standard_normal((n, m.shape[0], m.shape[1]))
    return np.asarray(m[None, :, :] + np.einsum("ij,njk->nik", chol, eps), dtype=float)


# ---------------------------------------------------------------------------
# signature contexts + OT coupling + interpolant
# ---------------------------------------------------------------------------


def context_logsignature(paths: Array, order: int = 2) -> Array:
    """Per-path context features: logsignature of the time-augmented path.

    Each row of ``paths`` (n, m, d) is augmented with its normalized-time
    channel (making the level-2 Lie content non-trivial for d = 1) and its
    truncated logsignature at ``order`` (1..4, per
    ``path_signatures.logsignature``) is returned — a (n, d_sig) array.
    Deterministic; translation- and reparametrization-invariant statistics
    of the observed prefix (Lyons 1998), used as the signature conditioning
    vector c.
    """
    arr = _as_paths3(paths, name="context paths")
    m = _check_count(order, "order")
    if m > 4:
        raise ValueError(f"order must be <= 4 (logsignature limit); got {order}")
    n, steps, _ = arr.shape
    t = np.linspace(0.0, 1.0, steps, dtype=float)
    t_batch = np.broadcast_to(t[None, :, None], (n, steps, 1))
    aug = np.concatenate([t_batch, arr], axis=2)  # (n, m, d+1)
    feats = [logsignature(aug[i], order=m) for i in range(n)]
    return np.stack(feats, axis=0)


@dataclass(frozen=True)
class GSliceContext:
    """Path-space conditioning C for the conditional G-SLiCE (paper §3).

    ``obs_times``/``obs_values`` define the observed prefix points the GP
    posterior prior μ(·|C) is fitted on; ``features`` carries the signature
    conditioning vector appended as constant driving channels. All arrays
    are per-path: n_context paths, m shared observation times.
    """

    obs_times: Array  # (n, m) or shared (m,) — stored per-path (n, m)
    obs_values: Array  # (n, m, d)
    features: Array  # (n, d_sig)


def build_context(
    obs_times: Array | Sequence[float],
    obs_values: Array,
    *,
    order: int = 2,
) -> GSliceContext:
    """Build a :class:`GSliceContext` from observed prefix points.

    ``obs_times`` may be shared across paths (m,) or per-path (n, m);
    ``obs_values`` is (n, m, d). Requires m >= 2 (a single observation has
    no path content to summarize — fail closed). Signature features are the
    logsignature of the time-augmented observed path per row.
    """
    t_obs = np.asarray(obs_times, dtype=float)
    y = _as_paths3(obs_values, name="obs_values")
    n, m, _ = y.shape
    if t_obs.ndim == 1:
        if t_obs.shape[0] != m:
            raise ValueError(f"shared obs_times must have length m={m}; got {t_obs.shape[0]}")
        t_obs = np.broadcast_to(t_obs[None, :], (n, m)).copy()
    elif t_obs.ndim == 2 and t_obs.shape == (n, m):
        t_obs = t_obs.copy()
    else:
        raise ValueError(f"obs_times must be (m,) or (n, m) with n={n}, m={m}; got {t_obs.shape}")
    if m < 2:
        raise ValueError(f"context needs at least 2 observation times; got {m}")
    if not bool(np.all(np.isfinite(t_obs))):
        raise ValueError("obs_times must be finite")
    for i in range(n):
        if not bool(np.all(np.diff(t_obs[i]) > 0.0)):
            raise ValueError(f"obs_times must be strictly increasing (row {i})")
    # signature features over the (time, value) path per row
    feats = np.stack(
        [logsignature(np.column_stack([t_obs[i], y[i]]), order=order) for i in range(n)],
        axis=0,
    )
    return GSliceContext(obs_times=t_obs, obs_values=y, features=feats)


def ot_coupling(x0: Array, x1: Array) -> Array:
    """Mini-batch OT coupling: permutation perm with x0[i] ↔ x1[perm[i]].

    Exact Hungarian assignment (scipy ``linear_sum_assignment``) on the
    squared Frobenius cost of flattened paths — the mini-batched OT plan
    q(X^(0), X^(1)) of Tong et al. (2023) used by the paper's trainer.
    Returns the column indices assigned to each row; x0 and x1 must have the
    same leading dimension.
    """
    a = _as_paths3(x0, name="x0")
    b = _as_paths3(x1, name="x1")
    if a.shape != b.shape:
        raise ValueError(f"x0 and x1 must share shape; got {a.shape} vs {b.shape}")
    n = a.shape[0]
    fa = a.reshape(n, -1)
    fb = b.reshape(n, -1)
    aa = np.sum(fa * fa, axis=1, keepdims=True)
    bb = np.sum(fb * fb, axis=1, keepdims=True).T
    cost = aa + bb - 2.0 * (fa @ fb.T)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(n, dtype=int)
    perm[rows] = cols
    return perm


def interpolant(x0: Array, x1: Array, s: Array | float) -> Array:
    """Straight-line path-space interpolant X^(s) = (1-s) X^(0) + s X^(1).

    ``s`` is a scalar or a per-path vector of shape (n,) broadcast along the
    grid and channel axes. Values outside [0, 1] fail closed — s is a flow
    time, not an extrapolation parameter.
    """
    a = _as_paths3(x0, name="x0")
    b = _as_paths3(x1, name="x1")
    if a.shape != b.shape:
        raise ValueError(f"x0 and x1 must share shape; got {a.shape} vs {b.shape}")
    sv = np.asarray(s, dtype=float)
    if sv.ndim == 0:
        sv = np.full(a.shape[0], float(sv))
    if sv.shape != (a.shape[0],) or not bool(np.all(np.isfinite(sv))):
        raise ValueError(f"s must be a scalar or finite (n,) vector; got {sv.shape}")
    if bool(np.any(sv < 0.0)) or bool(np.any(sv > 1.0)):
        raise ValueError("s must lie in [0, 1]")
    w = sv[:, None, None]
    return (1.0 - w) * a + w * b


# ---------------------------------------------------------------------------
# proper-score diagnostics on path space (proper scores only — AGENTS.md)
# ---------------------------------------------------------------------------


def path_energy_score(generated: Array, observed: Array) -> float:
    """Mean energy score of a generated ensemble vs held-out paths.

    Paths are flattened to vectors in R^{n_grid * d} and each held-out path
    is scored against the full generated ensemble with the strictly proper
    U-statistic energy score (Gneiting & Raftery 2007) from
    ``metrics.energy_score``; the mean over held-out paths is returned.
    """
    gen = _as_paths3(generated, name="generated")
    obs = _as_paths3(observed, name="observed")
    if gen.shape[1:] != obs.shape[1:]:
        raise ValueError(
            f"generated/observed must share (n_grid, d); got {gen.shape[1:]} vs {obs.shape[1:]}"
        )
    if gen.shape[0] < 2:
        raise ValueError("generated ensemble needs >= 2 paths")
    gflat = gen.reshape(gen.shape[0], -1)
    oflat = obs.reshape(obs.shape[0], -1)
    return float(np.mean([energy_score(gflat, oflat[i]) for i in range(oflat.shape[0])]))


def path_band_coverage(
    generated: Array, observed: Array, *, level: float = 0.8
) -> dict[str, float]:
    """Marginal coverage of the generated ensemble's central band.

    At each (grid time, channel) the empirical central ``level`` band of the
    generated ensemble is formed by the (1-level)/2 and 1-(1-level)/2
    quantiles; ``coverage`` is the fraction of observed points inside and
    ``mean_width`` the average band width. Marginal calibration diagnostic,
    not a joint-path guarantee.
    """
    gen = _as_paths3(generated, name="generated")
    obs = _as_paths3(observed, name="observed")
    if gen.shape[1:] != obs.shape[1:]:
        raise ValueError(
            f"generated/observed must share (n_grid, d); got {gen.shape[1:]} vs {obs.shape[1:]}"
        )
    if gen.shape[0] < 2:
        raise ValueError("generated ensemble needs >= 2 paths")
    lev = float(level)
    if not math.isfinite(lev) or not (0.0 < lev < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}")
    alpha = 1.0 - lev
    lo = np.quantile(gen, alpha / 2.0, axis=0)
    hi = np.quantile(gen, 1.0 - alpha / 2.0, axis=0)
    inside = (obs >= lo[None]) & (obs <= hi[None])
    return {
        "coverage": float(np.mean(inside)),
        "mean_width": float(np.mean(hi - lo)),
    }


def path_pit_values(generated: Array, observed: Array) -> Array:
    """Empirical PIT of each observed point under the generated ensemble.

    pit[i, t, c] = fraction of generated paths with value <= observed[i, t, c],
    clipped into the open interval (1/(2n), 1 - 1/(2n)). Under a correctly
    specified path law these are iid Uniform(0,1) marginally (Rosenblatt
    1952) — feed to ``metrics.density_forecast.pit_histogram``.
    """
    gen = _as_paths3(generated, name="generated")
    obs = _as_paths3(observed, name="observed")
    if gen.shape[1:] != obs.shape[1:]:
        raise ValueError(
            f"generated/observed must share (n_grid, d); got {gen.shape[1:]} vs {obs.shape[1:]}"
        )
    n = gen.shape[0]
    if n < 2:
        raise ValueError("generated ensemble needs >= 2 paths")
    pits = np.mean(gen[None] <= obs[:, None], axis=1)
    eps = 0.5 / float(n)
    return np.clip(pits, eps, 1.0 - eps)


def evaluate_samples(
    generated: Array,
    observed: Array,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
) -> dict[str, float]:
    """Scorecard-ready proper-score evaluation of a generated ensemble.

    Returns flat ``gslice_*`` keys: energy score (flattened path space),
    marginal band coverage and width per level, and PIT mean/sd/KS
    uniformity p-value. Proper scores only — no Sharpe-family content.
    """
    gen = _as_paths3(generated, name="generated")
    obs = _as_paths3(observed, name="observed")
    if len(levels) == 0:
        raise ValueError("levels must be non-empty")
    out: dict[str, float] = {
        "gslice_energy_score_mean": path_energy_score(gen, obs),
        "gslice_n_generated": float(gen.shape[0]),
        "gslice_n_observed": float(obs.shape[0]),
    }
    for lev in levels:
        band = path_band_coverage(gen, obs, level=float(lev))
        tag = f"{int(round(float(lev) * 100)):02d}"
        out[f"gslice_coverage_{tag}"] = band["coverage"]
        out[f"gslice_width_{tag}"] = band["mean_width"]
    pits = path_pit_values(gen, obs).reshape(-1)
    out["gslice_pit_mean"] = float(np.mean(pits))
    out["gslice_pit_sd"] = float(np.std(pits))
    out["gslice_pit_ks_pvalue"] = float(kstest(pits, "uniform").pvalue)
    return out


# ---------------------------------------------------------------------------
# SYNTHETIC path laws (correctness tests only — never market evidence)
# ---------------------------------------------------------------------------


def synthetic_gp_paths(
    n_paths: int,
    grid: Array | Sequence[float],
    *,
    kernel: str = "rbf",
    length_scale: float = 0.3,
    amplitude: float = 1.0,
    drift: float = 0.0,
    seed: RNG | None = None,
) -> Array:
    """SYNTHETIC data law: GP paths plus optional linear drift.

    ``drift`` adds drift * (t - t_0) to every path, giving a non-centered
    law the flow must transport to. Labeled SYNTHETIC: a correctness test
    bench, not market evidence.
    """
    t = _as_grid(grid)
    paths = sample_gp_paths(
        n_paths,
        t,
        kernel=kernel,
        length_scale=length_scale,
        amplitude=amplitude,
        rng=seed,
    )
    if drift != 0.0:
        d = float(drift)
        if not math.isfinite(d):
            raise ValueError(f"drift must be finite; got {drift!r}")
        paths = paths + (d * (t - t[0]))[None, :, None]
    return paths


def synthetic_switching_paths(
    n_paths: int,
    grid: Array | Sequence[float],
    *,
    vol_low: float = 0.4,
    vol_high: float = 1.6,
    p_stay: float = 0.9,
    seed: RNG | None = None,
) -> Array:
    """SYNTHETIC data law: random-walk paths with two-state vol regimes.

    Increments are Gaussian with per-step volatility switched by a sticky
    two-state Markov chain (Hamilton 1989-style regime switching) — a
    non-Gaussian path law whose multi-scale structure the SLiCE backbone
    must capture. Labeled SYNTHETIC: a correctness test bench, not market
    evidence.
    """
    n = _check_count(n_paths, "n_paths")
    t = _as_grid(grid)
    vl = _check_positive(vol_low, "vol_low")
    vh = _check_positive(vol_high, "vol_high")
    ps = _check_probability(p_stay, "p_stay")
    gen = _resolve_rng(seed)
    steps = t.shape[0] - 1
    dt = np.diff(t)
    state = np.zeros((n, steps), dtype=int)
    state[:, 0] = gen.integers(0, 2, size=n)
    switch = gen.uniform(size=(n, steps)) > ps
    for j in range(1, steps):
        state[:, j] = np.where(switch[:, j], 1 - state[:, j - 1], state[:, j - 1])
    vols = np.where(state == 0, vl, vh)
    inc = vols * np.sqrt(dt)[None, :] * gen.standard_normal((n, steps))
    paths = np.concatenate([np.zeros((n, 1)), np.cumsum(inc, axis=1)], axis=1)
    return paths[:, :, None]


# ---------------------------------------------------------------------------
# torch layer (lazy — optional ``nn`` extra)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GSliceConfig:
    """Hyperparameters for :func:`train_gslice` / :meth:`GSliceModel.sample`.

    ``block_size`` controls the SLiCE structure family: 1 = diagonal
    (non-universal; ablation), ``hidden_dim`` = dense Linear-NCDE, values in
    between = maximally expressive block-diagonal SLiCE. ``hidden_dim`` must
    be divisible by ``block_size``.
    """

    hidden_dim: int = 16
    n_blocks: int = 2
    block_size: int = 16
    kernel: str = "rbf"
    length_scale: float = 0.3
    amplitude: float = 1.0
    obs_noise: float = 1e-4
    sig_order: int = 2
    flow_steps: int = 24
    epochs: int = 60
    lr: float = 3e-3
    batch_size: int = 64
    ot_couple: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        _check_count(self.hidden_dim, "hidden_dim")
        _check_count(self.n_blocks, "n_blocks")
        _check_count(self.block_size, "block_size")
        if self.hidden_dim % self.block_size != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by block_size={self.block_size}"
            )
        _check_kernel(self.kernel)
        _check_positive(self.length_scale, "length_scale")
        _check_positive(self.amplitude, "amplitude")
        if not math.isfinite(float(self.obs_noise)) or self.obs_noise < 0.0:
            raise ValueError(f"obs_noise must be finite and >= 0; got {self.obs_noise!r}")
        _check_count(self.sig_order, "sig_order")
        if self.sig_order > 4:
            raise ValueError("sig_order must be <= 4 (logsignature limit)")
        _check_count(self.flow_steps, "flow_steps")
        _check_count(self.epochs, "epochs")
        _check_positive(self.lr, "lr")
        _check_count(self.batch_size, "batch_size")


def _expm_scaled(torch: Any, mat: Any, *, taylor_order: int = 8, squarings: int = 4) -> Any:
    """Batched matrix exponential via scaling-and-squaring + truncated Taylor.

    ``expm(M) = (P_order(M / 2^q))^{2^q}`` with ``P_order`` the order-8 Taylor
    polynomial. On the increment scales used by the SLiCE transition
    (‖Σ_i A^i Δω^i‖ ≲ 1) the residual error is ≲ (‖M‖/2^q)^{order+1} ~ 1e-9 —
    the same exact-flow semantics as ``torch.linalg.matrix_exp`` up to
    roundoff, implemented in plain matmuls so autograd differentiates
    through ordinary kernels (the native ``matrix_exp`` backward is not
    used: it is both slower and, on some builds, unreliable here).
    """
    scaled = mat / float(2**squarings)
    eye = torch.eye(mat.shape[-1], dtype=mat.dtype).expand_as(scaled)
    term = eye.clone()
    acc = eye + scaled
    power = scaled
    for k in range(2, taylor_order + 1):
        power = power @ scaled
        term = power / math.factorial(k)
        acc = acc + term
    for _ in range(squarings):
        acc = acc @ acc
    return acc


def _build_field(torch: Any, cfg: GSliceConfig, d_x: int, d_ctx: int, d_aux: int) -> Any:
    """Build the stacked residual SLiCE vector field (torch modules).

    Driving channels ω = [t, x, s, signature-features (d_ctx), aux (d_aux)]
    where aux carries the deterministic context path channels of the paper's
    §2.1 clause ("ω may include time, observed values, masks, lags"): the
    observed-knot mask and the GP-posterior mean path, d_aux = 1 + d_x for
    conditional models and 0 otherwise.
    """

    class _SliceBlock(torch.nn.Module):
        """One SLiCE layer: structured A^i, exact-flow recurrence, readout."""

        def __init__(self) -> None:
            super().__init__()
            n_blocks_mat = cfg.hidden_dim // cfg.block_size
            self.n_mat = n_blocks_mat
            self.b = cfg.block_size
            self.d_omega = 2 + d_x + d_ctx + d_aux  # t + x + s + ctx + aux
            # A^i per driving channel, block-diagonal (K, b, b) per channel.
            # Cold init: iid entries ~N(0, scale²) of a b×b block have spectral
            # radius ~ scale*sqrt(b); keeping it << 1 makes exp(Σ_i A^i Δω^i)
            # start near-identity so the recurrence does not blow up.
            scale = 0.05 / math.sqrt(self.d_omega * self.b)
            self.weight = torch.nn.Parameter(
                torch.randn(self.d_omega, n_blocks_mat, self.b, self.b) * scale
            )
            self.init = torch.nn.Linear(d_x + d_ctx, cfg.hidden_dim)
            self.readout = torch.nn.Linear(cfg.hidden_dim, d_x)

        def hidden(self, omega: Any, h0: Any) -> Any:
            """Exact-flow recurrence h_{j+1} = exp(Σ_i A^i Δω^i_j) h_j."""
            n, g, _ = omega.shape
            d_om = omega[:, 1:, :] - omega[:, :-1, :]  # (n, J, d_ω)
            # per-block generator B_j = Σ_i A^i Δω^i_j → (n, J, K, b, b)
            gen = torch.einsum("kquv,njk->njquv", self.weight, d_om)
            phi = _expm_scaled(torch, gen)  # (n, J, K, b, b) exact-flow
            h = h0.reshape(n, self.n_mat, self.b)
            traj = [h.reshape(n, cfg.hidden_dim)]
            for j in range(g - 1):
                h = torch.einsum("nquv,nqv->nqu", phi[:, j], h)
                traj.append(h.reshape(n, cfg.hidden_dim))
            return torch.stack(traj, dim=1)  # (n, G, d_h)

    class _Field(torch.nn.Module):
        """G-SLiCE vector field F_θ(s, X): residual stack of SLiCE blocks."""

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([_SliceBlock() for _ in range(cfg.n_blocks)])

        def forward(self, x: Any, s: Any, ctx: Any, aux: Any) -> Any:
            n, g, _ = x.shape
            t = torch.linspace(0.0, 1.0, g, dtype=x.dtype).reshape(1, g, 1).expand(n, g, 1)
            sc = s.reshape(n, 1, 1).expand(n, g, 1)
            if d_ctx > 0:
                cc = ctx.reshape(n, 1, d_ctx).expand(n, g, d_ctx)
            else:
                cc = x.new_zeros((n, g, 0))
            z = x
            h0_in = x[:, 0, :] if d_ctx == 0 else torch.cat([x[:, 0, :], ctx], dim=1)
            for k, block in enumerate(self.blocks):
                omega = torch.cat([t, z, sc, cc, aux], dim=2)
                h = block.hidden(omega, block.init(h0_in))  # ξ_φ(X_{t0}, c)
                step = block.readout(h)  # (n, G, d_x)
                z = step if k == 0 else z + step  # residual stack (App. F)
            return z

    return _Field()


def _context_tables(
    cfg: GSliceConfig, grid: Array, ctx: GSliceContext
) -> tuple[Array, list[Array], Array]:
    """Per-row GP-posterior mean, Cholesky, and observed-knot mask.

    The posterior mean path and mask are the deterministic context channels
    the paper's §2.1 allows ω to carry ("observed values, masks"); the
    Cholesky factors are reused to draw μ(·|C_i) prior paths. ``mask[i,j]``
    is 1 when ``grid[j]`` lies within half a grid step of an observation
    time of context row ``i``.
    """
    n_ctx, _, d_x = ctx.obs_values.shape
    n_grid = grid.shape[0]
    means = np.empty((n_ctx, n_grid, d_x), dtype=float)
    chols: list[Array] = []
    half_step = 0.5 * float(grid[1] - grid[0])
    mask = np.zeros((n_ctx, n_grid, 1), dtype=float)
    for i in range(n_ctx):
        mean, cov = gp_posterior_moments(
            grid,
            ctx.obs_times[i],
            ctx.obs_values[i],
            kernel=cfg.kernel,
            length_scale=cfg.length_scale,
            amplitude=cfg.amplitude,
            noise=cfg.obs_noise,
        )
        means[i] = mean
        chols.append(_cholesky_jittered(cov, name="posterior covariance"))
        near = np.min(np.abs(ctx.obs_times[i][:, None] - grid[None, :]), axis=0)
        mask[i, near <= half_step, 0] = 1.0
    return means, chols, mask


def _draw_prior(
    torch: Any,
    cfg: GSliceConfig,
    grid: Array,
    n_paths: int,
    d_x: int,
    gen: np.random.Generator,
) -> Any:
    """Unconditional GP-prior draws; channels independent under the prior."""
    arr = sample_gp_paths(
        n_paths * d_x,
        grid,
        kernel=cfg.kernel,
        length_scale=cfg.length_scale,
        amplitude=cfg.amplitude,
        rng=gen,
    ).reshape(n_paths, grid.shape[0], d_x)
    return torch.as_tensor(arr, dtype=torch.float32)


def _draw_posterior(
    torch: Any,
    means: Array,
    chols: list[Array],
    idx: NDArray[np.integer[Any]],
    gen: np.random.Generator,
) -> Any:
    """Draw n = len(idx) prior paths from the per-row GP posteriors."""
    n = int(idx.shape[0])
    n_grid = means.shape[1]
    d_x = means.shape[2]
    eps = gen.standard_normal((n, n_grid, d_x))
    draws = np.stack(
        [
            means[i] + np.einsum("ij,jk->ik", chols[i], eps[r])
            for r, i in enumerate(np.asarray(idx).tolist())
        ],
        axis=0,
    )
    return torch.as_tensor(draws, dtype=torch.float32)


def train_gslice(
    data_paths: Array,
    *,
    grid: Array | Sequence[float] | None = None,
    context: GSliceContext | None = None,
    config: GSliceConfig | None = None,
) -> GSliceModel:
    """Train the G-SLiCE path-space flow by conditional flow matching.

    Per epoch the data batch is permuted by the seeded generator; for each
    minibatch, noise paths are drawn from the prior μ (unconditional GP, or
    per-path GP posterior on the context), coupled to data by exact
    mini-batch OT assignment when ``config.ot_couple``, and the CFM loss
    E‖F_θ(s, X^(s)) − (X^(1) − X^(0))‖² is minimized with Adam. Deterministic
    given ``config.seed`` on CPU single-thread. ``data_paths`` is
    (n, n_grid, d) — ``grid`` defaults to linspace(0, 1, n_grid).
    """
    torch = _torch()
    cfg = config if config is not None else GSliceConfig()
    data = _as_paths3(data_paths, name="data_paths")
    n_data, n_grid, d_x = data.shape
    _check_count(n_data, "data_paths", minimum=4)
    g = np.linspace(0.0, 1.0, n_grid) if grid is None else _as_grid(grid)
    if g.shape[0] != n_grid:
        raise ValueError(f"grid length {g.shape[0]} != data n_grid {n_grid}")
    d_ctx = 0
    d_aux = 0
    feats: Array | None = None
    post_mean: Array | None = None
    post_chol: list[Array] = []
    aux_np: Array | None = None
    if context is not None:
        if context.obs_values.shape[0] != n_data:
            raise ValueError(
                f"context must have one row per data path ({n_data}); "
                f"got {context.obs_values.shape[0]}"
            )
        if context.obs_values.shape[2] != d_x:
            raise ValueError(f"context channel count {context.obs_values.shape[2]} != data d={d_x}")
        feats = np.asarray(context.features, dtype=float)
        if feats.ndim != 2 or feats.shape[0] != n_data:
            raise ValueError("context.features must be (n_data, d_sig)")
        d_ctx = int(feats.shape[1])
        post_mean, post_chol, knot_mask = _context_tables(cfg, g, context)
        aux_np = np.concatenate([knot_mask, post_mean], axis=2)  # (n, G, 1+d)
        d_aux = 1 + d_x

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    gen = np.random.default_rng(cfg.seed)
    net = _build_field(torch, cfg, d_x, d_ctx, d_aux)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    data_t = torch.as_tensor(data, dtype=torch.float32)
    ctx_t = (
        torch.as_tensor(feats, dtype=torch.float32)
        if feats is not None
        else torch.zeros((n_data, 0), dtype=torch.float32)
    )
    aux_t = (
        torch.as_tensor(aux_np, dtype=torch.float32)
        if aux_np is not None
        else torch.zeros((n_data, n_grid, 0), dtype=torch.float32)
    )
    curve: list[float] = []
    net.train()
    for _epoch in range(cfg.epochs):
        perm = gen.permutation(n_data)
        epoch_losses: list[float] = []
        for start in range(0, n_data, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            b = int(idx.shape[0])
            x1 = data_t[idx]
            c_b = ctx_t[idx]
            a_b = aux_t[idx]
            if context is None:
                x0 = _draw_prior(torch, cfg, g, b, d_x, gen)
            else:
                if not (post_mean is not None):
                    raise ValueError("post_mean is not None")  # set when context is given
                x0 = _draw_posterior(torch, post_mean, post_chol, idx, gen)
            if cfg.ot_couple and b >= 2:
                perm_ot = ot_coupling(x0.numpy(), x1.numpy())
                perm_t = torch.as_tensor(perm_ot)
                x1 = x1[perm_t]
                c_b = c_b[perm_t]
                a_b = a_b[perm_t]
            s = torch.as_tensor(gen.uniform(0.0, 1.0, size=b), dtype=torch.float32)
            xs = (1.0 - s.reshape(b, 1, 1)) * x0 + s.reshape(b, 1, 1) * x1
            target = x1 - x0
            pred = net(xs, s, c_b, a_b)
            loss = torch.mean((pred - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            lval = float(loss.detach().numpy())
            if not math.isfinite(lval):
                raise ValueError(
                    "non-finite CFM loss during training (diverged field; lower lr or hidden_dim)"
                )
            epoch_losses.append(lval)
        curve.append(float(np.mean(epoch_losses)))
    net.eval()
    return GSliceModel(
        net=net,
        grid=g,
        config=cfg,
        d_x=d_x,
        d_ctx=d_ctx,
        d_aux=d_aux,
        loss_curve=np.asarray(curve, dtype=float),
        train_n=n_data,
    )


@dataclass(frozen=True)
class GSliceModel:
    """Trained G-SLiCE path-space flow + training diagnostics.

    ``net`` is the torch vector field (frozen); sampling methods integrate
    the learned path-ODE from the (possibly conditional) GP prior.
    ``loss_curve`` is per-epoch mean CFM loss.
    """

    net: Any = field(repr=False, compare=False)
    grid: Array = field(compare=False)
    config: GSliceConfig = field(compare=False)
    d_x: int = field(compare=False)
    d_ctx: int = field(compare=False)
    d_aux: int = field(compare=False)
    loss_curve: Array = field(compare=False)
    train_n: int = field(compare=False)

    def _velocity(self, torch: Any, x_t: Any, s: float, ctx_t: Any, aux_t: Any) -> Any:
        s_t = torch.full((int(x_t.shape[0]),), float(s), dtype=torch.float32)
        return self.net(x_t, s_t, ctx_t, aux_t)

    def _integrate(
        self,
        torch: Any,
        x_t: Any,
        ctx_t: Any,
        aux_t: Any,
        flow_steps: int,
        method: str,
        pin_knots: NDArray[np.intp] | None = None,
    ) -> Any:
        """Integrate dX/ds = F_θ(s, X) over s ∈ [0, 1] (Euler or midpoint).

        ``pin_knots`` lists observed grid indices: their values are held at
        the GP-posterior draw after every step (imputation-style pinning),
        so generated paths reproduce the observation up to obs noise while
        the learned flow drives all unobserved positions.
        """
        dt = 1.0 / float(flow_steps)
        x_pin = x_t
        with torch.no_grad():
            for k in range(flow_steps):
                s = k * dt
                if method == "euler":
                    x_t = x_t + dt * self._velocity(torch, x_t, s, ctx_t, aux_t)
                else:  # midpoint RK2
                    k1 = self._velocity(torch, x_t, s, ctx_t, aux_t)
                    mid = x_t + 0.5 * dt * k1
                    k2 = self._velocity(torch, mid, s + 0.5 * dt, ctx_t, aux_t)
                    x_t = x_t + dt * k2
                if pin_knots is not None and len(pin_knots) > 0:
                    x_t[:, pin_knots, :] = x_pin[:, pin_knots, :]
        return x_t

    def sample(
        self,
        n_paths: int,
        *,
        context: GSliceContext | None = None,
        flow_steps: int | None = None,
        method: str = "euler",
        seed: RNG | None = None,
    ) -> Array:
        """Generate paths: X^(0) ~ μ (GP prior / posterior), flow to s = 1.

        Unconditional: fresh GP-prior draws; the model must have been
        trained without a signature context (d_ctx = 0). Conditional: pass
        one ``GSliceContext`` whose single row describes the observed prefix
        — every sample shares its signature features and draws from its GP
        posterior. ``method`` is ``"euler"`` or ``"midpoint"``.
        Deterministic given ``seed``.
        """
        torch = _torch()
        n = _check_count(n_paths, "n_paths")
        steps = (
            self.config.flow_steps if flow_steps is None else _check_count(flow_steps, "flow_steps")
        )
        if method not in ("euler", "midpoint"):
            raise ValueError(f"method must be 'euler' or 'midpoint'; got {method!r}")
        gen = _resolve_rng(0 if seed is None else seed)
        pin_knots: NDArray[np.intp] | None = None
        if context is None:
            if self.d_ctx > 0:
                raise ValueError("model was trained with a signature context; pass context=...")
            x0 = _draw_prior(torch, self.config, self.grid, n, self.d_x, gen)
            ctx_t = torch.zeros((n, 0), dtype=torch.float32)
            aux_t = torch.zeros((n, self.grid.shape[0], 0), dtype=torch.float32)
        else:
            if self.d_aux == 0:
                raise ValueError(
                    "model was trained without a context; conditional sampling "
                    "requires a context-trained model"
                )
            if context.obs_values.shape[0] != 1:
                raise ValueError(
                    "sampling expects a single shared context row "
                    f"(n_context=1); got {context.obs_values.shape[0]}"
                )
            if context.features.shape[1] != self.d_ctx:
                raise ValueError(
                    f"context feature dim {context.features.shape[1]} != trained "
                    f"d_ctx {self.d_ctx} (rebuild with sig_order={self.config.sig_order})"
                )
            if context.obs_values.shape[2] != self.d_x:
                raise ValueError(
                    f"context channel count {context.obs_values.shape[2]} != trained d={self.d_x}"
                )
            post_mean, post_chol, knot_mask = _context_tables(self.config, self.grid, context)
            idx = np.zeros(n, dtype=int)
            x0 = _draw_posterior(torch, post_mean, post_chol, idx, gen)
            aux_np = np.concatenate([knot_mask, post_mean], axis=2)
            aux_t = torch.as_tensor(aux_np, dtype=torch.float32).expand(n, -1, -1)
            ctx_t = torch.as_tensor(context.features, dtype=torch.float32)
            ctx_t = ctx_t.expand(n, self.d_ctx)
            pin_knots = np.flatnonzero(knot_mask[0, :, 0] > 0.0)
        out = self._integrate(torch, x0, ctx_t, aux_t, steps, method, pin_knots=pin_knots)
        return np.asarray(out.numpy(), dtype=float)

    def sample_conditional(
        self,
        n_paths: int,
        obs_times: Array | Sequence[float],
        obs_values: Array,
        *,
        flow_steps: int | None = None,
        method: str = "euler",
        seed: RNG | None = None,
    ) -> Array:
        """Conditional generation given one observed prefix (paper §3).

        Convenience wrapper: builds the :class:`GSliceContext` from the
        observation (signature features at ``config.sig_order``) and samples
        from the GP-posterior prior transported by the learned flow.
        ``obs_values`` is (1, m, d) or (m, d).
        """
        y = np.asarray(obs_values, dtype=float)
        if y.ndim == 2:
            y = y[None, :, :]
        ctx = build_context(obs_times, y, order=self.config.sig_order)
        return self.sample(n_paths, context=ctx, flow_steps=flow_steps, method=method, seed=seed)


def bench_gslice(
    *,
    n_train: int = 256,
    n_eval: int = 64,
    n_grid: int = 17,
    seed: int = 0,
    config: GSliceConfig | None = None,
) -> dict[str, float]:
    """Seeded SYNTHETIC bench: G-SLiCE learns a regime-switching path law.

    Trains on ``synthetic_switching_paths`` (non-Gaussian increments),
    samples an ensemble, and scores it against a held-out independently
    seeded batch of the same law plus a raw GP-prior ensemble baseline.
    All keys are ``gslice_synth_*`` — SYNTHETIC correctness evidence only,
    never market evidence (AGENTS.md).
    """
    cfg = config if config is not None else GSliceConfig(epochs=40, hidden_dim=16)
    grid = np.linspace(0.0, 1.0, n_grid)
    train = synthetic_switching_paths(n_train, grid, seed=seed)
    held = synthetic_switching_paths(n_eval, grid, seed=seed + 10_000)
    model = train_gslice(train, grid=grid, config=cfg)
    gen = model.sample(n_eval, seed=seed + 20_000)
    prior = sample_gp_paths(
        n_eval,
        grid,
        kernel=cfg.kernel,
        length_scale=cfg.length_scale,
        amplitude=cfg.amplitude,
        rng=seed + 30_000,
    )
    ev_gen = evaluate_samples(gen, held)
    ev_prior = evaluate_samples(prior, held)
    out: dict[str, float] = {"gslice_synth_" + k[len("gslice_") :]: v for k, v in ev_gen.items()}
    out["synthetic_gslice_synth_energy_score_prior"] = ev_prior["gslice_energy_score_mean"]
    out["synthetic_gslice_synth_energy_score_gain"] = (
        ev_prior["gslice_energy_score_mean"] - ev_gen["gslice_energy_score_mean"]
    )
    out["synthetic_gslice_synth_final_loss"] = float(model.loss_curve[-1])
    out["synthetic_gslice_synth_loss_drop"] = float(model.loss_curve[0] - model.loss_curve[-1])
    return out
