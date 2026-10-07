"""Deep kernel hedging: hedge in an RKHS with a learned neural kernel.

Research-grade minimal implementation of the deep-kernel hedging framework:

- Dupret, J.-L., Hainaut, D. & Motte, E. (2026), "Deep kernel hedging",
  arXiv:2609.34474, https://doi.org/10.48550/arXiv.2609.34474 (cs.LG;
  q-fin.CP), submitted 28 Sep 2026. The hedging functional xi = phi(X) is
  restricted to the RKHS H_{K_theta} of a *deep kernel* K_theta(x, x') =
  K_RBF(psi_theta(x), psi_theta(x')) whose latent map psi_theta =
  psi_tilde_omega / gamma is an l2-normalized shallow MLP output divided by a
  learned scale gamma (Dupret et al. 2026, §4; deep kernel learning after
  Wilson, Hu, Salakhutdinov & Xing 2016, "Deep kernel learning", AISTATS,
  arXiv:1511.02222). The regularized empirical problem (their eq. (3.5))
  minimizes (1/N) sum_i L(E_i(phi)) + lambda ||phi||^2 over the terminal
  hedging error E_i(phi; v0) = H^i - B_T^i v0 - sum_k phi(X^i_{t_k})
  G^i_{t_{k+1}} with self-financing discounted gains G^i_{t_{k+1}} =
  B_T^i (S^i_{t_{k+1}}/B^i_{t_{k+1}} - S^i_{t_k}/B^i_{t_k}).

What is implemented here, keyed to the paper:

1. *Representer-theorem reduction* (Dupret et al. 2026, Thm 3.1 and the
   remark after it; generalized representer theorem in the sense of
   Schölkopf, Herbrich & Smola 2001, "A generalized representer theorem",
   COLT, LNCS 2111, 416–426): for fixed theta the unique minimizer is
   phi*(.) = sum_i alpha_i g_{theta,i}(.) with the gain-weighted kernel
   sections g_{theta,i}(.) = sum_k G^i_{t_{k+1}} K_theta(X^i_{t_k}, .), and
   alpha solves the finite-dimensional problem min_alpha (1/N) sum_i
   L((H^{v0} - Q_theta alpha)_i) + lambda alpha' Q_theta alpha over the
   *hedging Gram matrix* Q_ij = sum_{k,l} G^i_{t_{k+1}} G^j_{t_{l+1}}
   K_theta(X^i_{t_k}, X^j_{t_l}) (their eq. (3.7)). For the square loss the
   reduction is the regularized linear solve alpha* = (Q + N lambda I)^{-1}
   H^{v0}. NOTE: this is a solve against the payoff target through the
   gain-weighted Gram matrix — not kernel ridge regression against a
   "hedging loss gradient"; the loss enters only through the convex
   objective (closed form for the square loss). Implemented torch-free in
   :func:`exact_kernel_hedge` / :func:`hedging_gram` /
   :func:`solve_representer_alpha` with the base RBF kernel on standardized
   signature features (psi = identity, the paper's "kernel hedging"
   benchmark), i.e. v0 = 0 and B = 1 (zero rates) are fixed; the paper
   learns v0 for quadratic hedging, which leaves the optimal phi unchanged
   when the gains are martingale increments (mu = 0).
2. *Time-augmented signature features* (Dupret et al. 2026, §4.2 and
   Appendix B): X_{t_k} is the truncated time-augmented signature
   Sig^{<=M}((t, Z)_{[t_0, t_k]}) of the market path, computed with the
   tensor-algebra conventions of :mod:`quant_fund.models.path_signatures`
   (Chen, K.-T. 1958; Chevyrev & Kormilitzin 2016/2025, "A primer on the
   signature method in machine learning", §3.1.4 base-point augmentation).
   Channels are (elapsed-time fraction, log-price relative to S_0), so every
   path starts at the origin and the base-point augmentation is a no-op.
   Prefix signatures are accumulated incrementally via Chen's identity
   S <- S (x) exp(dx) (exact; cross-checked in tests against the public
   :func:`~quant_fund.models.path_signatures.signature`), or computed
   directly per prefix with ``incremental=False``.
3. *Random Fourier features* (Rahimi & Recht 2007, "Random features for
   large-scale kernel machines", NeurIPS; Dupret et al. 2026, §2.3, §3.3):
   y_theta(x) = sqrt(2/D) cos(W psi_theta(x) + b) with (W_j, b_j) ~
   N(0, I_p) x U(0, 2 pi) sampled ONCE and kept fixed (w.r.t. the unit
   bandwidth RBF; gamma is absorbed into the latent map, their eq. (4.2)).
   The primal RFF reduction (their Prop 2.1 / Lemma 3.2) gives
   phi*_D(x) = beta' y_theta(x) with, for the square loss,
   beta* = (Z'Z + lambda N I_D)^{-1} Z'H^{v0}, where the aggregated
   feature rows are Z_i = sum_k y_theta(X^i_{t_k}) G^i_{t_{k+1}} (their
   eq. (3.14)). Approximation error: the RFF estimator is unbiased and,
   uniformly over x, x' (and theta on compacta),
   P(sup |K~_{theta,D} - K_theta| > eps) <= C eps^{-2} exp(-c D eps^2)
   (their Prop 3.2, extending Rahimi & Recht 2007 Claim 1), i.e. the
   sup-norm error decays as O(D^{-1/2}) and the hedging Gram matrices
   converge in operator norm. Tests pin the empirical decay.
4. *Joint deep training* (their Algorithms 1 and 2): full-batch AdamW
   (weight decay 5e-3, cosine lr schedule) on (omega, log gamma, beta[,
   zeta]) minimizing the quadratic objective (1/N)||H^{v0} - Z_theta
   beta||^2 + lambda ||beta||^2 or the CVaR objective zeta + (1/(alpha N))
   sum_i (E_i - zeta)_+ + lambda ||beta||^2 via the Rockafellar–Uryasev
   representation (Rockafellar & Uryasev 2000, Journal of Risk 2(3), 21–41;
   Dupret et al. 2026, §4.3.2, v0 fixed). The paper uses mini-batch
   updates; full-batch steps are used here for determinism, matching the
   sibling :mod:`quant_fund.models.deep_hedging` convention. Convergence of
   the RFF hedging problems as D -> infinity is their Thm 3.3 (optimal
   values and strategies converge a.s.).

Baselines and simulation live in :mod:`quant_fund.models.deep_hedging`
(Buehler, Gonon, Teichmann & Wood 2019, "Deep hedging", Quantitative
Finance 19(8), 1271–1291, arXiv:1802.03042); with B = 1 and v0 = 0 the
paper's terminal hedging error E_i(phi; 0) coincides exactly with that
module's per-path ``loss`` at ``cost_rate=0``, which tests assert.
Transaction costs are OUTSIDE the paper's model (costs make E_i nonlinear
in phi and break the representer reduction); learned strategies can still
be evaluated under friction via
:func:`~quant_fund.models.deep_hedging.hedged_pnl_components`.

Honesty: everything here runs on SYNTHETIC simulated paths. Outputs are
risk measures (variance / CVaR) of simulated hedged P&L — algorithmic
correctness evidence, never market evidence. No Sharpe/Sortino/P&L
headline; no live-trading claims (AGENTS.md honesty contract). Comparisons
on the training batch are in-sample; evaluate strategies on fresh
independently seeded paths for an honest train/eval split.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` exactly like ``deep_hedging.py``, so this module imports
cleanly without torch and every torch entry point raises ``ImportError``
with install guidance. The numpy core (signature features, Gram matrix,
representer solve, RFF draw/features/solve, exact and RFF kernel hedges)
is fully usable torch-free. Fail-closed edges: ``ValueError`` on invalid
paths/payoffs, non-positive lambda/gamma/bandwidth/feature counts, unknown
loss kinds, out-of-range CVaR levels or signature orders, and shape
mismatches. Training is CPU single-thread full-batch and deterministic
given ``seed`` (GPU determinism is not claimed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.path_signatures import signature

Array = NDArray[np.float64]

__all__ = [
    "DeepKernelHedgeResult",
    "ExactKernelHedge",
    "RffKernelHedge",
    "deep_kernel_hedge",
    "exact_kernel_hedge",
    "hedged_error",
    "hedging_gram",
    "rbf_kernel_matrix",
    "rff_aggregate",
    "rff_draw",
    "rff_features",
    "rff_kernel_hedge",
    "signature_features",
    "solve_representer_alpha",
    "solve_rff_beta",
    "standardize_features",
]

_LOSS_KINDS = ("quadratic", "cvar")
_MAX_SIG_ORDER = 6  # mirrors path_signatures._MAX_ORDER (public bound)
_SIG_CHANNELS = 2  # (time fraction, log-relative price)
_NORM_EPS = 1e-6  # l2-normalization floor (paper: eps_norm = 1e-6)
_STD_FLOOR = 1e-12


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "deep kernel hedging needs the optional 'nn' extra (torch): uv sync --extra nn"
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


def _check_finite(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return v


def _as_paths(paths: Array | Sequence[Sequence[float]], *, name: str = "paths") -> Array:
    """Validate price paths (same contract as ``deep_hedging._as_paths``)."""
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


def _check_order(order: int) -> int:
    m = int(order)
    if isinstance(order, bool) or m != order or m < 1 or m > _MAX_SIG_ORDER:
        raise ValueError(f"order must be an integer in [1, {_MAX_SIG_ORDER}]; got {order!r}")
    return m


def _check_payoff(payoff: Array | Sequence[float], n_paths: int) -> Array:
    pay = np.asarray(payoff, dtype=float).reshape(-1)
    if pay.shape[0] != n_paths or not bool(np.isfinite(pay).all()):
        raise ValueError(f"payoff must be a finite vector of length n_paths={n_paths}")
    return pay


def signature_features(
    paths: Array | Sequence[Sequence[float]],
    order: int,
    *,
    incremental: bool = True,
) -> Array:
    """Time-augmented truncated signature features per rebalancing time.

    For each path, the adapted feature at rebalancing time ``t_k`` is the
    truncated signature of order ``order`` of the time-augmented 2-channel
    path ``((t_j / T, log(S_j / S_0)))_{j <= k}`` — Dupret, Hainaut & Motte
    (2026, §4.2, Appendix B), the ``X_{t_k} = hat-Sig^{<=M}_{t_k}`` choice
    (Abi Jaber & Gérard 2025, arXiv:2508.02759; Cuchiero, Primavera &
    Svaluto-Ferro 2025, Finance and Stochastics). Channels start at the
    origin, so the base-point augmentation of Chevyrev & Kormilitzin (2025,
    §3.1.4) is a no-op. Level 0 (the scalar 1) is omitted, matching
    :func:`~quant_fund.models.path_signatures.signature`; at ``k = 0`` the
    prefix is a single point and all levels >= 1 vanish (zero features).

    ``incremental=True`` (default) accumulates prefix signatures by Chen's
    identity ``S <- S (x) exp(dx)`` (Chen 1958) with ``exp(dx)`` the exact
    signature of one straight increment (levels ``dx^{(x)k} / k!``); the
    layout and values are identical to calling the public
    :func:`~quant_fund.models.path_signatures.signature` per prefix
    (``incremental=False``), which tests pin exactly.

    Returns ``(n_paths, n_steps, dim)`` with ``dim = sum_{l=1..order} 2^l =
    2^(order+1) - 2``; SYNTHETIC-path feature construction only.
    """
    arr = _as_paths(paths)
    m = _check_order(order)
    n_paths, n_cols = arr.shape
    n_steps = n_cols - 1
    dim = 2 ** (m + 1) - 2
    t_frac = np.arange(n_cols, dtype=float) / float(n_steps)
    log_rel = np.log(arr / arr[:, :1])
    feats = np.zeros((n_paths, n_steps, dim), dtype=float)
    for i in range(n_paths):
        pts = np.column_stack([t_frac, log_rel[i]])  # (n_cols, 2), starts at origin
        if not incremental:
            for k in range(1, n_steps):
                feats[i, k] = signature(pts[: k + 1], m)
            continue
        levels = [np.zeros((_SIG_CHANNELS,) * k, dtype=float) for k in range(1, m + 1)]
        for k in range(1, n_steps):
            dx = pts[k] - pts[k - 1]
            # exp(dx): level k of the signature of the straight increment.
            inc: list[Array] = [dx.reshape(_SIG_CHANNELS)]
            for j in range(2, m + 1):
                inc.append(np.multiply.outer(inc[-1], dx) / float(j))
            merged: list[Array] = []
            for lev in range(1, m + 1):
                acc = levels[lev - 1] + inc[lev - 1]
                for a in range(1, lev):
                    acc = acc + np.multiply.outer(levels[a - 1], inc[lev - a - 1])
                merged.append(acc)
            levels = merged
            feats[i, k] = np.concatenate([t.reshape(-1) for t in levels])
    return feats


def standardize_features(
    feats: Array,
    mean: Array | None = None,
    std: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Per-coordinate standardization of ``(n_paths, n_steps, d)`` features.

    The paper standardizes inputs before the neural network (Dupret et al.
    2026, §4.1). With ``mean``/``std`` given (training statistics), applies
    them; otherwise computes them over the (path, time) axes. Zero-variance
    coordinates fall back to unit scale (never divide by ~0).
    """
    x = np.asarray(feats, dtype=float)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(
            f"feats must be 3-D (n_paths, n_steps, d) with positive sizes; got {x.shape}"
        )
    if not bool(np.isfinite(x).all()):
        raise ValueError("feats entries must be finite (NaN/inf rejected)")
    if mean is None and std is None:
        mu = x.mean(axis=(0, 1))
        sd = x.std(axis=(0, 1))
        sd = np.where(sd > _STD_FLOOR, sd, 1.0)
    else:
        mu = np.asarray(mean, dtype=float).reshape(-1)
        sd = np.asarray(std, dtype=float).reshape(-1)
        if mu.shape != (x.shape[2],) or sd.shape != (x.shape[2],):
            raise ValueError(
                f"mean/std must have length d={x.shape[2]}; got {mu.shape} and {sd.shape}"
            )
        if not bool(np.isfinite(mu).all()) or not bool(np.isfinite(sd).all()):
            raise ValueError("mean/std must be finite")
        if not bool(np.all(sd > 0.0)):
            raise ValueError("std entries must be strictly positive")
    return (x - mu) / sd, mu, sd


def rbf_kernel_matrix(a: Array, b: Array, gamma: float) -> Array:
    """Base RBF kernel ``exp(-||z - z'||^2 / (2 gamma^2))`` on latent rows.

    Normalized (k(z, z) = 1), symmetric, positive definite; the base kernel
    of the deep construction K_theta = K_RBF(psi_theta(x), psi_theta(x'))
    (Dupret et al. 2026, eq. (2.4), §4.1). ``a`` is ``(m, p)``, ``b`` is
    ``(n, p)``; returns ``(m, n)``.
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    g = _check_positive(gamma, "gamma")
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] < 1 or B.shape[0] < 1:
        raise ValueError(f"a and b must be non-empty 2-D latent matrices; got {A.shape}, {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"latent dims must match; got {A.shape[1]} and {B.shape[1]}")
    if not bool(np.isfinite(A).all()) or not bool(np.isfinite(B).all()):
        raise ValueError("latent entries must be finite (NaN/inf rejected)")
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(np.exp(-d2 / (2.0 * g * g)), dtype=float)


def hedging_gram(kernel_flat: Array, gains: Array) -> Array:
    """Hedging Gram matrix Q_theta (Dupret et al. 2026, eq. (3.7)).

    ``kernel_flat`` is the ``(N*n, N*n)`` kernel matrix over all flattened
    training points ``(i, k)`` (path i, rebalancing time t_k), ``gains`` the
    ``(N, n)`` self-financing gains ``G^i_{t_{k+1}}``. Returns the symmetric
    ``(N, N)`` matrix ``Q_ij = sum_{k,l} G^i_k G^j_l K(x^i_k, x^j_l)`` —
    the inner products <g_i, g_j> of the gain-weighted kernel sections
    g_i = sum_k G^i_k K(x^i_k, .). Costs O(N^2 n^2) time and memory, the
    bottleneck the paper's RFF scheme removes.
    """
    K = np.asarray(kernel_flat, dtype=float)
    G = np.asarray(gains, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] < 1:
        raise ValueError(f"kernel_flat must be a square non-empty matrix; got {K.shape}")
    if not bool(np.isfinite(K).all()):
        raise ValueError("kernel_flat entries must be finite (NaN/inf rejected)")
    if G.ndim != 2 or G.shape[0] < 1 or G.shape[1] < 1:
        raise ValueError(f"gains must be a non-empty 2-D (n_paths, n_steps) array; got {G.shape}")
    n_paths, n_steps = G.shape
    if K.shape[0] != n_paths * n_steps:
        raise ValueError(
            f"kernel_flat must be (n_paths*n_steps)^2 = ({n_paths * n_steps},) ^2; got {K.shape}"
        )
    if not bool(np.isfinite(G).all()):
        raise ValueError("gains entries must be finite (NaN/inf rejected)")
    K4 = K.reshape(n_paths, n_steps, n_paths, n_steps)
    Q = np.einsum("ik,ikjl,jl->ij", G, K4, G, optimize=True)
    return np.asarray((Q + Q.T) / 2.0, dtype=float)  # symmetrize float noise


def solve_representer_alpha(Q: Array, target: Array, lam: float) -> Array:
    """Square-loss representer coefficients alpha* = (Q + N lam I)^{-1} H.

    The closed-form reduction of the finite-dimensional problem (Dupret et
    al. 2026, eq. (3.8) with L(u) = u^2 and the remark after Thm 3.1;
    N = len(target), v0 = 0 so H^{v0} = payoff vector). Q is PSD and
    lam > 0 makes the system positive definite; a singular system fails
    closed with ``ValueError``.
    """
    Qm = np.asarray(Q, dtype=float)
    h = np.asarray(target, dtype=float).reshape(-1)
    lmb = _check_positive(lam, "lam")
    if Qm.ndim != 2 or Qm.shape[0] != Qm.shape[1] or Qm.shape[0] < 1:
        raise ValueError(f"Q must be a square non-empty matrix; got {Qm.shape}")
    if Qm.shape[0] != h.shape[0]:
        raise ValueError(f"target must have length N={Qm.shape[0]}; got {h.shape[0]}")
    if not bool(np.isfinite(Qm).all()) or not bool(np.isfinite(h).all()):
        raise ValueError("Q and target entries must be finite (NaN/inf rejected)")
    n = h.shape[0]
    system = Qm + float(n) * lmb * np.eye(n)
    try:
        alpha = np.linalg.solve(system, h)
    except np.linalg.LinAlgError as exc:
        raise ValueError("representer system is singular (lam > 0 should prevent this)") from exc
    if not bool(np.isfinite(alpha).all()):
        raise ArithmeticError("representer solve produced non-finite coefficients")
    return alpha


def rff_draw(p_dim: int, n_features: int, seed: int) -> tuple[Array, Array]:
    """Draw fixed RFF parameters ``(W, b)`` for the unit-bandwidth RBF kernel.

    Spectral measure of K_RBF with unit bandwidth: ``W`` rows ~ N(0, I_p),
    ``b`` ~ U(0, 2 pi) — sampled ONCE and kept fixed throughout training
    (Rahimi & Recht 2007; Dupret et al. 2026, §4.1: gamma is absorbed into
    the latent map so the sampling law never depends on learned parameters).
    Returns ``W`` of shape ``(n_features, p_dim)`` and ``b`` of shape
    ``(n_features,)``, seeded via ``numpy.random.default_rng(seed)``.
    """
    p = _check_count(p_dim, "p_dim")
    d = _check_count(n_features, "n_features")
    rng = np.random.default_rng(int(seed))
    W = rng.standard_normal((d, p))
    b = rng.uniform(0.0, 2.0 * math.pi, d)
    return W, b


def rff_features(latent: Array, W: Array, b: Array) -> Array:
    """Random-Fourier feature map y(x) = sqrt(2/D) cos(W z + b).

    Unbiased estimator of the unit-bandwidth RBF kernel:
    ``y(z)' y(z') -> K_RBF(z, z')`` a.s. as D -> infinity, with the uniform
    tail bound P(sup |y'y - K| > eps) <= C eps^{-2} exp(-c D eps^2)
    (Rahimi & Recht 2007, Claim 1; Dupret et al. 2026, Prop 3.2). ``latent``
    is ``(m, p)`` (already scaled by 1/gamma); returns ``(m, D)``.
    """
    Z = np.asarray(latent, dtype=float)
    Wm = np.asarray(W, dtype=float)
    bv = np.asarray(b, dtype=float).reshape(-1)
    if Z.ndim != 2 or Z.shape[0] < 1:
        raise ValueError(f"latent must be a non-empty 2-D array; got {Z.shape}")
    if Wm.ndim != 2 or Wm.shape[0] < 1:
        raise ValueError(f"W must be a non-empty 2-D (D, p) array; got {Wm.shape}")
    if Z.shape[1] != Wm.shape[1]:
        raise ValueError(f"latent dim p must match W columns; got {Z.shape[1]} and {Wm.shape[1]}")
    if bv.shape[0] != Wm.shape[0]:
        raise ValueError(f"b must have length D={Wm.shape[0]}; got {bv.shape[0]}")
    if not (
        bool(np.isfinite(Z).all()) and bool(np.isfinite(Wm).all()) and bool(np.isfinite(bv).all())
    ):
        raise ValueError("latent/W/b entries must be finite (NaN/inf rejected)")
    d_feat = Wm.shape[0]
    return np.asarray(math.sqrt(2.0 / d_feat) * np.cos(Z @ Wm.T + bv[None, :]), dtype=float)


def rff_aggregate(y_feats: Array, gains: Array) -> Array:
    """Aggregated RFF design matrix Z_i = sum_k y(x_{i,k}) G^i_k.

    ``y_feats`` is ``(N, n, D)`` per-step RFF features, ``gains`` is
    ``(N, n)``; returns ``(N, D)`` — the paper's Z_theta (Dupret et al.
    2026, eq. (3.14)), whose Gram Z Z' approximates the hedging Gram
    Q_theta (their Lemma 3.2).
    """
    Y = np.asarray(y_feats, dtype=float)
    G = np.asarray(gains, dtype=float)
    if Y.ndim != 3 or Y.shape[0] < 1 or Y.shape[1] < 1 or Y.shape[2] < 1:
        raise ValueError(f"y_feats must be 3-D (n_paths, n_steps, D); got {Y.shape}")
    if G.shape != Y.shape[:2]:
        raise ValueError(f"gains must have shape {Y.shape[:2]}; got {G.shape}")
    if not bool(np.isfinite(Y).all()) or not bool(np.isfinite(G).all()):
        raise ValueError("y_feats/gains entries must be finite (NaN/inf rejected)")
    return np.asarray(np.einsum("ikd,ik->id", Y, G), dtype=float)


def solve_rff_beta(Z: Array, target: Array, lam: float) -> Array:
    """Square-loss RFF primal solve beta* = (Z'Z + lam N I_D)^{-1} Z' H.

    Primal normal equations of the RFF hedging problem (Dupret et al. 2026,
    eq. (2.11)/(3.16) with L(u) = u^2, v0 = 0); equivalent to the dual
    alpha-formulation on Q~ = Z Z' by their Prop 2.1 / Lemma 3.2. Cost
    O(N D^2) + O(D^3) versus O(N^2 n^2) for the exact Gram. Fails closed on
    a singular system (lam > 0 makes Z'Z + lam N I positive definite).
    """
    Zm = np.asarray(Z, dtype=float)
    h = np.asarray(target, dtype=float).reshape(-1)
    lmb = _check_positive(lam, "lam")
    if Zm.ndim != 2 or Zm.shape[0] < 1 or Zm.shape[1] < 1:
        raise ValueError(f"Z must be a non-empty 2-D (N, D) matrix; got {Zm.shape}")
    if Zm.shape[0] != h.shape[0]:
        raise ValueError(f"target must have length N={Zm.shape[0]}; got {h.shape[0]}")
    if not bool(np.isfinite(Zm).all()) or not bool(np.isfinite(h).all()):
        raise ValueError("Z and target entries must be finite (NaN/inf rejected)")
    n, d_feat = Zm.shape
    system = Zm.T @ Zm + lmb * float(n) * np.eye(d_feat)
    try:
        beta = np.linalg.solve(system, Zm.T @ h)
    except np.linalg.LinAlgError as exc:
        raise ValueError("RFF normal equations are singular (lam > 0 should prevent this)") from exc
    if not bool(np.isfinite(beta).all()):
        raise ArithmeticError("RFF solve produced non-finite coefficients")
    return beta


def hedged_error(paths: Array, positions: Array, payoff: Array) -> Array:
    """Terminal hedging error E_i(phi; v0=0) with B = 1 (zero rates).

    ``E_i = H^i - sum_k phi(x^i_{t_k}) (S^i_{t_{k+1}} - S^i_{t_k})`` — the
    paper's eq. (3.4) at v0 = 0, B_T = 1. Equals
    :func:`~quant_fund.models.deep_hedging.hedged_loss` at ``cost_rate=0``
    (tests pin this). No transaction costs: frictions are outside the
    paper's representer framework.
    """
    arr = _as_paths(paths)
    n_paths, n_steps = arr.shape[0], arr.shape[1] - 1
    pos = np.asarray(positions, dtype=float)
    if pos.shape != (n_paths, n_steps):
        raise ValueError(f"positions must have shape ({n_paths}, {n_steps}); got {pos.shape}")
    if not bool(np.isfinite(pos).all()):
        raise ValueError("positions must be finite (NaN/inf rejected)")
    pay = _check_payoff(payoff, n_paths)
    gains = arr[:, 1:] - arr[:, :-1]
    return pay - np.sum(pos * gains, axis=1)


def _gains(arr: Array) -> Array:
    """Self-financing gains G^i_k = S^i_{t_{k+1}} - S^i_{t_k} (B = 1, r = 0)."""
    return arr[:, 1:] - arr[:, :-1]


def _fit_inputs(
    paths: Array | Sequence[Sequence[float]],
    payoff: Array | Sequence[float],
    order: int,
) -> tuple[Array, Array, Array, Array]:
    """Shared numpy front-end: (raw paths, payoff, gains, raw features)."""
    arr = _as_paths(paths)
    pay = _check_payoff(payoff, arr.shape[0])
    feats = signature_features(arr, order)
    return arr, pay, _gains(arr), feats


@dataclass(frozen=True)
class ExactKernelHedge:
    """Exact representer-theorem hedge with the base RBF kernel (psi = identity).

    The paper's "kernel hedging" benchmark (Dupret et al. 2026, §4.1):
    alpha* = (Q + N lam I)^{-1} H from the hedging Gram Q on standardized
    time-augmented signature features, and the optimal functional
    phi*(x) = sum_i alpha_i sum_k G^i_k K_RBF(x^i_k, x; gamma) evaluated via
    :meth:`hedge_positions`. Purely numpy; O(N^2 n^2) cost.
    """

    alpha: Array  # (N,) representer coefficients
    weights: Array  # (N*n,) alpha_i * G^i_k, flattened representer weights
    train_latent: Array  # (N*n, d) standardized training features
    gamma: float  # RBF bandwidth
    lam: float
    order: int
    n_steps: int
    train_positions: Array  # (N, n_steps) phi* on the training paths
    feat_mean: Array
    feat_std: Array

    def hedge_positions(self, paths: Array | Sequence[Sequence[float]]) -> Array:
        """Positions phi*(x_{t_k}) on fresh paths (eval mode)."""
        arr = _as_paths(paths)
        if arr.shape[1] - 1 != self.n_steps:
            raise ValueError(
                f"paths must have n_steps+1={self.n_steps + 1} columns to match the "
                f"fitted hedge; got {arr.shape[1]}"
            )
        feats = signature_features(arr, self.order)
        z, _, _ = standardize_features(feats, self.feat_mean, self.feat_std)
        flat = z.reshape(-1, z.shape[-1])
        k_cross = rbf_kernel_matrix(flat, self.train_latent, self.gamma)
        return np.asarray(k_cross @ self.weights).reshape(arr.shape[0], self.n_steps)


def exact_kernel_hedge(
    paths: Array | Sequence[Sequence[float]],
    payoff: Array | Sequence[float],
    *,
    gamma: float,
    lam: float,
    order: int = 3,
) -> ExactKernelHedge:
    """Fit the exact RKHS hedge by the representer-theorem linear solve.

    Fixed base RBF kernel on standardized time-augmented signature features
    (psi = identity, bandwidth ``gamma``); square loss, v0 = 0, B = 1. This
    is the closed-form reduction of Dupret et al. (2026, Thm 3.1): no
    gradient iteration — one ``(N, N)`` linear solve. SYNTHETIC-path
    correctness tool, never market evidence.
    """
    g = _check_positive(gamma, "gamma")
    lmb = _check_positive(lam, "lam")
    m = _check_order(order)
    arr, pay, gains, feats = _fit_inputs(paths, payoff, m)
    n_paths, n_steps = arr.shape[0], arr.shape[1] - 1
    z, mu, sd = standardize_features(feats)
    flat = z.reshape(-1, z.shape[-1])
    kernel_flat = rbf_kernel_matrix(flat, flat, g)
    Q = hedging_gram(kernel_flat, gains)
    alpha = solve_representer_alpha(Q, pay, lmb)
    weights = (alpha[:, None] * gains).reshape(-1)
    train_positions = (kernel_flat @ weights).reshape(n_paths, n_steps)
    return ExactKernelHedge(
        alpha=alpha,
        weights=weights,
        train_latent=flat,
        gamma=g,
        lam=lmb,
        order=m,
        n_steps=n_steps,
        train_positions=train_positions,
        feat_mean=mu,
        feat_std=sd,
    )


@dataclass(frozen=True)
class RffKernelHedge:
    """RFF-primal hedge with the base RBF kernel (psi = identity / gamma).

    beta* = (Z'Z + lam N I_D)^{-1} Z'H with aggregated features
    Z_i = sum_k y(x^i_k / gamma) G^i_k; strategy phi_D(x) = beta' y(x /
    gamma) (Dupret et al. 2026, Lemma 3.2 and eq. (3.17)). Purely numpy;
    O(N n D) memory versus O(N^2 n^2) for :class:`ExactKernelHedge`, with
    the documented O(D^{-1/2}) uniform approximation error of the kernel.
    """

    beta: Array  # (D,) primal RFF coefficients
    W: Array  # (D, d) fixed random frequencies
    b: Array  # (D,) fixed random phases
    gamma: float
    lam: float
    order: int
    n_steps: int
    n_rff: int
    train_positions: Array  # (N, n_steps) phi_D on the training paths
    feat_mean: Array
    feat_std: Array

    def hedge_positions(self, paths: Array | Sequence[Sequence[float]]) -> Array:
        """Positions phi_D(x_{t_k}) on fresh paths (eval mode)."""
        arr = _as_paths(paths)
        if arr.shape[1] - 1 != self.n_steps:
            raise ValueError(
                f"paths must have n_steps+1={self.n_steps + 1} columns to match the "
                f"fitted hedge; got {arr.shape[1]}"
            )
        feats = signature_features(arr, self.order)
        z, _, _ = standardize_features(feats, self.feat_mean, self.feat_std)
        flat = z.reshape(-1, z.shape[-1]) / self.gamma
        y = rff_features(flat, self.W, self.b)
        return np.asarray(y @ self.beta).reshape(arr.shape[0], self.n_steps)


def rff_kernel_hedge(
    paths: Array | Sequence[Sequence[float]],
    payoff: Array | Sequence[float],
    *,
    gamma: float,
    lam: float,
    n_rff: int = 100,
    seed: int = 0,
    order: int = 3,
) -> RffKernelHedge:
    """Fit the RFF-primal hedge (fixed base kernel, no learned embedding).

    Numpy-only mirror of the paper's scalable formulation with psi =
    identity/gamma — their "kernel hedging" benchmark at scale. ``n_rff``
    (D) controls the O(D^{-1/2}) kernel approximation error; the draw is
    seeded and fixed. SYNTHETIC-path correctness tool, never market evidence.
    """
    g = _check_positive(gamma, "gamma")
    lmb = _check_positive(lam, "lam")
    d_feat = _check_count(n_rff, "n_rff")
    m = _check_order(order)
    arr, pay, gains, feats = _fit_inputs(paths, payoff, m)
    n_paths, n_steps = arr.shape[0], arr.shape[1] - 1
    z, mu, sd = standardize_features(feats)
    flat = z.reshape(-1, z.shape[-1]) / g
    W, b = rff_draw(z.shape[-1], d_feat, seed)
    y = rff_features(flat, W, b).reshape(n_paths, n_steps, d_feat)
    Z = rff_aggregate(y, gains)
    beta = solve_rff_beta(Z, pay, lmb)
    train_positions = np.asarray(np.einsum("ikd,d->ik", y, beta))
    return RffKernelHedge(
        beta=beta,
        W=W,
        b=b,
        gamma=g,
        lam=lmb,
        order=m,
        n_steps=n_steps,
        n_rff=d_feat,
        train_positions=train_positions,
        feat_mean=mu,
        feat_std=sd,
    )


def _build_embedding(torch: Any, d_in: int, hidden: Sequence[int], p_dim: int) -> Any:
    """Latent feature network psi_omega: MLP with tanh hidden layers.

    Paper architecture: one hidden layer of 16 tanh units and a linear
    output of dimension p = 16 (Dupret et al. 2026, §4.1); the l2
    normalization and 1/gamma scaling are applied outside the network.
    """
    layers: list[Any] = []
    d = int(d_in)
    for h in hidden:
        layers += [torch.nn.Linear(d, int(h)), torch.nn.Tanh()]
        d = int(h)
    layers.append(torch.nn.Linear(d, int(p_dim)))
    return torch.nn.Sequential(*layers)


@dataclass(frozen=True)
class DeepKernelHedgeResult:
    """Learned deep kernel hedge and training diagnostics.

    ``beta`` are the RFF primal coefficients, ``gamma`` the learned RBF
    scale and ``net`` the latent embedding psi_omega (torch module; use
    :meth:`strategy_positions` for inference on fresh paths). ``objective``
    is the final training objective value (regularized empirical risk).
    """

    positions: Array  # (N, n_steps) learned positions on the training paths
    train_error: Array  # (N,) terminal hedging errors E_i on the training paths
    beta: Array  # (D,) RFF primal coefficients
    gamma: float  # learned RBF scale (exp of the trained log-parameter)
    zeta: float  # learned CVaR level (quadratic loss: the fixed init)
    objective: float  # final training objective (risk + lambda ||beta||^2)
    loss_kind: str
    loss_curve: Array  # per-epoch objective (value before each update)
    seed: int
    epochs: int
    order: int
    n_steps: int
    lam: float
    n_rff: int
    feat_mean: Array = field(repr=False)
    feat_std: Array = field(repr=False)
    net: Any = field(repr=False, compare=False)
    W: Any = field(repr=False, compare=False)  # (D, p) torch tensor, fixed
    b: Any = field(repr=False, compare=False)  # (D,) torch tensor, fixed

    def strategy_positions(self, paths: Array | Sequence[Sequence[float]]) -> Array:
        """Learned positions phi_D(x_{t_k}) on fresh paths (eval, no grad)."""
        arr = _as_paths(paths)
        if arr.shape[1] - 1 != self.n_steps:
            raise ValueError(
                f"paths must have n_steps+1={self.n_steps + 1} columns to match the "
                f"trained hedge; got {arr.shape[1]}"
            )
        torch = _torch()
        torch.set_num_threads(1)
        feats = signature_features(arr, self.order)
        z, _, _ = standardize_features(feats, self.feat_mean, self.feat_std)
        n_paths = arr.shape[0]
        flat = torch.as_tensor(z.reshape(-1, z.shape[-1]), dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            pos = _deep_positions(
                torch, flat, self.net, self.gamma, self.W, self.b, self.beta
            ).reshape(n_paths, self.n_steps)
        return np.asarray(pos.numpy(), dtype=float)


def _deep_latent(torch: Any, flat: Any, net: Any, log_gamma: Any) -> Any:
    """Deep kernel latent map psi_theta(x) = l2-normalize(psi_omega(x)) / gamma."""
    psi = net(flat)
    norm = psi.norm(dim=-1, keepdim=True).clamp_min(_NORM_EPS)
    return psi / norm / torch.exp(log_gamma)


def _deep_positions(
    torch: Any, flat: Any, net: Any, gamma: float, W: Any, b: Any, beta: Array
) -> Any:
    """Per-point positions beta' y_theta(x) for rows of ``flat`` (no grad)."""
    d_feat = int(W.shape[0])
    log_gamma = torch.tensor(math.log(gamma), dtype=torch.float32)
    latent = _deep_latent(torch, flat, net, log_gamma)
    y = math.sqrt(2.0 / d_feat) * torch.cos(latent @ W.T + b[None, :])
    beta_t = torch.as_tensor(np.asarray(beta, dtype=float), dtype=torch.float32)
    return y @ beta_t


def deep_kernel_hedge(
    paths: Array | Sequence[Sequence[float]],
    payoff: Array | Sequence[float],
    *,
    loss: str = "quadratic",
    alpha_cvar: float = 0.1,
    lam: float = 1e-6,
    gamma0: float = 1.0,
    hidden: Sequence[int] = (16,),
    p_dim: int = 16,
    n_rff: int = 100,
    order: int = 3,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 5e-3,
    seed: int = 0,
) -> DeepKernelHedgeResult:
    """Train a deep kernel hedge by joint full-batch AdamW (Algorithms 1 & 2).

    Deep kernel hedging (Dupret, Hainaut & Motte 2026, arXiv:2609.34474):
    the hedging functional is phi_D(x) = beta' y_theta(x) in the RFF image
    of the RKHS of the deep kernel K_theta = K_RBF(psi_theta(x),
    psi_theta(x')), with psi_theta = l2-normalize(psi_omega(x)) / gamma from
    a small tanh MLP. Jointly optimized over (omega, log gamma, beta[, zeta])
    on inputs ``X_{t_k}`` = time-augmented truncated signatures (order
    ``order``) of the standardized (time, log-price) channels:

    - ``loss='quadratic'``: (1/N) sum_i (H^i - sum_k phi(x^i_k) G^i_k)^2
      + lam ||beta||^2 with v0 = 0 fixed (their eq. (4.3); learning v0
      jointly changes the objective by a path-independent constant when the
      gains are martingale increments and leaves phi* unchanged).
    - ``loss='cvar'``: zeta + (1/(alpha_cvar N)) sum_i (E_i - zeta)_+
      + lam ||beta||^2 — the Rockafellar–Uryasev representation of the
      upper CVaR at level 1 - alpha_cvar of the terminal hedging error
      (their eq. (4.6)/(4.7); v0 fixed as required there).

    RFF frequencies/phases (W, b) ~ N(0, I_p) x U(0, 2 pi) are drawn once
    from the seeded torch RNG and never resampled. beta is warm-started at
    the closed-form RFF primal solve :func:`solve_rff_beta` for the initial
    embedding (deterministic given ``seed``), so training refines an
    already-valid hedge. Optimizer: AdamW with weight decay
    ``weight_decay`` (paper default 5e-3) and a cosine learning rate
    schedule ``lr`` -> ``lr`` / 10 over ``epochs`` (the paper's 1e-3 ->
    1e-4); weight decay is applied to the embedding weights and beta but
    not to log gamma / zeta (scale and level parameters, where decay would
    only bias the optimum). The paper uses mini-batch updates; full-batch
    steps are used here for determinism (CPU, single thread, seeded) — GPU
    determinism is not claimed. SYNTHETIC-path training only; evaluate on
    fresh paths for an honest train/eval split. No transaction costs
    (outside the paper's representer framework).
    """
    torch = _torch()
    if loss not in _LOSS_KINDS:
        raise ValueError(f"unknown loss kind {loss!r}; expected one of {_LOSS_KINDS}")
    a_cvar = float(alpha_cvar)
    if loss == "cvar" and (not math.isfinite(a_cvar) or not (0.0 < a_cvar < 1.0)):
        raise ValueError(f"alpha_cvar must be in (0, 1) for loss='cvar'; got {alpha_cvar!r}")
    lmb = _check_positive(lam, "lam")
    g0 = _check_positive(gamma0, "gamma0")
    p = _check_count(p_dim, "p_dim")
    d_feat = _check_count(n_rff, "n_rff")
    m = _check_order(order)
    hidden_widths = tuple(int(h) for h in hidden)
    if not hidden_widths or any(h < 1 for h in hidden_widths):
        raise ValueError(f"hidden must be a non-empty sequence of positive widths; got {hidden!r}")
    if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) < 1:
        raise ValueError(f"epochs must be an int >= 1; got {epochs!r}")
    n_epochs = int(epochs)
    rate = _check_positive(lr, "lr")
    wd = _check_finite(weight_decay, "weight_decay")
    if wd < 0.0:
        raise ValueError(f"weight_decay must be non-negative; got {weight_decay!r}")

    arr, pay, gains, feats = _fit_inputs(paths, payoff, m)
    n_paths, n_steps = arr.shape[0], arr.shape[1] - 1
    z, mu, sd = standardize_features(feats)
    flat_np = z.reshape(-1, z.shape[-1])

    torch.set_num_threads(1)
    flat = torch.as_tensor(flat_np, dtype=torch.float32)
    gains_t = torch.as_tensor(gains, dtype=torch.float32)
    pay_t = torch.as_tensor(pay, dtype=torch.float32)
    # Fork the global stream: the seeded embedding init + RFF draw must be
    # reproducible here while leaving the caller's torch RNG state untouched.
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        net = _build_embedding(torch, int(flat.shape[1]), hidden_widths, p)
        # RFF draw AFTER the network init so the seeded stream is fully determined.
        W = torch.randn(d_feat, p, dtype=torch.float32)
        b = torch.rand(d_feat, dtype=torch.float32) * (2.0 * math.pi)
    log_gamma = torch.tensor(math.log(g0), dtype=torch.float32, requires_grad=True)
    # zeta init: empirical (1 - alpha_cvar)-quantile of the unhedged errors.
    zeta_init = float(np.quantile(pay, 1.0 - a_cvar)) if loss == "cvar" else 0.0
    zeta = torch.tensor(zeta_init, dtype=torch.float32, requires_grad=loss == "cvar")
    scale = math.sqrt(2.0 / d_feat)
    # Warm start: beta_0 is the closed-form RFF primal solve (solve_rff_beta)
    # at the initial embedding, so joint training refines an already-valid
    # random-kitchen-sinks hedge instead of growing beta from zero against
    # the tiny aggregated-feature scale (deterministic given seed).
    with torch.no_grad():
        latent0 = _deep_latent(torch, flat, net, log_gamma)
        y0 = scale * torch.cos(latent0 @ W.T + b[None, :])
        Z0 = (y0.view(n_paths, n_steps, d_feat) * gains_t[:, :, None]).sum(dim=1)
    beta0 = solve_rff_beta(np.asarray(Z0.numpy(), dtype=float), pay, lmb)
    beta = torch.tensor(beta0, dtype=torch.float32, requires_grad=True)

    decay: list[Any] = [*net.parameters(), beta]
    no_decay: list[Any] = [log_gamma]
    if loss == "cvar":
        no_decay.append(zeta)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=rate,
    )
    curve: list[float] = []
    net.train()
    for epoch in range(n_epochs):
        # Cosine lr decay lr -> lr/10 over the epochs (paper: 1e-3 -> 1e-4).
        lr_e = rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * epoch / n_epochs)))
        for group in opt.param_groups:
            group["lr"] = lr_e
        opt.zero_grad(set_to_none=True)
        latent = _deep_latent(torch, flat, net, log_gamma)
        y = scale * torch.cos(latent @ W.T + b[None, :])
        Z = (y.view(n_paths, n_steps, d_feat) * gains_t[:, :, None]).sum(dim=1)
        err = pay_t - Z @ beta
        if loss == "cvar":
            tail = torch.clamp(err - zeta, min=0.0)
            objective = zeta + tail.mean() / a_cvar + lmb * (beta * beta).sum()
        else:
            objective = (err * err).mean() + lmb * (beta * beta).sum()
        curve.append(float(objective.detach().numpy()))
        objective.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        latent = _deep_latent(torch, flat, net, log_gamma)
        y = scale * torch.cos(latent @ W.T + b[None, :])
        Y = y.view(n_paths, n_steps, d_feat)
        positions_t = Y @ beta
        err_t = pay_t - (Y * gains_t[:, :, None]).sum(dim=1) @ beta
        if loss == "cvar":
            final_obj = zeta + torch.clamp(err_t - zeta, min=0.0).mean() / a_cvar
            final_obj = final_obj + lmb * (beta * beta).sum()
        else:
            final_obj = (err_t * err_t).mean() + lmb * (beta * beta).sum()
        positions = np.asarray(positions_t.numpy(), dtype=float)
        train_error = np.asarray(err_t.numpy(), dtype=float)
        beta_np = np.asarray(beta.detach().numpy(), dtype=float)
        gamma_hat = float(math.exp(float(log_gamma.detach().numpy())))
        zeta_hat = float(zeta.detach().numpy())
    if not bool(np.isfinite(positions).all()):
        raise ArithmeticError("deep kernel hedging produced non-finite positions")
    return DeepKernelHedgeResult(
        positions=positions,
        train_error=train_error,
        beta=beta_np,
        gamma=gamma_hat,
        zeta=zeta_hat,
        objective=float(final_obj.numpy()),
        loss_kind=loss,
        loss_curve=np.asarray(curve, dtype=float),
        seed=int(seed),
        epochs=n_epochs,
        order=m,
        n_steps=n_steps,
        lam=lmb,
        n_rff=d_feat,
        feat_mean=mu,
        feat_std=sd,
        net=net,
        W=W,
        b=b,
    )
