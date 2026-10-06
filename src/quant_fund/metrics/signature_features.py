"""Signature-feature metrics: Goursat-PDE signature kernel and path MMD.

This lane ADDS the untruncated signature kernel and a path-valued two-sample
layer on top of the wave-11 rough-path primitives — it does NOT reimplement
them. Composition notes:

- ``quant_fund.models.path_signatures.signature`` / ``logsignature`` /
  ``lead_lag_transform`` are reused by import: the incremental Chen-product
  solver, the Lyndon/Hall logsignature basis, and the lead-lag augmentation
  are already landed and unit-tested there. The wave-11
  ``signature_kernel`` is the *order-m truncated inner product*
  ``k_m(a, b; σ) = Σ_{k≤m} σ^{2k} ⟨Sig^k(a), Sig^k(b)⟩``; this module keeps
  it available (``kernel="truncated"``) as the fast Gram option and as the
  convergence reference, and implements the *untruncated* kernel that the
  truncated form approximates.
- ``quant_fund.models.leadlag`` is intentionally NOT reused: it answers the
  cross-sectional "which asset leads which" question (peak-lag
  cross-correlation networks, Hayashi–Yoshida) — a different object from the
  lead-lag path augmentation used here.
- ``quant_fund.metrics.twosample`` is intentionally NOT reused: its KS /
  energy / CvM tests act on scalar samples; the MMD here acts on path-valued
  samples through the signature kernel Gram.

References (verified against arXiv abstract pages on 2026-09-30)
---------------------------------------------------------------
- Chevyrev, I. & Oberhauser, H. (2018/2022), "Signature moments to
  characterize laws of stochastic processes", Ann. Appl. Probab. 32(1),
  213–258, arXiv:1810.10971 — the signature kernel induces an MMD-type
  metric on laws of stochastic processes; under bounded-variation inputs it
  metrizes weak convergence, which is what makes the path two-sample test
  here consistent.
- Salvi, C., Cass, T., Foster, J., Lyons, T. & Yang, W. (2021), "The
  signature kernel is the solution of a Goursat PDE", SIAM J. Math. Data
  Sci. 3(3), 873–899, arXiv:2006.14794 — the untruncated kernel
  K(s, t) = ⟨Sig(x_{[0,s]}), Sig(y_{[0,t]})⟩ solves the hyperbolic Goursat
  problem ∂²K/∂s∂t = ⟨Ẋ_s, Ẏ_t⟩ K with boundary K(0,·) = K(·,0) = 1, so it
  can be computed on a grid directly from path increments without ever
  forming signatures. (The wave brief suggested arXiv:2006.14742 for this
  citation; that id is an unrelated number-theory paper — corrected here.)
- Salvi, C., Lemercier, M., Liu, C., Horvath, B., Damoulas, T. & Lyons, T.
  (2021), "Higher order kernel mean embeddings to capture filtrations of
  stochastic processes", NeurIPS 34, arXiv:2109.03582 — motivation for
  carrying the time augmentation inside the path (a plain signature kernel
  on the raw channels is blind to the information carried by the sampling
  clock).
- Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B. & Smola, A. (2008/
  2012), "A kernel method for the two-sample problem", NeurIPS 19 / JMLR 13,
  723–773, arXiv:0805.2368 — the biased V-statistic / unbiased U-statistic
  MMD² estimators and the permutation calibration used here.
- Gyurkó, L.G., Lyons, T., Kontkowski, M. & Field, J. (2013), "Extracting
  information from the signature of a financial data stream",
  arXiv:1307.7244 — signatures as features of financial paths (the
  microstructure/return-path framing this lane follows).
- Chevyrev, I. & Kormilitzin, A. (2016), "A primer on the signature method
  in machine learning", arXiv:1603.03788 — discretization conventions and
  the lead-lag augmentation (implemented upstream in
  ``models.path_signatures`` and reused).

Machinery
---------
1. Augmentation utilities — ``time_augmentation`` appends a normalized
   clock channel (cheap filtration sensitivity, cf. arXiv:2109.03582 §4);
   ``augment_path`` composes it with the reused ``lead_lag_transform``
   (order documented: augmentations apply left to right).
2. ``signature_feature_vector`` / ``signature_feature_matrix`` — batched
   truncated signature or Lyndon-basis logsignature features for ensembles
   of piecewise-linear paths (2–3 channels, orders 3–5 supported for the
   signature basis, ≤4 for logsignature — the upstream Lyndon-enumeration
   bound; larger orders fail closed). The incremental Chen-product solver
   of the composed module is the efficient per-path computation; this layer
   only batches and validates.
3. ``dyadic_refine`` / ``signature_kernel_pde`` — the Goursat-PDE kernel of
   arXiv:2006.14794 on a dyadically refined grid. Inserting collinear grid
   points leaves the piecewise-linear path (hence its signature, hence the
   true kernel) unchanged — refinement raises only the discretization's
   accuracy, which is why consecutive levels must converge. Two schemes:
   ``pde1`` (first-order trapezoidal cell rule) and ``pde2`` (the
   second-order cell rule of the paper's reference implementation,
   default). For straight-line paths the kernel has the closed form
   ``I_0(2·sqrt(⟨V_a, V_b⟩)) = Σ_k ⟨V_a, V_b⟩^k / (k!)²`` (each level of a
   linear path's signature is V^{⊗k}/k!, so level k contributes
   ⟨V_a,V_b⟩^k/(k!)² — NOT the exp(⟨V_a,V_b⟩) of factorial-weighted
   variants); tests assert the PDE converges to it and to the truncated
   inner-product kernel at high order.
4. ``signature_gram`` / ``signature_mmd`` / ``signature_mmd_test`` — Gram
   assembly and the Gretton et al. two-sample machinery specialized to
   path-valued samples. The permutation test reuses ONE pooled Gram for
   all label permutations (the kernel matrix is label-free; permutations
   only re-select submatrices), so the test cost is one Gram build plus
   O(n²) per permutation — this is the efficient-computation story for the
   MMD layer.
5. ``bench_signature_features(seed)`` — seeded SYNTHETIC bench (AGENTS.md
   honesty contract #2): GBM vs mean-reverting OU path ensembles and
   vol-regime shifts, emitting MMD power/false-positive rates, a
   logsignature-feature drift-regression R², PDE-vs-truncated consistency
   and Gram sanity metrics. Correctness evidence only — every key is
   prefixed ``synthetic_``.

Honesty: outputs are path-geometry diagnostics (kernel values, MMD², p
-values, feature regressions) — proper two-sample/calibration evidence only.
No Sharpe/Sortino/P&L/NAV content is produced and ``bench_*`` reports are
SYNTHETIC correctness numbers, never market evidence. Fail-closed
throughout: degenerate, short, mismatched-channel, non-finite or invalid
inputs raise (``ValueError``; ``ArithmeticError`` when a PDE cell rule goes
non-finite) rather than silently returning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "AUGMENTATIONS",
    "MMD_KERNELS",
    "PDE_SCHEMES",
    "augment_path",
    "bench_signature_features",
    "dyadic_refine",
    "signature_feature_matrix",
    "signature_feature_vector",
    "signature_gram",
    "signature_kernel_pde",
    "signature_kernel_pde_diagnostics",
    "signature_mmd",
    "signature_mmd_test",
    "time_augmentation",
]

PDE_SCHEMES: tuple[str, str] = ("pde1", "pde2")
"""Goursat cell rules: ``pde1`` first-order trapezoidal, ``pde2`` second-order."""

AUGMENTATIONS: tuple[str, str] = ("time", "leadlag")
"""Supported channel augmentations for ``augment_path`` / feature matrices."""

MMD_KERNELS: tuple[str, str] = ("pde", "truncated")
"""Gram kernels: ``pde`` (untruncated Goursat kernel) or ``truncated`` (order-m
inner product reused from ``models.path_signatures``)."""

_BASES: tuple[str, str] = ("signature", "logsignature")
_MAX_SIGNATURE_ORDER = 6  # upstream bound in models.path_signatures
_MAX_LOGSIGNATURE_ORDER = 4  # upstream Lyndon-enumeration bound
_MAX_DYADIC_LEVEL = 8  # 2**8 = 256× grid blow-up cap
_MAX_REFINED_SEGMENTS = 4096  # absolute grid cap per path
_MIN_PATHS_MMD = 2  # unbiased estimator needs n, m >= 2
_EPS_DENOM = 1e-12


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_path(path: Array | Sequence[Sequence[float]], name: str = "path") -> Array:
    """Validate and normalize to a finite float64 (T+1, d) array, T >= 1."""
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (T+1, d); got ndim={arr.ndim}")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must have at least 2 points (T+1 >= 2); got {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must have at least one channel (d >= 1)")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} entries must be finite (NaN/inf rejected)")
    return arr


def _as_ensemble(
    paths: Sequence[Array] | Array,
    name: str,
    *,
    min_paths: int = 1,
) -> list[Array]:
    """Normalize an ensemble of paths to a list of validated 2-D arrays.

    Accepts a sequence of (T_i+1, d) arrays or a stacked (n, T+1, d) array.
    Every path must carry the same channel count d (kernel/feature stacking
    is meaningless otherwise); lengths may differ. Fail-closed on empty
    ensembles, non-2-D members, NaN/inf, or channel mismatch.
    """
    if isinstance(paths, np.ndarray):
        if paths.ndim != 3 or paths.shape[0] == 0:
            raise ValueError(f"{name} as an array must be a non-empty (n, T+1, d) stack")
        members = [np.asarray(paths[i], dtype=float) for i in range(paths.shape[0])]
    else:
        try:
            members = [np.asarray(p, dtype=float) for p in paths]
        except TypeError:
            raise ValueError(f"{name} must be a sequence of paths or a 3-D array") from None
    if len(members) < min_paths:
        raise ValueError(f"{name} must hold at least {min_paths} path(s); got {len(members)}")
    dims = int(members[0].shape[1]) if members[0].ndim == 2 else -1
    for idx, member in enumerate(members):
        if member.ndim != 2:
            raise ValueError(f"{name}[{idx}] must be a 2-D (T+1, d) path; got ndim={member.ndim}")
        if member.shape[1] != dims:
            raise ValueError(
                f"{name} paths must share one channel count; [{idx}] has {member.shape[1]} vs {dims}"
            )
        _as_path(member, f"{name}[{idx}]")
    return members


def _check_int(value: int, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(value)
    if v < lo or v > hi:
        raise ValueError(f"{name} must lie in [{lo}, {hi}]; got {v}")
    return v


def _check_sigma(sigma: float) -> float:
    s = float(sigma)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError(f"sigma must be positive and finite; got {sigma!r}")
    return s


def _check_scheme(scheme: str) -> str:
    if scheme not in PDE_SCHEMES:
        raise ValueError(f"scheme must be one of {PDE_SCHEMES}; got {scheme!r}")
    return scheme


# ---------------------------------------------------------------------------
# 1. Channel augmentations (time clock + reused lead-lag)
# ---------------------------------------------------------------------------


def time_augmentation(path: Array | Sequence[Sequence[float]]) -> Array:
    """Append a normalized clock channel: (T+1, d) -> (T+1, d+1).

    The extra channel is ``linspace(0, 1, T+1)`` — the path's own index
    parametrization, so irregular spacing in an upstream resample is NOT
    re-encoded here (feed clock-aware resamples if that matters). A time
    channel makes the signature sensitive to the sampling clock / carry of
    the filtration (arXiv:2109.03582) and lifts level-1 features to include
    elapsed time.
    """
    arr = _as_path(path)
    clock = np.linspace(0.0, 1.0, arr.shape[0], dtype=float)
    return np.concatenate([arr, clock[:, None]], axis=1)


def augment_path(
    path: Array | Sequence[Sequence[float]],
    *,
    augmentations: Sequence[str] = ("time",),
) -> Array:
    """Apply channel augmentations left to right, in the order given.

    ``"time"`` appends the normalized clock channel; ``"leadlag"`` applies
    the reused ``lead_lag_transform`` (interleaved lead/lag copy, doubling
    the channel count). Applied in sequence they compose — e.g.
    ``("time", "leadlag")`` clock-augmented then lead-lag interleaved, a
    (T+1, d) path becoming (2T+1, 2(d+1)). An empty tuple returns the
    validated path unchanged; an unknown name fails closed.
    """
    # Lazy: path_signatures lives in the analytics layer above metrics —
    # the sanctioned way to break that upward edge.
    from quant_fund.models.path_signatures import lead_lag_transform  # noqa: PLC0415

    arr = _as_path(path)
    for aug in augmentations:
        if aug == "time":
            arr = time_augmentation(arr)
        elif aug == "leadlag":
            arr = lead_lag_transform(arr)
        else:
            raise ValueError(f"augmentation must be one of {AUGMENTATIONS}; got {aug!r}")
    return arr


# ---------------------------------------------------------------------------
# 2. Batched signature / logsignature features
# ---------------------------------------------------------------------------


def _resolve_basis_order(basis: str, order: int) -> tuple[str, int]:
    if basis not in _BASES:
        raise ValueError(f"basis must be one of {_BASES}; got {basis!r}")
    cap = _MAX_SIGNATURE_ORDER if basis == "signature" else _MAX_LOGSIGNATURE_ORDER
    m = _check_int(order, "order", 1, cap)
    return basis, m


def signature_feature_vector(
    path: Array | Sequence[Sequence[float]],
    *,
    order: int = 3,
    basis: str = "signature",
    augmentations: Sequence[str] = ("time",),
) -> Array:
    """Truncated signature or Lyndon-basis logsignature of one path.

    The path is first channel-augmented (default: time channel), then the
    reused upstream solver runs — incremental Chen updates for
    ``basis="signature"`` (orders 1..6; flat word-order layout, d + d² + …
    terms) or series-log + Lyndon extraction for ``basis="logsignature"``
    (orders 1..4; length Σ_k l_k(d) with Witt numbers l_k). The signature
    is invariant to translation and reparametrization — features describe
    the *order of events*, not the sampling clock (unless the time channel
    is augmented in).
    """
    # Lazy: path_signatures lives in the analytics layer above metrics —
    # the sanctioned way to break that upward edge.
    from quant_fund.models.path_signatures import (  # noqa: PLC0415
        logsignature,
        signature,
    )

    basis_resolved, m = _resolve_basis_order(basis, order)
    arr = augment_path(path, augmentations=augmentations)
    if basis_resolved == "signature":
        return signature(arr, m)
    return logsignature(arr, m)


def signature_feature_matrix(
    paths: Sequence[Array] | Array,
    *,
    order: int = 3,
    basis: str = "signature",
    augmentations: Sequence[str] = ("time",),
) -> Array:
    """Stacked feature matrix (n_paths, n_features) over an ensemble.

    All members share one channel count (ensemble guard), so every feature
    column is comparable. Fail-closed per member and on empty ensembles.
    """
    members = _as_ensemble(paths, "paths")
    feats = [
        signature_feature_vector(p, order=order, basis=basis, augmentations=augmentations)
        for p in members
    ]
    return np.stack(feats, axis=0)


# ---------------------------------------------------------------------------
# 3. Signature kernel via the Goursat PDE (dyadic refinement)
# ---------------------------------------------------------------------------


def dyadic_refine(path: Array | Sequence[Sequence[float]], level: int) -> Array:
    """Refine a piecewise-linear path by inserting ``2**level - 1`` collinear
    points on every segment.

    The refined path is the SAME piecewise-linear path: splitting a linear
    segment v = v_1 + v_2 with v_1 ∥ v_2 gives exp(v_1) ⊗ exp(v_2) = exp(v)
    in the tensor algebra, so the signature (and the true signature kernel)
    is exactly preserved — only the PDE discretization's accuracy improves.
    This invariance is what makes dyadic refinement a valid convergence
    sequence. Fail-closed on ``level`` outside [0, 8] or a refined segment
    count above 4096.
    """
    arr = _as_path(path)
    lv = _check_int(level, "dyadic_level", 0, _MAX_DYADIC_LEVEL)
    if lv == 0:
        return arr.copy()
    n_segments = arr.shape[0] - 1
    refined_segments = n_segments * (2**lv)
    if refined_segments > _MAX_REFINED_SEGMENTS:
        raise ValueError(
            f"refined path would have {refined_segments} segments (> {_MAX_REFINED_SEGMENTS})"
        )
    d = arr.shape[1]
    steps = 2**lv
    frac = np.arange(steps, dtype=float).reshape(-1, 1) / float(steps)
    starts = arr[:-1]  # (T, d)
    deltas = arr[1:] - arr[:-1]  # (T, d)
    interp = starts[:, None, :] + frac[None, :, :] * deltas[:, None, :]  # (T, steps, d)
    refined = np.concatenate([interp.reshape(-1, d), arr[-1:, :]], axis=0)
    return np.asarray(refined, dtype=float)


def _goursat_solve(a_increments: Array, scheme: str) -> float:
    """Solve the Goursat PDE on the grid defined by two increment sequences.

    ``a_increments[i, j] = ⟨dx_i, dy_j⟩`` is the increment inner-product
    matrix of the (already refined) paths — the PDE source term, constant
    per cell because both paths are piecewise linear. Boundary K(0,·) =
    K(·,0) = 1. Cells are filled along anti-diagonals s = i + j: cell
    (i, j) reads K[i+1, j], K[i, j+1], K[i, j] which were all completed at
    earlier diagonals, so the whole diagonal updates vectorized.

    pde1 (first order, trapezoidal cell mean):
        K[i+1,j+1] = (K[i+1,j] + K[i,j+1] − K[i,j]·(1 − a/2)) / (1 − a/2)
    pde2 (second order, paper's reference scheme — default):
        K[i+1,j+1] = (K[i+1,j] + K[i,j+1])·(1 + a/2 + a²/12)
                     − K[i,j]·(1 − a²/12)

    A non-finite cell value (e.g. the pde1 denominator crossing zero on
    large increments) raises ArithmeticError — fail-closed.
    """
    m_a, m_b = a_increments.shape
    kernel = np.ones((m_a + 1, m_b + 1), dtype=float)
    for s in range(m_a + m_b - 1):
        i_lo = max(0, s - m_b + 1)
        i_hi = min(m_a - 1, s)
        ii = np.arange(i_lo, i_hi + 1)
        jj = s - ii
        a = a_increments[ii, jj]
        below = kernel[ii + 1, jj]
        left = kernel[ii, jj + 1]
        corner = kernel[ii, jj]
        if scheme == "pde2":
            upd = (below + left) * (1.0 + 0.5 * a + a * a / 12.0) - corner * (1.0 - a * a / 12.0)
        else:
            denom = 1.0 - 0.5 * a
            if float(np.min(np.abs(denom))) < _EPS_DENOM:
                raise ArithmeticError(
                    "pde1 cell rule hit a vanishing denominator (1 − a/2 ≈ 0); "
                    "refine dyadically or use scheme='pde2'"
                )
            upd = (below + left - corner * (1.0 - 0.5 * a)) / denom
        if not bool(np.all(np.isfinite(upd))):
            raise ArithmeticError(
                f"{scheme} cell update produced non-finite values on diagonal {s}"
            )
        kernel[ii + 1, jj + 1] = upd
    value = float(kernel[m_a, m_b])
    if not math.isfinite(value):
        raise ArithmeticError("Goursat solve returned a non-finite kernel value")
    return value


def signature_kernel_pde(
    path_a: Array | Sequence[Sequence[float]],
    path_b: Array | Sequence[Sequence[float]],
    *,
    dyadic_level: int = 1,
    sigma: float = 1.0,
    scheme: str = "pde2",
) -> float:
    """Untruncated signature kernel K(a, b) via the Goursat PDE.

    K(a, b) = ⟨Sig(a), Sig(b)⟩ over ALL signature levels — the object the
    upstream truncated ``signature_kernel`` approximates at order m. Solved
    on the grid of path increments refined ``dyadic_level`` times per side
    (refinement leaves the true kernel invariant and only raises accuracy;
    ``signature_kernel_pde_diagnostics`` exposes the convergence sequence).
    ``sigma`` rescales increments: K_σ(a, b) = K_1(σa, σb), the same
    convention as the truncated kernel. Positive definite as the inner
    product of a feature map (a ↦ Sig(σa)). Fail-closed on channel
    mismatch, invalid level/scheme/sigma, and non-finite cell updates.
    """
    arr_a = _as_path(path_a, "path_a")
    arr_b = _as_path(path_b, "path_b")
    if int(arr_a.shape[1]) != int(arr_b.shape[1]):
        raise ValueError(
            f"paths must share the same channel count d; got {arr_a.shape[1]} and {arr_b.shape[1]}"
        )
    s = _check_sigma(sigma)
    scheme_resolved = _check_scheme(scheme)
    lv = _check_int(dyadic_level, "dyadic_level", 0, _MAX_DYADIC_LEVEL)
    ref_a = dyadic_refine(arr_a, lv)
    ref_b = dyadic_refine(arr_b, lv)
    dx = np.diff(ref_a, axis=0)
    dy = np.diff(ref_b, axis=0)
    a_mat = (dx @ dy.T) * (s * s)
    return _goursat_solve(a_mat, scheme_resolved)


def signature_kernel_pde_diagnostics(
    path_a: Array | Sequence[Sequence[float]],
    path_b: Array | Sequence[Sequence[float]],
    *,
    max_dyadic_level: int = 3,
    sigma: float = 1.0,
    scheme: str = "pde2",
) -> dict[str, Any]:
    """Dyadic convergence sequence of the PDE kernel, levels 0..max.

    Returns the per-level kernel estimates, consecutive absolute gaps, and
    the order-``max(order_probe, ...)`` truncated inner-product value as an
    independent cross-check (the truncation approximates the same limit
    from below — its remainder is the factorial-decayed tail). The gaps
    must shrink geometrically; a non-shrinking gap is surfaced honestly in
    ``monotone_shrinking_gaps`` rather than hidden.
    """
    # Lazy: path_signatures lives in the analytics layer above metrics —
    # the sanctioned way to break that upward edge.
    from quant_fund.models.path_signatures import signature_kernel  # noqa: PLC0415

    _as_path(path_a, "path_a")
    _as_path(path_b, "path_b")
    cap = _check_int(max_dyadic_level, "max_dyadic_level", 0, _MAX_DYADIC_LEVEL)
    estimates = [
        signature_kernel_pde(path_a, path_b, dyadic_level=lv, sigma=sigma, scheme=scheme)
        for lv in range(cap + 1)
    ]
    gaps = [abs(estimates[lv] - estimates[lv - 1]) for lv in range(1, cap + 1)]
    # Truncated inner product at the maximum supported order as a reference
    # for the same untruncated limit (level-0 term 1 included upstream).
    reference = signature_kernel(path_a, path_b, order=_MAX_SIGNATURE_ORDER, sigma=sigma)
    return {
        "estimates": estimates,
        "gaps": gaps,
        "final": estimates[-1],
        "truncated_reference_order6": reference,
        "abs_gap_to_reference": abs(estimates[-1] - reference),
        "monotone_shrinking_gaps": all(gaps[i] <= gaps[i - 1] for i in range(1, len(gaps))),
    }


# ---------------------------------------------------------------------------
# 4. Gram assembly + path MMD
# ---------------------------------------------------------------------------


def _kernel_eval(
    arr_a: Array,
    arr_b: Array,
    kernel: str,
    dyadic_level: int,
    order: int,
    sigma: float,
    scheme: str,
) -> float:
    # Lazy: path_signatures lives in the analytics layer above metrics —
    # the sanctioned way to break that upward edge.
    from quant_fund.models.path_signatures import signature_kernel  # noqa: PLC0415

    if kernel == "pde":
        return signature_kernel_pde(
            arr_a, arr_b, dyadic_level=dyadic_level, sigma=sigma, scheme=scheme
        )
    return signature_kernel(arr_a, arr_b, order=order, sigma=sigma)


def _level_column_weights(dims: int, order: int, sigma: float) -> Array:
    """Per-feature-column weight sigma^{2k} for level-k signature columns.

    The flat signature layout holds level k in d^k consecutive columns, so
    the weighted inner product
    ``k_m(a, b; σ) = 1 + ⟨Sig^{≤m}(a), Sig^{≤m}(b)⟩_w``
    is a single matrix product — the batch-efficient form of the upstream
    pairwise ``signature_kernel`` (identical sums, formed once per path).
    """
    w = np.concatenate(
        [np.full(dims**k, sigma ** (2 * k), dtype=float) for k in range(1, order + 1)]
    )
    return w


def _gram_truncated(
    members_a: list[Array],
    members_b: list[Array] | None,
    order: int,
    sigma: float,
) -> Array:
    """Batched order-m truncated Gram: signatures once per path, then matmul."""
    # Lazy: path_signatures lives in the analytics layer above metrics —
    # the sanctioned way to break that upward edge.
    from quant_fund.models.path_signatures import signature  # noqa: PLC0415

    feats_a = np.stack([signature(p, order) for p in members_a], axis=0)
    w = _level_column_weights(int(members_a[0].shape[1]), order, sigma)
    if members_b is None:
        gram = 1.0 + (feats_a * w) @ feats_a.T
        return np.asarray(gram, dtype=float)
    feats_b = np.stack([signature(p, order) for p in members_b], axis=0)
    gram = 1.0 + (feats_a * w) @ feats_b.T
    return np.asarray(gram, dtype=float)


def _prepare_kernel_args(
    kernel: str, dyadic_level: int, order: int, sigma: float, scheme: str
) -> tuple[int, int, float, str]:
    if kernel not in MMD_KERNELS:
        raise ValueError(f"kernel must be one of {MMD_KERNELS}; got {kernel!r}")
    lv = _check_int(dyadic_level, "dyadic_level", 0, _MAX_DYADIC_LEVEL)
    m = _check_int(order, "order", 1, _MAX_SIGNATURE_ORDER)
    s = _check_sigma(sigma)
    sch = _check_scheme(scheme)
    return lv, m, s, sch


def signature_gram(
    paths_a: Sequence[Array] | Array,
    paths_b: Sequence[Array] | Array | None = None,
    *,
    kernel: str = "pde",
    dyadic_level: int = 1,
    order: int = 4,
    sigma: float = 1.0,
    scheme: str = "pde2",
) -> Array:
    """Gram/kernel matrix over path ensembles.

    ``signature_gram(paths)`` returns the symmetric (n, n) Gram computed on
    the upper triangle only and mirrored — exact symmetry by construction.
    ``signature_gram(a, b)`` returns the cross matrix (n_a, n_b) with every
    entry evaluated. ``kernel="pde"`` uses the untruncated Goursat kernel at
    ``dyadic_level`` (one PDE solve per pair); ``kernel="truncated"``
    reuses the upstream order-m signature — formed ONCE per path, with the
    σ-weighted inner product assembled by a single matrix product (the
    same values ``models.path_signatures.signature_kernel`` computes
    pairwise, minus the redundant recomputation).
    """
    lv, m, s, sch = _prepare_kernel_args(kernel, dyadic_level, order, sigma, scheme)
    members_a = _as_ensemble(paths_a, "paths_a")
    members_b = _as_ensemble(paths_b, "paths_b") if paths_b is not None else None
    if members_b is not None and int(members_a[0].shape[1]) != int(members_b[0].shape[1]):
        raise ValueError(
            "paths_a and paths_b must share one channel count; "
            f"got {members_a[0].shape[1]} vs {members_b[0].shape[1]}"
        )
    if kernel == "truncated":
        # Batched path: one signature per path, then a weighted matmul —
        # the same sums the upstream pairwise kernel would produce, minus
        # the redundant per-pair signature recomputation.
        return _gram_truncated(members_a, members_b, m, s)
    if members_b is None:
        n = len(members_a)
        gram = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                gram[i, j] = _kernel_eval(members_a[i], members_a[j], kernel, lv, m, s, sch)
                gram[j, i] = gram[i, j]
        return gram
    gram = np.empty((len(members_a), len(members_b)), dtype=float)
    for i, pa in enumerate(members_a):
        for j, pb in enumerate(members_b):
            gram[i, j] = _kernel_eval(pa, pb, kernel, lv, m, s, sch)
    return gram


def _mmd2_from_gram(gram: Array, n_a: int) -> dict[str, float]:
    """Biased V-statistic and unbiased U-statistic MMD² from a pooled Gram.

    The pooled Gram of the concatenated sample [a; b] splits as
    K_xx (n_a×n_a), K_yy (n_b×n_b), K_xy (n_a×n_b):
        MMD²_b = mean(K_xx) + mean(K_yy) − 2·mean(K_xy)
        MMD²_u = off-diag mean(K_xx) + off-diag mean(K_yy) − 2·mean(K_xy)
    (Gretton et al. 2008/2012, §2–3). The biased statistic can go slightly
    negative on identical samples — reported as-is (it is a statistic, not
    a distance estimate to clip).
    """
    n_b = int(gram.shape[0]) - n_a
    k_xx = gram[:n_a, :n_a]
    k_yy = gram[n_a:, n_a:]
    k_xy = gram[:n_a, n_a:]
    biased = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    off_xx = float((k_xx.sum() - np.trace(k_xx)) / (n_a * (n_a - 1)))
    off_yy = float((k_yy.sum() - np.trace(k_yy)) / (n_b * (n_b - 1)))
    unbiased = float(off_xx + off_yy - 2.0 * k_xy.mean())
    return {"mmd2_biased": biased, "mmd2_unbiased": unbiased}


def signature_mmd(
    paths_a: Sequence[Array] | Array,
    paths_b: Sequence[Array] | Array,
    *,
    kernel: str = "pde",
    dyadic_level: int = 1,
    order: int = 4,
    sigma: float = 1.0,
    scheme: str = "pde2",
) -> dict[str, float]:
    """Kernel MMD² between two path ensembles (Chevyrev & Oberhauser 2018).

    With the PDE kernel this is the untruncated signature-kernel MMD — the
    metric that metrizes weak convergence of path laws under bounded
    variation. With ``kernel="truncated"`` it is the order-m feature-space
    approximation. Returns both estimators plus ensemble sizes and the
    pooled Gram's minimum eigenvalue (a PSD sanity readout — the truncated
    kernel is exactly PSD; the PDE kernel is PSD up to discretization
    error, so ``gram_min_eig`` should sit at a small negative or positive
    value, never deep-negative).
    """
    members_a = _as_ensemble(paths_a, "paths_a", min_paths=_MIN_PATHS_MMD)
    members_b = _as_ensemble(paths_b, "paths_b", min_paths=_MIN_PATHS_MMD)
    if int(members_a[0].shape[1]) != int(members_b[0].shape[1]):
        raise ValueError(
            "path ensembles must share one channel count; "
            f"got {members_a[0].shape[1]} vs {members_b[0].shape[1]}"
        )
    pooled = members_a + members_b
    gram = signature_gram(
        pooled,
        kernel=kernel,
        dyadic_level=dyadic_level,
        order=order,
        sigma=sigma,
        scheme=scheme,
    )
    stats = _mmd2_from_gram(gram, len(members_a))
    min_eig = float(np.linalg.eigvalsh(gram).min())
    return {
        **stats,
        "mmd_biased": math.sqrt(max(stats["mmd2_biased"], 0.0)),
        "mmd_unbiased": math.sqrt(max(stats["mmd2_unbiased"], 0.0)),
        "n_a": float(len(members_a)),
        "n_b": float(len(members_b)),
        "gram_min_eig": min_eig,
    }


def signature_mmd_test(
    paths_a: Sequence[Array] | Array,
    paths_b: Sequence[Array] | Array,
    *,
    seed: int | np.random.Generator,
    n_permutations: int = 99,
    kernel: str = "pde",
    dyadic_level: int = 1,
    order: int = 4,
    sigma: float = 1.0,
    scheme: str = "pde2",
) -> dict[str, Any]:
    """Permutation-calibrated two-sample test on the signature-kernel MMD.

    One pooled Gram is built once — under a label permutation the kernel
    values are unchanged, so each permutation only re-selects submatrices
    (the efficient-computation contract for this layer). The test statistic
    is the biased MMD²; ``p_value = (1 + #{MMD_perm ≥ MMD_obs}) /
    (1 + n_permutations)`` is the standard randomized-test p-value (never
    exactly 0). Deterministic: pass an int seed (default_rng) or a
    Generator. Returns the statistic, p-value, the α=0.05 rejection flag
    and the permutation quantile threshold.
    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        rng = np.random.default_rng(int(seed))
    else:
        raise ValueError("seed must be an int or a numpy Generator")
    n_perm = _check_int(n_permutations, "n_permutations", 1, 100000)
    members_a = _as_ensemble(paths_a, "paths_a", min_paths=_MIN_PATHS_MMD)
    members_b = _as_ensemble(paths_b, "paths_b", min_paths=_MIN_PATHS_MMD)
    if int(members_a[0].shape[1]) != int(members_b[0].shape[1]):
        raise ValueError(
            "path ensembles must share one channel count; "
            f"got {members_a[0].shape[1]} vs {members_b[0].shape[1]}"
        )
    n_a, n_b = len(members_a), len(members_b)
    pooled = members_a + members_b
    gram = signature_gram(
        pooled,
        kernel=kernel,
        dyadic_level=dyadic_level,
        order=order,
        sigma=sigma,
        scheme=scheme,
    )
    observed = _mmd2_from_gram(gram, n_a)["mmd2_biased"]

    perm_stats = np.empty(n_perm, dtype=float)
    total = n_a + n_b
    for rep in range(n_perm):
        idx = rng.permutation(total)
        a_idx = idx[:n_a]
        k_xx = gram[np.ix_(a_idx, a_idx)]
        b_idx = idx[n_a:]
        k_yy = gram[np.ix_(b_idx, b_idx)]
        k_xy = gram[np.ix_(a_idx, b_idx)]
        perm_stats[rep] = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    exceedances = int(np.count_nonzero(perm_stats >= observed))
    p_value = float((1 + exceedances) / (1 + n_perm))
    threshold = float(np.quantile(perm_stats, 0.95))
    return {
        "mmd2_biased": float(observed),
        "p_value": p_value,
        "reject_5pct": bool(p_value <= 0.05),
        "n_permutations": int(n_perm),
        "perm_threshold_95": threshold,
        "perm_stat_max": float(perm_stats.max()),
        "perm_stat_mean": float(perm_stats.mean()),
        "n_a": int(n_a),
        "n_b": int(n_b),
    }


# ---------------------------------------------------------------------------
# 5. Seeded SYNTHETIC bench (AGENTS.md honesty contract #2)
# ---------------------------------------------------------------------------


def _simulate_gbm_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    *,
    mu_low: float,
    mu_high: float,
    sigma: float,
) -> tuple[Array, Array]:
    """Drifted-Brownian log-price paths plus the per-path true drift.

    increments ~ N(μ_k·dt, σ²·dt), μ_k ~ U(mu_low, mu_high), dt = 1/n_steps.
    Returns (n, T+1, 1) paths started at 0 and the (n,) drift labels.
    """
    dt = 1.0 / float(n_steps)
    mus = rng.uniform(mu_low, mu_high, n_paths)
    increments = rng.normal(0.0, math.sqrt(dt), (n_paths, n_steps)) * sigma
    increments += (mus * dt)[:, None]
    paths = np.concatenate(
        [np.zeros((n_paths, 1, 1)), np.cumsum(increments, axis=1)[:, :, None]], axis=1
    )
    return np.asarray(paths, dtype=float), mus


def _simulate_ou_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    *,
    theta: float,
    sigma: float,
) -> Array:
    """Mean-reverting OU paths dx = −θ x dt + σ dW, x_0 = 0, (n, T+1, 1)."""
    dt = 1.0 / float(n_steps)
    shocks = rng.normal(0.0, math.sqrt(dt), (n_paths, n_steps)) * sigma
    paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] - theta * paths[:, t] * dt + shocks[:, t]
    return paths[:, :, None]


def _linear_r2(
    features: Array, targets: Array, *, train_frac: float = 0.7, ridge: float = 1e-8
) -> dict[str, float]:
    """Deterministic-split ridge OLS R²: first `train_frac` rows fit, rest test.

    Closed-form ridge normal equations; no sklearn dependency. Fail-closed
    on a constant test target (R² undefined) or non-finite fit.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 10:
        raise ValueError("features must be (n, p) matching targets with n >= 10")
    if not bool(np.all(np.isfinite(x))) or not bool(np.all(np.isfinite(y))):
        raise ValueError("features and targets must be finite")
    n_train = int(round(x.shape[0] * train_frac))
    if n_train < 5 or x.shape[0] - n_train < 3:
        raise ValueError("train/test split too small")
    x_tr, x_te = x[:n_train], x[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    mu_x, sd_x = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd_x = np.where(sd_x <= 0.0, 1.0, sd_x)  # constant column stays unused
    z_tr = (x_tr - mu_x) / sd_x
    z_te = (x_te - mu_x) / sd_x
    gram = z_tr.T @ z_tr + ridge * np.eye(z_tr.shape[1])
    coef = np.linalg.solve(gram, z_tr.T @ y_tr)
    intercept = float(y_tr.mean())
    pred_te = z_te @ coef + intercept
    pred_tr = z_tr @ coef + intercept
    ss_te = float(np.sum((y_te - pred_te) ** 2))
    sst_te = float(np.sum((y_te - float(y_te.mean())) ** 2))
    if sst_te <= 0.0:
        raise ValueError("test targets are constant — R² undefined")
    ss_tr = float(np.sum((y_tr - pred_tr) ** 2))
    sst_tr = float(np.sum((y_tr - float(y_tr.mean())) ** 2))
    return {
        "r2_test": float(1.0 - ss_te / sst_te),
        "r2_train": float(1.0 - ss_tr / sst_tr) if sst_tr > 0.0 else float("nan"),
    }


def bench_signature_features(seed: int) -> dict[str, float]:
    """Seeded SYNTHETIC bench for the signature-feature lane.

    World: scalar log-price paths (T=64) plus the normalized clock channel
    (2 channels), GBM-with-drift vs mean-reverting OU and a low/high vol
    split (sigma 0.4 vs 1.6, 3 replicates). Emits:

    - ``synthetic_mmd_power_gbm_vs_ou`` / ``synthetic_mmd_fp_gbm_vs_gbm``:
      empirical rejection rate at α=0.05 of the truncated-kernel
      permutation test over B=6 seeded replicates (power under the
      alternative; false-positive rate under the GBM-vs-GBM null).
    - ``synthetic_mmd_power_volshift``: same rejection-rate machinery on
      a harder contrast — same GBM law at σ=0.4 vs 1.6 — over 6 seeded
      replicates; honest power at a near-boundary shift.
    - ``synthetic_pde_mmd2_*`` / ``synthetic_pde_mmd_pval_*``: the same
      contrast with the untruncated PDE kernel on one pooled ensemble each.
    - ``synthetic_pde_refine_gap_l01`` / ``_l12``: consecutive dyadic-level
      kernel gaps (convergence evidence), and
      ``synthetic_pde_vs_truncated_abs_gap``: |PDE − order-6 truncated|
      cross-check on one pair.
    - ``synthetic_logsig_drift_r2_test`` / ``_train``: Lyndon-basis
      logsignature(order 3) ridge-regression R² for per-path drift labels
      — the level-1 displacement coordinate should carry most of it.
    - ``synthetic_gram_min_eig`` / ``_symmetry_err`` / ``_diag_min``: PSD /
      symmetry / positivity sanity on a PDE Gram over a mixed ensemble.

    SYNTHETIC correctness numbers only — never market evidence.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    n_steps = 64
    n_paths = 24
    n_perm = 99
    n_replicates = 6

    def _ensembles() -> tuple[Array, Array, Array]:
        gbm_a, _ = _simulate_gbm_paths(rng, n_paths, n_steps, mu_low=-0.05, mu_high=0.05, sigma=0.9)
        ou = _simulate_ou_paths(rng, n_paths, n_steps, theta=8.0, sigma=0.7)
        gbm_b, _ = _simulate_gbm_paths(rng, n_paths, n_steps, mu_low=-0.05, mu_high=0.05, sigma=0.9)
        return gbm_a, ou, gbm_b

    def _augmented(batch: Array) -> list[Array]:
        return [time_augmentation(batch[i]) for i in range(batch.shape[0])]

    # --- MMD power / false-positive rates (truncated kernel, fast Grams) ---
    power_hits = 0
    fp_hits = 0
    mmd2_obs_first = math.nan
    pval_obs_first = math.nan
    for rep in range(n_replicates):
        gbm_a, ou, gbm_b = _ensembles()
        a_aug = _augmented(gbm_a)
        alt = signature_mmd_test(
            a_aug,
            _augmented(ou),
            seed=int(rng.integers(0, 2**31 - 1)),
            n_permutations=n_perm,
            kernel="truncated",
            order=3,
            sigma=2.0,
        )
        null = signature_mmd_test(
            a_aug,
            _augmented(gbm_b),
            seed=int(rng.integers(0, 2**31 - 1)),
            n_permutations=n_perm,
            kernel="truncated",
            order=3,
            sigma=2.0,
        )
        if rep == 0:
            mmd2_obs_first = float(alt["mmd2_biased"])
            pval_obs_first = float(alt["p_value"])
        power_hits += int(bool(alt["reject_5pct"]))
        fp_hits += int(bool(null["reject_5pct"]))
    power = power_hits / n_replicates
    fp_rate = fp_hits / n_replicates

    # --- Vol-regime contrast: same law (GBM), different sigma ---
    # Same B=6 replicate scheme as the GBM/OU power: the σ-shift is a
    # harder contrast, so its rejection rate is honest partial power.
    vol_rejects = 0
    vol_mmd2_first = math.nan
    vol_pval_first = math.nan
    for rep in range(n_replicates):
        low_vol, _ = _simulate_gbm_paths(rng, n_paths, n_steps, mu_low=0.0, mu_high=0.0, sigma=0.4)
        high_vol, _ = _simulate_gbm_paths(rng, n_paths, n_steps, mu_low=0.0, mu_high=0.0, sigma=1.6)
        vol_test = signature_mmd_test(
            _augmented(low_vol),
            _augmented(high_vol),
            seed=seed * 7919 + 17 + rep,
            n_permutations=n_perm,
            kernel="truncated",
            order=3,
            sigma=2.0,
        )
        if rep == 0:
            vol_mmd2_first = float(vol_test["mmd2_biased"])
            vol_pval_first = float(vol_test["p_value"])
        vol_rejects += int(bool(vol_test["reject_5pct"]))
    vol_power = vol_rejects / n_replicates

    # --- PDE-kernel MMD headline + Gram sanity ---
    gbm_a, ou, gbm_b = _ensembles()
    pde_alt = signature_mmd_test(
        _augmented(gbm_a),
        _augmented(ou),
        seed=seed * 104729 + 3,
        n_permutations=n_perm,
        kernel="pde",
        dyadic_level=1,
        sigma=2.0,
    )
    pde_null = signature_mmd_test(
        _augmented(gbm_a),
        _augmented(gbm_b),
        seed=seed * 1299709 + 5,
        n_permutations=n_perm,
        kernel="pde",
        dyadic_level=1,
        sigma=2.0,
    )
    pooled = _augmented(gbm_a) + _augmented(ou)
    gram = signature_gram(pooled, kernel="pde", dyadic_level=1, sigma=2.0)
    gram_min_eig = float(np.linalg.eigvalsh(gram).min())
    gram_diag_min = float(np.min(np.diag(gram)))

    # --- PDE convergence + truncated cross-check on one pair ---
    pa = _augmented(gbm_a)[0]
    pb = _augmented(ou)[0]
    diag = signature_kernel_pde_diagnostics(pa, pb, max_dyadic_level=2)
    gap_l01 = float(diag["gaps"][0]) if len(diag["gaps"]) >= 1 else math.nan
    gap_l12 = float(diag["gaps"][1]) if len(diag["gaps"]) >= 2 else math.nan
    gap_ref = float(diag["abs_gap_to_reference"])

    # --- Logsignature drift regression: labels = per-path true drift ---
    drift_paths, drift_labels = _simulate_gbm_paths(
        rng, 60, n_steps, mu_low=-0.5, mu_high=0.5, sigma=0.15
    )
    features_aug = signature_feature_matrix(
        drift_paths, order=3, basis="logsignature", augmentations=("time",)
    )
    r2 = _linear_r2(features_aug, drift_labels)

    return {
        "synthetic_mmd_power_gbm_vs_ou": float(power),
        "synthetic_mmd_fp_gbm_vs_gbm": float(fp_rate),
        "synthetic_mmd2_gbm_vs_ou_rep0": mmd2_obs_first,
        "synthetic_mmd_pval_gbm_vs_ou_rep0": pval_obs_first,
        "synthetic_mmd2_volshift": vol_mmd2_first,
        "synthetic_mmd_pval_volshift": vol_pval_first,
        "synthetic_mmd_power_volshift": vol_power,
        "synthetic_pde_mmd2_gbm_vs_ou": float(pde_alt["mmd2_biased"]),
        "synthetic_pde_mmd_pval_gbm_vs_ou": float(pde_alt["p_value"]),
        "synthetic_pde_mmd_reject_gbm_vs_ou": float(bool(pde_alt["reject_5pct"])),
        "synthetic_pde_mmd2_gbm_vs_gbm": float(pde_null["mmd2_biased"]),
        "synthetic_pde_mmd_pval_gbm_vs_gbm": float(pde_null["p_value"]),
        "synthetic_pde_refine_gap_l01": gap_l01,
        "synthetic_pde_refine_gap_l12": gap_l12,
        "synthetic_pde_vs_truncated_abs_gap": gap_ref,
        "synthetic_gram_min_eig": gram_min_eig,
        "synthetic_gram_diag_min": gram_diag_min,
        "synthetic_gram_symmetry_err": float(np.max(np.abs(gram - gram.T))),
        "synthetic_logsig_drift_r2_test": float(r2["r2_test"]),
        "synthetic_logsig_drift_r2_train": float(r2["r2_train"]),
        "synthetic_logsig_feature_dim": float(features_aug.shape[1]),
        "synthetic_n_replicates": float(n_replicates),
        "synthetic_n_paths": float(n_paths),
        "synthetic_n_steps": float(n_steps),
        "synthetic_n_permutations": float(n_perm),
    }
