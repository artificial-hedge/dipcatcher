"""Anytime-valid rank inference for model comparison on shared items.

Leaderboard-style diagnostics: several models are scored on a COMMON set of
benchmark items, so their per-item scores are dependent in unknown ways, and
the standings are inspected repeatedly while the evaluation is still running.
Fixed-sample rank intervals break under such monitoring; the constructions
below keep their error guarantees at every sample size simultaneously, hence
under any (even data-dependent) stopping rule.

1. Rank confidence sequences (Khosravi & Huo, 2026, "Rank Confidence
   Sequences: Anytime-Valid Leaderboards", arXiv:2609.32211). For every model
   j a set of ranks R_{j,t} that contains its true rank R_j(theta) =
   1 + #{l: theta_l > theta_j} simultaneously for all models and at all times
   with probability >= 1 - alpha (their Section 2 "Goal"). Construction
   (their Section 3):
   - Step 1, pairwise betting wealths (their Eq. (2)-(3)): for each ordered
     pair (j, l), on paired item differences Z_t^{jl} = X_{tj} - X_{tl} in
     [-1, 1],
         E_{lambda,t}^{jl} = E_{lambda,t-1}^{jl} (1 + lambda (Z_t - b_t)/(1 + b_t)),
         E_t^{jl} = mean over lambda in Lambda of E_{lambda,t}^{jl},
     with a fixed bet grid Lambda (default their recommended
     {0.03, 0.06, 0.12, 0.25, 0.5}) and a PREDICTABLE offset b_t bounding
     E[Z_t | F_{t-1}] under H_{jl}: theta_j <= theta_l: b_t = 0 for the i.i.d.
     superpopulation model (S), and b_t = max{-0.99, min{1, -S_{t-1}/(N-t+1)}}
     for a finite benchmark of N items evaluated in random order (F) — the
     sampling-without-replacement betting of Waudby-Smith & Ramdas (2024,
     JRSS-B 86(1):1-27, arXiv:2010.09686), whose linear-bet capital process
     Eq. (2) is (their Lemma B.1 gives the supermartingale property; any
     within-item dependence across models is allowed).
   - Step 2, closed testing over WEAK ORDERS (ties allowed; Marcus, Peritz &
     Gabriel, 1976, Biometrika 63(3):655-660; partitioning principle of
     Finner & Strassburger, 2002): each weak order W is tested by the average
     wealth E_t^W = mean over T(W) = {(j,l): j <=_W l} of E_t^{jl} (an average
     of e-processes is an e-process under arbitrary dependence, Vovk & Wang,
     2021, Annals of Statistics 49(3):1736-1754); W is eliminated permanently
     once max_{s<=t} E_s^W >= 1/alpha. Two exact routes are provided:
     * their Algorithm 1 (enumeration of all Fubini(M) weak orders — 13 for
       M=3, 545,835 for M=8), exact and practical up to the documented cap
       ``_RCS_MAX_K_EXACT = 8``; it alone yields the EXACT rank sets
       R_{j,t} = {R_j(W): W survives};
     * their Algorithm 3 (integer programming, exact for any M) is NOT
       implemented here — the enumeration covers small M exactly and their
       polynomial-time Algorithm 2 covers any M.
   - Their Algorithm 2 (polynomial-time shortcut, any M): certify j > l once
     B_t^{jl} = E_t^{jl} + sum_{m not in {j,l}} min{E_t^{jm}, E_t^{ml}} >=
     M(M-1)/alpha (transitivity pooling; at least as much as e-Bonferroni,
     a subset of the exact D_t), then take the transitive closure.
   - Step 3, reports (their Theorems 4.2-4.4): certified dominances D_t,
     rank intervals L_{j,t} = 1 + #{l: (l,j) in D_t}, U_{j,t} =
     M - #{l: (j,l) in D_t}, top-k status (U_j <= k inside, L_j > k outside),
     and tiers (1 + longest certified chain above j). If every ordering is
     eliminated (probability <= alpha) the report is flagged ``inconsistent``
     and fails closed to the widest intervals.

2. BB-EDGE (Gao, Zhang, Xie, Jing, Wang & Liu, 2026, "Anytime-Valid LLM
   Leaderboards via Benchmark-weighted and Block-Factorized e-Processes",
   arXiv:2609.32248). Estimand: benchmark-mean differences
   Delta_{a,b} = theta_a^B - theta_b^B on a FIXED benchmark observed over
   successive complete replicates (time index = completed replicate, not
   item); directional nulls H_e^(tau): Delta <= tau for a margin tau >= 0.
   - Benchmark-weighted block-factorized empirical-Bernstein e-process
     (their Eqs. (3)-(7)): items are partitioned into protocol-defined blocks
     C_m with weights w_m = |C_m|/N; block means Y_{e,m,r} of
     X = (S^(a) - S^(b) + 1)/2 in [0, 1] preserve the benchmark average
     (their Eq. (3)); stakes are WEIGHT-PROPORTIONAL lambda_{g,m} =
     lambda_g w_m / w_star, w_star = max_m w_m, which their Proposition 1
     shows is necessary and sufficient for nonpositive linear drift under the
     heterogeneous benchmark-average null (individual block means may exceed
     mu_0 = (1+tau)/2). Per-replicate factor (their Eq. (6), the predictable
     empirical-Bernstein construction of Waudby-Smith & Ramdas, 2024):
         F_{e,r}(lambda_g) = exp{ sum_m [ lambda_{g,m} (Y_{e,m,r} - mu_0)
                              - psi_E(lambda_{g,m}) (Y_{e,m,r} - Yhat_{e,m,r})^2 ] },
         psi_E(l) = -log(1-l) - l,
     mixed over a stake grid (default: their G.4 choice, 41 equally spaced
     stakes on [0, 0.95] with uniform mixture weights) and multiplied across
     replicates. The predictable prediction Yhat_{e,m,r} is the block running
     mean initialized at mu_0, Yhat_{e,m,r} = (mu_0 + sum_{s<r} Y_{e,m,s})/r
     (their G.4; mirrors the regularized running estimators of
     :mod:`quant_fund.metrics.confidence_sequences`). M = 1 block permits
     arbitrary within-replicate dependence (their Remark 1 fallback).
   - Direct e-Holm across the L(L-1) directions at every replicate (their
     Eq. (8), Hartog & Lei, 2025, "Family-wise error rate control with
     e-values", arXiv:2501.09015, Theorem 4.2): with
     J_r = {e: E_{e,r} < 1/alpha}, threshold
     c_r = 1/alpha + sum_{e in J_r} (1/alpha - E_{e,r}), certify
     E_{e,r} >= c_r. Their Theorem 1: anytime FWER <= alpha under arbitrary
     within-block and cross-pair dependence. NOTE: this is direct e-Holm on
     e-PROCESSES, distinct from the static e-BH machinery in
     :mod:`quant_fund.metrics.anytime_fdr`; it is re-derived here (one line,
     their Eq. (8)) rather than imported.
   - Top-k certification (their Appendix A, Theorem 2): independent pilot
     replicates select the candidate set T_0 (ties by model index); e-Holm at
     level alpha is then run on the restricted family of k(L-k) cross-set
     directions from the confirmatory replicates; T_0 is certified once all
     are rejected at one replicate. Anytime false-certification probability
     <= alpha even if the pilot candidate is wrong.
   - Simultaneous rank intervals (their Appendix B, Eq. (13), Corollary 1):
     I_{l,r} = [1 + |A_{l,r}|, L - |D_{l,r}|] from the transitive closure of
     the certified graph; with probability >= 1 - alpha they cover the true
     benchmark ranks of all models at every replicate r <= R.

Re-derivation note (composition boundary with wave-12):
:func:`quant_fund.metrics.confidence_sequences.wsr_cs` and
:func:`~quant_fund.metrics.confidence_sequences.empirical_bernstein_cs`
INVERT wealth/boundary constructions into confidence intervals for one
stream mean; neither exposes the wealth PATHS that closed testing (averages of
pairwise e-processes over orderings) and direct e-Holm require, and the
Khosravi-Huo bets are grid-averaged constant-fraction bets with a
without-replacement offset rather than WSR's hedged truncated bets. Both
constructions are therefore re-derived here from the cited equations, in the
same bounded-difference betting family (Waudby-Smith & Ramdas, 2024);
``psi_E`` matches ``quant_fund.metrics.evalues._psi_exp`` (the sub-exponential
CGF bound). Conventions (fail-closed validation, e_cap clipping, SYNTHETIC
seeded Monte-Carlo in tests) follow the wave-12 module.

Honesty: rank inference over model benchmark scores is a RESEARCH DIAGNOSTIC.
Nothing here is market evidence, a performance claim (no Sharpe/Sortino/P&L
content), or a live-trading capability; all Monte-Carlo validation in the
accompanying tests uses seeded SYNTHETIC item-score matrices and is a
correctness check of the error control only. Every function is deterministic
in its inputs (no internal randomness); under sampling model (F) the caller
must supply the benchmark in a pre-randomized order drawn from a recorded
seed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

__all__ = [
    "BBEdgeResult",
    "BBEdgeTopKResult",
    "RankConfidenceSequenceResult",
    "bb_edge_certify",
    "bb_edge_e_values",
    "bb_edge_topk_certify",
    "check_dominance_fwer",
    "check_edge_fwer",
    "check_rank_set_coverage",
    "check_rank_time_uniform_coverage",
    "closed_testing_rank_sets",
    "direct_e_holm",
    "direction_pairs",
    "enumerate_weak_orders",
    "pairwise_betting_wealth",
    "rank_confidence_sequence",
    "shortcut_certify",
    "topk_status",
]

# Paper defaults / caps.
_RCS_GRID_DEFAULT: tuple[float, ...] = (0.03, 0.06, 0.12, 0.25, 0.5)
# Khosravi & Huo (2026, Section 3 "Computing Step 2" and Appendix A):
# enumeration of the Fubini(M) weak orders is exact and "practical up to
# about M = 8" (Fubini(8) = 545,835). Beyond the cap, method="exact"
# fails closed; method="shortcut" (their Algorithm 2) supports any M.
_RCS_MAX_K_EXACT = 8
_BB_STAKE_GRID_DEFAULT_N = 41
_BB_STAKE_GRID_DEFAULT_MAX = 0.95
_E_MAX = 1e300
_FUBINI = {1: 1, 2: 3, 3: 13, 4: 75, 5: 541, 6: 4683, 7: 47293, 8: 545835}

_WEAK_ORDER_CACHE: dict[int, IntArray] = {}
_LE_MASK_CACHE: dict[int, tuple[BoolArray, IntArray]] = {}
_MASK_CHUNK = 1 << 17


# ---------------------------------------------------------------------------
# validation helpers (fail-closed, mirroring confidence_sequences.py)
# ---------------------------------------------------------------------------


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must lie in the open interval (0, 1)")
    return a


def _check_tau(tau: float) -> float:
    t = float(tau)
    if not np.isfinite(t) or not 0.0 <= t < 1.0:
        raise ValueError("tau must lie in the half-open interval [0, 1)")
    return t


def _check_score_matrix(x: Array | Iterable[Iterable[float]], name: str = "scores") -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be a nonempty (n_items, n_models>=2) 2-D array")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    if bool(np.any(arr < 0.0)) or bool(np.any(arr > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1] (out-of-support scores rejected)")
    return arr


def _check_panels(panels: Array | Iterable[object], name: str = "panels") -> Array:
    arr = np.asarray(panels, dtype=float)
    if arr.ndim != 3 or min(arr.shape) < 1 or arr.shape[2] < 2:
        raise ValueError(f"{name} must be a (n_replicates, n_items, n_models>=2) 3-D array")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    if bool(np.any(arr < 0.0)) or bool(np.any(arr > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1] (out-of-support scores rejected)")
    return arr


def _check_grid(grid: Sequence[float] | None, default: Sequence[float], name: str) -> Array:
    seq = default if grid is None else grid
    arr = np.asarray(list(seq), dtype=float).reshape(-1)
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be a nonempty finite sequence")
    if bool(np.any(arr < 0.0)) or bool(np.any(arr >= 1.0)):
        raise ValueError(f"{name} entries must lie in [0, 1)")
    return arr


def _check_wealth(wealth: Array, name: str = "wealth") -> Array:
    w = np.asarray(wealth, dtype=float)
    if w.ndim != 3 or w.shape[0] < 2 or w.shape[1] < 2 or w.shape[1] != w.shape[2]:
        raise ValueError(f"{name} must be a (T+1>=2, M>=2, M) wealth-path array")
    if not bool(np.all(np.isfinite(w))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    if bool(np.any(w < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    return w


def direction_pairs(m: int) -> list[tuple[int, int]]:
    """Ordered pairs (a, b), a != b, in lexicographic order (column layout)."""
    k = int(m)
    if k < 2:
        raise ValueError("direction_pairs needs at least 2 models")
    return [(a, b) for a in range(k) for b in range(k) if a != b]


def _psi_e(lam: Array) -> Array:
    """psi_E(l) = -log(1-l) - l, the sub-exponential CGF bound on [0, 1].

    Same function as ``quant_fund.metrics.evalues._psi_exp`` (Choe-Ramdas /
    Waudby-Smith & Ramdas 2024); used by BB-EDGE Eq. (6) (Gao et al., 2026).
    Valid (finite) for l in [0, 1); callers enforce the stake grid range.
    """
    out = -np.log(1.0 - lam) - lam
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# weak orders (Khosravi & Huo 2026, Section 3, Step 2)
# ---------------------------------------------------------------------------


def enumerate_weak_orders(k: int) -> IntArray:
    """All weak orders (tie-allowed rankings) on k models, as level vectors.

    Row w holds levels (v_1, ..., v_k) with j <=_W l iff v_j <= v_l (higher
    level = better; the paper's representation, Section 3 "Step 2"). The count
    is the Fubini (ordered Bell) number: 3, 13, 75, 541, 4683, 47293, 545835
    for k = 2..8. Generation is incremental and deterministic: every weak
    order on j+1 models restricts to a unique weak order on the first j, and
    model j+1 is inserted either tied with one of the existing levels or
    strictly at one of (#levels + 1) fresh positions, so each order appears
    exactly once. Cached per k.
    """
    kk = int(k)
    if kk < 1:
        raise ValueError("k must be >= 1")
    if kk in _WEAK_ORDER_CACHE:
        return _WEAK_ORDER_CACHE[kk]
    arr = np.zeros((1, 1), dtype=np.int64)
    for j in range(1, kk):
        maxlv = arr.max(axis=1)
        pieces: list[IntArray] = []
        col_range = np.arange(j + 1)
        # tie model j with existing level m (requires m <= maxlv per row)
        for m in col_range[:j]:
            sel = maxlv >= m
            if bool(np.any(sel)):
                block = arr[sel]
                piece = np.hstack([block, np.full((int(block.shape[0]), 1), int(m))])
                pieces.append(piece)
        # strict insertion with exactly s levels strictly below (s = 0..maxlv+1)
        for s in col_range:
            sel = (maxlv + 1) >= s
            if bool(np.any(sel)):
                block = arr[sel]
                shifted = np.where(block >= s, block + 1, block)
                piece = np.hstack([shifted, np.full((int(block.shape[0]), 1), int(s))])
                pieces.append(piece)
        arr = np.vstack(pieces).astype(np.int64)
    if kk in _FUBINI and int(arr.shape[0]) != _FUBINI[kk]:  # pragma: no cover
        raise ValueError(f"internal error: expected Fubini({kk}) = {_FUBINI[kk]} weak orders")
    _WEAK_ORDER_CACHE[kk] = arr
    return arr


def _le_mask_table(k: int) -> tuple[BoolArray, IntArray]:
    """Boolean (n_W, P) mask of T(W) = {(j,l): v_j <= v_l} plus |T(W)| counts.

    Column order of the mask matches :func:`direction_pairs` (and hence the
    off-diagonal layout of the wealth matrices).
    """
    if k in _LE_MASK_CACHE:
        return _LE_MASK_CACHE[k]
    levels = enumerate_weak_orders(k)
    pairs = direction_pairs(k)
    pj = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    pl = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    mask = levels[:, pj] <= levels[:, pl]  # (n_W, P)
    counts = mask.sum(axis=1).astype(np.int64)
    if int(counts.min()) < 1:
        raise ValueError("internal error: empty T(W) encountered")  # pragma: no cover
    _LE_MASK_CACHE[k] = (mask, counts)
    return mask, counts


def _order_wealth(mask: BoolArray, counts: IntArray, e_pairs: Array) -> Array:
    """E_t^W = mean over T(W) of E_t^{jl} for every weak order (chunked)."""
    n_w = int(mask.shape[0])
    out = np.empty(n_w, dtype=np.float64)
    for start in range(0, n_w, _MASK_CHUNK):
        blk = mask[start : start + _MASK_CHUNK].astype(np.float64)
        out[start : start + blk.shape[0]] = blk @ e_pairs
    return out / counts


def _transitive_closure(rel: BoolArray) -> BoolArray:
    """Transitive closure of a (M, M) relation by repeated boolean squaring."""
    r = np.array(rel, dtype=bool, copy=True)
    np.fill_diagonal(r, False)
    m = int(r.shape[0])
    steps = max(1, int(np.ceil(np.log2(max(m, 2)))) + 1)
    for _ in range(steps):
        r8 = r.astype(np.uint8)
        new = r | ((r8 @ r8) > 0)
        if np.array_equal(new, r):
            break
        r = new
    return r


def _tiers_from_dominance(dom: BoolArray) -> IntArray:
    """tier(j) = 1 + max{tier(l): l > j certified} (longest certified chain).

    Khosravi & Huo (2026, Step 3 "Tiers"). On a strict partial order the
    fixed point is reached in <= M sweeps; a cycle (bad event, flagged
    separately) leaves the last sweep's values.
    """
    m = int(dom.shape[0])
    tiers = np.ones(m, dtype=np.int64)
    for _ in range(m):
        cand = np.where(dom, tiers[:, None], 0)
        new = 1 + cand.max(axis=0)
        if np.array_equal(new, tiers):
            break
        tiers = new
    return tiers


def _report_from_dominance(dom: BoolArray, m: int) -> tuple[IntArray, IntArray, IntArray]:
    """Eq. (6) rank interval endpoints and tiers from a closed dominance set."""
    lower = 1 + dom.sum(axis=0).astype(np.int64)  # #{l: l > j}
    upper = m - dom.sum(axis=1).astype(np.int64)  # M - #{l: j > l}
    tiers = _tiers_from_dominance(dom)
    return lower, upper, tiers


def topk_status(rank_lower: IntArray, rank_upper: IntArray, k: int) -> dict[str, BoolArray]:
    """Certified top-k membership from rank intervals (Khosravi & Huo, Step 3).

    ``inside[j]`` iff U_j <= k (model j is among the k best; ties can put more
    than k models inside), ``outside[j]`` iff L_j > k. Accepts (T+1, M) or
    (M,) endpoints; entries that are neither are unresolved.
    """
    lo = np.asarray(rank_lower, dtype=np.int64)
    up = np.asarray(rank_upper, dtype=np.int64)
    if lo.shape != up.shape or lo.ndim not in (1, 2) or lo.size == 0:
        raise ValueError("rank_lower/rank_upper must be matching nonempty (M,) or (T+1, M)")
    kk = int(k)
    if kk < 1 or kk > lo.shape[-1]:
        raise ValueError("k must lie in [1, M]")
    return {"inside": up <= kk, "outside": lo > kk}


# ---------------------------------------------------------------------------
# Step 1: pairwise betting wealths (Khosravi & Huo 2026, Eq. (2)-(3))
# ---------------------------------------------------------------------------


def pairwise_betting_wealth(
    scores: Array,
    *,
    sampling: str = "superpopulation",
    grid: Sequence[float] | None = None,
    e_cap: float = _E_MAX,
) -> Array:
    """Pairwise betting e-processes for H_{jl}: theta_j <= theta_l.

    ``scores`` is the (N, M) item-by-model matrix in [0, 1], rows in
    EVALUATION ORDER (under the finite-benchmark model (F) the caller must
    pre-randomize the order from a recorded seed; under (S) rows are i.i.d.
    draws). Returns the wealth paths E_t^{jl} of their Eq. (2), shape
    (N+1, M, M), with E_0 = 1 and the diagonal fixed at 1; entry [t, j, l]
    is the evidence at time t that model j outscores model l. Under either
    sampling model and ANY within-item dependence across models, each path is
    a nonnegative test supermartingale (e-process) under its null (their
    Lemma B.1), so Ville gives P(any t: E_t^{jl} >= 1/alpha) <= alpha.

    ``sampling``: "superpopulation" (S, offset b_t = 0) or "finite" (F, offset
    b_t = max{-0.99, min{1, -S_{t-1}/(N-t+1)}} for sampling without
    replacement; ``scores`` must then be the COMPLETE benchmark of N items —
    the offset formula is invalid for a prefix, which the function cannot
    detect, so callers must fail closed on partial input).

    ``grid`` is the bet grid Lambda (default their recommended
    {0.03, 0.06, 0.12, 0.25, 0.5}); the grid affects power only, never
    validity (any predictable bet in [0, 1) keeps the factor nonnegative and
    the process a supermartingale). Wealths are capped at ``e_cap`` to avoid
    float overflow on long dominant streams; the cap dwarfs every decision
    threshold used downstream.
    """
    x = _check_score_matrix(scores)
    mode = str(sampling)
    if mode not in ("superpopulation", "finite"):
        raise ValueError("sampling must be 'superpopulation' or 'finite'")
    lam = _check_grid(grid, _RCS_GRID_DEFAULT, "grid")
    cap = float(e_cap)
    if not np.isfinite(cap) or cap <= 1.0:
        raise ValueError("e_cap must be finite and > 1")
    n_items, m = int(x.shape[0]), int(x.shape[1])
    z = x[:, :, None] - x[:, None, :]  # z[t, j, l] = X_{tj} - X_{tl}
    if mode == "finite":
        s_prev = np.concatenate(
            [np.zeros((1, m, m), dtype=float), np.cumsum(z, axis=0)[:-1]], axis=0
        )
        remaining = (n_items - np.arange(1, n_items + 1, dtype=float) + 1.0)[:, None, None]
        b = np.clip(-s_prev / remaining, -0.99, 1.0)
    else:
        b = np.zeros_like(z)
    adj = (z - b) / (1.0 + b)  # in [-1, inf); with the offset, >= -1 always
    with np.errstate(over="ignore"):
        acc = np.zeros((n_items, m, m), dtype=float)
        for g in lam:
            acc += np.cumprod(1.0 + float(g) * adj, axis=0)
        wealth = np.empty((n_items + 1, m, m), dtype=float)
        wealth[0] = 1.0
        wealth[1:] = np.clip(acc / float(lam.size), 0.0, cap)
    idx = np.arange(m)
    wealth[:, idx, idx] = 1.0
    return wealth


# ---------------------------------------------------------------------------
# Steps 2-3: closed testing (exact) and the polynomial-time shortcut
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankConfidenceSequenceResult:
    """Output of :func:`rank_confidence_sequence` (times t = 0..N on axis 0).

    ``dominance[t, j, l]`` is True iff j > l is certified at time t (already
    transitively closed); ``rank_lower``/``rank_upper`` are the Eq. (6)
    interval endpoints; ``rank_sets`` (exact method only; None for the
    shortcut) is the exact rank set membership [t, j, rank-1]; ``tiers`` is
    1 + the longest certified chain above each model; ``inconsistent[t]``
    flags the bad event (no surviving ordering / a dominance cycle), which
    has probability <= alpha, after which reports fail closed to the widest
    intervals. On the good event all certified dominances are true at all
    times simultaneously (their Theorem 4.2), and the intervals/rank sets
    cover the true ranks of ALL models at ALL times (their Theorem 4.3).
    """

    wealth: Array
    dominance: BoolArray
    rank_lower: IntArray
    rank_upper: IntArray
    rank_sets: BoolArray | None
    tiers: IntArray
    inconsistent: BoolArray
    alpha: float
    method: str
    sampling: str

    @property
    def n_models(self) -> int:
        """Number of models M."""
        return int(self.rank_lower.shape[-1])

    @property
    def n_times(self) -> int:
        """Number of recorded times T+1 (including t = 0)."""
        return int(self.rank_lower.shape[0])

    def top_k(self, k: int) -> dict[str, BoolArray]:
        """Certified top-k inside/outside status at every time (Step 3)."""
        return topk_status(self.rank_lower, self.rank_upper, k)


def closed_testing_rank_sets(
    wealth: Array,
    *,
    alpha: float = 0.05,
    sampling: str = "superpopulation",
) -> RankConfidenceSequenceResult:
    """Exact rank confidence sequence by enumeration (their Algorithm 1).

    ``wealth`` holds the (T+1, M, M) pairwise wealth paths (from
    :func:`pairwise_betting_wealth` or any Assumption-4.1-conformant
    e-process family). Every weak order W is eliminated permanently once
    max_{s<=t} E_s^W >= 1/alpha, E^W the average wealth over T(W) (their
    Eq. (4)); certified dominances are the pairs every survivor agrees on
    (Eq. (5)) and exact rank sets are the survivors' rank projections. Cost
    is O(T * Fubini(M) * M^2); requires M <= ``_RCS_MAX_K_EXACT`` (8). If all
    orderings die (probability <= alpha) the result is flagged inconsistent
    and fails closed (empty dominance, widest intervals, empty rank sets).
    """
    a = _check_alpha(alpha)
    w = _check_wealth(wealth)
    n_times, m = int(w.shape[0]), int(w.shape[1])
    if m > _RCS_MAX_K_EXACT:
        raise ValueError(
            f"exact closed testing enumerates Fubini(M) weak orders and is capped at "
            f"M={_RCS_MAX_K_EXACT} (got M={m}); use method='shortcut' (Algorithm 2) instead"
        )
    levels = enumerate_weak_orders(m)
    mask, counts = _le_mask_table(m)
    pairs = direction_pairs(m)
    pj = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    pl = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    inv_alpha = 1.0 / a
    n_w = int(levels.shape[0])
    run_max = np.ones(n_w, dtype=float)
    alive = np.ones(n_w, dtype=bool)
    dominance = np.zeros((n_times, m, m), dtype=bool)
    lower = np.ones((n_times, m), dtype=np.int64)
    upper = np.full((n_times, m), m, dtype=np.int64)
    tiers = np.ones((n_times, m), dtype=np.int64)
    rank_sets = np.zeros((n_times, m, m), dtype=bool)
    inconsistent = np.zeros(n_times, dtype=bool)
    for t in range(n_times):
        if t > 0:
            e_pairs = w[t][pj, pl]
            ew = _order_wealth(mask, counts, e_pairs)
            np.maximum(run_max, ew, out=run_max)
            alive &= run_max < inv_alpha
        if not bool(np.any(alive)):
            # Bad event (probability <= alpha): fail closed, report nothing.
            inconsistent[t:] = True
            break
        lv = levels[alive]
        # exact rank sets: R_j(W) = 1 + #{l: v_j < v_l}
        ranks = 1 + (lv[:, :, None] < lv[:, None, :]).sum(axis=2)  # (n_alive, M)
        onehot = np.zeros((int(lv.shape[0]), m, m), dtype=bool)
        rows = np.arange(int(lv.shape[0]))[:, None]
        cols = np.arange(m)[None, :]
        onehot[rows, cols, ranks - 1] = True
        rank_sets[t] = onehot.any(axis=0)
        # certified dominances: l <_W j for EVERY survivor
        dom_pairs = ~mask[alive].any(axis=0)
        dominance[t][pj, pl] = dom_pairs
        lower[t], upper[t], tiers[t] = _report_from_dominance(dominance[t], m)
    return RankConfidenceSequenceResult(
        wealth=w,
        dominance=dominance,
        rank_lower=lower,
        rank_upper=upper,
        rank_sets=rank_sets,
        tiers=tiers,
        inconsistent=inconsistent,
        alpha=a,
        method="exact",
        sampling=str(sampling),
    )


def shortcut_certify(
    wealth: Array,
    *,
    alpha: float = 0.05,
    sampling: str = "superpopulation",
) -> RankConfidenceSequenceResult:
    """Polynomial-time shortcut certification (their Algorithm 2, any M).

    Certify j > l once the transitivity-pooled statistic
    B_t^{jl} = E_t^{jl} + sum_{m not in {j,l}} min{E_t^{jm}, E_t^{ml}}
    reaches M(M-1)/alpha; certifications are permanent and transitively
    closed after every update. The certified set is a subset of the exact
    D_t and a superset of e-Bonferroni (every pair with E_t^{jl} >=
    M(M-1)/alpha). Cost O(T * M^3). ``rank_sets`` is None: exact rank sets
    require enumeration (Algorithm 1); the Eq. (6) intervals remain valid
    (their Theorem 4.3(b) holds for ANY certified set). A dominance cycle
    (bad event, probability <= alpha) is flagged inconsistent and the
    offending time's report fails closed to empty dominance and the widest
    intervals.
    """
    a = _check_alpha(alpha)
    w = _check_wealth(wealth)
    n_times, m = int(w.shape[0]), int(w.shape[1])
    thresh = float(m * (m - 1)) / a
    dominance = np.zeros((n_times, m, m), dtype=bool)
    lower = np.ones((n_times, m), dtype=np.int64)
    upper = np.full((n_times, m), m, dtype=np.int64)
    tiers = np.ones((n_times, m), dtype=np.int64)
    inconsistent = np.zeros(n_times, dtype=bool)
    cert = np.zeros((m, m), dtype=bool)
    keep = np.ones((m, m, m), dtype=bool)
    diag = np.arange(m)
    keep[diag, :, diag] = False  # exclude m == j
    keep[:, diag, diag] = False  # exclude m == l
    for t in range(n_times):
        if t > 0:
            e = w[t]
            pooled = np.minimum(e[:, None, :], e.T[None, :, :])  # [j,l,m]=min(E_jm,E_ml)
            b_stat = e + (np.where(keep, pooled, 0.0)).sum(axis=2)
            cert |= b_stat >= thresh
            np.fill_diagonal(cert, False)
        closed = _transitive_closure(cert)
        if bool(np.any(np.diag(closed))):
            inconsistent[t:] = True  # cycle: bad event, fail closed from here on
            break
        dominance[t] = closed
        lower[t], upper[t], tiers[t] = _report_from_dominance(closed, m)
    return RankConfidenceSequenceResult(
        wealth=w,
        dominance=dominance,
        rank_lower=lower,
        rank_upper=upper,
        rank_sets=None,
        tiers=tiers,
        inconsistent=inconsistent,
        alpha=a,
        method="shortcut",
        sampling=str(sampling),
    )


def rank_confidence_sequence(
    scores: Array,
    *,
    alpha: float = 0.05,
    method: str = "exact",
    sampling: str = "superpopulation",
    grid: Sequence[float] | None = None,
) -> RankConfidenceSequenceResult:
    """Anytime-valid rank sets/intervals for M models scored on shared items.

    Khosravi & Huo (2026, arXiv:2609.32211, Sections 2-3): ``scores`` is the
    (N, M) matrix of per-item scores in [0, 1] in evaluation order; with
    probability >= 1 - alpha the returned rank sets (``method="exact"``,
    M <= 8, their Algorithm 1) or rank intervals (both methods; Eq. (6))
    contain the true ranks of ALL models at ALL times t = 0..N
    simultaneously — valid under any stopping rule and any within-item
    dependence. ``method="shortcut"`` runs their polynomial-time Algorithm 2
    for any M (subset of the exact certifications, superset of
    e-Bonferroni). ``sampling`` selects the offset of their Eq. (3)
    ("superpopulation" for i.i.d. items, "finite" for a complete benchmark in
    pre-randomized order). Research diagnostic on model scores — never
    market evidence, never a live-trading claim.
    """
    x = _check_score_matrix(scores)
    mode = str(sampling)
    if mode not in ("superpopulation", "finite"):
        raise ValueError("sampling must be 'superpopulation' or 'finite'")
    meth = str(method)
    if meth not in ("exact", "shortcut"):
        raise ValueError("method must be 'exact' or 'shortcut'")
    if meth == "exact" and int(x.shape[1]) > _RCS_MAX_K_EXACT:
        raise ValueError(
            f"method='exact' is capped at M={_RCS_MAX_K_EXACT} models "
            f"(Fubini(M) enumeration); use method='shortcut'"
        )
    a = _check_alpha(alpha)
    wealth = pairwise_betting_wealth(x, sampling=mode, grid=grid)
    if meth == "exact":
        return closed_testing_rank_sets(wealth, alpha=a, sampling=mode)
    return shortcut_certify(wealth, alpha=a, sampling=mode)


# ---------------------------------------------------------------------------
# BB-EDGE (Gao et al. 2026, arXiv:2609.32248)
# ---------------------------------------------------------------------------


def direct_e_holm(e_values: Array, alpha: float) -> tuple[float, BoolArray]:
    """Direct e-Holm step (Hartog & Lei 2025, Thm 4.2; Gao et al. Eq. (8)).

    With J = {e: E_e < 1/alpha}, the data-dependent threshold is
    c = 1/alpha + sum_{e in J} (1/alpha - E_e) and hypothesis e is rejected
    iff E_e >= c. The threshold interpolates between 1/alpha (all e-values
    large) and n/alpha (all small), dominates Holm, and is valid under
    ARBITRARY dependence among the e-values/e-processes. Returns
    (threshold, rejected mask).
    """
    a = _check_alpha(alpha)
    e = np.asarray(e_values, dtype=float).reshape(-1)
    if e.size == 0 or not bool(np.all(np.isfinite(e))) or bool(np.any(e < 0.0)):
        raise ValueError("e_values must be nonempty, finite and nonnegative")
    inv = 1.0 / a
    not_yet = e < inv
    c = inv + float(np.sum(inv - e[not_yet]))
    return c, np.asarray(e >= c, dtype=bool)


def bb_edge_e_values(
    panels: Array,
    *,
    blocks: Sequence[int] | IntArray | None = None,
    tau: float = 0.0,
    stake_grid: Sequence[float] | None = None,
    mixture: Sequence[float] | None = None,
    directions: Sequence[tuple[int, int]] | None = None,
) -> Array:
    """Benchmark-weighted block-factorized e-processes (their Eqs. (3)-(7)).

    ``panels`` is the (R, N, L) array of scores in [0, 1]: R complete
    replicates of every one of L models on all N benchmark items (the time
    index is the completed replicate). ``blocks`` assigns each item a block
    id (default: a single block — their Remark 1 fallback, which allows
    ARBITRARY within-replicate dependence). ``tau`` is the superiority margin
    (null H_e: Delta_{a,b} <= tau); ``stake_grid`` defaults to their G.4
    choice (41 equally spaced stakes on [0, 0.95]); ``mixture`` defaults to
    uniform; ``directions`` defaults to all L(L-1) ordered pairs.

    For direction e = (a->b): X_{i,r} = (S^{(a)} - S^{(b)} + 1)/2 in [0, 1],
    block means Y_{e,m,r} preserve the benchmark average (Eq. (3)); stakes
    lambda_{g,m} = lambda_g w_m / w_star are weight-proportional, which their
    Proposition 1 shows is necessary and sufficient for nonpositive linear
    drift under the heterogeneous benchmark-average null; the factor is the
    predictable empirical-Bernstein term of Eq. (6) with psi_E(l) =
    -log(1-l) - l (Waudby-Smith & Ramdas 2024) and the running-mean
    prediction initialized at mu_0 = (1+tau)/2 (their G.4). Returns the
    mixture e-process values E_{e,r} of Eq. (7), shape (R, n_directions),
    each a nonnegative supermartingale across replicates under its null
    (their Theorem 1), capped at ``_E_MAX``.
    """
    p = _check_panels(panels)
    n_rep, n_items, n_models = (int(s) for s in p.shape)
    t = _check_tau(tau)
    lam = _check_grid(
        stake_grid,
        tuple(np.linspace(0.0, _BB_STAKE_GRID_DEFAULT_MAX, _BB_STAKE_GRID_DEFAULT_N).tolist()),
        "stake_grid",
    )
    if mixture is None:
        rho = np.full(lam.size, 1.0 / lam.size)
    else:
        rho = np.asarray(list(mixture), dtype=float).reshape(-1)
        if rho.shape != lam.shape or not bool(np.all(np.isfinite(rho))) or bool(np.any(rho < 0.0)):
            raise ValueError("mixture must match stake_grid in shape, be finite and nonnegative")
        total = float(rho.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("mixture weights must sum to a positive value")
        rho = rho / total
    if blocks is None:
        ids = np.zeros(n_items, dtype=np.int64)
    else:
        ids = np.asarray(list(blocks) if not isinstance(blocks, np.ndarray) else blocks)
        ids = ids.astype(np.int64).reshape(-1)
        if ids.size != n_items or ids.size == 0:
            raise ValueError("blocks must assign one nonnegative id per item")
        if bool(np.any(ids < 0)):
            raise ValueError("blocks must assign one nonnegative id per item")
    uniq = np.unique(ids)
    weights = np.array([(ids == u).sum() for u in uniq], dtype=float) / float(n_items)
    w_star = float(weights.max())
    stakes = lam[:, None] * (weights / w_star)[None, :]  # (G, Mb), in [0, 1)
    psi = _psi_e(stakes)  # (G, Mb)
    mu0 = 0.5 * (1.0 + t)
    dirs = (
        list(direction_pairs(n_models))
        if directions is None
        else [(int(a), int(b)) for a, b in directions]
    )
    if len(dirs) == 0:
        raise ValueError("directions must be nonempty")
    for a, b in dirs:
        if not (0 <= a < n_models) or not (0 <= b < n_models) or a == b:
            raise ValueError("directions must be ordered pairs of distinct valid model indices")
    out = np.empty((n_rep, len(dirs)), dtype=float)
    r_idx = np.arange(1, n_rep + 1, dtype=float)[:, None]
    for col, (a, b) in enumerate(dirs):
        x_diff = 0.5 * (p[:, :, a] - p[:, :, b] + 1.0)  # (R, N) in [0, 1]
        y = np.stack([x_diff[:, ids == u].mean(axis=1) for u in uniq], axis=1)  # (R, Mb)
        pred = mu0 + np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)[:-1]], 0)
        pred = pred / r_idx  # Yhat_r = (mu0 + sum_{s<r} Y_s) / r, predictable, in [0, 1]
        lin = (y - mu0) @ stakes.T  # (R, G)
        pen = ((y - pred) ** 2) @ psi.T  # (R, G)
        log_f = lin - pen
        with np.errstate(over="ignore"):
            log_e = np.minimum(np.cumsum(log_f, axis=0), np.log(_E_MAX))
            e_g = np.exp(log_e)  # (R, G) per-stake e-processes
        out[:, col] = np.clip(e_g @ rho, 0.0, _E_MAX)
    return out


@dataclass(frozen=True)
class BBEdgeResult:
    """Output of :func:`bb_edge_certify` (replicates r = 1..R on axis 0).

    ``e_values[r-1, d]`` is the Eq. (7) e-process value of direction
    ``directions[d]`` after r completed replicates; ``edges`` are the DIRECT
    e-Holm certifications of Eq. (8) at each replicate (not nested over time
    by design); ``edges_closed`` is their transitive closure (a path a->...->b
    of length d certifies Delta_{a,b} > d*tau); ``rank_lower``/``rank_upper``
    are the simultaneous rank intervals I_{l,r} = [1+|A_{l,r}|, L-|D_{l,r}|]
    of their Appendix B Eq. (13), valid at every replicate with probability
    >= 1 - alpha (their Corollary 1). ``inconsistent[r-1]`` flags a closure
    cycle (bad event, probability <= alpha) after which reports fail closed.
    """

    e_values: Array
    edges: BoolArray
    edges_closed: BoolArray
    thresholds: Array
    directions: tuple[tuple[int, int], ...]
    rank_lower: IntArray
    rank_upper: IntArray
    inconsistent: BoolArray
    alpha: float
    tau: float
    n_blocks: int

    @property
    def n_models(self) -> int:
        """Number of models L."""
        return int(self.rank_lower.shape[-1])

    @property
    def n_replicates(self) -> int:
        """Number of completed replicates R."""
        return int(self.e_values.shape[0])


def bb_edge_certify(
    panels: Array,
    *,
    blocks: Sequence[int] | IntArray | None = None,
    tau: float = 0.0,
    alpha: float = 0.05,
    stake_grid: Sequence[float] | None = None,
    mixture: Sequence[float] | None = None,
    directions: Sequence[tuple[int, int]] | None = None,
) -> BBEdgeResult:
    """BB-EDGE confidence graph with anytime FWER control (their Theorem 1).

    Runs :func:`bb_edge_e_values` and applies direct e-Holm (their Eq. (8))
    across the direction family after every completed replicate: with
    probability >= 1 - alpha, NO false edge (Delta_{a,b} <= tau) is certified
    at ANY replicate r <= R, under arbitrary within-block and cross-pair
    dependence. Rank intervals follow their Appendix B. Research diagnostic
    on benchmark scores — never market evidence.
    """
    p = _check_panels(panels)
    a = _check_alpha(alpha)
    n_rep, _, n_models = (int(s) for s in p.shape)
    e_vals = bb_edge_e_values(
        p,
        blocks=blocks,
        tau=tau,
        stake_grid=stake_grid,
        mixture=mixture,
        directions=directions,
    )
    dirs = (
        direction_pairs(n_models)
        if directions is None
        else [(int(x), int(y)) for x, y in directions]
    )
    edges = np.zeros((n_rep, n_models, n_models), dtype=bool)
    edges_closed = np.zeros((n_rep, n_models, n_models), dtype=bool)
    thresholds = np.full(n_rep, 1.0 / a, dtype=float)
    lower = np.ones((n_rep, n_models), dtype=np.int64)
    upper = np.full((n_rep, n_models), n_models, dtype=np.int64)
    inconsistent = np.zeros(n_rep, dtype=bool)
    for r in range(n_rep):
        c, rejected = direct_e_holm(e_vals[r], a)
        thresholds[r] = c
        for d, (x, y) in enumerate(dirs):
            edges[r, x, y] = bool(rejected[d])
        closed = _transitive_closure(edges[r])
        if bool(np.any(np.diag(closed))):
            # Cycle: bad event (probability <= alpha). Fail closed from here.
            inconsistent[r:] = True
            break
        edges_closed[r] = closed
        lower[r] = 1 + closed.sum(axis=0).astype(np.int64)  # |A_l|
        upper[r] = n_models - closed.sum(axis=1).astype(np.int64)  # L - |D_l|
    return BBEdgeResult(
        e_values=e_vals,
        edges=edges,
        edges_closed=edges_closed,
        thresholds=thresholds,
        directions=tuple(dirs),
        rank_lower=lower,
        rank_upper=upper,
        inconsistent=inconsistent,
        alpha=a,
        tau=float(tau),
        n_blocks=1 if blocks is None else int(np.unique(np.asarray(blocks)).size),
    )


@dataclass(frozen=True)
class BBEdgeTopKResult:
    """Output of :func:`bb_edge_topk_certify` (their Appendix A, Theorem 2).

    ``candidate`` is the pilot-selected set T_0 (k model indices); the
    confirmatory family is the k(L-k) cross-set directions; ``certified`` /
    ``certified_at`` report whether (and at which 0-based confirmatory
    replicate index) all family hypotheses were rejected by direct e-Holm at
    ONE replicate. On certification, with probability >= 1 - alpha,
    min_{a in T_0} theta_a > max_{b not in T_0} theta_b + tau — even when the
    pilot candidate itself was wrong (an incorrect candidate is certified
    with probability <= alpha).
    """

    candidate: IntArray
    certified: bool
    certified_at: int | None
    e_values: Array
    edges: BoolArray
    directions: tuple[tuple[int, int], ...]
    alpha: float
    tau: float
    k: int


def bb_edge_topk_certify(
    pilot_panels: Array,
    confirm_panels: Array,
    k: int,
    *,
    blocks: Sequence[int] | IntArray | None = None,
    tau: float = 0.0,
    alpha: float = 0.05,
    stake_grid: Sequence[float] | None = None,
    mixture: Sequence[float] | None = None,
) -> BBEdgeTopKResult:
    """Pilot-targeted anytime-valid Top-k certification (their Appendix A).

    ``pilot_panels`` (R_0, N, L) and ``confirm_panels`` (R, N, L) must be
    INDEPENDENT replicates of the same benchmark (their Assumption 1(3)):
    the pilot only selects the candidate set T_0 (top-k pilot means, ties by
    increasing model index) and fixes the tested family; all certification
    evidence comes from the confirmatory replicates, whose e-processes start
    at 1. T_0 is certified once every one of the k(L-k) cross-set directions
    is rejected by direct e-Holm at the same replicate (their Step 3).
    """
    pilot = _check_panels(pilot_panels, name="pilot_panels")
    confirm = _check_panels(confirm_panels, name="confirm_panels")
    a = _check_alpha(alpha)
    t = _check_tau(tau)
    if pilot.shape[1:] != confirm.shape[1:]:
        raise ValueError("pilot and confirmatory panels must share (n_items, n_models)")
    n_models = int(confirm.shape[2])
    kk = int(k)
    if not 1 <= kk < n_models:
        raise ValueError("k must satisfy 1 <= k < L")
    pilot_means = pilot.mean(axis=(0, 1))
    order = np.lexsort((np.arange(n_models), -pilot_means))  # desc mean, ties by index
    candidate = np.sort(order[:kk]).astype(np.int64)
    in_t0 = np.zeros(n_models, dtype=bool)
    in_t0[candidate] = True
    dirs = [
        (a_i, b_i)
        for a_i in range(n_models)
        for b_i in range(n_models)
        if in_t0[a_i] and not in_t0[b_i]
    ]
    e_vals = bb_edge_e_values(
        confirm,
        blocks=blocks,
        tau=t,
        stake_grid=stake_grid,
        mixture=mixture,
        directions=dirs,
    )
    n_rep = int(e_vals.shape[0])
    edges = np.zeros((n_rep, len(dirs)), dtype=bool)
    certified_at: int | None = None
    for r in range(n_rep):
        _, rejected = direct_e_holm(e_vals[r], a)
        edges[r] = rejected
        if certified_at is None and bool(np.all(rejected)):
            certified_at = r
    edges_shaped = np.zeros((n_rep, n_models, n_models), dtype=bool)
    for d, (x, y) in enumerate(dirs):
        edges_shaped[:, x, y] = edges[:, d]
    return BBEdgeTopKResult(
        candidate=candidate,
        certified=certified_at is not None,
        certified_at=certified_at,
        e_values=e_vals,
        edges=edges_shaped,
        directions=tuple(dirs),
        alpha=a,
        tau=t,
        k=kk,
    )


# ---------------------------------------------------------------------------
# Monte-Carlo validation diagnostics (SYNTHETIC correctness checks only)
# ---------------------------------------------------------------------------


def check_rank_time_uniform_coverage(
    rank_lower: Array,
    rank_upper: Array,
    true_ranks: Array,
) -> dict[str, object]:
    """Time-uniform rank-interval coverage over replicates (diagnostic).

    ``rank_lower``/``rank_upper`` are (n_reps, T, M) or (T, M) endpoints;
    ``true_ranks`` is (M,) or (n_reps, M) with entries in [1, M]. A replicate
    counts as covered only if EVERY model's true rank lies in its interval at
    EVERY time (the time-uniform event of Khosravi & Huo Thm 4.3 / Gao et al.
    Cor 1). Keys: ``time_uniform_coverage``, ``violation_rate``, ``n_reps``,
    ``per_time_coverage`` (T,) — fraction of replicates covering all models
    at that time — and ``first_violations`` (n_reps,) 0-based first violation
    times, -1 where never violated.
    """
    lo = np.asarray(rank_lower, dtype=float)
    up = np.asarray(rank_upper, dtype=float)
    tr = np.asarray(true_ranks, dtype=float)
    if lo.shape != up.shape or lo.ndim not in (2, 3) or lo.size == 0:
        raise ValueError("endpoints must be matching nonempty (T, M) or (n_reps, T, M)")
    lo3 = lo if lo.ndim == 3 else lo[None, ...]
    up3 = up if up.ndim == 3 else up[None, ...]
    n_reps, n_t, m = (int(s) for s in lo3.shape)
    if tr.shape == (m,):
        tr3 = np.broadcast_to(tr[None, None, :], lo3.shape)
    elif tr.shape == (n_reps, m):
        tr3 = np.broadcast_to(tr[:, None, :], lo3.shape)
    else:
        raise ValueError("true_ranks must be (M,) or (n_reps, M)")
    if bool(np.any(np.isnan(tr3))) or bool(np.any(tr3 < 1)) or bool(np.any(tr3 > m)):
        raise ValueError("true_ranks must be finite and lie in [1, M]")
    covered = (lo3 <= tr3) & (tr3 <= up3)  # (R, T, M)
    all_models = covered.all(axis=-1)  # (R, T)
    ever_missed = ~all_models.all(axis=-1)
    first = np.where(ever_missed, np.argmax(~all_models, axis=-1), -1).astype(np.int64)
    return {
        "time_uniform_coverage": float(1.0 - ever_missed.mean()),
        "violation_rate": float(ever_missed.mean()),
        "n_reps": n_reps,
        "n_times": n_t,
        "per_time_coverage": np.asarray(all_models.mean(axis=0), dtype=np.float64),
        "first_violations": first,
    }


def check_rank_set_coverage(rank_sets: Array, true_ranks: Array) -> dict[str, object]:
    """Time-uniform coverage of EXACT rank sets (Khosravi & Huo Thm 4.3(a)).

    ``rank_sets`` is (n_reps, T, M, M) or (T, M, M) boolean membership with
    the last axis indexed by rank-1; ``true_ranks`` is (M,) or (n_reps, M).
    Same keys as :func:`check_rank_time_uniform_coverage`.
    """
    rs = np.asarray(rank_sets, dtype=bool)
    tr = np.asarray(true_ranks, dtype=np.int64)
    if rs.ndim not in (3, 4) or rs.size == 0 or rs.shape[-1] != rs.shape[-2]:
        raise ValueError("rank_sets must be (T, M, M) or (n_reps, T, M, M) membership")
    rs4 = rs if rs.ndim == 4 else rs[None, ...]
    n_reps, n_t, m = int(rs4.shape[0]), int(rs4.shape[1]), int(rs4.shape[2])
    if tr.shape == (m,):
        tr2 = np.broadcast_to(tr[None, :], (n_reps, m))
    elif tr.shape == (n_reps, m):
        tr2 = tr
    else:
        raise ValueError("true_ranks must be (M,) or (n_reps, M)")
    if bool(np.any(tr2 < 1)) or bool(np.any(tr2 > m)):
        raise ValueError("true_ranks must lie in [1, M]")
    target = np.broadcast_to((tr2 - 1)[:, None, :], (n_reps, n_t, m))[..., None]
    picked = np.take_along_axis(rs4, target, axis=3)[..., 0]  # (R, T, M)
    per_rep_time = picked.all(axis=-1)  # (R, T)
    ever_missed = ~per_rep_time.all(axis=-1)
    first = np.where(ever_missed, np.argmax(~per_rep_time, axis=-1), -1).astype(np.int64)
    return {
        "time_uniform_coverage": float(1.0 - ever_missed.mean()),
        "violation_rate": float(ever_missed.mean()),
        "n_reps": n_reps,
        "n_times": n_t,
        "per_time_coverage": np.asarray(per_rep_time.mean(axis=0), dtype=np.float64),
        "first_violations": first,
    }


def check_dominance_fwer(
    dominance: Array,
    theta: Array,
) -> dict[str, object]:
    """Ever-false-dominance event for one run (Khosravi & Huo Thm 4.2, Eq. 8).

    ``dominance`` is the (T, M, M) certified-dominance path; ``theta`` the
    (M,) true abilities. A pair (j, l) is FALSE iff dominance[t, j, l] and
    theta_j <= theta_l (ties count as false: nothing is strictly better).
    Keys: ``ever_false`` (bool), ``first_false_time`` (int, -1 if never),
    ``n_false_pairs`` (count over all times).
    """
    dom = np.asarray(dominance, dtype=bool)
    th = np.asarray(theta, dtype=float).reshape(-1)
    if dom.ndim != 3 or dom.shape[1] != dom.shape[2] or dom.shape[1] != th.size:
        raise ValueError("dominance must be (T, M, M) matching theta (M,)")
    false_pair = (th[:, None] <= th[None, :]) & ~np.eye(th.size, dtype=bool)
    false_hits = dom & false_pair[None, :, :]
    any_hit = false_hits.reshape(int(dom.shape[0]), -1).any(axis=1)
    return {
        "ever_false": bool(any_hit.any()),
        "first_false_time": int(np.argmax(any_hit)) if bool(any_hit.any()) else -1,
        "n_false_pairs": int(false_hits.sum()),
    }


def check_edge_fwer(
    edges: Array,
    delta_true: Array,
    tau: float = 0.0,
) -> dict[str, object]:
    """Ever-false-edge event for one BB-EDGE run (Gao et al. Eq. (1)).

    ``edges`` is the (R, L, L) per-replicate certification array (direct or
    closed); ``delta_true[a, b]`` = theta_a - theta_b. Edge (a->b) at
    replicate r is FALSE iff certified and delta_true[a, b] <= tau. Keys:
    ``ever_false`` (bool), ``first_false_replicate`` (int, -1 if never),
    ``n_false_edges`` (count over all replicates).
    """
    ed = np.asarray(edges, dtype=bool)
    dl = np.asarray(delta_true, dtype=float)
    t = _check_tau(tau)
    if ed.ndim != 3 or ed.shape[1] != ed.shape[2] or dl.shape != ed.shape[1:]:
        raise ValueError("edges must be (R, L, L) and delta_true (L, L)")
    false_dir = (dl <= t) & ~np.eye(dl.shape[0], dtype=bool)
    hits = ed & false_dir[None, :, :]
    any_hit = hits.reshape(int(ed.shape[0]), -1).any(axis=1)
    return {
        "ever_false": bool(any_hit.any()),
        "first_false_replicate": int(np.argmax(any_hit)) if bool(any_hit.any()) else -1,
        "n_false_edges": int(hits.sum()),
    }
