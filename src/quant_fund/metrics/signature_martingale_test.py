"""Expected-signature martingale hypothesis test (signature e-statistic).

Tests whether a sampled price / log-price / discounted-price path is
consistent with the martingale property — the weak-form EMH statement that
increments are martingale differences, ``E[dx_t | F_{t-1}] = 0``. The test
statistic is built from *signature moments* of the time-augmented path:
features of the truncated signature whose expectation is zero under the
null. Implements the expected-signature testing layer of:

    Chevyrev, I. & Oberhauser, H. (2018/2022). "Signature moments to
    characterize laws of stochastic processes." Ann. Appl. Probab. 32(1),
    213–258. arXiv:1810.10971 — verified against the arXiv abstract page
    (authors/title match, fetched 2026-09-30). Signature moments
    characterize the law of the process; their hypothesis test compares an
    empirical mean signature against the null-implied value with a
    Hotelling-type statistic and resampling calibration, the construction
    specialized here to the martingale null.

Supporting / applied references (verified 2026-09-30 unless noted):

- Andrès, H., Boumezoued, A. & Jourdain, B. (2022). "Signature-based
  validation of real-world economic scenarios." arXiv:2208.07251 —
  signature-based validation of stochastic-process path ensembles; the
  scenario-validation usage where the martingale property of discounted
  paths is exactly the arbitrage-free check.
- Fermanian, A. (2019/2021). "Embedding and learning with signatures."
  Comput. Stat. 36(4), 2425–2470. arXiv:1911.13211 — expected signatures
  as statistical features of path distributions.
- Vovk, V. & Wang, R. (2019/2021). "E-values: Calibration, combination,
  and applications." Ann. Statist. 49(3), 1736–1754. arXiv:1912.06116 —
  the e-variable formalism (two-sided Gaussian factor ``cosh(λs)e^{-λ²/2}``,
  product martingales, Ville-type decisions at ``1/alpha``).
- Domínguez, M. A. & Lobato, I. N. (2003). "Testing the martingale
  difference hypothesis." Econometric Reviews 22(4), 351–377 — journal-only;
  the classical (non-signature) martingale-difference test family this lane
  extends to path features.
- Kuan, C.-M. & Lee, W.-M. (2004). "Testing for martingale difference
  hypothesis." Econometric Reviews 23(4), 355–376 — journal-only;
  robust martingale-difference testing under heteroskedasticity.

Citation deviation note: the wave brief suggested arXiv:2206.14174 as a
candidate "expected signature martingale" reference — that id resolves to
Martin, Salehi & Yamashita, "The (4,p)-arithmetic hyperbolic lattices,
p >= 2, in three dimensions" (unrelated number theory), and the suggested
"Katsirelos & Wiese" attribution does not exist in the signature
literature. No dedicated arXiv preprint on an expected-signature
martingale test was found; the module therefore cites the verified
expected-signature theory (arXiv:1810.10971), its scenario-validation
application (arXiv:2208.07251), the e-value calculus (arXiv:1912.06116),
and the classical martingale-difference tests as journal-only references.

Machinery
---------
1. ``time_augmented_path`` / ``window_signature_features`` — split the
   increment stream into ``n_windows`` non-overlapping windows of ``L``
   increments (incomplete tail dropped, counted). On each window the
   time-augmented path ``(u_j, X_j)``, ``u_j = j / L``, is fed through the
   truncated signature of ``models.path_signatures.signature`` (composed,
   not reimplemented). Three signature-moment features per window, each
   ``E[· | F_a] = 0`` under the martingale null and standardized to
   unit null variance by the within-window increment scale
   ``σ̂_w = sqrt(mean(dx_w²))``:

   - ``z1 = ΔX_w / (σ̂_w √L)`` — level-1 signature moment ``Sig^1_X``;
     drift-sensitive, ordering-invariant.
   - ``z2 = Sig^2_{u,X} / (σ̂_w sd_2)``, ``sd_2 = sqrt((4L² − 1) / (12L))``
     — the symmetric level-2 time-price moment ``∫(v − a) dX_v``, a
     martingale stochastic integral under H0; late-weighted increment,
     drift- AND ordering-sensitive.
   - ``z3 = Σ dx_j dx_{j+1} / (σ̂_w² √(L − 1))`` — lag-1 increment product
     (ordered signature content: cross-products of successive increments);
     mean-reversion/momentum-sensitive, ordering-sensitive.

   Shuffle-product note: on the normalized clock,
   ``Sig^2_{u,X} + Sig^2_{X,u} = Sig^1_u · Sig^1_X = ΔX_w``, so the
   antisymmetric Lévy-area term ``A = Sig^2_{X,u} − Sig^2_{u,X} =
   ΔX_w − 2·Sig^2_{u,X}`` is a linear function of (z1, z2) — it is
   documented, not duplicated in the feature vector (keeps the Hotelling
   covariance non-singular). The signed area is drift-invariant by
   construction (a straight-line trend adds zero signed area), which is
   why pure drift detection rides on the symmetric moments z1/z2 while
   curvature/serial structure rides on z3 and the ordering statistic.
2. ``expected_signature_test`` — the Chevyrev–Oberhauser-flavored
   expected-signature test: Hotelling statistic
   ``T = n_w · z̄ᵀ (Σ̂ + εI)⁻¹ z̄`` over the window feature vectors, with a
   chi-square reference ``p_chi2`` (df = 3) and a Rademacher wild-bootstrap
   ``p_boot`` (sign-flip the window feature vectors; destroys the mean
   under H0 — exact-ish under symmetric martingale differences,
   approximate under asymmetric increments — documented honesty caveat).
3. ``ordering_permutation_test`` — signed ordering statistic
   ``S = Σ_w (z2_w + z3_w)`` calibrated by within-window increment
   permutation (destroys serial order, preserves each window's increment
   multiset → exact under exchangeable increments). The two-sided
   p-value measures departure from the permutation centre:
   ``p = (1 + #{|S_b − m̄| ≥ |S_obs − m̄|})/(1 + B)`` with ``m̄`` the mean
   of the permutation draws — valid because under H0 ``S_obs`` is
   exchangeable with every ``S_b``. Blind to pure level drift by
   construction (a uniform mean survives permutation inside the centre
   term) — the drift arm is the bootstrap/e-process; this arm carries
   power vs mean-reversion, momentum and curvature alternatives.
4. ``martingale_e_process`` — anytime-valid e-process over window
   boundaries: per-window factor on the unit-variance composite margin
   ``s_w = (z1 + z2 + z3) / sqrt(3 + 2ρ)`` with the deterministic null
   correlation ``ρ = L√3 / √(4L² − 1)``. Two factor modes:
   ``gaussian_lr`` (default) ``e_w = cosh(λ s_w) e^{−λ²/2}`` — the
   Vovk–Wang two-sided Gaussian e-variable, exactly unit-mean when
   ``s_w ~ N(0, 1)`` (the iid-Gaussian martingale null; approximate
   under heavier-than-Gaussian or skewed nulls — documented);
   ``bounded`` ``e_w = 1 + λ tanh(s_w)`` — exactly unit-mean whenever
   ``s_w`` is symmetric about zero under the null. Ville decision via
   ``metrics.evalues.e_process_threshold`` (composed, not reimplemented):
   reject when the running product reaches ``1/alpha``.
5. ``signature_martingale_test`` — orchestrator returning the full blob:
   features, Hotelling ``T``/``p_chi2``/``p_boot``, ordering ``S``/
   ``p_perm``, the e-path/``e_final``/Ville fields, the Bonferroni omnibus
   ``p_omnibus = min(1, 2·min(p_boot, p_perm))``, and rejection flags.
6. ``simulate_martingale_paths`` / ``bench_signature_martingale_test`` —
   seeded SYNTHETIC generators and bench (AGENTS.md honesty contract #2):
   iid-Gaussian and GARCH martingale-difference nulls (size), constant-
   drift, OU mean-reversion and AR(1) momentum alternatives (power),
   e-value trajectories. Correctness evidence only, never market evidence.

Honesty: outputs are proper inference diagnostics only — p-values,
e-values, rejection flags, signature-moment statistics. No
Sharpe/Sortino/Calmar/P&L/NAV content. ``live_pnl_claim`` is False in every
blob; bench keys are prefixed ``synthetic_``. Fail-closed throughout:
non-1-D, non-finite, too-short or degenerate (zero-variance window)
inputs raise ``ValueError`` rather than silently returning.

Composition: ``models.path_signatures.signature`` is reused for the
truncated signature of each windowed path — pulled through a
function-level (lazy) import so the metrics→models layer-order arch guard
stays clean (``configs/arch_boundaries.toml`` forbids new baseline
entries; module-scope metrics→models edges are violations).
``metrics.evalues.e_process_threshold`` is reused for the Ville decision.
``metrics.anytime_fdr`` and ``metrics.conformal_martingale`` are
deliberately NOT reused: the former is multiple-testing over e-value
streams, the latter tests exchangeability of conformal scores — different
objects from the martingale-property null here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from quant_fund.metrics.evalues import e_process_threshold
from quant_fund.utils.series import as_named_1d

Array = NDArray[np.float64]

SCHEMA = "signature_martingale_test.v1"
KIND = "signature_martingale_test"
FEATURE_NAMES: tuple[str, ...] = (
    "increment",
    "time_weighted_increment",
    "lag1_increment_product",
)

MIN_WINDOW = 4
MIN_WINDOWS = 4
DEFAULT_WINDOW = 16
DEFAULT_N_RESAMPLES = 499
DEFAULT_LAM = 0.5
DEFAULT_ALPHA = 0.05
E_MODES: tuple[str, str] = ("gaussian_lr", "bounded")
_COV_RIDGE = 1e-9
_E_MAX = 1e300

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_LAM",
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_WINDOW",
    "E_MODES",
    "FEATURE_NAMES",
    "KIND",
    "MIN_WINDOW",
    "MIN_WINDOWS",
    "SCHEMA",
    "bench_signature_martingale_test",
    "expected_signature_test",
    "martingale_e_process",
    "ordering_permutation_test",
    "signature_martingale_test",
    "simulate_martingale_paths",
    "time_augmented_path",
    "window_signature_features",
]


# ---------------------------------------------------------------------------
# Validation helpers (fail-closed — AGENTS.md contract #4)
# ---------------------------------------------------------------------------


def _as_series(x: Array | Iterable[float], name: str) -> Array:
    v = as_named_1d(name, np.asarray(x, dtype=float))
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return v


def _check_window(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError("window must be an integer")
    w = int(window)
    if w < MIN_WINDOW:
        raise ValueError(f"window must be >= {MIN_WINDOW} (lag-1 and ordering need it)")
    return w


def _check_resamples(n: int, name: str) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(n) < 19:
        raise ValueError(f"{name} must be >= 19 for a meaningful resampling grid")
    return int(n)


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not math.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_lam(lam: float) -> float:
    x = float(lam)
    if not math.isfinite(x) or not 0.0 < x < 1.0:
        raise ValueError("lam must be in (0, 1)")
    return x


def _check_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed)


def _resolve_rng(rng: np.random.Generator | int | None, seed: int) -> np.random.Generator:
    """Seeded Generator — the only source of randomness (determinism contract)."""
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy Generator or None")
        return rng
    return np.random.default_rng(_check_seed(seed))


def _n_windows(n_increments: int, window: int) -> int:
    """Complete non-overlapping windows of ``window`` increments (tail dropped)."""
    return n_increments // window


def _require_windows(n_increments: int, window: int, minimum: int) -> int:
    n_w = _n_windows(n_increments, window)
    if n_w < minimum:
        raise ValueError(
            f"series too short: {n_increments} increments give {n_w} complete "
            f"windows of {window}; need >= {minimum}"
        )
    return n_w


# ---------------------------------------------------------------------------
# 1. Windowed signature-moment features
# ---------------------------------------------------------------------------


def time_augmented_path(values: Array | Iterable[float]) -> Array:
    """Append a normalized-clock channel: ``(j/T, X_j)`` for j = 0..T.

    Time augmentation makes the signature sensitive to the sampling clock —
    required for martingale testing, whose null is about *when* increments
    arrive, not just their multiset (Salvi et al. 2021 framing; plain
    signatures are reparametrization-invariant and clock-blind).
    """
    x = _as_series(values, "values")
    t = np.linspace(0.0, 1.0, x.shape[0])
    out = np.empty((x.shape[0], 2), dtype=np.float64)
    out[:, 0] = t
    out[:, 1] = x
    return out


def _window_level2_ux(path_2d: Array) -> float:
    """Sig^2_{u,X} of a normalized-clock windowed path via the composed solver.

    The metrics→models upward edge is taken lazily (function-level) so the
    layer-order arch guard stays clean — module-scope it would be a
    violation per configs/arch_boundaries.toml.
    """
    from quant_fund.models.path_signatures import signature

    sig = signature(path_2d, order=2)
    # flat word order, d=2: [Sig^1_u, Sig^1_X, Sig^2_uu, Sig^2_uX, Sig^2_Xu, Sig^2_XX]
    return float(sig[3])


def _time_weight_scale(window: int) -> float:
    """Null sd of Sig^2_{u,X}: sqrt(Var(Σ mid_j dx_j)) / σ = sqrt((4L²−1)/(12L))."""
    return float(math.sqrt((4.0 * window * window - 1.0) / (12.0 * window)))


def _increment_time_corr(window: int) -> float:
    """Deterministic null correlation between z1 and z2: L√3 / √(4L²−1)."""
    return float(window * math.sqrt(3.0) / math.sqrt(4.0 * window * window - 1.0))


def _window_z(dx_w: Array, sig2_ux: float, window: int) -> Array:
    """Feature vector (z1, z2, z3) of one window, fail-closed on zero scale."""
    sigma_hat_sq = float(np.mean(dx_w * dx_w))
    if not math.isfinite(sigma_hat_sq) or sigma_hat_sq <= 0.0:
        raise ValueError("degenerate window: zero increment variance (σ̂_w = 0)")
    sigma_hat = math.sqrt(sigma_hat_sq)
    z1 = float(np.sum(dx_w)) / (sigma_hat * math.sqrt(window))
    z2 = sig2_ux / (sigma_hat * _time_weight_scale(window))
    lag1 = float(np.sum(dx_w[:-1] * dx_w[1:]))
    z3 = lag1 / (sigma_hat_sq * math.sqrt(window - 1.0))
    return np.asarray([z1, z2, z3], dtype=float)


def window_signature_features(
    x: Array | Iterable[float],
    window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Signature-moment feature matrix of a path, one row per window.

    Returns the standardized feature matrix ``z`` of shape
    ``(n_windows, 3)`` with columns ``FEATURE_NAMES``
    ``(increment, time_weighted_increment, lag1_increment_product)``, plus
    the per-window increment scale ``sigma_hat``, raw ``increments``, and
    window bookkeeping. Under the martingale null every column has
    conditional mean zero and unit marginal variance (up to the σ̂_w
    plug-in); that is the expected-signature statement being tested.
    Requires at least ``MIN_WINDOWS`` complete windows.
    """
    v = _as_series(x, "x")
    w = _check_window(window)
    dx = np.diff(v)
    n_w = _require_windows(dx.shape[0], w, MIN_WINDOWS)
    dropped = dx.shape[0] - n_w * w
    z = np.empty((n_w, 3), dtype=float)
    sigma_hat = np.empty(n_w, dtype=float)
    for i in range(n_w):
        seg = v[i * w : (i + 1) * w + 1]
        sig2_ux = _window_level2_ux(time_augmented_path(seg))
        z[i] = _window_z(dx[i * w : (i + 1) * w], sig2_ux, w)
        sigma_hat[i] = math.sqrt(float(np.mean(dx[i * w : (i + 1) * w] ** 2)))
    return {
        "z": z,
        "sigma_hat": sigma_hat,
        "increments": dx,
        "n_windows": n_w,
        "window": w,
        "dropped_tail_increments": dropped,
        "feature_names": FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# 2. Expected-signature Hotelling test + Rademacher wild bootstrap
# ---------------------------------------------------------------------------


def _feature_covariance(z: Array) -> Array:
    """Window-feature sample covariance with a deterministic ridge floor."""
    n_w = int(z.shape[0])
    cov_raw = np.cov(z, rowvar=False, ddof=1) if n_w > 1 else np.eye(z.shape[1])
    cov: Array = np.atleast_2d(np.asarray(cov_raw, dtype=np.float64))
    if not bool(np.all(np.isfinite(cov))):
        raise ValueError("non-finite feature covariance")
    ridge = _COV_RIDGE * max(1.0, float(np.trace(cov)) / cov.shape[0])
    out: Array = cov + ridge * np.eye(cov.shape[0])
    return out


def _hotelling(z: Array) -> tuple[float, Array, Array]:
    """T = n_w · z̄ᵀ Σ̂⁻¹ z̄ over window feature vectors (expected-signature)."""
    z_bar = np.mean(z, axis=0)
    cov = _feature_covariance(z)
    t_stat = float(z.shape[0] * z_bar @ np.linalg.solve(cov, z_bar))
    return t_stat, z_bar, cov


def expected_signature_test(
    x: Array | Iterable[float],
    window: int = DEFAULT_WINDOW,
    *,
    n_boot: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Expected-signature test: do the window signature moments vanish?

    Under the martingale null ``E[z_w] = 0``; the Hotelling statistic
    ``T = n_w · z̄ᵀ Σ̂⁻¹ z̄`` measures the departure. ``p_chi2`` is the
    asymptotic χ²₃ reference; ``p_boot`` is the Rademacher wild-bootstrap
    calibration (window feature vectors sign-flipped — a valid resample of
    the null under symmetric martingale differences, the multiplier-
    bootstrap default; approximate under skewed increment laws — honest
    caveat, never silently exact).
    """
    features = window_signature_features(x, window)
    z = np.asarray(features["z"], dtype=float)
    n_b = _check_resamples(n_boot, "n_boot")
    gen = _resolve_rng(rng, seed)
    t_obs, z_bar, cov = _hotelling(z)
    cov_inv = np.linalg.inv(cov)
    signs = gen.choice(np.array([-1.0, 1.0]), size=(n_b, z.shape[0]))
    z_bs = np.einsum("bw,wc->bc", signs, z) / z.shape[0]
    t_boot = np.asarray(z.shape[0] * np.einsum("bi,ij,bj->b", z_bs, cov_inv, z_bs), dtype=float)
    p_boot = float((1.0 + float(np.count_nonzero(t_boot >= t_obs))) / (n_b + 1.0))
    return {
        "t_stat": t_obs,
        "df": float(z.shape[1]),
        "p_chi2": float(chi2.sf(t_obs, df=z.shape[1])),
        "p_boot": p_boot,
        "z_bar": z_bar,
        "feature_cov": cov,
        "n_windows": int(z.shape[0]),
        "n_boot": n_b,
        "calibration": "rademacher_wild_bootstrap",
    }


# ---------------------------------------------------------------------------
# 3. Within-window permutation test of the ordering statistic
# ---------------------------------------------------------------------------


def _ordering_stat_closed(dx_w: Array, window: int) -> tuple[float, float]:
    """Closed-form (z2, z3) of one window's increments — permutation fast path.

    Sig^2_{u,X} on the normalized clock is exactly ``Σ mid_j dx_j`` with
    ``mid_j = (j − 1/2)/L`` for the piecewise-linear path — identical to
    feeding the permuted path through the composed signature solver (unit
    test pins the identity); the closed form keeps resampling O(L) per draw.
    """
    sigma_hat_sq = float(np.mean(dx_w * dx_w))
    if sigma_hat_sq <= 0.0:
        raise ValueError("degenerate window: zero increment variance (σ̂_w = 0)")
    mids = (np.arange(window, dtype=float) + 0.5) / window
    sig2_ux = float(np.sum(mids * dx_w))
    sigma_hat = math.sqrt(sigma_hat_sq)
    z2 = sig2_ux / (sigma_hat * _time_weight_scale(window))
    lag1 = float(np.sum(dx_w[:-1] * dx_w[1:]))
    z3 = lag1 / (sigma_hat_sq * math.sqrt(window - 1.0))
    return z2, z3


def _ordering_stat(z: Array) -> float:
    """S = Σ_w (z2_w + z3_w) — signed ordering-sensitive evidence.

    Signed (not squared): mean reversion pushes z3 negative, momentum
    positive — the departure direction is the evidence. Departure from
    the permutation centre is assessed two-sided.
    """
    return float(np.sum(z[:, 1] + z[:, 2]))


def ordering_permutation_test(
    x: Array | Iterable[float],
    window: int = DEFAULT_WINDOW,
    *,
    n_perm: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Within-window increment-permutation test of ordering structure.

    ``S = Σ_w (z2_w + z3_w)`` is recalibrated under independent uniform
    permutations of the increments *inside each window* — the increments'
    per-window multiset is preserved, their order destroyed. Exact under
    exchangeable (iid) increments (S_obs is then exchangeable with every
    permuted draw) and valid for detecting serial structure (mean
    reversion, momentum, curvature); structurally blind to pure level
    drift — a uniform mean is permutation-invariant, so the drift arm is
    ``expected_signature_test`` / ``martingale_e_process``, not this test.
    The p-value is two-sided around the permutation centre.
    """
    v = _as_series(x, "x")
    w = _check_window(window)
    dx = np.diff(v)
    n_w = _require_windows(dx.shape[0], w, MIN_WINDOWS)
    n_p = _check_resamples(n_perm, "n_perm")
    gen = _resolve_rng(rng, seed)
    windows = dx[: n_w * w].reshape(n_w, w)
    z = np.asarray(window_signature_features(v, w)["z"], dtype=float)
    s_obs = _ordering_stat(z)
    # Vectorized permutation draws: row-wise independent permutations of the
    # (n_windows, window) increment block; closed-form z2/z3 (identical to the
    # signature-solver values — pinned by unit test).
    mids = (np.arange(w, dtype=float) + 0.5) / w
    sig_sq = np.mean(windows * windows, axis=1)
    if bool(np.any(sig_sq <= 0.0)):
        raise ValueError("degenerate window: zero increment variance (σ̂_w = 0)")
    sd2 = _time_weight_scale(w)
    lag_denom = np.sqrt(w - 1.0)
    s_perm = np.empty(n_p, dtype=float)
    for b in range(n_p):
        perm = gen.permuted(windows, axis=1)
        z2 = (perm @ mids) / (np.sqrt(sig_sq) * sd2)
        z3 = np.sum(perm[:, :-1] * perm[:, 1:], axis=1) / (sig_sq * lag_denom)
        s_perm[b] = float(np.sum(z2 + z3))
    centre = float(np.mean(s_perm))
    dev_obs = abs(s_obs - centre)
    p_perm = float(
        (1.0 + float(np.count_nonzero(np.abs(s_perm - centre) >= dev_obs))) / (n_p + 1.0)
    )
    return {
        "s_stat": s_obs,
        "p_perm": p_perm,
        "s_perm_center": centre,
        "s_perm_median": float(np.median(s_perm)),
        "n_windows": n_w,
        "n_perm": n_p,
        "calibration": "within_window_increment_permutation",
    }


# ---------------------------------------------------------------------------
# 4. Anytime-valid martingale e-process over window boundaries
# ---------------------------------------------------------------------------


def _composite_margin(z: Array, window: int) -> Array:
    """Unit-null-variance composite margin s_w = θᵀz_w / √(θᵀΣ₀θ).

    Σ₀ is the deterministic martingale-null feature covariance
    ``[[1, ρ, 0], [ρ, 1, 0], [0, 0, 1]]`` with
    ``ρ = L√3 / √(4L² − 1)`` — a function of the window length only, so the
    normalization is predictable (fixed before any data is seen), not a
    plug-in over the observed windows.
    """
    rho = _increment_time_corr(window)
    var = 3.0 + 2.0 * rho
    return (z[:, 0] + z[:, 1] + z[:, 2]) / math.sqrt(var)


def martingale_e_process(
    x: Array | Iterable[float],
    window: int = DEFAULT_WINDOW,
    *,
    lam: float = DEFAULT_LAM,
    alpha: float = DEFAULT_ALPHA,
    e_mode: str = "gaussian_lr",
) -> dict[str, Any]:
    """Anytime-valid e-process betting against the martingale null.

    Per-window factor on the composite margin ``s_w``:

    - ``gaussian_lr`` (default): ``e_w = cosh(λ s_w) e^{−λ²/2}``, the
      Vovk–Wang two-sided Gaussian e-variable — exactly unit-mean under
      ``s_w ~ N(0, 1)`` (the iid-Gaussian martingale null, so the size
      simulation is exact); under symmetric sub-Gaussian nulls it stays a
      valid supermartingale, and under skewed or heavier-than-Gaussian
      nulls it is an *approximate* e-value — an honest caveat, never a
      hidden exactness claim.
    - ``bounded``: ``e_w = 1 + λ tanh(s_w)`` — exactly unit-mean under any
      null where ``s_w`` is symmetric about zero (tanh is odd), at the
      price of slower growth.

    The running product is the evidence trajectory; rejection is the
    Ville-type decision ``E ≥ 1/alpha`` delegated to
    ``metrics.evalues.e_process_threshold`` (composed, not reimplemented).
    """
    a = _check_alpha(alpha)
    lam_v = _check_lam(lam)
    if e_mode not in E_MODES:
        raise ValueError(f"e_mode must be one of {E_MODES}")
    features = window_signature_features(x, window)
    z = np.asarray(features["z"], dtype=float)
    s = _composite_margin(z, int(features["window"]))
    if e_mode == "gaussian_lr":
        factors = np.cosh(lam_v * s) * math.exp(-0.5 * lam_v * lam_v)
    else:
        factors = 1.0 + lam_v * np.tanh(s)
    factors = np.asarray(np.clip(factors, 1e-300, _E_MAX), dtype=float)
    log_run = np.minimum(np.cumsum(np.log(factors)), np.log(_E_MAX))
    e_path = np.asarray(np.exp(log_run), dtype=float)
    decision = e_process_threshold(e_path, level=a)
    return {
        "e": e_path,
        "e_final": float(e_path[-1]),
        "s_margin": s,
        "lam": lam_v,
        "e_mode": e_mode,
        "first_cross": decision["first_cross"],
        "reject": bool(decision["reject"]),
        "threshold": float(decision["threshold"]),  # type: ignore[arg-type]
        "alpha": a,
        "n_windows": int(features["n_windows"]),
    }


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------


def signature_martingale_test(
    x: Array | Iterable[float],
    window: int = DEFAULT_WINDOW,
    *,
    n_boot: int = DEFAULT_N_RESAMPLES,
    n_perm: int = DEFAULT_N_RESAMPLES,
    lam: float = DEFAULT_LAM,
    alpha: float = DEFAULT_ALPHA,
    e_mode: str = "gaussian_lr",
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Full signature-martingale battery on one price-like series.

    Runs all three arms — the expected-signature Hotelling test
    (``p_chi2`` asymptotic, ``p_boot`` wild-bootstrap), the ordering
    permutation test (``p_perm``), and the anytime-valid e-process
    (``e_final``, Ville ``reject`` at ``alpha``) — and reports the
    Bonferroni omnibus ``p_omnibus = min(1, 2·min(p_boot, p_perm))``.
    Returns the research blob with schema, honesty flags, config echo and
    per-arm fields; ``live_pnl_claim`` is False.
    """
    v = _as_series(x, "x")
    w = _check_window(window)
    a = _check_alpha(alpha)
    _check_lam(lam)
    if e_mode not in E_MODES:
        raise ValueError(f"e_mode must be one of {E_MODES}")
    gen = _resolve_rng(rng, seed)
    # one Generator split deterministically into independent child streams
    child_seeds = gen.integers(0, np.iinfo(np.int64).max, size=3)
    esp = expected_signature_test(v, w, n_boot=n_boot, rng=np.random.default_rng(child_seeds[0]))
    order_test = ordering_permutation_test(
        v, w, n_perm=n_perm, rng=np.random.default_rng(child_seeds[1])
    )
    eproc = martingale_e_process(v, w, lam=lam, alpha=a, e_mode=e_mode)
    p_omnibus = float(min(1.0, 2.0 * min(float(esp["p_boot"]), float(order_test["p_perm"]))))
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "claim": "research_only",
        "live_pnl_claim": False,
        "citation": (
            "Chevyrev & Oberhauser 2022 arXiv:1810.10971; "
            "Andrès, Boumezoued & Jourdain 2022 arXiv:2208.07251; "
            "Vovk & Wang 2021 arXiv:1912.06116"
        ),
        "n": int(v.shape[0]),
        "window": w,
        "alpha": a,
        "e_mode": e_mode,
        "feature_names": FEATURE_NAMES,
        "n_windows": int(esp["n_windows"]),
        "t_stat": float(esp["t_stat"]),
        "p_chi2": float(esp["p_chi2"]),
        "p_boot": float(esp["p_boot"]),
        "s_stat": float(order_test["s_stat"]),
        "p_perm": float(order_test["p_perm"]),
        "p_omnibus": p_omnibus,
        "e_final": float(eproc["e_final"]),
        "e_first_cross": eproc["first_cross"],
        "e_reject": bool(eproc["reject"]),
        "e_threshold": float(eproc["threshold"]),
        "reject_omnibus": bool(p_omnibus <= a),
        "reject_boot": bool(float(esp["p_boot"]) <= a),
        "reject_perm": bool(float(order_test["p_perm"]) <= a),
        "reject_chi2": bool(float(esp["p_chi2"]) <= a),
    }


# ---------------------------------------------------------------------------
# 6. SYNTHETIC generators + bench (correctness evidence only)
# ---------------------------------------------------------------------------

_SIM_KINDS: tuple[str, ...] = ("brownian", "garch", "drift", "ou", "ar1")


def simulate_martingale_paths(
    kind: str,
    n: int,
    seed: int,
    *,
    drift: float = 0.0,
    theta: float = 0.1,
    phi: float = 0.3,
    sigma: float = 1.0,
    alpha_garch: float = 0.10,
    beta_garch: float = 0.85,
) -> dict[str, Any]:
    """Seeded SYNTHETIC path generator for the martingale-test controls.

    Kinds (increments model):

    - ``brownian`` — iid Gaussian increments: the martingale null (size).
    - ``garch`` — GARCH(1,1) martingale differences: heteroskedastic null
      (size robustness — increments still unpredictable, variance
      clustered).
    - ``drift`` — Gaussian increments plus constant per-step ``drift``:
      the trend alternative (power via bootstrap/e-process arm).
    - ``ou`` — Ornstein–Uhlenbeck ``dx = −θ x dt + σ dB`` at the path
      level: the mean-reversion alternative (power via ordering arm).
    - ``ar1`` — AR(1) increments ``dx_t = φ dx_{t−1} + ε`` with ``φ > 0``:
      the momentum alternative (power via ordering arm).

    Returns ``increments``, ``path`` (cumulative level with a zero start),
    ``kind`` and the config echo. All draws flow through one seeded
    Generator — repeated calls are bit-identical.
    """
    if kind not in _SIM_KINDS:
        raise ValueError(f"kind must be one of {_SIM_KINDS}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 8:
        raise ValueError("n must be an integer >= 8")
    n = int(n)
    s = _check_seed(seed)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    for name, value in (("drift", drift), ("theta", theta), ("phi", phi)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= phi < 1.0:
        raise ValueError("phi must lie in [0, 1)")
    if not theta >= 0.0:
        raise ValueError("theta must be non-negative")
    if not 0.0 < float(alpha_garch) + float(beta_garch) < 1.0:
        raise ValueError("GARCH persistence alpha+beta must lie in (0, 1)")
    if float(alpha_garch) < 0.0 or float(beta_garch) < 0.0:
        raise ValueError("GARCH coefficients must be non-negative")

    gen = np.random.default_rng(s)
    if kind in ("brownian", "drift"):
        dx = gen.normal(float(drift), sigma, n)
    elif kind == "garch":
        z = gen.normal(0.0, 1.0, n)
        sig2 = np.empty(n, dtype=float)
        sig2[0] = sigma * sigma / (1.0 - float(alpha_garch) - float(beta_garch))
        dx = np.empty(n, dtype=float)
        dx[0] = math.sqrt(sig2[0]) * z[0]
        for t in range(1, n):
            sig2[t] = (
                sigma * sigma
                + float(alpha_garch) * dx[t - 1] ** 2
                + float(beta_garch) * sig2[t - 1]
            )
            dx[t] = math.sqrt(sig2[t]) * z[t]
    elif kind == "ou":
        dx = np.empty(n, dtype=float)
        level = 0.0
        for t in range(n):
            step = -theta * level + sigma * gen.normal(0.0, 1.0)
            dx[t] = step
            level += step
    else:  # ar1
        z = gen.normal(0.0, sigma, n)
        dx = np.empty(n, dtype=float)
        dx[0] = z[0]
        for t in range(1, n):
            dx[t] = phi * dx[t - 1] + z[t]
    path = np.concatenate([np.zeros(1), np.cumsum(dx)])
    return {
        "increments": dx,
        "path": path,
        "kind": kind,
        "config": {
            "data_label": "SYNTHETIC",
            "seed": s,
            "n": n,
            "drift": float(drift),
            "theta": float(theta),
            "phi": float(phi),
            "sigma": float(sigma),
            "note": "planted correctness world — never market evidence",
        },
    }


def _run_battery(
    kind: str,
    n: int,
    n_mc: int,
    seed: int,
    *,
    window: int,
    n_boot: int,
    n_perm: int,
    lam: float,
    kwargs: dict[str, float],
) -> list[dict[str, Any]]:
    """Run the full battery over ``n_mc`` seeded draws of one config."""
    out: list[dict[str, Any]] = []
    for rep in range(n_mc):
        world = simulate_martingale_paths(kind, n, seed + 7919 * rep, **kwargs)
        out.append(
            signature_martingale_test(
                world["path"],
                window,
                n_boot=n_boot,
                n_perm=n_perm,
                lam=lam,
                seed=seed + 104729 * rep,
            )
        )
    return out


def bench_signature_martingale_test(seed: int) -> dict[str, float]:
    """Seeded SYNTHETIC bench: size on martingale nulls, power on alternatives.

    Controls: iid-Gaussian and GARCH martingale-difference nulls must hold
    size ≈ alpha (nominal 0.05/0.10); constant-drift, OU mean-reversion and
    AR(1)-momentum alternatives must gain power with the planted strength.
    Also reports e-process evidence trajectories (median final e-value,
    Ville crossing rates). Flat ``synthetic_``-prefixed float keys —
    correctness evidence only, never market evidence.
    """
    s = _check_seed(seed)
    window = DEFAULT_WINDOW
    n, n_mc, n_boot, n_perm = 336, 48, 199, 199
    out: dict[str, float] = {
        "synthetic_seed": float(s),
        "synthetic_n": float(n),
        "synthetic_n_mc": float(n_mc),
        "synthetic_window": float(window),
        "synthetic_n_boot": float(n_boot),
        "synthetic_n_perm": float(n_perm),
    }

    # --- size: iid Gaussian martingale null -----------------------------------
    null_bm = _run_battery(
        "brownian",
        n,
        n_mc,
        s + 11,
        window=window,
        n_boot=n_boot,
        n_perm=n_perm,
        lam=DEFAULT_LAM,
        kwargs={},
    )
    out["synthetic_size_bm_005"] = float(np.mean([r["p_omnibus"] <= 0.05 for r in null_bm]))
    out["synthetic_size_bm_010"] = float(np.mean([r["p_omnibus"] <= 0.10 for r in null_bm]))
    out["synthetic_size_bm_boot_005"] = float(np.mean([r["p_boot"] <= 0.05 for r in null_bm]))
    out["synthetic_size_bm_perm_005"] = float(np.mean([r["p_perm"] <= 0.05 for r in null_bm]))
    out["synthetic_t_null_median"] = float(np.median([r["t_stat"] for r in null_bm]))
    out["synthetic_eval_null_median"] = float(np.median([r["e_final"] for r in null_bm]))
    out["synthetic_ville_rate_null_005"] = float(np.mean([r["e_reject"] for r in null_bm]))

    # --- size robustness: heteroskedastic (GARCH) martingale-difference null --
    null_g = _run_battery(
        "garch",
        n,
        n_mc,
        s + 23,
        window=window,
        n_boot=n_boot,
        n_perm=n_perm,
        lam=DEFAULT_LAM,
        kwargs={"alpha_garch": 0.12, "beta_garch": 0.82},
    )
    out["synthetic_size_garch_005"] = float(np.mean([r["p_omnibus"] <= 0.05 for r in null_g]))

    # --- power: constant-drift grid -------------------------------------------
    for tag, d in (("0p05", 0.05), ("0p10", 0.10), ("0p20", 0.20)):
        runs = _run_battery(
            "drift",
            n,
            n_mc,
            s + 37,
            window=window,
            n_boot=n_boot,
            n_perm=n_perm,
            lam=DEFAULT_LAM,
            kwargs={"drift": d},
        )
        out[f"synthetic_power_drift_{tag}_005"] = float(
            np.mean([r["p_omnibus"] <= 0.05 for r in runs])
        )
        out[f"synthetic_power_drift_boot_{tag}_005"] = float(
            np.mean([r["p_boot"] <= 0.05 for r in runs])
        )

    # reuse the drift=0.20 battery for the e-value trajectory evidence
    out["synthetic_eval_drift_hi_median"] = float(np.median([r["e_final"] for r in runs]))
    out["synthetic_ville_rate_drift_hi_005"] = float(np.mean([r["e_reject"] for r in runs]))

    # --- power: OU mean-reversion grid ----------------------------------------
    for tag, th in (("0p10", 0.10), ("0p30", 0.30)):
        runs = _run_battery(
            "ou",
            n,
            n_mc,
            s + 53,
            window=window,
            n_boot=n_boot,
            n_perm=n_perm,
            lam=DEFAULT_LAM,
            kwargs={"theta": th},
        )
        out[f"synthetic_power_ou_t{tag}_005"] = float(
            np.mean([r["p_omnibus"] <= 0.05 for r in runs])
        )
        out[f"synthetic_power_ou_perm_t{tag}_005"] = float(
            np.mean([r["p_perm"] <= 0.05 for r in runs])
        )

    # --- power: AR(1) momentum -------------------------------------------------
    runs_ar = _run_battery(
        "ar1",
        n,
        n_mc,
        s + 67,
        window=window,
        n_boot=n_boot,
        n_perm=n_perm,
        lam=DEFAULT_LAM,
        kwargs={"phi": 0.30},
    )
    out["synthetic_power_ar1_phi0p30_005"] = float(
        np.mean([r["p_omnibus"] <= 0.05 for r in runs_ar])
    )
    out["synthetic_power_ar1_perm_phi0p30_005"] = float(
        np.mean([r["p_perm"] <= 0.05 for r in runs_ar])
    )
    out["synthetic_live_pnl_claim"] = 0.0
    return out
