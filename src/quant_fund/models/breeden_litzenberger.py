"""Breeden-Litzenberger risk-neutral density extraction + implied moments.

SYNTHETIC research module: every smile here is model-generated (an in-module
lognormal-mixture "lognormal + curvature" DGP), never market data. No
live-trading or market-evidence claim is made or implied.

Pieces
------
1. :func:`extract_rnd` — the Breeden-Litzenberger (1978) identity

       f_Q(K) = e^{rT} * d2 C(K) / dK2

   recovers the risk-neutral density (RND) of the terminal price ``S_T`` from
   the second strike-derivative of the call-price function.  Raw second
   differences on noisy mid prices explode (the Aït-Sahalia & Lo 1998
   motivation for nonparametric smoothing), so the pipeline is the
   Shimko (1993) / Malz (2014) / Figlewski (2010) one:

   a. input no-arbitrage screen — call bounds ``e^{-rT} (F-K)+ <= C <=
      e^{-rT} F`` and strike-monotonicity, both within a relative
      ``price_tol`` (raw convexity is *not* screened: fair noisy mids are
      expected to violate it at noise scale, which is exactly what the
      smoothing step is for);
   b. a smoothing cubic spline fit through the mid prices in price space
      (:func:`scipy.interpolate.UnivariateSpline`; ``smoothing=None`` picks
      the residual target adaptively from a noise-variance estimate — see
      :func:`_noise_variance` — an explicit ``smoothing`` value fixes the
      FITPACK ``s`` target and ``smoothing=0`` forces exact interpolation,
      the right choice for near-exact prices; the call-price grid is used
      directly — a total-variance grid would invert mids through an
      implied-vol map and is left out by design);
   c. evaluation on a uniform dense strike grid (the stable grid the finite
      differences need — uneven input strikes make the naive stencil
      ill-conditioned);
   d. finite-difference second derivative (3-point interior, 3-point
      one-sided edges) times ``e^{rT}`` — the raw RND;
   e. repair — clip the (small) negative butterfly mass, then graft
      Figlewski-style exponential tails whose log-slope matches the density
      at each grid edge and which carry exactly the missing mass, then a
      final renormalization so the discrete density integrates to 1.

2. :func:`implied_moments` — risk-neutral mean, variance, skewness and
   excess kurtosis by trapezoid integration of the extracted density.

   Honest distinction: these are moments *of the terminal-price density*
   ``S_T``.  The second central moment ``E_Q[(S_T - E_Q S_T)^2]`` is NOT
   the model-free VIX-style variance-swap strike: the variance swap
   replicates the log contract ``-2/T E_Q[ln(S_T/F)]`` (expected realised
   variance under continuity), which loads on every moment of the density,
   not just the second.  For a lognormal ``S_T`` with total vol ``s`` the
   two differ exactly: ``Var[S_T] = F^2 (e^{s^2} - 1)`` vs. a fair strike of
   ``s^2/T``.  See ``quant_fund.models.variance_swap`` for the log-contract
   estimator; this module deliberately does not reimplement it.

3. :func:`arbitrage_report` — diagnostics on the extracted density:
   non-negativity violation mass/fraction on the raw estimate, mass defect
   before and after repair, forward-consistency gap ``(E_Q[S_T] - F)/F``
   (the martingale check — a properly extracted RND prices the underlying
   back to the forward), and, when a reference model density and/or model
   moments are supplied, L1 distance, KL divergence ``KL(q || f)`` and
   per-moment errors.

4. :func:`synthetic_mixture_smile` — the comparison harness DGP: a
   two-component lognormal mixture ("lognormal + curvature") whose weights,
   medians and vols are pinned so that ``E[S_T] = F`` exactly (mixture
   martingale calibration).  Call prices are the closed-form mixture of
   Black-76 components; the true density and true moments (raw-moment
   closed forms for a lognormal mixture) come along for scoring.  Do NOT
   import wave-23 SVI; the smile is generated in-module.

Fail-closed: non-finite inputs, non-positive or non-increasing strikes,
length mismatches, non-positive forward/maturity, prices violating the
input no-arbitrage screen beyond ``price_tol``, degenerate smoothing/grid
parameters, and non-normalizable extracted densities all raise
``ValueError``.  Deterministic: all randomness (mid-price noise) lives in
``numpy.random.default_rng(seed)``; extraction itself is fully
deterministic.

Honesty (AGENTS.md contract): every number a test or the bench emits is a
correctness check on the SYNTHETIC fixture — density L1/KL error, moment
error, mass defect — never market evidence.  Bench keys carry the
``synthetic_`` prefix.

Composition: complements ``quant_fund.models.options``
(:func:`risk_neutral_density` there is the raw second-difference estimator
with no smoothing, tail or moment layer — this module is the
arbitrage-audited version), ``quant_fund.models.local_stoch_vol`` (Dupire
uses the same butterfly object ``d2C/dK2`` as the local-variance
denominator across the whole surface; here it is the terminal density
itself), and ``quant_fund.models.fourier_pricing`` (characteristic-function
pricing is parametric; the RND here is model-free from quoted prices).
None of those files is edited.

References
----------
- Breeden, D. T., Litzenberger, R. H. (1978). Prices of state-contingent
  claims implicit in option prices. Journal of Business 51(4), 621-651.
  doi:10.1086/296025.
- Aït-Sahalia, Y., Lo, A. W. (1998). Nonparametric estimation of
  state-price densities implicit in financial asset prices. Journal of
  Finance 53(2), 499-547. doi:10.1111/0022-1082.215228.
- Figlewski, S. (2010). Estimating the implied risk neutral density for
  the U.S. market portfolio. In Volatility and Time Series Econometrics:
  Essays in Honor of Robert F. Engle (T. Bollerslev, J. R. Russell,
  M. W. Watson, eds.). Oxford University Press.
- Shimko, D. (1993). Bounds of probability. RISK 6(4), 33-37.
- Malz, A. M. (2014). A simple and reliable way to compute option-based
  risk-neutral distributions. Federal Reserve Bank of New York Staff
  Report No. 677.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline
from scipy.special import binom
from scipy.stats import norm

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__ = [
    "RNDResult",
    "SmileFixture",
    "arbitrage_report",
    "bench_breeden_litzenberger",
    "extract_rnd",
    "implied_moments",
    "synthetic_mixture_smile",
]

SYNTHETIC_LABEL = "SYNTHETIC: model-generated smiles only; not market data or market evidence"

#: Sentinel recorded in diagnostics when the adaptive noise estimate picked s.
_LAM_GCV_SENTINEL = -1.0

#: Floor for edge log-slopes when the edge density is flat/rising: tails then
#: decay over ~a quarter of the observed strike range.
_TAIL_SLOPE_FALLBACK_FRAC = 0.25

#: KL guard: densities are floored at eps * max(model density) before KL.
_KL_EPS_FRAC = 1e-12


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")


def _check_strikes_calls(
    strikes: Array | list[float],
    call_prices: Array | list[float],
    min_len: int = 5,
) -> tuple[Array, Array]:
    k = np.asarray(strikes, dtype=float).ravel()
    c = np.asarray(call_prices, dtype=float).ravel()
    if k.size != c.size or k.size < min_len:
        raise ValueError(
            f"strikes/call_prices must share length >= {min_len}, got {k.size}/{c.size}"
        )
    if not (np.isfinite(k).all() and np.isfinite(c).all()):
        raise ValueError("strikes and call_prices must be finite")
    if (k <= 0.0).any():
        raise ValueError("strikes must be positive")
    order = np.argsort(k, kind="stable")
    k = k[order]
    c = c[order]
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("strikes must be distinct (strictly increasing after sort)")
    return k, c


def _check_term(forward: float, r: float, maturity: float) -> None:
    if not np.isfinite(forward) or forward <= 0.0:
        raise ValueError("forward must be positive and finite")
    if not np.isfinite(r):
        raise ValueError("r must be finite")
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be positive and finite")


def _input_arb_screen(
    k: Array, c: Array, forward: float, r: float, t: float, price_tol: float
) -> None:
    """Hard bounds + monotonicity on the raw quotes (convexity left to the spline)."""
    disc = float(np.exp(-r * t))
    tol = price_tol * forward * disc
    lower = disc * np.maximum(forward - k, 0.0)
    upper = disc * forward
    if (c < lower - tol).any():
        raise ValueError("call prices fail no-arbitrage screen: below intrinsic lower bound")
    if (c > upper + tol).any():
        raise ValueError("call prices fail no-arbitrage screen: above upper bound")
    if (np.diff(c) > tol).any():
        raise ValueError("call prices fail no-arbitrage screen: increasing in strike")


def _noise_variance(c: Array) -> float:
    """Mid-price noise variance from normalized higher-order differences.

    For iid mid noise, E[(Δ^j c)^2] = binom(2j, j) sigma^2 while signal
    curvature enters at O(h^j c^{(j)}), so j in {3, 4} suppresses the smile's
    own curvature relative to the noise. Taking the min over j keeps the
    estimator from firing on near-exact prices (where all curvature is real).
    """
    best = 0.0
    for j in (3, 4):
        d = np.diff(c, n=j)
        if d.size == 0:
            continue
        est = float(np.mean(d * d) / binom(2 * j, j))
        best = est if best == 0.0 else min(best, est)
    return best


def _second_derivative_uniform(y: Array, h: float) -> Array:
    """Second derivative on a uniform grid: 3-pt interior, 3-pt one-sided edges."""
    d2 = np.empty_like(y)
    d2[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (h * h)
    d2[0] = (2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]) / (h * h)
    d2[-1] = (2.0 * y[-1] - 5.0 * y[-2] + 4.0 * y[-3] - y[-4]) / (h * h)
    return d2


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RNDResult:
    """Extracted risk-neutral density on the (possibly tail-extended) grid."""

    strikes: Array  # full evaluation grid: left tail + observed range + right tail
    density: Array  # repaired RND: non-negative, integrates to ~1
    density_raw: Array  # raw e^{rT} d2C/dK2 inside the observed range, NaN in tails
    observed_mask: BoolArray  # True over the observed-strike region
    forward: float
    r: float
    maturity: float
    diagnostics: dict[str, float]


def _edge_log_slope(k: Array, f: Array, side: str, window: int = 8) -> float:
    """d log f / dK at a grid edge from a short positive-density edge window."""
    n = min(window, f.size)
    if side == "right":
        kx, fx = k[-n:], f[-n:]
    else:
        kx, fx = k[:n], f[:n]
    pos = fx > 0.0
    if pos.sum() < 3:
        return 0.0
    slope = float(np.polyfit(kx[pos], np.log(fx[pos]), 1)[0])
    return slope


def _graft_tails(
    kg: Array,
    f_clip: Array,
    deficit: float,
    tail_points: int,
) -> tuple[Array, Array, float]:
    """Exponential tails carrying ``deficit`` mass, slopes matched at the edges.

    Returns ``(strikes, density, grafted_mass)`` on the extended grid. Each
    tail decays exponentially from its edge density with a log-slope matched
    to the fitted edge (flat/rising edges fall back to decay over ~a quarter
    of the observed range); tails are scaled so their combined trapezoid
    mass equals ``deficit``. A side with a degenerate edge density (<= 0) or
    no room (left floor reaching the edge) is skipped.
    """
    h = float(kg[1] - kg[0])
    width = float(kg[-1] - kg[0])
    lam_floor = 1.0 / (_TAIL_SLOPE_FALLBACK_FRAC * width)

    k_left = np.empty(0)
    f_left = np.empty(0)
    f_lo = float(f_clip[0])
    if f_lo > 0.0:
        s_l = _edge_log_slope(kg, f_clip, "left")
        lam_l = s_l if s_l > 0.0 else lam_floor
        k_floor = max(kg[0] - tail_points * h, 0.02 * kg[0])
        if k_floor < kg[0] - h:
            k_left = np.linspace(k_floor, kg[0], tail_points + 1)[:-1]
            f_left = f_lo * np.exp(-lam_l * (kg[0] - k_left))

    k_right = np.empty(0)
    f_right = np.empty(0)
    f_hi = float(f_clip[-1])
    if f_hi > 0.0:
        s_r = _edge_log_slope(kg, f_clip, "right")
        lam_r = -s_r if s_r < 0.0 else lam_floor
        k_right = kg[-1] + h * np.arange(1, tail_points + 1)
        f_right = f_hi * np.exp(-lam_r * (k_right - kg[-1]))

    if k_left.size == 0 and k_right.size == 0:
        return kg, f_clip.copy(), 0.0

    tail_mass = 0.0
    if k_left.size:
        tail_mass += float(np.trapezoid(f_left, k_left))
    if k_right.size:
        tail_mass += float(np.trapezoid(f_right, k_right))
    scale = deficit / tail_mass if tail_mass > 0.0 else 0.0
    grafted = deficit if tail_mass > 0.0 else 0.0

    k_full = np.concatenate([k_left, kg, k_right])
    f_full = np.concatenate([f_left * scale, f_clip, f_right * scale])
    return k_full, f_full, grafted


def extract_rnd(
    strikes: Array | list[float],
    call_prices: Array | list[float],
    *,
    forward: float,
    r: float = 0.0,
    maturity: float,
    smoothing: float | None = None,
    grid_points: int = 401,
    tail_points: int = 64,
    price_tol: float = 0.02,
) -> RNDResult:
    """Breeden-Litzenberger RND from a smile of European call mid prices.

    Pipeline: input no-arbitrage screen -> smoothing spline (adaptive
    noise-variance target when ``smoothing=None``; ``smoothing=0`` forces
    exact interpolation) -> uniform-grid finite-difference ``d2C/dK2`` ->
    clip + exponential tail graft + renormalization. See module docstring.
    """
    k, c = _check_strikes_calls(strikes, call_prices)
    _check_term(forward, r, maturity)
    if smoothing is not None and (not np.isfinite(smoothing) or smoothing < 0.0):
        raise ValueError("smoothing must be non-negative (None = adaptive)")
    if not isinstance(grid_points, int) or isinstance(grid_points, bool) or grid_points < 16:
        raise ValueError("grid_points must be an int >= 16")
    if not isinstance(tail_points, int) or isinstance(tail_points, bool) or tail_points < 0:
        raise ValueError("tail_points must be an int >= 0")
    if not np.isfinite(price_tol) or price_tol <= 0.0:
        raise ValueError("price_tol must be positive and finite")
    _input_arb_screen(k, c, forward, r, maturity, price_tol)

    s_target = k.size * _noise_variance(c) if smoothing is None else float(smoothing)
    spline = UnivariateSpline(k, c, s=s_target)
    kg = np.linspace(k[0], k[-1], grid_points)
    cg = spline(kg)
    h = float(kg[1] - kg[0])
    f_raw = float(np.exp(r * maturity)) * _second_derivative_uniform(cg, h)

    mass_raw = float(np.trapezoid(f_raw, kg))
    neg = np.minimum(f_raw, 0.0)
    negative_mass_raw = float(-np.trapezoid(neg, kg))
    negativity_violation_frac = float(np.mean(f_raw < 0.0))
    mass_defect_raw = 1.0 - mass_raw

    f_clip = np.maximum(f_raw, 0.0)
    grid_mass = float(np.trapezoid(f_clip, kg))
    deficit = 1.0 - grid_mass

    if tail_points > 0 and deficit > 0.0:
        k_full, f_full, grafted = _graft_tails(kg, f_clip, deficit, tail_points)
    else:
        k_full, f_full, grafted = kg, f_clip.copy(), 0.0

    total_mass = float(np.trapezoid(f_full, k_full))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("extracted density is not normalizable (mass <= 0)")
    f_full = f_full / total_mass
    mass_defect_after = float(abs(1.0 - np.trapezoid(f_full, k_full)))

    observed_mask = (k_full >= kg[0]) & (k_full <= kg[-1])
    density_raw = np.full(k_full.size, np.nan)
    density_raw[observed_mask] = f_raw

    diagnostics: dict[str, float] = {
        "mass_raw": mass_raw,
        "mass_defect_raw": mass_defect_raw,
        "negative_mass_raw": negative_mass_raw,
        "negativity_violation_frac": negativity_violation_frac,
        "grid_mass_clipped": grid_mass,
        "tail_mass_grafted": grafted,
        "mass_defect_after_repair": mass_defect_after,
        "smoothing_lambda": float(smoothing) if smoothing is not None else _LAM_GCV_SENTINEL,
        "noise_variance_est": float(_noise_variance(c)),
        "n_input_strikes": float(k.size),
        "synthetic_safe": 1.0,
    }
    return RNDResult(
        strikes=k_full,
        density=f_full,
        density_raw=density_raw,
        observed_mask=observed_mask,
        forward=forward,
        r=r,
        maturity=maturity,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# moments + arbitrage diagnostics
# ---------------------------------------------------------------------------


def implied_moments(strikes: Array | list[float], density: Array | list[float]) -> dict[str, float]:
    """Risk-neutral moments of ``S_T`` from a density on a strike grid.

    Returns ``mass``, ``mean``, ``variance``, ``std``, ``skewness`` and
    ``excess_kurtosis``. These are moments of the *terminal-price* density,
    not the model-free log-contract/variance-swap quantities (see module
    docstring for the VIX-style distinction).
    """
    k = np.asarray(strikes, dtype=float).ravel()
    f = np.asarray(density, dtype=float).ravel()
    if k.size != f.size or k.size < 3:
        raise ValueError("strikes/density must share length >= 3")
    if not (np.isfinite(k).all() and np.isfinite(f).all()):
        raise ValueError("strikes and density must be finite")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("strikes must be strictly increasing")
    if (f < 0.0).any():
        raise ValueError("density must be non-negative")
    mass = float(np.trapezoid(f, k))
    if mass <= 0.0:
        raise ValueError("density must have positive mass")
    g = f / mass
    mean = float(np.trapezoid(k * g, k))
    m2 = float(np.trapezoid((k - mean) ** 2 * g, k))
    m3 = float(np.trapezoid((k - mean) ** 3 * g, k))
    m4 = float(np.trapezoid((k - mean) ** 4 * g, k))
    if m2 <= 0.0:
        raise ValueError("density has zero variance")
    return {
        "mass": mass,
        "mean": mean,
        "variance": m2,
        "std": float(np.sqrt(m2)),
        "skewness": m3 / m2**1.5,
        "excess_kurtosis": m4 / (m2 * m2) - 3.0,
    }


def arbitrage_report(
    result: RNDResult,
    *,
    model_density: Array | list[float] | None = None,
    reference_moments: dict[str, float] | None = None,
) -> dict[str, float]:
    """Arbitrage/quality diagnostics on an extracted RND.

    Always reports the raw non-negativity and mass-defect diagnostics plus
    the forward-consistency gap ``(E_Q[S_T] - F) / F``. With ``model_density``
    (evaluated on ``result.strikes``) adds ``l1_distance`` and
    ``kl_divergence`` (KL(true || recovered), both renormalized on the grid
    with a small floor). With ``reference_moments`` adds per-moment errors.
    """
    if not isinstance(result, RNDResult):
        raise ValueError("result must be an RNDResult")
    mom = implied_moments(result.strikes, result.density)
    out: dict[str, float] = {
        "mass": mom["mass"],
        "mass_defect_raw": result.diagnostics["mass_defect_raw"],
        "mass_defect_after_repair": result.diagnostics["mass_defect_after_repair"],
        "negative_mass_raw": result.diagnostics["negative_mass_raw"],
        "negativity_violation_frac": result.diagnostics["negativity_violation_frac"],
        "tail_mass_grafted": result.diagnostics["tail_mass_grafted"],
        "forward_gap": (mom["mean"] - result.forward) / result.forward,
        "mean": mom["mean"],
        "variance": mom["variance"],
        "skewness": mom["skewness"],
        "excess_kurtosis": mom["excess_kurtosis"],
    }
    if model_density is not None:
        q = np.asarray(model_density, dtype=float).ravel()
        if q.size != result.strikes.size:
            raise ValueError("model_density must match result.strikes length")
        if not np.isfinite(q).all() or (q < 0.0).any():
            raise ValueError("model_density must be finite and non-negative")
        q_mass = float(np.trapezoid(q, result.strikes))
        if q_mass <= 0.0:
            raise ValueError("model_density must have positive mass on the grid")
        q = q / q_mass
        f = result.density / float(np.trapezoid(result.density, result.strikes))
        out["l1_distance"] = float(np.trapezoid(np.abs(f - q), result.strikes))
        eps = _KL_EPS_FRAC * float(q.max())
        out["kl_divergence"] = float(
            np.trapezoid(q * np.log((q + eps) / (f + eps)), result.strikes)
        )
    if reference_moments is not None:
        for key in ("mean", "variance", "skewness", "excess_kurtosis"):
            if key not in reference_moments or not np.isfinite(reference_moments[key]):
                raise ValueError(f"reference_moments[{key!r}] must be present and finite")
        out["mean_abs_err"] = abs(mom["mean"] - reference_moments["mean"])
        out["mean_rel_err"] = abs(mom["mean"] - reference_moments["mean"]) / abs(
            reference_moments["mean"]
        )
        out["variance_rel_err"] = abs(mom["variance"] - reference_moments["variance"]) / abs(
            reference_moments["variance"]
        )
        out["skewness_abs_err"] = abs(mom["skewness"] - reference_moments["skewness"])
        out["excess_kurtosis_abs_err"] = abs(
            mom["excess_kurtosis"] - reference_moments["excess_kurtosis"]
        )
    return out


# ---------------------------------------------------------------------------
# synthetic comparison harness (lognormal-mixture smile DGP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmileFixture:
    """A synthetic smile: quoted mids plus the true density/moments for scoring."""

    strikes: Array  # quoted strike grid
    call_prices: Array  # mid prices (== call_prices_clean when noise_std == 0)
    call_prices_clean: Array
    forward: float
    r: float
    maturity: float
    mix_weights: Array  # component weights (sum 1)
    mix_medians: Array  # component lognormal medians
    mix_vols: Array  # component total log-vols (sigma * sqrt(T))
    fine_grid: Array
    true_density: Array  # mixture density on fine_grid
    true_moments: dict[str, float]

    def density_on(self, k: Array | list[float]) -> Array:
        """True mixture lognormal density evaluated on ``k``."""
        x = np.asarray(k, dtype=float).ravel()
        if (x <= 0.0).any():
            raise ValueError("density_on requires positive strikes")
        out = np.zeros_like(x)
        for w, m, s in zip(self.mix_weights, self.mix_medians, self.mix_vols, strict=True):
            out += w * norm.pdf(np.log(x / m) / s) / (x * s)
        return out

    def call_price_on(self, k: Array | list[float]) -> Array:
        """Clean mixture call price (Black-76 components, discounted)."""
        x = np.asarray(k, dtype=float).ravel()
        if (x <= 0.0).any():
            raise ValueError("call_price_on requires positive strikes")
        undisc = np.zeros_like(x)
        for w, m, s in zip(self.mix_weights, self.mix_medians, self.mix_vols, strict=True):
            d1 = (np.log(m / x) + s * s) / s
            d2 = d1 - s
            undisc += w * (m * np.exp(0.5 * s * s) * norm.cdf(d1) - x * norm.cdf(d2))
        return float(np.exp(-self.r * self.maturity)) * undisc


def _mixture_raw_moments(weights: Array, medians: Array, vols: Array) -> Array:
    """Raw moments E[S^k], k = 1..4, of a lognormal mixture (closed form)."""
    ks = np.arange(1, 5)[:, None]
    # E[S^k | comp] = m^k * exp(k^2 s^2 / 2)
    comp = medians[None, :] ** ks * np.exp(0.5 * ks**2 * vols[None, :] ** 2)
    return comp @ weights


def _mixture_true_moments(weights: Array, medians: Array, vols: Array) -> dict[str, float]:
    m1, m2, m3, m4 = _mixture_raw_moments(weights, medians, vols)
    var = m2 - m1 * m1
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1**4
    return {
        "mean": float(m1),
        "variance": float(var),
        "std": float(np.sqrt(var)),
        "skewness": float(c3 / var**1.5),
        "excess_kurtosis": float(c4 / (var * var) - 3.0),
    }


def synthetic_mixture_smile(
    *,
    seed: int = 0,
    n_strikes: int = 25,
    noise_std: float = 0.0,
    forward: float = 100.0,
    r: float = 0.03,
    maturity: float = 0.5,
    weight: float = 0.30,
    crash_log_median: float = -0.55,
    vol1: float = 0.45,
    vol2: float = 0.18,
    strike_lo_log: float = -1.4,
    strike_hi_log: float = 0.9,
    fine_points: int = 2001,
) -> SmileFixture:
    """Two-component lognormal-mixture smile with known density and moments.

    Component 1 (``weight``) is a low-median high-vol "crash" piece giving the
    curvature/skew; component 2's median is solved so the mixture mean equals
    ``forward`` exactly (risk-neutral martingale calibration). ``noise_std``
    is the mid-price noise std as a fraction of the discounted forward.
    """
    _check_seed(seed)
    _check_term(forward, r, maturity)
    if not isinstance(n_strikes, int) or isinstance(n_strikes, bool) or n_strikes < 5:
        raise ValueError("n_strikes must be an int >= 5")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if not np.isfinite(weight) or not 0.0 < weight < 1.0:
        raise ValueError("weight must lie in (0, 1)")
    if not (np.isfinite(vol1) and vol1 > 0.0 and np.isfinite(vol2) and vol2 > 0.0):
        raise ValueError("vol1 and vol2 must be positive")
    if not np.isfinite(crash_log_median) or crash_log_median >= 0.0:
        raise ValueError("crash_log_median must be negative")
    if not (
        np.isfinite(strike_lo_log) and np.isfinite(strike_hi_log) and strike_lo_log < strike_hi_log
    ):
        raise ValueError("require strike_lo_log < strike_hi_log")
    if not isinstance(fine_points, int) or isinstance(fine_points, bool) or fine_points < 101:
        raise ValueError("fine_points must be an int >= 101")

    med1 = forward * float(np.exp(crash_log_median))
    med2 = (forward - weight * med1 * np.exp(0.5 * vol1 * vol1)) / (
        (1.0 - weight) * np.exp(0.5 * vol2 * vol2)
    )
    if not np.isfinite(med2) or med2 <= 0.0:
        raise ValueError("mixture parameters imply non-positive second median")

    weights = np.array([weight, 1.0 - weight])
    medians = np.array([med1, med2])
    vols = np.array([vol1, vol2])

    strikes = forward * np.exp(np.linspace(strike_lo_log, strike_hi_log, n_strikes))
    clean = np.zeros_like(strikes)
    for w, m, s in zip(weights, medians, vols, strict=True):
        d1 = (np.log(m / strikes) + s * s) / s
        d2 = d1 - s
        clean += w * (m * np.exp(0.5 * s * s) * norm.cdf(d1) - strikes * norm.cdf(d2))
    clean *= float(np.exp(-r * maturity))

    rng = np.random.default_rng(seed)
    mids = clean + rng.normal(0.0, noise_std * forward * np.exp(-r * maturity), strikes.size)

    fine_grid = forward * np.exp(np.linspace(strike_lo_log - 0.3, strike_hi_log + 0.6, fine_points))
    true_density = np.zeros_like(fine_grid)
    for w, m, s in zip(weights, medians, vols, strict=True):
        true_density += w * norm.pdf(np.log(fine_grid / m) / s) / (fine_grid * s)

    return SmileFixture(
        strikes=strikes,
        call_prices=mids,
        call_prices_clean=clean,
        forward=forward,
        r=r,
        maturity=maturity,
        mix_weights=weights,
        mix_medians=medians,
        mix_vols=vols,
        fine_grid=fine_grid,
        true_density=true_density,
        true_moments=_mixture_true_moments(weights, medians, vols),
    )


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def bench_breeden_litzenberger(seed: int = 20240717) -> dict[str, float]:
    """Flat dict of SYNTHETIC benchmark numbers for the RND pipeline.

    Runs extraction on the noise-free and noisy (0.4%-of-forward mid noise)
    mixture smiles and reports density L1/KL error, per-moment errors, mass
    defect before/after repair, the forward martingale gap, and noise
    degradation. All keys are ``synthetic_``-prefixed floats — correctness
    metrics, never market evidence.
    """
    _check_seed(seed)

    fx_clean = synthetic_mixture_smile(seed=seed, noise_std=0.0)
    res_clean = extract_rnd(
        fx_clean.strikes,
        fx_clean.call_prices,
        forward=fx_clean.forward,
        r=fx_clean.r,
        maturity=fx_clean.maturity,
    )
    rep_clean = arbitrage_report(
        res_clean,
        model_density=fx_clean.density_on(res_clean.strikes),
        reference_moments=fx_clean.true_moments,
    )

    fx_noisy = synthetic_mixture_smile(seed=seed + 1, noise_std=0.004)
    res_noisy = extract_rnd(
        fx_noisy.strikes,
        fx_noisy.call_prices,
        forward=fx_noisy.forward,
        r=fx_noisy.r,
        maturity=fx_noisy.maturity,
    )
    rep_noisy = arbitrage_report(
        res_noisy,
        model_density=fx_noisy.density_on(res_noisy.strikes),
        reference_moments=fx_noisy.true_moments,
    )

    return {
        "synthetic_l1_density_clean": rep_clean["l1_distance"],
        "synthetic_kl_divergence_clean": rep_clean["kl_divergence"],
        "synthetic_mean_rel_err": rep_clean["mean_rel_err"],
        "synthetic_variance_rel_err": rep_clean["variance_rel_err"],
        "synthetic_skewness_abs_err": rep_clean["skewness_abs_err"],
        "synthetic_exkurt_abs_err": rep_clean["excess_kurtosis_abs_err"],
        "synthetic_mass_defect_raw": rep_clean["mass_defect_raw"],
        "synthetic_mass_defect_after_repair": rep_clean["mass_defect_after_repair"],
        "synthetic_negative_mass_raw": rep_clean["negative_mass_raw"],
        "synthetic_negativity_violation_frac": rep_clean["negativity_violation_frac"],
        "synthetic_tail_mass_grafted": rep_clean["tail_mass_grafted"],
        "synthetic_forward_gap": abs(rep_clean["forward_gap"]),
        "synthetic_l1_density_noisy": rep_noisy["l1_distance"],
        "synthetic_mean_rel_err_noisy": rep_noisy["mean_rel_err"],
        "synthetic_variance_rel_err_noisy": rep_noisy["variance_rel_err"],
        "synthetic_negative_mass_noisy": rep_noisy["negative_mass_raw"],
        "synthetic_noise_l1_degradation": rep_noisy["l1_distance"] - rep_clean["l1_distance"],
    }
