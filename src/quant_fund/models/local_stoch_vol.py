"""Dupire (1994) local volatility + stochastic-local-volatility (SLV) Monte Carlo.

SYNTHETIC research module: every surface here is model-generated (closed-form
SABR/Heston or Monte Carlo), never market data. No live-trading or market
evidence claim is made or implied.

Pieces
------
1. :func:`dupire_local_vol` — local-vol surface sigma_L(K,T) from a discrete
   call surface C(K,T) via the Dupire formula (Dupire 1994, "Pricing with a
   smile", Risk 7(1)):

       sigma_L^2(K,T) = (dC/dT + (r-q) K dC/dK + q C) / (0.5 K^2 d2C/dK2)

   (the q=0 form is Dupire's original; q is carried for dividend-paying
   underlyings). Raw finite differences explode on noisy surfaces, so prices
   first pass a separable Tikhonov filter solved exactly by Sylvester
   eigendecomposition:

       min_C~ ||C~ - C||^2 + lam_T ||D1 C~||^2 + lam_K ||C~ D2^T||^2
       <=>  (I + lam_T D1^T D1) C~ + C~ (lam_K D2^T D2) = C

   with D1/D2 unit-spacing first/second difference stencils. Fail-closed
   no-arbitrage screens run on the raw input: call bounds, monotonicity and
   convexity in K (butterfly), calendar monotonicity of e^{rT} C.

2. :func:`local_vol_mc` — Euler/Milstein Monte Carlo under
   dS = (r-q) S dt + sigma_L(S,t) S dW with bilinear (T, lnK) interpolation
   (clipped extrapolation). The fundamental Dupire consistency check — implied
   vols of local-vol MC prices reproduce the input surface's implied vols — is
   asserted in tests.

3. :func:`heston_mc` — inline log-Euler Heston path simulator. The repo's
   ``quant_fund.quant_models.heston`` is characteristic-function pricing
   (no paths), so the SLV scheme needs its own path engine here. ``xi = 0`` is
   allowed and gives the deterministic-variance (pure local-vol) degenerate
   case used by the degenerate tests.

4. :func:`slv_leverage_function` — van der Stoep, Grzelak & Oosterlee (2014),
   "The Heston stochastic-local volatility model: efficient Monte Carlo
   simulation" (IJTAF 17(3):1450016, arXiv:1211.2993), mixing-MC calibration
   of the Gatheral (2006) leverage function L(S,t) in the SLV dynamics

       dS = (r-q) S dt + L(S,t) sqrt(v_t) S dW_S.

   L is piecewise constant on (S,t) bins and updated by the conditional
   variance fixed point

       L2_new(s,t) = L2_old(s,t) * E_hat_Heston[V_t | S_t in bin(s)]
                                  / E_hat_SLV[V_t | S_t in bin(s)],

   so the SLV model reproduces the target Heston implied surface; a few
   iterations suffice on test-grade grids. Bergomi (2009), "Smile dynamics
   IV", Risk 22(9), gives the consistency background (skew vs conditional
   variance).

Degenerate checks (asserted in tests): leverage == 1 with xi == 0 recovers a
pure deterministic local-vol model matching the closed-form Black price;
leverage == 1 with xi > 0 recovers plain Heston MC bitwise (same streams).

Fail-closed: non-monotone/non-positive grids, non-finite values,
arbitrage-violating input surfaces, |rho| >= 1, xi < 0, or degenerate
butterfly density beyond ``max_degenerate_frac`` raise ``ValueError``. All
simulators are seeded and deterministic, antithetic by default, and draw
counter-based Philox normals from :mod:`quant_fund.mc_engine.philox` with
stream ids >= ``USER_STREAM_ID_MIN`` so they cannot collide with engine
streams.

References
----------
- Dupire, B. (1994). Pricing with a smile. Risk 7(1), 32-39.
- Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley.
- van der Stoep, A. W., Grzelak, L. A., Oosterlee, C. W. (2014). The Heston
  stochastic-local volatility model: efficient Monte Carlo simulation.
  Int. J. Theor. Appl. Finance 17(3):1450016. arXiv:1211.2993.
- Bergomi, L. (2009). Smile dynamics IV. Risk 22(9), 100-105.
- Heston, S. (1993). A closed-form solution for options with stochastic
  volatility with applications to bond and currency options. RFS 6(2).
- Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002). Managing smile
  risk. Wilmott Magazine (seed surfaces via ``quant_fund.models.sabr``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil, exp, log

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh

from quant_fund.mc_engine.philox import USER_STREAM_ID_MIN, philox_normals
from quant_fund.models.options import implied_vol

Array = NDArray[np.float64]

__all__ = [
    "LocalVolSurface",
    "dupire_local_vol",
    "heston_mc",
    "local_vol_mc",
    "make_leverage_interpolator",
    "mc_implied_vols",
    "slv_leverage_function",
    "slv_price_calls",
]

SYNTHETIC_LABEL = (
    "SYNTHETIC: model-generated surfaces/paths only; not market data or market evidence"
)

# Default variance clamp for the Dupire extraction: (0.5%)^2 .. (500%)^2.
_VAR_FLOOR_DEFAULT = 2.5e-5
_VAR_CAP_DEFAULT = 25.0

# Philox stream ids (all >= USER_STREAM_ID_MIN, engine-reserved ids are 1-15).
_STREAM_LOCAL_VOL_MC = USER_STREAM_ID_MIN
_STREAM_HESTON = USER_STREAM_ID_MIN + 2
_STREAM_SLV_PRICE = USER_STREAM_ID_MIN + 6
_STREAM_SLV_CALIB = USER_STREAM_ID_MIN + 8


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")


def _check_1d(x: Array | list[float] | tuple[float, ...], name: str, min_len: int) -> Array:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} entries, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _check_strictly_increasing_positive(
    x: Array | list[float] | tuple[float, ...], name: str, min_len: int
) -> Array:
    arr = _check_1d(x, name, min_len)
    if (arr <= 0.0).any():
        raise ValueError(f"{name} must be > 0")
    if not (np.diff(arr) > 0.0).all():
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _check_paths(n_paths: int, antithetic: bool) -> None:
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 100:
        raise ValueError("n_paths must be an int >= 100")
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic=True")


def _antithetic_normals(
    seed: int, n_paths: int, n_draws: int, *, stream_id: int, antithetic: bool
) -> Array:
    """Philox normals, shape (n_paths, n_draws), optionally sign-mirrored."""
    _check_seed(seed)
    _check_paths(n_paths, antithetic)
    n_base = n_paths // 2 if antithetic else n_paths
    z = philox_normals(seed, np.arange(n_base, dtype=np.int64), n_draws, stream_id=stream_id)
    if antithetic:
        z = np.concatenate([z, -z], axis=0)
    return z


def _build_time_grid(nodes: Array, max_dt: float) -> tuple[Array, list[int]]:
    """Refine maturity nodes to steps <= max_dt; return (grid, record indices)."""
    if not np.isfinite(max_dt) or max_dt <= 0.0:
        raise ValueError("max_dt must be finite and > 0")
    grid: list[float] = [0.0]
    record: list[int] = []
    prev = 0.0
    for t in nodes.tolist():
        n_sub = max(1, int(ceil((float(t) - prev) / max_dt)))
        for j in range(1, n_sub):
            grid.append(prev + (float(t) - prev) * j / n_sub)
        grid.append(float(t))
        record.append(len(grid) - 1)
        prev = float(t)
    return np.asarray(grid, dtype=float), record


def _diff_matrix(n: int, order: int) -> Array:
    """Unit-spacing difference stencil matrix (order 1 or 2), shape (n-order, n)."""
    if order == 1:
        d = np.zeros((n - 1, n))
        rows = np.arange(n - 1)
        d[rows, rows] = -1.0
        d[rows, rows + 1] = 1.0
        return d
    if order == 2:
        d = np.zeros((n - 2, n))
        rows = np.arange(n - 2)
        d[rows, rows] = 1.0
        d[rows, rows + 1] = -2.0
        d[rows, rows + 2] = 1.0
        return d
    raise ValueError("order must be 1 or 2")


def _smooth_tikhonov(prices: Array, lambda_t: float, lambda_k: float) -> tuple[Array, float]:
    """Separable Tikhonov smoothing, exact via Sylvester eigendecomposition.

    Solves ``min ||X - C||^2 + lambda_t ||D1 X||^2 + lambda_k ||X D2^T||^2``,
    i.e. ``(I + lambda_t D1^T D1) X + X (lambda_k D2^T D2) = C``. Both system
    matrices are symmetric (positive definite / semidefinite), so simultaneous
    diagonalization gives the closed form. Returns (smoothed, rms change).
    """
    if lambda_t < 0.0 or lambda_k < 0.0:
        raise ValueError("lambda_t and lambda_k must be >= 0")
    if lambda_t == 0.0 and lambda_k == 0.0:
        return prices.copy(), 0.0
    n_t, n_k = prices.shape
    # Normal equations: M_t X + X N_k = C with M_t = I + lam_t D1^T D1 (SPD)
    # and N_k = lam_k D2^T D2 (PSD). Simultaneous diagonalization gives the
    # closed form Y_ij = (U_t^T C U_k)_ij / (e_t_i + e_k_j), X = U_t Y U_k^T.
    if lambda_t > 0.0:
        d1 = _diff_matrix(n_t, 1)
        e_t, u_t = eigh(np.eye(n_t) + lambda_t * (d1.T @ d1))
        e_t = np.clip(e_t, 1e-12, None)
    else:
        e_t, u_t = np.ones(n_t), np.eye(n_t)
    if lambda_k > 0.0:
        d2 = _diff_matrix(n_k, 2)
        e_k, u_k = eigh(lambda_k * (d2.T @ d2))
        e_k = np.clip(e_k, 0.0, None)
    else:
        e_k, u_k = np.zeros(n_k), np.eye(n_k)
    core = (u_t.T @ prices @ u_k) / (e_t[:, None] + e_k[None, :])
    smoothed = u_t @ core @ u_k.T
    delta = float(np.sqrt(np.mean((smoothed - prices) ** 2)))
    return smoothed, delta


def _second_derivative_k(prices: Array, strikes: Array) -> Array:
    """Non-uniform 3-point d2C/dK2; edge nodes replicate the nearest interior."""
    h1 = strikes[1:-1] - strikes[:-2]
    h2 = strikes[2:] - strikes[1:-1]
    out = np.empty_like(prices)
    out[:, 1:-1] = 2.0 * (
        prices[:, :-2] / (h1 * (h1 + h2))[None, :]
        - prices[:, 1:-1] / (h1 * h2)[None, :]
        + prices[:, 2:] / (h2 * (h1 + h2))[None, :]
    )
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def _ffill_bad(var: Array, good: NDArray[np.bool_]) -> Array:
    """Fill non-good entries with the nearest good value (row ffill then bfill)."""
    out = np.where(good, var, np.nan)
    n_t, n_k = out.shape
    cols = np.arange(n_k)
    for row in range(n_t):
        g = good[row]
        if g.all():
            continue
        if not g.any():
            continue  # fully degenerate row: left NaN, caught by caller
        idx = np.where(g, cols, 0)
        np.maximum.accumulate(idx, out=idx)
        filled = out[row, idx]
        first = int(np.argmax(g))
        filled[:first] = out[row, first]
        out[row] = filled
    return out


@dataclass(eq=False)
class LocalVolSurface:
    """Local-vol surface sigma_L on a (T, K) grid with a bilinear interpolator.

    ``vol`` interpolates linearly in (T, lnK) and clamps queries to the grid
    box (constant extrapolation), which is the stable choice for MC: wings and
    t < T_min reuse the nearest calibrated value instead of extrapolating a
    finite-difference artifact. ``diagnostics`` carries extraction metadata.
    """

    strikes: Array
    maturities: Array
    local_vol: Array  # shape (n_t, n_k)
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        s = np.asarray(self.strikes, dtype=float).ravel()
        t = np.asarray(self.maturities, dtype=float).ravel()
        v = np.asarray(self.local_vol, dtype=float)
        if s.size < 3 or t.size < 2:
            raise ValueError("LocalVolSurface needs >= 3 strikes and >= 2 maturities")
        if not np.isfinite(s).all() or (s <= 0.0).any() or not (np.diff(s) > 0.0).all():
            raise ValueError("strikes must be finite, > 0 and strictly increasing")
        if not np.isfinite(t).all() or (t <= 0.0).any() or not (np.diff(t) > 0.0).all():
            raise ValueError("maturities must be finite, > 0 and strictly increasing")
        if v.shape != (t.size, s.size):
            raise ValueError(f"local_vol shape {v.shape} != ({t.size}, {s.size})")
        if not np.isfinite(v).all() or (v <= 0.0).any():
            raise ValueError("local_vol must be finite and > 0")
        self.strikes = s
        self.maturities = t
        self.local_vol = v
        self._lnk = np.log(s)
        self._interp = RegularGridInterpolator(
            (t, self._lnk), v, method="linear", bounds_error=False, fill_value=None
        )

    def vol(self, s: Array | float, t: float) -> Array:
        """sigma_L(S, t) broadcast over S (scalar t). Always returns an array."""
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        if not np.isfinite(s_arr).all() or (s_arr <= 0.0).any():
            raise ValueError("spot values must be finite and > 0")
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("t must be finite and >= 0")
        tt = min(max(float(t), float(self.maturities[0])), float(self.maturities[-1]))
        lns = np.clip(np.log(s_arr), self._lnk[0], self._lnk[-1])
        pts = np.column_stack((np.full(lns.shape, tt), lns))
        return np.asarray(self._interp(pts), dtype=float).ravel()

    def dvol_dlns(self, s: Array, t: float, h: float = 0.02) -> Array:
        """d sigma_L / d lnS by central difference (Milstein correction term)."""
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("h must be finite and > 0")
        s_arr = np.asarray(s, dtype=float)
        return (self.vol(s_arr * exp(h), t) - self.vol(s_arr * exp(-h), t)) / (2.0 * h)


def dupire_local_vol(
    strikes: Array,
    maturities: Array,
    call_prices: Array,
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    lambda_k: float = 0.5,
    lambda_t: float = 0.0,
    gamma_floor: float = 1e-10,
    var_floor: float = _VAR_FLOOR_DEFAULT,
    var_cap: float = _VAR_CAP_DEFAULT,
    max_degenerate_frac: float = 0.02,
    arb_tol: float = 1e-6,
) -> LocalVolSurface:
    """Dupire (1994) local-vol extraction from a discrete call surface C(K,T).

    ``call_prices`` has shape ``(len(maturities), len(strikes))`` (row = tenor).
    Inputs must pass no-arbitrage screens within ``arb_tol`` (absolute price
    units; raise it for noisy/MC-generated surfaces — the Tikhonov filter
    ``lambda_k``/``lambda_t`` handles the noise itself). ``lambda_* = 0`` gives
    raw finite differences. Degenerate butterflies (d2C/dK2 <= gamma_floor)
    beyond ``max_degenerate_frac`` of grid nodes raise; isolated degenerate
    nodes are filled from the nearest valid node in the same tenor row, and the
    variance is clamped to [var_floor, var_cap].
    """
    k = _check_strictly_increasing_positive(strikes, "strikes", 5)
    t = _check_strictly_increasing_positive(maturities, "maturities", 3)
    c = np.asarray(call_prices, dtype=float)
    if c.ndim != 2 or c.shape != (t.size, k.size):
        raise ValueError(f"call_prices shape {c.shape} != ({t.size}, {k.size})")
    if not np.isfinite(c).all():
        raise ValueError("call_prices must be finite")
    spot = float(spot)
    r = float(r)
    q = float(q)
    if not np.isfinite([spot, r, q]).all() or spot <= 0.0:
        raise ValueError("spot must be finite and > 0; r, q finite")
    if lambda_k < 0.0 or lambda_t < 0.0:
        raise ValueError("lambda_k and lambda_t must be >= 0")
    if gamma_floor <= 0.0 or not np.isfinite(gamma_floor):
        raise ValueError("gamma_floor must be finite and > 0")
    if var_floor <= 0.0 or var_cap <= var_floor:
        raise ValueError("need 0 < var_floor < var_cap")
    if not 0.0 <= max_degenerate_frac <= 1.0:
        raise ValueError("max_degenerate_frac must be in [0, 1]")
    if not np.isfinite(arb_tol) or arb_tol < 0.0:
        raise ValueError("arb_tol must be finite and >= 0")

    df_r = np.exp(-r * t)[:, None]
    df_q = np.exp(-q * t)[:, None]
    lower = np.maximum(spot * df_q - k[None, :] * df_r, 0.0)
    upper = spot * df_q * np.ones_like(k)[None, :]

    def _viol(mask: NDArray[np.bool_]) -> int:
        return int(np.count_nonzero(mask))

    n_viol = [
        ("below intrinsic lower bound", _viol(c < lower - arb_tol)),
        ("above upper bound", _viol(c > upper + arb_tol)),
        ("negative prices", _viol(c < -arb_tol)),
        ("increasing in K", _viol(np.diff(c, axis=1) > arb_tol)),
        ("non-convex in K (butterfly)", _viol(np.diff(c, n=2, axis=1) < -arb_tol)),
        (
            "calendar (e^{rT} C decreasing)",
            _viol(np.diff(c * np.exp(r * t)[:, None], axis=0) < -arb_tol),
        ),
    ]
    bad = [f"{n} at {m} nodes" for n, m in n_viol if m > 0]
    if bad:
        raise ValueError("call surface fails no-arbitrage screen: " + "; ".join(bad))

    smoothed, rms_delta = _smooth_tikhonov(c, lambda_t, lambda_k)
    # Dupire in log-strike x = lnK (better conditioned than raw K: K C_K = C_x
    # and K^2 C_KK = C_xx - C_x, so the K^2 factor never inflates the stencil):
    #   sigma_L^2 = (C_T + (r-q) C_x + q C) / (0.5 (C_xx - C_x))
    x = np.log(k)
    c_t = np.gradient(smoothed, t, axis=0, edge_order=2)
    c_x = np.gradient(smoothed, x, axis=1, edge_order=2)
    c_xx = _second_derivative_k(smoothed, x)

    numer = c_t + (r - q) * c_x + q * smoothed
    gamma_k2 = c_xx - c_x  # == K^2 d2C/dK2
    good = (gamma_k2 > gamma_floor) & np.isfinite(numer) & np.isfinite(gamma_k2)
    degenerate_frac = float(1.0 - good.mean())
    if degenerate_frac > max_degenerate_frac:
        raise ValueError(
            f"degenerate butterfly density (K^2 d2C/dK2 <= {gamma_floor:g}) at "
            f"{degenerate_frac:.1%} of nodes > max_degenerate_frac"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        var_raw = numer / (0.5 * gamma_k2)
    var_filled = _ffill_bad(var_raw, good)
    if not np.isfinite(var_filled).all():
        raise ValueError("Dupire variance is non-finite after degenerate-node fill")
    var_clipped = np.clip(var_filled, var_floor, var_cap)
    local_vol = np.sqrt(var_clipped)

    diagnostics: dict[str, float] = {
        "lambda_k": float(lambda_k),
        "lambda_t": float(lambda_t),
        "tikhonov_rms_delta": rms_delta,
        "gamma_min": float(gamma_k2.min()),
        "degenerate_frac": degenerate_frac,
        "frac_clamped_low": float((var_filled < var_floor).mean()),
        "frac_clamped_high": float((var_filled > var_cap).mean()),
        "var_min": float(var_clipped.min()),
        "var_max": float(var_clipped.max()),
        "synthetic_safe": 1.0,  # extraction is deterministic; labels live in results
    }
    return LocalVolSurface(k, t, local_vol, diagnostics)


def _price_calls(S_term: Array, strikes: Array, T: float, r: float) -> tuple[Array, Array]:
    """Discounted European call prices + standard errors from terminal spots."""
    pay = np.maximum(S_term[:, None] - strikes[None, :], 0.0)
    disc = exp(-r * T)
    n = S_term.size
    prices = disc * pay.mean(axis=0)
    stderr = disc * pay.std(axis=0, ddof=1) / np.sqrt(n)
    return prices, stderr


def mc_implied_vols(
    prices: Array,
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    strikes: Array,
    maturities: Array,
) -> Array:
    """Invert call prices to Black implied vols (composes options.implied_vol).

    Dividends enter through the escrowed-spot map S -> S e^{-qT}, which turns
    the Black-Scholes inversion into exact Black-76 on the forward.
    """
    p = np.asarray(prices, dtype=float)
    k = _check_1d(strikes, "strikes", 1)
    t = _check_1d(maturities, "maturities", 1)
    if p.ndim != 2 or p.shape != (t.size, k.size):
        raise ValueError(f"prices shape {p.shape} != ({t.size}, {k.size})")
    if not np.isfinite(p).all() or (p < 0.0).any():
        raise ValueError("prices must be finite and >= 0")
    spot = float(spot)
    if not np.isfinite([spot, r, q]).all() or spot <= 0.0:
        raise ValueError("spot must be finite and > 0; r, q finite")
    out = np.empty_like(p)
    for i in range(t.size):
        s_eff = spot * exp(-q * float(t[i]))
        for j in range(k.size):
            try:
                out[i, j] = implied_vol(float(p[i, j]), s_eff, float(k[j]), float(t[i]), r)
            except ValueError as exc:
                raise ValueError(f"implied-vol inversion failed at K={k[j]:g}, T={t[i]:g}") from exc
    return out


def local_vol_mc(
    surface: LocalVolSurface,
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    strikes: Array,
    seed: int,
    n_paths: int = 20_000,
    max_dt: float = 1.0 / 12.0,
    scheme: str = "euler",
    antithetic: bool = True,
    stream_id: int = _STREAM_LOCAL_VOL_MC,
) -> dict[str, object]:
    """Monte Carlo under the extracted sigma_L(S,t): dS = (r-q)S dt + sigma_L S dW.

    Log-Euler (``scheme="euler"``, vol frozen at the step start) or Milstein
    (adds 0.5 sigma (dsigma/dlnS) dt (Z^2 - 1)). Time grid refines the surface
    maturities to steps <= ``max_dt``; spots are recorded exactly at maturity
    nodes. Returns a dict with prices/stderr of shape ``(n_t, n_k)``.
    """
    if not isinstance(surface, LocalVolSurface):
        raise ValueError("surface must be a LocalVolSurface")
    scheme_l = str(scheme).lower()
    if scheme_l not in ("euler", "milstein"):
        raise ValueError("scheme must be 'euler' or 'milstein'")
    k = _check_strictly_increasing_positive(strikes, "strikes", 1)
    spot = float(spot)
    r = float(r)
    q = float(q)
    if not np.isfinite([spot, r, q]).all() or spot <= 0.0:
        raise ValueError("spot must be finite and > 0; r, q finite")
    if stream_id < USER_STREAM_ID_MIN:
        raise ValueError(f"stream_id must be >= USER_STREAM_ID_MIN ({USER_STREAM_ID_MIN})")
    _check_seed(seed)
    _check_paths(n_paths, antithetic)

    grid, record = _build_time_grid(surface.maturities, max_dt)
    n_steps = grid.size - 1
    z = _antithetic_normals(seed, n_paths, n_steps, stream_id=stream_id, antithetic=antithetic)
    record_at = {idx: j for j, idx in enumerate(record)}

    x = np.full(n_paths, log(spot))
    s_at = np.empty((len(record), n_paths))
    for i in range(n_steps):
        t_cur = float(grid[i])
        dt = float(grid[i + 1] - grid[i])
        sdt = np.sqrt(dt)
        sig = surface.vol(np.exp(x), t_cur)
        if scheme_l == "milstein":
            dsig = surface.dvol_dlns(np.exp(x), t_cur)
            zz = z[:, i]
            x += (
                (r - q - 0.5 * sig * sig) * dt
                + sig * sdt * zz
                + 0.5 * sig * dsig * dt * (zz * zz - 1.0)
            )
        else:
            x += (r - q - 0.5 * sig * sig) * dt + sig * sdt * z[:, i]
        j = record_at.get(i + 1)
        if j is not None:
            s_at[j] = np.exp(x)

    n_t, n_k = len(record), k.size
    prices = np.empty((n_t, n_k))
    stderr = np.empty((n_t, n_k))
    for j, T in enumerate(surface.maturities.tolist()):
        prices[j], stderr[j] = _price_calls(s_at[j], k, T, r)
    return {
        "prices": prices,
        "stderr": stderr,
        "strikes": k,
        "maturities": surface.maturities,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "scheme": scheme_l,
        "seed": seed,
        "antithetic": bool(antithetic),
        "synthetic": True,
        "label": SYNTHETIC_LABEL,
    }


def heston_mc(
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    times: Array,
    n_paths: int,
    seed: int,
    leverage: Callable[[Array, float], Array] | None = None,
    antithetic: bool = True,
    vol_floor: float = 1e-10,
    stream_id: int = _STREAM_HESTON,
) -> dict[str, object]:
    """Inline log-Euler Heston simulator with optional SLV leverage L(S,t).

    dS = (r-q)S dt + L(S,t) sqrt(v) S dW_S;
    dv = kappa(theta - v)dt + xi sqrt(v) dW_v;  corr = rho.

    ``xi = 0`` is allowed (deterministic variance => pure local-vol limit).
    Returns full paths ``S`` and ``v`` of shape ``(n_paths, len(times))``.
    """
    tm = _check_1d(times, "times", 2)
    if abs(tm[0]) > 1e-12:
        raise ValueError("times must start at 0")
    if not (np.diff(tm) > 0.0).all():
        raise ValueError("times must be strictly increasing")
    kappa = float(kappa)
    theta = float(theta)
    xi = float(xi)
    rho = float(rho)
    v0 = float(v0)
    spot = float(spot)
    if not np.isfinite([kappa, theta, xi, rho, v0, spot, r, q]).all():
        raise ValueError("all Heston params must be finite")
    if kappa <= 0.0 or theta <= 0.0:
        raise ValueError("kappa and theta must be > 0")
    if xi < 0.0:
        raise ValueError("xi must be >= 0 (xi = 0 is the deterministic-vol degenerate case)")
    if abs(rho) >= 1.0:
        raise ValueError("rho must be in (-1, 1)")
    if v0 <= 0.0 or spot <= 0.0:
        raise ValueError("v0 and spot must be > 0")
    if not np.isfinite(vol_floor) or vol_floor <= 0.0:
        raise ValueError("vol_floor must be finite and > 0")
    if stream_id < USER_STREAM_ID_MIN:
        raise ValueError(f"stream_id must be >= USER_STREAM_ID_MIN ({USER_STREAM_ID_MIN})")
    if leverage is not None and not callable(leverage):
        raise ValueError("leverage must be None or callable (S_array, t) -> Array")
    _check_seed(seed)
    _check_paths(n_paths, antithetic)

    n_steps = tm.size - 1
    idx = np.arange(n_paths // 2 if antithetic else n_paths, dtype=np.int64)
    z1 = philox_normals(seed, idx, n_steps, stream_id=stream_id)
    z2 = philox_normals(seed, idx, n_steps, stream_id=stream_id + 1)
    if antithetic:
        z1 = np.concatenate([z1, -z1], axis=0)
        z2 = np.concatenate([z2, -z2], axis=0)
    w2 = rho * z1 + np.sqrt(1.0 - rho * rho) * z2

    s_paths = np.empty((n_paths, tm.size))
    v_paths = np.empty((n_paths, tm.size))
    x = np.full(n_paths, log(spot))
    v_cur = np.full(n_paths, v0)
    s_paths[:, 0] = spot
    v_paths[:, 0] = v0
    for i in range(n_steps):
        dt = float(tm[i + 1] - tm[i])
        sdt = np.sqrt(dt)
        lev = (
            np.ones(n_paths)
            if leverage is None
            else np.asarray(leverage(np.exp(x), float(tm[i])), dtype=float)
        )
        a = lev * np.sqrt(np.maximum(v_cur, vol_floor))
        x += (r - q - 0.5 * a * a) * dt + a * sdt * z1[:, i]
        if xi > 0.0:
            v_cur = np.maximum(
                v_cur
                + kappa * (theta - v_cur) * dt
                + xi * np.sqrt(np.maximum(v_cur, vol_floor)) * sdt * w2[:, i],
                vol_floor,
            )
        s_paths[:, i + 1] = np.exp(x)
        v_paths[:, i + 1] = v_cur
    return {
        "S": s_paths,
        "v": v_paths,
        "times": tm,
        "n_paths": n_paths,
        "seed": seed,
        "leverage": leverage is not None,
        "synthetic": True,
        "label": SYNTHETIC_LABEL,
    }


def make_leverage_interpolator(
    time_nodes: Array, bin_centers: Array, leverage: Array
) -> Callable[[Array, float], Array]:
    """Bilinear (t, lnS) interpolator for a binned leverage grid -> L(S, t).

    ``leverage`` has shape ``(len(time_nodes), len(bin_centers))``; queries are
    clamped to the grid box (constant extrapolation), the stable choice for MC.
    """
    tn = _check_strictly_increasing_positive(time_nodes, "time_nodes", 2)
    bc = _check_strictly_increasing_positive(bin_centers, "bin_centers", 2)
    lev = np.asarray(leverage, dtype=float)
    if lev.shape != (tn.size, bc.size):
        raise ValueError(f"leverage shape {lev.shape} != ({tn.size}, {bc.size})")
    if not np.isfinite(lev).all() or (lev <= 0.0).any():
        raise ValueError("leverage must be finite and > 0")
    ln_b = np.log(bc)
    interp = RegularGridInterpolator(
        (tn, ln_b), lev, method="linear", bounds_error=False, fill_value=None
    )
    t_lo, t_hi = float(tn[0]), float(tn[-1])
    l_lo, l_hi = float(ln_b[0]), float(ln_b[-1])

    def lev_fn(s: Array, t: float) -> Array:
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        if not np.isfinite(s_arr).all() or (s_arr <= 0.0).any():
            raise ValueError("leverage query spots must be finite and > 0")
        tt = min(max(float(t), t_lo), t_hi)
        lns = np.clip(np.log(s_arr), l_lo, l_hi)
        pts = np.column_stack((np.full(lns.shape, tt), lns))
        return np.asarray(interp(pts), dtype=float).ravel()

    return lev_fn


def slv_price_calls(
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    T: float,
    strikes: Array,
    n_paths: int = 8_000,
    n_steps: int = 40,
    seed: int = 0,
    leverage: Callable[[Array, float], Array] | None = None,
    stream_id: int = _STREAM_SLV_PRICE,
) -> dict[str, object]:
    """Price European calls by SLV/Heston MC over [0, T] on a uniform grid.

    ``leverage=None`` prices under the pure stochastic-vol model (L == 1).
    """
    T = float(T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and > 0")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError("n_steps must be an int >= 2")
    k = _check_strictly_increasing_positive(strikes, "strikes", 1)
    times = np.linspace(0.0, T, n_steps + 1)
    sim = heston_mc(
        spot=spot,
        r=r,
        q=q,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        v0=v0,
        times=times,
        n_paths=n_paths,
        seed=seed,
        leverage=leverage,
        stream_id=stream_id,
    )
    s_term = np.asarray(sim["S"], dtype=float)[:, -1]
    prices, stderr = _price_calls(s_term, k, T, r)
    return {
        "prices": prices,
        "stderr": stderr,
        "strikes": k,
        "T": T,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "seed": seed,
        "synthetic": True,
        "label": SYNTHETIC_LABEL,
    }


def _cond_mean_v(
    s_paths: Array, v_paths: Array, node_idx: Array, edges: Array, n_bins: int
) -> tuple[Array, Array]:
    """Per (time node, log-S bin) mean of v_t and path counts."""
    n_nodes = node_idx.size
    means = np.full((n_nodes, n_bins), np.nan)
    counts = np.zeros((n_nodes, n_bins))
    for a, i in enumerate(node_idx.tolist()):
        i = int(i)
        b = np.clip(np.digitize(np.log(s_paths[:, i]), edges) - 1, 0, n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins).astype(float)
        vsum = np.bincount(b, weights=v_paths[:, i], minlength=n_bins)
        counts[a] = cnt
        ok = cnt > 0.0
        means[a, ok] = vsum[ok] / cnt[ok]
    return means, counts


def slv_leverage_function(
    *,
    spot: float,
    r: float,
    q: float = 0.0,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    T: float,
    calib_strikes: Array,
    target_implied_vols: Array,
    n_steps: int = 40,
    n_paths: int = 8_000,
    n_iter: int = 3,
    seed: int = 0,
    n_s_bins: int = 14,
    n_time_nodes: int = 9,
    damping: float = 1.0,
    min_bin_count: int = 30,
    l2_min: float = 0.02,
    l2_max: float = 50.0,
    ratio_cap: float = 4.0,
    bin_span_mult: float = 3.0,
    stream_id: int = _STREAM_SLV_CALIB,
    target_kappa: float | None = None,
    target_theta: float | None = None,
    target_xi: float | None = None,
    target_rho: float | None = None,
    target_v0: float | None = None,
) -> dict[str, object]:
    """van der Stoep, Grzelak & Oosterlee (2014) mixing-MC leverage calibration.

    Iteratively updates a piecewise-constant leverage L(S,t) on (lnS, t) bins so
    the SLV model dS = (r-q)S dt + L sqrt(v_t) S dW reproduces the conditional
    variance of the target Heston model:

        L2_new(s,t) = L2_old(s,t) * E_hat_target[V_t | S_t in bin]
                                   / E_hat_SLV[V_t | S_t in bin]

    The SLV driver uses (kappa, theta, xi, rho, v0); the target conditional
    moments come from the same parameters unless ``target_*`` overrides are
    given. The scheme's contraction is established in the paper's matched
    setting (target == driver Heston, whose exact fixed point is L == 1);
    with a strongly mismatched target the fixed point may not be representable
    and the calibration raises once bins saturate the leverage clamps
    (fail-closed). With identical target/driver parameters, L == 1 is the
    exact fixed point, which serves as a degenerate consistency check.

    Bins with fewer than ``min_bin_count`` paths in either simulation keep their
    previous value (ratio 1); ratios are capped at ``ratio_cap`` and L2 clamped
    to [``l2_min``, ``l2_max``]. ``damping`` in (0, 1] damps the multiplicative
    update. Iteration 0 scores the un-levered driver (pure Heston MC) against
    ``target_implied_vols`` (caller supplies them, e.g. from the repo's
    closed-form Heston CF pricer), so convergence is measurable. If more than
    90% of bins are under-populated the calibration raises (fail-closed).

    Returns a dict with the leverage grid ``leverage`` (vol units, sqrt of the
    L2 grid) on (``time_nodes``, ``bin_centers``), per-iteration diagnostics
    (``implied_rmse``, ``l2_delta_rms``, ...), and the final SLV implied vols.
    SYNTHETIC: target and diagnostics are all model-generated.
    """
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be an int >= 1")
    if not np.isfinite([spot, r, q, kappa, theta, xi, rho, v0]).all():
        raise ValueError("SLV driver params must be finite")
    if kappa <= 0.0 or theta <= 0.0 or v0 <= 0.0 or spot <= 0.0:
        raise ValueError("kappa, theta, v0, spot must be > 0")
    if xi < 0.0 or abs(rho) >= 1.0:
        raise ValueError("need xi >= 0 and |rho| < 1")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 4:
        raise ValueError("n_steps must be an int >= 4")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    if isinstance(n_s_bins, bool) or not isinstance(n_s_bins, int) or n_s_bins < 3:
        raise ValueError("n_s_bins must be an int >= 3")
    if (
        isinstance(n_time_nodes, bool)
        or not isinstance(n_time_nodes, int)
        or not 2 <= n_time_nodes <= n_steps
    ):
        raise ValueError("n_time_nodes must be an int in [2, n_steps]")
    if isinstance(min_bin_count, bool) or not isinstance(min_bin_count, int) or min_bin_count < 2:
        raise ValueError("min_bin_count must be an int >= 2")
    if not 0.0 < l2_min < 1.0 < l2_max:
        raise ValueError("need 0 < l2_min < 1 < l2_max")
    if not np.isfinite(ratio_cap) or ratio_cap <= 1.0:
        raise ValueError("ratio_cap must be finite and > 1")
    if not np.isfinite(bin_span_mult) or bin_span_mult <= 0.0:
        raise ValueError("bin_span_mult must be finite and > 0")
    T = float(T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and > 0")
    ks = _check_strictly_increasing_positive(calib_strikes, "calib_strikes", 2)
    tv = np.asarray(target_implied_vols, dtype=float).ravel()
    if tv.shape != ks.shape or not np.isfinite(tv).all() or (tv <= 0.0).any():
        raise ValueError("target_implied_vols must match calib_strikes, finite and > 0")

    times = np.linspace(0.0, T, n_steps + 1)
    node_idx = np.unique(np.round(np.linspace(1, n_steps, n_time_nodes)).astype(int))
    t_nodes = times[node_idx]
    half_span = bin_span_mult * float(np.sqrt(max(theta, v0)) * T)
    edges = np.linspace(log(spot) - half_span, log(spot) + half_span, n_s_bins + 1)
    centers = np.exp(0.5 * (edges[:-1] + edges[1:]))

    t_kappa = kappa if target_kappa is None else float(target_kappa)
    t_theta = theta if target_theta is None else float(target_theta)
    t_xi = xi if target_xi is None else float(target_xi)
    t_rho = rho if target_rho is None else float(target_rho)
    t_v0 = v0 if target_v0 is None else float(target_v0)
    if not np.isfinite([t_kappa, t_theta, t_xi, t_rho, t_v0]).all():
        raise ValueError("target_* params must be finite")
    if t_kappa <= 0.0 or t_theta <= 0.0 or t_v0 <= 0.0 or t_xi < 0.0 or abs(t_rho) >= 1.0:
        raise ValueError("target_* params must define a valid Heston model")

    # Target conditional moments under the (possibly different) target Heston.
    target = heston_mc(
        spot=spot,
        r=r,
        q=q,
        kappa=t_kappa,
        theta=t_theta,
        xi=t_xi,
        rho=t_rho,
        v0=t_v0,
        times=times,
        n_paths=n_paths,
        seed=seed * 1000 + 101,
        stream_id=stream_id,
    )
    tgt_means, tgt_counts = _cond_mean_v(
        np.asarray(target["S"]), np.asarray(target["v"]), node_idx, edges, n_s_bins
    )

    l2 = np.ones((node_idx.size, n_s_bins))
    iterations: list[dict[str, float]] = []
    slv_iv = np.full(ks.size, np.nan)

    def _price_and_score(lev_fn: Callable[[Array, float], Array] | None, price_seed: int) -> Array:
        priced = slv_price_calls(
            spot=spot,
            r=r,
            q=q,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
            T=T,
            strikes=ks,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=price_seed,
            leverage=lev_fn,
            stream_id=stream_id + 4,
        )
        return mc_implied_vols(
            np.asarray(priced["prices"], dtype=float).reshape(1, -1),
            spot=spot,
            r=r,
            q=q,
            strikes=ks,
            maturities=np.array([T]),
        ).ravel()

    # Iteration 0 baseline: score the un-levered (pure Heston MC) starting point
    # so convergence of the mixing scheme is measurable.
    slv_iv = _price_and_score(None, seed * 1000 + 505)
    rmse0 = float(np.sqrt(np.mean((slv_iv - tv) ** 2)))
    iterations.append(
        {
            "iter": 0.0,
            "implied_rmse": rmse0,
            "implied_max_abs": float(np.max(np.abs(slv_iv - tv))),
            "l2_delta_rms": 0.0,
            "l2_mean": 1.0,
            "l2_min": 1.0,
            "l2_max": 1.0,
            "bins_updated_frac": 0.0,
        }
    )
    for it in range(n_iter):
        lev_fn = make_leverage_interpolator(t_nodes, centers, np.sqrt(l2))
        sim = heston_mc(
            spot=spot,
            r=r,
            q=q,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
            times=times,
            n_paths=n_paths,
            seed=seed * 1000 + 202 + it,
            leverage=lev_fn,
            stream_id=stream_id + 2,
        )
        slv_means, slv_counts = _cond_mean_v(
            np.asarray(sim["S"]), np.asarray(sim["v"]), node_idx, edges, n_s_bins
        )
        valid = (
            (tgt_counts >= min_bin_count)
            & (slv_counts >= min_bin_count)
            & np.isfinite(tgt_means)
            & np.isfinite(slv_means)
            & (slv_means > 0.0)
        )
        if float(valid.mean()) < 0.1:
            raise ValueError(
                "SLV mixing MC: > 90% of (S,t) bins under-populated; "
                "increase n_paths/n_s_bins resolution mismatch or T"
            )
        safe_slv = np.where(slv_means > 0.0, slv_means, 1.0)
        ratio = np.where(valid, tgt_means / safe_slv, 1.0)
        ratio = np.clip(ratio, 1.0 / ratio_cap, ratio_cap)
        l2_new = np.clip(l2 * ratio**damping, l2_min, l2_max)
        saturated = float(np.mean((l2_new <= l2_min + 1e-12) | (l2_new >= l2_max - 1e-12)))
        if saturated > 0.25:
            raise ValueError(
                "SLV mixing MC diverged: > 25% of bins pinned at the leverage clamps; "
                "the target conditional variance is not representable by this "
                "driver + leverage (the scheme is contractive in the matched "
                "van der Stoep et al. 2014 setting)"
            )
        l2_delta = float(np.sqrt(np.mean((l2_new - l2) ** 2)))
        l2 = l2_new

        slv_iv = _price_and_score(
            make_leverage_interpolator(t_nodes, centers, np.sqrt(l2)), seed * 1000 + 505 + it
        )
        rmse = float(np.sqrt(np.mean((slv_iv - tv) ** 2)))
        iterations.append(
            {
                "iter": float(it + 1),
                "implied_rmse": rmse,
                "implied_max_abs": float(np.max(np.abs(slv_iv - tv))),
                "l2_delta_rms": l2_delta,
                "l2_mean": float(l2.mean()),
                "l2_min": float(l2.min()),
                "l2_max": float(l2.max()),
                "bins_updated_frac": float(valid.mean()),
            }
        )

    return {
        "leverage": np.sqrt(l2),
        "leverage_sq": l2,
        "bin_centers": centers,
        "bin_edges": edges,
        "time_nodes": t_nodes,
        "iterations": iterations,
        "final_implied_vols": slv_iv,
        "target_implied_vols": tv,
        "final_implied_rmse": float(iterations[-1]["implied_rmse"]),
        "n_paths": n_paths,
        "n_steps": n_steps,
        "n_iter": n_iter,
        "seed": seed,
        "synthetic": True,
        "label": SYNTHETIC_LABEL,
    }
