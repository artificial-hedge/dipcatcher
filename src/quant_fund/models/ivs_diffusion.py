"""AD-Seq-Vol: conditional diffusion for dynamic implied-volatility surfaces.

Han, Y., Zhang, J.Y., Torres, M., Acero, F. & Xu, R. (2026), "Diffusion models
for dynamic volatility surface generation and data-driven hedging",
arXiv:2609.13402 (v3; citation verified against the arXiv abstract page on
2026-09-30 -- title, authors, and algorithm match this lane spec). The paper
jointly learns the conditional evolution of the underlying return and the
high-dimensional implied-volatility surface (IVS), then generates ADAPTED
multi-period scenarios by sequentially updating the realized market history
("Adaptive Sequential Diffusion"). AD-Seq-Vol-FT adds post-training penalties
for violations of static no-arbitrage conditions, cutting violations to ~zero
on SPX option data; generated scenarios then drive an optimization-based
hedge whose tracking error stays near zero with shrunk tail risk.

Implemented here (research-grade reduction, CPU/tiny budgets):

- ``IVSGrid``: a fixed (log-moneyness, maturity) mesh; surfaces are implied
  vols of shape ``(n, n_k, n_tau)``, fail-closed on shape/finite errors.
- Conditional DDPM (Ho, J., Jain, A. & Abbeel, P. 2020, "Denoising Diffusion
  Probabilistic Models", NeurIPS 33:6840-6855, arXiv:2006.11239) over the
  joint next-step target ``y = (r_{t+1}, V_{t+1} - V_t)`` given a flattened
  history window ``X_t = (r, V)_{t-H+1..t}``: ``x_t = sqrt(abar_t) y_0 +
  sqrt(1 - abar_t) eps``, ``eps_theta(x_t, onehot(t), X_t)`` an MLP, reverse
  sampling through the standard posterior coefficients
  ``mu = c0 x0_hat + c1 x_t``, ``var = beta_tilde`` composed from
  ``models/diffusion_forecaster.noise_schedule`` (its DiffPTS LSNM posterior
  specializes to DDPM's at ``f_phi = 0``; the ``c2`` coefficient is unused).
  Cosine schedule default (Nichol & Dhariwal 2021, ICML, arXiv:2102.09672).
- Sequential adaptive sampling: :meth:`IVSDiffusion.generate_scenarios`
  re-embeds the latest generated ``(r, V)`` into the window at every step --
  the paper's defining feature, kept exactly.
- Static no-arbitrage report/penalty on a generated surface ``sigma(k,tau)``:
  (i) NONNEGATIVITY ``sigma > 0``; (ii) CALENDAR -- total variance
  ``w(k, tau) = sigma^2 tau`` non-decreasing in ``tau`` per ``k`` (no calendar
  arbitrage, e.g. Gatheral, J. 2006, *The Volatility Surface*, Wiley, §2);
  (iii) BUTTERFLY -- call prices convex in strike, measured as the second
  divided difference of ``C(K, tau) = BS(S=1, K=e^k, T=tau, sigma)`` on the
  non-uniform strike mesh (convexity of the piecewise-linear price
  interpolant). ``arb_violation_report`` measures rates + magnitudes in
  numpy; ``finetune_no_arb`` penalizes differentiable surrogates of the same
  three conditions on UNROLLED reverse-diffusion samples (reparameterized
  fresh noises, the standard eps-MSE kept as a stability anchor) -- the
  paper's post-training penalty evaluated on what generation actually
  produces, faithful to arXiv:2609.13402's finetune phase.
- Optimization-based hedge evaluation (:func:`evaluate_hedge`): the paper's
  data-driven hedge is solved in closed form -- a ridge/OLS position in the
  underlying minimizing the squared tracking error of a short grid option
  under the generated one-step scenario ensemble. Reported diagnostics are
  tracking RMSE / mean absolute residual / ES_tail of the hedging loss --
  proper-risk diagnostics only; any simulator P&L stays ``sim_internal_*``
  and never headlines (AGENTS.md honesty contract).
- ``historical_resample_scenarios``: block-resample baseline standing in for
  the paper's data-driven benchmarks (it reports a GAN and classical
  benchmarks; a faithful GAN trainer is out of scope for this lane -- the
  stationary block resample and the unconditional Gaussian baseline bracket
  the same "no conditional structure" comparison. Documented deviation).

Composition (imported, not reimplemented): ``_torch`` lazy-import pattern
mirrors ``models/deep_hedging.py``; the noise schedule / training-loop
conventions (seeded numpy rng driving noise draws, ``torch.manual_seed`` for
init, CPU single thread, cosine LR, loss-before-update curves) follow
``models/diffusion_forecaster.py`` and its ``NoiseSchedule``/``noise_schedule``
are reused directly; call prices come from
``quant_models.black_scholes.bs_price`` (vectorized BSM); scenario quality is
scored with ``metrics/energy_score.energy_score`` (Gneiting & Raftery 2007,
strictly proper). ``models/iv_approx`` inversion primitives are NOT needed:
all arbitrage checks run in forward IV->price space.

Honesty: every result is computed on SYNTHETIC seeded streams (SVI-driven
surfaces with a known law) -- correctness evidence for the algorithm, never
market evidence. Metrics are proper scores (energy score) and hedging-error
diagnostics; no Sharpe/Sortino/Calmar/P&L/NAV headline, no live-trading
claims. Deterministic given ``seed`` on CPU (GPU determinism not claimed).
Torch is the optional ``nn`` extra imported lazily via :func:`_torch`, so the
module imports cleanly without it and only training/finetuning raises
``ImportError``; ALL sampling/evaluation runs in numpy on extracted float64
parameters. Fail-closed: invalid grids/hyperparameters, non-finite or
mismatched inputs, too-few samples, unfitted use, non-finite training loss,
and degenerate windows raise ``ValueError``/``RuntimeError``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.metrics.energy_score import energy_score
from quant_fund.models.diffusion_forecaster import NoiseSchedule, noise_schedule
from quant_fund.quant_models.black_scholes import bs_price

Array = NDArray[np.float64]

__all__ = [
    "ArbReport",
    "IVSDiffusion",
    "IVSFinetuneInfo",
    "IVSFitInfo",
    "IVSGrid",
    "arb_violation_report",
    "bench_ivs_diffusion",
    "build_supervised_pairs",
    "call_price_surface",
    "evaluate_hedge",
    "historical_resample_scenarios",
    "scenario_energy_score",
    "surface_codes",
    "synthetic_ivs_stream",
]

_SCHEDULES = ("cosine", "linear")
_MIN_FIT_STEPS = 12  # need >= window + a few (context, target) pairs to learn from
_TARGET_CLIP_MARGIN = 0.25  # standardized-unit margin on the DDPM x0_hat clip
_SIGMA_FLOOR = 1e-8  # implied-vol floor used only INSIDE price evaluation
_BUTTERFLY_TOL = 1e-9  # divided-difference tolerance (float64 price scale)
_RIDGE_FLOOR = 1e-10  # numerical ridge on the closed-form hedge denominator

_Layers = tuple[tuple[Array, Array], ...]


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "ivs diffusion needs the optional 'nn' extra (torch): uv sync --extra nn"
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


def _check_nonneg(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite; got {value!r}")
    return v


def _check_choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")
    return value


def _require_finite_loss(value: float, epoch: int) -> float:
    """Fail-closed guard on the training loss (mirrors diffusion_forecaster)."""
    if not math.isfinite(value):
        raise ValueError(
            f"IVS diffusion training loss is not finite at epoch {epoch} "
            f"(loss={value!r}); reduce lr or check inputs"
        )
    return value


# ---------------------------------------------------------------------------
# IVS grid and surface validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class IVSGrid:
    """Fixed (log-moneyness, maturity) mesh for implied-vol surfaces.

    ``log_moneyness`` ``k = log(K / S)`` strictly increasing; ``maturities``
    in years, strictly increasing and positive. Surfaces on this grid are
    ``(n, n_k, n_tau)`` arrays of implied volatilities (a single surface may
    be passed as ``(n_k, n_tau)`` and is promoted to ``n = 1``).
    """

    log_moneyness: Array
    maturities: Array

    def __post_init__(self) -> None:
        k = np.asarray(self.log_moneyness, dtype=float).reshape(-1)
        tau = np.asarray(self.maturities, dtype=float).reshape(-1)
        if k.size < 3:
            raise ValueError(f"log_moneyness needs >= 3 grid points (butterfly); got {k.size}")
        if tau.size < 2:
            raise ValueError(f"maturities needs >= 2 tenors (calendar); got {tau.size}")
        if not (bool(np.all(np.isfinite(k))) and bool(np.all(np.isfinite(tau)))):
            raise ValueError("grid coordinates must be finite")
        if not bool(np.all(np.diff(k) > 0.0)):
            raise ValueError("log_moneyness must be strictly increasing")
        if not (bool(np.all(np.diff(tau) > 0.0)) and bool(tau[0] > 0.0)):
            raise ValueError("maturities must be strictly increasing and positive")
        object.__setattr__(self, "log_moneyness", k)
        object.__setattr__(self, "maturities", tau)

    @property
    def n_k(self) -> int:
        return int(self.log_moneyness.size)

    @property
    def n_tau(self) -> int:
        return int(self.maturities.size)

    @property
    def n_points(self) -> int:
        """Flattened surface dimension ``n_k * n_tau`` (row-major, k fastest)."""
        return self.n_k * self.n_tau

    @property
    def strikes(self) -> Array:
        """Strike mesh ``K = exp(k)`` (non-uniform), shape ``(n_k,)``."""
        return np.asarray(np.exp(self.log_moneyness), dtype=float)

    def check_surfaces(self, surfaces: Array, *, positive: bool, name: str) -> Array:
        """Validate surfaces to ``(n, n_k, n_tau)`` float64; fail closed.

        ``positive=True`` (training/observed data) requires strictly positive
        vols; ``positive=False`` (generated data under audit) requires only
        finiteness so violations can be MEASURED, never rejected or clipped.
        """
        arr = np.asarray(surfaces, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(f"{name} must be (n, n_k, n_tau) implied vols; got ndim={arr.ndim}")
        if arr.shape[1:] != (self.n_k, self.n_tau):
            raise ValueError(
                f"{name} surface shape {arr.shape[1:]} does not match grid "
                f"({self.n_k}, {self.n_tau})"
            )
        if arr.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one surface")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"{name} must be finite (NaN/inf rejected)")
        if positive and not bool(np.all(arr > 0.0)):
            raise ValueError(f"{name} entries must be strictly positive implied vols")
        return np.asarray(arr, dtype=float)


def _check_returns(returns: Array, name: str = "returns") -> Array:
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(r))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return r


def _check_stream(returns: Array, surfaces: Array, grid: IVSGrid) -> tuple[Array, Array]:
    """Aligned (returns, surfaces) stream validation; returns must be simple."""
    r = _check_returns(returns)
    v = grid.check_surfaces(surfaces, positive=True, name="surfaces")
    if r.shape[0] != v.shape[0]:
        raise ValueError(
            f"returns and surfaces must have equal length; got {r.shape[0]} vs {v.shape[0]}"
        )
    if r.shape[0] < 2:
        raise ValueError("stream needs at least two time points")
    return r, v


# ---------------------------------------------------------------------------
# static no-arbitrage measurement (numpy)
# ---------------------------------------------------------------------------


def call_price_surface(surfaces: Array, grid: IVSGrid) -> Array:
    """BS forward call prices on the grid: ``C(k, tau)`` with ``S = 1``, ``r = q = 0``.

    ``C = N(d1) - e^k N(d2)``, ``d1 = (-k + w/2) / sqrt(w)``, ``d2 = d1 -
    sqrt(w)``, ``w = sigma^2 tau`` (Black & Scholes 1973). Non-positive vols
    are floored at ``_SIGMA_FLOOR`` INSIDE the pricing map so violation
    magnitude stays measurable instead of raising -- the floor is documented
    and only ever applies to surfaces being audited.
    """
    sig = grid.check_surfaces(surfaces, positive=False, name="surfaces")
    sig = np.maximum(sig, _SIGMA_FLOOR)
    k = grid.log_moneyness[:, None]
    t = grid.maturities[None, :]
    prices = bs_price(
        np.ones((), dtype=float),
        np.exp(k),
        np.broadcast_to(t, (grid.n_k, grid.n_tau)),
        0.0,
        0.0,
        sig,
        "call",
    )
    return np.asarray(prices, dtype=float)


def _butterfly_residuals(prices: Array, strikes: Array) -> Array:
    """Second divided difference of call prices in strike, shape (n, n_k-2, n_tau).

    For a piecewise-linear interpolant of C on the (non-uniform) strike mesh,
    the interpolant is convex iff every second divided difference is >= 0
    (Rockafellar 1970, *Convex Analysis*, §24 -- midpoint convexity of the
    interpolant at mesh resolution). Returns the raw divided differences;
    negative entries are butterfly-arbitrage violations.
    """
    k1, k2, k3 = strikes[:-2], strikes[1:-1], strikes[2:]
    c1, c2, c3 = prices[:, :-2, :], prices[:, 1:-1, :], prices[:, 2:, :]
    dd = (
        c1 / ((k1 - k2) * (k1 - k3))[:, None]
        + c2 / ((k2 - k1) * (k2 - k3))[:, None]
        + c3 / ((k3 - k1) * (k3 - k2))[:, None]
    )
    return np.asarray(2.0 * dd, dtype=float)


@dataclass(frozen=True)
class ArbReport:
    """Static no-arbitrage audit of a batch of IVS snapshots.

    Rates are fractions of the relevant cells/pairs; magnitudes are mean
    violation sizes (in total-variance units for calendar, price-second-
    difference units for butterfly, vol units for nonnegativity) computed over
    violating entries only -- 0.0 when there are none. ``surface_rate`` is the
    fraction of surfaces carrying ANY violation, the paper's headline number
    (arXiv:2609.13402: AD-Seq-Vol-FT cuts it to ~zero).
    """

    nonneg_rate: float
    nonneg_magnitude: float
    calendar_rate: float
    calendar_magnitude: float
    butterfly_rate: float
    butterfly_magnitude: float
    surface_rate: float
    n_surfaces: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}nonneg_rate": self.nonneg_rate,
            f"{prefix}nonneg_magnitude": self.nonneg_magnitude,
            f"{prefix}calendar_rate": self.calendar_rate,
            f"{prefix}calendar_magnitude": self.calendar_magnitude,
            f"{prefix}butterfly_rate": self.butterfly_rate,
            f"{prefix}butterfly_magnitude": self.butterfly_magnitude,
            f"{prefix}surface_rate": self.surface_rate,
        }


def arb_violation_report(surfaces: Array, grid: IVSGrid) -> ArbReport:
    """Measure static no-arbitrage violations of a batch of surfaces.

    (i) NONNEGATIVITY: cells with ``sigma <= 0`` (vols are floored inside the
    pricing map so downstream checks stay defined). (ii) CALENDAR: total
    variance ``w = sigma^2 tau`` must be non-decreasing in ``tau`` for every
    ``k``; violations are pairs ``w_j - w_{j+1} > tol``. (iii) BUTTERFLY:
    negative second divided differences of the call-price surface in strike.
    Fail-closed on shape/non-finite inputs; degenerate violations are
    MEASURED, never rejected.
    """
    sig = grid.check_surfaces(surfaces, positive=False, name="surfaces")
    n = int(sig.shape[0])
    neg = sig <= 0.0
    nonneg_rate = float(np.mean(neg))
    nonneg_mag = float(np.mean(-sig[neg])) if bool(neg.any()) else 0.0

    sig_eff = np.maximum(sig, _SIGMA_FLOOR)
    w = sig_eff * sig_eff * grid.maturities[None, None, :]
    d_w = w[:, :, 1:] - w[:, :, :-1]  # (n, n_k, n_tau-1); must be >= 0
    cal_bad = d_w < -_BUTTERFLY_TOL
    calendar_rate = float(np.mean(cal_bad))
    calendar_mag = float(np.mean(-d_w[cal_bad])) if bool(cal_bad.any()) else 0.0

    prices = call_price_surface(sig, grid)
    dd = _butterfly_residuals(prices, grid.strikes)
    bfly_bad = dd < -_BUTTERFLY_TOL
    butterfly_rate = float(np.mean(bfly_bad))
    butterfly_mag = float(np.mean(-dd[bfly_bad])) if bool(bfly_bad.any()) else 0.0

    any_bad = neg.any(axis=(1, 2)) | cal_bad.any(axis=(1, 2)) | bfly_bad.any(axis=(1, 2))
    return ArbReport(
        nonneg_rate=nonneg_rate,
        nonneg_magnitude=nonneg_mag,
        calendar_rate=calendar_rate,
        calendar_magnitude=calendar_mag,
        butterfly_rate=butterfly_rate,
        butterfly_magnitude=butterfly_mag,
        surface_rate=float(np.mean(any_bad)),
        n_surfaces=n,
    )


# ---------------------------------------------------------------------------
# SYNTHETIC surface stream with a known law (correctness fixture only)
# ---------------------------------------------------------------------------


def _svi_total_variance(k: Array, a: float, b: float, rho: float, m: float, s: float) -> Array:
    """Raw-SVI total variance ``w(k) = a + b (rho (k - m) + sqrt((k-m)^2 + s^2))``.

    Gatheral, J. & Jacquier, A. (2014), "Arbitrage-free SVI volatility
    surfaces", Quantitative Finance 14(1), 59-71, arXiv:1204.0646.
    """
    return np.asarray(a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s)), dtype=float)


def synthetic_ivs_stream(
    n_steps: int,
    grid: IVSGrid,
    *,
    seed: int = 0,
    jitter: float = 0.0,
    atm_level: float = 0.18,
    vol_of_vol: float = 0.35,
    mean_reversion: float = 0.06,
    momentum: float = 0.8,
    leverage: float = -0.55,
    dt: float = 1.0 / 252.0,
) -> tuple[Array, Array]:
    """Seeded SYNTHETIC joint (return, IVS) stream with a known law.

    Latent volatility level follows an OU process ``u_t``; each maturity slice
    carries a time-varying raw-SVI wing on top of a cumulative base
    ``a_j(t)``, so total variance ``w(k, tau_j) = a_j(t) + wing_j(k, t)`` is
    non-decreasing in ``tau`` by construction (``a_j`` strictly increasing in
    ``j`` and ``wing_j >= 0``) and smiles stay convex in strike for the mild
    SVI parameters drawn below -- the DGP is arbitrage-free on the mesh up to
    ``jitter``, an optional multiplicative lognormal noise
    ``sigma -> sigma * exp(jitter * eps)`` (always positive, like real
    quoted vols) that DOES create violations (used to exercise the arb
    report and the no-arb finetune). The level recursion carries a momentum
    term ``momentum * (u_{t-1} - u_{t-2})`` (vol clustering: level changes
    trend), which makes the next-step surface increment GENUINELY
    predictable from history -- the regime where the paper's
    conditional-sampling claim is testable; ``momentum=0`` reduces to a
    plain OU where increments are ~unforecastable. Returns are
    ``r_t = u_t * sqrt(dt) * eps_t`` with ``eps_t`` correlated with the
    level innovation at ``leverage`` -- the joint (return, surface)
    dependence AD-Seq-Vol is built to learn.

    Correctness material only: SYNTHETIC, seeded, never market evidence.
    """
    n = _check_count(n_steps, "n_steps") + 1  # +1: entry 0 seeds the first pair
    jit = _check_nonneg(jitter, "jitter")
    u_bar = _check_positive(atm_level, "atm_level")
    vv = _check_nonneg(vol_of_vol, "vol_of_vol")
    kappa = _check_nonneg(mean_reversion, "mean_reversion")
    phi = float(momentum)
    if not math.isfinite(phi) or not (0.0 <= phi < 1.0):
        raise ValueError(f"momentum must be in [0, 1); got {momentum!r}")
    lev = float(leverage)
    if not math.isfinite(lev) or not (-1.0 <= lev <= 1.0):
        raise ValueError(f"leverage must be a correlation in [-1, 1]; got {leverage!r}")
    step = _check_positive(dt, "dt")
    rng = np.random.default_rng(int(seed))
    k = grid.log_moneyness
    nk, nt = grid.n_k, grid.n_tau

    z_u = rng.standard_normal(n)
    z_r = rng.standard_normal(n)
    eps_r = lev * z_u + math.sqrt(1.0 - lev * lev) * z_r

    u = np.empty(n, dtype=float)
    u[0] = u_bar
    # innovations scaled so the unconditional std of u stays ~ u_bar * vv
    s_u = u_bar * vv * math.sqrt(2.0 * max(kappa, 1e-3)) * math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        du_prev = u[t - 1] - u[t - 2] if t > 1 else 0.0
        u[t] = u[t - 1] + kappa * (u_bar - u[t - 1]) + phi * du_prev + s_u * z_u[t]
    u = np.maximum(u, 0.02 * u_bar)  # level floor keeps the DGP sane

    # slowly varying smile state (skew/curvature OU), shared across maturities
    rho_path = np.empty(n, dtype=float)
    curv_path = np.empty(n, dtype=float)
    rho_path[0], curv_path[0] = -0.45, 0.9
    z_rho = rng.standard_normal(n)
    z_curv = rng.standard_normal(n)
    for t in range(1, n):
        rho_path[t] = rho_path[t - 1] + 0.10 * (-0.45 - rho_path[t - 1]) + 0.04 * z_rho[t]
        curv_path[t] = curv_path[t - 1] + 0.10 * (0.9 - curv_path[t - 1]) + 0.05 * z_curv[t]
    rho_path = np.clip(rho_path, -0.75, 0.25)
    curv_path = np.clip(curv_path, 0.4, 1.6)

    surfaces = np.empty((n - 1, nk, nt), dtype=float)
    returns = np.empty(n - 1, dtype=float)
    for t in range(1, n):
        atm_var = u[t] * u[t]
        w_prev = 0.0
        row = np.empty((nk, nt), dtype=float)
        for j in range(nt):
            tau_j = float(grid.maturities[j])
            # base increment a_j(t) > 0 keeps cumulative w non-decreasing in tau
            a_j = (
                atm_var
                * tau_j
                * (0.55 + 0.05 * j)
                / (0.55 + 0.05 * (nt - 1))
                * (nt / (nt - j + 0.5))
            )
            wing = _svi_total_variance(
                k,
                a=0.0,
                b=0.12 * curv_path[t] * atm_var * math.sqrt(tau_j),
                rho=float(rho_path[t]),
                m=0.05,
                s=0.12 + 0.02 * j,
            )
            wing = np.maximum(wing - float(np.min(wing)), 0.0)
            w_j = w_prev + np.maximum(a_j, 1e-6 * atm_var * tau_j) + wing
            row[:, j] = np.sqrt(w_j / tau_j)
            w_prev = w_j
        surfaces[t - 1] = row
        returns[t - 1] = u[t] * math.sqrt(step) * eps_r[t]
    if jit > 0.0:
        surfaces = surfaces * np.exp(jit * rng.standard_normal(surfaces.shape))
    return np.asarray(returns, dtype=float), np.asarray(surfaces, dtype=float)


# ---------------------------------------------------------------------------
# supervised pairs: history window -> next-step joint target
# ---------------------------------------------------------------------------


def _code_matrix(grid: IVSGrid) -> Array:
    """OLS projection ``(3, n_k)`` mapping each maturity's vol row to smile codes.

    For every maturity slice ``sigma(.)`` the codes are the least-squares
    coefficients on the fixed basis ``[1, k, k^2]`` -- a level / slope /
    curvature triple. The map depends only on grid geometry (no fitted
    state), so training and inference compress identically.
    """
    k = grid.log_moneyness
    xk = np.column_stack([np.ones(grid.n_k), k, k * k])
    return np.asarray(np.linalg.pinv(xk), dtype=float)


def surface_codes(surfaces: Array, grid: IVSGrid) -> Array:
    """Smile codes of a surface batch, shape ``(n, 3 * n_tau)``.

    Row ``3*j : 3*j+3`` holds the (level, slope, curvature) OLS codes of
    maturity ``j`` -- a compressed, denoised summary of the surface used
    for history-window conditioning.
    """
    v = grid.check_surfaces(surfaces, positive=False, name="surfaces")
    p = _code_matrix(grid)
    return np.asarray(np.einsum("pk,nkt->ntp", p, v).reshape(v.shape[0], -1), dtype=float)


def build_supervised_pairs(
    returns: Array,
    surfaces: Array,
    grid: IVSGrid,
    window: int,
) -> tuple[Array, Array, Array]:
    """Flattened ``(X_t, y_t, V_t)`` supervised pairs for AD-Seq-Vol.

    For each ``t = window-1 .. n-2``: the context ``X_t`` stacks the last
    ``window`` returns, the LAST surface's smile code, and the ``window-1``
    coded increments ``codes(V_s) - codes(V_{s-1})`` over the window (shape
    ``h + 3 n_tau + (h-1) * 3 n_tau = h * (1 + 3 n_tau)``). The raw surface
    is compressed to its per-maturity level/slope/curvature codes AND the
    increments are exposed explicitly: vol surfaces are near-unit-root, so a
    small denoiser learns far better conditioning from (level, increment)
    than from levels it would have to difference internally. The target
    ``y_t = [r_{t+1}, V_{t+1} - V_t]`` stays the FULL surface increment
    (shape ``1 + d``), and ``V_t`` is the last realized surface used to
    reconstruct generated surfaces from predicted increments.
    """
    h = _check_count(window, "window")
    r, v = _check_stream(returns, surfaces, grid)
    n, nk, nt = v.shape
    if n - 1 - (h - 1) < 1:
        raise ValueError(f"window={h} leaves no supervised pairs on n={n} steps")
    d = nk * nt
    codes = surface_codes(v, grid)  # (n, 3 n_tau)
    dcodes = codes[1:] - codes[:-1]  # (n-1, 3 n_tau); index s-1 -> s
    n_pairs = n - h
    X = np.empty((n_pairs, h * (1 + codes.shape[1])), dtype=float)
    y = np.empty((n_pairs, 1 + d), dtype=float)
    v_last = np.empty((n_pairs, nk, nt), dtype=float)
    for i, t in enumerate(range(h - 1, n - 1)):
        lo = t - h + 1
        # dcodes[j] = codes[j+1]-codes[j]: intra-window increments are j = lo..t-1
        X[i] = np.concatenate([r[lo : t + 1], codes[t], dcodes[lo:t].reshape(-1)])
        y[i, 0] = r[t + 1]
        y[i, 1:] = (v[t + 1] - v[t]).reshape(-1)
        v_last[i] = v[t]
    return X, y, v_last


def _standardize(a: Array) -> tuple[Array, Array, Array]:
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)  # constant columns: unit scale
    return (
        np.asarray((a - mean) / std, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(std, dtype=float),
    )


# ---------------------------------------------------------------------------
# scenario baselines and proper scoring
# ---------------------------------------------------------------------------


def historical_resample_scenarios(
    returns: Array,
    surfaces: Array,
    grid: IVSGrid,
    window: int,
    *,
    n_scenarios: int,
    horizon: int,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Block-resample baseline: historical (return, surface) step sequences.

    Draws ``n_scenarios`` contiguous blocks of length ``horizon`` from the
    observed stream (uniform over admissible start indices) -- the paper's
    data-driven benchmark family placeholder (see module docstring for the
    documented GAN deviation). Returns ``(r_paths, v_paths)`` with shapes
    ``(n_scenarios, horizon)`` and ``(n_scenarios, horizon, n_k, n_tau)``;
    surface paths are LEVELS following the resampled increments, anchored at
    the surface realized ``window`` steps after each block start.
    Deterministic given ``seed``.
    """
    m = _check_count(n_scenarios, "n_scenarios")
    hz = _check_count(horizon, "horizon")
    h = _check_count(window, "window")
    r, v = _check_stream(returns, surfaces, grid)
    n = int(r.shape[0])
    last_start = n - h - hz
    if last_start < 0:
        raise ValueError(f"need n >= window + horizon (={h + hz}); got {n}")
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, last_start + 1, size=m)
    r_paths = np.empty((m, hz), dtype=float)
    v_paths = np.empty((m, hz, grid.n_k, grid.n_tau), dtype=float)
    for i, s in enumerate(starts):
        base = v[s + h - 1]
        r_paths[i] = r[s + h : s + h + hz]
        v_paths[i] = base + np.cumsum(v[s + h : s + h + hz] - v[s + h - 1 : s + h + hz - 1], axis=0)
    return r_paths, np.asarray(v_paths, dtype=float)


def scenario_energy_score(samples: Array, realized: Array) -> float:
    """Energy score of a scenario ensemble vs the realized vector (proper).

    Thin composition wrapper over ``quant_fund.metrics.energy_score``
    (Gneiting & Raftery 2007): ``samples`` ``(m, d)`` are model/baseline
    one-step joint targets (standardized), ``realized`` ``(d,)`` the realized
    target in the same units. Lower is better; strictly proper.
    """
    s = np.asarray(samples, dtype=float)
    y = np.asarray(realized, dtype=float).ravel()
    return float(energy_score(s, y))


# ---------------------------------------------------------------------------
# optimization-based hedge evaluation (numpy; proper-risk diagnostics only)
# ---------------------------------------------------------------------------


def evaluate_hedge(
    scenario_returns: Array,
    scenario_surfaces: Array,
    base_surface: Array,
    grid: IVSGrid,
    *,
    k_idx: int,
    tau_idx: int,
    alpha: float = 0.95,
    ridge: float = 1e-8,
) -> dict[str, float]:
    """Optimization-based one-step hedge of a short grid option under scenarios.

    The book is short the call at grid cell ``(k_idx, tau_idx)``. Each
    scenario ``s`` gives a next-step return ``r_s`` and surface ``V_s``; the
    option is revalued sticky-strike: spot ``S_s = exp(r_s)`` (base ``S_0 =
    1``), same strike ``K = e^{k*}``, same maturity ``tau*``, implied vol the
    scenario's own cell value ``V_s[k*, tau*]`` (floored at ``_SIGMA_FLOOR``
    inside pricing only). The hedge is a static position ``delta`` in the
    underlying minimizing the scenario tracking error
    ``sum_s (dC_s - delta dS_s)^2`` -- closed-form ridge solution
    ``delta* = sum(dS dC) / (sum(dS^2) + ridge * m)``, the linear case of the
    paper's optimization-based hedge (arXiv:2609.13402 §hedging).

    Diagnostics (hedging-error terms, never a P&L headline): residual rmse,
    mean absolute residual, expected shortfall of the residual loss at
    ``alpha`` (Rockafellar & Uryasev 2000 empirical estimator), and the same
    for the unhedged book and the frozen Black-Scholes delta hedge.
    """
    r = _check_returns(scenario_returns, "scenario_returns")
    v = grid.check_surfaces(scenario_surfaces, positive=False, name="scenario_surfaces")
    base = grid.check_surfaces(base_surface, positive=True, name="base_surface")
    if base.shape[0] != 1:
        raise ValueError("base_surface must be a single surface (n = 1)")
    if r.shape[0] != v.shape[0]:
        raise ValueError(
            f"scenario_returns and scenario_surfaces must share n; got {r.shape[0]} vs {v.shape[0]}"
        )
    if r.shape[0] < 4:
        raise ValueError(f"need >= 4 scenarios for a stable hedge; got {r.shape[0]}")
    a = float(alpha)
    if not math.isfinite(a) or not (0.0 < a < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")
    rg = _check_nonneg(ridge, "ridge")
    if not (0 <= int(k_idx) < grid.n_k):
        raise ValueError(f"k_idx must be in [0, {grid.n_k}); got {k_idx!r}")
    if not (0 <= int(tau_idx) < grid.n_tau):
        raise ValueError(f"tau_idx must be in [0, {grid.n_tau}); got {tau_idx!r}")
    i_k, i_t = int(k_idx), int(tau_idx)
    kk = float(grid.log_moneyness[i_k])
    tt = float(grid.maturities[i_t])
    sig0 = float(base[0, i_k, i_t])
    c0 = float(bs_price(1.0, math.exp(kk), tt, 0.0, 0.0, sig0, "call"))

    s1 = np.exp(r)
    sig1 = np.maximum(v[:, i_k, i_t], _SIGMA_FLOOR)
    c1 = np.asarray(bs_price(s1, math.exp(kk), tt, 0.0, 0.0, sig1, "call"), dtype=float)
    d_c = c1 - c0
    d_s = s1 - 1.0

    m = float(r.shape[0])
    ss = float(np.sum(d_s * d_s))
    if ss <= _RIDGE_FLOOR:
        raise ValueError("degenerate scenarios: all scenario returns are identical")
    delta_star = float(np.sum(d_s * d_c) / (ss + rg * m))

    # frozen BS delta at the base surface: N(d1) with S0 = 1, K = e^{k*}
    w0 = sig0 * sig0 * tt
    d1 = (-kk + 0.5 * w0) / math.sqrt(w0)
    delta_bs = float(norm.cdf(d1))

    def _stats(loss: Array) -> tuple[float, float, float]:
        rmse = float(np.sqrt(np.mean(loss * loss)))
        mean_abs = float(np.mean(np.abs(loss)))
        cnt = max(1, int(math.ceil((1.0 - a) * loss.size)))
        es = float(np.sort(loss)[-cnt:].mean())
        return rmse, mean_abs, es

    res_opt = d_c - delta_star * d_s
    res_unh = d_c
    res_bs = d_c - delta_bs * d_s
    rmse_opt, mabs_opt, es_opt = _stats(res_opt)
    rmse_unh, mabs_unh, es_unh = _stats(res_unh)
    rmse_bs, mabs_bs, es_bs = _stats(res_bs)
    return {
        "hedge_delta": delta_star,
        "hedge_bs_delta": delta_bs,
        "hedge_tracking_rmse": rmse_opt,
        "hedge_mean_abs": mabs_opt,
        "hedge_es_tail": es_opt,
        "unhedged_tracking_rmse": rmse_unh,
        "unhedged_mean_abs": mabs_unh,
        "unhedged_es_tail": es_unh,
        "bsdelta_tracking_rmse": rmse_bs,
        "bsdelta_mean_abs": mabs_bs,
        "bsdelta_es_tail": es_bs,
        "option_price_base": c0,
        "n_scenarios": float(r.shape[0]),
    }


# ---------------------------------------------------------------------------
# torch training core (guarded by _torch); numpy inference on extracted params
# ---------------------------------------------------------------------------


def _build_mlp(torch: Any, d_in: int, hidden: Sequence[int], d_out: int) -> Any:
    layers: list[Any] = []
    d = int(d_in)
    for hh in hidden:
        layers += [torch.nn.Linear(d, int(hh)), torch.nn.Tanh()]
        d = int(hh)
    layers.append(torch.nn.Linear(d, int(d_out)))
    return torch.nn.Sequential(*layers)


def _extract_mlp_layers(torch: Any, net: Any) -> _Layers:
    out: list[tuple[Array, Array]] = []
    for module in net:
        if isinstance(module, torch.nn.Linear):
            w = np.array(module.weight.detach().numpy(), dtype=float)
            b = np.array(module.bias.detach().numpy(), dtype=float)
            out.append((w, b))
    return tuple(out)


def _np_mlp(x: Array, layers: _Layers) -> Array:
    z = x
    last = len(layers) - 1
    for i, (w, b) in enumerate(layers):
        z = z @ w.T + b
        if i < last:
            z = np.tanh(z)
    return np.asarray(z, dtype=float)


def _arb_coefficients_butterfly(strikes: Array) -> Array:
    """Row coefficients turning call prices into second divided differences.

    ``B[i] @ C`` is the second divided difference over strikes
    ``(i, i+1, i+2)`` -- shared between the numpy report path and the torch
    penalty so the two measure exactly the same functional.
    """
    k1, k2, k3 = strikes[:-2], strikes[1:-1], strikes[2:]
    b = np.zeros((strikes.size - 2, strikes.size), dtype=float)
    rng = np.arange(strikes.size - 2)
    b[rng, rng] = 1.0 / ((k1 - k2) * (k1 - k3))
    b[rng, rng + 1] = 1.0 / ((k2 - k1) * (k2 - k3))
    b[rng, rng + 2] = 1.0 / ((k3 - k1) * (k3 - k2))
    return np.asarray(2.0 * b, dtype=float)


def _torch_arb_penalty(
    torch: Any,
    sigma_hat: Any,
    grid: IVSGrid,
    butterfly_b: Any,
    scale_s: float,
    scale_w: float,
    scale_b: float,
) -> Any:
    """Differentiable static-arb penalty on a batch of predicted surfaces.

    Mirrors :func:`arb_violation_report` with bounded per-cell surrogates:
    each condition's violation size ``u`` is normalized by its detached
    clean-data scale and squashed by ``u / (1 + u)``, so every violating
    cell contributes at most ~1 -- the three conditions stay balanced (an
    unbounded surrogate lets the largest-scale condition dominate and the
    optimizer sacrifices the others). Conditions: ``relu(-sigma)`` (nonneg),
    ``relu(w_j - w_{j+1})`` (calendar, ``w = clamp(sigma,0)^2 tau``),
    ``relu(-ddC)`` (butterfly, torch BS call prices at ``w`` floored).
    """

    def _bounded(u: Any) -> Any:
        v = torch.relu(u)
        return torch.mean(v / (1.0 + v))

    tau_t = torch.as_tensor(grid.maturities, dtype=sigma_hat.dtype)
    pos = torch.clamp(sigma_hat, min=0.0)
    w = pos * pos * tau_t
    pen_nonneg = _bounded(-sigma_hat / max(scale_s, 1e-8))

    d_w = w[:, :, 1:] - w[:, :, :-1]
    pen_cal = _bounded(-d_w / max(scale_w, 1e-12))

    w_floor = torch.clamp(w, min=1e-8)
    k_t = torch.as_tensor(grid.log_moneyness, dtype=sigma_hat.dtype)
    sqrt_w = torch.sqrt(w_floor)
    d1 = (-k_t[:, None] + 0.5 * w_floor) / sqrt_w
    d2 = d1 - sqrt_w
    calls = torch.special.ndtr(d1) - torch.exp(k_t[:, None]) * torch.special.ndtr(d2)
    ddc = torch.einsum("ij,njt->nit", butterfly_b, calls)
    pen_bfly = _bounded(-ddc / max(scale_b, 1e-12))
    return pen_nonneg + pen_cal + pen_bfly


def _train_eps_net(
    torch: Any,
    Xs: Array,
    Ys: Array,
    v_last: Array,
    grid: IVSGrid,
    *,
    sched: NoiseSchedule,
    hidden: Sequence[int],
    epochs: int,
    lr: float,
    batch_size: int | None,
    seed: int,
    y_mean: Array,
    y_std: Array,
    y_clip_lo: Array,
    y_clip_hi: Array,
    penalty_weight: float,
    net: Any = None,
) -> tuple[Any, list[float], list[float]]:
    """Adam on the conditional eps-MSE objective (+ optional arb penalty).

    ``x_t = sqrt(abar_t) y0 + sqrt(1 - abar_t) eps0`` with ``t ~ U{1..T}``
    and ``eps0 ~ N(0, I)`` from a seeded numpy generator (torch's RNG is
    reserved for initialization). With ``penalty_weight > 0`` each update
    additionally unrolls the full reverse chain with fresh reparameterized
    noises and adds ``weight * arb_penalty(V_last + dV_sampled)`` -- the
    AD-Seq-Vol-FT post-training penalty on generated surfaces, with the
    eps-MSE kept as a stability anchor; ``net`` may be a warmed module for
    continued training. Returns ``(net, loss_curve, pen_curve)``.
    """
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    n, dx = int(Xs.shape[0]), int(Xs.shape[1])
    dy = int(Ys.shape[1])
    t_steps = int(sched.n_steps)
    if net is None:
        net = _build_mlp(torch, dy + t_steps + dx, hidden, dy)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    bsz = n if batch_size is None else min(int(batch_size), n)
    steps_per_epoch = max(1, -(-n // bsz))
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(epochs) * steps_per_epoch, eta_min=float(lr) / 10.0
    )
    x_all = torch.as_tensor(Xs, dtype=torch.float32)
    y_all = torch.as_tensor(Ys, dtype=torch.float32)
    vlast_all = torch.as_tensor(v_last.reshape(n, -1), dtype=torch.float32)
    sab = torch.as_tensor(sched.sqrt_alpha_bar, dtype=torch.float32)
    sbb = torch.as_tensor(sched.sqrt_beta_bar, dtype=torch.float32)
    ymean_t = torch.as_tensor(y_mean, dtype=torch.float32)
    ystd_t = torch.as_tensor(y_std, dtype=torch.float32)
    butterfly_b = torch.as_tensor(_arb_coefficients_butterfly(grid.strikes), dtype=torch.float32)
    c0_t = torch.as_tensor(sched.c0, dtype=torch.float32)
    c1_t = torch.as_tensor(sched.c1, dtype=torch.float32)
    bt_t = torch.as_tensor(sched.beta_tilde, dtype=torch.float32)
    clip_lo = torch.as_tensor(y_clip_lo, dtype=torch.float32)
    clip_hi = torch.as_tensor(y_clip_hi, dtype=torch.float32)
    nk, nt = grid.n_k, grid.n_tau
    # detached clean-data scales for the normalized penalties: scale_w is the
    # mean total variance of the observed surfaces; scale_b the mean |second
    # divided difference| of the observed call-price surface -- each penalty
    # is then measured in units of the violation's own natural scale.
    sig_clean = torch.clamp(vlast_all, min=0.0).reshape(n, nk, nt)
    scale_s = max(1e-6, float(torch.mean(sig_clean).detach().numpy()))
    w_clean = sig_clean * sig_clean * torch.as_tensor(grid.maturities, dtype=torch.float32)
    scale_w = max(1e-8, float(torch.mean(w_clean).detach().numpy()))
    w_cl = torch.clamp(w_clean, min=1e-8)
    k_cl = torch.as_tensor(grid.log_moneyness, dtype=torch.float32)
    sq_cl = torch.sqrt(w_cl)
    d1c = (-k_cl[:, None] + 0.5 * w_cl) / sq_cl
    d2c = d1c - sq_cl
    calls_clean = torch.special.ndtr(d1c) - torch.exp(k_cl[:, None]) * torch.special.ndtr(d2c)
    ddc_clean = torch.einsum("ij,njt->nit", butterfly_b, calls_clean)
    scale_b = max(1e-6, float(torch.mean(torch.abs(ddc_clean)).detach().numpy()))
    rng = np.random.default_rng(int(seed))
    loss_curve: list[float] = []
    pen_curve: list[float] = []
    update = 0
    for _epoch in range(int(epochs)):
        order = np.arange(n) if batch_size is None else rng.permutation(n)
        for start in range(0, n, bsz):
            idx = order[start : start + bsz]
            xb = x_all[idx]
            yb = y_all[idx]
            vb = vlast_all[idx]
            nb = int(idx.size)
            t_idx = torch.as_tensor(rng.integers(0, t_steps, size=nb), dtype=torch.int64)
            eps0 = torch.as_tensor(rng.standard_normal((nb, dy)), dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            xt = sab[t_idx][:, None] * yb + sbb[t_idx][:, None] * eps0
            onehot = torch.nn.functional.one_hot(t_idx, num_classes=t_steps).to(torch.float32)
            eps_hat = net(torch.cat([xt, onehot, xb], dim=-1))
            loss = torch.mean((eps0 - eps_hat) * (eps0 - eps_hat))
            pen_val = torch.zeros((), dtype=torch.float32)
            if penalty_weight > 0.0:
                # Unrolled reverse pass with reparameterized noises -- the
                # paper's post-training penalty evaluated on what sampling
                # actually produces (documented: full chain, MSE anchor kept).
                xg = torch.as_tensor(rng.standard_normal((nb, dy)), dtype=torch.float32)
                for tt in range(t_steps, 0, -1):
                    oh = torch.full((nb, t_steps), 0.0, dtype=torch.float32)
                    oh[:, tt - 1] = 1.0
                    eh = net(torch.cat([xg, oh, xb], dim=-1))
                    x0 = (xg - sbb[tt - 1] * eh) / sab[tt - 1]
                    x0 = torch.clamp(x0, clip_lo, clip_hi)
                    if tt == 1:
                        xg = x0
                    else:
                        zg = torch.as_tensor(rng.standard_normal((nb, dy)), dtype=torch.float32)
                        xg = c0_t[tt - 1] * x0 + c1_t[tt - 1] * xg + torch.sqrt(bt_t[tt - 1]) * zg
                y_raw = xg * ystd_t + ymean_t
                dv = y_raw[:, 1:].reshape(nb, nk, nt)
                sigma_hat = vb.reshape(nb, nk, nt) + dv
                pen_val = _torch_arb_penalty(
                    torch, sigma_hat, grid, butterfly_b, scale_s, scale_w, scale_b
                )
                loss = loss + float(penalty_weight) * pen_val
            loss_curve.append(_require_finite_loss(float(loss.detach().numpy()), update))
            pen_curve.append(float(pen_val.detach().numpy()))
            loss.backward()
            opt.step()
            lr_sched.step()
            update += 1
    return net, loss_curve, pen_curve


@dataclass(frozen=True)
class IVSFitInfo:
    """Training trace of :meth:`IVSDiffusion.fit` (loss recorded BEFORE updates)."""

    epochs: int
    n_steps: int
    schedule: str
    window: int
    seed: int
    loss_curve: list[float]


@dataclass(frozen=True)
class IVSFinetuneInfo:
    """Trace of :meth:`IVSDiffusion.finetune_no_arb` (AD-Seq-Vol-FT).

    ``violation_rate_*`` is the share of PROBED surfaces carrying any
    violation; ``cell_rate_*`` is the mean of the three per-cell violation
    rates (nonneg, calendar, butterfly) -- the cell-level metric stays
    informative when surface_rate saturates at 1.0, as happens whenever
    per-cell violation probability is non-negligible.
    """

    epochs: int
    penalty_weight: float
    loss_curve: list[float]
    penalty_curve: list[float]
    violation_rate_before: float
    violation_rate_after: float
    cell_rate_before: float
    cell_rate_after: float


@dataclass(frozen=True, eq=False)
class _NumpyIVSParams:
    """Float64 inference parameters extracted from the torch denoiser."""

    schedule: NoiseSchedule
    eps_layers: _Layers
    window: int
    ctx_mean: Array
    ctx_std: Array
    y_mean: Array
    y_std: Array
    y_clip_lo: Array
    y_clip_hi: Array
    n_k: int
    n_tau: int


def _np_eps(params: _NumpyIVSParams, xt: Array, ctx: Array, t: int) -> Array:
    """eps_theta(x_t, onehot(t), ctx) in numpy; (n, dy) -> (n, dy)."""
    sched = params.schedule
    if not 1 <= t <= sched.n_steps:
        raise ValueError(f"t must be in 1..{sched.n_steps}; got {t}")
    n, dy = xt.shape
    onehot = np.broadcast_to(np.eye(sched.n_steps, dtype=float)[t - 1], (n, sched.n_steps))
    inp = np.concatenate([xt, onehot, ctx], axis=1)
    return np.asarray(_np_mlp(inp, params.eps_layers), dtype=float)


def _np_sample(params: _NumpyIVSParams, ctx: Array, rng: np.random.Generator) -> Array:
    """Conditional DDPM reverse pass; (n, dx) context -> (n, dy) standardized."""
    sched = params.schedule
    n, dy = ctx.shape[0], params.y_mean.size
    x = rng.standard_normal((n, dy))
    for t in range(sched.n_steps, 0, -1):
        eps_hat = _np_eps(params, x, ctx, t)
        sab_t = float(sched.sqrt_alpha_bar[t - 1])
        sbb_t = float(sched.sqrt_beta_bar[t - 1])
        x0 = (x - sbb_t * eps_hat) / sab_t
        x0 = np.clip(x0, params.y_clip_lo[None, :], params.y_clip_hi[None, :])
        if t == 1:
            x = x0
        else:
            z = rng.standard_normal((n, dy))
            x = (
                sched.c0[t - 1] * x0
                + sched.c1[t - 1] * x
                + math.sqrt(float(sched.beta_tilde[t - 1])) * z
            )
    return np.asarray(x, dtype=float)


class IVSDiffusion:
    """AD-Seq-Vol conditional diffusion over (next return, next IVS increment).

    ``fit`` trains the eps-net on flattened history windows; all inference is
    numpy on extracted parameters (house torch-gating convention). Use
    :meth:`finetune_no_arb` for the AD-Seq-Vol-FT post-training penalty and
    :meth:`generate_scenarios` for the paper's adapted sequential sampling.
    """

    def __init__(
        self,
        n_steps: int = 10,
        schedule: str = "cosine",
        hidden: Sequence[int] = (64, 64),
        epochs: int = 200,
        lr: float = 3e-3,
        batch_size: int | None = 64,
        seed: int = 0,
    ) -> None:
        if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) < 2:
            raise ValueError(f"n_steps must be an int >= 2; got {n_steps!r}")
        _check_choice(schedule, _SCHEDULES, "schedule")
        hidden_widths = tuple(int(hh) for hh in hidden)
        if not hidden_widths or any(hh < 1 for hh in hidden_widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths; got {hidden!r}"
            )
        if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) < 1:
            raise ValueError(f"epochs must be an int >= 1; got {epochs!r}")
        if batch_size is not None and (
            isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) < 1
        ):
            raise ValueError(f"batch_size must be an int >= 1 or None; got {batch_size!r}")
        self.n_steps = int(n_steps)
        self.schedule = str(schedule)
        self.hidden = hidden_widths
        self.epochs = int(epochs)
        self.lr = _check_positive(lr, "lr")
        self.batch_size = None if batch_size is None else int(batch_size)
        self.seed = int(seed)
        self._params: _NumpyIVSParams | None = None
        self._grid: IVSGrid | None = None
        self._net: Any = None
        self.fit_info: IVSFitInfo | None = None
        self.finetune_info: IVSFinetuneInfo | None = None

    # -- state ----------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    def _require_params(self) -> _NumpyIVSParams:
        if self._params is None:
            raise RuntimeError("IVSDiffusion is not fitted")
        return self._params

    def _require_grid(self) -> IVSGrid:
        if self._grid is None:
            raise RuntimeError("IVSDiffusion is not fitted")
        return self._grid

    # -- training (torch-gated) -------------------------------------------------

    def fit(
        self,
        returns: Array,
        surfaces: Array,
        grid: IVSGrid,
        *,
        window: int = 8,
    ) -> IVSDiffusion:
        """Fit the conditional DDPM on the joint next-step target.

        Validates inputs BEFORE torch is touched (contract errors are
        ``ValueError`` even without the ``nn`` extra); a successful fit needs
        torch and raises ``ImportError`` with guidance when absent.
        """
        h = _check_count(window, "window")
        r, v = _check_stream(returns, surfaces, grid)
        if r.shape[0] < _MIN_FIT_STEPS + h:
            raise ValueError(
                f"stream too short: need n >= window + {_MIN_FIT_STEPS}; "
                f"got {r.shape[0]} with window={h}"
            )
        X, y, v_last = build_supervised_pairs(r, v, grid, h)
        Xs, x_mean, x_std = _standardize(X)
        ys, y_mean, y_std = _standardize(y)
        sched = noise_schedule(self.n_steps, self.schedule)
        y_lo = np.asarray(ys.min(axis=0) - _TARGET_CLIP_MARGIN, dtype=float)
        y_hi = np.asarray(ys.max(axis=0) + _TARGET_CLIP_MARGIN, dtype=float)

        torch = _torch()
        net, loss_curve, _pen = _train_eps_net(
            torch,
            Xs,
            ys,
            v_last,
            grid,
            sched=sched,
            hidden=self.hidden,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seed,
            y_mean=y_mean,
            y_std=y_std,
            y_clip_lo=y_lo,
            y_clip_hi=y_hi,
            penalty_weight=0.0,
        )
        self._net = net
        self._grid = grid
        self._params = _NumpyIVSParams(
            schedule=sched,
            eps_layers=_extract_mlp_layers(torch, net),
            window=h,
            ctx_mean=x_mean,
            ctx_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            y_clip_lo=y_lo,
            y_clip_hi=y_hi,
            n_k=grid.n_k,
            n_tau=grid.n_tau,
        )
        self.fit_info = IVSFitInfo(
            epochs=self.epochs,
            n_steps=self.n_steps,
            schedule=self.schedule,
            window=h,
            seed=self.seed,
            loss_curve=loss_curve,
        )
        self.finetune_info = None
        return self

    def finetune_no_arb(
        self,
        returns: Array,
        surfaces: Array,
        *,
        penalty_weight: float = 30.0,
        epochs: int = 40,
        lr: float = 2e-3,
        n_probe: int = 128,
        seed: int | None = None,
    ) -> IVSFinetuneInfo:
        """AD-Seq-Vol-FT: continue training with the static-arb penalty active.

        Adds ``penalty_weight * bounded`` violations of (nonneg, calendar,
        butterfly) evaluated on unrolled reverse-diffusion samples
        (reparameterized fresh noises; the eps-MSE stays on as a stability
        anchor), reconstructed to implied-vol space via the window's last
        realized surface. ``violation_rate_before/after`` are the
        ``arb_violation_report`` surface rates on ``n_probe`` one-step
        conditional samples before vs after finetuning (same probe seed) --
        the paper's headline: FT cuts the violation rate toward zero.
        """
        params = self._require_params()
        grid = self._require_grid()
        if self._net is None:
            raise RuntimeError("torch denoiser is unavailable; refit the model")
        w = _check_positive(penalty_weight, "penalty_weight")
        ep = _check_count(epochs, "epochs")
        rate_lr = _check_positive(lr, "lr")
        npb = _check_count(n_probe, "n_probe")
        ft_seed = self.seed + 1 if seed is None else int(seed)
        r, v = _check_stream(returns, surfaces, grid)
        X, y, v_last = build_supervised_pairs(r, v, grid, params.window)
        Xs = np.asarray((X - params.ctx_mean) / params.ctx_std, dtype=float)
        ys = np.asarray((y - params.y_mean) / params.y_std, dtype=float)

        probe_ctx = Xs[: min(npb, Xs.shape[0])]
        probe_v = v_last[: min(npb, v_last.shape[0])]
        rng_probe = np.random.default_rng(self.seed + 777_001)
        before, before_cell = self._sampled_surface_rate(probe_ctx, probe_v, rng_probe)

        torch = _torch()
        net, loss_curve, pen_curve = _train_eps_net(
            torch,
            Xs,
            ys,
            v_last,
            grid,
            sched=params.schedule,
            hidden=self.hidden,
            epochs=ep,
            lr=rate_lr,
            batch_size=self.batch_size,
            seed=ft_seed,
            y_mean=params.y_mean,
            y_std=params.y_std,
            y_clip_lo=params.y_clip_lo,
            y_clip_hi=params.y_clip_hi,
            penalty_weight=w,
            net=self._net,
        )
        self._net = net
        self._params = _NumpyIVSParams(
            schedule=params.schedule,
            eps_layers=_extract_mlp_layers(torch, net),
            window=params.window,
            ctx_mean=params.ctx_mean,
            ctx_std=params.ctx_std,
            y_mean=params.y_mean,
            y_std=params.y_std,
            y_clip_lo=params.y_clip_lo,
            y_clip_hi=params.y_clip_hi,
            n_k=params.n_k,
            n_tau=params.n_tau,
        )
        rng_probe = np.random.default_rng(self.seed + 777_001)
        after, after_cell = self._sampled_surface_rate(probe_ctx, probe_v, rng_probe)
        info = IVSFinetuneInfo(
            epochs=ep,
            penalty_weight=w,
            loss_curve=loss_curve,
            penalty_curve=pen_curve,
            violation_rate_before=float(before),
            violation_rate_after=float(after),
            cell_rate_before=float(before_cell),
            cell_rate_after=float(after_cell),
        )
        self.finetune_info = info
        return info

    def _sampled_surface_rate(
        self, ctx_std: Array, v_last: Array, rng: np.random.Generator
    ) -> tuple[float, float]:
        """Probe (surface_rate, cell_rate) over one-step conditional samples.

        ``cell_rate`` is the mean of the three per-cell violation rates.
        """
        params = self._require_params()
        grid = self._require_grid()
        y_std = _np_sample(params, ctx_std, rng)
        dv = (y_std[:, 1:] * params.y_std[1:] + params.y_mean[1:]).reshape(
            -1, params.n_k, params.n_tau
        )
        sig = v_last + dv
        rep = arb_violation_report(sig, grid)
        cell = (rep.nonneg_rate + rep.calendar_rate + rep.butterfly_rate) / 3.0
        return rep.surface_rate, cell

    # -- numpy inference --------------------------------------------------------

    def _context(self, r_window: Array, v_window: Array) -> Array:
        """Standardized flattened context of one shared history window."""
        params = self._require_params()
        grid = self._require_grid()
        h = params.window
        r = _check_returns(r_window, "r_window")
        if r.shape[0] != h:
            raise ValueError(f"r_window must have length window={h}; got {r.shape[0]}")
        v = grid.check_surfaces(v_window, positive=True, name="v_window")
        if v.shape[0] != h:
            raise ValueError(f"v_window must have length window={h}; got {v.shape[0]}")
        codes = surface_codes(v, grid)
        dcodes = codes[1:] - codes[:-1]  # (h-1, 3 n_tau) intra-window increments
        x = np.concatenate([r, codes[h - 1], dcodes.reshape(-1)])[None, :]
        return np.asarray((x - params.ctx_mean) / params.ctx_std, dtype=float)

    def sample_next(
        self,
        r_window: Array,
        v_window: Array,
        *,
        n_samples: int,
        seed: int = 0,
    ) -> tuple[Array, Array, Array]:
        """One-step conditional samples ``(r', dV', V')`` from ``p(. | window)``.

        Returns next-step simple return ``(m,)``, surface increment
        ``(m, n_k, n_tau)``, and next implied-vol level ``V' = V_last + dV``
        (unclipped -- violations are measured, not hidden). Deterministic
        given ``seed``.
        """
        params = self._require_params()
        grid = self._require_grid()
        m = _check_count(n_samples, "n_samples")
        ctx = self._context(r_window, v_window)
        ctx = np.broadcast_to(ctx, (m, ctx.shape[1])).copy()
        rng = np.random.default_rng(int(seed))
        y_std = _np_sample(params, ctx, rng)
        y = y_std * params.y_std + params.y_mean
        r_next = np.asarray(y[:, 0], dtype=float)
        dv = np.asarray(y[:, 1:].reshape(m, params.n_k, params.n_tau), dtype=float)
        v_last = grid.check_surfaces(v_window, positive=True, name="v_window")[-1]
        return r_next, dv, np.asarray(v_last[None, :, :] + dv, dtype=float)

    def generate_scenarios(
        self,
        r_window: Array,
        v_window: Array,
        *,
        n_scenarios: int,
        horizon: int,
        seed: int = 0,
    ) -> tuple[Array, Array]:
        """Adapted multi-period scenarios (the paper's sequential generation).

        At every step the model samples the joint next ``(r, dV)`` given the
        CURRENT window, appends the generated observation, and slides the
        window -- conditioning adapts to its own realized history. Returns
        ``(r_paths, v_paths)`` of shapes ``(n_scenarios, horizon)`` and
        ``(n_scenarios, horizon, n_k, n_tau)``; surface paths are levels
        (increment chain anchored at the last realized surface).
        Deterministic given ``seed``.
        """
        params = self._require_params()
        grid = self._require_grid()
        m = _check_count(n_scenarios, "n_scenarios")
        hz = _check_count(horizon, "horizon")
        h = params.window
        r0 = _check_returns(r_window, "r_window")
        if r0.shape[0] != h:
            raise ValueError(f"r_window must have length window={h}; got {r0.shape[0]}")
        v0 = grid.check_surfaces(v_window, positive=True, name="v_window")
        if v0.shape[0] != h:
            raise ValueError(f"v_window must have length window={h}; got {v0.shape[0]}")

        rng = np.random.default_rng(int(seed))
        nc = 3 * grid.n_tau
        r_hist = np.broadcast_to(r0[None, :], (m, h)).copy()
        v_hist = np.broadcast_to(v0[None, :, :, :], (m, h, grid.n_k, grid.n_tau)).copy()
        r_paths = np.empty((m, hz), dtype=float)
        v_paths = np.empty((m, hz, grid.n_k, grid.n_tau), dtype=float)
        x_mean, x_std = params.ctx_mean, params.ctx_std
        for t in range(hz):
            # context layout shared with build_supervised_pairs:
            # [r_{t-h+1..t}, code(V_t), dcodes over the window]
            codes = surface_codes(v_hist.reshape(m * h, grid.n_k, grid.n_tau), grid).reshape(
                m, h, nc
            )
            dcodes = codes[:, 1:] - codes[:, :-1]
            ctx = np.concatenate([r_hist, codes[:, -1], dcodes.reshape(m, -1)], axis=1)
            ctxs = (ctx - x_mean) / x_std
            y_std = _np_sample(params, ctxs, rng)
            y = y_std * params.y_std + params.y_mean
            r_next = y[:, 0]
            v_next = v_hist[:, -1] + y[:, 1:].reshape(m, grid.n_k, grid.n_tau)
            r_paths[:, t] = r_next
            v_paths[:, t] = v_next
            r_hist = np.concatenate([r_hist[:, 1:], r_next[:, None]], axis=1)
            v_hist = np.concatenate([v_hist[:, 1:], v_next[:, None, :, :]], axis=1)
        return np.asarray(r_paths, dtype=float), np.asarray(v_paths, dtype=float)

    def conditional_targets(
        self, r_window: Array, v_window: Array, *, n_samples: int, seed: int = 0
    ) -> Array:
        """Standardized one-step joint-target samples ``(m, 1 + d)``.

        Used for proper-scoring comparisons (energy score) against realized
        standardized targets -- the model's conditional law in the training
        metric space.
        """
        params = self._require_params()
        m = _check_count(n_samples, "n_samples")
        ctx = self._context(r_window, v_window)
        ctx = np.broadcast_to(ctx, (m, ctx.shape[1])).copy()
        rng = np.random.default_rng(int(seed))
        return np.asarray(_np_sample(params, ctx, rng), dtype=float)


# ---------------------------------------------------------------------------
# SYNTHETIC validation bench
# ---------------------------------------------------------------------------


def bench_ivs_diffusion(
    n_stream: int = 520,
    window: int = 8,
    horizon: int = 4,
    n_scenarios: int = 48,
    n_train: int = 420,
    seed: int = 0,
    n_steps: int = 10,
    epochs: int = 600,
    ft_epochs: int = 40,
    penalty_weight: float = 30.0,
    hidden: Sequence[int] = (96, 96),
    jitter: float = 0.02,
    batch_size: int | None = 64,
) -> dict[str, float | str]:
    """AD-Seq-Vol on a seeded SYNTHETIC stream: proper scores + arb diagnostics.

    End-to-end correctness bench for the paper's claims (arXiv:2609.13402):
    the conditional diffusion beats a historical block-resample baseline on
    the ENERGY SCORE of one-step joint targets and on optimization-based
    hedging tail risk, and AD-Seq-Vol-FT cuts the generated surfaces' static
    no-arbitrage violation rate below the (jittered) training data's. All
    keys are ``float`` metrics or ``str`` labels; ``synthetic`` /
    ``claim=research_metric_only`` mark the honesty boundary -- correctness
    evidence, never market evidence, no Sharpe/P&L headline.
    """
    grid = IVSGrid(
        log_moneyness=np.linspace(-0.35, 0.25, 7),
        maturities=np.array([0.10, 0.25, 0.50, 1.0]),
    )
    r, v = synthetic_ivs_stream(n_stream, grid, seed=int(seed), jitter=jitter)
    r_tr, v_tr = r[:n_train], v[:n_train]
    model = IVSDiffusion(
        n_steps=n_steps,
        epochs=epochs,
        hidden=hidden,
        batch_size=batch_size,
        seed=int(seed),
    ).fit(r_tr, v_tr, grid, window=window)

    # --- one-step conditional quality: energy score on standardized targets
    X_all, y_all, v_last_all = build_supervised_pairs(r, v, grid, window)
    params = model._require_params()
    ev = slice(n_train - window, X_all.shape[0])
    ctxs = np.asarray((X_all[ev] - params.ctx_mean) / params.ctx_std, dtype=float)
    y_std = np.asarray((y_all - params.y_mean) / params.y_std, dtype=float)
    rng = np.random.default_rng(int(seed) + 11)
    n_ev = int(ctxs.shape[0])
    es_model = np.empty(n_ev, dtype=float)
    es_resample = np.empty(n_ev, dtype=float)
    y_tr_std = np.asarray(
        (build_supervised_pairs(r_tr, v_tr, grid, window)[1] - params.y_mean) / params.y_std,
        dtype=float,
    )
    for i in range(n_ev):
        ctx = np.broadcast_to(ctxs[i], (n_scenarios, ctxs.shape[1])).copy()
        draw = _np_sample(params, ctx, rng)
        es_model[i] = scenario_energy_score(draw, y_std[ev][i])
        idx = rng.integers(0, y_tr_std.shape[0], size=n_scenarios)
        es_resample[i] = scenario_energy_score(y_tr_std[idx], y_std[ev][i])

    # --- sequential scenarios vs resample baseline on the hedge eval
    base_r = r[n_train - window : n_train]
    base_v = v[n_train - window : n_train]
    r_paths, v_paths = model.generate_scenarios(
        base_r, base_v, n_scenarios=n_scenarios, horizon=horizon, seed=int(seed) + 3
    )
    rr_paths, rv_paths = historical_resample_scenarios(
        r_tr, v_tr, grid, window, n_scenarios=n_scenarios, horizon=horizon, seed=int(seed) + 3
    )
    k_idx = grid.n_k // 2
    tau_idx = grid.n_tau - 1  # hedge the longest-maturity ATM option
    model_hedge = evaluate_hedge(
        r_paths[:, 0], v_paths[:, 0], base_v[-1:], grid, k_idx=k_idx, tau_idx=tau_idx
    )
    res_hedge = evaluate_hedge(
        rr_paths[:, 0], rv_paths[:, 0], base_v[-1:], grid, k_idx=k_idx, tau_idx=tau_idx
    )

    # --- no-arb audit: training data vs generated vs fine-tuned
    arb_data = arb_violation_report(v_tr, grid)
    arb_gen = arb_violation_report(v_paths[:, 0], grid)
    ft = model.finetune_no_arb(
        r_tr,
        v_tr,
        penalty_weight=penalty_weight,
        epochs=ft_epochs,
        lr=2e-3,
        seed=int(seed) + 5,
    )
    r_paths_ft, v_paths_ft = model.generate_scenarios(
        base_r, base_v, n_scenarios=n_scenarios, horizon=1, seed=int(seed) + 3
    )
    arb_ft = arb_violation_report(v_paths_ft[:, 0], grid)

    out: dict[str, float | str] = {
        "es_model": float(np.mean(es_model)),
        "es_resample": float(np.mean(es_resample)),
        "es_gain_vs_resample": float(np.mean(es_resample) - np.mean(es_model)),
        "hedge_es_tail_model": model_hedge["hedge_es_tail"],
        "hedge_es_tail_resample": res_hedge["hedge_es_tail"],
        "hedge_rmse_model": model_hedge["hedge_tracking_rmse"],
        "hedge_rmse_resample": res_hedge["hedge_tracking_rmse"],
        "hedge_delta_model": model_hedge["hedge_delta"],
        "arb_rate_data": arb_data.surface_rate,
        "arb_rate_generated": arb_gen.surface_rate,
        "arb_rate_finetuned": arb_ft.surface_rate,
        "arb_cell_generated": (arb_gen.nonneg_rate + arb_gen.calendar_rate + arb_gen.butterfly_rate)
        / 3.0,
        "arb_cell_finetuned": (arb_ft.nonneg_rate + arb_ft.calendar_rate + arb_ft.butterfly_rate)
        / 3.0,
        "ft_probe_rate_before": ft.violation_rate_before,
        "ft_probe_rate_after": ft.violation_rate_after,
        "ft_probe_cell_before": ft.cell_rate_before,
        "ft_probe_cell_after": ft.cell_rate_after,
        "n_stream": float(n_stream),
        "n_train": float(n_train),
        "seed": float(seed),
        "dgp": "synthetic_svi_leverage",
        "claim": "research_metric_only",
        "synthetic": "seeded_ivs_stream",
    }
    return out
