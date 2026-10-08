"""XVA suite: CVA / FVA / MVA by Monte Carlo on a seeded SYNTHETIC book.

Counterparty-risk valuation adjustments computed from Euler-simulated
portfolio value paths of a synthetic book (payer IR swap + FX forward +
equity call on shared Vasicek-rate / GBM-FX / GBM-equity factors).

Formulas
--------
- **CVA** (Pykhtin & Zhu 2007, GSI RM 2(1); Gregory 2020, ch. 7):

      CVA = (1 - R) * sum_i E[D(0, t_i) V^+(t_i)] * (S(t_{i-1}) - S(t_i)),

  with survival ``S(t) = exp(-Lambda(t))`` from a constant or piecewise-
  constant (term) hazard curve.  Optional wrong-way-risk (WWR) mode couples
  the counterparty default time to a market factor through a one-factor
  Gaussian copula (Li 2000), matching the latent convention of
  :mod:`quant_fund.models.gaussian_copula_default`: ``L = sqrt(rho) M +
  sqrt(1 - rho) eps``, default by ``T`` iff ``L <= Phi^{-1}(PD_T)``, and the
  default time is the inverse-cumulative-hazard transform ``tau =
  Lambda^{-1}(-ln(1 - Phi(L)))`` so the marginal default distribution is
  exactly the hazard curve for any rho.

- **FVA** (Hull & White 2012, "The FVA Debate", Risk 25(1)): one-sided
  borrowing-cost convention — the dealer funds the uncollateralised positive
  MtM at OIS + ``funding_spread`` and earns no spread on negative MtM:

      FVA = s_f * E[ int_0^T D(0, t) max(V_t - C_t, 0) dt ].

  The optional two-sided variant (``include_funding_benefit=True``) nets the
  symmetric lending benefit; it is NOT the default because Hull & White argue
  the benefit is not economically available to a dealer that cannot lend the
  surplus at its own borrowing spread.

- **MVA** (Andersen, Choudhury & Xing 2019, "Computationally Efficient
  Margin", JPM 45(4)): capital ``K(t) = alpha * VaR_q[V^+(t + delta) |
  F_t]`` over a margin-period-of-risk window ``delta``, charged at
  ``capital_cost_rate`` and discounted:

      MVA = k * E[ int_0^T D(0, t) K_t dt ].

  The conditional MPoR-window VaR uses the computationally efficient
  frozen-local-normal approximation ``VaR_q ~ max(V_t + z_q sigma_t sqrt(delta),
  0)`` with the cross-path local exposure volatility ``sigma_t`` (no nested
  Monte Carlo); this is an approximation, biased where exposure diffusion is
  strongly state-dependent.

Closed-form anchors (tested): deterministic exposure ``EE`` with zero rates
gives ``CVA = (1 - R) * PD * EE`` exactly (telescoping sum); CVA is monotone
in hazard and LGD; FVA -> 0 as ``funding_spread`` -> 0; MVA -> 0 as
``alpha`` -> 0; WWR CVA exceeds independent CVA on a planted positively
correlated default/market scenario (and reverses under the opposite sign).

Rate-leg discretisation mirrors :mod:`quant_fund.models.short_rate` (exact
Vasicek AR(1) step and the ``A exp(-B r)`` zero-coupon bond formula); GBM
legs use log-Euler.  The Gaussian-copula diagnostics reuse
:func:`quant_fund.models.gaussian_copula_default.conditional_default_prob`.

References
----------
Gregory, J. (2020). *The XVA Challenge*, 2nd ed. Wiley.
Pykhtin, M. & Zhu, S. (2007). "A Guide to Modelling Counterparty Credit
    Risk". GSI RM 2(1).
Hull, J. & White, A. (2012). "The FVA Debate". Risk 25(1).
Andersen, L., Choudhury, S. & Xing, H. (2019). "Computationally Efficient
    Margin". Journal of Portfolio Management 45(4).
Li, D.X. (2000). "On Default Correlation: A Copula Function Approach".
    Journal of Fixed Income 9(4).

HONESTY: everything here runs on seeded SYNTHETIC simulated paths — an
engine-correctness diagnostic for the XVA machinery, never market evidence.
Outputs are risk/price diagnostics: no live counterparty claims, no broker
connectivity, no trading recommendations.  Fail-closed on invalid inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from quant_fund.models.gaussian_copula_default import conditional_default_prob

Array = NDArray[np.float64]

DEFAULT_SEED = 20260929
_SOURCE = "SYNTHETIC"
_CLAIM = "diagnostics_only"
_MATURITY_TOL = 1e-9
_VALID_WWR_FACTORS = ("rate", "fx", "equity")


def _as_float(x: object) -> float:
    """Fail-closed extraction of a float from a mixed diagnostic dict."""
    if not isinstance(x, float) or not math.isfinite(x):
        raise ValueError(f"expected a finite float, got {x!r}")
    return float(x)


# --------------------------------------------------------------- rate helpers


def _vasicek_zcb(r: Array, tau: float | Array, kappa: float, theta: float, sigma: float) -> Array:
    """Vector Vasicek zero-coupon bond price ``A(tau) exp(-B(tau) r)``.

    Mirrors :func:`quant_fund.models.short_rate.vasicek_bond_price` (scalar)
    over an array of short-rate states; ``tau`` may be a scalar or an array
    of maturities broadcasting against ``r``.
    """
    b = (1.0 - np.exp(-kappa * tau)) / kappa
    a = np.exp((theta - sigma**2 / (2.0 * kappa**2)) * (b - tau) - sigma**2 * b**2 / (4.0 * kappa))
    return np.asarray(a * np.exp(-b * np.asarray(r, dtype=float)), dtype=float)


def _bs_call(S: Array, K: float, tau: float, r: Array, q: float, sigma: float) -> Array:
    """Vectorised Black-Scholes-Merton European call with dividend yield ``q``."""
    sq = sigma * math.sqrt(tau)
    d1 = (
        np.log(np.asarray(S, dtype=float) / K)
        + (np.asarray(r, dtype=float) - q + 0.5 * sigma**2) * tau
    ) / sq
    d2 = d1 - sq
    out = np.asarray(S, dtype=float) * np.exp(-q * tau) * np.asarray(
        stats.norm.cdf(d1)
    ) - K * np.exp(-np.asarray(r, dtype=float) * tau) * np.asarray(stats.norm.cdf(d2))
    return np.asarray(out, dtype=float)


def _trapz_paths(a: Array, times: Array) -> Array:
    """Per-path trapezoidal integral over the time grid: shape (n_paths,)."""
    dt = np.diff(times)
    return np.asarray(np.sum(0.5 * (a[:, :-1] + a[:, 1:]) * dt[None, :], axis=1), dtype=float)


def _validate_times(times: Array) -> Array:
    t = np.asarray(times, dtype=float).ravel()
    if t.size < 2 or not np.isfinite(t).all():
        raise ValueError("times must be finite with >= 2 entries")
    if abs(t[0]) > 1e-12 or not np.all(np.diff(t) > 0.0):
        raise ValueError("times must start at 0 and be strictly increasing")
    return t


# ------------------------------------------------------------------ the book


@dataclass(frozen=True)
class SyntheticBook:
    """Seeded SYNTHETIC derivatives book on three shared risk factors.

    Instruments (all valued in domestic currency):

    - a fixed-rate IR swap (payer by default) on the Vasicek short rate,
      floating leg priced with the continuously-reset par identity
      ``1 - P(t, T)``, fixed leg as a schedule of zero-coupon bonds;
    - an FX forward on a GBM spot with constant foreign rate;
    - a European equity call on a GBM with dividend yield.

    Defaults are par-anchored: ``swap_fixed=None`` sets the par swap rate and
    ``fx_fwd_strike=None`` sets the initial forward, so both start at zero
    value and exposure builds purely from factor moves.  Fail-closed on
    invalid parameters (non-PSD correlations, non-positive vols, maturities
    beyond the horizon, ...).
    """

    # Vasicek short-rate factor (exact AR(1) discretisation, short_rate.py)
    r0: float = 0.03
    kappa_r: float = 0.8
    theta_r: float = 0.03
    sigma_r: float = 0.01
    # FX factor (log-Euler GBM), constant foreign rate
    fx0: float = 1.25
    sigma_fx: float = 0.10
    foreign_rate: float = 0.01
    # Equity factor (log-Euler GBM)
    eq0: float = 100.0
    sigma_eq: float = 0.25
    eq_div_yield: float = 0.0
    # Brownian correlations, order (rate, fx, equity)
    rho_rate_fx: float = -0.2
    rho_rate_eq: float = -0.3
    rho_fx_eq: float = 0.15
    # Payer/receiver IR swap
    swap_notional: float = 1_000_000.0
    swap_fixed: float | None = None
    swap_maturity: float = 5.0
    swap_freq: int = 2
    swap_is_payer: bool = True
    # FX forward (long foreign currency by default)
    fx_fwd_notional: float = 1_000_000.0
    fx_fwd_strike: float | None = None
    fx_fwd_maturity: float = 2.5
    fx_fwd_is_long: bool = True
    # European equity call (units of the equity)
    eq_opt_notional: float = 10_000.0
    eq_opt_strike: float = 100.0
    eq_opt_maturity: float = 3.0
    # Simulation horizon; None -> max instrument maturity
    horizon: float | None = None

    # derived (set in __post_init__)
    _horizon: float = field(init=False, repr=False)
    _chol: Array = field(init=False, repr=False)
    _swap_times: Array = field(init=False, repr=False)
    _swap_deltas: Array = field(init=False, repr=False)
    _swap_fixed_res: float = field(init=False, repr=False)
    _fx_strike_res: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "r0",
            "kappa_r",
            "theta_r",
            "sigma_r",
            "fx0",
            "sigma_fx",
            "foreign_rate",
            "eq0",
            "sigma_eq",
            "eq_div_yield",
            "rho_rate_fx",
            "rho_rate_eq",
            "rho_fx_eq",
            "swap_notional",
            "swap_maturity",
            "fx_fwd_notional",
            "fx_fwd_maturity",
            "eq_opt_notional",
            "eq_opt_strike",
            "eq_opt_maturity",
        ):
            v = getattr(self, name)
            if not (isinstance(v, float | int) and math.isfinite(float(v))):
                raise ValueError(f"book parameter {name} must be finite, got {v!r}")
        if self.kappa_r <= 0.0:
            raise ValueError("kappa_r must be positive")
        for name in ("sigma_r", "sigma_fx", "sigma_eq"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.fx0 <= 0.0 or self.eq0 <= 0.0:
            raise ValueError("fx0 and eq0 must be positive")
        for name in ("swap_notional", "fx_fwd_notional", "eq_opt_notional"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.eq_opt_strike <= 0.0:
            raise ValueError("eq_opt_strike must be positive")
        if self.swap_freq not in (1, 2, 4, 12):
            raise ValueError("swap_freq must be one of 1, 2, 4, 12")
        if (
            abs(self.swap_maturity * self.swap_freq - round(self.swap_maturity * self.swap_freq))
            > 1e-9
        ):
            raise ValueError("swap_maturity must be a whole multiple of 1/swap_freq")
        for name in ("rho_rate_fx", "rho_rate_eq", "rho_fx_eq"):
            if not -1.0 < float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must lie in (-1, 1)")
        corr = np.array(
            [
                [1.0, self.rho_rate_fx, self.rho_rate_eq],
                [self.rho_rate_fx, 1.0, self.rho_fx_eq],
                [self.rho_rate_eq, self.rho_fx_eq, 1.0],
            ]
        )
        eigs = np.linalg.eigvalsh(corr)
        if float(np.min(eigs)) <= 1e-8:
            raise ValueError("factor correlation matrix must be positive definite")
        mats = np.array([self.swap_maturity, self.fx_fwd_maturity, self.eq_opt_maturity])
        if not np.isfinite(mats).all() or np.any(mats <= 0.0):
            raise ValueError("instrument maturities must be positive")
        horizon = float(mats.max()) if self.horizon is None else float(self.horizon)
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("horizon must be positive")
        if float(mats.max()) > horizon + 1e-12:
            raise ValueError("instrument maturity exceeds horizon")
        if self.swap_fixed is not None and not (
            isinstance(self.swap_fixed, float | int) and math.isfinite(float(self.swap_fixed))
        ):
            raise ValueError("swap_fixed must be finite or None (par)")
        if self.fx_fwd_strike is not None and not (
            isinstance(self.fx_fwd_strike, float | int)
            and math.isfinite(float(self.fx_fwd_strike))
            and float(self.fx_fwd_strike) > 0.0
        ):
            raise ValueError("fx_fwd_strike must be positive/finite or None (initial forward)")

        object.__setattr__(self, "_horizon", horizon)
        object.__setattr__(self, "_chol", np.linalg.cholesky(corr))
        n_pay = int(round(self.swap_maturity * self.swap_freq))
        pay_times = np.arange(1, n_pay + 1, dtype=float) / float(self.swap_freq)
        object.__setattr__(self, "_swap_times", pay_times)
        object.__setattr__(self, "_swap_deltas", np.full(n_pay, 1.0 / float(self.swap_freq)))
        p_end = float(
            _vasicek_zcb(
                np.array([self.r0]), self.swap_maturity, self.kappa_r, self.theta_r, self.sigma_r
            )[0]
        )
        annuity0 = float(
            np.sum(
                self._swap_deltas
                * _vasicek_zcb(
                    np.array([self.r0]), pay_times, self.kappa_r, self.theta_r, self.sigma_r
                )
            )
        )
        par = (1.0 - p_end) / annuity0
        fixed = par if self.swap_fixed is None else float(self.swap_fixed)
        object.__setattr__(self, "_swap_fixed_res", fixed)
        p_fx = float(
            _vasicek_zcb(
                np.array([self.r0]), self.fx_fwd_maturity, self.kappa_r, self.theta_r, self.sigma_r
            )[0]
        )
        fwd0 = self.fx0 * math.exp(-self.foreign_rate * self.fx_fwd_maturity) / p_fx
        object.__setattr__(
            self,
            "_fx_strike_res",
            fwd0 if self.fx_fwd_strike is None else float(self.fx_fwd_strike),
        )

    @property
    def horizon_resolved(self) -> float:
        return float(self._horizon)

    @property
    def cholesky(self) -> Array:
        return np.array(self._chol)

    @property
    def swap_fixed_rate(self) -> float:
        return float(self._swap_fixed_res)

    @property
    def fx_fwd_strike_price(self) -> float:
        return float(self._fx_strike_res)

    @property
    def swap_payment_times(self) -> Array:
        return np.array(self._swap_times)

    @property
    def swap_payment_deltas(self) -> Array:
        return np.array(self._swap_deltas)


# ------------------------------------------------------------- simulation


@dataclass(frozen=True)
class ExposureSimulation:
    """Portfolio value paths with pathwise discount factors.

    ``values[p, i]`` is the book MtM on path ``p`` at ``times[i]``;
    ``discounts[p, i]`` is ``D(0, t_i)`` along that path.  ``components``
    holds per-instrument value curves; ``factor_terminals`` holds the
    terminal factor states used as WWR copula latents (empty for
    hand-built simulations, which therefore reject ``copula_wwr`` mode).
    """

    times: Array
    values: Array
    discounts: Array
    seed: int
    book: SyntheticBook | None = None
    components: dict[str, Array] = field(default_factory=dict)
    factor_terminals: dict[str, Array] = field(default_factory=dict)
    source: str = _SOURCE

    def __post_init__(self) -> None:
        t = _validate_times(self.times)
        object.__setattr__(self, "times", t)
        v = np.asarray(self.values, dtype=float)
        d = np.asarray(self.discounts, dtype=float)
        if v.ndim != 2 or v.shape[1] != t.size or v.shape[0] < 1:
            raise ValueError(f"values must be (n_paths >= 1, {t.size}), got {v.shape}")
        if d.shape != v.shape:
            raise ValueError(f"discounts shape {d.shape} != values shape {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("values contain non-finite entries")
        if not np.isfinite(d).all() or np.any(d <= 0.0):
            raise ValueError("discounts must be finite and strictly positive")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "discounts", d)
        for name, arr in {**self.components, **self.factor_terminals}.items():
            a = np.asarray(arr, dtype=float)
            if not np.isfinite(a).all():
                raise ValueError(f"{name} contains non-finite entries")

    @property
    def n_paths(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.times.size)

    @classmethod
    def from_arrays(
        cls,
        times: Array,
        values: Array,
        discounts: Array,
        *,
        seed: int = 0,
        book: SyntheticBook | None = None,
    ) -> ExposureSimulation:
        """Build a simulation from hand-made arrays (closed-form anchors)."""
        return cls(times=times, values=values, discounts=discounts, seed=int(seed), book=book)

    def positive_exposure(self) -> Array:
        return np.asarray(np.maximum(self.values, 0.0), dtype=float)

    def discounted_positive_exposure(self) -> Array:
        return np.asarray(self.discounts * np.maximum(self.values, 0.0), dtype=float)


def simulate_exposure(
    book: SyntheticBook,
    *,
    n_paths: int = 4000,
    n_steps: int = 240,
    seed: int = DEFAULT_SEED,
) -> ExposureSimulation:
    """Simulate the SYNTHETIC book's value paths (seeded, deterministic).

    Factors: exact-discretisation Vasicek short rate (same AR(1) coefficients
    as :func:`quant_fund.models.short_rate.vasicek_simulate`) plus log-Euler
    GBMs for FX and equity, with correlated Brownian increments from the
    book's Cholesky factor.  The book is re-valued on every grid date under
    the simulated state (affine bond prices, analytic BS for the call).
    """
    if not isinstance(book, SyntheticBook):
        raise ValueError("book must be a SyntheticBook instance")
    if int(n_paths) < 2 or int(n_steps) < 2:
        raise ValueError("n_paths and n_steps must be >= 2")
    n_paths, n_steps = int(n_paths), int(n_steps)
    T = book.horizon_resolved
    times = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps, 3))
    dz = np.asarray(z @ book.cholesky.T, dtype=float)

    e = math.exp(-book.kappa_r * dt)
    sd_r = book.sigma_r * math.sqrt((1.0 - e * e) / (2.0 * book.kappa_r))
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = book.r0
    log_fx = np.empty((n_paths, n_steps + 1))
    log_fx[:, 0] = math.log(book.fx0)
    log_eq = np.empty((n_paths, n_steps + 1))
    log_eq[:, 0] = math.log(book.eq0)
    sq_dt = math.sqrt(dt)
    for k in range(n_steps):
        rk = r[:, k]
        r[:, k + 1] = rk * e + book.theta_r * (1.0 - e) + sd_r * dz[:, k, 0]
        log_fx[:, k + 1] = log_fx[:, k] + (
            (rk - book.foreign_rate - 0.5 * book.sigma_fx**2) * dt
            + book.sigma_fx * sq_dt * dz[:, k, 1]
        )
        log_eq[:, k + 1] = log_eq[:, k] + (
            (rk - book.eq_div_yield - 0.5 * book.sigma_eq**2) * dt
            + book.sigma_eq * sq_dt * dz[:, k, 2]
        )

    # pathwise discount factors D(0, t) = exp(-int r ds), trapezoidal in r
    incr = 0.5 * (r[:, :-1] + r[:, 1:]) * dt
    discounts = np.empty((n_paths, n_steps + 1))
    discounts[:, 0] = 1.0
    discounts[:, 1:] = np.exp(-np.cumsum(incr, axis=1))

    values, comp = _value_book(book, times, r, np.exp(log_fx), np.exp(log_eq))
    if not np.isfinite(values).all():
        raise ValueError("simulation produced non-finite portfolio values")
    factor_terminals = {
        "rate": np.array(r[:, -1]),
        "fx": np.array(log_fx[:, -1] - math.log(book.fx0)),
        "equity": np.array(log_eq[:, -1] - math.log(book.eq0)),
    }
    return ExposureSimulation(
        times=times,
        values=values,
        discounts=discounts,
        seed=int(seed),
        book=book,
        components=comp,
        factor_terminals=factor_terminals,
    )


def _value_book(
    book: SyntheticBook, times: Array, r: Array, s_fx: Array, s_eq: Array
) -> tuple[Array, dict[str, Array]]:
    """Value swap / FX forward / equity call on the grid; return (total, parts)."""
    n_paths, n_times = r.shape
    total = np.zeros((n_paths, n_times))
    comp = {
        "swap": np.zeros((n_paths, n_times)),
        "fx_forward": np.zeros((n_paths, n_times)),
        "equity_call": np.zeros((n_paths, n_times)),
    }
    zcb = _vasicek_zcb
    kr, th, sg = book.kappa_r, book.theta_r, book.sigma_r
    sign_sw = 1.0 if book.swap_is_payer else -1.0
    sign_fx = 1.0 if book.fx_fwd_is_long else -1.0
    pay_t = book.swap_payment_times
    pay_d = book.swap_payment_deltas
    for i in range(n_times):
        t = float(times[i])
        rt, sft, seqt = r[:, i], s_fx[:, i], s_eq[:, i]
        # IR swap: floating leg = 1 - P(t, T_sw); fixed leg = K * sum d_j P(t, T_j)
        tau_sw = book.swap_maturity - t
        if tau_sw > _MATURITY_TOL:
            rem = pay_t[pay_t > t + _MATURITY_TOL] - t
            p_end = zcb(rt, tau_sw, kr, th, sg)
            annuity = np.zeros(n_paths)
            for tau_j, d_j in zip(rem, pay_d[pay_t > t + _MATURITY_TOL], strict=True):
                annuity += d_j * zcb(rt, float(tau_j), kr, th, sg)
            comp["swap"][:, i] = (
                sign_sw * book.swap_notional * ((1.0 - p_end) - book.swap_fixed_rate * annuity)
            )
        # FX forward: long foreign -> S e^{-r_f tau} - K P(t, tau)
        tau_f = book.fx_fwd_maturity - t
        if tau_f > _MATURITY_TOL:
            comp["fx_forward"][:, i] = (
                sign_fx
                * book.fx_fwd_notional
                * (
                    sft * math.exp(-book.foreign_rate * tau_f)
                    - book.fx_fwd_strike_price * zcb(rt, tau_f, kr, th, sg)
                )
            )
        # Equity call: analytic BS on the simulated state, payoff at maturity
        tau_e = book.eq_opt_maturity - t
        if tau_e > _MATURITY_TOL:
            comp["equity_call"][:, i] = book.eq_opt_notional * _bs_call(
                seqt, book.eq_opt_strike, tau_e, rt, book.eq_div_yield, book.sigma_eq
            )
        elif abs(tau_e) <= _MATURITY_TOL:
            comp["equity_call"][:, i] = book.eq_opt_notional * np.maximum(
                seqt - book.eq_opt_strike, 0.0
            )
        total[:, i] = comp["swap"][:, i] + comp["fx_forward"][:, i] + comp["equity_call"][:, i]
    return total, comp


# --------------------------------------------------------------- exposures


def exposure_profile(sim: ExposureSimulation) -> dict[str, Array]:
    """Exposure curves EPE(t) / EE(t) / ENE(t) and quantiles (Gregory 2020).

    ``ee`` = E[V^+(t)]; ``ene`` = E[max(-V(t), 0)] (magnitude, >= 0);
    ``epe`` = Gregory's time-averaged EPE, ``(1/t) int_0^t EE(s) ds`` (with
    ``epe(0) = ee(0)``); ``q95``/``q99`` are cross-path quantiles of V^+;
    ``discounted_ee`` = E[D(0,t) V^+(t)] is the Pykhtin-Zhu CVA integrand.
    """
    if not isinstance(sim, ExposureSimulation):
        raise ValueError("sim must be an ExposureSimulation")
    vp = sim.positive_exposure()
    ee = np.asarray(vp.mean(axis=0), dtype=float)
    ene = np.asarray(np.maximum(-sim.values, 0.0).mean(axis=0), dtype=float)
    dt = np.diff(sim.times)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (ee[1:] + ee[:-1]) * dt)])
    epe = ee.copy()
    nz = sim.times > 0.0
    epe[nz] = cum[nz] / sim.times[nz]
    q95 = np.asarray(np.quantile(vp, 0.95, axis=0), dtype=float)
    q99 = np.asarray(np.quantile(vp, 0.99, axis=0), dtype=float)
    return {
        "times": np.array(sim.times),
        "ee": ee,
        "epe": epe,
        "ene": ene,
        "q95": q95,
        "q99": q99,
        "discounted_ee": np.asarray(sim.discounted_positive_exposure().mean(axis=0), dtype=float),
        "mean_value": np.asarray(sim.values.mean(axis=0), dtype=float),
    }


# ----------------------------------------------------------- hazard curves


def cumulative_hazard(times: Array, hazard: float | Array) -> Array:
    """Cumulative hazard ``Lambda(t_i)`` for a constant or term hazard curve.

    ``hazard`` is either a scalar (constant intensity) or an array of length
    ``len(times) - 1`` of piecewise-constant intensities, ``hazard[i-1]``
    applying on ``[t_{i-1}, t_i]``.  Fail-closed on negative/non-finite.
    """
    t = _validate_times(times)
    dt = np.diff(t)
    if isinstance(hazard, float | int):
        lam_val = float(hazard)
        if not math.isfinite(lam_val) or lam_val < 0.0:
            raise ValueError(f"hazard must be finite and >= 0, got {hazard!r}")
        lam = np.full(dt.size, lam_val)
    else:
        lam = np.asarray(hazard, dtype=float).ravel()
        if lam.size != dt.size or not np.isfinite(lam).all() or np.any(lam < 0.0):
            raise ValueError("term hazard must be finite, >= 0, length len(times) - 1")
    return np.asarray(np.concatenate([[0.0], np.cumsum(lam * dt)]), dtype=float)


def survival_prob(times: Array, hazard: float | Array) -> Array:
    """Survival curve ``S(t) = exp(-Lambda(t))``."""
    return np.asarray(np.exp(-cumulative_hazard(times, hazard)), dtype=float)


def default_prob_increments(times: Array, hazard: float | Array) -> Array:
    """Interval default probabilities ``S(t_{i-1}) - S(t_i)``, length n-1."""
    s = survival_prob(times, hazard)
    return np.asarray(-np.diff(s), dtype=float)


# --------------------------------------------------------------------- CVA


def compute_cva(
    sim: ExposureSimulation,
    *,
    recovery: float = 0.4,
    hazard: float | Array = 0.01,
    mode: str = "independent",
    rho_wwr: float = 0.0,
    wwr_factor: str = "rate",
    wwr_sign: int = 1,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | str | Array | bool]:
    """CVA = (1 - R) * sum_i E[D(0,t_i) V^+(t_i)] * dPD_i (Pykhtin & Zhu 2007).

    ``mode="independent"`` weights the discounted EE curve by the exact
    hazard-curve default probabilities (deterministic; supports constant and
    term hazards).  ``mode="copula_wwr"`` instead simulates one default time
    per path through a one-factor Gaussian copula (Li 2000) between the
    counterparty latent ``L = sqrt(rho) * (wwr_sign * M) + sqrt(1 - rho) * eps``
    and the standardised terminal factor latent ``M`` (``wwr_factor`` in
    {"rate", "fx", "equity"}), with ``tau = Lambda^{-1}(-ln(1 - Phi(L)))``.
    The marginal law of ``tau`` is exactly the hazard curve for any rho, so
    ``rho_wwr = 0`` reproduces the independent CVA up to Monte Carlo error.
    ``wwr_sign = -1`` places defaults in high-factor states, ``+1`` in
    low-factor states: e.g. ``wwr_factor="equity", wwr_sign=-1`` plants the
    classic option-writer WWR (default when equity rallies, exposure peaks),
    while ``wwr_factor="rate", wwr_sign=+1`` plants defaults in low-rate
    states (WWR for receiver-side rate exposure).
    """
    if not isinstance(sim, ExposureSimulation):
        raise ValueError("sim must be an ExposureSimulation")
    if not (isinstance(recovery, float | int) and math.isfinite(float(recovery))):
        raise ValueError(f"recovery must be finite, got {recovery!r}")
    if not 0.0 <= float(recovery) < 1.0:
        raise ValueError(f"recovery must lie in [0, 1), got {recovery!r}")
    if mode not in ("independent", "copula_wwr"):
        raise ValueError(f"mode must be 'independent' or 'copula_wwr', got {mode!r}")
    times = sim.times
    surv = survival_prob(times, hazard)
    dpd = -np.diff(surv)
    lgd = 1.0 - float(recovery)
    dee = sim.discounted_positive_exposure().mean(axis=0)

    if mode == "independent":
        contrib = lgd * np.asarray(dee, dtype=float)[1:] * dpd
        return {
            "cva": float(contrib.sum()),
            "mode": "independent",
            "recovery": float(recovery),
            "survival": surv,
            "pd_increments": dpd,
            "cva_increments": contrib,
            "discounted_ee": np.asarray(dee, dtype=float),
        }

    # ---- copula WWR mode
    if not (isinstance(rho_wwr, float | int) and math.isfinite(float(rho_wwr))):
        raise ValueError(f"rho_wwr must be finite, got {rho_wwr!r}")
    rho = float(rho_wwr)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho_wwr must lie in [0, 1), got {rho_wwr!r}")
    if wwr_factor not in _VALID_WWR_FACTORS:
        raise ValueError(f"wwr_factor must be one of {_VALID_WWR_FACTORS}, got {wwr_factor!r}")
    if int(wwr_sign) not in (-1, 1):
        raise ValueError(f"wwr_sign must be +1 or -1, got {wwr_sign!r}")
    if wwr_factor not in sim.factor_terminals:
        raise ValueError(
            f"copula_wwr needs a factor-simulated book; latent {wwr_factor!r} absent "
            "(ExposureSimulation.from_arrays carries no factor terminals)"
        )
    x = np.asarray(sim.factor_terminals[wwr_factor], dtype=float).ravel()
    sd = float(np.std(x))
    if sd < 1e-12:
        raise ValueError(f"WWR factor {wwr_factor!r} is degenerate (zero variance)")
    m = float(wwr_sign) * (x - float(np.mean(x))) / sd
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(sim.n_paths)
    latent = math.sqrt(rho) * m + math.sqrt(1.0 - rho) * eps
    u = np.clip(np.asarray(stats.norm.cdf(latent), dtype=float), 0.0, 1.0 - 1e-15)
    energy = -np.log1p(-u)  # = -ln(1 - Phi(L)), standard-exponential draw
    lam_grid = cumulative_hazard(times, hazard)
    tau = np.full(sim.n_paths, np.inf)
    inside = energy <= lam_grid[-1]
    tau[inside] = np.interp(energy[inside], lam_grid, times)
    idx = np.searchsorted(times, tau, side="left")
    rows = np.arange(sim.n_paths)
    ok = np.isfinite(tau) & (idx >= 1) & (idx < times.size)
    loss = np.zeros(sim.n_paths)
    loss[ok] = (
        lgd * sim.discounts[rows[ok], idx[ok]] * np.maximum(sim.values[rows[ok], idx[ok]], 0.0)
    )
    contrib_full = (
        np.asarray(np.bincount(idx[ok], weights=loss[ok], minlength=times.size), dtype=float)
        / sim.n_paths
    )
    cva = float(loss.mean())
    T = float(times[-1])
    pd_T = float(1.0 - surv[-1])
    m_nodes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    if 0.0 < rho < 1.0 and 0.0 < pd_T < 1.0:
        cond_pd = np.array(
            [float(conditional_default_prob(np.array([pd_T]), rho, float(mm))[0]) for mm in m_nodes]
        )
    else:
        cond_pd = np.full(m_nodes.size, pd_T)
    return {
        "cva": cva,
        "mode": "copula_wwr",
        "recovery": float(recovery),
        "rho_wwr": rho,
        "wwr_factor": str(wwr_factor),
        "wwr_sign": int(wwr_sign),
        "seed": int(seed),
        "survival": surv,
        "pd_increments": dpd,
        "cva_increments": contrib_full[1:],
        "default_rate_horizon": float(np.mean(tau <= T)),
        "pd_horizon": pd_T,
        "wwr_default_times": tau,
        "wwr_latent_signed_m": m,
        "wwr_m_nodes": m_nodes,
        "wwr_conditional_pd": cond_pd,
    }


# --------------------------------------------------------------------- FVA


def compute_fva(
    sim: ExposureSimulation,
    *,
    funding_spread: float,
    collateral: float | Array | None = None,
    include_funding_benefit: bool = False,
) -> dict[str, float | str | Array | bool]:
    """FVA under the Hull & White (2012) borrowing-cost convention.

    ``FVA = s_f * E[int_0^T D(0,t) max(V_t - C_t, 0) dt]`` — the cost of
    funding the uncollateralised positive MtM at OIS + ``funding_spread``.
    One-sided by default: no lending benefit on negative funding gaps (Hull &
    White argue the benefit is not available to the dealer); set
    ``include_funding_benefit=True`` for the symmetric two-sided net variant
    ``cost - benefit``.  ``collateral`` may be None (uncollateralised), a
    scalar, a ``(n_times,)`` schedule, or a full ``(n_paths, n_times)`` path
    matrix.  Trapezoidal time integration on the simulation grid.
    """
    if not isinstance(sim, ExposureSimulation):
        raise ValueError("sim must be an ExposureSimulation")
    if not (isinstance(funding_spread, float | int) and math.isfinite(float(funding_spread))):
        raise ValueError(f"funding_spread must be finite, got {funding_spread!r}")
    sf = float(funding_spread)
    if sf < 0.0:
        raise ValueError(f"funding_spread must be >= 0, got {funding_spread!r}")
    if collateral is None:
        coll = np.zeros_like(sim.values)
    else:
        coll = np.asarray(
            np.broadcast_to(np.asarray(collateral, dtype=float), sim.values.shape), dtype=float
        )
        if not np.isfinite(coll).all():
            raise ValueError("collateral contains non-finite entries")
    gap_pos = np.maximum(sim.values - coll, 0.0)
    gap_neg = np.maximum(coll - sim.values, 0.0)
    cost = sf * float(np.mean(_trapz_paths(sim.discounts * gap_pos, sim.times)))
    benefit = sf * float(np.mean(_trapz_paths(sim.discounts * gap_neg, sim.times)))
    fva = cost - benefit if include_funding_benefit else cost
    return {
        "fva": float(fva),
        "fva_funding_cost": cost,
        "fva_funding_benefit": benefit,
        "funding_spread": sf,
        "include_funding_benefit": bool(include_funding_benefit),
        "convention": (
            "hull_white_2012_two_sided_net"
            if include_funding_benefit
            else "hull_white_2012_one_sided_borrowing_cost"
        ),
        "funding_gap_curve": np.asarray(gap_pos.mean(axis=0), dtype=float),
        "discounted_gap_curve": np.asarray((sim.discounts * gap_pos).mean(axis=0), dtype=float),
    }


# --------------------------------------------------------------------- MVA


def compute_mva(
    sim: ExposureSimulation,
    *,
    alpha: float = 0.06,
    quantile: float = 0.99,
    mpor_years: float = 10.0 / 252.0,
    capital_cost_rate: float = 0.10,
) -> dict[str, float | str | Array]:
    """MVA: charged, discounted expected capital over MPoR windows.

    Andersen, Choudhury & Xing (2019) KVA-style: capital at node ``t_i`` is
    ``K_i = alpha * VaR_q[V^+(t_i + delta) | F_{t_i}]`` with margin period of
    risk ``delta = mpor_years``.  The conditional VaR uses their
    "computationally efficient" frozen-local-normal approximation
    ``K_i = alpha * max(V_i + z_q * sigma_i * sqrt(delta), 0)`` where
    ``sigma_i`` is the cross-path local volatility of the portfolio value
    over ``[t_i, t_{i+1}]`` (no nested Monte Carlo).  Then
    ``MVA = capital_cost_rate * E[int_0^T D(0,t) K_t dt]``.  Exactly 0 when
    ``alpha = 0``; linear in ``alpha`` and ``capital_cost_rate``.
    """
    if not isinstance(sim, ExposureSimulation):
        raise ValueError("sim must be an ExposureSimulation")
    if not (isinstance(alpha, float | int) and math.isfinite(float(alpha))) or float(alpha) < 0.0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha!r}")
    if not (isinstance(quantile, float | int) and math.isfinite(float(quantile))):
        raise ValueError(f"quantile must be finite, got {quantile!r}")
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError(f"quantile must lie in (0.5, 1), got {quantile!r}")
    if not (isinstance(mpor_years, float | int) and math.isfinite(float(mpor_years))):
        raise ValueError(f"mpor_years must be finite, got {mpor_years!r}")
    if float(mpor_years) <= 0.0:
        raise ValueError(f"mpor_years must be > 0, got {mpor_years!r}")
    if not (isinstance(capital_cost_rate, float | int) and math.isfinite(float(capital_cost_rate))):
        raise ValueError(f"capital_cost_rate must be finite, got {capital_cost_rate!r}")
    if float(capital_cost_rate) < 0.0:
        raise ValueError(f"capital_cost_rate must be >= 0, got {capital_cost_rate!r}")
    if sim.n_times < 3 or sim.n_paths < 2:
        raise ValueError("MVA needs n_times >= 3 and n_paths >= 2")
    dt = np.diff(sim.times)
    dv = sim.values[:, 1:] - sim.values[:, :-1]
    sigma = np.asarray(np.std(dv, axis=0) / np.sqrt(dt), dtype=float)
    sigma = np.concatenate([sigma, [sigma[-1]]])
    z_q = float(stats.norm.ppf(float(quantile)))
    horizon_shock = z_q * sigma * math.sqrt(float(mpor_years))
    capital = float(alpha) * np.maximum(sim.values + horizon_shock[None, :], 0.0)
    mva = float(capital_cost_rate) * float(
        np.mean(_trapz_paths(sim.discounts * capital, sim.times))
    )
    return {
        "mva": mva,
        "capital_curve": np.asarray(capital.mean(axis=0), dtype=float),
        "exposure_vol_curve": sigma,
        "alpha": float(alpha),
        "quantile": float(quantile),
        "mpor_years": float(mpor_years),
        "capital_cost_rate": float(capital_cost_rate),
        "convention": "andersen_choudhury_xing_2019_local_normal_mpor",
    }


# ------------------------------------------------------------------- suite


def compute_xva_suite(
    sim: ExposureSimulation,
    *,
    recovery: float = 0.4,
    hazard: float | Array = 0.01,
    funding_spread: float = 0.005,
    alpha: float = 0.06,
    quantile: float = 0.99,
    mpor_years: float = 10.0 / 252.0,
    capital_cost_rate: float = 0.10,
    mode: str = "independent",
    rho_wwr: float = 0.0,
    wwr_factor: str = "rate",
    wwr_sign: int = 1,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | str]:
    """CVA + FVA + MVA in one call (SYNTHETIC diagnostics, never market evidence)."""
    cva_out = compute_cva(
        sim,
        recovery=recovery,
        hazard=hazard,
        mode=mode,
        rho_wwr=rho_wwr,
        wwr_factor=wwr_factor,
        wwr_sign=wwr_sign,
        seed=seed,
    )
    fva_out = compute_fva(sim, funding_spread=funding_spread)
    mva_out = compute_mva(
        sim,
        alpha=alpha,
        quantile=quantile,
        mpor_years=mpor_years,
        capital_cost_rate=capital_cost_rate,
    )
    prof = exposure_profile(sim)
    return {
        "cva": _as_float(cva_out["cva"]),
        "fva": _as_float(fva_out["fva"]),
        "mva": _as_float(mva_out["mva"]),
        "cva_mode": str(cva_out["mode"]),
        "epe_final": float(np.asarray(prof["epe"], dtype=float)[-1]),
        "ee_q99_final": float(np.asarray(prof["q99"], dtype=float)[-1]),
        "source": sim.source,
        "claim": _CLAIM,
    }


# ------------------------------------------------------------------- bench


def bench_xva(
    *,
    n_paths: int = 6000,
    n_steps: int = 180,
    seed: int = DEFAULT_SEED,
    hazard: float = 0.02,
    recovery: float = 0.4,
    funding_spread: float = 0.005,
    alpha: float = 0.06,
    quantile: float = 0.99,
    rho_wwr: float = 0.6,
) -> dict[str, float | str]:
    """SYNTHETIC fixture bench for the XVA lane. Price diagnostics only.

    Engine-correctness evidence on seeded simulated paths (never market
    evidence, no live counterparty claims): the CVA/FVA/MVA levels, the
    Gaussian-copula WWR uplift vs the rho = 0 common-random-numbers baseline
    and the right-way mirror (planted equity-writer scenario: the
    counterparty defaults when equity rallies, i.e. ``wwr_factor="equity"``
    with ``wwr_sign=-1``, exactly when the long call is in the money), the
    closed-form deterministic-exposure CVA anchor error, and the
    exposure-profile summary.
    """
    if int(n_paths) < 2 or int(n_steps) < 2:
        raise ValueError("n_paths and n_steps must be >= 2")
    book = SyntheticBook()
    sim = simulate_exposure(book, n_paths=int(n_paths), n_steps=int(n_steps), seed=int(seed))
    suite = compute_xva_suite(
        sim,
        recovery=recovery,
        hazard=hazard,
        funding_spread=funding_spread,
        alpha=alpha,
        quantile=quantile,
    )
    mc0 = _as_float(
        compute_cva(
            sim,
            recovery=recovery,
            hazard=hazard,
            mode="copula_wwr",
            rho_wwr=0.0,
            wwr_factor="equity",
            wwr_sign=-1,
            seed=int(seed) + 1,
        )["cva"]
    )
    wwr = _as_float(
        compute_cva(
            sim,
            recovery=recovery,
            hazard=hazard,
            mode="copula_wwr",
            rho_wwr=float(rho_wwr),
            wwr_factor="equity",
            wwr_sign=-1,
            seed=int(seed) + 1,
        )["cva"]
    )
    rwr = _as_float(
        compute_cva(
            sim,
            recovery=recovery,
            hazard=hazard,
            mode="copula_wwr",
            rho_wwr=float(rho_wwr),
            wwr_factor="equity",
            wwr_sign=1,
            seed=int(seed) + 1,
        )["cva"]
    )
    if mc0 <= 0.0:
        raise ValueError("bench requires a strictly positive hazard/exposure (mc0 CVA <= 0)")
    # closed-form anchor: deterministic exposure -> CVA = (1-R) PD EE exactly
    t_cf = np.linspace(0.0, 10.0, 201)
    det = ExposureSimulation.from_arrays(
        t_cf, np.full((1, t_cf.size), 100.0), np.ones((1, t_cf.size))
    )
    cf = _as_float(compute_cva(det, recovery=recovery, hazard=hazard)["cva"])
    expected = (1.0 - float(recovery)) * 100.0 * (1.0 - math.exp(-float(hazard) * 10.0))
    prof = exposure_profile(sim)
    surv_T = float(survival_prob(sim.times, hazard)[-1])
    return {
        "synthetic_xva_cva": _as_float(suite["cva"]),
        "synthetic_xva_fva": _as_float(suite["fva"]),
        "synthetic_xva_mva": _as_float(suite["mva"]),
        "synthetic_xva_cva_mc_rho0": mc0,
        "synthetic_xva_cva_wwr": wwr,
        "synthetic_xva_cva_rwr": rwr,
        "synthetic_xva_wwr_uplift": wwr / mc0 - 1.0,
        "synthetic_xva_rwr_relief": 1.0 - rwr / mc0,
        "synthetic_xva_wwr_factor": "equity",
        "synthetic_xva_epe_final": _as_float(suite["epe_final"]),
        "synthetic_xva_ee_q99_max": float(np.max(np.asarray(prof["q99"], dtype=float))),
        "synthetic_xva_pd_horizon": 1.0 - surv_T,
        "synthetic_xva_closed_form_abs_err": abs(cf - expected),
        "synthetic_n_paths": float(n_paths),
        "synthetic_n_steps": float(n_steps),
        "synthetic_seed": float(seed),
        "synthetic_hazard": float(hazard),
        "synthetic_recovery": float(recovery),
        "synthetic_funding_spread": float(funding_spread),
        "synthetic_alpha": float(alpha),
        "synthetic_quantile": float(quantile),
        "synthetic_rho_wwr": float(rho_wwr),
        "synthetic_dgp": "fixture",
        "synthetic_source": _SOURCE,
        "synthetic_claim": _CLAIM,
    }
