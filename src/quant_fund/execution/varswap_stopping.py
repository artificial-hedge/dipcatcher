"""Optimal entry/exit rules for a perpetual variance swap in closed form.

Implements the optimal-stopping solution of

    Maeda, J. (2026). "Optimal entry and exit for variance swaps: closed-form
    rules for the perpetual contract." arXiv:2609.19102 [q-fin.MF] (v2,
    28 Sep 2026; citation verified against the arXiv abstract page).

for a perpetual, continuously settled variance swap under a two-measure
Heston/CIR variance law.  Under the physical measure P the instantaneous
variance follows ``dv = kappa_P (theta_P - v) dt + gamma sqrt(v) dW``; under
the pricing measure Q the same CIR form holds at speed ``kappa_Q``, with the
affine link ``kappa_P * theta_P = kappa_Q * theta_Q`` maintained throughout
(the paper's eq. 12) and the variance risk premium carried by the single
parameter ``lam := kappa_Q - kappa_P`` (a difference of mean-reversion
speeds).  ``theta_Q`` is therefore derived, not supplied: passing the paper's
rounded four-parameter calibrations would violate the link by a rounding
residual.

Reductions implemented (paper section references)
-------------------------------------------------
- Sec. 3-4: dated contract.  ``A_t + Phi_0(t, v)`` is a Q-martingale
  (Lemma 2.1) so no genuine impatience exists; under P the accrued variance
  separates exactly (Prop. 3.1) and the contract enters the free-boundary
  problem only through the forcing term ``A Phi = eps lam g_Q(t) v +
  (s - c_m)`` (Prop. 4.1), whose sign fixes the exercise-region geometry
  (Table 1).  ``myopic_level`` is the zero-carry benchmark (Def. 4.2).
- Sec. 6: perpetual contract paying ``(v - K) dt`` until an independent
  exponential time of rate ``delta`` (Carr 1998 randomised maturity).  The
  running reward is removed exactly (Lemma 6.4) leaving the affine reduced
  reward ``h~ = eps D(v) - s_bar + c_m/delta = alpha + beta v``
  (Prop. 6.5) where ``D = P_Q - P_P`` is the premium embedded in the
  perpetual rate (eq. 34) and the entry strike cancels (Cor. 6.9).  The exit
  threshold is the unique root of the smooth-pasting equation
  ``beta X(b) - h~(b) X'(b) = 0`` in the confluent hypergeometric eigenpairs
  ``F = M(a, nu; sigma v)`` (increasing) and ``G = U(a, nu; sigma v)``
  (decreasing) of the P-generator at level delta (Thm. 6.11): ``X = F``
  when ``beta > 0`` (exercise the upper set) and ``X = G`` when
  ``beta < 0`` (exercise the lower set).
- Sec. 7: the entry problem (``J = sup E[e^{-delta zeta} rho(v_zeta)]`` with
  obstacle ``rho = U - h~ - s_bar - s_e_bar``) is solved by the same
  smooth-pasting construction on the *other* eigenfunction (Thm. 7.8).  The
  holding costs admitting a reachable entry-exit pair form an interval
  (Prop. 7.6, Fig. 2) whose edges we solve numerically; both thresholds are
  strictly increasing in ``c_m`` with the closed-form sensitivities of
  Prop. 7.9.  ``c_0`` (opportunity cost of idle capital) shifts the entry
  obstacle by ``c_0/delta`` (Sec. 7.3) and is the only cost that can pull the
  entry threshold out of the deep upper tail; at ``c_0/delta >= s_bar +
  s_e_bar`` the flat trader enters at once.

Honesty
-------
This module produces *timing thresholds* and proper diagnostics only: the
stationary-Gamma CDF levels (reachability percentiles), smooth-pasting
residuals, second-order/maximum checks, and comparative-static signs.  No
Sharpe/Sortino/P&L/NAV-style quantity is computed or reported; any simulated
value used by tests or ``bench_varswap_stopping`` is a SYNTHETIC correctness
check (labelled ``synthetic_*``), never market evidence.  No live-trading
claims: the thresholds are mathematical rules, not broker instructions.
Fail-closed throughout: degenerate parameters (non-positive speeds, vols,
termination rate; violated Feller ``nu > 1``), non-finite inputs, a broken
affine link, and a non-bracketing or non-converging boundary solve all raise
rather than return a silently wrong threshold.

Composition
-----------
Reuses ``models.short_rate.cir_simulate`` for seeded CIR paths in tests (it
owns CIR path simulation; this module needs only the law, not the sampler) and
follows the proper-score conventions of ``metrics.*`` for its diagnostics.
``models.variance_swap.variance_swap_fair_strike`` is deliberately NOT reused:
it prices the model-free replication strike of a *dated* swap from an OTM
option strip, a different object from the perpetual fair rate ``K(v)`` of
eq. (30).  The confluent-hypergeometric eigenpairs are scipy's ``hyp1f1`` /
``hyperu`` with the Buchholz (1969) derivative identities (paper eq. 27).

References
----------
- Maeda, J. (2026). Optimal entry and exit for variance swaps: closed-form
  rules for the perpetual contract. arXiv:2609.19102 [q-fin.MF].
- Dayanik, S. and Karatzas, I. (2003). On the optimal stopping problem for
  one-dimensional diffusions. (delta-excessive majorant reduction, Prop. 5.2/5.3)
- Leung, T., Li, X. and Wang, Z. (2014). Optimal starting-stopping and
  switching of a CIR process. (affine reward on CIR)
- Dynkin, E.B. (1963). Optimal choice of the stopping moment of a Markov
  process. (Thm. 5.1)
- Going-Jaeschke, A. and Yor, M. (2003). A survey and some generalizations of
  Bessel processes. (F/G monotonicity)
- Buchholz, H. (1969). The Confluent Hypergeometric Function. (eq. 27)
- Carr, P. (1998). Randomization and the American put. Review of Financial
  Studies. (exponential maturity device)
- Gatheral, J. (2006). The Volatility Surface. Wiley. (fair-strike form)
- Carr, P. and Wu, L. (2009). Variance risk premiums. Review of Financial
  Studies. (empirical sign of the premium)
- Cox, J., Ingersoll, J. and Ross, S. (1985). A theory of the term structure
  of interest rates. Econometrica 53(2). (CIR dynamics)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincinv, hyp1f1, hyperu

Array = NDArray[np.float64]
ScalarOrArray = float | Array

#: Reachability tolerance of the paper's Definition 7.5.
DEFAULT_ETA = 0.005
#: Upper argmax scan cap: sigma * v beyond this the M-series is numerically
#: useless (exp overflow far above ~700); reachable thresholds never get close.
_Z_HI_CAP = 400.0
#: Root bracketing / grid resolution.
_N_GRID = 512
_RTOL = 1e-12
_XTOL = 1e-14


def _require_finite_positive(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {x}")
    return v


def _require_finite_nonneg(x: float, name: str) -> float:
    v = float(x)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {x}")
    return v


def _scalar(x: object) -> float:
    """float() over the float|Array union (0-d arrays route through .item())."""
    return float(np.asarray(x, dtype=float).item())


@dataclass(frozen=True)
class CIRTwoMeasure:
    """P/Q CIR variance dynamics under the affine link (paper eq. 11-12).

    ``dv = kappa (theta - v) dt + gamma sqrt(v) dW`` under each measure, with
    ``theta_q = kappa_p * theta_p / kappa_q`` implied by the link
    ``kappa_P theta_P = kappa_Q theta_Q`` and ``lam = kappa_Q - kappa_P`` the
    (single) variance-risk-premium parameter.  The Feller condition
    ``nu = 2 kappa_P theta_P / gamma^2 > 1`` (paper eq. 7, identical under both
    measures by the link) makes 0 an entrance boundary and the state space
    ``(0, inf)``.  Fail-closed on non-positive or non-finite parameters and on
    ``nu <= 1``.
    """

    kappa_p: float
    theta_p: float
    kappa_q: float
    gamma: float

    def __post_init__(self) -> None:
        kp = _require_finite_positive(self.kappa_p, "kappa_p")
        tp = _require_finite_positive(self.theta_p, "theta_p")
        kq = _require_finite_positive(self.kappa_q, "kappa_q")
        gm = _require_finite_positive(self.gamma, "gamma")
        object.__setattr__(self, "kappa_p", kp)
        object.__setattr__(self, "theta_p", tp)
        object.__setattr__(self, "kappa_q", kq)
        object.__setattr__(self, "gamma", gm)
        if not self.nu > 1.0:
            raise ValueError(
                f"Feller condition violated: nu = 2 kappa_p theta_p / gamma^2 "
                f"= {self.nu} must be > 1 (0 must be an entrance boundary)"
            )

    @property
    def theta_q(self) -> float:
        """Pricing-measure long-run variance implied by the affine link."""
        return self.kappa_p * self.theta_p / self.kappa_q

    @property
    def lam(self) -> float:
        """Variance risk premium lam = kappa_Q - kappa_P (paper eq. 12)."""
        return self.kappa_q - self.kappa_p

    @property
    def nu(self) -> float:
        """Stationary-law shape nu = 2 kappa_P theta_P / gamma^2 (eq. 24)."""
        return 2.0 * self.kappa_p * self.theta_p / self.gamma**2

    @property
    def sigma_tilde(self) -> float:
        """Stationary-law rate varsigma = 2 kappa_P / gamma^2 (eq. 24)."""
        return 2.0 * self.kappa_p / self.gamma**2


@dataclass(frozen=True)
class PerpetualVarSwap:
    """The perpetual contract + cost vector (paper Sec. 3.2, 6.1, 7.3).

    The contract pays ``eps (v - K) dt`` per unit variance notional until an
    independent exponential termination time of rate ``delta`` (eq. 31);
    ``1/delta`` is the expected contract life.  Unwinding is a tear-up at flat
    concession ``s_exit >= 0``; entry concession ``s_entry >= 0``; running
    holding charge ``c_hold`` of either sign; opportunity cost of idle capital
    ``c_idle >= 0`` (paper Sec. 7.3, charged while flat).  ``side`` is
    ``+1`` (long) or ``-1`` (short; the only side worth opening under the
    empirical premium sign, Remark 7.3).
    """

    cir: CIRTwoMeasure
    delta: float
    s_exit: float
    s_entry: float
    c_hold: float
    c_idle: float = 0.0
    side: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.cir, CIRTwoMeasure):
            raise ValueError("cir must be a CIRTwoMeasure spec")
        object.__setattr__(self, "delta", _require_finite_positive(self.delta, "delta"))
        object.__setattr__(self, "s_exit", _require_finite_nonneg(self.s_exit, "s_exit"))
        object.__setattr__(self, "s_entry", _require_finite_nonneg(self.s_entry, "s_entry"))
        object.__setattr__(self, "c_idle", _require_finite_nonneg(self.c_idle, "c_idle"))
        ch = float(self.c_hold)
        if not np.isfinite(ch):
            raise ValueError(f"c_hold must be finite; got {self.c_hold}")
        object.__setattr__(self, "c_hold", ch)
        if self.side not in (+1, -1):
            raise ValueError(f"side must be +1 (long) or -1 (short); got {self.side}")

    @property
    def a(self) -> float:
        """Hypergeometric first parameter a = delta / kappa_P (eq. 24)."""
        return self.delta / self.cir.kappa_p

    @property
    def eps(self) -> float:
        """Position sign eps in {+1, -1}."""
        return float(self.side)

    @property
    def k_round(self) -> float:
        """Round-trip cost k = s_bar + s_e_bar (Lemma 7.7 notation)."""
        return self.s_exit + self.s_entry

    @property
    def beta(self) -> float:
        """Reduced-reward slope beta = eps * D' (Prop. 6.5 / eq. 35)."""
        c = self.cir
        return (
            self.eps
            * (c.kappa_p - c.kappa_q)
            / ((self.delta + c.kappa_q) * (self.delta + c.kappa_p))
        )

    @property
    def alpha(self) -> float:
        """Reduced-reward intercept alpha (Prop. 6.5 eq. 33)."""
        c = self.cir
        d0 = (
            (c.theta_q - c.theta_p) / self.delta
            - c.theta_q / (self.delta + c.kappa_q)
            + c.theta_p / (self.delta + c.kappa_p)
        )
        return self.eps * d0 - self.s_exit + self.c_hold / self.delta

    @property
    def alpha_idle(self) -> float:
        """Idle-cost shift of the entry obstacle: c_0 / delta (Sec. 7.3)."""
        return self.c_idle / self.delta

    def reduced_reward(self, v: ScalarOrArray) -> ScalarOrArray:
        """Affine reduced reward h~(v) = alpha + beta v (Prop. 6.5)."""
        return self.alpha + self.beta * np.asarray(v, dtype=float)

    def premium_gap(self, v: ScalarOrArray) -> ScalarOrArray:
        """D(v) = P_Q(v) - P_P(v), the premium in the perpetual rate (eq. 34).

        Affine with slope ``-lam / ((delta + kappa_Q)(delta + kappa_P))``;
        ``h~ = eps D - s_bar + c_m / delta`` (eq. 35).
        """
        return variance_perpetuity(
            np.asarray(v, dtype=float), self.cir.kappa_q, self.cir.theta_q, self.delta
        ) - variance_perpetuity(
            np.asarray(v, dtype=float), self.cir.kappa_p, self.cir.theta_p, self.delta
        )

    def hold_value(self, v: ScalarOrArray, k_rate: float) -> ScalarOrArray:
        """R(v; K): value of holding to termination (Lemma 6.4 / eq. 28 under P)."""
        kk = float(k_rate)
        if not np.isfinite(kk):
            raise ValueError("k_rate must be finite")
        return (
            self.eps
            * variance_perpetuity(
                np.asarray(v, dtype=float), self.cir.kappa_p, self.cir.theta_p, self.delta
            )
            - self.eps * kk / self.delta
            - self.c_hold / self.delta
        )


def fair_rate(v: ScalarOrArray, kappa: float, theta: float, delta: float) -> ScalarOrArray:
    """Perpetual fair rate K(v) = theta + delta (v - theta)/(delta + kappa) (eq. 30)."""
    kk = _require_finite_positive(kappa, "kappa")
    tt = _require_finite_positive(theta, "theta")
    dd = _require_finite_positive(delta, "delta")
    vv = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite")
    return tt + dd * (vv - tt) / (dd + kk)


def variance_perpetuity(
    v: ScalarOrArray, kappa: float, theta: float, delta: float
) -> ScalarOrArray:
    """P_M(v) = theta/delta + (v - theta)/(delta + kappa) (eq. 29).

    Survival-weighted perpetuity of variance under measure M; affine in v
    with slope 1/(delta + kappa).
    """
    kk = _require_finite_positive(kappa, "kappa")
    tt = _require_finite_positive(theta, "theta")
    dd = _require_finite_positive(delta, "delta")
    vv = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite")
    return tt / dd + (vv - tt) / (dd + kk)


def stationary_cdf(cir: CIRTwoMeasure, v: ScalarOrArray) -> ScalarOrArray:
    """F_inf(v): Gamma(shape=nu, rate=sigma_tilde) stationary CDF of v under P.

    With ``scipy.stats.gamma`` the rate is ``scale = 1/rate``; computed here as
    the regularised lower incomplete gamma ``gammainc(nu, sigma_tilde * v)``,
    which needs no stats import and is exact at v = 0.
    """
    vv = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(vv)) or bool(np.any(vv < 0.0)):
        raise ValueError("v must be finite and >= 0")
    return np.asarray(gammainc(cir.nu, cir.sigma_tilde * vv), dtype=float)


def stationary_ppf(cir: CIRTwoMeasure, p: float) -> float:
    """Inverse stationary CDF via the regularised lower incomplete gamma."""
    pp = float(p)
    if not np.isfinite(pp) or not 0.0 < pp < 1.0:
        raise ValueError("p must be in (0, 1)")
    return float(gammaincinv(cir.nu, pp) / cir.sigma_tilde)


# ---------------------------------------------------------------------------
# Confluent-hypergeometric eigenfunctions of the CIR generator (Sec. 5, eq.
# 24-27): F = M(a, nu; sigma v) increasing, G = U(a, nu; sigma v) decreasing,
# both solving (L - delta) u = 0.  Derivatives via Buchholz (1969) / eq. (27):
#   M'(p, q; z) = (p/q) M(p+1, q+1; z),  U'(p, q; z) = -p U(p+1, q+1; z).
# ---------------------------------------------------------------------------


def _fg_params(spec: PerpetualVarSwap) -> tuple[float, float, float]:
    return spec.a, spec.cir.nu, spec.cir.sigma_tilde


def _F(spec: PerpetualVarSwap, v: ScalarOrArray) -> ScalarOrArray:
    a, nu, sg = _fg_params(spec)
    with np.errstate(all="ignore"):
        out = hyp1f1(a, nu, sg * np.asarray(v, dtype=float))
    return np.asarray(out, dtype=float)


def _G(spec: PerpetualVarSwap, v: ScalarOrArray) -> ScalarOrArray:
    a, nu, sg = _fg_params(spec)
    with np.errstate(all="ignore"):
        out = hyperu(a, nu, sg * np.asarray(v, dtype=float))
    return np.asarray(out, dtype=float)


def _Fp(spec: PerpetualVarSwap, v: ScalarOrArray) -> ScalarOrArray:
    a, nu, sg = _fg_params(spec)
    with np.errstate(all="ignore"):
        out = sg * (a / nu) * hyp1f1(a + 1.0, nu + 1.0, sg * np.asarray(v, dtype=float))
    return np.asarray(out, dtype=float)


def _Gp(spec: PerpetualVarSwap, v: ScalarOrArray) -> ScalarOrArray:
    a, nu, sg = _fg_params(spec)
    with np.errstate(all="ignore"):
        out = -sg * a * hyperu(a + 1.0, nu + 1.0, sg * np.asarray(v, dtype=float))
    return np.asarray(out, dtype=float)


def _Fpp(spec: PerpetualVarSwap, v: ScalarOrArray) -> ScalarOrArray:
    a, nu, sg = _fg_params(spec)
    with np.errstate(all="ignore"):
        out = (
            sg**2
            * (a * (a + 1.0) / (nu * (nu + 1.0)))
            * hyp1f1(a + 2.0, nu + 2.0, sg * np.asarray(v, dtype=float))
        )
    return np.asarray(out, dtype=float)


def _Gpp(spec: PerpetualVarSwap, v: ScalarOrArray) -> ScalarOrArray:
    a, nu, sg = _fg_params(spec)
    with np.errstate(all="ignore"):
        out = sg**2 * a * (a + 1.0) * hyperu(a + 2.0, nu + 2.0, sg * np.asarray(v, dtype=float))
    return np.asarray(out, dtype=float)


def _brentq_root(
    f,
    lo: float,
    hi: float,
    *,
    what: str,
) -> float:
    """Bracketed brentq; fail-closed on non-bracket or non-convergence."""
    flo = float(f(lo))
    fhi = float(f(hi))
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        raise RuntimeError(f"{what}: non-finite endpoint evaluation")
    if flo * fhi > 0.0:
        raise RuntimeError(
            f"{what}: no sign change on [{lo:.6g}, {hi:.6g}] (f(lo)={flo:.6g}, f(hi)={fhi:.6g})"
        )
    try:
        root = float(brentq(f, lo, hi, xtol=_XTOL, rtol=_RTOL, maxiter=200, full_output=False))
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{what}: boundary solve did not converge: {exc}") from exc
    if not np.isfinite(root):
        raise RuntimeError(f"{what}: non-finite root returned")
    return root


def _v_upper(spec: PerpetualVarSwap) -> float:
    """Scan cap: a quantile far into the stationary upper tail, z-capped."""
    q = stationary_ppf(spec.cir, 1.0 - 1e-9)
    while spec.cir.sigma_tilde * q > _Z_HI_CAP:
        q *= 0.5
    return float(max(q * 1.5, 10.0 * spec.cir.theta_p))


def _sign_change_roots(
    f,
    lo: float,
    hi: float,
    *,
    n: int = _N_GRID,
) -> list[float]:
    """All sign-change roots of ``f`` on the log grid over (lo, hi].

    Non-finite evaluations are skipped (never interpolated); each bracket is
    solved by fail-closed brentq.  Deterministic grid: fixed count, geometric
    spacing.
    """
    lo2 = max(lo, np.finfo(float).tiny)
    xs = np.exp(np.linspace(np.log(lo2), np.log(hi), n))
    fs = np.asarray(f(xs), dtype=float)
    roots: list[float] = []
    for i in range(xs.size - 1):
        f0, f1 = float(fs[i]), float(fs[i + 1])
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if f0 == 0.0:
            roots.append(float(xs[i]))
        elif f0 * f1 < 0.0:
            roots.append(
                _brentq_root(
                    lambda z: float(f(np.asarray(z))),
                    float(xs[i]),
                    float(xs[i + 1]),
                    what="sign-change root",
                )
            )
    return roots


@dataclass(frozen=True)
class ExitSolution:
    """Result of the perpetual exit solve (Thm. 6.11).

    ``region`` is ``"upper"`` (exercise on [b*, inf), beta > 0), ``"lower"``
    (exercise on (0, b*], beta < 0), ``"all"`` (beta = 0 with h~ > 0: exercise
    at once everywhere, Cor. 6.7), or ``"empty"`` (h~ <= 0 everywhere: hold to
    termination, Prop. 7.1 regime).  ``residual`` is |Psi(b*)| at the returned
    threshold; ``dpsi`` is the analytic first derivative of the smooth-pasting
    lhs at b* (``-h~(b*) X''(b*)``), negative at the unique root.
    """

    exists: bool
    threshold: float
    region: str
    alpha: float
    beta: float
    v0_star: float
    vs_star: float
    residual: float
    dpsi: float
    lam_coeff: float  # U(v) = lam_coeff * X(v) on the continuation region


def exercise_geometry(eps: float, lam: float) -> str:
    """Exercise-region geometry off the sign of eps * lam (Table 1 / Cor. 6.6).

    ``eps * lam < 0`` -> exercise into a spike (upper set, ``"upper"``);
    ``eps * lam > 0`` -> exercise into a collapse (lower set, ``"lower"``);
    ``eps * lam == 0`` -> no optionality (Cor. 6.7), ``"flat"``.
    """
    s = float(eps) * float(lam)
    if s < 0.0:
        return "upper"
    if s > 0.0:
        return "lower"
    return "flat"


def solve_exit(spec: PerpetualVarSwap) -> ExitSolution:
    """Perpetual exit threshold b* of Thm. 6.11.

    beta > 0: root of ``beta F - h~ F' = 0`` on ``(max(v0*, vs*), inf)``
    (Psi_F strictly decreasing there since ``Psi_F' = -h~ F''``); exists iff
    ``v_s* > 0``.  beta < 0: root of ``beta G - h~ G' = 0`` on
    ``(0, min(v0*, vs*))``; exists iff ``h~(0) = alpha > 0`` (equivalently
    ``c_m > c_m*`` of Prop. 7.1 for the short).  beta = 0 is the degenerate
    Cor. 6.7 case: ``h~`` constant, exercised at once iff positive.
    """
    beta = spec.beta
    alpha = spec.alpha
    kp, tp, delta = spec.cir.kappa_p, spec.cir.theta_p, spec.delta
    v0 = float("nan") if beta == 0.0 else -alpha / beta
    vs = float("nan") if beta == 0.0 else (beta * kp * tp - delta * alpha) / (beta * (kp + delta))

    if beta == 0.0:
        exists = alpha > 0.0
        return ExitSolution(
            exists=exists,
            threshold=0.0 if exists else float("nan"),
            region="all" if exists else "empty",
            alpha=alpha,
            beta=beta,
            v0_star=v0,
            vs_star=vs,
            residual=0.0,
            dpsi=float("nan"),
            lam_coeff=float("nan"),
        )

    def psi_f(b: ScalarOrArray) -> ScalarOrArray:
        return beta * _F(spec, b) - spec.reduced_reward(b) * _Fp(spec, b)

    def psi_g(b: ScalarOrArray) -> ScalarOrArray:
        return beta * _G(spec, b) - spec.reduced_reward(b) * _Gp(spec, b)

    eps_tiny = np.finfo(float).eps
    if beta > 0.0:
        # Region is upper; root lies above max(v0*, vs*).  Existence iff
        # Psi_F(0+) > 0, equivalently v_s* > 0.
        if not vs > 0.0:
            return ExitSolution(
                False,
                float("nan"),
                "empty",
                alpha,
                beta,
                v0,
                vs,
                float("nan"),
                float("nan"),
                float("nan"),
            )
        lo = max(v0, vs, 0.0)
        lo = lo * (1.0 + 1e-9) + 1e-12
        if float(psi_f(lo)) <= 0.0:
            return ExitSolution(
                False,
                float("nan"),
                "empty",
                alpha,
                beta,
                v0,
                vs,
                float("nan"),
                float("nan"),
                float("nan"),
            )
        hi = max(lo * 4.0, _v_upper(spec) * 0.25)
        while float(psi_f(hi)) > 0.0:
            hi *= 4.0
            if spec.cir.sigma_tilde * hi > _Z_HI_CAP:
                raise RuntimeError("exit solve: bracket did not close below z-cap")
        b = _brentq_root(lambda z: float(psi_f(z)), lo, hi, what="exit (beta>0)")
        resid = _scalar(abs(psi_f(b)))
        dpsi = _scalar(-spec.reduced_reward(b) * _Fpp(spec, b))
        lam_coeff = _scalar(spec.reduced_reward(b) / _F(spec, b))
        return ExitSolution(True, b, "upper", alpha, beta, v0, vs, resid, dpsi, lam_coeff)

    # beta < 0: region lower; root on (0, min(v0*, vs*)); exists iff alpha > 0.
    if not alpha > 0.0:
        return ExitSolution(
            False,
            float("nan"),
            "empty",
            alpha,
            beta,
            v0,
            vs,
            float("nan"),
            float("nan"),
            float("nan"),
        )
    hi = min(v0, vs)
    if not hi > 0.0:
        raise RuntimeError("exit solve: v0*/vs* non-positive with alpha > 0")
    lo = max(hi * 1e-9, eps_tiny)
    # Psi_G decreasing on (0, v0*): + at lo, - at hi.
    if float(psi_g(lo)) <= 0.0 or float(psi_g(hi)) >= 0.0:
        raise RuntimeError("exit solve: Psi_G bracket orientation violated")
    b = _brentq_root(lambda z: float(psi_g(z)), lo, hi, what="exit (beta<0)")
    resid = _scalar(abs(psi_g(b)))
    dpsi = _scalar(-spec.reduced_reward(b) * _Gpp(spec, b))
    lam_coeff = _scalar(spec.reduced_reward(b) / _G(spec, b))
    return ExitSolution(True, b, "lower", alpha, beta, v0, vs, resid, dpsi, lam_coeff)


def exit_value(spec: PerpetualVarSwap, sol: ExitSolution, v: ScalarOrArray) -> ScalarOrArray:
    """U(v): reduced-problem value (smallest delta-excessive majorant of h~).

    ``h~`` on the exercise region, ``h~(b*) X(v)/X(b*)`` on the continuation
    region; identically zero when the exercise region is empty.
    """
    vv = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite")
    if not sol.exists:
        return np.zeros_like(vv)
    if sol.region == "all":
        return spec.reduced_reward(vv)
    if sol.region == "upper":
        cont = sol.lam_coeff * _F(spec, vv)
        return np.asarray(np.where(vv < sol.threshold, cont, spec.reduced_reward(vv)))
    cont = sol.lam_coeff * _G(spec, vv)
    return np.asarray(np.where(vv > sol.threshold, cont, spec.reduced_reward(vv)))


def position_value(
    spec: PerpetualVarSwap, sol: ExitSolution, v: ScalarOrArray, k_rate: float
) -> ScalarOrArray:
    """V(v; K) = R(v; K) + U(v) of the unwind problem (Lemma 6.4)."""
    return spec.hold_value(v, k_rate) + exit_value(spec, sol, v)


def entry_obstacle(spec: PerpetualVarSwap, sol: ExitSolution, v: ScalarOrArray) -> ScalarOrArray:
    """rho(v) = U - h~ - k + c_0/delta: the effective entry obstacle (eq. 40, 46).

    On the exit exercise region ``U = h~`` so ``rho = -k + c_0/delta``; on the
    continuation region ``rho`` is the explicit ``Lam * X - h~ - k + c_0/delta``
    of Lemma 7.7, C^1 across b*.  When the exercise region is empty
    (``U == 0``) ``rho`` is the affine ``-h~ - k + c_0/delta`` (eq. 43).
    """
    vv = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite")
    return exit_value(spec, sol, vv) - spec.reduced_reward(vv) - spec.k_round + spec.alpha_idle


def _entry_obstacle_prime(
    spec: PerpetualVarSwap, sol: ExitSolution, v: ScalarOrArray
) -> ScalarOrArray:
    """rho'(v) on the continuation branch (eq. 49); ``0`` on the exercise set."""
    vv = np.asarray(v, dtype=float)
    if not sol.exists:
        return np.asarray(-spec.beta + 0.0 * vv)
    if sol.region == "upper":
        return np.asarray(
            np.where(vv < sol.threshold, sol.lam_coeff * _Fp(spec, vv) - spec.beta, 0.0)
        )
    return np.asarray(np.where(vv > sol.threshold, sol.lam_coeff * _Gp(spec, vv) - spec.beta, 0.0))


@dataclass(frozen=True)
class EntrySolution:
    """Result of the perpetual entry solve (Thm. 7.8).

    ``exists`` — the entry region is non-empty (rho reaches a positive
    maximum); ``immediate`` — the global argmax of ``rho/Y`` sits at the
    domain boundary, i.e. the trader enters at once whatever the level
    (Sec. 7.3, c_0/delta >= k).  ``threshold`` is d* in v units (nan when
    empty, 0.0 when immediate).  ``residual`` is |Psi_d(d*)|; ``dpsi`` is the
    analytic (gamma^2 d / 2) derivative form of eq. (56), negative at the
    unique maximum.
    """

    exists: bool
    immediate: bool
    threshold: float
    side_set: str  # "upper" entry set [d, inf) or "lower" (0, d]
    residual: float
    dpsi: float
    rho_at_d: float
    n_candidates: int


def _l_d_rho(spec: PerpetualVarSwap, sol: ExitSolution, v: float) -> float:
    """(L - delta) rho at v (eq. 47 extended by the c_0 shift).

    On the continuation branch ``rho = Lam X - h~ - k + c_0/delta`` so
    ``rho' = Lam X' - beta`` and ``rho'' = Lam X''`` — with
    ``(L - delta) X = 0`` this reduces to ``-beta * drift - delta * rho``
    plus the surviving diffusion piece ``(gamma^2 v / 2) (Lam X'' - 0)``;
    equivalently ``kappa (theta - v) rho' + (gamma^2 v/2) rho'' - delta rho``.
    """
    c = spec.cir
    r = float(entry_obstacle(spec, sol, v))
    rp = float(_entry_obstacle_prime(spec, sol, v))
    if sol.exists:
        xpp = _Fpp if sol.region == "upper" else _Gpp
        rpp = sol.lam_coeff * float(xpp(spec, v))
        if sol.region == "upper":
            rpp = float(np.where(v < sol.threshold, rpp, 0.0))
        else:
            rpp = float(np.where(v > sol.threshold, rpp, 0.0))
    else:
        rpp = 0.0
    return float(0.5 * c.gamma**2 * v * rpp + c.kappa_p * (c.theta_p - v) * rp - spec.delta * r)


def solve_entry(spec: PerpetualVarSwap) -> EntrySolution:
    """Perpetual entry threshold d* of Thm. 7.8 (unique smooth-pasting root).

    For beta < 0 (the paper's short) the entry region is the upper set
    ``[d*, inf)``: ``d*`` maximises ``rho/F`` on ``(b*, inf)``, equivalently
    the unique root of ``rho' F - rho F' = 0`` where ``(rho/F)'' < 0``.
    For beta > 0 the symmetric statement holds with F, G interchanged and the
    entry region the lower set ``(0, d*]``.  With ``c_idle`` the obstacle is
    ``rho + c_0/delta`` (Sec. 7.3): if the shifted value at the boundary of
    the exit exercise region is non-negative the flat trader enters at once.

    In the degenerate-exit regime (``U == 0``, Prop. 7.1) ``rho`` is the
    affine ``-h~ - k + c_0/delta`` and the same scan applies on ``(0, inf)``.
    ``beta == 0`` carries no optionality: enter at once iff the constant
    obstacle is positive, never otherwise.
    """
    beta = spec.beta
    if beta == 0.0:
        sol = solve_exit(spec)
        r0 = float(entry_obstacle(spec, sol, 1e-12))
        if r0 > 0.0:
            return EntrySolution(True, True, 0.0, "upper", 0.0, float("nan"), r0, 0)
        return EntrySolution(False, False, float("nan"), "upper", float("nan"), float("nan"), r0, 0)

    sol = solve_exit(spec)
    hi_cap = _v_upper(spec)
    const_ex = -spec.k_round + spec.alpha_idle  # rho on the exit exercise region

    if beta < 0.0:
        y, yp = _F, _Fp
        side_set = "upper"
        lo = sol.threshold if sol.exists else 0.0
        if sol.exists:
            # rho/Y on (0, b*] = const/F: sup is const at v -> 0 when
            # const > 0, else const/F(b*) at the right edge.
            boundary_v = 0.0 if const_ex > 0.0 else sol.threshold
            boundary_ratio = (
                const_ex if const_ex > 0.0 else const_ex / float(_F(spec, sol.threshold))
            )
        else:
            boundary_v = 0.0
            boundary_ratio = float(entry_obstacle(spec, sol, 0.0))  # rho(0) / F(0)
    else:
        y, yp = _G, _Gp
        side_set = "lower"
        lo = 0.0
        if sol.exists and const_ex > 0.0:
            # rho/G = const/G on (b*, inf) is unbounded: enter at once.
            return EntrySolution(True, True, 0.0, side_set, 0.0, float("nan"), const_ex, 0)
        boundary_v = sol.threshold if sol.exists else hi_cap
        boundary_ratio = const_ex / float(_G(spec, sol.threshold)) if sol.exists else 0.0

    def psi_d(d: ScalarOrArray) -> ScalarOrArray:
        return _entry_obstacle_prime(spec, sol, d) * y(spec, d) - entry_obstacle(spec, sol, d) * yp(
            spec, d
        )

    # Scan (lo, hi] for interior stationary points of rho/Y.  Thm. 7.8's d*
    # is the interior root; per Table 3 it is adopted whenever one exists
    # (interior maxima outrank the v -> 0 boundary until they disappear, which
    # is precisely the paper's "enter at once" regime for large c_0).
    roots = _sign_change_roots(psi_d, max(lo, np.finfo(float).tiny), hi_cap)
    best_v = -1.0
    best_ratio = float("-inf")
    n_cand = 0
    for r in roots:
        # keep only local maxima: d/dv (rho/Y) changes + -> -
        rr = float(r)
        num = float(psi_d(rr * (1.0 - 1e-7)))
        den = float(psi_d(rr * (1.0 + 1e-7)))
        if num > 0.0 > den:
            n_cand += 1
            ratio = float(entry_obstacle(spec, sol, rr) / y(spec, rr))
            if ratio > best_ratio:
                best_ratio = ratio
                best_v = rr

    if best_v < 0.0 or best_ratio <= 0.0:
        # No interior maximum (or none reaching a positive obstacle): the
        # supremum of rho/Y sits at a domain boundary — "enter at once"
        # when that value is positive, empty entry region otherwise.
        if boundary_ratio > 0.0:
            return EntrySolution(
                True,
                True,
                0.0,
                side_set,
                0.0,
                float("nan"),
                boundary_ratio * float(y(spec, boundary_v)),
                n_cand,
            )
        return EntrySolution(
            False,
            False,
            float("nan"),
            side_set,
            float("nan"),
            float("nan"),
            boundary_ratio * float(y(spec, boundary_v)),
            n_cand,
        )

    resid = _scalar(abs(psi_d(best_v)))
    dpsi = (
        2.0 / (spec.cir.gamma**2 * best_v) * _scalar(y(spec, best_v)) * _l_d_rho(spec, sol, best_v)
    )
    rho_d = _scalar(entry_obstacle(spec, sol, best_v))
    if rho_d <= 0.0:
        return EntrySolution(False, False, float("nan"), side_set, resid, dpsi, rho_d, n_cand)
    return EntrySolution(True, False, best_v, side_set, resid, dpsi, rho_d, n_cand)


def entry_value(
    spec: PerpetualVarSwap, sol: ExitSolution, esol: EntrySolution, v: ScalarOrArray
) -> ScalarOrArray:
    """J(v): value of the entry problem (eq. 40), including the idle cost.

    ``J = -c_0/delta + sup E[e^{-delta zeta} (rho + c_0/delta)(v_zeta)]``
    (Sec. 7.3): on the continuation side ``rho(d*) Y(v)/Y(d*)``, on the
    entry region ``rho(v)``, with ``Y = F`` (upper set) or ``G`` (lower).
    """
    vv = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite")
    g = entry_obstacle(spec, sol, vv)
    if not esol.exists:
        return np.zeros_like(vv)
    if esol.immediate:
        return g
    if esol.side_set == "upper":
        cont = esol.rho_at_d * _F(spec, vv) / _F(spec, esol.threshold)
        return np.asarray(np.where(vv < esol.threshold, cont, g))
    cont = esol.rho_at_d * _G(spec, vv) / _G(spec, esol.threshold)
    return np.asarray(np.where(vv > esol.threshold, cont, g))


def carry_star(spec: PerpetualVarSwap) -> float:
    """c_m* of Prop. 7.1: below it the exercise region is empty (U == 0).

    ``c_m* = delta (s_bar - sup_v eps D(v))``; under Heston ``eps D`` is affine
    so the sup sits at an endpoint: ``eps D(0)`` when ``eps D' < 0`` (the
    short's case: ``delta (s_bar + D(0))``), ``+inf`` when ``eps D' > 0``
    (the long's degenerate side, Remark 7.3).
    """
    slope = (
        spec.eps
        * (spec.cir.kappa_p - spec.cir.kappa_q)
        / ((spec.delta + spec.cir.kappa_q) * (spec.delta + spec.cir.kappa_p))
    )
    if slope > 0.0:
        return float("-inf")
    sup_ed = float(spec.eps * spec.premium_gap(0.0))
    return float(spec.delta * (spec.s_exit - sup_ed))


def threshold_sensitivities(
    spec: PerpetualVarSwap, sol: ExitSolution, esol: EntrySolution
) -> dict[str, float]:
    """Prop. 7.9 comparative statics: d(b*)/dc_m and d(d*)/dc_m in closed form.

    beta < 0 (paper's short): ``db/dc_m = -G'(b*) / (delta h~(b*) G''(b*))``
    (eq. 50) and ``dd/dc_m = (gamma^2 d* / 2 delta) *
    (W(d*)/G(b*) - F'(d*)) / (F(d*) (L - delta) rho(d*))`` (eq. 51) with
    ``W = F' G - F G'``; both strictly positive.  beta > 0 interchanges F, G
    (both strictly negative, thresholds falling in the charge).
    """
    if not sol.exists or not esol.exists or esol.immediate:
        raise ValueError("sensitivities require a non-degenerate entry-exit pair")
    b, d, delta = sol.threshold, esol.threshold, spec.delta
    h_b = float(spec.reduced_reward(b))
    if sol.region == "lower":
        db = float(-_Gp(spec, b) / (delta * h_b * _Gpp(spec, b)))
        w_d = float(_Fp(spec, d) * _G(spec, d) - _F(spec, d) * _Gp(spec, d))
        dd = float(
            (spec.cir.gamma**2 * d / (2.0 * delta))
            * (w_d / _G(spec, b) - _Fp(spec, d))
            / (_F(spec, d) * _l_d_rho(spec, sol, d))
        )
    else:
        db = float(-_Fp(spec, b) / (delta * h_b * _Fpp(spec, b)))
        w_d = float(_Gp(spec, d) * _F(spec, d) - _G(spec, d) * _Fp(spec, d))
        dd = float(
            (spec.cir.gamma**2 * d / (2.0 * delta))
            * (w_d / _F(spec, b) - _Gp(spec, d))
            / (_G(spec, d) * _l_d_rho(spec, sol, d))
        )
    return {"db_dcm": db, "dd_dcm": dd}


@dataclass(frozen=True)
class EntryExitWindow:
    """The c_m interval admitting a non-degenerate pair (Def. 7.5, Fig. 2).

    ``c_star`` is the Prop. 7.1 degeneracy bound; ``c_lower`` the reachable
    lower edge (smallest charge with ``F_inf(b) > eta``); ``c_upper`` the
    upper edge (largest charge with ``F_inf(d) < 1 - eta``).  ``nonempty`` is
    False when no charge qualifies (window empty).  Edges are NaN when absent.
    """

    nonempty: bool
    eta: float
    c_star: float
    c_lower: float
    c_upper: float
    width: float
    b_at_lower: float
    d_at_upper: float
    exit_cdf_at_lower: float
    entry_cdf_at_upper: float


def _with_hold(spec: PerpetualVarSwap, c_hold: float) -> PerpetualVarSwap:
    return PerpetualVarSwap(
        cir=spec.cir,
        delta=spec.delta,
        s_exit=spec.s_exit,
        s_entry=spec.s_entry,
        c_hold=float(c_hold),
        c_idle=spec.c_idle,
        side=spec.side,
    )


def entry_exit_window(
    spec: PerpetualVarSwap,
    eta: float = DEFAULT_ETA,
    *,
    c_grid_max: float = 8.0,
) -> EntryExitWindow:
    """Solve the reachable-carry interval of Def. 7.5 / Fig. 2.

    ``b(c_m)`` and ``d(c_m)`` are both strictly increasing in the charge
    (Prop. 7.9), so the two edges are 1-d monotone solves: ``c_lower`` is the
    root of ``F_inf(b(c)) - eta = 0`` above ``c_star``, ``c_upper`` the root
    of ``F_inf(d(c)) - (1 - eta) = 0``.  An empty window returns
    ``nonempty=False`` rather than raising — an absent interval is a
    legitimate answer (premium below the Figure-2 opening).  Supported
    directly for ``beta < 0``; for ``beta > 0`` the same construction holds
    with the inequalities reversed (Prop. 7.6 remark), which this computes
    symmetrically.
    """
    ee = float(eta)
    if not np.isfinite(ee) or not 0.0 < ee < 0.5:
        raise ValueError("eta must be in (0, 1/2)")
    if spec.beta >= 0.0:
        raise ValueError(
            "entry_exit_window is implemented for beta < 0 (the paper's short; "
            "eps * lam < 0 fails for beta > 0 geometry)"
        )
    c_star = carry_star(spec)
    if not np.isfinite(c_star):
        # eps D' > 0: exercise region always non-empty; the lower edge is
        # driven by reachability alone from c = -inf — not the paper's case.
        raise ValueError("carry_star is infinite (eps D' > 0): window undefined")

    def exit_cdf(c_hold: float) -> float:
        s = _with_hold(spec, c_hold)
        sol = solve_exit(s)
        if not sol.exists:
            return 0.0
        return float(stationary_cdf(s.cir, sol.threshold))

    def entry_cdf(c_hold: float) -> float:
        s = _with_hold(spec, c_hold)
        es = solve_entry(s)
        if not es.exists:
            return 0.0
        if es.immediate:
            return 1.0
        return float(stationary_cdf(s.cir, es.threshold))

    def _bisect_inc(f, lo: float, target: float, *, what: str) -> float:
        flo = f(lo) - target
        hi = lo + max(abs(lo), 1e-4)
        fhi = f(hi) - target
        tries = 0
        while fhi < 0.0:
            hi = lo + (hi - lo) * 4.0
            fhi = f(hi) - target
            tries += 1
            if hi > c_grid_max or tries > 64:
                raise RuntimeError(f"{what}: no bracket found below c={c_grid_max}")
        if flo > 0.0:
            return lo
        return _brentq_root(lambda c: f(c) - target, lo, hi, what=what)

    # Lower edge: smallest c with F_inf(b(c)) > eta (root above c_star).
    c_lo_start = c_star + 1e-6
    f0 = exit_cdf(c_lo_start)
    if f0 >= ee:
        c_lower = c_star  # reachable immediately at the bound
    else:
        c_lower = _bisect_inc(exit_cdf, c_lo_start, ee, what="window lower edge")
    # Upper edge: largest c with F_inf(d(c)) < 1 - eta (monotone increasing).
    # If the entry region dies (or turns immediate) before the CDF target is
    # reached, the admissible interval ends at that existence boundary instead.
    upper_target = 1.0 - ee
    if entry_cdf(c_lower) >= upper_target:
        return EntryExitWindow(
            False,
            ee,
            c_star,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    def _entry_alive(c_hold: float) -> bool:
        es = solve_entry(_with_hold(spec, c_hold))
        return es.exists and not es.immediate

    hi = c_lower + max(abs(c_lower), 1e-4)
    tries = 0
    while _entry_alive(hi) and entry_cdf(hi) < upper_target:
        hi = c_lower + (hi - c_lower) * 4.0
        tries += 1
        if hi > c_grid_max or tries > 64:
            raise RuntimeError("window upper edge: no bracket found below c_max")
    if _entry_alive(hi):
        # entry_cdf(hi) >= upper_target: the reachable edge brackets.
        c_upper = _brentq_root(
            lambda c: entry_cdf(c) - upper_target,
            c_lower,
            hi,
            what="window upper edge",
        )
    else:
        # entry dies on (last_alive, hi]: upper edge = existence boundary.
        lo_alive, hi_dead = c_lower, hi
        for _ in range(80):
            mid = 0.5 * (lo_alive + hi_dead)
            if _entry_alive(mid):
                lo_alive = mid
            else:
                hi_dead = mid
        c_upper = lo_alive
    if not c_upper > c_lower:
        return EntryExitWindow(
            False,
            ee,
            c_star,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    b_lo = solve_exit(_with_hold(spec, c_lower)).threshold
    d_hi = solve_entry(_with_hold(spec, c_upper)).threshold
    return EntryExitWindow(
        True,
        ee,
        c_star,
        c_lower,
        c_upper,
        c_upper - c_lower,
        b_lo,
        d_hi,
        exit_cdf(c_lower),
        entry_cdf(c_upper),
    )


# ---------------------------------------------------------------------------
# Dated-contract machinery (Sec. 3-4): the forcing term and its geometry.
# ---------------------------------------------------------------------------


def dated_variance_duration(t: float, maturity: float, kappa_q: float) -> float:
    """g_Q(t) = (1 - exp(-k_Q (T - t))) / k_Q (eq. 8): residual variance duration."""
    tt = float(t)
    mm = _require_finite_positive(maturity, "maturity")
    kk = _require_finite_positive(kappa_q, "kappa_q")
    if not np.isfinite(tt) or tt < 0.0 or tt > mm:
        raise ValueError("t must be finite and in [0, maturity]")
    return float((1.0 - np.exp(-kk * (mm - tt))) / kk)


def dated_mark_phi0(
    t: float, v: float, maturity: float, strike_var: float, cir: CIRTwoMeasure
) -> float:
    """Phi_0(t, v) of eq. (10): the non-accrued part of the dated-swap mark."""
    tt = float(t)
    vv = float(v)
    mm = _require_finite_positive(maturity, "maturity")
    ss = float(strike_var)
    if not (np.isfinite(tt) and np.isfinite(vv) and np.isfinite(ss)):
        raise ValueError("t, v, strike_var must be finite")
    if not 0.0 <= tt <= mm:
        raise ValueError("t must be in [0, maturity]")
    g = dated_variance_duration(tt, mm, cir.kappa_q)
    return float(cir.theta_q * (mm - tt) - mm * ss + g * (vv - cir.theta_q))


def dated_mark_state(
    t: float,
    accrued: float,
    v: float,
    maturity: float,
    strike_var: float,
    cir: CIRTwoMeasure,
) -> float:
    """h(t) = A_t + Phi_0(t, v): the Q-martingale of Lemma 2.1."""
    aa = float(accrued)
    if not np.isfinite(aa):
        raise ValueError("accrued must be finite")
    return aa + dated_mark_phi0(t, v, maturity, strike_var, cir)


def dated_fair_strike(t: float, v: float, maturity: float, cir: CIRTwoMeasure) -> float:
    """xi(t, v) of eq. (53): fair strike for the residual maturity."""
    tt = float(t)
    mm = _require_finite_positive(maturity, "maturity")
    vv = float(v)
    if not (np.isfinite(tt) and np.isfinite(vv)):
        raise ValueError("t and v must be finite")
    if not 0.0 <= tt < mm:
        raise ValueError("t must be in [0, maturity)")
    g = dated_variance_duration(tt, mm, cir.kappa_q)
    return float(cir.theta_q + (vv - cir.theta_q) * g / (mm - tt))


def dated_breakeven_level(
    t: float, accrued: float, maturity: float, strike_var: float, cir: CIRTwoMeasure
) -> float:
    """v*_{A_t} of eq. (39): the level at which the dated swap marks at zero."""
    tt = float(t)
    aa = float(accrued)
    mm = _require_finite_positive(maturity, "maturity")
    ss = float(strike_var)
    if not (np.isfinite(tt) and np.isfinite(aa) and np.isfinite(ss)):
        raise ValueError("t, accrued, strike_var must be finite")
    if not 0.0 <= tt < mm:
        raise ValueError("t must be in [0, maturity)")
    g = dated_variance_duration(tt, mm, cir.kappa_q)
    if not g > 0.0:
        raise ValueError("residual duration is zero at maturity")
    return float(
        cir.theta_q
        - (cir.kappa_q / (1.0 - np.exp(-cir.kappa_q * (mm - tt))))
        * (aa - tt * ss + (mm - tt) * (cir.theta_q - ss))
    )


def carry_forcing(
    t: float,
    v: ScalarOrArray,
    maturity: float,
    side: int,
    cir: CIRTwoMeasure,
    s: float,
    c_m: float,
) -> ScalarOrArray:
    """A Phi(t, v) = eps lam g_Q(t) v + (s - c_m): the forcing term (Prop. 4.1).

    The dated-contract carry identity: risk-premium carry proportional to the
    state and the residual variance duration, signed by the side, net of
    spread and holding charge.  Its sign in v (that of eps * lam, g_Q > 0 on
    [0, T)) fixes the exercise-region geometry of Table 1.
    """
    if side not in (+1, -1):
        raise ValueError("side must be +1 or -1")
    ss = float(s)
    cm = float(c_m)
    if not (np.isfinite(ss) and np.isfinite(cm)) or ss < 0.0:
        raise ValueError("s must be finite >= 0 and c_m finite")
    vv = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(vv)):
        raise ValueError("v must be finite")
    g = dated_variance_duration(float(t), maturity, cir.kappa_q)
    return float(side) * cir.lam * g * vv + (ss - cm)


def myopic_level(
    t: float,
    maturity: float,
    side: int,
    cir: CIRTwoMeasure,
    s: float,
    c_m: float,
) -> float:
    """v_c(t) = (c_m - s) / (eps lam g_Q(t)) of Def. 4.2: the zero-carry level."""
    eps_lam = float(side) * cir.lam
    g = dated_variance_duration(float(t), maturity, cir.kappa_q)
    if eps_lam == 0.0:
        raise ValueError("eps * lam = 0: no myopic level (Cor. 6.7 degenerate)")
    if not g > 0.0:
        raise ValueError("no residual duration at maturity")
    ss = float(s)
    cm = float(c_m)
    if not (np.isfinite(ss) and np.isfinite(cm)):
        raise ValueError("s, c_m must be finite")
    return float((cm - ss) / (eps_lam * g))


def dated_exercise_geometry(
    side: int, cir: CIRTwoMeasure, s: float, c_m: float
) -> dict[str, object]:
    """Table-1 geometry off the forcing-term sign and the cost comparison.

    eps*lam < 0 -> exercise region upper set ``{v > b_T(t)}`` (exit into a
    spike), requires ``s > c_m``; eps*lam > 0 -> lower set ``{v <= b_T(t)}``
    (exit into a collapse), requires ``c_m > s``.  The ``requires`` flag is
    the Table-1 admissibility condition; without it the exercise region is
    empty (the position is held to expiry).
    """
    if side not in (+1, -1):
        raise ValueError("side must be +1 or -1")
    ss = float(s)
    cm = float(c_m)
    if not (np.isfinite(ss) and np.isfinite(cm)) or ss < 0.0:
        raise ValueError("s must be finite >= 0 and c_m finite")
    eps_lam = float(side) * cir.lam
    geom = exercise_geometry(float(side), cir.lam)
    if geom == "upper":
        requires = bool(ss > cm)
        interp = "exit into a spike" if requires else "held to expiry"
    elif geom == "lower":
        requires = bool(cm > ss)
        interp = "exit into a collapse" if requires else "held to expiry"
    else:
        requires = False
        interp = "no optionality (lam = 0)"
    return {
        "region": geom,
        "eps_lam": eps_lam,
        "requires_satisfied": requires,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Bench: flat dict[str, float] of seeded SYNTHETIC diagnostics.
# ---------------------------------------------------------------------------


def bench_varswap_stopping(seed: int = 7) -> dict[str, float]:
    """Seeded SYNTHETIC bench (~seconds).  All keys are ``synthetic_*``.

    Uses the paper's Section-8.2 in-window calibration (kappa_Q = 1.0,
    theta_Q = 0.045, gamma = 0.20, lam = -0.6 -> kappa_P = 1.60,
    theta_P = 0.028125, nu = 2.25) plus a seeded stationary-law Monte Carlo
    check of the entry-tail percentile.  Correctness material only — never
    market evidence.
    """
    rng = np.random.default_rng(int(seed))
    cir = CIRTwoMeasure(kappa_p=1.6, theta_p=0.028125, kappa_q=1.0, gamma=0.20)
    spec = PerpetualVarSwap(cir=cir, delta=0.5, s_exit=0.002, s_entry=0.002, c_hold=0.012, side=-1)
    sol = solve_exit(spec)
    esol = solve_entry(spec)
    win = entry_exit_window(spec)
    sens = threshold_sensitivities(spec, sol, esol)

    out: dict[str, float] = {}
    out["synthetic_smooth_pasting_max_resid"] = max(sol.residual, esol.residual)
    out["synthetic_exit_threshold_short"] = sol.threshold
    out["synthetic_entry_threshold_short"] = esol.threshold
    out["synthetic_exit_rate_at_threshold"] = float(
        np.sqrt(fair_rate(sol.threshold, cir.kappa_q, cir.theta_q, spec.delta))
    )
    out["synthetic_entry_rate_at_threshold"] = float(
        np.sqrt(fair_rate(esol.threshold, cir.kappa_q, cir.theta_q, spec.delta))
    )
    out["synthetic_entry_tail_cdf_at_trigger"] = float(stationary_cdf(cir, esol.threshold))
    out["synthetic_exit_cdf_at_trigger"] = float(stationary_cdf(cir, sol.threshold))
    out["synthetic_entry_interval_width"] = win.width
    out["synthetic_carry_star"] = win.c_star
    out["synthetic_window_lower_edge"] = win.c_lower
    out["synthetic_window_upper_edge"] = win.c_upper
    out["synthetic_db_dcm"] = sens["db_dcm"]
    out["synthetic_dd_dcm"] = sens["dd_dcm"]
    out["synthetic_l_minus_delta_rho_at_d"] = _l_d_rho(spec, sol, esol.threshold)
    out["synthetic_entry_dpsi_sign"] = float(np.sign(esol.dpsi))
    # Forcing-term sign flip (Prop. 4.1): short vs long at a mid state.
    f_short = float(carry_forcing(0.0, cir.theta_p, 1.0, -1, cir, s=0.002, c_m=0.012))
    f_long = float(carry_forcing(0.0, cir.theta_p, 1.0, +1, cir, s=0.002, c_m=0.012))
    out["synthetic_forcing_short_minus_long"] = f_short - f_long
    out["synthetic_forcing_sign_flip_geometry"] = float(exercise_geometry(-1.0, cir.lam) == "lower")
    # Seeded stationary-law check: empirical P(v_inf > d*) vs analytic.
    v_inf = rng.gamma(cir.nu, 1.0 / cir.sigma_tilde, size=200_000)
    emp_tail = float(np.mean(v_inf > esol.threshold))
    out["synthetic_mc_entry_tail_empirical"] = emp_tail
    out["synthetic_mc_entry_tail_abs_err"] = abs(
        emp_tail - (1.0 - out["synthetic_entry_tail_cdf_at_trigger"])
    )
    # Eigenfunction residual: (L - delta) F == 0 pointwise (eq. 25 check).
    vv = np.linspace(0.005, 0.15, 9)
    lf = (
        0.5 * cir.gamma**2 * vv * _Fpp(spec, vv)
        + cir.kappa_p * (cir.theta_p - vv) * _Fp(spec, vv)
        - spec.delta * _F(spec, vv)
    )
    out["synthetic_eigen_eq_max_resid"] = float(np.max(np.abs(lf)))
    return out
