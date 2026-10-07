"""Conformal prediction under exponential-tilt joint shift (ExTRA-WCP / -WCP-T).

Implements Choi, "Conformal Prediction under Exponential-Tilt Joint Shift",
arXiv:2609.30886 (stat.ML, submitted 2026-09-25; fetched and verified —
the summary below follows its Sections 2-4 and Appendices A-B). The ExTRA
estimator (Exponential Tilt Reweighting Alignment) is from Maity et al.,
"Understanding new tasks through the lens of training data via exponential
tilting", ICLR 2023. Setting: labeled source pairs from P, unlabeled target
inputs from Q_X, and Q << P may shift BOTH the input distribution and the
input-response relationship (a joint shift). Covariate-shift weighting
(Tibshirani et al. 2019, :mod:`quant_fund.models.weighted_conformal`) and
label-shift adjustment each cover only one half of this shift.

Shift model and estimator (paper Section 2):

1. Exponential tilt (Eq. 1): ``w_beta(x,y) = h_beta(x,y)/Z_P(beta)`` with
   ``h_beta(x,y) = exp(beta . phi(x,y))``. The regression family implemented
   here is the paper's separable sign tilt (Eq. 23), ``phi(x,y) = (u,
   sign(y))`` for inputs ``x = (u, v)``. The response feature is BOUNDED, and
   that is load-bearing: a tilt linear in ``y`` drives candidate weights past
   the ``alpha*W/(1-alpha)`` forced-inclusion threshold and yields prediction
   sets of infinite length on one response tail (Eq. 21 and the paragraph
   following it). The V-interaction family of paper Prop. 2 (which is exactly
   non-identified from target inputs) is deliberately excluded.
2. Marginal matching (Section 2.2, Eq. 4-6): each tilt implies an input
   distribution through the conditional moment ``M_beta(x) = E_P[h_beta |
   X=x]`` (Eq. 2-3). :func:`fit_exponential_tilt` maximizes the penalized
   plug-in objective ``L^(beta) = mean_j log M^_beta(X_j^Q) - log mean_i
   h_beta(X_i^P, Y_i^P) - (lambda/2)||beta||^2`` (Eq. 6) with L-BFGS-B under
   prespecified coefficient bounds and multiple starts (App. B); maximizing
   ``L^`` is the KL projection of the observed target inputs onto the induced
   input distributions, ``KL(Q_X||Q_beta,X) = KL(Q_X||P_X) - L(beta)``.

Two procedures sharing one fitted ratio (paper Section 3):

- :func:`extra_weighted_calibration` — ExTRA-WCP: the weighted conformal
  quantile of Eq. 26 with candidate-dependent weights ``H_z(x) = h_beta^(x,
  z)`` per response half-line. This generalizes the Tibshirani et al. (2019)
  order statistic (their unit candidate atom becomes the candidate's own
  tilt weight); when the candidate weight equals the mean calibration
  weight it reduces EXACTLY to
  :func:`quant_fund.models.weighted_conformal.weighted_conformal_quantile`.
- :func:`extra_tilted_predictive` — ExTRA-WCP-T: additionally tilts the
  fitted source predictive distribution, ``p_beta(y|x) = h_beta p_0 /
  M^_beta(x)`` (Eq. 9), score ``S_beta = S_0 - log h_beta + log Z^pred``
  (Eq. 10). For the sign-tilt family this is the closed-form gate-logit
  shift ``+2b`` (within-mode components unchanged); input-only tilts cancel
  from the conditional law (Section 3.1).

Coverage (paper Section 4): with the fitted weights BOTH scores cover at
``>= 1 - alpha`` under the surrogate ``Q^ = w^.P`` they define (Eq. 15), and
share the bound ``>= 1 - alpha - d_TV(Q, Q^)`` under the true target (Eq.
16). Actual coverage can still differ sharply because the score decides
WHERE misses land relative to ratio error (Eq. 17-19): in the paper's
synthetic experiments tilting cuts mean set length ~30% at near-nominal
coverage when the models match the DGP and target inputs are informative
about the shift (Table 2: WCP 0.904/2.190 vs WCP-T 0.899/1.525 at
(a*,b*)=(1,1.2), paired length reduction 30.34%), but loses substantial
coverage in classification (0.924 -> 0.814) and when target inputs carry
little information about the response shift (eta=0: WCP 0.727, WCP-T 0.493 —
weights-only also degrades there because the fitted JOINT ratio itself is
unidentified; it never collapses as far as tilting).

:func:`when_to_tilt_diagnostic` — deciding when to tilt from source labels
and target inputs alone is an OPEN problem (paper abstract and Section 6:
"Good coverage from weighted calibration alone does not ensure that adding
predictive tilting will preserve coverage"). The diagnostic operationalizes
the observable signals the paper does offer — the Prop. 1 identification
requirement (nonconstant fitted mode probability; at constant mode
probability Eq. 25 makes target inputs uninformative about b), the Section
5.4 coefficient-instability observation (wrong-sign / bound-saturated fits
across optimizer starts accompany coverage losses), and weight ESS — as a
documented HEURISTIC with the conservative weights-only default. The paper
states these post hoc associations "do not yield a score-selection rule
without target responses"; nothing here claims otherwise.

Honesty: every number is validated on the SYNTHETIC fixture
:func:`synthetic_bimodal_shift` (the paper's Section 5.2 bimodal
truncated-Gaussian DGP, Eq. 22, with the eta mode-signal ladder of Section
5.4) — a correctness test, never market evidence. Research outputs are
proper-score quantities only (coverage, prediction-set length, TV-style
diagnostics, ESS); no P&L / Sharpe-family metrics. Determinism: all
randomness lives inside seeded generators; every estimator (IRLS, L-BFGS-B
from fixed starts, Brent root-find, Gauss-Legendre quadrature, vectorized
set inversion) is closed-form or deterministic. Fail-closed: invalid inputs,
empty modes, and non-converged optimizers raise ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log, pi, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize
from scipy.special import expit, log_ndtr, logsumexp, ndtr, ndtri

Array = NDArray[np.float64]

_LOG2PI: float = log(2.0 * pi)

#: Paper App. B coefficient bounds for the separable regression tilt family.
PAPER_TILT_BOUNDS: tuple[tuple[float, float], tuple[float, float]] = (
    (-1.5, 1.5),
    (-3.0, 3.0),
)

#: Paper App. B optimizer starts for the separable family: (0,0), (0,±1.5),
#: retaining the converged solution with the largest objective.
DEFAULT_TILT_STARTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.5),
    (0.0, -1.5),
)

# Paper App. B source-model bounds: location intercepts [-3,3], location
# slopes [-1,1], log-scale intercept [-4,0], log-scale slope [-1.5,1.5].
_MLE_BOUNDS: tuple[tuple[float, float], ...] = (
    (-3.0, 3.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-3.0, 3.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-4.0, 0.0),
    (-1.5, 1.5),
)

# Gauss-Legendre nodes for E_V over Uniform[-1,1] (density 1/2 on [-1,1]).
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(200)
_GL_NODES = np.asarray(_GL_NODES, dtype=float)
_GL_WEIGHTS = np.asarray(_GL_WEIGHTS, dtype=float) / 2.0

#: Source positive-mode probability p_+ at eta=1, d=-2 (paper Section 5.2:
#: [log(1+e) - log(1+e^-5)]/6 ~= 0.218), held fixed across the eta ladder by
#: adjusting d_eta (paper App. B).
SOURCE_POSITIVE_MODE_PROB: float = float(np.sum(_GL_WEIGHTS * expit(-2.0 + 3.0 * _GL_NODES)))


def _finite_1d(x: object, name: str) -> Array:
    """Fail-closed coercion to a finite, non-empty 1-d float array."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _finite_2d(x: object, name: str) -> Array:
    """Fail-closed coercion to a finite, non-empty 2-d float array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2-d array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _input_2d(x: object, name: str) -> Array:
    """Fail-closed coercion of fixture inputs to an (n, 2) float array."""
    arr = _finite_2d(x, name)
    if arr.shape[1] != 2:
        raise ValueError(f"{name} must have exactly two input columns (u, v)")
    return arr


def _signed_1d(y: object, name: str) -> Array:
    """Fail-closed response signs: the model has P(Y=0)=0, so y=0 is invalid."""
    arr = _finite_1d(y, name)
    if np.any(arr == 0.0):
        raise ValueError(f"{name} must be nonzero (sign(y) is undefined at 0)")
    return np.sign(arr)


def _positive_1d(w: object, name: str) -> Array:
    """Fail-closed coercion to finite, strictly positive weights (paper: H>0)."""
    arr = _finite_1d(w, name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def _check_beta(beta: object, name: str = "beta") -> Array:
    """Fail-closed coercion of a tilt parameter to a finite 2-vector (a, b)."""
    arr = np.asarray(beta, dtype=float).ravel()
    if arr.size != 2:
        raise ValueError(f"{name} must be a 2-vector (a, b) for the separable tilt family")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def solve_mode_intercept(eta: float) -> float:
    """Intercept ``d_eta`` preserving ``p_+`` at mode-signal strength ``eta``.

    Solves ``E_V[logit^-1(d + 3*eta*V)] = p_+`` over ``V ~ Uniform[-1,1]`` by
    Gauss-Legendre quadrature plus Brent root-finding (paper App. B: d_eta ~=
    -2.000, -1.481, -1.331, -1.279 at eta = 1, 0.5, 0.25, 0). Deterministic.
    """
    e = float(eta)
    if not np.isfinite(e) or e < 0.0:
        raise ValueError("eta must be finite and >= 0")
    if e == 0.0:
        return float(log(SOURCE_POSITIVE_MODE_PROB) - log(1.0 - SOURCE_POSITIVE_MODE_PROB))

    def gap(d: float) -> float:
        return float(np.sum(_GL_WEIGHTS * expit(d + 3.0 * e * _GL_NODES))) - float(
            SOURCE_POSITIVE_MODE_PROB
        )

    return float(brentq(gap, -15.0, 15.0, xtol=1e-12))


@dataclass(frozen=True)
class SourceConditional:
    """Fitted source conditional law ``p^(y|x)`` for the fixture family.

    Paper App. B: logistic mode probability ``pi^_+(u,v) = expit(g . (1,u,v))``
    on the response sign; per-sign Gaussian locations ``mu_z(x)`` linear in
    ``(1,u,v)``; shared log scale ``log sigma(u) = s0 + s1*u``; each mode's
    density is the Gaussian TRUNCATED to the half-line ``{z*y > 0}`` (paper
    Eq. 22), so ``sign(y)`` observes the mode. ``mu_coef`` row 0 is z=+1,
    row 1 is z=-1.
    """

    gate: Array
    mu_coef: Array
    log_sigma_coef: Array

    def _gate_linear(self, x: Array) -> Array:
        return np.asarray(x[:, 0] * self.gate[1] + x[:, 1] * self.gate[2] + self.gate[0], float)

    def mode_prob(self, x: Array) -> Array:
        """Fitted positive-mode probability ``pi^_+(x)``."""
        return np.asarray(expit(self._gate_linear(_input_2d(x, "x"))), dtype=float)

    def _log_pi_z(self, x: Array, z: Array) -> Array:
        g = self._gate_linear(x)
        return np.asarray(
            np.where(z > 0.0, -np.logaddexp(0.0, -g), -np.logaddexp(0.0, g)), dtype=float
        )

    def mean_z(self, x: Array, z: Array) -> Array:
        """Per-sign location ``mu_z(x)`` (row-selected by sign)."""
        xp = _input_2d(x, "x")
        zz = np.asarray(z, dtype=float)
        design = np.column_stack([np.ones(xp.shape[0]), xp[:, 0], xp[:, 1]])
        mu_p = design @ self.mu_coef[0]
        mu_m = design @ self.mu_coef[1]
        return np.asarray(np.where(zz > 0.0, mu_p, mu_m), dtype=float)

    def log_scale(self, x: Array) -> Array:
        xp = _input_2d(x, "x")
        return np.asarray(self.log_sigma_coef[0] + self.log_sigma_coef[1] * xp[:, 0], dtype=float)

    def scale(self, x: Array) -> Array:
        return np.asarray(np.exp(self.log_scale(x)), dtype=float)

    def log_density(self, x: Array, y: Array) -> Array:
        """Log fitted conditional density ``log p^(y|x)`` (truncated modes)."""
        xp = _input_2d(x, "x")
        yy = _finite_1d(y, "y")
        if yy.size != xp.shape[0]:
            raise ValueError("x and y must have one value per row")
        z = _signed_1d(y, "y")
        mu = self.mean_z(xp, z)
        log_sig = self.log_scale(xp)
        t = z * mu * np.exp(-log_sig)
        return np.asarray(
            self._log_pi_z(xp, z)
            - 0.5 * _LOG2PI
            - log_sig
            - log_ndtr(t)
            - 0.5 * np.square((yy - mu) * np.exp(-log_sig)),
            dtype=float,
        )

    def score(self, x: Array, y: Array) -> Array:
        """Source nonconformity ``S_0(x,y) = -log p^(y|x)`` (paper Section 3.1)."""
        return np.asarray(-self.log_density(x, y), dtype=float)

    def log_moment(self, x: Array, beta: Array) -> Array:
        """``log M^_beta(x)``: log conditional moment of the tilt (paper Eq. 2/5).

        For the sign tilt ``h = exp(a*u + b*sign(y))`` the within-mode
        integral is exact: ``M^_beta(x) = e^{a*u} [pi^_+ e^b + (1-pi^_+) e^-b]``
        (matching the paper's App. A expression for ``M_{a,b}(u,v)``).
        """
        xp = _input_2d(x, "x")
        b = _check_beta(beta)
        g = self._gate_linear(xp)
        log_pi = -np.logaddexp(0.0, -g)
        log_1mpi = -np.logaddexp(0.0, g)
        log_gate = np.logaddexp(log_pi + b[1], log_1mpi - b[1])
        return np.asarray(b[0] * xp[:, 0] + log_gate, dtype=float)

    def moment(self, x: Array, beta: Array) -> Array:
        return np.asarray(np.exp(self.log_moment(x, beta)), dtype=float)


def _irls_logistic_gate(design: Array, labels: Array, l2: float) -> Array:
    """Ridge-penalized logistic gate by IRLS/Newton from a zero start.

    Deterministic; intercept unpenalized; singular Hessian or divergent
    iterates fail closed. Mirrors the cousin convention in
    :mod:`quant_fund.models.conformal_transfer`.
    """
    p = design.shape[1]
    reg = np.full(p, float(l2))
    reg[0] = 0.0
    beta = np.zeros(p)
    for _ in range(100):
        pr = np.asarray(expit(design @ beta), dtype=float)
        curvature = np.maximum(pr * (1.0 - pr), 1e-9)
        grad = design.T @ (labels - pr) - reg * beta
        hess = np.asarray((design * curvature[:, None]).T @ design + np.diag(reg), dtype=float)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise ValueError("gate logistic Hessian is singular") from exc
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            raise ValueError("gate logistic diverged")
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def _tn_objective(theta: Array, u: Array, v: Array, y: Array, z: Array) -> tuple[float, Array]:
    """Within-mode truncated-normal NLL and analytic gradient (joint value).

    ``theta = [p0,p1,p2, m0,m1,m2, s0,s1]``; ``mu_z = coef_z . (1,u,v)``;
    ``log sigma = s0 + s1*u``; observation i contributes
    ``0.5 e_i^2 + log sigma_i - log Phi(z_i mu_i / sigma_i)`` (constants
    dropped), the truncated-Gaussian likelihood of paper Eq. 22 / App. B.
    """
    design = np.column_stack([np.ones(u.size), u, v])
    mu = np.where(z > 0.0, design @ theta[0:3], design @ theta[3:6])
    log_sig = theta[6] + theta[7] * u
    sig = np.exp(log_sig)
    resid = (y - mu) / sig
    t = z * mu / sig
    log_phi_t = -0.5 * np.square(t) - 0.5 * _LOG2PI
    # Inverse Mills ratio Phi-bar: exp(log phi(t) - log Phi(t)), stable for
    # very negative t; clipped far beyond any value reachable under _MLE_BOUNDS.
    mills = np.clip(np.exp(np.minimum(log_phi_t - log_ndtr(t), 50.0)), 0.0, 1e12)
    nll = float(np.sum(0.5 * np.square(resid) + log_sig - log_ndtr(t)))
    if not np.isfinite(nll):
        return 1e12, np.zeros(8)
    d_mu = (-resid - z * mills) / sig
    d_log_sig = 1.0 - np.square(resid) + t * mills
    grad = np.zeros(8)
    pos = z > 0.0
    grad[0:3] = design[pos].T @ d_mu[pos]
    grad[3:6] = design[~pos].T @ d_mu[~pos]
    grad[6] = float(np.sum(d_log_sig))
    grad[7] = float(np.sum(d_log_sig * u))
    if not np.all(np.isfinite(grad)):
        return 1e12, np.zeros(8)
    return nll, grad


def fit_source_model(x_train: Array, y_train: Array, *, c_gate: float = 100.0) -> SourceConditional:
    """Fit the source conditional law on ``D_tr`` (paper App. B, two stages).

    Stage 1: ridge logistic mode probability on ``(1,u,v)`` with inverse
    regularization ``c_gate`` (paper: C=100). Stage 2: the six location and
    two log-scale coefficients jointly by truncated-normal maximum likelihood
    (L-BFGS-B under the paper's App. B bounds, deterministic analytic
    gradient, moment-based deterministic start). Fail-closed: raises
    ``ValueError`` on empty/non-finite inputs, zero responses, a missing
    response sign, or a non-finite optimum.
    """
    x = _input_2d(x_train, "x_train")
    y = _finite_1d(y_train, "y_train")
    if x.shape[0] != y.size:
        raise ValueError("x_train and y_train must have the same number of rows")
    c = float(c_gate)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("c_gate must be finite and > 0")
    z = _signed_1d(y, "y_train")
    pos = z > 0.0
    if not np.any(pos) or not np.any(~pos):
        raise ValueError("source fit requires both response signs (P(Y=0)=0 model)")

    design = np.column_stack([np.ones(x.shape[0]), x[:, 0], x[:, 1]])
    gate = _irls_logistic_gate(design, pos.astype(float), 1.0 / c)

    mu_init_p = float(np.clip(np.mean(y[pos]), -2.5, 2.5))
    mu_init_m = float(np.clip(np.mean(y[~pos]), -2.5, 2.5))
    resid = y - np.where(pos, mu_init_p, mu_init_m)
    s_init = float(np.clip(log(max(float(np.std(resid)), 1e-3)), -3.9, -0.05))
    theta0 = np.array([mu_init_p, 0.0, 0.0, mu_init_m, 0.0, 0.0, s_init, 0.0])
    theta0 = np.clip(theta0, [b[0] for b in _MLE_BOUNDS], [b[1] for b in _MLE_BOUNDS])
    res = minimize(
        _tn_objective,
        theta0,
        jac=True,
        args=(x[:, 0], x[:, 1], y, z),
        method="L-BFGS-B",
        bounds=_MLE_BOUNDS,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    theta = np.asarray(res.x, dtype=float)
    if not np.all(np.isfinite(theta)) or not np.isfinite(float(res.fun)) or float(res.fun) >= 1e11:
        raise ValueError("source truncated-normal MLE failed to reach a finite optimum")
    return SourceConditional(
        gate=np.asarray(gate, dtype=float),
        mu_coef=np.asarray(theta[0:6].reshape(2, 3), dtype=float),
        log_sigma_coef=np.asarray(theta[6:8], dtype=float),
    )


def regularization_check_ridge(n_shift: int, m_target: int) -> float:
    """The paper's App. B regularization-check penalty ``sqrt(1/n + 1/m)``.

    The main ExTRA penalty is ``1/n_shift``; App. B fixes this larger penalty
    for the regularization check, and Section 5.4 reports that it reverses the
    WCP/WCP-T ordering at weak mode signal — the observable penalty
    sensitivity that :func:`when_to_tilt_diagnostic` uses as its
    tilt-instability signal.
    """
    n = int(n_shift)
    m = int(m_target)
    if n < 1 or m < 1:
        raise ValueError("n_shift and m_target must be >= 1")
    return float(sqrt(1.0 / n + 1.0 / m))


@dataclass(frozen=True)
class TiltFit:
    """ExTRA marginal-matching fit (paper Eq. 6) with per-start records.

    ``beta`` is the retained solution: among converged starts with finite
    objectives, the one with the LARGEST penalized objective (paper App. B).
    ``start_betas`` / ``start_objectives`` / ``start_converged`` keep every
    start for the instability signal of :func:`when_to_tilt_diagnostic`
    (paper Section 5.4: divergent or wrong-sign fits across starts accompany
    coverage losses). ``beta[0]`` is the covariate coefficient ``a``,
    ``beta[1]`` the response (sign) coefficient ``b``.
    """

    beta: Array
    objective: float
    start_betas: Array
    start_objectives: Array
    start_converged: Array
    ridge: float
    n_shift: int
    m_target: int


def fit_exponential_tilt(
    source_model: SourceConditional,
    x_shift_src: Array,
    y_shift_src: Array,
    x_shift_target: Array,
    *,
    bounds: tuple[tuple[float, float], ...] = PAPER_TILT_BOUNDS,
    ridge: float | None = None,
    starts: tuple[tuple[float, float], ...] = DEFAULT_TILT_STARTS,
    max_iter: int = 500,
) -> TiltFit:
    """Estimate the joint tilt from labeled source pairs + unlabeled target inputs.

    Maximizes the paper's Eq. 6 penalized plug-in objective

    ``L^(beta) = (1/m) sum_j log M^_beta(X_j^Q)
    - log{(1/n_shift) sum_i h_beta(X_i^P, Y_i^P)} - (lambda/2)||beta||^2``

    — the empirical KL projection of the target input marginal onto the
    tilt-induced input distribution (Eq. 4: ``KL(Q_X||Q_beta,X) =
    KL(Q_X||P_X) - L(beta)``). ``M^_beta`` is evaluated ANALYTICALLY under
    the fitted source conditional (Eq. 5; closed form for the sign tilt, see
    :meth:`SourceConditional.log_moment`). Optimization is L-BFGS-B with the
    analytic gradient under prespecified coefficient bounds and the paper's
    multiple starts (App. B), retaining the converged solution with the
    largest objective; default ridge ``lambda = 1/n_shift`` (App. B main
    penalty). Numerical convergence does not certify a global maximum
    (paper's caveat, kept). Fail-closed: raises ``ValueError`` on empty /
    non-finite / shape-mismatched samples, zero responses, invalid bounds,
    starts outside bounds, negative ridge, or when no start converges to a
    finite objective.
    """
    if not isinstance(source_model, SourceConditional):
        raise TypeError("source_model must be a fitted SourceConditional")
    xp = _input_2d(x_shift_src, "x_shift_src")
    yp = _finite_1d(y_shift_src, "y_shift_src")
    if xp.shape[0] != yp.size:
        raise ValueError("x_shift_src and y_shift_src must have the same number of rows")
    xq = _input_2d(x_shift_target, "x_shift_target")
    z_p = _signed_1d(y_shift_src, "y_shift_src")
    u_p = xp[:, 0]
    u_q = xq[:, 0]

    bnds = tuple((float(lo), float(hi)) for lo, hi in bounds)
    if len(bnds) != 2 or any(not np.isfinite(lo + hi) or lo >= hi for lo, hi in bnds):
        raise ValueError("bounds must be two finite (lo, hi) pairs with lo < hi")
    lam = float(ridge) if ridge is not None else 1.0 / float(xp.shape[0])
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("ridge must be finite and >= 0")
    start_arr = np.asarray(starts, dtype=float).reshape(-1, 2)
    if start_arr.shape[0] == 0 or not np.all(np.isfinite(start_arr)):
        raise ValueError("starts must be a non-empty finite (k, 2) array")
    if np.any(start_arr[:, 0] < bnds[0][0]) or np.any(start_arr[:, 0] > bnds[0][1]):
        raise ValueError("starts must lie inside the coefficient bounds")
    if np.any(start_arr[:, 1] < bnds[1][0]) or np.any(start_arr[:, 1] > bnds[1][1]):
        raise ValueError("starts must lie inside the coefficient bounds")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1")

    g_q = source_model._gate_linear(xq)
    log_pi_q = np.asarray(-np.logaddexp(0.0, -g_q), dtype=float)
    log_1mpi_q = np.asarray(-np.logaddexp(0.0, g_q), dtype=float)
    n_shift = int(xp.shape[0])
    log_n = log(float(n_shift))

    def neg_objective(beta: Array) -> tuple[float, Array]:
        a, b = float(beta[0]), float(beta[1])
        log_g = np.logaddexp(log_pi_q + b, log_1mpi_q - b)
        term_q = float(np.mean(a * u_q + log_g))
        expo = a * u_p + b * z_p
        lse_raw = float(logsumexp(expo))
        obj = term_q - (lse_raw - log_n) - 0.5 * lam * (a * a + b * b)
        # Softmax-normalized tilt shares (sum to 1): the gradient of the log
        # empirical normalizer is the weighted mean of the features.
        h = np.exp(expo - lse_raw)
        d_log_g = np.exp(log_pi_q + b - log_g) - np.exp(log_1mpi_q - b - log_g)
        grad = np.array(
            [
                float(np.mean(u_q)) - float(np.sum(u_p * h)) - lam * a,
                float(np.mean(d_log_g)) - float(np.sum(z_p * h)) - lam * b,
            ]
        )
        if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
            return 1e12, np.zeros(2)
        return -obj, -grad

    betas: list[Array] = []
    objectives: list[float] = []
    converged: list[bool] = []
    for start in start_arr:
        res = minimize(
            neg_objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bnds,
            options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8},
        )
        beta_s = np.asarray(res.x, dtype=float)
        obj_s, _ = neg_objective(beta_s)
        obj_s = -obj_s if obj_s < 1e11 else float("-inf")
        ok = bool(res.success) and bool(np.isfinite(obj_s)) and bool(np.all(np.isfinite(beta_s)))
        betas.append(beta_s)
        objectives.append(float(obj_s) if ok else float("-inf"))
        converged.append(ok)
    if not any(converged):
        raise ValueError("ExTRA tilt fit failed to converge from every start")
    best = int(np.argmax(np.asarray(objectives)))
    return TiltFit(
        beta=np.asarray(betas[best], dtype=float),
        objective=float(objectives[best]),
        start_betas=np.asarray(np.vstack(betas), dtype=float),
        start_objectives=np.asarray(objectives, dtype=float),
        start_converged=np.asarray(converged, dtype=bool),
        ridge=lam,
        n_shift=n_shift,
        m_target=int(xq.shape[0]),
    )


def tilt_weights(x: Array, y: Array, beta: Array) -> Array:
    """Unnormalized joint tilt ``h_beta^(x,y) = exp(a*u + b*sign(y))``.

    The global normalizer ``Z_P(beta^)`` cancels from every weighted-rank
    expression (paper Section 3.2), so raw ``h`` values are the weights.
    """
    xp = _input_2d(x, "x")
    z = _signed_1d(y, "y")
    b = _check_beta(beta)
    if z.size != xp.shape[0]:
        raise ValueError("x and y must have one value per row")
    return np.asarray(np.exp(b[0] * xp[:, 0] + b[1] * z), dtype=float)


def extra_weighted_calibration(
    scores: Array, weights: Array, candidate_weight: float, alpha: float
) -> float:
    """ExTRA weighted conformal quantile with candidate-dependent weights.

    Paper Eq. 26: ``q = inf{t : sum_i H_i 1{V_i <= t} >= (1-alpha)(W + H_y)}``
    with ``inf(empty) = +inf``, where ``V_i`` are the fixed calibration
    scores, ``H_i = h_beta^(X_i, Y_i)`` the calibration weights, ``W = sum_i
    H_i``, and ``H_y = h_beta^(x, y)`` the CANDIDATE's own weight (the
    candidate atom of the weighted rank, Eq. 12). This generalizes the
    Tibshirani et al. (2019) construction — their unit atom becomes
    ``H_y`` — and reduces exactly to
    :func:`quant_fund.models.weighted_conformal.weighted_conformal_quantile`
    when ``H_y`` equals the mean calibration weight (that module's atom
    convention); with unit weights and ``H_y = 1`` it is the Vovk order
    statistic of :func:`quant_fund.metrics.conformal.conformal_quantile`.
    ``+inf`` is returned exactly when the candidate atom dominates,
    ``H_y > alpha*W/(1-alpha)`` — the forced-inclusion regime of paper Eq. 21
    where the candidate joins the set under EVERY score; unlike the cousin's
    max-score clip, the paper's convention keeps the vacuous threshold
    explicit so half-line sets stay honest about infinite length.
    """
    s = _finite_1d(scores, "scores")
    w = _positive_1d(weights, "weights")
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    hy = float(candidate_weight)
    if not np.isfinite(hy) or hy <= 0.0:
        raise ValueError("candidate_weight must be finite and > 0")
    a = _check_alpha(alpha)
    order = np.argsort(s, kind="mergesort")
    cumulative = np.cumsum(w[order])
    total = float(cumulative[-1])
    target = (1.0 - a) * (total + hy)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    if idx >= s.size:
        return float("inf")
    return float(s[order][idx])


def extra_weighted_rank(
    scores: Array, weights: Array, candidate_score: float, candidate_weight: float
) -> float:
    """Weighted conformal p-value of a candidate response (paper Eq. 12).

    ``p^(y) = [H_y + sum_i H_i 1{V_i >= v}] / [H_y + sum_i H_i]``; the
    numerator counts tied calibration scores (conservative direction, paper's
    convention). The candidate is retained when ``p^(y) > alpha``
    (Eq. 13), which on a response half-line is equivalent to
    ``v <= q`` of :func:`extra_weighted_calibration` (paper App. B).
    """
    s = _finite_1d(scores, "scores")
    w = _positive_1d(weights, "weights")
    if s.size != w.size:
        raise ValueError("scores and weights must have the same length")
    v = float(candidate_score)
    hy = float(candidate_weight)
    if not np.isfinite(v):
        raise ValueError("candidate_score must be finite")
    if not np.isfinite(hy) or hy <= 0.0:
        raise ValueError("candidate_weight must be finite and > 0")
    numerator = hy + float(np.sum(w * (s >= v).astype(float)))
    return float(numerator / (hy + float(np.sum(w))))


def extra_tilted_predictive(source_model: SourceConditional, beta: Array) -> TiltedPredictive:
    """Tilt the fitted source predictive distribution (paper Eq. 9-10).

    ``p_beta(y|x) = h_beta(x,y) p_0(y|x) / Z^pred_beta(x)`` with the fitted
    normalizer ``Z^pred = M^_beta(x)`` (paper Section 3.1 uses ``p_0 = p^_P``,
    so the predictive normalizer IS the conditional moment). For the sign
    tilt this is closed-form: the within-mode truncated-Gaussian components
    are unchanged (``h`` is constant on each mode) and the gate logit shifts
    by ``2b`` — a Saerens-style prior adjustment; the input-only factor
    ``e^{a*u}`` cancels from the conditional law (paper Section 3.1).
    """
    if not isinstance(source_model, SourceConditional):
        raise TypeError("source_model must be a fitted SourceConditional")
    return TiltedPredictive(source=source_model, beta=_check_beta(beta))


@dataclass(frozen=True)
class TiltedPredictive:
    """ExTRA-WCP-T predictive law: tilted gate, unchanged mode components."""

    source: SourceConditional
    beta: Array

    def mode_prob(self, x: Array) -> Array:
        """Tilted positive-mode probability ``expit(logit(pi^_+(x)) + 2b)``."""
        g = self.source._gate_linear(_input_2d(x, "x"))
        return np.asarray(expit(g + 2.0 * self.beta[1]), dtype=float)

    def log_normalizer(self, x: Array) -> Array:
        """``log Z^pred_beta(x) = log M^_beta(x)`` (paper Eq. 9)."""
        return self.source.log_moment(x, self.beta)

    def log_density(self, x: Array, y: Array) -> Array:
        """``log p_beta(y|x) = log p_0(y|x) + log h_beta - log M^_beta(x)``."""
        return np.asarray(
            self.source.log_density(x, y) + self._log_h(x, y) - self.log_normalizer(x),
            dtype=float,
        )

    def score(self, x: Array, y: Array) -> Array:
        """Tilted score ``S_beta = S_0 - log h_beta + log Z^pred`` (Eq. 10)."""
        return np.asarray(-self.log_density(x, y), dtype=float)

    def _log_h(self, x: Array, y: Array) -> Array:
        xp = _input_2d(x, "x")
        z = _signed_1d(y, "y")
        if z.size != xp.shape[0]:
            raise ValueError("x and y must have one value per row")
        return np.asarray(self.beta[0] * xp[:, 0] + self.beta[1] * z, dtype=float)


@dataclass(frozen=True)
class PredictionIntervals:
    """Per-half-line inverted prediction sets (paper App. B inversion).

    ``lower_pos/upper_pos`` bound the set on ``{y > 0}`` and
    ``lower_neg/upper_neg`` on ``{y < 0}``; NaN marks an EMPTY component
    (threshold below the mode's minimum score ``c_z(x)``) and ``+/-inf`` a
    forced full half-line (candidate-atom dominance, Eq. 21). ``thresholds_*``
    keep the raw Eq. 26 quantiles (possibly ``+inf``) for diagnostics. Total
    length is the SUM of component lengths, allowing disconnected sets
    (paper Section 5 convention).
    """

    lower_pos: Array
    upper_pos: Array
    lower_neg: Array
    upper_neg: Array
    thresholds_pos: Array
    thresholds_neg: Array
    alpha: float

    def total_length(self) -> Array:
        def length(lo: Array, hi: Array) -> Array:
            empty = np.isnan(lo) | np.isnan(hi)
            return np.asarray(np.where(empty, 0.0, hi - lo), dtype=float)

        return np.asarray(
            length(self.lower_pos, self.upper_pos) + length(self.lower_neg, self.upper_neg),
            dtype=float,
        )

    def covers(self, y: Array) -> Array:
        yy = _finite_1d(y, "y")
        if yy.size != self.lower_pos.size:
            raise ValueError("y must have one value per prediction row")
        in_pos = (yy >= self.lower_pos) & (yy <= self.upper_pos)
        in_neg = (yy >= self.lower_neg) & (yy <= self.upper_neg)
        return np.asarray((in_pos | in_neg).astype(float), dtype=float)

    def coverage(self, y: Array) -> float:
        return float(np.mean(self.covers(y)))


def extra_prediction_intervals(
    source_model: SourceConditional,
    beta: Array,
    x_cal: Array,
    y_cal: Array,
    x_test: Array,
    alpha: float,
    *,
    tilt_score: bool = False,
) -> PredictionIntervals:
    """Build ExTRA-WCP (-T if ``tilt_score``) sets over the full response line.

    Shared weights ``H_i = h_beta^(X_i, Y_i)`` on the calibration pairs; the
    score is ``S_0`` (WCP) or ``S_beta`` (WCP-T) for BOTH calibration and
    candidates (paper Figure 1: the two procedures differ only in the score).
    Per test input and half-line ``z``, the threshold is Eq. 26 with the
    candidate atom ``H_z(x) = h_beta^(x, z)`` (constant within a half-line
    for the sign tilt), and the set on that half-line inverts
    ``S(x, y) <= q`` analytically (App. B): ``S_0(x, y) = c_z(x) +
    (y - mu_z(x))^2 / (2 sigma^2(u))`` gives the radius ``r = sigma *
    sqrt(2 (q~ - c_z))`` intersected with ``{z*y > 0}``. Regression sets are
    checked over the FULL response space (paper Section 4.2: both half-lines
    are always inverted — candidate weights can otherwise force unbounded
    tails). Fail-closed on empty/non-finite inputs, zero responses, alpha
    outside (0,1), or a tilt parameter that is not a finite 2-vector.
    """
    if not isinstance(source_model, SourceConditional):
        raise TypeError("source_model must be a fitted SourceConditional")
    b = _check_beta(beta)
    a_coef = _check_alpha(alpha)
    x_c = _input_2d(x_cal, "x_cal")
    y_c = _finite_1d(y_cal, "y_cal")
    if x_c.shape[0] != y_c.size:
        raise ValueError("x_cal and y_cal must have the same number of rows")
    x_t = _input_2d(x_test, "x_test")
    z_c = _signed_1d(y_cal, "y_cal")

    weights = np.exp(b[0] * x_c[:, 0] + b[1] * z_c)
    if tilt_score:
        tilted = TiltedPredictive(source=source_model, beta=b)
        scores = tilted.score(x_c, y_c)
    else:
        scores = source_model.score(x_c, y_c)
    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1])

    n_t = x_t.shape[0]
    u_t = x_t[:, 0]
    log_moment_t = source_model.log_moment(x_t, b)
    sig_t = source_model.scale(x_t)
    log_sig_t = np.log(sig_t)
    gate_t = source_model._gate_linear(x_t)

    lowers: list[Array] = []
    uppers: list[Array] = []
    thresholds: list[Array] = []
    for z in (1.0, -1.0):
        z_arr = np.full(n_t, z)
        cand_weight = np.exp(b[0] * u_t + b[1] * z_arr)
        target = (1.0 - a_coef) * (total + cand_weight)
        idx = np.searchsorted(cumulative, target, side="left")
        safe_idx = np.minimum(idx, scores_sorted.size - 1)
        q = np.asarray(np.where(idx < scores_sorted.size, scores_sorted[safe_idx], np.inf), float)
        # Convert the tilted-score threshold back to source-score units:
        # S_beta <= q  <=>  S_0 <= q + log h_beta - log M^_beta(x)  (Eq. 10).
        q_src = q + (b[0] * u_t + b[1] * z_arr - log_moment_t) if tilt_score else q
        mu = source_model.mean_z(x_t, z_arr)
        log_pi_z = np.where(z > 0.0, -np.logaddexp(0.0, -gate_t), -np.logaddexp(0.0, gate_t))
        c_z = -log_pi_z + 0.5 * _LOG2PI + log_sig_t + log_ndtr(z * mu / sig_t)
        full = np.isinf(q_src)
        valid = (~full) & (q_src >= c_z)
        radius = sig_t * np.sqrt(2.0 * np.maximum(np.where(valid, q_src - c_z, 0.0), 0.0))
        lo = mu - radius
        hi = mu + radius
        if z > 0.0:
            lo = np.maximum(lo, 0.0)
        else:
            hi = np.minimum(hi, 0.0)
        empty = valid & (hi < lo)
        lo_out = np.where(full, 0.0 if z > 0.0 else -np.inf, np.where(valid & ~empty, lo, np.nan))
        hi_out = np.where(full, np.inf if z > 0.0 else 0.0, np.where(valid & ~empty, hi, np.nan))
        lowers.append(np.asarray(lo_out, dtype=float))
        uppers.append(np.asarray(hi_out, dtype=float))
        thresholds.append(np.asarray(q, dtype=float))
    return PredictionIntervals(
        lower_pos=lowers[0],
        upper_pos=uppers[0],
        lower_neg=lowers[1],
        upper_neg=uppers[1],
        thresholds_pos=thresholds[0],
        thresholds_neg=thresholds[1],
        alpha=a_coef,
    )


@dataclass(frozen=True)
class TiltDecision:
    """Output of :func:`when_to_tilt_diagnostic` (conservative by default)."""

    recommend_tilt: bool
    decision: str
    mode_signal_sd: float
    tilt_coefficient_spread: float
    weight_ess_percent: float
    rationale: str


def when_to_tilt_diagnostic(
    source_model: SourceConditional,
    tilt_fit: TiltFit,
    x_src: Array,
    cal_weights: Array,
    *,
    tilt_fit_check: TiltFit | None = None,
    min_mode_signal_sd: float = 0.05,
    max_tilt_coefficient_spread: float = 0.5,
    min_weight_ess_percent: float = 20.0,
) -> TiltDecision:
    """Heuristic tilt / no-tilt signal — the paper leaves this decision OPEN.

    Paper abstract and Section 6: deciding when predictive tilting is safe
    from source labels and target inputs alone "remains an open problem",
    and good weighted-calibration coverage does NOT ensure tilted coverage.
    This diagnostic therefore only operationalizes observable signals the
    paper offers, and defaults to WEIGHTS-ONLY (ExTRA-WCP):

    1. ``mode_signal_sd`` — SD of the fitted mode probability ``pi^_+(x)``
       over source inputs. Paper Prop. 1: a nonconstant mode probability is
       what makes the response tilt identified from target inputs; at a
       constant mode probability Eq. 25 shows target inputs carry NO
       information about ``b``, so tilting is unfounded.
    2. ``tilt_coefficient_spread`` — range of the response coefficient ``b^``
       over every converged optimizer start of ``tilt_fit`` and, when given,
       of ``tilt_fit_check`` (the same fit under the paper's larger App. B
       regularization-check penalty, see
       :func:`regularization_check_ridge`). Paper Section 5.4: unstable,
       wrong-sign, or bound-saturated fits accompany the tilting coverage
       losses, and the larger penalty reverses the WCP/WCP-T ordering at weak
       signal — explicitly a post hoc association, not a validated rule.
    3. ``weight_ess_percent`` — ESS% of the calibration tilt weights; low ESS
       flags a fragile ratio estimate for BOTH procedures (weights compose
       the cousin-module ESS convention).

    Passing all three thresholds is necessary, never sufficient, for
    ``recommend_tilt``. Fail-closed: fewer than two usable converged
    coefficients (across starts and penalties) yields an infinite spread
    (no tilt).
    """
    if not isinstance(source_model, SourceConditional):
        raise TypeError("source_model must be a fitted SourceConditional")
    if not isinstance(tilt_fit, TiltFit):
        raise TypeError("tilt_fit must be a TiltFit from fit_exponential_tilt")
    if tilt_fit_check is not None and not isinstance(tilt_fit_check, TiltFit):
        raise TypeError("tilt_fit_check must be None or a TiltFit")
    x = _input_2d(x_src, "x_src")
    w = _positive_1d(cal_weights, "cal_weights")
    for name, value in (
        ("min_mode_signal_sd", min_mode_signal_sd),
        ("max_tilt_coefficient_spread", max_tilt_coefficient_spread),
        ("min_weight_ess_percent", min_weight_ess_percent),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")

    sd = float(np.std(source_model.mode_prob(x)))
    fits = [tilt_fit] + ([tilt_fit_check] if tilt_fit_check is not None else [])
    response: list[float] = []
    for fit in fits:
        usable = np.asarray(fit.start_converged, dtype=bool) & np.isfinite(
            np.asarray(fit.start_objectives, dtype=float)
        )
        response.extend(float(b) for b in np.asarray(fit.start_betas, dtype=float)[usable, 1])
    spread = float(np.max(response) - np.min(response)) if len(response) >= 2 else float("inf")
    ess = float(np.sum(w)) ** 2 / (float(w.size) * float(np.sum(w * w)))
    ess_pct = 100.0 * ess

    passed = {
        "mode_signal": sd >= float(min_mode_signal_sd),
        "tilt_stability": spread <= float(max_tilt_coefficient_spread),
        "weight_ess": ess_pct >= float(min_weight_ess_percent),
    }
    recommend = all(passed.values())
    failed = sorted(name for name, ok_flag in passed.items() if not ok_flag)
    rationale = (
        "tilt: all observable signals pass (identified mode signal, stable tilt across "
        "starts, healthy weight ESS); still a heuristic — the paper proves no selection "
        "rule from source labels + target inputs"
        if recommend
        else "weights_only (conservative default): failed signals "
        + ", ".join(failed)
        + "; paper arXiv:2609.30886 leaves the tilt/no-tilt decision open and shows "
        "tilting can lose substantial coverage when target inputs are uninformative "
        "about the response shift"
    )
    return TiltDecision(
        recommend_tilt=bool(recommend),
        decision="tilt" if recommend else "weights_only",
        mode_signal_sd=sd,
        tilt_coefficient_spread=spread,
        weight_ess_percent=ess_pct,
        rationale=rationale,
    )


@dataclass(frozen=True)
class SyntheticExTRAData:
    """SYNTHETIC fixture samples (paper Section 2.1). Correctness, not market.

    Five mutually independent samples plus a labeled target test set
    (EVALUATION only) and an unlabeled surrogate pool (source-distributed,
    for resampling under the fitted-weights surrogate ``Q^`` of Eq. 14-15).
    """

    eta: float
    a_star: float
    b_star: float
    d_eta: float
    p_plus: float
    x_train: Array
    y_train: Array
    x_shift_src: Array
    y_shift_src: Array
    x_shift_target: Array
    x_cal: Array
    y_cal: Array
    x_test: Array
    y_test: Array
    x_surrogate_pool: Array
    y_surrogate_pool: Array

    def true_log_tilt(self, x: Array, y: Array) -> Array:
        """``log h*(x,y) = a* u + b* sign(y)`` (eval-only oracle diagnostic)."""
        return np.asarray(
            self.a_star * _input_2d(x, "x")[:, 0] + self.b_star * _signed_1d(y, "y"), dtype=float
        )


def _draw_uniform_inputs(n: int, rng: np.random.Generator) -> Array:
    return np.asarray(
        np.column_stack([rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n)]), dtype=float
    )


def _draw_truncated_response(u: Array, v: Array, pi_pos: Array, rng: np.random.Generator) -> Array:
    """Draw ``Y`` from paper Eq. 22: gate ``pi_pos``, then truncated normal."""
    n = u.size
    z = np.where(rng.random(n) < pi_pos, 1.0, -1.0)
    mu = 1.25 * z + 0.10 * v
    sig = 0.22 * np.exp(0.8 * u)
    cdf0 = ndtr(-mu / sig)
    unif = rng.random(n)
    arg_pos = np.clip(cdf0 + unif * (1.0 - cdf0), 1e-15, 1.0 - 1e-15)
    arg_neg = np.clip(unif * cdf0, 1e-15, 1.0 - 1e-15)
    y = np.where(z > 0.0, mu + sig * ndtri(arg_pos), mu + sig * ndtri(arg_neg))
    return np.asarray(y, dtype=float)


def _target_input_density(
    u: Array, v: Array, eta: float, d_eta: float, a: float, b: float
) -> Array:
    """Unnormalized target input density ``M*(u,v)`` (paper Eq. 3 / App. B)."""
    pi = expit(d_eta + 3.0 * eta * v)
    return np.asarray(np.exp(a * u) * (pi * np.exp(b) + (1.0 - pi) * np.exp(-b)), dtype=float)


def _draw_target_inputs(
    n: int, eta: float, d_eta: float, a: float, b: float, rng: np.random.Generator
) -> Array:
    """Rejection-sample target inputs from ``M*`` (paper App. B), seeded."""
    grid = np.linspace(-1.0, 1.0, 401)
    gu, gv = np.meshgrid(grid, grid, indexing="ij")
    m_max = float(np.max(_target_input_density(gu.ravel(), gv.ravel(), eta, d_eta, a, b)))
    if not np.isfinite(m_max) or m_max <= 0.0:
        raise ValueError("target input density bound must be finite and positive")
    m_max *= 1.0 + 1e-9
    out_u: list[Array] = []
    out_v: list[Array] = []
    have = 0
    for _ in range(200):
        batch = max(4 * (n - have), 1024)
        u = rng.uniform(-1.0, 1.0, batch)
        v = rng.uniform(-1.0, 1.0, batch)
        dens = _target_input_density(u, v, eta, d_eta, a, b)
        keep = rng.random(batch) < dens / m_max
        out_u.append(u[keep])
        out_v.append(v[keep])
        have += int(np.sum(keep))
        if have >= n:
            break
    if have < n:
        raise ValueError("target input rejection sampling did not fill the request")
    u_all = np.concatenate(out_u)[:n]
    v_all = np.concatenate(out_v)[:n]
    return np.asarray(np.column_stack([u_all, v_all]), dtype=float)


def synthetic_bimodal_shift(
    eta: float = 1.0,
    a_star: float = 1.0,
    b_star: float = 1.2,
    *,
    n_train: int = 1200,
    n_shift: int = 1200,
    m_target: int = 1500,
    n_cal: int = 1000,
    n_test: int = 1500,
    n_surrogate_pool: int = 2000,
    seed: int = 0,
) -> SyntheticExTRAData:
    """Seeded SYNTHETIC fixture: paper Section 5.2 DGP with the eta ladder.

    Source (Eq. 22): ``X = (U,V)`` iid Uniform[-1,1]; ``Z = sign(Y)`` with
    ``P(Z=+1|u,v) = logit^-1(d_eta + 3*eta*v)``; ``Y|u,v,z ~ N(1.25z +
    0.10v, (0.22 e^{0.8u})^2)`` restricted to ``{zy > 0}``. ``d_eta`` holds
    the positive-mode probability ``p_+`` fixed across the ladder (App. B).
    Target: joint tilt ``h* = exp(a* u + b* sign(y))`` (Eq. 23) — inputs
    drawn by rejection from the induced marginal ``M*``, conditional mode
    log-odds shifted by ``2 b*``, within-mode laws unchanged. ``eta`` is the
    mode-signal knob of Section 5.4: ``eta = 1`` is the informative design
    (target inputs identify both ``a*`` and ``b*``, Prop. 1); ``eta = 0``
    makes the mode probability constant, so Eq. 25 applies and target inputs
    carry NO information about ``b*``. Correctness test, never market
    evidence. Fail-closed on invalid sizes / parameters.
    """
    sizes = {
        "n_train": n_train,
        "n_shift": n_shift,
        "m_target": m_target,
        "n_cal": n_cal,
        "n_test": n_test,
        "n_surrogate_pool": n_surrogate_pool,
    }
    for name, value in sizes.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1")
    for pname, pvalue in (("eta", eta), ("a_star", a_star), ("b_star", b_star)):
        if not np.isfinite(float(pvalue)):
            raise ValueError(f"{pname} must be finite")
    if float(eta) < 0.0:
        raise ValueError("eta must be >= 0")

    e = float(eta)
    d_eta = solve_mode_intercept(e)
    rng = np.random.default_rng(int(seed))

    def source_pair(n: int) -> tuple[Array, Array]:
        x = _draw_uniform_inputs(n, rng)
        pi = expit(d_eta + 3.0 * e * x[:, 1])
        return x, _draw_truncated_response(x[:, 0], x[:, 1], pi, rng)

    x_train, y_train = source_pair(int(n_train))
    x_shift_src, y_shift_src = source_pair(int(n_shift))
    x_shift_target = _draw_target_inputs(int(m_target), e, d_eta, a_star, b_star, rng)
    x_cal, y_cal = source_pair(int(n_cal))
    x_test = _draw_target_inputs(int(n_test), e, d_eta, a_star, b_star, rng)
    pi_target = expit(d_eta + 3.0 * e * x_test[:, 1] + 2.0 * float(b_star))
    y_test = _draw_truncated_response(x_test[:, 0], x_test[:, 1], pi_target, rng)
    x_pool, y_pool = source_pair(int(n_surrogate_pool))
    return SyntheticExTRAData(
        eta=e,
        a_star=float(a_star),
        b_star=float(b_star),
        d_eta=d_eta,
        p_plus=float(SOURCE_POSITIVE_MODE_PROB),
        x_train=x_train,
        y_train=y_train,
        x_shift_src=x_shift_src,
        y_shift_src=y_shift_src,
        x_shift_target=x_shift_target,
        x_cal=x_cal,
        y_cal=y_cal,
        x_test=x_test,
        y_test=y_test,
        x_surrogate_pool=x_pool,
        y_surrogate_pool=y_pool,
    )


def surrogate_resample(
    x_pool: Array, y_pool: Array, beta: Array, n_draws: int, seed: int
) -> tuple[Array, Array]:
    """Resample source pairs under the surrogate ``Q^ = w^.P`` (paper Eq. 14).

    Draws ``n_draws`` pairs with probabilities proportional to the fitted
    tilt ``h_beta^`` (the normalizer cancels). Used to check the shared
    guarantee of Eq. 15: under the distribution the fitted weights DEFINE,
    both scores cover at ``>= 1 - alpha``. Seeded and deterministic.
    """
    x = _input_2d(x_pool, "x_pool")
    y = _finite_1d(y_pool, "y_pool")
    if x.shape[0] != y.size:
        raise ValueError("x_pool and y_pool must have the same number of rows")
    n = int(n_draws)
    if n < 1:
        raise ValueError("n_draws must be >= 1")
    h = tilt_weights(x, y, _check_beta(beta))
    probs = h / float(np.sum(h))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=n, replace=True, p=probs)
    return (
        np.asarray(x[idx], dtype=float),
        np.asarray(y[idx], dtype=float),
    )


@dataclass(frozen=True)
class ExtraRunResult:
    """One replication of the paper's controlled comparison (Section 5).

    All three methods share the learned source predictor, fitted ratio,
    calibration sample, and test observations; target responses are used for
    EVALUATION only. ``tilt_fit`` uses the main App. B penalty (1/n_shift);
    ``tilt_fit_check`` refits under the larger regularization-check penalty
    for the instability signal of :func:`when_to_tilt_diagnostic`.
    ``coverage_diff_wcp_t_minus_wcp`` and ``length_reduction_percent`` are
    the paper's paired statistics.
    """

    source_model: SourceConditional
    tilt_fit: TiltFit
    tilt_fit_check: TiltFit
    intervals_standard: PredictionIntervals
    intervals_wcp: PredictionIntervals
    intervals_wcp_t: PredictionIntervals
    coverage_standard: float
    length_standard: float
    coverage_wcp: float
    length_wcp: float
    coverage_wcp_t: float
    length_wcp_t: float
    length_reduction_percent: float
    coverage_diff_wcp_t_minus_wcp: float
    mode_signal_sd: float
    weight_ess_percent: float

    def tilt_coefficient_spread(self) -> float:
        """Range of ``b^`` over all usable starts of both penalties."""
        values: list[float] = []
        for fit in (self.tilt_fit, self.tilt_fit_check):
            usable = np.asarray(fit.start_converged, dtype=bool) & np.isfinite(
                np.asarray(fit.start_objectives, dtype=float)
            )
            values.extend(float(b) for b in np.asarray(fit.start_betas, dtype=float)[usable, 1])
        if len(values) < 2:
            return float("inf")
        return float(np.max(values) - np.min(values))


def run_extra_replication(
    data: SyntheticExTRAData,
    *,
    alpha: float = 0.10,
    bounds: tuple[tuple[float, float], ...] = PAPER_TILT_BOUNDS,
    ridge: float | None = None,
    starts: tuple[tuple[float, float], ...] = DEFAULT_TILT_STARTS,
    c_gate: float = 100.0,
) -> ExtraRunResult:
    """Fit source + ExTRA on one fixture draw and run WCP vs WCP-T (paper §5).

    Standard CP (uniform weights, source score) is the no-ratio baseline.
    Deterministic given the fixture. Fail-closed via the underlying
    estimators.
    """
    if not isinstance(data, SyntheticExTRAData):
        raise TypeError("data must be a SyntheticExTRAData fixture")
    a = _check_alpha(alpha)
    source_model = fit_source_model(data.x_train, data.y_train, c_gate=c_gate)
    tilt_fit = fit_exponential_tilt(
        source_model,
        data.x_shift_src,
        data.y_shift_src,
        data.x_shift_target,
        bounds=bounds,
        ridge=ridge,
        starts=starts,
    )
    tilt_fit_check = fit_exponential_tilt(
        source_model,
        data.x_shift_src,
        data.y_shift_src,
        data.x_shift_target,
        bounds=bounds,
        ridge=regularization_check_ridge(tilt_fit.n_shift, tilt_fit.m_target),
        starts=starts,
    )
    beta = tilt_fit.beta
    zero = np.zeros(2)
    intervals_standard = extra_prediction_intervals(
        source_model, zero, data.x_cal, data.y_cal, data.x_test, a
    )
    intervals_wcp = extra_prediction_intervals(
        source_model, beta, data.x_cal, data.y_cal, data.x_test, a, tilt_score=False
    )
    intervals_wcp_t = extra_prediction_intervals(
        source_model, beta, data.x_cal, data.y_cal, data.x_test, a, tilt_score=True
    )

    def summarize(iv: PredictionIntervals) -> tuple[float, float]:
        return iv.coverage(data.y_test), float(np.mean(iv.total_length()))

    cov_s, len_s = summarize(intervals_standard)
    cov_w, len_w = summarize(intervals_wcp)
    cov_t, len_t = summarize(intervals_wcp_t)
    weights = tilt_weights(data.x_cal, data.y_cal, beta)
    ess = float(np.sum(weights)) ** 2 / (float(weights.size) * float(np.sum(weights * weights)))
    return ExtraRunResult(
        source_model=source_model,
        tilt_fit=tilt_fit,
        tilt_fit_check=tilt_fit_check,
        intervals_standard=intervals_standard,
        intervals_wcp=intervals_wcp,
        intervals_wcp_t=intervals_wcp_t,
        coverage_standard=cov_s,
        length_standard=len_s,
        coverage_wcp=cov_w,
        length_wcp=len_w,
        coverage_wcp_t=cov_t,
        length_wcp_t=len_t,
        length_reduction_percent=100.0 * (1.0 - len_t / len_w) if len_w > 0.0 else float("nan"),
        coverage_diff_wcp_t_minus_wcp=cov_t - cov_w,
        mode_signal_sd=float(np.std(source_model.mode_prob(data.x_shift_src))),
        weight_ess_percent=100.0 * ess,
    )


def surrogate_coverage(
    data: SyntheticExTRAData, result: ExtraRunResult, *, seed: int, n_draws: int | None = None
) -> tuple[float, float]:
    """Coverage of WCP and WCP-T sets under the fitted-weights surrogate.

    Resamples the source-distributed surrogate pool with probabilities
    proportional to ``h_beta^`` (Eq. 14) and evaluates both interval families
    on the resampled pairs — the empirical face of the shared guarantee
    ``>= 1 - alpha`` of Eq. 15. Returns ``(coverage_wcp, coverage_wcp_t)``.
    Seeded; target responses are never involved.
    """
    if not isinstance(data, SyntheticExTRAData):
        raise TypeError("data must be a SyntheticExTRAData fixture")
    if not isinstance(result, ExtraRunResult):
        raise TypeError("result must be an ExtraRunResult")
    n = int(n_draws) if n_draws is not None else int(data.y_test.size)
    x_s, y_s = surrogate_resample(
        data.x_surrogate_pool, data.y_surrogate_pool, result.tilt_fit.beta, n, seed
    )
    # Intervals are input-dependent, so they are rebuilt at the resampled
    # inputs (calibration, fitted predictor, and fitted ratio stay fixed —
    # only the test distribution changes from Q to Q^, as in Eq. 15-16).
    rebuilt_w = extra_prediction_intervals(
        result.source_model,
        result.tilt_fit.beta,
        data.x_cal,
        data.y_cal,
        x_s,
        result.intervals_wcp.alpha,
        tilt_score=False,
    )
    rebuilt_t = extra_prediction_intervals(
        result.source_model,
        result.tilt_fit.beta,
        data.x_cal,
        data.y_cal,
        x_s,
        result.intervals_wcp_t.alpha,
        tilt_score=True,
    )
    return rebuilt_w.coverage(y_s), rebuilt_t.coverage(y_s)


def bench_extra_tilt(
    *,
    eta: float = 1.0,
    a_star: float = 1.0,
    b_star: float = 1.2,
    alpha: float = 0.10,
    seed: int = 17,
    replications: int = 10,
    n_train: int = 1200,
    n_shift: int = 1200,
    m_target: int = 1500,
    n_cal: int = 1000,
    n_test: int = 1500,
) -> dict[str, float | str]:
    """Coverage and set length of ExTRA-WCP vs WCP-T on the SYNTHETIC fixture.

    Proper scores only (coverage, prediction-set length, estimator
    diagnostics); no P&L / Sharpe-family keys; never market evidence. Means
    of replication-level quantities with the paper's paired statistics
    (length reduction %, coverage difference). ``eta=1`` is the informative
    design where the paper finds ~30% paired length reduction at near-nominal
    coverage (Table 2); ``eta=0`` is the uninformative warning case where
    tilting loses substantial coverage (Figure 3: WCP 0.727 vs WCP-T 0.493).
    Deterministic: replication r uses seed ``seed + 1000*r``.
    """
    reps = int(replications)
    if reps < 1:
        raise ValueError("replications must be >= 1")
    a = _check_alpha(alpha)
    rows: list[ExtraRunResult] = []
    for r in range(reps):
        data = synthetic_bimodal_shift(
            eta,
            a_star,
            b_star,
            n_train=n_train,
            n_shift=n_shift,
            m_target=m_target,
            n_cal=n_cal,
            n_test=n_test,
            seed=int(seed) + 1000 * r,
        )
        rows.append(run_extra_replication(data, alpha=a))

    def col(get: Callable[[ExtraRunResult], float]) -> Array:
        return np.asarray([get(row) for row in rows], dtype=float)

    betas = np.vstack([row.tilt_fit.beta for row in rows])
    spreads = np.asarray([row.tilt_coefficient_spread() for row in rows], dtype=float)

    def mean(get: Callable[[ExtraRunResult], float]) -> float:
        return float(np.mean(col(get)))

    return {
        "synthetic_coverage_standard_cp": mean(lambda r: r.coverage_standard),
        "synthetic_mean_length_standard_cp": mean(lambda r: r.length_standard),
        "synthetic_coverage_extra_wcp": mean(lambda r: r.coverage_wcp),
        "synthetic_mean_length_extra_wcp": mean(lambda r: r.length_wcp),
        "synthetic_coverage_extra_wcp_t": mean(lambda r: r.coverage_wcp_t),
        "synthetic_mean_length_extra_wcp_t": mean(lambda r: r.length_wcp_t),
        "synthetic_paired_length_reduction_percent": mean(lambda r: r.length_reduction_percent),
        "synthetic_coverage_diff_wcp_t_minus_wcp": mean(lambda r: r.coverage_diff_wcp_t_minus_wcp),
        "synthetic_a_hat_mean": float(np.mean(betas[:, 0])),
        "synthetic_b_hat_mean": float(np.mean(betas[:, 1])),
        "synthetic_b_hat_spread_mean": float(np.mean(spreads)),
        "synthetic_mode_signal_sd": mean(lambda r: r.mode_signal_sd),
        "synthetic_weight_ess_percent": mean(lambda r: r.weight_ess_percent),
        "synthetic_replications": float(reps),
        "synthetic_n": float(n_test),
        "synthetic_alpha": a,
        "synthetic_eta": float(eta),
        "synthetic_a_star": float(a_star),
        "synthetic_b_star": float(b_star),
        "synthetic_dgp": "fixture",
        "synthetic_claim": "research_metric_only",
        "synthetic_seed": float(seed),
    }
