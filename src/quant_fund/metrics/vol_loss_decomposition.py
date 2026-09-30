"""Loss-choice vs model-choice decomposition for volatility forecasts.

This lane ADDS a decomposition layer on top of the existing vol-comparison
machinery — ``research/vol_bench.py`` and ``metrics/vol_eval.py`` are reused
by import and never modified. Raw comparisons of volatility forecasts conflate
(a) persistent forecast-LEVEL differences induced by the training loss with
(b) differences in day-to-day forecast MOVEMENTS. Validation-based level
alignment removes (a) before evaluation so the residual comparison isolates
(b). Implements the comparison layer of:

    Tokajuk, A., Chudziak, J. A. (2026). "Loss Choice or Model Choice? The
    Role of Forecast Level in Cryptocurrency Volatility Forecasting."
    arXiv:2609.27024 [q-fin.CP]; accepted for the ADMA 2026 proceedings.
    Citation verified against https://arxiv.org/abs/2609.27024 and the full
    HTML text (fetched 2026-09-30).

Paper machinery implemented here (section references are to the fetched text)
---------------------------------------------------------------------------
1. ``validation_level_alignment`` — Eq. (1), §3.2: one constant per
   loss-model fit, ``c = (1/|V|) Σ_{t∈V} h_t / f_t``, applied as
   ``f_aligned_t = c · f_t`` on the TEST window only. ``c`` is the minimizer
   of validation QLIKE over constant rescalings:
   ``QLIKE_V(c) = mean(h/f)/c + log c + const`` is strictly convex in ``c``
   with stationary point ``c = mean(h/f)``. Fitted on validation NEVER test;
   overlapping validation/test masks fail closed (lookahead guard). A
   documented additive variant (``b = mean(h − f)``, least-squares optimal
   over constant shifts, minimizes validation MSE) is a lane extension — the
   paper uses the multiplicative form only.
2. ``pairwise_marginal_gaps`` / ``loss_model_decomposition`` — §3.2, RQ1:
   the paper's ACTUAL decomposition statistic is not an ANOVA. For a score
   block ``q[o, m]`` (mean test QLIKE for loss ``o``, model ``m``), with
   marginals ``q̄_o`` (over models) and ``q̄_m`` (over losses):
   ``Δ_L = mean_{o<o'} |q̄_o − q̄_o'|`` and ``Δ_M = mean_{m<m'} |q̄_m − q̄_m'|``;
   the ratio ``Δ_L/Δ_M`` above one means loss choice produces the larger
   marginal score differences. Paper headline: median ratio 2.91 (95% CI
   [2.27, 3.25]) raw → 0.67 (CI [0.56, 0.99]) aligned — the loss-dominated
   to model-dominated flip. Because the paper's statistic "does not separate
   loss-model interactions" (§3.2), ``two_way_score_shares`` is added as a
   documented supplementary two-way ANOVA-style eta² split whose residual
   share absorbs interaction + noise.
3. ``level_share`` — §3.2, RQ2: for two forecasts of the same model,
   ``d_t = log f_{o,t} − log f_{o',t}`` and
   ``LevelShare = 1 − Σ_t (d_t − d̄)² / Σ_t d_t²``. Near one ⇒ a constant
   multiplicative offset (level) explains the difference; near zero ⇒
   time-varying (movement) differences. Uses test forecasts only and never
   determines the alignment constants. Paper: 89% (HAR) down to 56% (LightGBM).
4. ``gaussian_var_threshold`` / ``var_breach_rates`` — §4/§5.3, RQ3: one-day
   Gaussian ``VaR_t = −q_alpha · sqrt(f_t)`` (``q_alpha`` the standard-normal
   quantile), breach when the return falls below ``−VaR_t``, with the Kupiec
   (1995) POF coverage diagnostic (reused from ``metrics.probability``).
   Paper: raw cross-loss breach rates span 1.3%–6.3% at the 5% level;
   alignment narrows them to 3.5%–3.6%, removing 97% of the cross-loss
   variation (while aggregate coverage stays below nominal — alignment
   optimizes QLIKE, not tail calibration).
5. ``simulate_loss_model_world`` / ``run_loss_model_decomposition_study`` —
   seeded SYNTHETIC planted world (AGENTS.md honesty contract #2):
   GARCH(1,1) returns with ``forecasts[m, l, t] = c_l · base_m(t) · noise``,
   level factors ``c_l`` calibrated to the paper's Table 4 VaR magnitudes
   (squared: −21.3% ⇒ 0.62 … +68.7% ⇒ 2.85), movement quality differing by
   model. Correctness evidence for the flip mechanism ONLY — the paper's
   empirical numbers come from real Binance data; nothing here is market
   evidence.

Scores are proper/robust rules only — QLIKE (Patton 2011, robust to the
noisy squared-return proxy), MSE, MSE-log — reused by import from
``metrics.scoring`` / ``metrics.vol_eval``. Never headline Sharpe / P&L.
Research-diagnostic only; ``live_pnl_claim`` is False everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from quant_fund.metrics.probability import kupiec_pof
from quant_fund.metrics.scoring import qlike as mean_qlike
from quant_fund.metrics.vol_eval import mse as mse_elementwise
from quant_fund.metrics.vol_eval import mse_log as mse_log_elementwise
from quant_fund.utils.series import as_named_1d, require_same_length

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

STUDY_SCHEMA = "vol_loss_decomposition.v1"
ALIGNMENT_MODES = ("multiplicative", "additive")
# Mirrors the metrics.vol_eval minimum-observation contract.
MIN_WINDOW = 10

# Paper §4/§5.3 defaults: seven training losses (Table 1), five models. The
# planted level factors are the Table 4 raw VaR magnitudes relative to QLIKE,
# squared to variance scale: (1 − 0.191)² ≈ 0.65 for MSE-log … (1 + 0.687)²
# ≈ 2.85 for HMSE. QLIKE/Patton target the conditional mean ⇒ factor ≈ 1.
DEFAULT_LOSS_NAMES: tuple[str, ...] = (
    "mse_log",
    "huber_log",
    "pinball",
    "qlike",
    "patton_bb",
    "hmae",
    "hmse",
)
DEFAULT_LEVEL_FACTORS: tuple[float, ...] = (0.65, 0.63, 0.62, 1.00, 1.02, 1.56, 2.85)
DEFAULT_MODEL_NAMES: tuple[str, ...] = ("model_0", "model_1", "model_2", "model_3")
# Movement quality per model: tracking weight on the true conditional variance
# (rest shrinks to the unconditional variance) and mean-preserving daily
# log-noise scale. Differences are deliberately mild so the RAW score matrix
# stays loss-dominated, matching the paper's raw ratio > 1.
DEFAULT_TRACKING_WEIGHTS: tuple[float, ...] = (0.98, 0.92, 0.86, 0.80)
DEFAULT_NOISE_SCALES: tuple[float, ...] = (0.05, 0.11, 0.18, 0.26)
# Small loss-specific movement perturbation so LevelShare is not exactly 1.0
# (the paper's level shares are 56–89%, never degenerate).
DEFAULT_LOSS_MOVEMENT_NOISE = 0.05


def _require_2d_scores(scores: Array, name: str) -> Array:
    q = np.asarray(scores, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"{name} must be 2d (n_losses, n_models)")
    if q.shape[0] < 2 or q.shape[1] < 2:
        raise ValueError(f"{name} needs at least 2 losses and 2 models; got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError(f"{name} must be finite")
    return q


def _mean_pairwise_abs_diff(x: Array) -> float:
    """Mean |x_i − x_j| over all i < j pairs (the paper's Δ averaging)."""
    n = int(x.size)
    if n < 2:
        raise ValueError("pairwise gaps need at least 2 entries")
    rows, cols = np.triu_indices(n, 1)
    return float(np.mean(np.abs(x[rows] - x[cols])))


# ---------------------------------------------------------------------------
# 1. Validation-based forecast-level alignment (paper Eq. 1, §3.2)
# ---------------------------------------------------------------------------


def validation_level_alignment(
    forecast: Array,
    proxy: Array,
    *,
    validation_mask: BoolArray,
    test_mask: BoolArray,
    mode: str = "multiplicative",
) -> dict[str, Any]:
    r"""Fit a per-fit forecast-level constant on VALIDATION data only (Eq. 1).

    Multiplicative (the paper's variant): ``c = (1/|V|) Σ_{t∈V} h_t / f_t``,
    ``f_aligned_t = c · f_t`` on test dates. ``c`` minimizes validation QLIKE
    over constant rescalings — ``QLIKE_V(c) = mean(h/f)/c + \log c + const``
    has its unique stationary point at ``c = mean(h/f)``. Applying one
    constant to every test date changes the forecast level while preserving
    relative day-to-day movements.

    Additive (documented lane extension, not in the paper): ``b = mean(h − f)``
    on validation — the least-squares-optimal constant shift, minimizing
    validation MSE — applied as ``f_aligned = f + b``. An additive shift can
    break variance positivity; any non-positive aligned test forecast fails
    closed rather than being clipped.

    Fail-closed guards (no silent fallbacks):
    - validation and test masks overlapping in ANY index → ValueError
      (lookahead guard: the constant must never see test data);
    - either window shorter than ``MIN_WINDOW`` → ValueError;
    - non-boolean masks, length mismatches, non-finite entries, non-positive
      forecasts, negative proxies, non-positive fitted constant → ValueError;
    - ``mode`` outside ``ALIGNMENT_MODES`` → ValueError.

    Returns ``constant``, ``aligned`` (full-length copy with ONLY test entries
    transformed; validation rows stay raw — the paper applies ``c`` to test
    forecasts), ``aligned_test``, window sizes, and the validation QLIKE / MSE
    before vs after the correction (the minimality evidence).
    """
    f = as_named_1d("forecast", np.asarray(forecast, dtype=float))
    h = as_named_1d("proxy", np.asarray(proxy, dtype=float))
    require_same_length(("forecast", f), ("proxy", h))
    v = np.asarray(validation_mask)
    t = np.asarray(test_mask)
    if v.dtype != np.bool_ or t.dtype != np.bool_:
        raise ValueError("validation_mask and test_mask must be boolean arrays")
    if v.ndim != 1 or t.ndim != 1 or v.shape[0] != f.shape[0] or t.shape[0] != f.shape[0]:
        raise ValueError("masks must be 1d and match the forecast length")
    if f.size == 0:
        raise ValueError("forecast must be non-empty")
    if not np.isfinite(f).all() or not np.isfinite(h).all():
        raise ValueError("forecast and proxy must be finite")
    if np.any(v & t):
        raise ValueError(
            "validation and test windows overlap — alignment must be fitted "
            "on validation data only (fail-closed lookahead guard)"
        )
    n_val = int(np.count_nonzero(v))
    n_test = int(np.count_nonzero(t))
    if n_val < MIN_WINDOW:
        raise ValueError(f"validation window needs >= {MIN_WINDOW} observations")
    if n_test < MIN_WINDOW:
        raise ValueError(f"test window needs >= {MIN_WINDOW} observations")
    if np.any(f <= 0.0):
        raise ValueError("forecast must be positive (variance scale)")
    if np.any(h < 0.0):
        raise ValueError("proxy must be non-negative")
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"mode must be one of {ALIGNMENT_MODES}")

    fv = f[v]
    hv = h[v]
    aligned = f.copy()
    if mode == "multiplicative":
        constant = float(np.mean(hv / fv))
        if not np.isfinite(constant) or constant <= 0.0:
            raise ValueError("fitted alignment constant must be finite and positive")
        aligned[t] = constant * f[t]
    else:
        constant = float(np.mean(hv - fv))
        if not np.isfinite(constant):
            raise ValueError("fitted alignment shift must be finite")
        candidate = f[t] + constant
        if np.any(candidate <= 0.0):
            raise ValueError(
                "additive alignment produced a non-positive test forecast; "
                "fail closed rather than clip (use the multiplicative variant)"
            )
        aligned[t] = candidate
    val_aligned = constant * fv if mode == "multiplicative" else fv + constant
    return {
        "mode": mode,
        "constant": constant,
        "aligned": aligned,
        "aligned_test": aligned[t].copy(),
        "n_validation": n_val,
        "n_test": n_test,
        "validation_qlike_raw": float(mean_qlike(hv, fv)),
        "validation_qlike_aligned": float(mean_qlike(hv, val_aligned)),
        "validation_mse_raw": float(np.mean(mse_elementwise(hv, fv))),
        "validation_mse_aligned": float(np.mean(mse_elementwise(hv, val_aligned))),
    }


# ---------------------------------------------------------------------------
# 2. Score-matrix decomposition (paper §3.2 RQ1 statistic + supplement)
# ---------------------------------------------------------------------------


def pairwise_marginal_gaps(scores: Array) -> dict[str, float]:
    """The paper's Δ_L / Δ_M marginal-gap statistic for one score block.

    ``scores[o, m]`` is the mean test score (log QLIKE in the paper) for
    training loss ``o`` and model ``m``. ``Δ_L`` averages |q̄_o − q̄_o'| over
    all loss pairs of the model-averaged marginals; ``Δ_M`` averages over
    model pairs of the loss-averaged marginals. Unequal numbers of losses and
    models make raw ranges incomparable — pairwise averaging is the paper's
    fix (§3.2). ``loss_to_model_ratio`` > 1 ⇒ loss choice produces the larger
    marginal score differences. Like the paper's statistic, this does NOT
    separate loss-model interaction; see ``two_way_score_shares``.

    Degenerate input fails closed: non-2d, fewer than 2 losses or 2 models,
    non-finite entries, or zero marginal variation on BOTH axes (the ratio is
    undefined). ``Δ_M = 0`` with ``Δ_L > 0`` honestly returns ``inf``.
    """
    q = _require_2d_scores(np.asarray(scores, dtype=float), "scores")
    loss_marginal = q.mean(axis=1)
    model_marginal = q.mean(axis=0)
    delta_loss = _mean_pairwise_abs_diff(loss_marginal)
    delta_model = _mean_pairwise_abs_diff(model_marginal)
    if delta_loss == 0.0 and delta_model == 0.0:
        raise ValueError("score matrix has no marginal variation — ratio undefined")
    ratio = float(delta_loss / delta_model) if delta_model > 0.0 else float("inf")
    n_losses, n_models = q.shape
    return {
        "delta_loss": delta_loss,
        "delta_model": delta_model,
        "loss_to_model_ratio": ratio,
        "n_loss_pairs": float(n_losses * (n_losses - 1) / 2),
        "n_model_pairs": float(n_models * (n_models - 1) / 2),
    }


def two_way_score_shares(scores: Array) -> dict[str, float]:
    """Supplementary two-way ANOVA-style eta² split of one score block.

    NOT the paper's statistic — documented supplement answering "how much of
    total score variation is attributable to loss choice vs model choice".
    ``loss_share`` / ``model_share`` are the main-effect sums of squares over
    the total; ``residual_share`` absorbs the loss×model interaction plus
    noise (the paper explicitly declines to separate interaction, §3.2).
    A constant score matrix has no variation to attribute → ValueError.
    """
    q = _require_2d_scores(np.asarray(scores, dtype=float), "scores")
    n_losses, n_models = q.shape
    grand = float(q.mean())
    ss_total = float(np.sum((q - grand) ** 2))
    if ss_total <= 0.0:
        raise ValueError("constant score matrix — variance shares undefined")
    ss_loss = float(n_models * np.sum((q.mean(axis=1) - grand) ** 2))
    ss_model = float(n_losses * np.sum((q.mean(axis=0) - grand) ** 2))
    ss_residual = ss_total - ss_loss - ss_model
    return {
        "loss_share": ss_loss / ss_total,
        "model_share": ss_model / ss_total,
        "residual_share": ss_residual / ss_total,
        "ss_total": ss_total,
    }


def _as_blocks(scores: Array, name: str) -> Array:
    q = np.asarray(scores, dtype=float)
    if q.ndim == 2:
        q = q.reshape(1, *q.shape)
    if q.ndim != 3:
        raise ValueError(f"{name} must be (n_losses, n_models) or (n_blocks, n_losses, n_models)")
    if q.shape[1] < 2 or q.shape[2] < 2:
        raise ValueError(f"{name} needs at least 2 losses and 2 models; got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError(f"{name} must be finite")
    return q


def loss_model_decomposition(scores_raw: Array, scores_aligned: Array) -> dict[str, Any]:
    """The paper's central before/after comparison of Δ_L/Δ_M (§3.2, §5).

    ``scores_raw`` / ``scores_aligned`` are matching ``(n_losses, n_models)``
    score matrices (or ``(n_blocks, n_losses, n_models)`` stacks — the paper
    takes the median ratio over its 25 asset-fold blocks). Per block, both
    the paper's pairwise-marginal statistic and the supplementary eta²
    shares are computed. ``flip_loss_to_model`` is True iff the median ratio
    moves from > 1 (loss-dominated, paper: 2.91) to < 1 (model-dominated,
    paper: 0.67) — the central claim this lane validates on SYNTHETIC data.
    Shape mismatch between the two inputs fails closed.
    """
    raw = _as_blocks(scores_raw, "scores_raw")
    aligned = _as_blocks(scores_aligned, "scores_aligned")
    if raw.shape != aligned.shape:
        raise ValueError(f"shape mismatch: scores_raw={raw.shape}, scores_aligned={aligned.shape}")

    def _side(blocks: Array) -> dict[str, Any]:
        gaps = [pairwise_marginal_gaps(blocks[b]) for b in range(blocks.shape[0])]
        shares = [two_way_score_shares(blocks[b]) for b in range(blocks.shape[0])]
        return {
            "ratio_median": float(np.median([g["loss_to_model_ratio"] for g in gaps])),
            "ratio_by_block": [g["loss_to_model_ratio"] for g in gaps],
            "delta_loss_median": float(np.median([g["delta_loss"] for g in gaps])),
            "delta_model_median": float(np.median([g["delta_model"] for g in gaps])),
            "loss_share_median": float(np.median([s["loss_share"] for s in shares])),
            "model_share_median": float(np.median([s["model_share"] for s in shares])),
            "residual_share_median": float(np.median([s["residual_share"] for s in shares])),
            "per_block_gaps": gaps,
            "per_block_shares": shares,
        }

    raw_side = _side(raw)
    aligned_side = _side(aligned)
    return {
        "n_blocks": int(raw.shape[0]),
        "raw": raw_side,
        "aligned": aligned_side,
        "flip_loss_to_model": bool(raw_side["ratio_median"] > 1.0 > aligned_side["ratio_median"]),
    }


def level_share(forecast_a: Array, forecast_b: Array) -> float:
    """LevelShare of two forecasts of the same model (§3.2, RQ2).

    ``d_t = log f_a,t − log f_b,t``; ``LevelShare = 1 − Σ(d_t − d̄)² / Σ d_t²``
    — the fraction of squared log-forecast differences explained by a
    constant multiplicative offset. Near 1 ⇒ a pure forecast-level
    difference; near 0 ⇒ time-varying (movement) differences. Uses the given
    (test) forecasts only and never determines alignment constants. Identical
    log forecasts (Σ d_t² = 0) leave nothing to attribute → ValueError.
    """
    a = as_named_1d("forecast_a", np.asarray(forecast_a, dtype=float))
    b = as_named_1d("forecast_b", np.asarray(forecast_b, dtype=float))
    require_same_length(("forecast_a", a), ("forecast_b", b))
    if a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("forecasts must be non-empty and finite")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("forecasts must be positive (variance scale)")
    d = np.log(a) - np.log(b)
    ss = float(d @ d)
    if ss <= 0.0:
        raise ValueError("identical log forecasts — level share undefined")
    centered_ss = float(np.sum((d - float(d.mean())) ** 2))
    return float(1.0 - centered_ss / ss)


# ---------------------------------------------------------------------------
# 3. One-day Gaussian VaR breach rates (paper §4, §5.3)
# ---------------------------------------------------------------------------


def gaussian_var_threshold(forecast_var: Array, alpha: float = 0.05) -> Array:
    r"""One-day Gaussian VaR magnitude: ``VaR_t = −q_alpha · sqrt(f_t)``.

    Paper §4: ``VaR_t = −q_{0.05} sqrt(f_t)`` with ``q`` the standard-normal
    quantile — a positive loss threshold at the ``alpha`` tail. ``alpha`` must
    lie in (0, 0.5) so the threshold is positive; forecasts must be finite
    and positive.
    """
    f = as_named_1d("forecast_var", np.asarray(forecast_var, dtype=float))
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must be in (0, 0.5) — a lower-tail VaR level")
    if f.size == 0 or not np.isfinite(f).all() or np.any(f <= 0.0):
        raise ValueError("forecast_var must be non-empty, finite, and positive")
    return np.asarray(-float(norm.ppf(float(alpha))) * np.sqrt(f), dtype=float)


def var_breach_rates(returns: Array, forecast_var: Array, *, alpha: float = 0.05) -> dict[str, Any]:
    """One-day VaR breach rate with the Kupiec (1995) POF coverage test.

    A breach is ``r_t < −VaR_t`` (the loss ``−r_t`` exceeds the Gaussian
    threshold implied by the variance forecast). ``kupiec_lr`` / ``kupiec_p``
    reuse ``metrics.probability.kupiec_pof`` at the miss level ``alpha``;
    short windows (n < 10) yield honest NaN there, never a fabricated p.
    Research-diagnostic only — breach rates are proper calibration evidence,
    not live risk or P&L claims.
    """
    r = as_named_1d("returns", np.asarray(returns, dtype=float))
    f = as_named_1d("forecast_var", np.asarray(forecast_var, dtype=float))
    require_same_length(("returns", r), ("forecast_var", f))
    if r.size == 0 or not np.isfinite(r).all():
        raise ValueError("returns must be non-empty and finite")
    threshold = gaussian_var_threshold(f, alpha)
    hits = r < -threshold
    rate, lr, p = kupiec_pof(hits.astype(float), float(alpha))
    return {
        "alpha": float(alpha),
        "breach_rate": float(np.mean(hits)),
        "n_breaches": int(np.count_nonzero(hits)),
        "n": int(r.size),
        "kupiec_lr": float(lr),
        "kupiec_p": float(p),
    }


# ---------------------------------------------------------------------------
# 4. Seeded SYNTHETIC planted world (AGENTS.md honesty contract #2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossModelWorld:
    """One seeded SYNTHETIC loss-vs-model world — correctness evidence only.

    ``forecasts_raw[m, l, t] = c_l · base_m(t) · exp(κ ζ − κ²/2)`` where
    ``base_m(t) = (w_m σ²_t + (1 − w_m) σ̄²) · exp(η_{m,t} − s_m²/2)``:
    the training loss enters ONLY through the persistent level factor ``c_l``
    (Table-4-calibrated), the model ONLY through movement quality (tracking
    weight ``w_m``, daily log-noise ``s_m``), plus a small loss-specific
    movement perturbation ``κ`` so LevelShare is high but not degenerate.
    ``proxy`` is the noisy squared-return proxy (Patton 2011 setting — QLIKE
    ranking is robust to it). Never market evidence.
    """

    returns: Array
    proxy: Array
    sigma2_true: Array
    unconditional_var: float
    forecasts_raw: Array  # (n_models, n_losses, n)
    loss_names: tuple[str, ...]
    model_names: tuple[str, ...]
    level_factors: Array  # (n_losses,)
    tracking_weights: Array  # (n_models,)
    noise_scales: Array  # (n_models,)
    config: dict[str, Any]


def _check_positive_floats(values: Any, name: str, n: int) -> Array:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != n or not np.isfinite(arr).all() or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be {n} finite positive numbers")
    return arr


def _check_names(names: Any, name: str, n: int) -> tuple[str, ...]:
    items = tuple(names)
    if len(items) != n or any(not isinstance(x, str) or not x.strip() for x in items):
        raise ValueError(f"{name} must be {n} non-empty strings")
    if len(set(items)) != n:
        raise ValueError(f"{name} must be unique")
    return items


def simulate_loss_model_world(
    n: int,
    seed: int,
    *,
    level_factors: Any = DEFAULT_LEVEL_FACTORS,
    tracking_weights: Any = DEFAULT_TRACKING_WEIGHTS,
    noise_scales: Any = DEFAULT_NOISE_SCALES,
    loss_names: Any = DEFAULT_LOSS_NAMES,
    model_names: Any = DEFAULT_MODEL_NAMES,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    burn: int = 256,
    loss_movement_noise: float = DEFAULT_LOSS_MOVEMENT_NOISE,
) -> LossModelWorld:
    """Simulate the planted GARCH(1,1) world (seeded, deterministic, SYNTHETIC).

    GARCH(1,1): ``σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}``, ``ε_t = σ_t z_t``,
    ``z ~ N(0, 1)``, started at the unconditional variance ``σ̄² = ω/(1−α−β)``
    with ``burn`` discarded steps. Persistence outside (0, 1), a degenerate
    window, or malformed factor/name/weight vectors fail closed.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 100:
        raise ValueError("n must be an integer >= 100")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(burn, bool) or not isinstance(burn, (int, np.integer)) or int(burn) < 0:
        raise ValueError("burn must be a non-negative integer")
    n = int(n)
    burn = int(burn)
    persistence = float(alpha) + float(beta)
    if not (np.isfinite(omega) and omega > 0.0):
        raise ValueError("omega must be finite and positive")
    if not 0.0 < persistence < 1.0:
        raise ValueError("GARCH(1,1) persistence alpha + beta must lie in (0, 1)")
    if not np.isfinite(loss_movement_noise) or loss_movement_noise < 0.0:
        raise ValueError("loss_movement_noise must be finite and non-negative")

    loss_label_tuple = tuple(loss_names)
    factors = _check_positive_floats(level_factors, "level_factors", len(loss_label_tuple))
    loss_labels = _check_names(loss_label_tuple, "loss_names", factors.size)
    weights = np.asarray(tracking_weights, dtype=float).reshape(-1)
    scales = np.asarray(noise_scales, dtype=float).reshape(-1)
    if weights.size != scales.size or weights.size < 2:
        raise ValueError("tracking_weights and noise_scales must match and hold >= 2 models")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0) or np.any(weights > 1.0):
        raise ValueError("tracking_weights must lie in (0, 1]")
    if not np.isfinite(scales).all() or np.any(scales < 0.0):
        raise ValueError("noise_scales must be finite and non-negative")
    model_labels = _check_names(model_names, "model_names", weights.size)

    rng = np.random.default_rng(int(seed))
    uvar = float(omega) / (1.0 - persistence)
    total = n + burn
    z = rng.normal(0.0, 1.0, total)
    sigma2 = np.empty(total, dtype=float)
    eps = np.empty(total, dtype=float)
    sigma2[0] = uvar
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, total):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    sigma2_path = np.asarray(sigma2[burn:], dtype=float)
    returns = np.asarray(eps[burn:], dtype=float)
    if not np.isfinite(sigma2_path).all() or np.any(sigma2_path <= 0.0):
        raise ValueError("GARCH recursion produced a degenerate variance path")
    proxy = returns**2

    n_models, n_losses = weights.size, factors.size
    eta = rng.normal(0.0, 1.0, (n_models, n)) * scales[:, None] - (scales[:, None] ** 2) / 2.0
    base = (weights[:, None] * sigma2_path[None, :] + (1.0 - weights[:, None]) * uvar) * np.exp(eta)
    zeta = (
        rng.normal(0.0, 1.0, (n_models, n_losses, n)) * loss_movement_noise
        - loss_movement_noise**2 / 2.0
    )
    forecasts_raw = np.asarray(
        factors[None, :, None] * base[:, None, :] * np.exp(zeta), dtype=float
    )
    config: dict[str, Any] = {
        "data_label": "SYNTHETIC",
        "seed": int(seed),
        "n": n,
        "burn": burn,
        "process": "garch_1_1",
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(persistence),
        "unconditional_var": uvar,
        "level_factors": [float(x) for x in factors],
        "tracking_weights": [float(x) for x in weights],
        "noise_scales": [float(x) for x in scales],
        "loss_movement_noise": float(loss_movement_noise),
        "proxy": "squared_return",
        "note": "planted correctness world — never market evidence",
    }
    return LossModelWorld(
        returns=returns,
        proxy=proxy,
        sigma2_true=sigma2_path,
        unconditional_var=uvar,
        forecasts_raw=forecasts_raw,
        loss_names=loss_labels,
        model_names=model_labels,
        level_factors=factors,
        tracking_weights=weights,
        noise_scales=scales,
        config=config,
    )


# ---------------------------------------------------------------------------
# 5. Study driver: raw vs aligned scores, decomposition flip, VaR narrowing
# ---------------------------------------------------------------------------


def run_loss_model_decomposition_study(
    seed: int = 0,
    *,
    n: int = 2600,
    n_validation: int = 600,
    n_test: int = 600,
    alpha_var: float = 0.05,
    world_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full planted-world study: alignment, flip, VaR narrowing.

    Windows are contiguous and disjoint by construction — validation
    ``[n − n_test − n_validation, n − n_test)``, test ``[n − n_test, n)`` —
    and every alignment call re-checks disjointness (fail-closed). Score
    matrices are ``(n_losses, n_models)`` mean test losses: QLIKE (primary,
    paper §4), MSE and MSE-log (robustness). Returns the bench blob with the
    decomposition (raw vs aligned), LevelShare per model, breach-rate matrices
    before/after alignment, cross-loss breach spreads and their narrowing,
    and the fitted constants (which recover ``1/c_l`` up to validation noise).
    SYNTHETIC research diagnostic; ``live_pnl_claim`` False.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    for name, value in (("n_validation", n_validation), ("n_test", n_test)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 30:
            raise ValueError(f"{name} must be an integer >= 30")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if int(n) < int(n_validation) + int(n_test) + MIN_WINDOW:
        raise ValueError("n cannot host the validation + test windows")
    if not np.isfinite(alpha_var) or not 0.0 < float(alpha_var) < 0.5:
        raise ValueError("alpha_var must be in (0, 0.5)")
    n, n_validation, n_test = int(n), int(n_validation), int(n_test)

    world = simulate_loss_model_world(n, int(seed), **(world_kwargs or {}))
    test_idx = np.arange(n - n_test, n)
    val_start = n - n_test - n_validation
    validation_mask = np.zeros(n, dtype=bool)
    validation_mask[val_start : n - n_test] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    proxy_test = world.proxy[test_idx]
    returns_test = world.returns[test_idx]

    n_models = len(world.model_names)
    n_losses = len(world.loss_names)
    scores_raw: dict[str, Array] = {
        key: np.empty((n_losses, n_models), dtype=float) for key in ("qlike", "mse", "mse_log")
    }
    scores_aligned: dict[str, Array] = {
        key: np.empty((n_losses, n_models), dtype=float) for key in ("qlike", "mse", "mse_log")
    }
    breach_raw = np.empty((n_models, n_losses), dtype=float)
    breach_aligned = np.empty((n_models, n_losses), dtype=float)
    kupiec_p_raw = np.empty((n_models, n_losses), dtype=float)
    constants = np.empty((n_models, n_losses), dtype=float)
    level_share_by_model = np.empty(n_models, dtype=float)

    for m in range(n_models):
        pair_shares: list[float] = []
        for o in range(n_losses):
            f_full = world.forecasts_raw[m, o]
            fit = validation_level_alignment(
                f_full,
                world.proxy,
                validation_mask=validation_mask,
                test_mask=test_mask,
            )
            f_raw_test = f_full[test_idx]
            f_aligned_test = np.asarray(fit["aligned_test"], dtype=float)
            constants[m, o] = float(fit["constant"])
            for key in ("qlike", "mse", "mse_log"):
                if key == "qlike":
                    raw_cell = float(mean_qlike(proxy_test, f_raw_test))
                    aligned_cell = float(mean_qlike(proxy_test, f_aligned_test))
                elif key == "mse":
                    raw_cell = float(np.mean(mse_elementwise(proxy_test, f_raw_test)))
                    aligned_cell = float(np.mean(mse_elementwise(proxy_test, f_aligned_test)))
                else:
                    raw_cell = float(np.mean(mse_log_elementwise(proxy_test, f_raw_test)))
                    aligned_cell = float(np.mean(mse_log_elementwise(proxy_test, f_aligned_test)))
                scores_raw[key][o, m] = raw_cell
                scores_aligned[key][o, m] = aligned_cell
            breach_raw_res = var_breach_rates(returns_test, f_raw_test, alpha=alpha_var)
            breach_raw[m, o] = float(breach_raw_res["breach_rate"])
            kupiec_p_raw[m, o] = float(breach_raw_res["kupiec_p"])
            breach_aligned[m, o] = float(
                var_breach_rates(returns_test, f_aligned_test, alpha=alpha_var)["breach_rate"]
            )
        for o in range(n_losses):
            for o2 in range(o + 1, n_losses):
                pair_shares.append(
                    level_share(
                        world.forecasts_raw[m, o][test_idx], world.forecasts_raw[m, o2][test_idx]
                    )
                )
        level_share_by_model[m] = float(np.mean(pair_shares))

    spread_raw_by_model = (breach_raw.max(axis=1) - breach_raw.min(axis=1)).astype(float)
    spread_aligned_by_model = (breach_aligned.max(axis=1) - breach_aligned.min(axis=1)).astype(
        float
    )
    mean_spread_raw = float(np.mean(spread_raw_by_model))
    mean_spread_aligned = float(np.mean(spread_aligned_by_model))
    decomposition_qlike = loss_model_decomposition(scores_raw["qlike"], scores_aligned["qlike"])
    decomposition_mse_log = loss_model_decomposition(
        scores_raw["mse_log"], scores_aligned["mse_log"]
    )
    return {
        "schema": STUDY_SCHEMA,
        "kind": "vol_loss_decomposition",
        "claim": "research_only",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "citation": "Tokajuk & Chudziak (2026), arXiv:2609.27024, ADMA 2026",
        "seed": int(seed),
        "n": n,
        "n_validation": n_validation,
        "n_test": n_test,
        "alpha_var": float(alpha_var),
        "loss_names": list(world.loss_names),
        "model_names": list(world.model_names),
        "world_config": world.config,
        "alignment_constants": constants.tolist(),
        "scores_raw": {key: matrix.tolist() for key, matrix in scores_raw.items()},
        "scores_aligned": {key: matrix.tolist() for key, matrix in scores_aligned.items()},
        "decomposition_qlike": decomposition_qlike,
        "decomposition_mse_log": decomposition_mse_log,
        "flip_loss_to_model": bool(decomposition_qlike["flip_loss_to_model"]),
        "level_share_by_model": level_share_by_model.tolist(),
        "level_share_mean": float(np.mean(level_share_by_model)),
        "var_breach_raw": breach_raw.tolist(),
        "var_breach_aligned": breach_aligned.tolist(),
        "kupiec_p_raw": kupiec_p_raw.tolist(),
        "breach_rate_by_loss_raw": breach_raw.mean(axis=0).tolist(),
        "breach_rate_by_loss_aligned": breach_aligned.mean(axis=0).tolist(),
        "breach_spread_raw_by_model": spread_raw_by_model.tolist(),
        "breach_spread_aligned_by_model": spread_aligned_by_model.tolist(),
        "breach_spread_raw_mean": mean_spread_raw,
        "breach_spread_aligned_mean": mean_spread_aligned,
        "breach_spread_narrowing": float(
            1.0 - mean_spread_aligned / mean_spread_raw if mean_spread_raw > 0.0 else np.nan
        ),
    }


__all__ = [
    "ALIGNMENT_MODES",
    "DEFAULT_LEVEL_FACTORS",
    "DEFAULT_LOSS_NAMES",
    "DEFAULT_MODEL_NAMES",
    "DEFAULT_LOSS_MOVEMENT_NOISE",
    "DEFAULT_NOISE_SCALES",
    "DEFAULT_TRACKING_WEIGHTS",
    "MIN_WINDOW",
    "STUDY_SCHEMA",
    "LossModelWorld",
    "gaussian_var_threshold",
    "level_share",
    "loss_model_decomposition",
    "pairwise_marginal_gaps",
    "run_loss_model_decomposition_study",
    "simulate_loss_model_world",
    "two_way_score_shares",
    "validation_level_alignment",
    "var_breach_rates",
]
