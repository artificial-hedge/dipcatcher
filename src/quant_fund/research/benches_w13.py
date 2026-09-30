"""Benchmark batteries for SOTA canon wave 13.

Covers the wave-13 module lanes: conformal coverage inference under temporal
dependence, minimax-optimal conformal e-detectors, rolling conformal
prediction for sequential model training, anytime-valid rank confidence
sequences / BB-EDGE leaderboards, prediction-interval-conditional prediction
intervals (PICPIs), replicable conformal calibration, reference-null
calibrated e-process thresholds, and delayed-feedback adaptive conformal
inference.

Seeded SYNTHETIC streams only — no panel or vendor data, no headline
performance ratios (proper scores / coverage / interval and sequential-testing
diagnostics only, per the AGENTS.md honesty contract). Each bench returns a
flat ``dict[str, float]`` of proper diagnostic statistics, or ``{}`` if its
synthetic setup cannot be constructed. Every bench is deterministic: repeated
calls are bit-identical. Monte-Carlo budgets are SHRUNK relative to the lane
test suites (documented per bench) so the whole battery stays inside its
runtime envelope; the accompanying research tests carry correspondingly
wider, documented tolerances.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm

from quant_fund.metrics.conformal_coverage_inference import (
    coverage_ztest,
    sigma_cov_squared,
    split_conformal_cutoff,
)
from quant_fund.metrics.conformal_e_detectors import pfa_delay_study
from quant_fund.metrics.conformal_martingale import conformal_p_values
from quant_fund.metrics.picpi import bench_picpi as _picpi_core_bench
from quant_fund.metrics.picpi import bench_picpi_width_rate as _picpi_width_rate_bench
from quant_fund.metrics.rank_confidence_sequences import (
    bb_edge_certify,
    bb_edge_topk_certify,
    check_edge_fwer,
    check_rank_time_uniform_coverage,
    rank_confidence_sequence,
)
from quant_fund.metrics.reference_null_calibration import (
    calibrate_conformal_martingale,
    crossing_time,
    detection_delay_summary,
    kt_histogram_martingale,
    restart_mixture_kt_martingale,
)
from quant_fund.models.conformal import AdaptiveConformal
from quant_fund.models.delayed_aci import (
    DelayedACI,
    delay_to_memory_ratio,
    memory_length_ar1,
    run_delayed_conformal,
)
from quant_fund.models.replicable_conformal import (
    agreement_experiment,
    selective_recalibration_experiment,
    validity_cost_experiment,
)
from quant_fund.models.rolling_conformal import (
    AlternatingFoldScorer,
    LastPointOverfitter,
    OnlineRidge,
    RollingConformal,
    SplitConformal,
    marginal_coverage_floor,
)

_SEED = 20260930


def _ar1(rng: np.random.Generator, size: int, phi: float, *, burn: int = 500) -> np.ndarray:
    """Stationary Gaussian AR(1) via lfilter; burn-in discarded (lane convention)."""
    eps = rng.standard_normal(size + burn)
    return np.asarray(lfilter([1.0], [1.0, -phi], eps)[burn:], dtype=float)


def _ar1_drift_stream(
    n: int, phi: float, seed: int, *, sigma_d: float = 1.0, sigma_z: float = 0.5
) -> np.ndarray:
    """AR(1)-drift residual stream eps_t = d_t + noise (the Sec. 6 sweep fixture)."""
    rng = np.random.default_rng(seed)
    sigma_eta = sigma_d * np.sqrt(1.0 - phi * phi)
    drift = np.asarray(
        lfilter([1.0], [1.0, -phi], rng.normal(0.0, sigma_eta, n)),
        dtype=float,
    )
    return np.asarray(drift + rng.normal(0.0, sigma_z, n), dtype=float)


def _copula_binary_panels(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    theta: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """SYNTHETIC binary item scores with a common item-difficulty factor.

    Z = sqrt(kappa) * U_item + sqrt(1 - kappa) * eps; X = 1{Z <= Phi^-1(theta_j)}
    (the lane suite's Gaussian-copula fixture): E[X_tj] = theta_j exactly and
    kappa > 0 makes within-item scores dependent across models arbitrarily.
    """
    m = int(np.asarray(theta).size)
    u_item = rng.standard_normal(shape + (1,))
    eps = rng.standard_normal(shape + (m,))
    z = np.sqrt(kappa) * u_item + np.sqrt(1.0 - kappa) * eps
    thr = norm.ppf(np.asarray(theta, dtype=float))
    return np.asarray((z <= thr).astype(float), dtype=float)


def bench_coverage_inference() -> dict[str, float]:
    """Realized-coverage inference under temporal dependence, SYNTHETIC (wave 13).

    Zhai, Cheng & Wu (2026), "Conformal Coverage of Time Series: Validity and
    Inference", arXiv:2609.33868 (CLT for realized split-conformal coverage
    via Wu (2005a) functional dependence; block SE of their Eq. (10) /
    Prop. 4; fixed-B scaled-t reference of their Eq. (24), Jones, Haran,
    Caffo & Neath 2006). One seeded AR(1) stream at phi = 0.8 against an
    i.i.d. contrast exhibits the long-run-variance inflation of the coverage
    indicators; the z-test size/power and SE-calibration cells are SHRUNK to
    30 seeded replicates (lane suite: 100-200), so the accompanying test
    tolerances are correspondingly wider. Seeded SYNTHETIC; inferential
    coverage diagnostics only, never market evidence.
    """
    try:
        alpha_cov = 0.10
        nominal = 1.0 - alpha_cov
        eta = 0.10
        n_reps = 30
        rng = np.random.default_rng(_SEED)

        # (a) block-SE^2 ratio: one AR(1) phi=0.8 stream vs one i.i.d. stream,
        # calibration indicators at each stream's own conformal cutoff.
        n_se = 1500
        v_ar = np.abs(_ar1(rng, n_se, 0.8))
        v_iid = np.abs(rng.standard_normal(n_se))
        q_ar = split_conformal_cutoff(v_ar, alpha_cov)
        q_iid = split_conformal_cutoff(v_iid, alpha_cov)
        s2_ar = sigma_cov_squared((v_ar <= q_ar).astype(float)).sigma2
        s2_iid = sigma_cov_squared((v_iid <= q_iid).astype(float)).sigma2

        # (b/c/d) z-test size (scaled-t reference, AR(1) null), power under a
        # miscalibrated predictor (test scores inflated x1.5), and the i.i.d.
        # SE-vs-binomial calibration check.
        n_z, m_z = 1500, 1500
        size_hits, power_hits = 0, 0
        ses: list[float] = []
        binom_se = float(np.sqrt(alpha_cov * nominal * (1.0 / n_z + 1.0 / m_z)))
        for _ in range(n_reps):
            v = np.abs(_ar1(rng, n_z + m_z, 0.8))
            r_null = coverage_ztest(
                v[:n_z], v[n_z:], alpha_coverage=alpha_cov, eta=eta, reference="student_t"
            )
            size_hits += int(r_null.p_value < eta)
            r_alt = coverage_ztest(
                v[:n_z], v[n_z:] * 1.5, alpha_coverage=alpha_cov, eta=eta, reference="student_t"
            )
            power_hits += int(r_alt.p_value < eta)
            v_i = np.abs(rng.standard_normal(n_z + m_z))
            r_iid = coverage_ztest(v_i[:n_z], v_i[n_z:], alpha_coverage=alpha_cov, eta=eta)
            ses.append(r_iid.se)
        return {
            "covinf_alpha_coverage": alpha_cov,
            "covinf_nominal": nominal,
            "covinf_eta": eta,
            "covinf_n_reps": float(n_reps),
            "covinf_ar1_se_ratio_vs_iid": float(s2_ar / s2_iid),
            "covinf_ztest_size": size_hits / float(n_reps),
            "covinf_ztest_power": power_hits / float(n_reps),
            "covinf_iid_se_binomial_ratio": float(np.mean(ses)) / binom_se,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_conformal_e_detectors() -> dict[str, float]:
    """Optimal restarted conformal e-detectors vs a Vovk CTM, SYNTHETIC (wave 13).

    Bhattacharyya & Ramdas (2026), "Change detection with conformal
    martingales: new optimal constructions, and suboptimality of existing
    methods", arXiv:2609.27179 (restarted mixture conformal e-processes, their
    Def. 2.6 / Alg. 1 / Thm. 2.7; polynomial restart weights, Cor. 3.6) with
    the Vovk (2021) conformal test martingale as the suboptimal comparator
    (their Example 2.2: Omega(T) delay under PFA control). A SHRUNK single-
    changepoint cell of the module's matched-PFA harness (T = 300, 16 seeded
    replicates, two methods — lane suite: T = 400 across the full method
    grid): at the SAME Ville threshold 1/alpha the restarted e-process must
    detect far earlier, with its pre-change false-alarm rate under control.
    Seeded SYNTHETIC; sequential-testing diagnostics only, never market
    evidence.
    """
    try:
        alpha = 0.05
        changepoint = 300
        n_reps = 16
        rows = {
            r.method: r
            for r in pfa_delay_study(
                changepoint,
                n_reps=n_reps,
                alpha=alpha,
                shift=1.2,
                seed=_SEED,
                methods=("vovk_ctm", "optimal_sum"),
            )
        }
        opt = rows["optimal_sum"]
        ctm = rows["vovk_ctm"]
        return {
            "ced_alpha": alpha,
            "ced_changepoint": float(changepoint),
            "ced_n_reps": float(n_reps),
            "ced_delay_optimal": float(opt.median_capped_delay),
            "ced_delay_vovk": float(ctm.median_capped_delay),
            "ced_delay_ratio": float(ctm.median_capped_delay / opt.median_capped_delay),
            "ced_fa_rate": float(opt.false_alarm_rate),
            "ced_detection_rate_optimal": float(opt.detection_rate),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_rolling_conformal() -> dict[str, float]:
    """Rolling conformal prediction on SYNTHETIC sequential-training streams (wave 13).

    Cheng, Liang & Barber (2026), "Rolling Conformal Prediction in Sequential
    Model Training", arXiv:2609.26951 (calibrate-then-roll, their Sec. 2;
    Theorem 1's universal 1 - 2*alpha marginal floor; Proposition 1 /
    App. A.4's alternating-fold worst case attaining it; Sec. 4.1 frozen
    split-CP width contrast; the naive arrival-time split has NO floor).
    SHRUNK Monte-Carlo cells vs the lane suite (M = 8-40 seeded streams
    instead of 40-150, n = 500-1200 instead of 500-2000): a stable online-
    ridge regime near nominal coverage, the adversarial fold at the factor-
    two floor, the rolling-vs-frozen-split width ratio, and the naive split's
    undercoverage under a deliberately unstable predictor. Seeded SYNTHETIC;
    coverage/width diagnostics only, never market evidence.
    """
    try:
        alpha = 0.10
        nominal = 1.0 - alpha

        # (a) STABLE regime (Thm. 4 territory): sequential ridge, i.i.d. linear
        # stream — coverage approaches the nominal 1 - alpha, not the floor.
        covs = []
        for k in range(24):
            rng = np.random.default_rng(_SEED + k)
            X = rng.normal(size=(500, 4))
            y = X[:, 0] + rng.normal(0.0, 0.5, 500)
            rng_h = np.random.default_rng(_SEED + 500_000 + k)
            Xt = rng_h.normal(size=(40, 4))
            yt = Xt[:, 0] + rng_h.normal(0.0, 0.5, 40)
            rc = RollingConformal(lambda: OnlineRidge(4, lam=0.5, score="sq"), alpha=alpha)
            covs.append(rc.run_stream(X, y, Xt, yt).coverage)
        marginal = float(np.mean(covs))

        # (b) WORST case (Prop. 1, App. A.4): alternating fold scores on
        # Unif(0,1) labels — coverage sits at 1 - 2*alpha + nu, and the
        # universal floor still HOLDS (that is what buys the guarantee).
        n_fold = 800
        fold_covs = []
        for k in range(30):
            rng = np.random.default_rng(_SEED + 1_000_000 + k)
            y = rng.uniform(0.0, 1.0, n_fold)
            rng_h = np.random.default_rng(_SEED + 1_500_000 + k)
            yh = rng_h.uniform(0.0, 1.0, 50)
            rc = RollingConformal(lambda: AlternatingFoldScorer(0.85, 1.0), alpha=alpha)
            rc.fit_stream(np.zeros((n_fold, 1)), y)
            fold_covs.append(float(np.mean(rc.evaluate(np.zeros((50, 1)), yh)[1])))
        fold_cov = float(np.mean(fold_covs))
        floor = marginal_coverage_floor(alpha, n_fold)
        floor_holds = 1.0 if fold_cov >= floor - 0.02 else 0.0

        # (c) Width contrast (Sec. 4.1, Fig. 2 setting 1): rolling-CP exploits
        # models trained on ALL n points; the frozen split model (trained on
        # the first m = 60 of n = 1200, d = 24) stays wider at similar coverage.
        rwidth, swidth = [], []
        for k in range(8):
            rng = np.random.default_rng(_SEED + 2_000_000 + k)
            X = rng.normal(size=(1200, 24))
            y = X[:, 0] + rng.normal(0.0, 0.2, 1200)
            rng_h = np.random.default_rng(_SEED + 2_500_000 + k)
            Xt = rng_h.normal(size=(25, 24))
            yt = Xt[:, 0] + rng_h.normal(0.0, 0.2, 25)

            def factory() -> OnlineRidge:
                return OnlineRidge(24, lam=0.1, score="abs")

            r = RollingConformal(factory, alpha=alpha).run_stream(X, y, Xt, yt)
            s = SplitConformal(factory, alpha=alpha, n_train=60).run_stream(X, y, Xt, yt)
            rwidth.append(r.mean_width)
            swidth.append(s.mean_width)
        width_ratio = float(np.mean(rwidth) / np.mean(swidth))

        # (d) NAIVE arrival-time split (no fixed-score premise => no floor)
        # under a deliberately unstable predictor: coverage breaks BELOW
        # 1 - 2*alpha while rolling-CP's same-state comparisons stay valid.
        naive_covs = []
        for k in range(40):
            rng = np.random.default_rng(_SEED + 3_000_000 + k)
            y = rng.normal(size=600)
            rng_h = np.random.default_rng(_SEED + 3_500_000 + k)
            yh = rng_h.normal(size=30)
            rc = RollingConformal(lambda: LastPointOverfitter(40.0, "linear"), alpha=alpha)
            rc.fit_stream(np.zeros((600, 1)), y)
            naive_covs.append(float(np.mean(rc.naive_evaluate(np.zeros((30, 1)), yh)[1])))
        return {
            "rcp_alpha": alpha,
            "rcp_nominal": nominal,
            "rcp_marginal_coverage": marginal,
            "rcp_worst_case_coverage": fold_cov,
            "rcp_worst_case_floor": float(floor),
            "rcp_worst_case_floor_holds": floor_holds,
            "rcp_vs_split_width_ratio": width_ratio,
            "rcp_naive_split_undercoverage": float(np.mean(naive_covs)),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_rank_cs() -> dict[str, float]:
    """Anytime-valid rank inference on SYNTHETIC leaderboard panels (wave 13).

    Khosravi & Huo (2026), "Rank Confidence Sequences: Anytime-Valid
    Leaderboards", arXiv:2609.32211 (pairwise betting wealths + closed
    testing over weak orders, their Alg. 1; time-uniform rank coverage,
    Thms. 4.2-4.3) and Gao, Zhang, Xie, Jing, Wang & Liu (2026), "Anytime-
    Valid LLM Leaderboards via Benchmark-weighted and Block-Factorized
    e-Processes" (BB-EDGE), arXiv:2609.32248 (direct e-Holm anytime FWER,
    their Eq. (8) / Thm. 1, via Hartog & Lei 2025; pilot Top-k certification,
    their App. A; Waudby-Smith & Ramdas 2024 empirical-Bernstein betting).
    M = 4 models scored on shared items with a common item-difficulty factor
    (arbitrary within-item dependence); SHRUNK to 20 seeded replicates per
    cell (lane suite: 40-100). Seeded SYNTHETIC; error-control diagnostics
    only, never market evidence.
    """
    try:
        alpha = 0.05
        rng = np.random.default_rng(_SEED)

        # (a) Time-uniform rank-interval coverage under within-item dependence.
        theta = np.array([0.62, 0.55, 0.48, 0.41])
        true_ranks = np.asarray(1 + (theta[:, None] < theta[None, :]).sum(axis=1), dtype=np.int64)
        n_reps, t_len = 20, 80
        panels = _copula_binary_panels(rng, (n_reps, t_len), theta, 0.5)
        los, ups = [], []
        for r in range(n_reps):
            res = rank_confidence_sequence(panels[r], alpha=alpha, method="exact")
            los.append(res.rank_lower)
            ups.append(res.rank_upper)
        cov = check_rank_time_uniform_coverage(
            np.asarray(los), np.asarray(ups), true_ranks.astype(np.float64)
        )
        rank_coverage = cast("float", cov["time_uniform_coverage"])

        # (b) Dominant model: rank-1 recovery at T and the median time the
        # full interval profile stabilizes (efficiency of the closed testing).
        theta_dom = np.array([0.9, 0.5, 0.45, 0.4])
        n_dom, t_dom = 20, 120
        dom_panels = _copula_binary_panels(rng, (n_dom, t_dom), theta_dom, 0.3)
        stabs: list[int] = []
        rank1_hits = 0
        for r in range(n_dom):
            res = rank_confidence_sequence(dom_panels[r], alpha=alpha, method="exact")
            lo, up = res.rank_lower, res.rank_upper
            same = ((lo == lo[-1][None, :]) & (up == up[-1][None, :])).all(axis=1)
            stabs.append(int(np.argmax(same)) if bool(same.any()) else t_dom)
            sets = res.rank_sets
            if sets is not None and bool(sets[-1, 0, 0]) and not bool(sets[-1, 0, 1:].any()):
                rank1_hits += 1
        median_stab_frac = float(np.median(np.asarray(stabs, dtype=float))) / float(t_dom)

        # (c) BB-EDGE anytime FWER under a global null with a dependent
        # item factor: every direction is a true null, so ANY certified edge
        # is a false edge (their Thm. 1 bad event, probability <= alpha).
        theta_null = np.full(4, 0.5)
        delta_null = theta_null[:, None] - theta_null[None, :]
        n_null, r_null, i_null, fwer_hits = 20, 25, 40, 0
        for _ in range(n_null):
            null_panels = _copula_binary_panels(rng, (r_null, i_null), theta_null, 0.35)
            res_null = bb_edge_certify(null_panels, tau=0.0, alpha=alpha)
            fwer_hits += int(
                bool(
                    check_edge_fwer(res_null.edges_closed.astype(np.float64), delta_null, tau=0.0)[
                        "ever_false"
                    ]
                )
            )

        # (d) BB-EDGE pilot Top-1 certification with a dominant model.
        theta_top = np.array([0.8, 0.45, 0.4])
        n_top, r_top, i_top, cert_hits = 20, 40, 40, 0
        for _ in range(n_top):
            confirm = _copula_binary_panels(rng, (r_top, i_top), theta_top, 0.35)
            pilot = _copula_binary_panels(rng, (3, i_top), theta_top, 0.35)
            tk = bb_edge_topk_certify(pilot, confirm, 1, tau=0.0, alpha=alpha)
            cert_hits += int(tk.certified)
        return {
            "rcs_alpha": alpha,
            "rcs_n_models": 4.0,
            "rcs_n_reps": float(n_reps),
            "rcs_time_uniform_rank_coverage": rank_coverage,
            "rcs_dominant_rank1_recovery": rank1_hits / float(n_dom),
            "rcs_median_stabilization_time": median_stab_frac,
            "bbedge_fwer_dependent": fwer_hits / float(n_null),
            "bbedge_topk_certification_rate": cert_hits / float(n_top),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_picpi() -> dict[str, float]:
    """Prediction-interval-conditional prediction intervals, SYNTHETIC (wave 13).

    Yang, Huang, Hou, Imbens & Jordan (2026), "PICPIs: Prediction-Interval-
    Conditional Prediction Intervals", arXiv:2609.25388 (self-consistency
    Def. 2.1; Algorithm 1's concentration-margin admission with Thm. 2.2's
    finite-sample guarantee; the margin-free naive-bin contrast that has NO
    validity; Thm. 3.2's n^(-1/3) width rate). Thin float-only adapter over
    the module's own ``bench_picpi`` / ``bench_picpi_width_rate`` (which
    return mixed float|str blobs): the numeric calibration-quality
    diagnostics are re-exposed under ``picpi_*`` keys, following the wave-12
    ``rwcv`` adapter precedent. The width-rate cell uses a SHRUNK documented
    n-grid subset (10k / 20k / 40k instead of the module default's
    10k-640k sweep) to stay inside the battery runtime budget; the adversarial
    spike cell reuses the lane suite's pinned seed 5. Seeded SYNTHETIC;
    calibration-quality diagnostics only, never market evidence.
    """
    try:
        calibrated = _picpi_core_bench(
            n_cal=20_000, n_test=20_000, num_bins=20, delta=0.1, min_count=200, seed=_SEED
        )
        adversarial = _picpi_core_bench(
            n_cal=20_000,
            n_test=20_000,
            num_bins=20,
            delta=0.1,
            min_count=200,
            spike_weight=0.015,
            seed=5,
        )
        width = _picpi_width_rate_bench(
            (10_000, 20_000, 40_000), num_bins=100, delta=0.1, n_test=2_000, seed=11
        )
        mapped = {
            "picpi_selfconsistency_violation_rate": float(
                calibrated["selfconsistency_violation_rate"]
            ),
            "picpi_selfconsistency_n_checked": float(calibrated["selfconsistency_n_checked"]),
            "picpi_naive_violation_rate": float(adversarial["naive_violation_rate"]),
            "picpi_adversarial_violation_rate": float(
                adversarial["selfconsistency_violation_rate"]
            ),
            "picpi_certified_fraction": float(calibrated["certified_fraction"]),
            "picpi_width_slope": float(width["width_slope"]),
            "picpi_width_first": float(width["width_first"]),
            "picpi_width_last": float(width["width_last"]),
            "picpi_num_bins": float(calibrated["num_bins"]),
            "picpi_delta": float(calibrated["delta"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_replicable_conformal() -> dict[str, float]:
    """Replicable conformal calibration (ReCal) on a SYNTHETIC PIT-uniform fixture.

    Papamichalis, Ruane & Papamichalis (2026), "Replicable Conformal
    Prediction", arXiv:2608.23638 (shared seed + upward grid rounding, their
    Alg. 1; Prop. 2's 1 - rho analyst agreement; Thm. 2(ii)'s unconditional
    marginal coverage; Cor. 4's set-size price of replicability; Cor. 3's
    min-of-M selection attack under which standard split CP silently
    undercovers while ReCal stays valid). Float-only adapter calling the
    module's harnesses directly with SHRUNK budgets (n = 12k, 40 analyst
    pairs / 40 trials / M = 10 redraws vs the module bench's n = 8k with
    100 / 100 / 10) — the module's own ``bench_replicable_conformal`` returns
    a mixed float|str blob. DEVIATION NOTE: ``repcon_gaming_undercover_standard``
    carries the selected COVERAGE (which must fall BELOW nominal — the
    silent-undercoverage claim), where the module bench maps the same key name
    to the coverage GAP. Seeded SYNTHETIC fixture (scores iid Unif[0,1]);
    agreement / coverage / set-size diagnostics only, never market evidence.
    """
    try:
        alpha, rho, n, m_draws = 0.10, 0.10, 12_000, 10
        n_pairs, n_trials = 40, 40
        nominal = 1.0 - alpha
        agree = agreement_experiment(n, alpha, rho, n_pairs, _SEED, kappa_hat=1.0)
        valid = validity_cost_experiment(n, alpha, rho, n_trials, _SEED + 1, kappa_hat=1.0)
        gaming = selective_recalibration_experiment(
            n, alpha, rho, m_draws, n_trials, _SEED + 2, kappa_hat=1.0
        )
        mapped = {
            "repcon_alpha": alpha,
            "repcon_rho": rho,
            "repcon_nominal": nominal,
            "repcon_agreement_rate": float(agree.identity_rate),
            "repcon_coverage": float(valid.mean_coverage_recal),
            "repcon_coverage_standard": float(valid.mean_coverage_standard),
            "repcon_size_cost_ratio": float(valid.size_cost_ratio),
            "repcon_gaming_undercover_standard": float(gaming.standard_selected_coverage),
            "repcon_gaming_selected_recal": float(gaming.recal_selected_coverage),
            "repcon_gaming_stability": float(gaming.recal_stability),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_reference_null() -> dict[str, float]:
    """Reference-null calibrated e-process thresholds, SYNTHETIC (wave 13).

    Ding, Wei, Zhu & Dai (2026), "Reference-Null Calibrated Thresholds for
    E-Processes with Applications to Conformal Martingales",
    arXiv:2609.32678 (Thm. 2.2's finite-sample rank calibration; Alg. 2's KT
    histogram betting martingale; Alg. 3's restart mixture; Sec. 5.1's paired
    calibrated-vs-Ville boundary design on conformal p-values, which are
    i.i.d. Unif(0,1) under every exchangeable score law — their Sec. 3.1
    pivot). SHRUNK bank study vs the lane suite: 2 banks x B = 199 reference
    nulls (vs 4-8 banks x 299-499), 160 evaluation paths at T = 250, and 80
    planted-shift delay replicates (vs 200) — bank-averaged type-I rates carry
    correspondingly wider MC slack in the tests. Seeded SYNTHETIC; type-I /
    detection-delay diagnostics only, never market evidence.
    """
    try:
        alpha = 0.05
        ville = 1.0 / alpha
        n_bins = 20
        horizon = 250
        n_eval = 160
        n_banks, bank_size = 2, 199

        # (a) Type-I: exchangeable null streams (iid N(0,1) scores -> conformal
        # p-values -> KT martingale paths), paired C-vs-V boundaries on the
        # SAME paths. Calibrated crossing is strict; Ville is non-strict.
        paths = []
        for rep in range(n_eval):
            rng = np.random.default_rng(_SEED + 100_000 + rep)
            scores = rng.standard_normal(horizon)
            p = conformal_p_values(scores, seed=_SEED + 200_000 + rep)
            paths.append(kt_histogram_martingale(p, n_bins))
        arr = np.stack(paths)
        fa_ville = float(np.mean([crossing_time(r, ville, strict=False) is not None for r in arr]))
        fa_banks: list[float] = []
        sharps: list[float] = []
        for b in range(n_banks):
            cal = calibrate_conformal_martingale(
                alpha,
                horizon=horizon,
                construction="kt",
                n_bins=n_bins,
                n_reference=bank_size,
                seed=_SEED + 900_000 + 1000 * b,
            )
            fa_banks.append(
                float(
                    np.mean([crossing_time(r, cal.threshold, strict=True) is not None for r in arr])
                )
            )
            sharps.append(cal.sharpness)

        # (b) Paired delay gain under a planted mean shift (restart-mixture
        # construction, their Sec. 5.1 change-point model): the SAME paths are
        # thresholded at the calibrated boundary (strict) and at Ville
        # (non-strict); RMDD uses their censoring convention.
        tau_shift, mu_shift, n_reps_delay = horizon // 2, 1.0, 80
        cal_d = calibrate_conformal_martingale(
            alpha,
            horizon=horizon,
            construction="restart_mixture",
            n_bins=n_bins,
            n_reference=bank_size,
            seed=_SEED + 950_000,
        )
        shift_paths = []
        for rep in range(n_reps_delay):
            rng = np.random.default_rng(_SEED + 300_000 + rep)
            scores = np.concatenate(
                [
                    rng.standard_normal(tau_shift),
                    rng.standard_normal(horizon - tau_shift) + mu_shift,
                ]
            )
            p = conformal_p_values(scores, seed=_SEED + 400_000 + rep)
            shift_paths.append(restart_mixture_kt_martingale(p, n_bins))
        shift_arr = np.stack(shift_paths)
        sum_c = detection_delay_summary(
            shift_arr, threshold=cal_d.threshold, change_time=tau_shift, strict=True
        )
        sum_v = detection_delay_summary(
            shift_arr, threshold=ville, change_time=tau_shift, strict=False
        )
        rmdd_c = cast("float", sum_c["rmdd"])
        rmdd_v = cast("float", sum_v["rmdd"])
        return {
            "rnc_alpha": alpha,
            "rnc_horizon": float(horizon),
            "rnc_n_banks": float(n_banks),
            "rnc_bank_size": float(bank_size),
            "rnc_n_eval": float(n_eval),
            "rnc_fa_rate_calibrated": float(np.mean(fa_banks)),
            "rnc_fa_rate_ville": fa_ville,
            "rnc_delay_gain_ratio": float(rmdd_c / rmdd_v),
            "rnc_threshold_sharpness": float(np.median(sharps)),
            "rnc_delay_threshold_ratio": float(cal_d.threshold / ville),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_delayed_aci() -> dict[str, float]:
    """tau-delayed adaptive conformal inference on a SYNTHETIC AR(1)-drift cell.

    El Halabi & Brandt (2026), "Adaptive Conformal Inference Under Delayed
    Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic",
    arXiv:2609.07251 (Algs. 1+2: sliding-window split bands with tau-delayed
    level updates; Eq. (11)'s finite-sample long-run coverage bound with
    explicit tau slack; tau = 1 reproduces Gibbs & Candes (2021) ACI exactly;
    Eq. (15)'s interval score — a proper score for interval forecasts; the
    r = tau/L delay-to-memory ratio with L = -1/log phi). ONE seeded cell of
    the lane suite's AR(1)-drift sweep (phi = 0.9, tau = 6 vs tau = 1 at
    T = 1500, R = 50 seeds), not the full (phi, tau) grid: the interval score
    must ORDER by the delay-to-memory ratio, the Eq. (11) bound must hold on
    every run, and the tau = 1 controller must match the repo's
    AdaptiveConformal bit-for-bit. Seeded SYNTHETIC; coverage and proper-
    score diagnostics only, never market evidence.
    """
    try:
        alpha, gamma, window = 0.10, 0.02, 200
        n_stream, phi = 1500, 0.9
        n_seeds = 50
        memory = memory_length_ar1(phi)

        # tau = 1 exactness pin: same closed-form update, same clip box, same
        # float ops as AdaptiveConformal — exact equality of the level path.
        rng = np.random.default_rng(_SEED)
        errs = np.asarray(rng.random(500) < 0.12, dtype=float)
        aci = AdaptiveConformal(alpha=alpha, gamma=0.05)
        ref = [aci.alpha_t]
        for e in errs:
            aci.update(float(e))
            ref.append(aci.alpha_t)
        daci = DelayedACI(alpha=alpha, gamma=0.05, tau=1)
        issued = [daci.step(None)]
        for e in errs:
            issued.append(daci.step(float(e)))
        tau1_flag = 1.0 if issued == ref else 0.0

        cov: dict[int, float] = {}
        isc: dict[int, float] = {}
        bound_ok = True
        for tau in (1, 6):
            covs, iscs = [], []
            for s in range(n_seeds):
                eps = _ar1_drift_stream(n_stream + tau, phi, _SEED + 1_000 * s + tau)
                res = run_delayed_conformal(eps, tau, alpha=alpha, gamma=gamma, window=window)
                covs.append(res.empirical_coverage)
                iscs.append(res.mean_interval_score)
                bound_ok = bound_ok and res.long_run_deviation <= res.coverage_bound + 1e-12
            cov[tau] = float(np.mean(covs))
            isc[tau] = float(np.mean(iscs))
        return {
            "daci_alpha": alpha,
            "daci_phi": phi,
            "daci_memory_length": float(memory),
            "daci_r_low": float(delay_to_memory_ratio(1, memory)),
            "daci_r_high": float(delay_to_memory_ratio(6, memory)),
            "daci_n_seeds": float(n_seeds),
            "daci_tau1_matches_aci": tau1_flag,
            "daci_bound_holds": 1.0 if bound_ok else 0.0,
            "daci_coverage_lowr": cov[1],
            "daci_coverage_highr": cov[6],
            "daci_is_lowr": isc[1],
            "daci_is_highr": isc[6],
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}
