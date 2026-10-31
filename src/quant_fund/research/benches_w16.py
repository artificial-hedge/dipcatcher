"""Benchmark batteries for SOTA canon wave 16 (conformal / selection / RL-MM).

Covers the wave-16 module lanes: the loss-choice-vs-model-choice decomposition
for volatility forecasts (validation level alignment and the loss-dominated to
model-dominated flip), generalized hierarchical conformal prediction (GHCP),
multi-source randomly localized conformal prediction (MS-RLCP), conformal
prediction under an exponential-tilt joint shift (ExTRA-WCP / -WCP-T),
target-alignment dilution accounting with cautious forecast selection, and the
(torch-gated) C51 distributional-RL market maker on the zero-intelligence LOB.

Honesty (AGENTS.md contract): seeded SYNTHETIC streams only — no panel or
vendor data, no headline performance ratios (proper scores / coverage / interval
width / calibration and correctness diagnostics only). Every bench returns a
flat ``dict[str, float]`` (float-only; the ``ghcp`` / ``ms_rlcp`` /
``extra_tilt`` adapters filter their modules' str stamps — the wave-12 ``rwcv``
and wave-14 ``xva`` precedent), or ``{}`` if its synthetic setup cannot be
constructed (the torch-gated ``rl_market_maker`` bench returns ``{}`` when the
optional ``nn`` extra is absent — the wave-12 ``deep_hedging`` precedent).
Every bench is deterministic: repeated calls are bit-identical (all randomness
lives in seeded generators; the RL-MM CPU trainer is single-threaded and seeded).
Monte-Carlo / training budgets are SHRUNK relative to the lane suites
(documented per bench) so the whole wave-16 battery stays inside its ~60 s
runtime envelope (measured ~7 s); the accompanying research tests carry
correspondingly wider, documented tolerances.

Documented deviations (wave-16 brief):
- ``forecast_selection`` maps ``fs_deviation_mirror_affine_r2`` to the module's
  PAIR-LEVEL ``pair_deviation_mirror_affine_r2`` (the paper's Section 9.2 mirror
  statistic, headline R^2 = 0.9999), NOT the date-level
  ``deviation_mirror_affine_r2`` (which folds in within-date sampling noise and
  lands ~0.84). Key-semantics note following the wave-13 ``repcon`` precedent;
  the lane suite asserts the pair-level R^2 > 0.99.
- ``rl_market_maker`` is TORCH-GATED (wave-12 ``deep_hedging`` precedent): ``{}``
  without the optional ``nn`` extra. It runs a TINY-BUDGET path (11-atom /
  hidden-16 / buffer-2000 C51, 3 training episodes under regime flow, 2 paired
  evaluation seeds, 600 s horizon) — NOT the paper-scale ``@slow``
  ``rl_mm_benchmark`` (101 atoms / hidden 256 / buffer 1e5 / hours).
  ``rlmm_inventory_cap_violations`` counts evaluation sessions where the C51
  agent's max-abs-inventory strictly EXCEEDS the hard cap ``q_max``; the
  one-sided inventory gating clamps AT the cap, so this is a SAFETY invariant
  (== 0), distinct from SATURATION (reaching the cap) which is the Avellaneda-
  Stoikov regime failure mode exposed by ``rlmm_regime_saturation_as``. The
  ``sim_internal_*`` PnL keys stay OUT of the blob (honesty contract: no
  headline P&L); only inventory / completion / config keys are emitted.
"""

from __future__ import annotations

import numpy as np

from quant_fund.metrics.forecast_selection import forecast_selection_benchmarks
from quant_fund.metrics.vol_loss_decomposition import run_loss_model_decomposition_study
from quant_fund.models.extra_tilt_conformal import bench_extra_tilt as _extra_tilt_core_bench
from quant_fund.models.hierarchical_conformal import bench_ghcp as _ghcp_core_bench
from quant_fund.models.multisource_conformal import (
    bench_ms_rlcp_two_bumps as _ms_rlcp_core_bench,
)

#: Wave-16 stamp seed (drives the vol-loss-decomposition study, the only family
#: whose driver takes a bare seed argument; the others reuse their lane-verified
#: default seeds, pinned explicitly below).
_SEED = 20261016

#: Lane-verified default seeds for the module bench entry points (pinned
#: explicitly so the blobs are reproducible independent of any default drift).
_GHCP_SEED = 11
_MSRLCP_SEED = 2609
_EXTRA_SEED = 17
_FS_SEED = 20260922

#: GHCP repetitions SHRUNK from the module default 200 (wave-16 runtime
#: budget); at n_reps = 100 the min coverage over m lands ~0.91 (nominal 0.90).
_GHCP_REPS = 100

#: ExTRA replications SHRUNK from the module default 10 (brief: <= 6).
_EXTRA_REPS = 6

#: Tiny-budget C51 RL market maker (NOT the paper-scale ``@slow`` benchmark).
_RL_SEED = 0
_RL_HORIZON = 600.0
_RL_TRAIN_EPISODES = 3
_RL_EVAL_SEEDS = 2
_RL_INVENTORY_CAP = 8
_RL_REGIME_TAU_R = 60.0
_RL_REGIME_OMEGA = 0.30


def bench_vol_loss_decomposition() -> dict[str, float]:
    """Loss-choice vs model-choice decomposition flip, SYNTHETIC (wave 16).

    Tokajuk & Chudziak (2026), "Loss Choice or Model Choice? The Role of
    Forecast Level in Cryptocurrency Volatility Forecasting", arXiv:2609.27024
    (ADMA 2026); Patton (2011, QLIKE robustness to the noisy squared-return
    proxy); Kupiec (1995, POF coverage). Thin float-only adapter over the
    module's ``run_loss_model_decomposition_study`` (which returns a mixed
    float|str|nested blob — the ``schema`` / ``kind`` / ``claim`` / ``citation``
    str stamps and the score matrices are filtered; wave-12 ``rwcv`` /
    wave-14 ``xva`` precedent). On the module's seeded SYNTHETIC planted
    GARCH(1,1) world (loss enters only through a persistent level factor, model
    only through movement quality), validation-based level alignment must FLIP
    the paper's central statistic: the raw pairwise-marginal ratio Delta_L /
    Delta_M is loss-dominated (> 1; paper median 2.91) while the level-aligned
    ratio is model-dominated (< 1; paper 0.67), and alignment narrows the
    cross-loss one-day VaR breach-rate spread (paper: removes 97% of the
    variation). SHRUNK to a SINGLE seed and a SINGLE score block (the 2-D
    (n_losses, n_models) QLIKE matrix — the minimal block count vs the paper's
    25 asset-fold blocks), at the module's default n = 2600 / 600 validation /
    600 test. Seeded SYNTHETIC; QLIKE-ratio / VaR-breach calibration
    diagnostics only, never market evidence.
    """
    try:
        raw = run_loss_model_decomposition_study(seed=_SEED, n=2600, n_validation=600, n_test=600)
        qlike = raw["decomposition_qlike"]
        mapped = {
            "vld_ratio_raw": float(qlike["raw"]["ratio_median"]),
            "vld_ratio_aligned": float(qlike["aligned"]["ratio_median"]),
            "vld_breach_spread_narrowing": float(raw["breach_spread_narrowing"]),
            "vld_flip_loss_to_model": 1.0 if raw["flip_loss_to_model"] else 0.0,
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hierarchical_conformal() -> dict[str, float]:
    """Generalized hierarchical conformal prediction (GHCP), SYNTHETIC (w16).

    Mallick, Tchetgen Tchetgen, Dobriban & Lee (2026), "Generalized Hierarchical
    Conformal Prediction", arXiv:2608.15500 (finite-sample distribution-free
    validity under group-level exchangeability, Thm 2.1 / Cor 2.6); extends Lee,
    Barber & Willett (2026), ACM J. Data Science (hierarchical CP). Thin
    float-only adapter over the module's ``bench_ghcp`` (which returns a mixed
    float|str blob — the ``dgp`` / ``claim`` str stamps are filtered; wave-12
    ``rwcv`` precedent): on the seeded hierarchical-Gaussian DGP (group effects
    B_j ~ N(0, gamma^2), within-group noise N(0, sigma^2)), the GHCP interval
    must hold marginal coverage across every test-stream depth m (min coverage
    >= nominal - 0.05), and adding in-stream observations must SHRINK the mean
    interval width (the m = 10 width is well below the m = 0 width — the
    paper's core efficiency gain from conditioning on o >= 1 records). SHRUNK
    to n_reps = 100 (module default 200), lane seed 11. Seeded SYNTHETIC;
    coverage / interval-width calibration diagnostics only, never market
    evidence.
    """
    try:
        raw = _ghcp_core_bench(n_reps=_GHCP_REPS, seed=_GHCP_SEED)
        mapped = {
            "ghcp_min_coverage": float(raw["synthetic_min_coverage"]),
            "ghcp_width_shrinks": float(raw["synthetic_width_shrinks"]),
            "ghcp_coverage_m0": float(raw["synthetic_coverage_m0"]),
            "ghcp_mean_width_m0": float(raw["synthetic_mean_width_m0"]),
            "ghcp_mean_width_m10": float(raw["synthetic_mean_width_m10"]),
            "ghcp_alpha": float(raw["synthetic_alpha"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_multisource_conformal() -> dict[str, float]:
    """Multi-source randomly localized conformal prediction, SYNTHETIC (w16).

    Hore, Chatterjee & Choudhury (2026), "Multi-source conformal prediction:
    leveraging heterogeneity via localization", arXiv:2609.14531 (MS-RLCP, the
    shared-P_{Y|X} covariate-shift setting and the Theorem 3.3 envelope bound);
    Hore & Barber (2025), JRSS-B 87(2), arXiv:2310.07850 (RLCP local weights);
    Tibshirani, Barber, Candes & Ramdas (2019), weighted CP. Thin float-only
    adapter over the module's ``bench_ms_rlcp_two_bumps`` (which returns a mixed
    float|str blob — the ``dgp`` / ``claim`` / ``kernel`` str stamps are
    filtered; wave-12 ``rwcv`` precedent): on the seeded two-bump DGP (sources
    at X ~ N(-/+sep, std^2), shared Y = sigma(X) eps, in-source test from the
    50/50 mixture, gap test from a thinly covered N(0, gap_std^2) region),
    MS-RLCP must keep in-source coverage near nominal (>= 0.865 at
    alpha = 0.10) while DEGRADING VISIBLY in the gap — the vacuous-set fraction
    and the poorly-represented rate both rise in the gap relative to in-source,
    and the Theorem 3.3 coverage bound goes vacuous there (== 1). The module
    defaults (400 train / 400 cal / 600 in-source test / 300 gap test,
    n_grid = 4001, seed 2609) already sit inside the wave-16 budget (no shrink;
    measured ~0.01 s). Seeded SYNTHETIC; coverage / set-vacuity calibration
    diagnostics only, never market evidence.
    """
    try:
        raw = _ms_rlcp_core_bench(seed=_MSRLCP_SEED)
        mapped = {
            "msrlcp_coverage_in_source": float(raw["synthetic_coverage_in_source"]),
            "msrlcp_frac_vacuous_gap": float(raw["synthetic_frac_vacuous_gap"]),
            "msrlcp_frac_vacuous_in_source": float(raw["synthetic_frac_vacuous_in_source"]),
            "msrlcp_poorly_represented_rate_gap": float(
                raw["synthetic_poorly_represented_rate_gap"]
            ),
            "msrlcp_bound_vacuous_gap": float(raw["synthetic_bound_vacuous_gap"]),
            "msrlcp_alpha": float(raw["synthetic_alpha"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_extra_tilt() -> dict[str, float]:
    """Conformal prediction under exponential-tilt joint shift, SYNTHETIC (w16).

    Choi (2026), "Conformal Prediction under Exponential-Tilt Joint Shift",
    arXiv:2609.30886 (ExTRA-WCP / -WCP-T: exponential-tilt reweighting that
    covers BOTH the input-distribution shift and the input-response shift); the
    ExTRA estimator is from Maity et al., ICLR 2023; Tibshirani et al. (2019),
    weighted CP. Thin float-only adapter over the module's ``bench_extra_tilt``
    (which returns a mixed float|str blob — the ``dgp`` / ``claim`` str stamps
    are filtered; wave-12 ``rwcv`` precedent): on the seeded SYNTHETIC bimodal
    joint-shift fixture at the INFORMATIVE design (eta = 1), the tilted WCP-T
    interval must cut paired set length by a wide margin (> 15%; the paper finds
    ~30% at Table 2) while its coverage stays close to the untilted WCP
    (|coverage difference| small under the informative shift), the mode-signal
    standard deviation stays non-degenerate, and the tilt weights keep a
    non-collapsed effective sample size. SHRUNK to replications = 6 (module
    default 10; brief <= 6) at the module's default sample sizes, lane seed 17.
    Seeded SYNTHETIC; coverage / set-length calibration diagnostics only, never
    market evidence.
    """
    try:
        raw = _extra_tilt_core_bench(replications=_EXTRA_REPS, seed=_EXTRA_SEED)
        mapped = {
            "extra_paired_length_reduction_percent": float(
                raw["synthetic_paired_length_reduction_percent"]
            ),
            "extra_coverage_diff_wcp_t_minus_wcp": float(
                raw["synthetic_coverage_diff_wcp_t_minus_wcp"]
            ),
            "extra_mode_signal_sd": float(raw["synthetic_mode_signal_sd"]),
            "extra_weight_ess_percent": float(raw["synthetic_weight_ess_percent"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_forecast_selection() -> dict[str, float]:
    """Target-alignment dilution accounting + cautious selection, SYNTHETIC (w16).

    Soleimani (2026), "Target alignment, dilution and forecast selection when
    cross-sectional forecasts share a common target", arXiv:2609.26303
    (International Journal of Forecasting; the Proposition 5 error-correlation
    mirror, the Proposition 9 per-diluter accounting, and the three-way cautious
    selection rule); Romano & Wolf (2005), Bretz & Westfall (2008), Newey-West
    standard errors. Thin float-only adapter over the module's
    ``forecast_selection_benchmarks`` (already a flat ``dict[str, float]`` — no
    str stamps to filter): on the seeded SYNTHETIC planted world (6 aligned
    forecasters + 16 zero-alignment diluters with cluster correlation 0.30), the
    scale-free three-way rule must NEVER admit a pure diluter
    (``fs_three_way_sf_diluters_admitted == 0`` — the scale-free basis never
    rewards dilution) and must remove a large fraction of the full-pool dilution
    loss (> 0.5), while the equal-weight basis DOES admit diluters (>= 0 — the
    replicated scale-mismatch failure mode). ``fs_deviation_mirror_affine_r2``
    is the PAIR-LEVEL mirror R^2 (Prop 5 / Section 9.2, paper headline 0.9999,
    assert > 0.9) — see the module docstring key-semantics note. Runs
    ``fast=True`` (800 dates vs 1600), lane seed 20260922. Seeded SYNTHETIC;
    relative-score-risk / admission calibration diagnostics only, never market
    evidence.
    """
    try:
        raw = forecast_selection_benchmarks(seed=_FS_SEED, fast=True)
        mapped = {
            "fs_three_way_sf_diluters_admitted": float(raw["three_way_sf_diluters_admitted"]),
            "fs_three_way_sf_removal_fraction": float(raw["three_way_sf_removal_fraction"]),
            "fs_three_way_ew_diluters_admitted": float(raw["three_way_ew_diluters_admitted"]),
            "fs_deviation_mirror_affine_r2": float(raw["pair_deviation_mirror_affine_r2"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_rl_market_maker() -> dict[str, float]:
    """C51 distributional-RL market maker vs classic AS, SYNTHETIC (w16, torch).

    Moret & Lillo (2026), "Deep learning of robust market making under
    regime-switching order flow", arXiv:2609.11614 (a distributional DQN market
    maker on a ZI / Santa Fe LOB; under regime-switching flow a stationarily
    oriented controller SATURATES inventory — the motivating failure mode);
    Bellemare, Dabney & Munos (2017), ICML, arXiv:1707.06887 (C51 atoms /
    categorical projection); Adams & MacKay (2007), the run-length flow filter.
    TORCH-GATED (wave-12 ``deep_hedging`` precedent): the module imports torch
    lazily, so a torch-less environment raises ImportError and this bench
    returns ``{}``. Runs a TINY-BUDGET path — train a 11-atom / hidden-16 /
    buffer-2000 C51 for 3 episodes under regime flow, then evaluate the greedy
    policy against Avellaneda-Stoikov (AS) and GLFT on matched seeds under
    STATIONARY and REGIME flow (2 paired seeds, 600 s horizon) — NOT the
    paper-scale ``@slow`` ``rl_mm_benchmark`` (101 atoms / hidden 256 / buffer
    1e5 / hours). The C51 agent must COMPLETE every session (== 1.0) and never
    BREACH the hard inventory cap ``q_max`` (``rlmm_inventory_cap_violations``
    counts max-abs-inventory strictly above the cap; the one-sided gating clamps
    AT the cap, so this safety invariant is == 0), while the classic AS policy
    SATURATES under regime flow (``rlmm_regime_saturation_as > 0`` — the
    motivating failure mode) but not under stationary flow
    (``rlmm_stationary_saturation_as == 0``). Honesty: the ``sim_internal_*``
    PnL keys stay OUT of the blob (only inventory / completion / config keys are
    emitted); SYNTHETIC simulator-internal diagnostics, never headline metrics,
    never market evidence, no live-trading claim.
    """
    try:
        # Local import: the RL-MM module needs torch only at call time, and the
        # ImportError guard below lets torch-less environments skip cleanly.
        from quant_fund.microstructure.rl_market_maker import (
            C51Config,
            C51MarketMaker,
            RLStateSpec,
            evaluate_rl_market_makers,
            paper_regime_flow,
            train_c51_market_maker,
        )
        from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

        config = santa_fe_config(seed=_RL_SEED)
        spec = RLStateSpec(
            aux_enabled=True, filter_tau_r=_RL_REGIME_TAU_R, inventory_cap=_RL_INVENTORY_CAP
        )
        agent_cfg = C51Config(
            n_atoms=11,
            hidden=(16,),
            buffer_capacity=2000,
            target_update_every=50,
            eps_decay_steps=200,
            batch_size=32,
            seed=_RL_SEED,
        )
        agent = C51MarketMaker(spec, agent_cfg)
        train_c51_market_maker(
            agent=agent,
            config=config,
            horizon=_RL_HORIZON,
            n_episodes=_RL_TRAIN_EPISODES,
            seed_base=_RL_SEED,
            flow_factory=lambda s: paper_regime_flow(
                seed=s, tau_r=_RL_REGIME_TAU_R, omega=_RL_REGIME_OMEGA
            ),
        )
        evaluation = evaluate_rl_market_makers(
            agent=agent,
            config=config,
            horizon=_RL_HORIZON,
            n_seeds=_RL_EVAL_SEEDS,
            seed_base=_RL_SEED + 7_777_777,
            regime_tau_r=_RL_REGIME_TAU_R,
            regime_omega=_RL_REGIME_OMEGA,
        )
        metrics = evaluation["metrics"]
        cap = evaluation["inventory_cap"]
        sessions = evaluation["sessions"]
        violations = 0.0
        for combo in ("c51_stationary", "c51_regime"):
            for row in sessions[combo]:
                if row["max_abs_inventory"] > cap:
                    violations += 1.0
        mapped = {
            "rlmm_c51_session_completion": float(
                min(
                    metrics["c51_stationary_session_completion_rate"],
                    metrics["c51_regime_session_completion_rate"],
                )
            ),
            "rlmm_inventory_cap_violations": float(violations),
            "rlmm_regime_saturation_as": float(metrics["as_regime_inventory_saturation_rate"]),
            "rlmm_stationary_saturation_as": float(
                metrics["as_stationary_inventory_saturation_rate"]
            ),
            "rlmm_inventory_cap": float(cap),
            "rlmm_horizon": float(evaluation["horizon"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except ImportError:
        return {}
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
