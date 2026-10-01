"""Benchmark family registry and forbidden research-metric keys.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import math

BENCHMARK_CATALOG_VERSION = 2
# Schema 2 stamps ``backtest_overfitting`` (PBO, DSR, PSR, MinTRL, trial counts).
# Schema 1 receipts remain valid: they predate the block and are not rewritten.
RESEARCH_RECEIPT_SCHEMA_VERSION = 2
RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED = frozenset({1, 2})
# Kept for the pre-split catalog API. These are diagnostic key groups.
REQUIRED_CHRISTOFFERSEN_CC_KEYS = frozenset({"christoffersen_cc_p", "christoffersen_cc_lr"})
PREFERRED_CHRISTOFFERSEN_IND_KEYS = frozenset({"christoffersen_ind_p", "christoffersen_ind_lr"})
REQUIRED_BENCHMARK_FAMILIES = frozenset(
    {
        "ranking",
        "alpha",
        "volatility",
        "distribution",
        "regime",
        "tail",
        "drawdown",
        "liquidity",
        "reinforcement",
        "conformal",
        "evalues",
        "jackknife_plus",
        "crc",
        "weighted_conformal",
        "interval_risk",
        "quantile_bandit",
        "cv_plus",
        "cpcv",
        "localized_conformal",
        "conformal_rank",
        "online_crc",
        "portfolio_conformal",
        "northset",
    }
)
# Optional families may appear in a receipt (and are then fully soft-verified)
# but are never required by the benchmark catalog. ``candle_order_book`` is the
# optional candlestick+L2 family emitted by the ``candle-book`` CLI; without
# this allow-list the verify.py candle honesty dispatch could never run.
OPTIONAL_BENCHMARK_FAMILIES = frozenset(
    {
        "candle_order_book",
        "robinhood_plus",
        # Descriptive proper-score diagnostics of the SYNTHETIC return panel
        # (see research/benches_extra.py); never live-P&L or headline ratios.
        "complexity",
        "roughness",
        "serial_randomness",
        # SOTA canon waves 8-10 batteries (see research/benches_w810.py):
        # anytime-valid inference, multivariate proper scores, TS conformal,
        # regime-conditional evaluation, the SYNTHETIC leaky-oracle red
        # team, and distributional-ML baselines. All seeded SYNTHETIC
        # streams; correctness diagnostics only, never promotion gates.
        "anytime_valid",
        "energy_score",
        "ts_conformal",
        "regime_eval",
        "leakage_redteam",
        "distributional_ml",
        # SOTA canon wave 11 batteries (see research/benches_w11.py):
        # rough-path signatures, optimal transport, Wasserstein DRO, and
        # conformal PID control. Seeded SYNTHETIC streams; correctness
        # diagnostics only, never promotion gates.
        "rough_paths",
        "optimal_transport",
        "wasserstein_dro",
        "conformal_pid",
        # SOTA canon wave 12 batteries (see research/benches_w12.py):
        # NexCP nonexchangeable conformal bounds, multi-horizon EnbPI,
        # Bayesian stacking / pseudo-BMA(+), regime-weighted conformal VaR,
        # sliced-Wasserstein two-sample testing, proper-score decompositions,
        # anytime-valid confidence sequences, and (torch-gated) deep hedging.
        # Seeded SYNTHETIC streams; correctness diagnostics only, never
        # promotion gates.
        "nexcp",
        "mh_enbpi",
        "stacking",
        "rwcv",
        "sliced_wasserstein",
        "score_decomposition",
        "confidence_sequences",
        "deep_hedging",
        # SOTA canon wave 13 batteries (see research/benches_w13.py):
        # conformal coverage inference under temporal dependence, minimax-
        # optimal conformal e-detectors, rolling conformal prediction for
        # sequential training, anytime-valid rank confidence sequences /
        # BB-EDGE leaderboards, PICPI self-consistency intervals, replicable
        # conformal calibration, reference-null calibrated e-thresholds, and
        # delayed-feedback ACI. Seeded SYNTHETIC streams; correctness
        # diagnostics only, never promotion gates.
        "coverage_inference",
        "conformal_e_detectors",
        "rolling_conformal",
        "rank_cs",
        "picpi",
        "replicable_conformal",
        "reference_null",
        "delayed_aci",
        # SOTA canon wave 14 batteries (see research/benches_w14.py): deep
        # finance math — martingale optimal transport model-free bounds,
        # large-deviations importance sampling, the trade-crowding mean-field
        # game, vine + GAS copulas, Malliavin-calculus Monte Carlo Greeks,
        # XVA (CVA/FVA/MVA + WWR), American LSM with dual bounds, Dupire local
        # vol + SLV mixing MC, the deep BSDE solver, the DeRegiME regime-mixture
        # head, two-stage odd residual flows, deep kernel hedging, the
        # zero-intelligence LOB simulator, and cash-constrained multi-asset
        # optimal execution. (models/fourier_pricing.py is deliberately NOT
        # wired — its COS European path has a documented open bug.) Seeded
        # SYNTHETIC streams; correctness diagnostics only, never promotion
        # gates.
        "martingale_ot",
        "large_deviations",
        "mean_field_games",
        "vine_copula",
        "malliavin_greeks",
        "xva",
        "american_lsm",
        "local_stoch_vol",
        "deep_bsde",
        "deep_regime_mixture",
        "odd_residual_flows",
        "deep_kernel_hedging",
        "zi_lob",
        "cash_constrained_oe",
        # SOTA canon wave 15 batteries (see research/benches_w15.py): agentic
        # research integrity + repaired Fourier pricing — dynamic-subspace
        # denoising under oblique structured noise, Transported Conformal
        # Calibration (TCC), HPD split conformal (C-USIM) for multimodal
        # predictive laws, EverMine capability-value Cap-swap accounting, the
        # anytime-valid frozen referee for LLM factor mining, revision-aware
        # vintage evaluation, the entropy-Shapley attribution hierarchy, and
        # the Fourier pricing suite (COS / COS-Bermudan / Hilbert barrier) now
        # that its European leg is REPAIRED and matches BS to ~1e-13 (wave 14
        # had deliberately skipped it; that comment block stays as history).
        # Seeded SYNTHETIC streams; correctness diagnostics only, never
        # promotion gates.
        "subspace_denoising",
        "conformal_transfer",
        "hpd_conformal",
        "capability_value",
        "agent_referee",
        "vintage_eval",
        "entropy_shapley",
        "fourier_pricing",
        # SOTA canon wave 16 batteries (see research/benches_w16.py):
        # loss-choice-vs-model-choice decomposition for volatility forecasts
        # (validation level alignment and the loss-dominated -> model-dominated
        # QLIKE-ratio flip), generalized hierarchical conformal prediction
        # (GHCP), multi-source randomly localized conformal prediction
        # (MS-RLCP), conformal prediction under an exponential-tilt joint shift
        # (ExTRA-WCP / -WCP-T), target-alignment dilution accounting with
        # cautious forecast selection, and (torch-gated) the C51 distributional
        # RL market maker on the zero-intelligence LOB. Seeded SYNTHETIC
        # streams; correctness diagnostics only, never promotion gates.
        "vol_loss_decomposition",
        "hierarchical_conformal",
        "multisource_conformal",
        "extra_tilt",
        "forecast_selection",
        "rl_market_maker",
        # SOTA canon wave 17 batteries (see research/benches_w17.py):
        # DiffPTS LSNM diffusion forecasting on an AR(1)-bimodal stream
        # (CRPS/coverage/PIT vs NGboost/QRF, DDPM-vs-DDIM NFE budgets),
        # extrapolated weighted conformal (+ harmonic-mean extension),
        # Cheridito-Weiss multi-level deep market making, Moret-Lillo
        # Algorithm-C C51 scenario-bandit robustness (both torch-gated),
        # sliced-graph-alignment multistep UQ certificates, Barzykin
        # passive-execution-vs-impact planning, and Nutz-Voss singular
        # stochastic tracking sharp-rate convergence. Seeded SYNTHETIC
        # streams; correctness diagnostics only, never promotion gates.
        "diffpts",
        "extra_conformal",
        "multilevel_mm",
        "rlmm_c51",
        "sga_uq",
        "passive_impact",
        "stochastic_tracking",
        # SOTA canon wave 18 batteries (see research/benches_w18.py):
        # G-SLiCE path-space flow matching vs a GP-prior baseline and the
        # latent neural SDE probabilistic forecaster vs its exact oracle
        # kernel (both torch-gated), plus StocBench fixed-budget sampler
        # evaluation (allocation, paired differentials, anytime-valid
        # significance, aleatoric/epistemic split, rollout drift). Seeded
        # SYNTHETIC streams; correctness diagnostics only, never promotion
        # gates.
        "gslice",
        "neural_sde",
        "stocbench",
        # agentic-LOB phase-transition diagnostics on the ZI-LOB, the
        # FASE self-evolving forecast-evaluation protocol, and the KiT
        # OHLCV candle-path pipeline (all pure-numpy cores; the KiT torch
        # lane is exercised by its own gated tests). Same SYNTHETIC
        # diagnostic contract.
        "agentic_lob",
        "fase_eval",
        "kit_paths",
        # SOTA canon wave 19 batteries (see research/benches_w19.py):
        # generalized-Langevin latent-liquidity impact (Itkin 2026,
        # arXiv:2609.37872), event-time order-flow memory /
        # operational-time impact (arXiv:2609.13715), Fukasawa's
        # first-order implied-variance representation (arXiv:2609.13961),
        # the AD-Seq-Vol conditional IVS diffusion with static no-arb
        # post-training penalties (arXiv:2609.13402, torch-gated), and the
        # RCCP retrieval-corrected (arXiv:2608.10553) + DCP
        # distribution-aware (arXiv:2605.26569) conformal frameworks.
        # Same SYNTHETIC diagnostic contract.
        "langevin_impact",
        "event_time_flow",
        "fukasawa_iv",
        "ivs_diffusion",
        "rccp",
        "dcp",
        # SOTA canon wave 20 batteries (see research/benches_w20.py):
        # perpetual variance-swap optimal stopping (Lorig-Lozano-Gomez),
        # belief-Markovian equilibrium pricing under a hidden Markov
        # dividend factor, Gaussian normalized coordinates / risk-neutral
        # CDF deformations (Sun 2026, arXiv:2609.14212), and liquidity-tail
        # LOB equilibrium under heavy-tailed demand (Cetin-Lin-Livieri,
        # arXiv:2607.01198), and adversarial-RL market making with Hawkes
        # order flow and price impact (Yang & Xu 2026, arXiv:2609.22785 —
        # torch-gated). Same SYNTHETIC diagnostic contract.
        "varswap_stopping",
        "hidden_markov_equilibrium",
        "gaussian_normalized_coords",
        "liquidity_tail_lob",
        "arl_mm",
        # SOTA canon wave 21 batteries (see research/benches_w21.py):
        # Bayesian online change-point detection (Adams & MacKay 2007,
        # arXiv:0710.3742) and rough-volatility pricing — fractional
        # Riccati rHeston CF + rBergomi/Volterra simulators (El Euch &
        # Rosenbaum 2019, arXiv:1609.02108; Bayer-Friz-Gatheral 2016;
        # Abi Jaber-Larsson-Pulido 2019, arXiv:1708.08796), and path
        # signatures — Goursat-PDE signature kernel + lead-lag MMD
        # two-sample scores (Chevyrev & Oberhauser 2022,
        # arXiv:1810.10971; Salvi-Cass-Foster-Lyons-Yang 2021,
        # arXiv:2006.14794). Same SYNTHETIC diagnostic contract.
        "bocpd_changepoint",
        "rough_heston_rbergomi",
        "signature_features",
        # SOTA canon wave 22 batteries (see research/benches_w22.py):
        # expected-signature martingale-validity test — omnibus
        # signature-moment scores, permutation/bootstrap p-calibration,
        # and an e-process arm (Chevyrev & Oberhauser 2022,
        # arXiv:1810.10971; see the module docstring for the verified
        # companion citation set). Same SYNTHETIC diagnostic contract.
        "signature_martingale_test",
        # SOTA canon wave 23 batteries (see research/benches_w23.py):
        # SVI/SSVI surface calibration + no-arb checks (Gatheral &
        # Jacquier 2014, arXiv:1204.0646), transient propagator impact
        # kernel estimation, queue-reactive CTMC limit-book dynamics,
        # Koopman/EDMD nonlinear-spectrum extraction, signature-
        # Wasserstein GAN generation (arXiv:2006.05421), and transformer
        # neural temporal point processes with Hawkes fallback. Same
        # SYNTHETIC diagnostic contract.
        "svi_surface",
        "propagator_impact",
        "queue_reactive",
        "koopman_edmd",
        "sig_gan",
        "neural_tpp",
    }
)
BENCHMARK_FAMILY_ORDER = (
    "ranking",
    "alpha",
    "volatility",
    "distribution",
    "regime",
    "tail",
    "drawdown",
    "liquidity",
    "reinforcement",
    "conformal",
    "evalues",
    "jackknife_plus",
    "crc",
    "weighted_conformal",
    "interval_risk",
    "quantile_bandit",
    "cv_plus",
    "cpcv",
    "localized_conformal",
    "conformal_rank",
    "online_crc",
    "portfolio_conformal",
    "northset",
)
# Headline P&L / ratio key *tokens* that must never appear in research family /
# scorecard blobs (not the paper analytics_export schema — that catalog allows
# equity/stress pnl/nav diagnostics under live_pnl_claim=false).
# Matching is by underscore-token on mapping keys (case-insensitive), so
# ``flag_high_sharpe`` / ``shock_down_pnl`` / ``calmar_diagnostic`` all fail closed.
FORBIDDEN_RESEARCH_METRIC_KEYS = frozenset(
    {
        "sharpe",
        "sortino",
        "calmar",
        "pnl",
        "nav",
    }
)


def _iter_mapping_keys(obj: object) -> list[str]:
    """Collect nested mapping keys (dicts only; list elements walked)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_iter_mapping_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_iter_mapping_keys(item))
    return keys


def family_blob_forbidden_metrics_absent(payload: object) -> bool:
    """Return True iff *payload* has no forbidden research-headline metric keys.

    Fail closed: any mapping key whose underscore tokens include sharpe / sortino /
    calmar / pnl / nav marks the blob unclean. Values are not scanned (keys only).

    Scope: research family / scorecard blobs only. Paper ``analytics_export`` may
    contain equity ``nav_*`` / stress ``*_pnl`` diagnostics; validate those with
    ``validate_analytics_export`` (live_pnl_claim fail-closed), not this helper.

    ``live_pnl_claim`` itself is exempt at any depth: it is the honesty flag,
    not a metric — receipts that embed other receipts carry it nested (e.g. a
    tournament manifest quoting its benchmark manifest).
    """
    for key in _iter_mapping_keys(payload):
        if key == "live_pnl_claim":
            continue
        parts = str(key).lower().replace("-", "_").split("_")
        if any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in parts if tok):
            return False
    return True


def family_blob_executed(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``executed`` twin)."""
    return bool(payload)


def family_blob_nonempty(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``nonempty`` twin)."""
    return bool(payload)


def family_blob_has_finite_observation(payload: object) -> bool:
    """True iff *payload* recursively contains at least one finite observation.

    Lifted from research.agent ``_benchmark_scorecard`` nested ``has_observation``
    (Day Wave 44). Semantics:
    - dict / list / tuple → any child is an observation
    - bool → True
    - int (incl. numpy integer scalars via ``.item()``) → True
    - float (incl. numpy floating) → ``math.isfinite``
    - str → nonempty strip and not ``nan`` / ``none`` (case-insensitive)
    - else → False

    Empty ``{}`` / all-NaN blobs → False. Used by soft scorecard forge verify.
    """
    if isinstance(payload, dict):
        return any(family_blob_has_finite_observation(item) for item in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(family_blob_has_finite_observation(item) for item in payload)
    if isinstance(payload, bool):
        return True
    if isinstance(payload, int):
        return True
    if isinstance(payload, float):
        return math.isfinite(payload)
    # Numpy scalars are not always subclasses of int/float — recurse via .item().
    if (
        hasattr(payload, "dtype")
        and hasattr(payload, "item")
        and not isinstance(payload, (bytes, bytearray, memoryview))
    ):
        try:
            return family_blob_has_finite_observation(payload.item())
        except (ValueError, TypeError, AttributeError):
            return False
    return (
        isinstance(payload, str)
        and bool(payload.strip())
        and payload.lower() not in {"nan", "none"}
    )


__all__ = [
    "BENCHMARK_CATALOG_VERSION",
    "BENCHMARK_FAMILY_ORDER",
    "FORBIDDEN_RESEARCH_METRIC_KEYS",
    "OPTIONAL_BENCHMARK_FAMILIES",
    "REQUIRED_BENCHMARK_FAMILIES",
    "PREFERRED_CHRISTOFFERSEN_IND_KEYS",
    "REQUIRED_CHRISTOFFERSEN_CC_KEYS",
    "RESEARCH_RECEIPT_SCHEMA_VERSION",
    "RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED",
    "family_blob_executed",
    "family_blob_forbidden_metrics_absent",
    "family_blob_has_finite_observation",
    "family_blob_nonempty",
]
