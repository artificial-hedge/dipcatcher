"""Benchmark batteries for SOTA canon wave 16 (integration lane).

Wires the three landed wave-16 module lanes as OPTIONAL scorecard families —
``multisource_conformal``, ``hierarchical_conformal`` and
``vol_loss_decomposition`` — as flat ``dict[str, float]`` blobs. This file is
integration-only: no new science lives here. Each bench is a thin adapter
over the wired module's own seeded SYNTHETIC bench helper / study driver
(the wave-12 ``rwcv`` / wave-15 ``tcc`` / ``cusim_bimodal`` adapter
precedent), filtering mixed ``float | str`` blobs down to float-only keys
under a family prefix, or ``{}`` if the synthetic setup cannot be
constructed.

Composition (import, never reimplement):
- ``models/multisource_conformal.py`` owns MS-RLCP (source selection,
  kernel localization, the Theorem 3.3 envelope bound) and its planted
  two-bump fixture ``bench_ms_rlcp_two_bumps`` — reused wholesale; the
  localized-vs-envelope and in-source-vs-gap honesty cells are already the
  lane-validated configuration.
- ``models/hierarchical_conformal.py`` owns GHCP (donor selection, the
  ``nu_donor`` calibration measure, trivial-set honesty) and its planted
  hierarchical-Gaussian runner ``bench_ghcp`` — reused wholesale; the
  authors' DGP is already encoded there.
- ``metrics/vol_loss_decomposition.py`` owns the loss-vs-model
  decomposition (validation-level alignment, pairwise marginal gaps,
  LevelShare, VaR breach rates) and its planted GARCH(1,1) study driver
  ``run_loss_model_decomposition_study`` — reused wholesale; the nested
  study blob is flattened to ``voldec_*`` scalars here.

Honesty (AGENTS.md contract): seeded SYNTHETIC streams only — no panel or
vendor data, no headline performance ratios (proper scores / coverage /
width / bound / decomposition diagnostics only). Every bench is
deterministic: repeated calls are bit-identical (all randomness lives in
the wired modules' seeded generators). Monte-Carlo budgets are SHRUNK
relative to the lane suites where the defaults needed it (documented per
bench) so the whole wave-16 battery stays inside its ~60 s runtime
envelope (measured < 1 s); the accompanying research tests carry
correspondingly wider, documented tolerances. Registration in
``OPTIONAL_BENCHMARK_FAMILIES`` is wired centrally by the controller AFTER
merge, so this module and its tests deliberately never import
``research.catalog`` / ``registry.py``.

References (full citation blocks and fetch-verification notes live in the
wired modules' docstrings):
- Hore, R., Chatterjee, S. & Choudhury, N. (2026), "Multi-source conformal
  prediction: leveraging heterogeneity via localization",
  arXiv:2609.14531 [stat.ML] (MS-RLCP; built on Hore & Barber 2025,
  JRSS-B 87(2), arXiv:2310.07850, RLCP).
- Mallick, Tchetgen Tchetgen, Dobriban & Lee (2026), "Generalized
  Hierarchical Conformal Prediction", arXiv:2608.15500 [stat.ME] (GHCP;
  extends Lee, Barber & Willett 2026, ACM J. Data Sci., HCP).
- Tokajuk, A. & Chudziak, J. A. (2026), "Loss Choice or Model Choice? The
  Role of Forecast Level in Cryptocurrency Volatility Forecasting",
  arXiv:2609.27024 [q-fin.CP], ADMA 2026.
"""

from __future__ import annotations

import numpy as np

from quant_fund.metrics.vol_loss_decomposition import (
    run_loss_model_decomposition_study as _voldec_core_study,
)
from quant_fund.models.hierarchical_conformal import bench_ghcp as _ghcp_core_bench
from quant_fund.models.multisource_conformal import (
    bench_ms_rlcp_two_bumps as _msrlcp_core_bench,
)

_SEED = 20261002

#: Lane-validated seeds for the wired helpers (outside each lane's
#: calibration/tuning seed sets; margins verified in the wave-16 tests).
_GHCP_SEED = 11
_VOLDEC_SEED = 11


def bench_multisource_conformal() -> dict[str, float]:
    """MS-RLCP two-bump envelope-honesty bench, SYNTHETIC (wave 16).

    Hore, Chatterjee & Choudhury (2026), arXiv:2609.14531 (Alg. 1 source
    selection, the Sec. 3 envelope / Theorem 3.3 coverage bound); Hore &
    Barber (2025), JRSS-B 87(2), arXiv:2310.07850 (RLCP); Tibshirani, Barber,
    Candès & Ramdas (2019, NeurIPS 32). Thin float-only adapter over the
    module's own ``bench_ms_rlcp_two_bumps`` (mixed ``float | str`` blob —
    the ``dgp`` / ``claim`` / ``kernel`` str stamps are filtered; wave-12
    ``rwcv`` precedent): on the planted two-source Gaussian-bump fixture
    (shared P_{Y|X}, covariate shift only, plus a thinly-covered gap test
    region), the MS-RLCP set must (a) cover in-source at nominal within MC
    tolerance and in both source regions, (b) pick the aligned source in
    region, (c) DEGRADE VISIBLY in the gap — higher poorly-represented and
    vacuous rates, a much larger representation/localization bound term and
    a much lower (more negative) Theorem-3.3 bound — and (d) satisfy the
    Lemma 3.1 identity ``max_k ||g_k||_inf == B`` through the bench's oracle
    grid constants. Module defaults are the lane-validated configuration
    (seed 2609, 400 train / 400 cal / 600+300 test / 4001-point envelope
    grid) and already run in ~0.1 s — no shrink needed. Seeded SYNTHETIC;
    coverage / width / bound diagnostics only, never market evidence.
    """
    try:
        raw = _msrlcp_core_bench()
        mapped = {
            "msrlcp_alpha": float(raw["alpha"]),
            "msrlcp_n_eff": float(raw["n_eff"]),
            "msrlcp_alignment_threshold": float(raw["alignment_threshold"]),
            "msrlcp_coverage_in_source": float(raw["coverage_in_source"]),
            "msrlcp_coverage_in_source_region_a": float(raw["coverage_in_source_region_a"]),
            "msrlcp_coverage_in_source_region_b": float(raw["coverage_in_source_region_b"]),
            "msrlcp_coverage_gap": float(raw["coverage_gap"]),
            "msrlcp_mean_width_in_source": float(raw["mean_width_in_source"]),
            "msrlcp_qhat_median_in_source": float(raw["qhat_median_in_source"]),
            "msrlcp_frac_vacuous_in_source": float(raw["frac_vacuous_in_source"]),
            "msrlcp_frac_vacuous_gap": float(raw["frac_vacuous_gap"]),
            "msrlcp_poorly_represented_rate_in_source": float(
                raw["poorly_represented_rate_in_source"]
            ),
            "msrlcp_poorly_represented_rate_gap": float(raw["poorly_represented_rate_gap"]),
            "msrlcp_selection_rate_a_region_a": float(raw["selection_rate_a_region_a"]),
            "msrlcp_selection_rate_b_region_b": float(raw["selection_rate_b_region_b"]),
            "msrlcp_selection_rate_a_gap": float(raw["selection_rate_a_gap"]),
            "msrlcp_bound_in_source": float(raw["bound_in_source"]),
            "msrlcp_bound_gap": float(raw["bound_gap"]),
            "msrlcp_bound_vacuous_in_source": float(raw["bound_vacuous_in_source"]),
            "msrlcp_bound_vacuous_gap": float(raw["bound_vacuous_gap"]),
            "msrlcp_localization_term_in_source": float(raw["localization_term_in_source"]),
            "msrlcp_localization_term_gap": float(raw["localization_term_gap"]),
            "msrlcp_representation_term_in_source": float(raw["representation_term_in_source"]),
            "msrlcp_representation_term_gap": float(raw["representation_term_gap"]),
            "msrlcp_envelope_b": float(raw["envelope_b"]),
            "msrlcp_envelope_g_source_sup": float(raw["envelope_g_source_sup"]),
            "msrlcp_envelope_g_test_sup_in_source": float(raw["envelope_g_test_sup_in_source"]),
            "msrlcp_envelope_g_test_sup_gap": float(raw["envelope_g_test_sup_gap"]),
            "msrlcp_mean_perturb_distance": float(raw["mean_perturb_distance"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hierarchical_conformal() -> dict[str, float]:
    """GHCP few-observations-per-group coverage bench, SYNTHETIC (wave 16).

    Mallick, Tchetgen Tchetgen, Dobriban & Lee (2026), "Generalized
    Hierarchical Conformal Prediction", arXiv:2608.15500 (donor selection,
    the ``nu_donor`` calibration measure, Theorem 2.1 finite-sample
    validity); Lee, Barber & Willett (2026), ACM J. Data Sci.,
    doi:10.1145/3786352 (the o = 0 HCP special case). Thin float-only
    adapter over the module's own ``bench_ghcp`` (mixed ``float | str``
    blob — the ``dgp`` / ``claim`` str stamps are filtered; wave-12
    ``rwcv`` precedent): on the authors' Section-3.1 hierarchical-Gaussian
    DGP (K = 20 reference groups of sizes 16..35, group effects
    ``N(0, 5^2)``, one held-out test group), the GHCP set for the
    ``(o+1)``-th test-group observation must (a) cover at nominal for every
    initial-observation count ``m in {0, 1, 3, 10}`` within MC tolerance,
    (b) produce NO trivial (+inf) sets at these sizes, and (c) shrink
    materially as ``m`` grows — the value of the generalization over
    first-observation HCP. SHRUNK ``n_reps = 200`` (the lane's science test
    uses 300; seed 11 is the lane-validated configuration). Seeded
    SYNTHETIC; coverage / set-width diagnostics only, never market
    evidence.
    """
    try:
        raw = _ghcp_core_bench(n_reps=200, seed=_GHCP_SEED)
        mapped = {
            "ghcp_alpha": float(raw["alpha"]),
            "ghcp_eta": float(raw["eta"]),
            "ghcp_n_reps": float(raw["n_reps"]),
            "ghcp_coverage_m0": float(raw["coverage_m0"]),
            "ghcp_coverage_m1": float(raw["coverage_m1"]),
            "ghcp_coverage_m3": float(raw["coverage_m3"]),
            "ghcp_coverage_m10": float(raw["coverage_m10"]),
            "ghcp_mean_width_m0": float(raw["mean_width_m0"]),
            "ghcp_mean_width_m1": float(raw["mean_width_m1"]),
            "ghcp_mean_width_m3": float(raw["mean_width_m3"]),
            "ghcp_mean_width_m10": float(raw["mean_width_m10"]),
            "ghcp_trivial_share_m0": float(raw["trivial_share_m0"]),
            "ghcp_trivial_share_m1": float(raw["trivial_share_m1"]),
            "ghcp_trivial_share_m3": float(raw["trivial_share_m3"]),
            "ghcp_trivial_share_m10": float(raw["trivial_share_m10"]),
            "ghcp_min_coverage": float(raw["min_coverage"]),
            "ghcp_width_shrinks": float(raw["width_shrinks"]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_vol_loss_decomposition() -> dict[str, float]:
    """Loss-vs-model vol-forecast decomposition bench, SYNTHETIC (wave 16).

    Tokajuk & Chudziak (2026), "Loss Choice or Model Choice? The Role of
    Forecast Level in Cryptocurrency Volatility Forecasting",
    arXiv:2609.27024 [q-fin.CP] (ADMA 2026) — the ``Delta_L / Delta_M``
    pairwise-marginal-gap ratio and the raw loss-dominated -> aligned
    model-dominated flip (paper medians 2.91 -> 0.67), LevelShare, and the
    cross-loss VaR breach-spread narrowing (paper: ~97% removed). Scalar
    adapter over the module's own ``run_loss_model_decomposition_study``
    (a nested ``dict[str, Any]`` blob — the ``voldec_*`` keys extract the
    study's scalar diagnostics; list/matrix fields stay in the lane suite):
    on the seeded planted GARCH(1,1) world with Table-4-calibrated level
    factors (0.62 .. 2.85) and per-model movement quality, (a) the QLIKE
    median ratio must flip from > 1 raw to < 1 aligned, (b) the
    supplementary eta2 shares flip the same way, (c) LevelShare stays high
    (planted cross-loss differences are mostly level), and (d) the
    cross-loss breach-rate spread narrows by more than half. SHRUNK
    windows (n = 1600, validation 400 / test 300 vs the lane's
    n = 2600 / 600 / 600; seed 11 is a lane-verified study seed). Seeded
    SYNTHETIC; QLIKE / MSE-log decomposition and breach diagnostics only,
    never market evidence.
    """
    try:
        res = _voldec_core_study(_VOLDEC_SEED, n=1600, n_validation=400, n_test=300)
        dec = res["decomposition_qlike"]
        dec_ml = res["decomposition_mse_log"]
        raw_by_loss = res["breach_rate_by_loss_raw"]
        mapped = {
            "voldec_alpha_var": float(res["alpha_var"]),
            "voldec_n": float(res["n"]),
            "voldec_flip_loss_to_model": float(res["flip_loss_to_model"]),
            "voldec_ratio_median_raw": float(dec["raw"]["ratio_median"]),
            "voldec_ratio_median_aligned": float(dec["aligned"]["ratio_median"]),
            "voldec_loss_share_raw": float(dec["raw"]["loss_share_median"]),
            "voldec_model_share_raw": float(dec["raw"]["model_share_median"]),
            "voldec_loss_share_aligned": float(dec["aligned"]["loss_share_median"]),
            "voldec_model_share_aligned": float(dec["aligned"]["model_share_median"]),
            "voldec_ratio_median_mse_log_raw": float(dec_ml["raw"]["ratio_median"]),
            "voldec_ratio_median_mse_log_aligned": float(dec_ml["aligned"]["ratio_median"]),
            "voldec_level_share_mean": float(res["level_share_mean"]),
            "voldec_breach_spread_raw_mean": float(res["breach_spread_raw_mean"]),
            "voldec_breach_spread_aligned_mean": float(res["breach_spread_aligned_mean"]),
            "voldec_breach_spread_narrowing": float(res["breach_spread_narrowing"]),
            "voldec_breach_rate_first_loss_raw": float(raw_by_loss[0]),
            "voldec_breach_rate_last_loss_raw": float(raw_by_loss[-1]),
        }
        if not all(np.isfinite(v) for v in mapped.values()):
            return {}
        return mapped
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
