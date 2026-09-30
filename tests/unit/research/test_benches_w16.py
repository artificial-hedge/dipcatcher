"""Tests for research/benches_w16.py — SOTA canon wave 16 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): the
values are deterministic (seeded through the wired modules' own helpers), so
re-running a bench per test buys nothing. The determinism test at the bottom
re-runs each bench once and compares against its fixture bit-for-bit.

Tolerance policy: the wave-16 benches run SHRUNK Monte-Carlo budgets where
the module defaults needed it (GHCP ``n_reps = 200`` vs the lane suite's
300; the vol-decomposition study at ``n = 1600``, validation 400 / test 300
vs the lane's ``n = 2600`` / 600 / 600; MS-RLCP runs the module's
lane-validated default fixture unshrunk — it already fits the envelope).
The science assertions below are therefore DIRECTIONAL with the lane's
tolerances widened where budgets shrank — they pin the qualitative claims of
the cited papers (in-source coverage with visible gap degradation for
MS-RLCP; nominal coverage + width shrinkage across the initial-observation
count for GHCP; the loss-dominated -> model-dominated decomposition flip,
LevelShare and breach-spread narrowing for the vol decomposition) without
re-asserting the lane suites' tight windows.

Registration (wave-16 integration-lane contract): family wiring into
``OPTIONAL_BENCHMARK_FAMILIES`` is done centrally by the controller AFTER
merge, so this test module never imports ``quant_fund.research.catalog`` /
``registry.py`` — the forbidden-key and finite-observation checks below are
flat-blob specializations mirroring ``family_blob_forbidden_metrics_absent``
/ ``family_blob_has_finite_observation`` semantics for ``dict[str, float]``
payloads, and no membership assertion exists here by design.
"""

from __future__ import annotations

import numpy as np
import pytest

import quant_fund.research.benches_w16 as benches_w16
from quant_fund.research.benches_w16 import (
    bench_hierarchical_conformal,
    bench_multisource_conformal,
    bench_vol_loss_decomposition,
)

_FAMILIES = (
    "multisource_conformal",
    "hierarchical_conformal",
    "vol_loss_decomposition",
)

_PREFIXES = {
    "multisource_conformal": "msrlcp_",
    "hierarchical_conformal": "ghcp_",
    "vol_loss_decomposition": "voldec_",
}

#: Mirrors FORBIDDEN_RESEARCH_METRIC_KEYS in research/catalog/registry.py —
#: kept local because the integration lane must not import the registry
#: before the controller wires family registration (see module docstring).
_FORBIDDEN_KEY_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _blob_forbidden_key_tokens_absent(blob: dict[str, float]) -> bool:
    """Flat-blob mirror of ``family_blob_forbidden_metrics_absent``."""
    for key in blob:
        parts = str(key).lower().replace("-", "_").split("_")
        if any(tok in _FORBIDDEN_KEY_TOKENS for tok in parts if tok):
            return False
    return True


@pytest.fixture(scope="module")
def multisource_conformal() -> dict[str, float]:
    return bench_multisource_conformal()


@pytest.fixture(scope="module")
def hierarchical_conformal() -> dict[str, float]:
    return bench_hierarchical_conformal()


@pytest.fixture(scope="module")
def vol_loss_decomposition() -> dict[str, float]:
    return bench_vol_loss_decomposition()


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_family_blobs_registration_shape(family: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(family)
    assert isinstance(blob, dict) and blob
    # Flat float-only blobs, namespaced under the family prefix.
    assert all(isinstance(v, float) for v in blob.values())
    assert all(k.startswith(_PREFIXES[family]) for k in blob)
    assert all(np.isfinite(v) for v in blob.values())


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_family_blobs_forbidden_metrics_absent(family: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(family)
    assert _blob_forbidden_key_tokens_absent(blob)


def test_msrlcp_in_source_coverage_and_selection(
    multisource_conformal: dict[str, float],
) -> None:
    nominal = 1.0 - multisource_conformal["msrlcp_alpha"]
    # MS paper Alg. 1: the RLCP weighted quantile of the selected source keeps
    # coverage at nominal (MC slack) overall and in BOTH source regions.
    assert multisource_conformal["msrlcp_coverage_in_source"] >= nominal - 0.05
    assert multisource_conformal["msrlcp_coverage_in_source_region_a"] >= nominal - 0.06
    assert multisource_conformal["msrlcp_coverage_in_source_region_b"] >= nominal - 0.06
    # Not vacuously over-covering, and the data-adaptive argmax picks the
    # aligned source in-region while the gap is near a coin flip.
    assert multisource_conformal["msrlcp_coverage_in_source"] <= 0.98
    assert multisource_conformal["msrlcp_selection_rate_a_region_a"] >= 0.85
    assert multisource_conformal["msrlcp_selection_rate_b_region_b"] >= 0.85
    assert 0.25 <= multisource_conformal["msrlcp_selection_rate_a_gap"] <= 0.75
    # Localization keeps in-source sets informative.
    assert multisource_conformal["msrlcp_mean_width_in_source"] > 0.0
    assert multisource_conformal["msrlcp_qhat_median_in_source"] < 3.0


def test_msrlcp_gap_degrades_visibly(multisource_conformal: dict[str, float]) -> None:
    # Theorem-3.3 honesty: the thinly-covered gap region degrades the bound
    # instead of hiding it — higher poorly-represented and vacuous rates,
    # much larger representation/localization terms, much lower bound.
    blob = multisource_conformal
    assert blob["msrlcp_poorly_represented_rate_gap"] >= (
        blob["msrlcp_poorly_represented_rate_in_source"] + 0.4
    )
    assert blob["msrlcp_representation_term_gap"] >= 0.9
    assert blob["msrlcp_representation_term_in_source"] <= 0.5
    assert blob["msrlcp_frac_vacuous_gap"] >= (blob["msrlcp_frac_vacuous_in_source"] + 0.1)
    assert blob["msrlcp_bound_gap"] < blob["msrlcp_bound_in_source"]
    assert blob["msrlcp_bound_vacuous_gap"] == 1.0
    assert blob["msrlcp_localization_term_gap"] > 100.0 * blob["msrlcp_localization_term_in_source"]
    assert (
        blob["msrlcp_envelope_g_test_sup_gap"]
        > 100.0 * blob["msrlcp_envelope_g_test_sup_in_source"]
    )


def test_msrlcp_envelope_lemma_and_sanity(
    multisource_conformal: dict[str, float],
) -> None:
    blob = multisource_conformal
    # Lemma 3.1 through the bench's oracle grid constants:
    # max_k ||g_k||_inf == B for the planted bump densities.
    assert blob["msrlcp_envelope_g_source_sup"] == pytest.approx(
        blob["msrlcp_envelope_b"], rel=1e-3
    )
    assert blob["msrlcp_n_eff"] > 0.0
    assert blob["msrlcp_alignment_threshold"] > 0.0
    assert blob["msrlcp_mean_perturb_distance"] > 0.0
    assert 0.0 <= blob["msrlcp_frac_vacuous_in_source"] < 1.0


def test_ghcp_coverage_at_every_m(hierarchical_conformal: dict[str, float]) -> None:
    blob = hierarchical_conformal
    nominal = 1.0 - blob["ghcp_alpha"]
    # Mallick et al. 2026 Thm 2.1: finite-sample coverage at every initial
    # observation count m in {0, 1, 3, 10}; no trivial (+inf) sets at these
    # donor-pool sizes (SHRUNK n_reps = 200 vs the lane's 300).
    for m in (0, 1, 3, 10):
        assert blob[f"ghcp_coverage_m{m}"] >= nominal - 0.04
        assert blob[f"ghcp_trivial_share_m{m}"] == 0.0
    assert blob["ghcp_min_coverage"] >= nominal - 0.04


def test_ghcp_width_shrinks_with_observations(
    hierarchical_conformal: dict[str, float],
) -> None:
    blob = hierarchical_conformal
    # The value of the generalization over first-observation HCP: mean set
    # width shrinks materially as the test group's observed prefix grows.
    widths = [blob[f"ghcp_mean_width_m{m}"] for m in (0, 1, 3, 10)]
    assert widths[3] < 0.75 * widths[0]
    for earlier, later in zip(widths, widths[1:], strict=False):
        assert later <= earlier * 1.05
    assert blob["ghcp_width_shrinks"] == 1.0


def test_voldec_decomposition_flip(vol_loss_decomposition: dict[str, float]) -> None:
    blob = vol_loss_decomposition
    # Tokajuk & Chudziak 2026 central claim on the planted world: raw score
    # variation is loss-dominated, aligned variation is model-dominated.
    assert blob["voldec_flip_loss_to_model"] == 1.0
    assert blob["voldec_ratio_median_raw"] > 2.0  # paper: median 2.91
    assert blob["voldec_ratio_median_aligned"] < 0.5  # paper: median 0.67
    assert blob["voldec_loss_share_raw"] > blob["voldec_model_share_raw"]
    assert blob["voldec_model_share_aligned"] > blob["voldec_loss_share_aligned"]
    # Robustness score (MSE-log) flips the same way.
    assert blob["voldec_ratio_median_mse_log_raw"] > 1.0
    assert blob["voldec_ratio_median_mse_log_aligned"] < 1.0


def test_voldec_level_share_and_breach_narrowing(
    vol_loss_decomposition: dict[str, float],
) -> None:
    blob = vol_loss_decomposition
    # Planted cross-loss differences are mostly LEVEL (paper LevelShare
    # 56-89%), and validation alignment removes most of the cross-loss VaR
    # breach-rate spread (paper: ~97% removed; planted world > 50%).
    assert blob["voldec_level_share_mean"] > 0.6
    assert blob["voldec_breach_spread_aligned_mean"] < blob["voldec_breach_spread_raw_mean"]
    assert blob["voldec_breach_spread_narrowing"] > 0.5
    # Raw pattern mirrors paper Table 4: log-error losses breach most, HMSE
    # most conservative.
    assert (
        blob["voldec_breach_rate_first_loss_raw"]
        > blob["voldec_alpha_var"]
        > blob["voldec_breach_rate_last_loss_raw"]
    )


def test_benches_fail_closed_through_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # {} contract: a failing synthetic setup surfaces as an empty blob,
    # never a partial/exceptional scorecard family.
    def _raise(*args: object, **kwargs: object) -> dict[str, float | str]:
        raise ValueError("planted failure")

    monkeypatch.setattr(benches_w16, "_msrlcp_core_bench", _raise)
    monkeypatch.setattr(benches_w16, "_ghcp_core_bench", _raise)
    monkeypatch.setattr(benches_w16, "_voldec_core_study", _raise)
    assert bench_multisource_conformal() == {}
    assert bench_hierarchical_conformal() == {}
    assert bench_vol_loss_decomposition() == {}


def test_benches_are_deterministic(
    multisource_conformal: dict[str, float],
    hierarchical_conformal: dict[str, float],
    vol_loss_decomposition: dict[str, float],
) -> None:
    # Seeded through the wired modules' own generators, so a fresh call must
    # reproduce each fixture bit-for-bit.
    assert bench_multisource_conformal() == multisource_conformal
    assert bench_hierarchical_conformal() == hierarchical_conformal
    assert bench_vol_loss_decomposition() == vol_loss_decomposition
