"""Tests for research/benches_w13.py — SOTA canon wave 13 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): every
value is deterministic (seeded from a module constant), so re-running a bench
per test buys nothing; the determinism test at the bottom re-runs the whole
battery once (it is cheap — the wave-13 budgets are shrunk) and compares
bit-for-bit.

Tolerance policy: the wave-13 benches run SHRUNK Monte-Carlo budgets relative
to the lane suites in tests/unit/core and tests/unit/models (e.g. 30 z-test
replicates instead of 100-200, 16 changepoint replicates instead of 16 at a
larger horizon and full method grid, 2 reference-null banks x B=199 instead of
4-8 banks x 299-499, 20 leaderboard replicates instead of 40-100). The science
assertions below are therefore DIRECTIONAL with wider documented slack than
the lane tests — they pin the qualitative claims of the cited papers (variance
inflation under dependence, log-vs-capped detection delay, the universal
1 - 2*alpha rolling floor, anytime error control, margin-based certification,
replicability agreement, calibrated-vs-Ville thresholds, interval-score
ordering by tau/L) without asserting the lane suites' tight MC windows.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w13 import (
    bench_conformal_e_detectors,
    bench_coverage_inference,
    bench_delayed_aci,
    bench_picpi,
    bench_rank_cs,
    bench_reference_null,
    bench_replicable_conformal,
    bench_rolling_conformal,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)

_FAMILIES = (
    "coverage_inference",
    "conformal_e_detectors",
    "rolling_conformal",
    "rank_cs",
    "picpi",
    "replicable_conformal",
    "reference_null",
    "delayed_aci",
)


@pytest.fixture(scope="module")
def coverage_inference() -> dict[str, float]:
    return bench_coverage_inference()


@pytest.fixture(scope="module")
def conformal_e_detectors() -> dict[str, float]:
    return bench_conformal_e_detectors()


@pytest.fixture(scope="module")
def rolling_conformal() -> dict[str, float]:
    return bench_rolling_conformal()


@pytest.fixture(scope="module")
def rank_cs() -> dict[str, float]:
    return bench_rank_cs()


@pytest.fixture(scope="module")
def picpi() -> dict[str, float]:
    return bench_picpi()


@pytest.fixture(scope="module")
def replicable_conformal() -> dict[str, float]:
    return bench_replicable_conformal()


@pytest.fixture(scope="module")
def reference_null() -> dict[str, float]:
    return bench_reference_null()


@pytest.fixture(scope="module")
def delayed_aci() -> dict[str, float]:
    return bench_delayed_aci()


def test_families_registered_as_optional() -> None:
    for fam in _FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_FAMILIES))
def test_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_coverage_inference_dependence_and_ztest(coverage_inference: dict[str, float]) -> None:
    # AR(1) phi=0.8 inflates the long-run variance of the coverage indicators
    # versus i.i.d. (their Sec. 3): the block-SE^2 ratio must be clearly > 1.
    # Single seeded stream pair (lane suite averages 200 reps), hence the wide
    # margin below the realized ratio (~3.8 on this seed).
    assert coverage_inference["covinf_ar1_se_ratio_vs_iid"] > 2.0
    # Scaled-t z-test size at eta = 0.10 under the AR(1) phi=0.8 null, 30 reps:
    # the lane suite permits up to 0.16 for the t reference at this dependence
    # and n=1500 (200 reps); at 30 reps binomial noise (1 sd ~ 0.055) widens
    # the honest ceiling to eta + 0.10 (realized 0.167).
    eta = coverage_inference["covinf_eta"]
    assert 0.0 <= coverage_inference["covinf_ztest_size"] <= eta + 0.10
    # A x1.5-inflated test window is decisively miscalibrated: power near 1.
    assert coverage_inference["covinf_ztest_power"] >= 0.80
    # i.i.d. anchor: the block SE tracks the binomial SE (lane window
    # (0.85, 1.08) at 200 reps; 30 reps justify the wider band).
    assert 0.80 <= coverage_inference["covinf_iid_se_binomial_ratio"] <= 1.15


def test_conformal_e_detectors_delay_and_fa(conformal_e_detectors: dict[str, float]) -> None:
    alpha = conformal_e_detectors["ced_alpha"]
    # The restarted e-process detects far earlier than the Vovk CTM at the
    # SAME Ville threshold 1/alpha (their Example 2.2: CTM delay is Omega(T);
    # at T=300 with 16 reps the CTM median sits at the cap, so only the
    # conservative spec floor of 2x is asserted).
    assert conformal_e_detectors["ced_delay_optimal"] >= 1.0
    assert conformal_e_detectors["ced_delay_vovk"] > conformal_e_detectors["ced_delay_optimal"]
    assert conformal_e_detectors["ced_delay_ratio"] >= 2.0
    # Ville validity: pre-change false-alarm rate <= alpha (+ 1-rep slack at
    # the shrunk 16-rep budget, where one false alarm is 1/16 = 0.0625).
    assert conformal_e_detectors["ced_fa_rate"] <= alpha + 0.02
    assert conformal_e_detectors["ced_detection_rate_optimal"] >= 0.5


def test_rolling_conformal_guarantees(rolling_conformal: dict[str, float]) -> None:
    alpha = rolling_conformal["rcp_alpha"]
    nominal = rolling_conformal["rcp_nominal"]
    # STABLE regime: coverage approaches 1 - alpha (their Thm. 4 territory);
    # 24 streams x 40 holdout draws (lane: 120 x 40) justify the +/- 0.05 window.
    assert abs(rolling_conformal["rcp_marginal_coverage"] - nominal) <= 0.05
    # ADVERSARIAL fold (their Prop. 1): the universal 1 - 2*alpha floor HOLDS
    # (flag), while coverage sits strictly BELOW nominal — a 1 - alpha claim
    # without stability is false there.
    assert rolling_conformal["rcp_worst_case_floor_holds"] == 1.0
    assert rolling_conformal["rcp_worst_case_coverage"] >= (1.0 - 2.0 * alpha) - 0.02
    assert rolling_conformal["rcp_worst_case_coverage"] < nominal
    # Rolling sets are narrower than the frozen split model's (their Fig. 2).
    assert rolling_conformal["rcp_vs_split_width_ratio"] < 1.0
    # The naive arrival-time split breaks below the floor under an unstable
    # predictor (no guarantee exists for it): 40 streams x 30 draws put the
    # realized mean ~0.085 below 1 - 2*alpha.
    assert rolling_conformal["rcp_naive_split_undercoverage"] < (1.0 - 2.0 * alpha) - 0.02


def test_rank_cs_anytime_error_control(rank_cs: dict[str, float]) -> None:
    alpha = rank_cs["rcs_alpha"]
    # Time-uniform rank coverage over 20 reps: the lane suite tolerates
    # alpha + 0.02 violations at 100 reps; at 20 reps one violation is 0.05,
    # so the honest shrunk-budget floor is 0.95 (realized: 1.0).
    assert rank_cs["rcs_time_uniform_rank_coverage"] >= 0.95
    # Dominant-model efficiency: rank-1 recovered at T in most reps, and the
    # interval profile stabilizes well before the horizon (lane: >= 0.9 and
    # <= 0.6 at 40 reps; shrunk to 20).
    assert rank_cs["rcs_dominant_rank1_recovery"] >= 0.8
    assert rank_cs["rcs_median_stabilization_time"] <= 0.6
    # BB-EDGE anytime FWER under a fully dependent global null: 0.0 expected;
    # 20 runs give 0.05 granularity, hence the alpha + 0.02 ceiling.
    assert rank_cs["bbedge_fwer_dependent"] <= alpha + 0.02
    assert rank_cs["bbedge_topk_certification_rate"] >= 0.8


def test_picpi_selfconsistency_and_width_rate(picpi: dict[str, float]) -> None:
    # Algorithm 1's margin screen: no held-out self-consistency violation on
    # the calibrated fixture, and none on the adversarial spike fixture —
    # while the margin-free naive bin IS violated there (the honest contrast).
    assert picpi["picpi_selfconsistency_n_checked"] > 0.0
    assert picpi["picpi_selfconsistency_violation_rate"] == 0.0
    assert picpi["picpi_adversarial_violation_rate"] == 0.0
    assert picpi["picpi_naive_violation_rate"] > 0.0
    # The calibrated fixture certifies (essentially) all prediction mass.
    assert picpi["picpi_certified_fraction"] >= 0.9
    # Theorem 3.2's n^{-1/3} width rate on the SHRUNK documented n-grid
    # subset (10k/20k/40k instead of 10k-640k): slope band from the lane
    # suite, widths strictly decreasing across the subset.
    assert -0.55 <= picpi["picpi_width_slope"] <= -0.18
    assert picpi["picpi_width_last"] < picpi["picpi_width_first"]


def test_replicable_conformal_agreement_and_gaming(
    replicable_conformal: dict[str, float],
) -> None:
    nominal = replicable_conformal["repcon_nominal"]
    # Prop. 2's rho-replicability contract at rho = 0.1: 40 analyst pairs give
    # 0.025 granularity, so the spec floor 1 - rho is asserted directly
    # (realized 0.95 at n = 12k).
    assert replicable_conformal["repcon_agreement_rate"] >= 1.0 - replicable_conformal["repcon_rho"]
    assert replicable_conformal["repcon_agreement_rate"] >= 0.90
    # Thm. 2(ii): rounding UP never loses marginal coverage (realized sits
    # ABOVE nominal by ~beta/2 — the measured price of replicability).
    assert replicable_conformal["repcon_coverage"] >= nominal - 0.01
    assert replicable_conformal["repcon_coverage_standard"] >= nominal - 0.01
    # Cor. 4: replicability is never free, and at this budget not ruinous.
    assert 1.0 < replicable_conformal["repcon_size_cost_ratio"] < 1.15
    # Cor. 3 selection attack: min-of-M recalibration silently pushes standard
    # split CP BELOW nominal, while every ReCal candidate stays valid.
    assert replicable_conformal["repcon_gaming_undercover_standard"] < nominal - 0.002
    assert replicable_conformal["repcon_gaming_selected_recal"] >= nominal
    assert replicable_conformal["repcon_gaming_stability"] >= 0.5


def test_reference_null_type_i_and_delay_gain(reference_null: dict[str, float]) -> None:
    alpha = reference_null["rnc_alpha"]
    # Thm. 2.2 / Ville: both boundaries control the finite-horizon type-I
    # error at alpha (+ 0.02 binomial slack; 2 banks x B=199 and 160 eval
    # paths are far below the lane suite's 4-8 banks x 299-499 / 240-500, so
    # the bank-averaged rate carries the wider documented slack).
    assert reference_null["rnc_fa_rate_calibrated"] <= alpha + 0.02
    assert reference_null["rnc_fa_rate_ville"] <= alpha + 0.02
    # Sec. 5.1's paired design: the calibrated boundary is sharper than Ville
    # on this bank, so the RMDD ratio on the SAME planted-shift paths is < 1.
    assert 0.0 < reference_null["rnc_threshold_sharpness"] < 1.0
    assert reference_null["rnc_delay_threshold_ratio"] < 1.0
    assert reference_null["rnc_delay_gain_ratio"] < 1.0


def test_delayed_aci_exactness_bound_and_ordering(delayed_aci: dict[str, float]) -> None:
    nominal = 1.0 - delayed_aci["daci_alpha"]
    # tau = 1 reproduces the repo's AdaptiveConformal level path bit-for-bit
    # (Gibbs & Candes 2021 reduction), and Eq. (11)'s long-run bound holds on
    # every seeded run of both cells.
    assert delayed_aci["daci_tau1_matches_aci"] == 1.0
    assert delayed_aci["daci_bound_holds"] == 1.0
    # The delay-to-memory ratio orders the cells (r_low < r_high) and the
    # proper interval score orders WITH it: the tau = 6 band pays more than
    # the tau = 1 band on the same AR(1)-drift law (their Sec. 6 mechanism).
    assert delayed_aci["daci_r_low"] < delayed_aci["daci_r_high"]
    assert delayed_aci["daci_is_lowr"] < delayed_aci["daci_is_highr"]
    # Coverage sanity at both delays: time-average coverage stays near nominal
    # (50 seeds x ~1300 construction steps per cell; +/- 0.05 window).
    assert abs(delayed_aci["daci_coverage_lowr"] - nominal) <= 0.05
    assert abs(delayed_aci["daci_coverage_highr"] - nominal) <= 0.05


def test_benches_are_deterministic() -> None:
    # The whole wave-13 battery is cheap (shrunk budgets, ~3s total), so every
    # bench is re-run once here: seeded from module constants, the repeated
    # calls must be bit-identical.
    assert bench_coverage_inference() == bench_coverage_inference()
    assert bench_conformal_e_detectors() == bench_conformal_e_detectors()
    assert bench_rolling_conformal() == bench_rolling_conformal()
    assert bench_rank_cs() == bench_rank_cs()
    assert bench_picpi() == bench_picpi()
    assert bench_replicable_conformal() == bench_replicable_conformal()
    assert bench_reference_null() == bench_reference_null()
    assert bench_delayed_aci() == bench_delayed_aci()
