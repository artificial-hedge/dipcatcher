"""Wave 21 bench wiring tests.

Every landed family must emit a finite ``dict[str, float]`` that clears the
catalog honesty gates; soft blobs return ``{}`` while their lane module is
absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w21 import (
    bench_bocpd_changepoint,
    bench_rough_heston_rbergomi,
    bench_signature_features,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)

_FAMILIES_LANDED = (
    "bocpd_changepoint",
    "rough_heston_rbergomi",
    "signature_features",
)
_LANDED_NUMPY_BLOBS = _FAMILIES_LANDED


@pytest.fixture(scope="module")
def bocpd_changepoint() -> dict[str, float]:
    return bench_bocpd_changepoint()


@pytest.fixture(scope="module")
def rough_heston_rbergomi() -> dict[str, float]:
    return bench_rough_heston_rbergomi()


@pytest.fixture(scope="module")
def signature_features() -> dict[str, float]:
    return bench_signature_features()


def test_landed_families_registered_as_optional() -> None:
    for fam in _FAMILIES_LANDED:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_LANDED_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


# -- science assertions -----------------------------------------------------


def test_bocpd_detects_planted_breaks(bocpd_changepoint: dict[str, float]) -> None:
    blob = bocpd_changepoint
    assert blob["synthetic_precision"] > 0.4
    assert blob["synthetic_recall"] > 0.4
    assert blob["synthetic_detection_delay_mean"] < 15.0


def test_bocpd_beats_nochange_and_tracks_oracle(
    bocpd_changepoint: dict[str, float],
) -> None:
    blob = bocpd_changepoint
    assert blob["synthetic_beats_nochange"] == 1.0
    # Oracle resets at true boundaries: BOCPD must stay within a small gap.
    assert 0.0 <= blob["synthetic_oracle_gap"] < 0.35


def test_rbergomi_signatures(rough_heston_rbergomi: dict[str, float]) -> None:
    blob = rough_heston_rbergomi
    assert blob["synthetic_rbergomi_lev_corr_short"] < -0.25
    assert 0.0 < blob["synthetic_rbergomi_hurst_recovered_short"] < 0.3
    assert 0.0 < blob["synthetic_rbergomi_hurst_recovered_long"] < 0.3


def test_rheston_machinery(rough_heston_rbergomi: dict[str, float]) -> None:
    blob = rough_heston_rbergomi
    assert blob["synthetic_riccati_classical_rel_err"] < 1e-2
    assert blob["synthetic_cf_mc_gap_in_se"] < 6.0
    assert 0.7 < blob["synthetic_fbm_var_ratio_to_theory"] < 1.05
    assert blob["synthetic_rheston_cf_call"] > 0.0


def test_signature_kernel_machinery(signature_features: dict[str, float]) -> None:
    blob = signature_features
    # Goursat-PDE kernel tracks the order-6 truncated-signature inner
    # product; dyadic refinement converges.
    assert blob["synthetic_pde_vs_truncated_abs_gap"] < 1e-2
    assert blob["synthetic_pde_refine_gap_l12"] < blob["synthetic_pde_refine_gap_l01"]
    assert blob["synthetic_gram_symmetry_err"] < 1e-9


def test_signature_discrimination(signature_features: dict[str, float]) -> None:
    blob = signature_features
    # MMD separates GBM from mean-reverting OU; log-sig features carry
    # drift information (out-of-sample R^2).
    assert blob["synthetic_mmd_power_gbm_vs_ou"] >= 0.5
    assert blob["synthetic_mmd_fp_gbm_vs_gbm"] < 0.5
    assert blob["synthetic_logsig_drift_r2_test"] > 0.3


def test_benches_deterministic() -> None:
    assert bench_bocpd_changepoint() == bench_bocpd_changepoint()
    assert bench_rough_heston_rbergomi() == bench_rough_heston_rbergomi()
    assert bench_signature_features() == bench_signature_features()
