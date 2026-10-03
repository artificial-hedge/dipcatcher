"""Wave 23 bench wiring tests.

Every landed family must emit a finite ``dict[str, float]`` that clears the
catalog honesty gates; soft blobs return ``{}`` while their lane module is
absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w23 import (
    bench_koopman_edmd,
    bench_neural_tpp,
    bench_propagator_impact,
    bench_queue_reactive,
    bench_sig_gan,
    bench_svi_surface,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)

_FAMILIES_LANDED = (
    "svi_surface",
    "propagator_impact",
    "queue_reactive",
    "koopman_edmd",
    "sig_gan",
    "neural_tpp",
)
_LANDED_NUMPY_BLOBS = _FAMILIES_LANDED


@pytest.fixture(scope="module")
def svi_surface() -> dict[str, float]:
    return bench_svi_surface()


@pytest.fixture(scope="module")
def propagator_impact() -> dict[str, float]:
    return bench_propagator_impact()


@pytest.fixture(scope="module")
def queue_reactive() -> dict[str, float]:
    return bench_queue_reactive()


@pytest.fixture(scope="module")
def koopman_edmd() -> dict[str, float]:
    return bench_koopman_edmd()


@pytest.fixture(scope="module")
def sig_gan() -> dict[str, float]:
    return bench_sig_gan()


@pytest.fixture(scope="module")
def neural_tpp() -> dict[str, float]:
    return bench_neural_tpp()


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


def test_svi_calibration_and_arb_detection(svi_surface: dict[str, float]) -> None:
    blob = svi_surface
    # exact SVI slice recalibrates to ~machine precision; arb detection
    # near-perfect on planted violating surfaces.
    assert blob["synthetic_calib_rmse_exact"] < 1e-6
    assert blob["synthetic_arb_f1"] > 0.9
    assert blob["synthetic_calendar_detection"] > 0.9


def test_propagator_kernel_recovery(propagator_impact: dict[str, float]) -> None:
    blob = propagator_impact
    assert blob["synthetic_kernel_recovery_relerr"] < 0.5
    assert blob["synthetic_determinism"] == 1.0


def test_queue_reactive_stationary_law(queue_reactive: dict[str, float]) -> None:
    blob = queue_reactive
    assert blob["synthetic_stationary_tv"] < 0.3
    assert blob["synthetic_determinism"] == 1.0


def test_koopman_spectrum_recovery(koopman_edmd: dict[str, float]) -> None:
    blob = koopman_edmd
    assert blob["synthetic_operator_residual"] < 0.5
    assert blob["synthetic_determinism_delta"] == 0.0


def test_sig_gan_distance_reduces(sig_gan: dict[str, float]) -> None:
    blob = sig_gan
    # signature distance strictly decreases after training on GBM target
    assert blob["synthetic_gbm_sig_distance_after"] < blob["synthetic_gbm_sig_distance_before"]
    assert blob["synthetic_determinism_delta"] == 0.0


def test_neural_tpp_fallback_recovery(neural_tpp: dict[str, float]) -> None:
    blob = neural_tpp
    assert blob["synthetic_hawkes_mu_relerr"] < 0.5
    assert blob["synthetic_determinism"] == 1.0
