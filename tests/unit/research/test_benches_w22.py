"""Wave 22 bench wiring tests.

Every landed family must emit a finite ``dict[str, float]`` that clears the
catalog honesty gates; soft blobs return ``{}`` while their lane module is
absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w22 import (
    bench_signature_martingale_test,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)

_FAMILIES_LANDED = ("signature_martingale_test",)
_LANDED_NUMPY_BLOBS = _FAMILIES_LANDED


@pytest.fixture(scope="module")
def signature_martingale_test() -> dict[str, float]:
    return bench_signature_martingale_test()


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


def test_martingale_size_and_power(signature_martingale_test: dict[str, float]) -> None:
    blob = signature_martingale_test
    # Size near nominal on Brownian nulls (loose MC band at n_mc=48);
    # power strictly increasing across the planted drift grid.
    assert blob["synthetic_size_bm_005"] < 0.25
    assert blob["synthetic_size_bm_boot_005"] < 0.25
    assert (
        blob["synthetic_power_drift_0p05_005"]
        < blob["synthetic_power_drift_0p10_005"]
        < blob["synthetic_power_drift_0p20_005"]
    )


def test_martingale_eprocess_and_alternatives(
    signature_martingale_test: dict[str, float],
) -> None:
    blob = signature_martingale_test
    assert blob["synthetic_ville_rate_null_005"] == 0.0
    assert blob["synthetic_eval_null_median"] < blob["synthetic_eval_drift_hi_median"]
    assert blob["synthetic_power_ar1_phi0p30_005"] > 0.5


def test_benches_deterministic() -> None:
    assert bench_signature_martingale_test() == bench_signature_martingale_test()
