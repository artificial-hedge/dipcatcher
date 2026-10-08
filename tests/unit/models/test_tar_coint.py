import numpy as np
import pytest

from quant_fund.models.tar_coint import bench_tar_coint, synth_tar, tar_cointegration


def test_bench_tar_coint_passes():
    r = bench_tar_coint()
    assert r["synthetic_score"] == 1.0


def test_tar_detects_asymmetry():
    y, x, _ = synth_tar(seed=9)
    r = tar_cointegration(y, x)
    assert r["rho_above"] < r["rho_below"]


def test_tar_symmetric_control_balanced():
    _, x, y2 = synth_tar(seed=11)
    r = tar_cointegration(y2, x)
    assert abs(r["rho_above"] - r["rho_below"]) < 0.25


def test_tar_asym_tstat_large():
    y, x, _ = synth_tar(seed=4)
    r = tar_cointegration(y, x)
    assert r["asym_t"] > 1.5


def test_tar_rejects_length_mismatch():
    with pytest.raises(ValueError):
        tar_cointegration(np.zeros(300), np.zeros(200))


def test_tar_mtar_mode_runs():
    y, x, _ = synth_tar(seed=2)
    r = tar_cointegration(y, x, mtar=True)
    assert np.isfinite(r["rho_above"])
