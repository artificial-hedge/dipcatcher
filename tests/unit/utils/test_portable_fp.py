"""Pinned float kernels stay on the sealed-receipt bits under a Haswell host."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quant_fund.utils.portable_fp import stable_dot, stable_exp, stable_pow

ROOT = Path(__file__).resolve().parents[3]

# Capacity ADV exp. libm rounds the other way by one ulp.
_EXP_X = float.fromhex("0x1.d110e00e6687ap+3")
_EXP_PIN = float.fromhex("0x1.f477d34ed63dap+20")

# Rough-vol kernel base whose AVX-512 pow is one ulp from libm.
_POW_PIN = float.fromhex("0x1.f3bbc0c2ac01cp-3")

# Length-4 HAR forecast dot. Haswell ``@`` lands on the neighbouring ulp.
_DOT_COEF = np.array(
    [
        float.fromhex("0x1.7e97440b976c5p-12"),
        float.fromhex("0x1.1414d445b5d2cp-3"),
        float.fromhex("0x1.166eedb1dcf3ap-1"),
        float.fromhex("-0x1.a81482eac10b7p+0"),
    ]
)
_DOT_X = np.array(
    [
        float.fromhex("0x1.0000000000000p+0"),
        float.fromhex("0x1.38bf9a3363578p-13"),
        float.fromhex("0x1.0367cf1dcfd2bp-13"),
        float.fromhex("0x1.8c69db23c3b0cp-13"),
    ]
)
_DOT_PIN = float.fromhex("0x1.23ba03e751feep-13")

_LANES = {
    "capacity": (
        [
            "capacity",
            "--dev",
            "--books",
            "uniform,concentrated",
            "--n-dates",
            "60",
            "--n-names",
            "8",
            "--seed",
            "11",
            "--aum-grid",
            "1e6,1e7",
            "--out-dir",
            "data/metadata/replay",
        ],
        "capacity_eval_f990aaadaa5b647b.json",
    ),
    "rankic": (
        [
            "rankic",
            "--panels",
            "pure_noise,linear_signal",
            "--challengers",
            "identity,noisy,lagged",
            "--n-assets",
            "16",
            "--n-dates",
            "60",
            "--horizons",
            "1,5",
            "--seed",
            "11",
            "--out-dir",
            "data/metadata/replay",
        ],
        "rankic_eval_507c328d7ea41c76.json",
    ),
    "vol": (
        [
            "vol-bench",
            "--models",
            "rv_roll,rv_ewma,har",
            "--shards",
            "garch_vol,rough_vol",
            "--horizons",
            "1",
            "--min-history",
            "60",
            "--n-origins",
            "10",
            "--n-bars",
            "200",
            "--seed",
            "7",
            "--out-dir",
            "data/metadata/replay",
        ],
        "vol_bench_c770505112aeabce.json",
    ),
}


def test_stable_exp_matches_pinned_kernel_not_libm() -> None:
    import math

    assert stable_exp(_EXP_X) == _EXP_PIN
    assert math.exp(_EXP_X) != _EXP_PIN


def test_stable_pow_matches_pinned_kernel_not_libm() -> None:
    import math

    assert stable_pow(34.0, 0.1 - 0.5) == _POW_PIN
    assert math.pow(34.0, 0.1 - 0.5) != _POW_PIN


def test_stable_dot_matches_pinned_har_forecast_dot() -> None:
    assert stable_dot(_DOT_COEF, _DOT_X) == _DOT_PIN


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_lane_replay_pins_under_haswell_avx2(lane: str, tmp_path: Path) -> None:
    """GitHub-hosted runners dispatch like AVX2 + Haswell/Zen OpenBLAS.

    The sealed replay files were written by the AVX-512 / SkylakeX kernels.
    Forcing that dispatch here must still emit the pinned names.
    """
    argv, pinned = _LANES[lane]
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    out = tmp_path / "data" / "metadata" / "replay"
    out.mkdir(parents=True)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OPENBLAS_CORETYPE": "HASWELL",
        "NPY_ENABLE_CPU_FEATURES": "AVX2",
    }
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; from quant_fund.cli.main import app; app()", *argv],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    written = sorted(path.name for path in out.glob("*.json"))
    assert written == [pinned]
