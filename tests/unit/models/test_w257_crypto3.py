"""Wave-257 crypto-3 canon tests."""

import numpy as np

from quant_fund.models.bfv_fhe import bench_bfv_fhe
from quant_fund.models.chaum_pedersen import bench_chaum_pedersen, cp_prove, cp_verify
from quant_fund.models.lwe_kex import _lwe_kex, bench_lwe_kex
from quant_fund.models.ntru_toy import _circulant, _mul, bench_ntru_toy
from quant_fund.models.sigma_or_proof import bench_sigma_or_proof, or_proof_prove, or_proof_verify
from quant_fund.models.sis_hash import bench_sis_hash, sis_hash


def test_lwe_agree():
    rng = np.random.RandomState(0)
    assert _lwe_kex(rng)


def test_lwe_bench():
    assert bench_lwe_kex()["synthetic_kex_agreement"] == 1.0


def test_ntru_conv():
    f = np.array([1, 0, 3])
    x = np.array([1, 2, 1])
    M = _circulant(f, 3)
    assert np.all((_mul(f, x, 3) % 7) == (M @ x) % 7)


def test_ntru_bench():
    assert bench_ntru_toy()["synthetic_ntru_roundtrip"] == 1.0


def test_bfv_bench():
    assert bench_bfv_fhe()["synthetic_bfv_add_correct"] == 1.0


def test_sis_deterministic():
    A = np.arange(8).reshape(2, 4) % 7
    x = np.array([1, 0, 1, 0])
    assert np.array_equal(sis_hash(A, x, 7), sis_hash(A, x, 7))


def test_sis_bench():
    assert bench_sis_hash()["synthetic_sis_binding"] == 1.0


def test_or_proof_basic():
    g, h, p = 5, 7, 2**31 - 1
    x = 12345
    y1, y2 = pow(g, x, p), pow(h, 99, p)
    proof = or_proof_prove(g, h, y1, y2, x, 0)
    assert or_proof_verify(g, h, y1, y2, proof)


def test_or_proof_bench():
    assert bench_sigma_or_proof()["synthetic_or_proof_valid"] == 1.0


def test_cp_basic():
    g, h, p = 5, 7, 2**31 - 1
    x = 777
    y, z = pow(g, x, p), pow(h, x, p)
    assert cp_verify(g, h, y, z, cp_prove(g, h, y, z, x))


def test_cp_bench():
    assert bench_chaum_pedersen()["synthetic_cp_valid"] == 1.0
