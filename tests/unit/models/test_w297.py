"""Unit tests for wave-297 post-quantum crypto canon modules."""

import numpy as np

from quant_fund.models.dilithium_sig import keygen as d_keygen
from quant_fund.models.dilithium_sig import sign, verify
from quant_fund.models.frodokem import decap as f_decap
from quant_fund.models.frodokem import encap as f_encap
from quant_fund.models.frodokem import keygen as f_keygen
from quant_fund.models.kyber_kem import decap, encap
from quant_fund.models.kyber_kem import keygen as k_keygen
from quant_fund.models.ntt_ring import N, Q, intt, ntt, ntt_mul
from quant_fund.models.sphincs_sig import sphincs_keygen, sphincs_sign, sphincs_verify
from quant_fund.models.xmss_sig import xmss_keygen, xmss_sign, xmss_verify


def test_ntt_roundtrip():
    rng = np.random.default_rng(0)
    a = rng.integers(0, Q, N)
    assert np.array_equal(intt(ntt(a)), a)


def test_ntt_mul_commutes():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, Q, N), rng.integers(0, Q, N)
    assert np.array_equal(ntt_mul(a, b), ntt_mul(b, a))


def test_kyber_roundtrip():
    pk, sk = k_keygen(b"seedseedseedseed")
    c, ss = encap(pk, b"randrdrdrdrdrdrd")
    assert decap(sk, c) == ss


def test_dilithium():
    rng = np.random.default_rng(2)
    pk, sk = d_keygen(b"seedseedseedseed")
    sig = sign(sk, pk, b"hello", rng)
    assert sig is not None and verify(pk, b"hello", sig)
    assert not verify(pk, b"bye", sig)


def test_frodo_roundtrip():
    pk, sk = f_keygen(b"seedseedseedseed")
    ct, ss = f_encap(pk, b"randrdrdrdrdrdrd")
    assert f_decap(sk, ct) == ss


def test_xmss():
    root, _ = xmss_keygen(b"seedseedseedseed")
    sig = xmss_sign(b"msg", b"seedseedseedseed", 0)
    assert xmss_verify(b"msg", sig, 0, root, b"seedseedseedseed")
    assert not xmss_verify(b"msg", sig, 1, root, b"seedseedseedseed")


def test_sphincs():
    pk = sphincs_keygen(b"seedseedseedseed")
    sig = sphincs_sign(b"msg", b"seedseedseedseed", 0)
    assert sphincs_verify(b"msg", sig, pk, b"seedseedseedseed")
    assert not sphincs_verify(b"bad", sig, pk, b"seedseedseedseed")
