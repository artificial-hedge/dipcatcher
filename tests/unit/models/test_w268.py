"""Wave-268 crypto-4 module tests."""

import numpy as np

from quant_fund.models.chacha_stream import block
from quant_fund.models.elgamal_enc import dec, enc, keygen
from quant_fund.models.fiat_shamir import prove, verify
from quant_fund.models.ot_12 import ot_12
from quant_fund.models.paillier_he import _N, _NN, _P, _Q, _lam, paillier_dec, paillier_enc
from quant_fund.models.poly1305_mac import poly1305


def test_elgamal_roundtrip() -> None:
    rng = np.random.RandomState(0)
    sk, pk = keygen(rng)
    c = enc(pk, 42, 7)
    assert dec(sk, c) == 42


def test_elgamal_homomorphic() -> None:
    rng = np.random.RandomState(1)
    sk, pk = keygen(rng)
    c1 = enc(pk, 6, 11)
    c2 = enc(pk, 7, 13)
    prod = (c1[0] * c2[0] % 104729, c1[1] * c2[1] % 104729)
    assert dec(sk, prod) == 42


def test_paillier_add() -> None:
    lam = _lam(_P, _Q)
    c = paillier_enc(100, 12345) * paillier_enc(23, 54321) % _NN
    assert paillier_dec(c, lam) == 123 % _N


def test_fiat_shamir_verify() -> None:
    y, a, z = prove(99, 12345)
    assert verify(y, a, z)


def test_ot_correct_choice() -> None:
    rng = np.random.RandomState(2)
    got, _ = ot_12(111, 222, 1, rng)
    assert got == 222


def test_chacha_deterministic() -> None:
    key = np.arange(8, dtype=np.uint32)
    assert np.array_equal(block(key, 0, 0), block(key, 0, 0))
    assert not np.array_equal(block(key, 0, 0), block(key, 1, 0))


def test_poly1305_detects_tamper() -> None:
    t = poly1305(b"hello world", 7, 13)
    assert t != poly1305(b"hello worle", 7, 13)
    assert t == poly1305(b"hello world", 7, 13)
