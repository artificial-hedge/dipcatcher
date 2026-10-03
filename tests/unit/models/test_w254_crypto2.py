"""Wave-254 applied-crypto-2 canon tests."""

import hashlib
import hmac as _hmac

from quant_fund.models.aead_etm import bench_aead_etm, open_, seal
from quant_fund.models.cbc_padding import bench_cbc_padding, pad, unpad
from quant_fund.models.hmac_construct import bench_hmac_construct, toy_hmac
from quant_fund.models.merkle_damgard import bench_merkle_damgard, md_hash
from quant_fund.models.pbkdf2_kdf import bench_pbkdf2_kdf, pbkdf2
from quant_fund.models.tls_handshake import Handshake, bench_tls_handshake


def test_tls_establishes():
    c, s = Handshake(b"k" * 32), Handshake(b"k" * 32)
    c.client_hello(["aes"])
    s.server_hello(c)
    s.encrypted_extensions(c)
    s.finished(c)
    assert len(c.traffic_key) == 32


def test_tls_bench():
    assert bench_tls_handshake()["synthetic_establish"] == 1.0


def test_hmac_vs_stdlib():
    assert toy_hmac(b"k", b"m") == _hmac.new(b"k", b"m", hashlib.sha256).digest()


def test_hmac_bench():
    assert bench_hmac_construct()["synthetic_hmac_matches_stdlib"] == 1.0


def test_aead_roundtrip():
    b = seal(b"e" * 32, b"m" * 32, b"n" * 8, b"hello", b"ad")
    assert open_(b"e" * 32, b"m" * 32, b, b"ad") == b"hello"
    assert open_(b"e" * 32, b"m" * 32, b, b"ad2") is None


def test_aead_bench():
    assert bench_aead_etm()["synthetic_tamper_reject"] == 1.0


def test_md_deterministic():
    assert md_hash(b"abc") == md_hash(b"abc") != md_hash(b"abd")


def test_md_bench():
    assert bench_merkle_damgard()["synthetic_length_ext_demo"] == 1.0


def test_padding_roundtrip():
    p = pad(b"xyz")
    assert unpad(p) == b"xyz"
    assert unpad(p[:-1] + b"\x99") is None


def test_padding_bench():
    assert bench_cbc_padding()["synthetic_roundtrip"] == 1.0


def test_pbkdf2_vs_stdlib():
    assert pbkdf2(b"pw", b"salt", 10) == hashlib.pbkdf2_hmac("sha256", b"pw", b"salt", 10)


def test_pbkdf2_bench():
    assert bench_pbkdf2_kdf()["synthetic_matches_stdlib"] == 1.0
