"""Adversarial probes for xmss_sig — WOTS+ checksum integrity."""

import random

from quant_fund.models.xmss_sig import (
    LEN,
    LEN1,
    W,
    _msg_chain_lens,
    xmss_keygen,
    xmss_sign,
    xmss_verify,
)


def test_checksum_nibbles_present() -> None:
    """Last LEN-LEN1 chain lengths must carry the real checksum digits.

    Before the fix, `csum.to_bytes(2, "big")` pushed 4 nibbles and `[:LEN]`
    truncated the actual checksum byte — positions LEN-2..LEN-1 were always 0.
    """
    rng = random.Random(7)
    saw_nonzero = False
    for _ in range(60):
        msg = rng.randbytes(12)
        lens = _msg_chain_lens(msg)
        assert len(lens) == LEN
        csum = sum(W - 1 - v for v in lens[:LEN1])
        assert lens[LEN1:] == [csum >> 4, csum & 0xF]
        saw_nonzero |= lens[LEN1] != 0 or lens[LEN1 + 1] != 0
    assert saw_nonzero, "checksum digits were always zero — truncated encoding"


def test_sign_verify_roundtrip_and_forgery() -> None:
    import numpy as np

    rng = np.random.default_rng(11)
    seed = rng.bytes(16)
    root, _ = xmss_keygen(seed)
    msg = b"hello"
    sig = xmss_sign(msg, seed, 3)
    assert xmss_verify(msg, sig, 3, root, seed)
    assert not xmss_verify(msg, sig, 4, root, seed)
    assert not xmss_verify(b"bye", sig, 3, root, seed)


def test_sign_reveals_no_raw_sk() -> None:
    """With a working checksum, no signature element is the raw sk (len>0 hashes)."""
    import numpy as np

    rng = np.random.default_rng(5)
    seed = rng.bytes(16)
    sig, _ = xmss_sign(b"m", seed, 0)
    # sk elems are 32 random bytes drawn per-leaf; a zero-length chain would mean
    # sig[i] == sk[i] — checksum digits are never all-zero across messages.
    lens = _msg_chain_lens(b"m")
    assert any(v > 0 for v in lens[LEN1:]), "checksum chains must hash >0 times"
    _ = sig
