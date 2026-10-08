from quant_fund.models.cbc_padding import bench_cbc_padding, pad, unpad


def test_roundtrip_empty_and_full_block() -> None:
    assert unpad(pad(b"")) == b""
    assert unpad(pad(b"a" * 16)) == b"a" * 16
    assert unpad(pad(b"abc")) == b"abc"


def test_unpad_rejects_corruption() -> None:
    m = b"hello world"
    p = pad(m)
    for i in range(len(p)):
        bad = p[:i] + bytes([p[i] ^ 0xFF]) + p[i + 1 :]
        out = unpad(bad)
        # a corrupted ciphertext must never silently return m
        assert out is None or out != m


def test_unpad_rejects_malformed() -> None:
    assert unpad(b"") is None
    assert unpad(b"\x00" * 16) is None
    assert unpad(bytes(range(17))) is None
    assert unpad(b"\x11" * 16) is None  # pad length 17 > block


def test_bench_corrupt_no_roundtrip_key() -> None:
    # the old bench computed a vacuous `... or True` counter that was
    # never emitted; the repaired metric must exist and be perfect.
    out = bench_cbc_padding()
    assert out["synthetic_corrupt_no_roundtrip"] == 1.0
    assert out["synthetic_roundtrip"] == 1.0
    assert out["synthetic_bad_pad_reject"] == 1.0
