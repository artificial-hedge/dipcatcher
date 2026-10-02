from quant_fund.models.dfpn import bench_dfpn, dfpn


def test_dfpn_prove():
    proved, calls = dfpn(9)
    assert proved is True
    assert calls > 0


def test_dfpn_disprove():
    proved, _ = dfpn(8)
    assert proved is False


def test_bench():
    out = bench_dfpn(seed=6)
    assert out["synthetic_dfpn_proved_31"] == 1.0
    assert out["synthetic_dfpn_disproved_32"] == 1.0
