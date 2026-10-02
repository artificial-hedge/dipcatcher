"""Wave-222 number-theory canon tests."""

from __future__ import annotations

from quant_fund.models.continued_fraction import bench_continued_fraction
from quant_fund.models.crt_garner import bench_crt_garner
from quant_fund.models.ec_scalar import bench_ec_scalar
from quant_fund.models.miller_rabin import bench_miller_rabin
from quant_fund.models.pollard_rho import bench_pollard_rho
from quant_fund.models.tonelli_shanks import bench_tonelli_shanks

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestMillerRabin:
    def test_bench(self) -> None:
        out = bench_miller_rabin()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_carmichael_caught"] == 1.0


class TestPollardRho:
    def test_bench(self) -> None:
        out = bench_pollard_rho()
        _clean(out)
        assert out["synthetic_recovered"] == 1.0
        assert out["synthetic_agree"] >= 0.9


class TestTonelli:
    def test_bench(self) -> None:
        out = bench_tonelli_shanks()
        _clean(out)
        assert out["synthetic_agree"] == 1.0


class TestContFrac:
    def test_bench(self) -> None:
        out = bench_continued_fraction()
        _clean(out)
        assert out["synthetic_pell_ok"] == 1.0
        assert out["synthetic_pell23"] == 1.0


class TestCRT:
    def test_bench(self) -> None:
        out = bench_crt_garner()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_known_23"] == 1.0


class TestECScalar:
    def test_bench(self) -> None:
        out = bench_ec_scalar()
        _clean(out)
        assert out["synthetic_order_cycle"] == 1.0
        assert out["synthetic_homomorphism"] == 1.0
        assert out["synthetic_on_curve"] == 1.0
