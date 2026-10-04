"""Wave-1023 finance-theory canon tests."""

from __future__ import annotations

from quant_fund.models.arbitrage_pricing import bench_arbitrage_pricing
from quant_fund.models.black_scholes import bench_black_scholes
from quant_fund.models.capm_model import bench_capm_model
from quant_fund.models.corporate_finance import bench_corporate_finance
from quant_fund.models.default_risk import bench_default_risk
from quant_fund.models.yield_curve import bench_yield_curve


def test_capm_model():
    assert bench_capm_model()["synthetic_capm_model"] == 1.0


def test_arbitrage_pricing():
    assert bench_arbitrage_pricing()["synthetic_arbitrage_pricing"] == 1.0


def test_black_scholes():
    assert bench_black_scholes()["synthetic_black_scholes"] == 1.0


def test_yield_curve():
    assert bench_yield_curve()["synthetic_yield_curve"] == 1.0


def test_default_risk():
    assert bench_default_risk()["synthetic_default_risk"] == 1.0


def test_corporate_finance():
    assert bench_corporate_finance()["synthetic_corporate_finance"] == 1.0
