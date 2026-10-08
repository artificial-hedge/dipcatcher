"""Tests for the Vasicek ASRF credit-loss distribution (models/vasicek_credit.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models import vasicek_credit as vc


def test_cdf_of_var_returns_q() -> None:
    """Consistency identity: CDF(VaR(q)) must equal q exactly."""
    for q in (0.5, 0.9, 0.99, 0.999):
        x = vc.vasicek_var(q, pd=0.02, rho=0.12)
        got = float(vc.vasicek_loss_cdf(np.array([x]), pd=0.02, rho=0.12)[0])
        assert got == pytest.approx(q, abs=1e-9)


def test_var_and_es_monotone_in_q() -> None:
    pd_, rho = 0.03, 0.15
    assert vc.vasicek_var(0.99, pd_, rho) > vc.vasicek_var(0.95, pd_, rho)
    assert vc.vasicek_es(0.99, pd_, rho) > vc.vasicek_var(0.99, pd_, rho)


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(1e-6, 0.999, 4000)
    pdf = vc.vasicek_loss_pdf(x, pd=0.02, rho=0.1)
    assert float(np.trapezoid(pdf, x)) == pytest.approx(1.0, abs=5e-3)


def test_param_guards() -> None:
    with pytest.raises(ValueError):
        vc.vasicek_var(0.99, pd=0.0, rho=0.1)
    with pytest.raises(ValueError):
        vc.vasicek_var(0.99, pd=0.02, rho=1.0)
    with pytest.raises(ValueError):
        vc.vasicek_es(1.0, pd=0.02, rho=0.1)
