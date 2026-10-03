"""Wave-278 econ-models-2 module tests."""

import numpy as np

from quant_fund.models.cobweb_model import cobweb_run
from quant_fund.models.nk_phillips import nkpc_irf
from quant_fund.models.olg_model import olg_path
from quant_fund.models.solow_model import solow_sim, solow_steady
from quant_fund.models.taylor_rule import taylor_sim


def test_nkpc_zero_gap() -> None:
    pi = nkpc_irf(0.99, 0.1, np.zeros(10))
    np.testing.assert_allclose(pi, 0.0)


def test_nkpc_terminal() -> None:
    pi = nkpc_irf(0.99, 0.1, np.ones(10))
    assert pi[-1] == 0.0


def test_taylor_damps() -> None:
    out = taylor_sim(1.5, 0.5, 100, 0.05)
    assert abs(out[-1]) < abs(out[0])


def test_solow_steady_zero_growth() -> None:
    k_star = solow_steady(0.2, 0.33, 0.08)
    path = solow_sim(k_star, 0.2, 0.33, 0.08, 10)
    np.testing.assert_allclose(path, k_star, rtol=1e-6)


def test_olg_converges() -> None:
    path = olg_path(0.1, 0.33, 0.3, 100)
    k_star = (0.3 * 0.67) ** (1.0 / 0.67)
    assert abs(path[-1] - k_star) / k_star < 0.05


def test_cobweb_oscillates() -> None:
    p = cobweb_run(1.0, 0.5, 10, 1.0)
    assert p[0] * p[1] < 0  # sign flips each step
