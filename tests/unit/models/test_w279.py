"""Wave-279 control-theory-3 module tests."""

import numpy as np

from quant_fund.models.flat_track import _ref
from quant_fund.models.l2_gain import _run as l2_run
from quant_fund.models.luen_obsv import _simulate
from quant_fund.models.mrac_adapt import _run as mrac_run


def test_luen_error_shrinks() -> None:
    xs, xhs, _ = _simulate(1)
    e0 = np.abs(xs[:20] - xhs[:20]).mean()
    e1 = np.abs(xs[-20:] - xhs[-20:]).mean()
    assert e1 < e0


def test_ref_endpoints() -> None:
    y0, _, _ = _ref(0.0)
    y1, _, _ = _ref(8.0)
    assert y0 == 0.0 and y1 == 1.0


def test_mrac_adaptive_beats_frozen() -> None:
    assert mrac_run(2, adaptive=True) < mrac_run(2, adaptive=False)


def test_l2_positive() -> None:
    assert l2_run(3, feedback=True) > 0
