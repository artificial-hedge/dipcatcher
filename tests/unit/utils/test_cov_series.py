"""Direct coverage for the shared series/panel guards in ``utils/series.py``.

The guards are reached indirectly through model fits, so these tests pin the
public contract (flattening, filtering, naming) and every fail-closed branch
at the seam where callers rely on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.utils.series import (
    as_named_1d,
    finite_1d,
    finite_nonconstant_series,
    finite_observations,
    finite_series,
    require_panel,
    require_same_length,
)


def test_finite_series_flattens_2d_input() -> None:
    out = finite_series(np.array([[1.0, 2.0], [3.0, 4.0]]), min_length=4)
    assert out.shape == (4,)
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_finite_series_accepts_exact_minimum_length() -> None:
    out = finite_series([1.0, 2.0, 3.0], min_length=3)
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_finite_series_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="length >= 3"):
        finite_series([1.0, 2.0], min_length=3)


def test_finite_series_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="series must be finite"):
        finite_series([1.0, np.nan, 3.0], min_length=1)
    with pytest.raises(ValueError, match="series must be finite"):
        finite_series([1.0, np.inf], min_length=1)


def test_finite_nonconstant_series_passes_varying_input() -> None:
    out = finite_nonconstant_series([1.0, 2.0, 4.0], min_length=2)
    assert out.tolist() == [1.0, 2.0, 4.0]


def test_finite_nonconstant_series_rejects_constant_input() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        finite_nonconstant_series([2.0, 2.0, 2.0], min_length=1)


def test_finite_nonconstant_series_propagates_length_failure() -> None:
    with pytest.raises(ValueError, match="length >= 5"):
        finite_nonconstant_series([1.0, 2.0], min_length=5)


def test_finite_observations_drops_non_finite_and_keeps_order() -> None:
    out = finite_observations([3.0, np.nan, 1.0, np.inf, 2.0], min_obs=3)
    assert out.tolist() == [3.0, 1.0, 2.0]


def test_finite_observations_uses_name_in_failure() -> None:
    with pytest.raises(ValueError, match="returns must contain at least 2"):
        finite_observations([1.0, np.nan], "returns", min_obs=2)


def test_finite_1d_keeps_empty_input_empty() -> None:
    out = finite_1d([])
    assert out.shape == (0,)


def test_finite_1d_filters_non_finite() -> None:
    out = finite_1d([[1.0, np.nan], [np.inf, -2.0]])
    assert out.tolist() == [1.0, -2.0]


def test_as_named_1d_returns_a_float_vector() -> None:
    out = as_named_1d("y", [1, 2, 3])
    assert out.shape == (3,)
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_as_named_1d_wraps_a_scalar() -> None:
    assert as_named_1d("y", 4.5).tolist() == [4.5]


def test_as_named_1d_rejects_row_and_column_vectors() -> None:
    with pytest.raises(ValueError, match="signal must be 1d"):
        as_named_1d("signal", np.zeros((1, 3)))
    with pytest.raises(ValueError, match="y must be 1d"):
        as_named_1d("y", np.zeros((3, 1)))


def test_require_same_length_accepts_equal_lengths() -> None:
    require_same_length(("a", np.zeros(3)), ("b", np.ones(3)))


def test_require_same_length_reports_each_name_and_length() -> None:
    with pytest.raises(ValueError) as excinfo:
        require_same_length(("a", np.zeros(3)), ("b", np.ones(2)))
    message = str(excinfo.value)
    assert "a=3" in message
    assert "b=2" in message


def test_require_panel_returns_float_panel() -> None:
    panel = require_panel(np.arange(6, dtype=int).reshape(3, 2), min_n=2, min_t=3)
    assert panel.shape == (3, 2)
    assert panel.dtype == np.float64


def test_require_panel_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match=r"panel must be \(T >= 2, N >= 1\)"):
        require_panel(np.zeros(4), min_n=1, min_t=2)


def test_require_panel_rejects_narrow_or_short_panel() -> None:
    with pytest.raises(ValueError, match=r"T >= 4"):
        require_panel(np.zeros((3, 2)), min_n=2, min_t=4)
    with pytest.raises(ValueError, match=r"N >= 3"):
        require_panel(np.zeros((5, 2)), min_n=3, min_t=1)


def test_require_panel_rejects_non_finite_entry() -> None:
    panel = np.zeros((2, 2))
    panel[1, 0] = np.nan
    with pytest.raises(ValueError, match="panel must be finite"):
        require_panel(panel, min_n=1, min_t=1)
