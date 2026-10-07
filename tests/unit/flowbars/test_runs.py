import numpy as np
import pytest

from quant_fund.flowbars.runs import (
    dollar_run_bar_ids,
    max_run_profile,
    run_bar_ids,
    tick_run_bar_ids,
    volume_run_bar_ids,
)

pytestmark = pytest.mark.synthetic


def test_tick_run_bars_close_on_streak() -> None:
    signs = np.array([1, 1, 1, -1, -1, 1])
    ids = tick_run_bar_ids(signs, 3.0)
    # streak of 3 ups closes at idx2; 2 downs + 1 up never reach 3 again
    np.testing.assert_array_equal(ids, [0, 0, 0, 1, 1, 1])


def test_runs_reset_when_side_flips() -> None:
    signs = np.array([1, 1, -1, 1, 1])
    ids = tick_run_bar_ids(signs, 2.0)
    # +2 closes at idx1; flip resets; +2 (idx3,4) closes at idx4
    np.testing.assert_array_equal(ids, [0, 0, 1, 1, 1])


def test_volume_run_bars_weight_by_size() -> None:
    signs = np.array([1, 1, -1])
    sizes = np.array([1.0, 4.0, 2.0])
    ids = volume_run_bar_ids(signs, sizes, 5.0)
    # buy run reaches 5.0 at idx1 → close; sell run 2.0 stays
    np.testing.assert_array_equal(ids, [0, 0, 1])


def test_dollar_run_bars_match_generic() -> None:
    signs = np.array([1, -1, -1, 1, 1, -1])
    dollars = np.array([2.0, 1.0, 3.0, 1.0, 4.0, 1.0])
    generic = run_bar_ids(signs, dollars, 4.0)
    named = dollar_run_bar_ids(signs, dollars, 4.0)
    np.testing.assert_array_equal(generic, named)


def test_max_run_profile() -> None:
    signs = np.array([1, 1, -1, -1, -1])
    profile = max_run_profile(signs, np.ones(5))
    np.testing.assert_allclose(profile, [1.0, 2.0, 1.0, 2.0, 3.0])


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        run_bar_ids(np.array([1, -1]), np.array([1.0, 1.0]), 0.0)
    with pytest.raises(ValueError):
        run_bar_ids(np.array([1, -1]), np.array([1.0]), 2.0)
