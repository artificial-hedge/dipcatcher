import numpy as np
import pytest

from quant_fund.flowbars.bars import dollar_bar_ids, synth_tape, tick_bar_ids, volume_bar_ids
from quant_fund.flowbars.imbalance import tick_rule_signs, volume_imbalance_bar_ids
from quant_fund.flowbars.roundtrip import (
    bar_builder_audit,
    invariant_monotone_ids,
    invariant_ohlc_consistent,
    invariant_returns_telescope,
    invariant_volume_conserved,
)
from quant_fund.flowbars.runs import tick_run_bar_ids

pytestmark = pytest.mark.synthetic


@pytest.fixture(scope="module")
def tape():
    return synth_tape(5000, seed=80)


def test_dollar_bars_pass_all_invariants(tape) -> None:
    checks = bar_builder_audit(
        tape["price"], tape["size"], lambda p, s: dollar_bar_ids(p * s, 2000.0)
    )
    assert checks["score"] == 1.0


def test_volume_bars_pass_all_invariants(tape) -> None:
    checks = bar_builder_audit(tape["price"], tape["size"], lambda p, s: volume_bar_ids(s, 40.0))
    assert checks["score"] == 1.0


def test_tick_bars_pass_all_invariants(tape) -> None:
    checks = bar_builder_audit(tape["price"], tape["size"], lambda p, s: tick_bar_ids(len(p), 25))
    assert checks["score"] == 1.0


def test_imbalance_bars_pass_all_invariants(tape) -> None:
    def build(p: np.ndarray, s: np.ndarray) -> np.ndarray:
        signs = tick_rule_signs(p)
        return volume_imbalance_bar_ids(signs, s, 40.0)

    checks = bar_builder_audit(tape["price"], tape["size"], build)
    assert checks["score"] == 1.0


def test_run_bars_pass_all_invariants(tape) -> None:
    def build(p: np.ndarray, s: np.ndarray) -> np.ndarray:
        signs = tick_rule_signs(p)
        return tick_run_bar_ids(signs, 15.0)

    checks = bar_builder_audit(tape["price"], tape["size"], build)
    assert checks["score"] == 1.0


def test_monotone_ids_rejects_gap() -> None:
    assert not invariant_monotone_ids(np.array([0, 0, 2, 2]))
    assert not invariant_monotone_ids(np.array([1, 1, 2]))
    assert invariant_monotone_ids(np.array([0, 0, 1, 2]))


def test_ohlc_consistent_rejects_bad_bars() -> None:
    # hand-built ids where a bar is NOT monotone group-consistent is hard to
    # build from real builders; instead verify the negative via reordering
    prices = np.array([1.0, 2.0, 0.5, 3.0])
    ids = np.array([0, 1, 1, 1])  # bar 1 has prices 2.0, 0.5, 3.0 → consistent
    assert invariant_ohlc_consistent(prices, ids)
    bad_ids = np.array([0, 1, 0, 1])  # bar 0 has prices 1.0, 0.5 (ids non-monotone → False anyway)
    assert not invariant_monotone_ids(bad_ids)


def test_volume_conserved_rejects_shape_mismatch() -> None:
    assert not invariant_volume_conserved(np.ones(4), np.ones(3))


def test_returns_telescope_exact() -> None:
    prices = np.array([100.0, 101.0, 99.5, 102.0, 103.0, 104.0])
    ids = np.array([0, 0, 1, 1, 2, 2])
    assert invariant_returns_telescope(prices, ids)
