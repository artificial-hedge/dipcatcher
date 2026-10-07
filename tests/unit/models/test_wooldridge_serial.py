"""Tests for wooldridge_serial (wave-57)."""

import numpy as np
import pytest

from quant_fund.models.wooldridge_serial import (
    bench_wooldridge,
    synth_wooldridge,
    wooldridge_serial,
)


def test_ar_errors_reject() -> None:
    e_ar, _, u = synth_wooldridge(seed=1)
    r = wooldridge_serial(e_ar, u)
    assert r["reject5"] == 1.0


def test_iid_accept() -> None:
    _, e_iid, u = synth_wooldridge(seed=2)
    r = wooldridge_serial(e_iid, u)
    assert r["reject5"] == 0.0


def test_fail_closed() -> None:
    e_ar, _, u = synth_wooldridge(seed=3)
    with pytest.raises(ValueError):
        wooldridge_serial(e_ar[:30], u[:30])
    with pytest.raises(ValueError):
        wooldridge_serial(np.full(200, np.nan), u[:200])
    with pytest.raises(ValueError):
        wooldridge_serial(np.ones(200), u[:200])


def test_determinism() -> None:
    e_ar, _, u = synth_wooldridge(seed=4)
    assert wooldridge_serial(e_ar, u) == wooldridge_serial(e_ar, u)


def test_bench_schema_and_score() -> None:
    r = bench_wooldridge()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["synthetic_score"] == 1.0
