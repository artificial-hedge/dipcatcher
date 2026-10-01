"""Tests for diffbacktest.backend_parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.diffbacktest.backend_parity import backend_parity_bench, parity_check
from quant_fund.diffbacktest.spec import STRATEGIES, StrategyParams


def test_parity_tsmom_default() -> None:
    pytest.importorskip("jax")
    rng = np.random.default_rng(1)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (120, 4)), axis=0))
    row = parity_check(prices, "tsmom", StrategyParams())
    assert row["parity"], row["per_array"]


def test_bench_seals_and_verifies() -> None:
    pytest.importorskip("jax")
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    receipt = backend_parity_bench(n_prices=1, seed=3)
    assert receipt["claim"]["status"] == "ok"
    assert receipt["claim"]["n_strategies"] == len(STRATEGIES)
    verdict = verify_receipt_payload(receipt, Path("receipts") / "backend_parity.json")
    assert verdict["valid"], verdict["errors"]


def test_skipped_when_jax_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.diffbacktest import jax_core

    monkeypatch.setattr(jax_core, "available", lambda: False)
    receipt = backend_parity_bench(n_prices=1)
    assert receipt["claim"]["status"] == "skipped"
    assert "receipt_sha256" not in receipt
