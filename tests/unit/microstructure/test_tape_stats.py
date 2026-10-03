"""Contracts for tape_stats — the receipt→table loader must fail closed."""

import json

import pytest

from quant_fund.microstructure.tape_stats import (
    TapeStats,
    load_tape_stats,
    scaled_size_pmf,
)


class TestLoadTapeStats:
    def test_loads_amzn(self):
        stats = load_tape_stats("receipts", "amzn")
        assert stats.ticker == "amzn"
        assert sum(w for _, w in stats.size_pmf) > 0
        assert stats.rate("exec") > 0
        assert stats.mean_spread_ticks == pytest.approx(13.086)

    def test_missing_root_fails_closed(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            load_tape_stats(tmp_path / "nope")

    def test_missing_receipt_fails_closed(self, tmp_path):
        with pytest.raises(ValueError, match="missing tape receipt"):
            load_tape_stats(tmp_path, "amzn")

    def test_missing_real_section_fails_closed(self, tmp_path):
        for name in ("round_lot_amzn", "event_matrix_amzn", "spread_dynamics_amzn"):
            (tmp_path / f"{name}.json").write_text(json.dumps({"schema": "x.v1"}))
        with pytest.raises(ValueError, match="'real' section"):
            load_tape_stats(tmp_path, "amzn")

    def test_missing_rate_key_fails_closed(self):
        stats = load_tape_stats("receipts", "amzn")
        with pytest.raises(ValueError, match="lacks rate"):
            stats.rate("nonexistent_kind")


class TestScaledSizePmf:
    def test_scale_and_floor(self):
        pmf = ((10, 0.5), (200, 0.5))
        out = scaled_size_pmf(pmf, 0.001)
        assert out == ((1, 0.5), (1, 0.5)) or all(s >= 1 for s, _ in out)
        assert tuple(w for _, w in out) == (0.5, 0.5)

    def test_bad_scale_fails_closed(self):
        with pytest.raises(ValueError, match="scale"):
            scaled_size_pmf(((1, 1.0),), -1.0)
        with pytest.raises(ValueError, match="scale"):
            scaled_size_pmf(((1, 1.0),), float("nan"))

    def test_stats_frozen(self):
        stats = load_tape_stats("receipts", "amzn")
        assert isinstance(stats, TapeStats)
        with pytest.raises(AttributeError):
            stats.mean_spread_ticks = 0.0  # type: ignore[misc]
