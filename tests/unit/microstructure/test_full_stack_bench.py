"""Tests for full_stack_bench."""

from quant_fund.microstructure.full_stack_bench import (
    _pins_ok,
    full_stack_bench,
)


def test_full_stack_shape() -> None:
    out = full_stack_bench(tape_dir=None, horizon=6000, seed=7)
    assert out["schema"] == "full_stack.v1"
    assert out["data_label"] == "MIXED"
    assert out["research_only"] is True
    assert out["tape"] is None
    assert len(out["cells"]) == 6
    assert set(out["claims"]) == {
        "cells_measured",
        "full_stack_found",
        "joint_cell_reseeds",
        "repost_preserves_joint_pins",
        "tape_remeasures_in_band",
    }
    for c in out["cells"]:
        assert set(c["pins"]) == {
            "crown",
            "empty",
            "spread",
            "hidden",
            "reseed_rate",
            "reseed_touch",
            "reveal_gap",
        }
        assert c["n_pins_ok"] == sum(c["pins"].values())
        assert isinstance(c["deltas"], dict)
    assert len(out["receipt_sha256"]) == 64


def test_pin_verdicts_are_band_semantics() -> None:
    good = {
        "crown_share_of_visible": 0.20,
        "empty_share": 0.47,
        "spread_mean": 15.0,
        "hidden_fill_share": 0.21,
        "reseed_rate_500": 0.54,
        "reseed_as_touch_share": 0.75,
        "reveal_gap_ticks_mean": 3.8,
    }
    assert all(_pins_ok(good).values())
    bad = dict(good, spread_mean=2.0, reseed_rate_500=0.9)
    pins = _pins_ok(bad)
    assert not pins["spread"] and not pins["reseed_rate"]
    # absent fields are skipped, not failed
    assert "spread" not in _pins_ok({"reseed_rate_500": 0.5})
