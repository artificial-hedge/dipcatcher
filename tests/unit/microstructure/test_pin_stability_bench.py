"""Tests for pin_stability_bench."""

from quant_fund.microstructure.pin_stability_bench import pin_stability_bench


def test_pin_stability_shape() -> None:
    out = pin_stability_bench(horizon=3000, seed=7)
    assert out["schema"] == "pin_stability.v1"
    assert out["research_only"] is True
    assert len(out["rows"]) == 2
    assert [r["flow_intensity"] for r in out["rows"]] == [None, 2.0]
    for r in out["rows"]:
        assert set(r["pin_hold_rate"]) == {
            "crown",
            "empty",
            "hidden",
            "reseed_rate",
            "reseed_touch",
            "reveal_gap",
            "spread",
        }
        assert all(0.0 <= v <= 1.0 for v in r["pin_hold_rate"].values())
        assert 0 <= r["n_all_pins"] <= r["n_seeds"]
    assert set(out["claims"]) == {
        "panel_measured",
        "closure_is_knife_edge",
        "fragility_quantified",
        "iid_spread_fails",
        "split_instant_holds_k200_overshoots",
    }
    assert len(out["receipt_sha256"]) == 64
