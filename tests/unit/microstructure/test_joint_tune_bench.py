"""Tests for joint_tune_bench."""

from quant_fund.microstructure.joint_tune_bench import joint_tune_bench


def test_joint_tune_shape() -> None:
    out = joint_tune_bench(horizon=400, seed=7)
    assert out["schema"] == "joint_tune.v1"
    assert out["research_only"] is True
    assert out["data_label"] == "MIXED"
    assert len(out["cells"]) == 8
    for c in out["cells"]:
        assert c["n_draws"] == 2
        assert 0.0 <= c["n_pins_mean"] <= 7.0
        for rate in c["pin_rates"].values():
            assert 0.0 <= rate <= 1.0
    assert set(out["claims"]) == {
        "cells_measured",
        "joint_closure_found",
        "grammar_keeps_pins",
        "kernel_carried",
    }
    assert len(out["receipt_sha256"]) == 64


# SYNTHETIC correctness cases: these do not regenerate research receipts.
def _synthetic_draw(**overrides):
    from quant_fund.microstructure.full_stack_bench import _PIN_FIELDS

    draw = {
        "pins": {pin: True for pin, _ in _PIN_FIELDS},
        "n_pins": 7,
        "card": {"life_events_p50_executed": 25.5},
        "instant_signed_ticks": 0.887,
        "k200": 4.64,
    }
    draw.update(overrides)
    return draw


def _mock_bench(monkeypatch, draws):
    import importlib

    bench = importlib.import_module("quant_fund.microstructure.joint_tune_bench")
    monkeypatch.setattr(bench, "_CELLS", (("synthetic", 12, 50, 0.9, 110),))
    by_seed = dict(zip((7007, 7011), draws, strict=True))
    monkeypatch.setattr(bench, "_cell", lambda *args, **kwargs: by_seed[kwargs["seed"]])
    monkeypatch.setattr(
        bench,
        "_reseed_fates",
        lambda *args, **kwargs: {
            "reseed_rate_500": None,
            "reseed_as_touch_share": None,
            "reseed_latency_p50": None,
        },
    )
    monkeypatch.setattr(bench, "git_revision", lambda: "SYNTHETIC")
    return bench.joint_tune_bench()


def test_closure_requires_all_seven_pins_on_every_draw(monkeypatch):
    first, second = _synthetic_draw(), _synthetic_draw()
    second["pins"]["reveal_gap"] = False
    second["n_pins"] = 6
    out = _mock_bench(monkeypatch, [first, second])
    cell = out["cells"][0]
    assert cell["n_pins_mean"] == 6.5
    assert cell["aggregate_closure"] is True
    assert cell["joint_closure_by_draw"] == [True, False]
    assert cell["joint_closure_rate"] == 0.5
    assert out["claims"]["joint_closure_found"] is False


def test_opposite_draw_failures_cannot_average_into_closure(monkeypatch):
    first = _synthetic_draw(card={"life_events_p50_executed": 5.0}, k200=2.0)
    second = _synthetic_draw(card={"life_events_p50_executed": 55.0}, k200=7.0)
    out = _mock_bench(monkeypatch, [first, second])
    cell = out["cells"][0]
    assert cell["aggregate_closure"] is True
    assert cell["joint_closure_by_draw"] == [False, False]
    assert cell["joint_closure_rate"] == 0.0
    assert out["claims"]["joint_closure_found"] is False


def test_missing_measurement_cannot_disappear_from_closure(monkeypatch):
    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw(k200=None)])
    assert out["cells"][0]["aggregate_closure"] is True
    assert out["claims"]["joint_closure_found"] is False


def test_rounding_cannot_admit_out_of_band_draw(monkeypatch):
    out = _mock_bench(monkeypatch, [_synthetic_draw(k200=6.00001)] * 2)
    assert out["cells"][0]["k200_mean"] == 6.0
    assert out["cells"][0]["aggregate_closure"] is True
    assert out["claims"]["joint_closure_found"] is False


def test_complete_draws_close_and_seal_the_claim_semantics(monkeypatch):
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw()])
    assert out["cells"][0]["joint_closure_by_draw"] == [True, True]
    assert out["cells"][0]["joint_closure_rate"] == 1.0
    assert out["claims"]["joint_closure_found"] is True
    assert out["claim_semantics"]["joint_closure_found"].startswith("all_draws.v1:")
    body = {key: value for key, value in out.items() if key != "receipt_sha256"}
    assert out["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))


def test_draw_closure_fails_on_incomplete_or_nonboolean_pins():
    from quant_fund.microstructure.joint_tune_bench import _draw_closed

    draw = _synthetic_draw()
    assert _draw_closed(draw)
    for pins in ({}, {"invented": True}, {**draw["pins"], "crown": 1}):
        assert not _draw_closed(_synthetic_draw(pins=pins))
    pins = dict(draw["pins"])
    del pins["crown"]
    assert not _draw_closed(_synthetic_draw(pins=pins))


def test_draw_closure_rejects_nonfinite_or_missing_channels():
    from quant_fund.microstructure.joint_tune_bench import _draw_closed

    for value in (None, float("nan"), float("inf"), -float("inf")):
        assert not _draw_closed(_synthetic_draw(k200=value))
        assert not _draw_closed(_synthetic_draw(instant_signed_ticks=value))
        assert not _draw_closed(_synthetic_draw(card={"life_events_p50_executed": value}))


def test_draw_closure_preserves_inclusive_life_and_kernel_boundaries():
    from quant_fund.microstructure.joint_tune_bench import _TAPE_LIFE_EV, _draw_closed

    for life in (0.5 * _TAPE_LIFE_EV, 2.0 * _TAPE_LIFE_EV):
        for k200 in (3.0, 6.0):
            assert _draw_closed(_synthetic_draw(card={"life_events_p50_executed": life}, k200=k200))


def test_instant_band_includes_boundaries_but_not_adjacent_outside_floats():
    import math

    from quant_fund.microstructure.joint_tune_bench import _TAPE_INSTANT, _draw_closed

    low, high = _TAPE_INSTANT - 0.35, _TAPE_INSTANT + 0.35
    for instant in (low, high, math.nextafter(low, high), math.nextafter(high, low)):
        assert _draw_closed(_synthetic_draw(instant_signed_ticks=instant))
    for instant in (math.nextafter(low, -math.inf), math.nextafter(high, math.inf)):
        assert not _draw_closed(_synthetic_draw(instant_signed_ticks=instant))
