"""Tests for joint_tune_bench."""

from quant_fund.microstructure.joint_tune_bench import joint_tune_bench


def test_joint_tune_shape() -> None:
    out = joint_tune_bench(horizon=400, seed=7)
    assert out["schema"] == "joint_tune.v2"
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
    from quant_fund.microstructure.joint_tune_contract import contract_errors

    assert contract_errors(out) == []


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
    by_seed = dict(zip((7007, 7011), draws, strict=True))
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import surface_inputs

    def cell(zone, ttl, intensity, **kwargs):
        draw = deepcopy(by_seed[kwargs["seed"]])
        inputs = surface_inputs(
            zone,
            ttl,
            kwargs["requote"],
            kwargs["fr_delay"],
            horizon=kwargs["horizon"],
            seed=kwargs["seed"],
            intensity=intensity,
        )
        draw["surface_inputs"] = {k: dict(inputs) for k in ("card", "crown", "reseed", "kernel")}
        return draw

    monkeypatch.setattr(bench, "_cell", cell)
    monkeypatch.setattr(
        bench,
        "_reseed_fates",
        lambda zone, ttl, rq, inten, **kwargs: {
            "inputs": surface_inputs(
                zone,
                ttl,
                rq,
                kwargs["fr_delay"],
                horizon=kwargs["horizon"],
                seed=kwargs["seed"],
                intensity=inten,
            ),
            "reseed_rate_500": 0.5376,
            "reseed_as_touch_share": 0.75,
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
    assert out["claim_semantics"]["joint_closure_found"].startswith("all_draws_iid.v2:")
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


def test_joint_tune_requests_uniform_iid_on_every_draw(monkeypatch):
    import importlib

    bench = importlib.import_module("quant_fund.microstructure.joint_tune_bench")
    seen = []

    def cell(zone, ttl, intensity, **kwargs):
        seen.append((intensity, kwargs["uniform_flow"], kwargs["seed"]))
        return _synthetic_draw()

    monkeypatch.setattr(bench, "_cell", cell)
    monkeypatch.setattr(bench, "_CELLS", (("synthetic", 12, 50, 0.9, 110),))
    monkeypatch.setattr(
        bench,
        "_reseed_fates",
        lambda *a, **kw: {
            "inputs": {},
            "reseed_rate_500": None,
            "reseed_as_touch_share": None,
            "reseed_latency_p50": None,
        },
    )
    bench.joint_tune_bench(horizon=1)
    assert seen == [(None, True, 7007), (None, True, 7011)]


def test_uniform_iid_cell_routes_every_surface_and_preserves_legacy_default(monkeypatch):
    import importlib

    cell = importlib.import_module("quant_fund.microstructure.zone_ttl_bench")
    seen = {}

    def card(zone, ttl, intensity, **kwargs):
        seen["card"] = (intensity, kwargs["seed"])
        return {}

    def crown(*args, **kwargs):
        seen["crown"] = (kwargs["flow_intensity"], kwargs["seed"])
        return {"n_reveals": 0, "n_fills": 0}

    def reseed(*args, **kwargs):
        seen["reseed"] = (kwargs["flow_intensity"], kwargs["seed"])
        return {}

    def kernel(config, flow, horizon):
        seen["kernel_flow"] = flow
        return {"instant_signed_ticks": None, "kernel_mean_ticks": {}}

    monkeypatch.setattr(cell, "_card", card)
    monkeypatch.setattr(cell, "_sim_crown", crown)
    monkeypatch.setattr(cell, "sim_reseed", reseed)
    monkeypatch.setattr(cell, "_pins_ok", lambda _: {})
    monkeypatch.setattr(cell, "_measure", kernel)
    cell._cell(12, 50, None, horizon=1, seed=7007, uniform_flow=True)
    assert seen == {
        "card": (None, 7007),
        "crown": (None, 7007),
        "reseed": (None, 7007),
        "kernel_flow": None,
    }
    legacy = cell._cell(12, 50, None, horizon=1, seed=7007)
    assert "surface_inputs" not in legacy
    assert seen["crown"] == (3.0, 7007)
    assert seen["kernel_flow"] is None


def test_v2_contract_rederives_closure_and_keeps_v1_registered(monkeypatch):
    from copy import deepcopy

    from quant_fund.research.script_receipts import (
        SCRIPT_RECEIPT_CONTRACTS,
        measurement_receipt_contract_errors,
        script_receipt_contract_errors,
    )

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw(k200=7.0)])

    def check(payload):
        return script_receipt_contract_errors("joint_tune.v2", payload)

    assert SCRIPT_RECEIPT_CONTRACTS["joint_tune.v1"] is measurement_receipt_contract_errors
    assert check(out) == []
    for key, value in (
        ("joint_closure_by_draw", [True, True]),
        ("joint_closure_rate", 1.0),
        ("n_draws", 3),
    ):
        forged = deepcopy(out)
        forged["cells"][0][key] = value
        assert check(forged)
    forged = deepcopy(out)
    forged["claims"]["joint_closure_found"] = True
    assert "joint_closure_found_mismatch" in check(forged)
    forged = deepcopy(out)
    forged["config"]["flow"] = "split"
    assert "joint_tune_flow_not_iid" in check(forged)
    for cells in ([], None, [{"draws": []}], [{"draws": [{}]}]):
        forged = deepcopy(out)
        forged["cells"] = cells
        assert check(forged)


def test_v2_contract_rejects_dropping_a_failed_sample(monkeypatch):
    from quant_fund.research.script_receipts import script_receipt_contract_errors

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw(k200=7.0)])
    cell = out["cells"][0]
    cell["draws"] = cell["draws"][:1]
    cell["n_draws"] = 1
    cell["joint_closure_by_draw"] = [True]
    cell["joint_closure_rate"] = 1.0
    out["claims"]["joint_closure_found"] = True
    errors = script_receipt_contract_errors("joint_tune.v2", out)
    assert "cell_0:draw_count_mismatch" in errors
    assert "cells_measured_mismatch" in errors


def test_v2_requires_complete_seed_and_cell_design(monkeypatch):
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import contract_errors

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw(k200=7.0)])
    assert contract_errors(out) == []
    for seeds in ([7], [11, 7], [7, 7], [True, 11], [7.0, 11], []):
        sample = deepcopy(out)
        sample["config"]["seeds"] = seeds
        assert "joint_tune_seed_set_mismatch" in contract_errors(sample)
    sample = deepcopy(out)
    sample["config"]["seeds"] = [7]
    for cell in sample["cells"]:
        cell["draws"] = cell["draws"][:1]
    assert "joint_tune_seed_set_mismatch" in contract_errors(sample)
    for key in ("config", "cells"):
        sample = deepcopy(out)
        if key == "config":
            sample[key]["cells"] = sample[key]["cells"][:1]
        else:
            sample[key] = sample[key][:1]
        assert contract_errors(sample)


def test_v2_binds_each_surface_to_draw_seed_and_configuration(monkeypatch):
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import contract_errors

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw()])
    for surface in ("card", "crown", "reseed", "kernel"):
        for key, value in (
            ("flow_intensity", 3.0),
            ("seed", 7011),
            ("seed", True),
            ("horizon", 1),
            ("maker_ttl", 75),
            ("fill_repost_frac", 0.5),
        ):
            sample = deepcopy(out)
            sample["cells"][0]["draws"][0]["surface_inputs"][surface][key] = value
            assert "cell_0:draw_0:surface_inputs_mismatch" in contract_errors(sample)
    sample = deepcopy(out)
    del sample["cells"][0]["draws"][0]["surface_inputs"]
    assert "cell_0:draw_0:surface_inputs_mismatch" in contract_errors(sample)
    sample = deepcopy(out)
    sample["cells"][0]["draws"][1] = deepcopy(sample["cells"][0]["draws"][0])
    assert "cell_0:draw_1:surface_inputs_mismatch" in contract_errors(sample)
    sample = deepcopy(out)
    sample["cells"][0]["draws"][0]["reseed_fates"]["inputs"]["seed"] = 7011
    assert "cell_0:draw_0:reseed_inputs_mismatch" in contract_errors(sample)


def test_v2_rederives_all_claims_and_every_cell_summary(monkeypatch):
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import contract_errors, summaries

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw(k200=7.0)])
    for key in out["claims"]:
        sample = deepcopy(out)
        sample["claims"][key] = not sample["claims"][key]
        assert key + "_mismatch" in contract_errors(sample)
    for key in summaries(out["cells"][0]["draws"]):
        sample = deepcopy(out)
        sample["cells"][0][key] = "invalid"
        assert "cell_0:" + key + "_mismatch" in contract_errors(sample)
    sample = deepcopy(out)
    sample["cells"][0]["draws"][0]["n_pins"] = 6
    assert "cell_0:draw_0:n_pins_mismatch" in contract_errors(sample)
    sample = deepcopy(out)
    sample["claims"]["extra_result"] = True
    assert "joint_tune_claim_set_mismatch" in contract_errors(sample)


def test_v2_reference_is_exact_and_independent_of_live_benchmark(monkeypatch):
    import importlib
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import contract_errors

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw()])
    for key in out["tape_reference"]:
        sample = deepcopy(out)
        sample["tape_reference"][key] = None
        assert "joint_tune_tape_reference_mismatch" in contract_errors(sample)
    bench = importlib.import_module("quant_fund.microstructure.joint_tune_bench")
    monkeypatch.setattr(bench, "_draw_closed", lambda d: False)
    monkeypatch.setattr(bench, "_TAPE_LIFE_EV", -1)
    assert contract_errors(out) == []


def test_v2_best_cell_tiebreak_is_pin_count_then_life_then_first():
    from quant_fund.microstructure.joint_tune_contract import claims

    def cell(pins, life, instant, k200):
        return dict(
            n_draws=2,
            n_pins_mean=pins,
            life_ev_p50_mean=life,
            instant_mean=instant,
            k200_mean=k200,
            joint_closure_by_draw=[False, False],
        )

    fail = cell(7, 30, 5, 20)
    passed = cell(7, 25.5, 0.887, 4.64)
    assert claims([fail, passed])["kernel_carried"] is True
    assert claims([passed, cell(7, 25.5, 5, 20)])["kernel_carried"] is True
    assert claims([cell(7, 25.5, 5, 20), passed])["kernel_carried"] is False
    assert claims([cell(7, 100, 5, 20), cell(6, 25.5, 0.887, 4.64)])["kernel_carried"] is False


def test_v2_contract_rejects_malformed_nested_values_without_raising(monkeypatch):
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import contract_errors

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw()])
    for key in ("config", "cells", "claims", "seed", "horizon", "tape_reference"):
        for value in (None, True, [], "invalid"):
            sample = deepcopy(out)
            sample[key] = value
            assert contract_errors(sample)
    for key in ("pins", "card", "reseed_fates", "surface_inputs", "instant_signed_ticks", "k200"):
        for value in (True, [], "invalid", float("nan"), float("inf")):
            sample = deepcopy(out)
            sample["cells"][0]["draws"][0][key] = value
            assert contract_errors(sample)


def test_v2_rejects_impossible_domains_and_evidence_labels(monkeypatch):
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import contract_errors

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw()])
    for section, key, value in (
        ("card", "life_events_p50_executed", -1),
        ("reseed_fates", "reseed_rate_500", 1.1),
        ("reseed_fates", "reseed_as_touch_share", -0.1),
        ("reseed_fates", "reseed_latency_p50", -1),
    ):
        sample = deepcopy(out)
        sample["cells"][0]["draws"][0][section][key] = value
        assert "cell_0:draw_0:measurement_domain_invalid" in contract_errors(sample)
    for key, value in (("data_label", "REAL"), ("kind", "market_evidence")):
        sample = deepcopy(out)
        sample[key] = value
        assert "joint_tune_evidence_label_mismatch" in contract_errors(sample)
    sample = deepcopy(out)
    sample["claim_semantics"]["grammar_keeps_pins"] = "every draw closes"
    assert "joint_tune_claim_semantics_mismatch" in contract_errors(sample)


def test_v2_aggregate_instant_preserves_original_absolute_difference():
    from quant_fund.microstructure.joint_tune_contract import claims

    for instant in (0.537, 1.237, 0.887):
        cells = [
            dict(
                n_draws=2,
                n_pins_mean=7,
                life_ev_p50_mean=25.5,
                instant_mean=instant,
                k200_mean=4.64,
                joint_closure_by_draw=[True, True],
            )
        ]
        assert claims(cells)["kernel_carried"] is (abs(instant - 0.887) <= 0.35)


def test_v2_binds_reseed_pins_to_embedded_same_run_measurements(monkeypatch):
    from copy import deepcopy

    from quant_fund.microstructure.joint_tune_contract import claims, contract_errors, summaries

    out = _mock_bench(monkeypatch, [_synthetic_draw(), _synthetic_draw()])
    for key in ("reseed_rate_500", "reseed_as_touch_share"):
        for value in (None, 0.0):
            sample = deepcopy(out)
            for cell in sample["cells"]:
                for draw in cell["draws"]:
                    draw["reseed_fates"][key] = value
                cell.update(summaries(cell["draws"]))
            sample["claims"] = claims(sample["cells"])
            assert "cell_0:draw_0:reseed_pin_mismatch" in contract_errors(sample)
