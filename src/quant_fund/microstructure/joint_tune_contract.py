"""Frozen numerical and provenance specification for joint_tune.v2.

Changing these definitions requires a new schema. Input provenance is a
self-contained execution record, not authentication of a simulator run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_CELLS: tuple[tuple[str, int, int, float, int], ...] = (
    ("ttl50_rq90_d110", 12, 50, 0.9, 110),
    ("ttl50_rq90_d60", 12, 50, 0.9, 60),
    ("ttl75_rq90_d60", 12, 75, 0.9, 60),
    ("ttl75_rq90_d110", 12, 75, 0.9, 110),
    ("ttl100_rq75_d60", 12, 100, 0.75, 60),
    ("ttl100_rq75_d110", 12, 100, 0.75, 110),
    ("ttl150_rq75_d60", 12, 150, 0.75, 60),
    ("ttl200_rq60_d60", 12, 200, 0.6, 60),
)

_SEEDS = (7, 11)

_TAPE_LIFE_EV = 2.213 * 11.528
_TAPE_K200 = 4.64
_TAPE_INSTANT = 0.887

_PIN_NAMES = ("crown", "empty", "spread", "hidden", "reseed_rate", "reseed_touch", "reveal_gap")


def claim_semantics() -> dict[str, str]:
    return {
        "joint_closure_found": "all_draws_iid.v2: all seven pins plus life, instant, and k200 "
        "within tolerance on every draw of at least one cell; every surface uses iid "
        "flow with the same configuration and seed; uses unrounded values",
        "grammar_keeps_pins": "aggregate: mean life <= 2x tape and mean pin count >= 6",
        "kernel_carried": "aggregate: mean instant and k200 in tolerance on best-pins cell",
        "aggregate_closure": "legacy diagnostic: mean pins >= 6.5 plus mean life, instant, "
        "and k200 in tolerance; does not establish per-draw joint closure",
    }


def tape_reference() -> dict[str, Any]:
    return {
        "life_p50_events": round(_TAPE_LIFE_EV, 1),
        "life_p50_events_unrounded": _TAPE_LIFE_EV,
        "k200_ticks": _TAPE_K200,
        "instant_ticks": _TAPE_INSTANT,
        "life_band": [0.5 * _TAPE_LIFE_EV, 2.0 * _TAPE_LIFE_EV],
        "instant_band": [_TAPE_INSTANT - 0.35, _TAPE_INSTANT + 0.35],
        "k200_band": [3.0, 6.0],
        "reseed_rate_band": [0.7 * 0.5376, 1.3 * 0.5376],
        "reseed_touch_min": 0.6,
        "sources": ["order_lifetime.v1", "impact_persist.v1"],
    }


def cell_config(cell: tuple[str, int, int, float, int]) -> dict[str, Any]:
    label, zone, ttl, rq, delay = cell
    return dict(
        label=label, zone_embargo=zone, maker_ttl=ttl, maker_requote=rq, fill_repost_delay=delay
    )


def surface_inputs(
    zone: int,
    ttl: int,
    rq: float,
    delay: int,
    *,
    horizon: int,
    seed: int,
    intensity: float | None,
    fr_frac: float = 0.8,
) -> dict[str, Any]:
    return dict(
        zone_embargo=zone,
        maker_ttl=ttl,
        maker_requote=rq,
        fill_repost_delay=delay,
        fill_repost_frac=fr_frac,
        horizon=horizon,
        seed=seed,
        flow_intensity=intensity,
    )


def _finite(value: Any) -> bool:
    try:
        return type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        return False


def _inside(value: Any, low: float, high: float) -> bool:
    return _finite(value) and low <= value <= high


def draw_closed(draw: Mapping[str, Any]) -> bool:
    pins = draw.get("pins")
    card = draw.get("card")
    return (
        isinstance(pins, Mapping)
        and set(pins) == set(_PIN_NAMES)
        and all(v is True for v in pins.values())
        and isinstance(card, Mapping)
        and _inside(card.get("life_events_p50_executed"), 0.5 * _TAPE_LIFE_EV, 2 * _TAPE_LIFE_EV)
        and _inside(draw.get("instant_signed_ticks"), _TAPE_INSTANT - 0.35, _TAPE_INSTANT + 0.35)
        and _inside(draw.get("k200"), 3.0, 6.0)
    )


def _aggregate_instant(value: Any) -> bool:
    # Preserve v2's original rounded aggregate arithmetic, including float edges.
    return _finite(value) and abs(value - _TAPE_INSTANT) <= 0.35


def summaries(draws: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[Any]) -> float | None:
        present = [v for v in values if v is not None]
        return round(sum(present) / len(present), 4) if present else None

    flags = [draw_closed(d) for d in draws]
    out: dict[str, Any] = dict(
        n_draws=len(draws),
        n_pins_mean=round(sum(sum(d["pins"].values()) for d in draws) / len(draws), 2),
        pin_rates={p: sum(d["pins"][p] for d in draws) / len(draws) for p in _PIN_NAMES},
        life_ev_p50_mean=mean([d["card"]["life_events_p50_executed"] for d in draws]),
        instant_mean=mean([d["instant_signed_ticks"] for d in draws]),
        k200_mean=mean([d["k200"] for d in draws]),
        joint_closure_by_draw=flags,
        joint_closure_rate=sum(flags) / len(draws),
    )
    for key in ("reseed_rate_500", "reseed_as_touch_share", "reseed_latency_p50"):
        out[key] = mean([d["reseed_fates"][key] for d in draws])
    out["aggregate_closure"] = (
        out["n_pins_mean"] >= 6.5
        and _inside(out["life_ev_p50_mean"], 0.5 * _TAPE_LIFE_EV, 2 * _TAPE_LIFE_EV)
        and _aggregate_instant(out["instant_mean"])
        and _inside(out["k200_mean"], 3.0, 6.0)
    )
    return out


def claims(cells: list[dict[str, Any]]) -> dict[str, bool]:
    best = max(
        cells,
        key=lambda c: (c["n_pins_mean"], -abs((c["life_ev_p50_mean"] or 999) - _TAPE_LIFE_EV)),
    )
    return dict(
        cells_measured=all(c["n_draws"] == len(_SEEDS) for c in cells),
        joint_closure_found=any(all(c["joint_closure_by_draw"]) for c in cells),
        grammar_keeps_pins=any(
            c["life_ev_p50_mean"] is not None
            and c["life_ev_p50_mean"] <= 2 * _TAPE_LIFE_EV
            and c["n_pins_mean"] >= 6
            for c in cells
        ),
        kernel_carried=(
            _aggregate_instant(best["instant_mean"]) and _inside(best["k200_mean"], 3.0, 6.0)
        ),
    )


def _same(actual: Any, expected: Any) -> bool:
    # bool must never compare equal to a numeric measurement or seed.
    if isinstance(expected, dict):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_same(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if type(expected) in (int, float):
        return (
            _finite(actual)
            and actual == expected
            and (type(expected) is not int or type(actual) is int)
        )
    return type(actual) is type(expected) and actual == expected


def contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("kind") != "microstructure_bench" or payload.get("data_label") != "MIXED":
        errors.append("joint_tune_evidence_label_mismatch")
    if not _same(payload.get("claim_semantics"), claim_semantics()):
        errors.append("joint_tune_claim_semantics_mismatch")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return ["joint_tune_config_missing"]
    if config.get("flow") != "iid":
        errors.append("joint_tune_flow_not_iid")
    if not _same(config.get("seeds"), list(_SEEDS)):
        errors.append("joint_tune_seed_set_mismatch")
    if not _same(config.get("cells"), [cell_config(c) for c in _CELLS]):
        errors.append("joint_tune_cell_set_mismatch")
    if not _same(payload.get("tape_reference"), tape_reference()):
        errors.append("joint_tune_tape_reference_mismatch")
    seed, horizon = payload.get("seed"), payload.get("horizon")
    if type(seed) is not int or seed < 0 or type(horizon) is not int or horizon <= 0:
        return errors + ["joint_tune_run_inputs_invalid"]
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != len(_CELLS):
        return errors + ["joint_tune_cells_missing"]
    derived = []
    for i, (cell, spec) in enumerate(zip(cells, _CELLS, strict=True)):
        prefix = f"cell_{i}:"
        if not isinstance(cell, Mapping):
            errors.append(prefix + "not_object")
            continue
        expected_config = cell_config(spec)
        expected_config["regime"] = expected_config.pop("label")
        if any(not _same(cell.get(k), v) for k, v in expected_config.items()):
            errors.append(prefix + "configuration_mismatch")
        draws = cell.get("draws")
        if not isinstance(draws, list) or len(draws) != len(_SEEDS):
            errors.extend([prefix + "draw_count_mismatch", "cells_measured_mismatch"])
            continue
        valid = True
        for j, (draw, offset) in enumerate(zip(draws, _SEEDS, strict=True)):
            dp = prefix + f"draw_{j}:"
            if not isinstance(draw, Mapping):
                errors.append(dp + "not_object")
                valid = False
                continue
            _, zone, ttl, rq, delay = spec
            inputs = surface_inputs(
                zone, ttl, rq, delay, horizon=horizon, seed=seed * 1000 + offset, intensity=None
            )
            if not _same(
                draw.get("surface_inputs"),
                {k: inputs for k in ("card", "crown", "reseed", "kernel")},
            ):
                errors.append(dp + "surface_inputs_mismatch")
            fates = draw.get("reseed_fates")
            if not isinstance(fates, Mapping) or not _same(fates.get("inputs"), inputs):
                errors.append(dp + "reseed_inputs_mismatch")
            pins, card = draw.get("pins"), draw.get("card")
            if (
                not isinstance(pins, Mapping)
                or set(pins) != set(_PIN_NAMES)
                or any(type(v) is not bool for v in pins.values())
                or not isinstance(card, Mapping)
                or not isinstance(fates, Mapping)
            ):
                errors.append(dp + "measurements_invalid")
                valid = False
                continue
            fields = [
                (draw, "instant_signed_ticks"),
                (draw, "k200"),
                (card, "life_events_p50_executed"),
            ]
            fields += [
                (fates, k)
                for k in ("reseed_rate_500", "reseed_as_touch_share", "reseed_latency_p50")
            ]
            if any(k not in d or (d[k] is not None and not _finite(d[k])) for d, k in fields):
                errors.append(dp + "measurements_invalid")
                valid = False
                continue
            life = card["life_events_p50_executed"]
            latency = fates["reseed_latency_p50"]
            rates = [fates["reseed_rate_500"], fates["reseed_as_touch_share"]]
            if (
                (life is not None and life < 0)
                or (latency is not None and latency < 0)
                or any(v is not None and not 0 <= v <= 1 for v in rates)
            ):
                errors.append(dp + "measurement_domain_invalid")
                valid = False
                continue
            expected_reseed = {
                "reseed_rate": _inside(fates["reseed_rate_500"], 0.7 * 0.5376, 1.3 * 0.5376),
                "reseed_touch": _inside(fates["reseed_as_touch_share"], 0.6, 1.0),
            }
            if any(pins[k] is not v for k, v in expected_reseed.items()):
                errors.append(dp + "reseed_pin_mismatch")
            if not _same(draw.get("n_pins"), sum(pins.values())):
                errors.append(dp + "n_pins_mismatch")
        if not valid:
            continue
        summary = summaries(draws)
        for k, v in summary.items():
            if not _same(cell.get(k), v):
                errors.append(prefix + k + "_mismatch")
        derived.append(summary)
    if len(derived) == len(_CELLS):
        expected_claims = claims(derived)
        recorded = payload.get("claims")
        if not isinstance(recorded, Mapping) or set(recorded) != set(expected_claims):
            errors.append("joint_tune_claim_set_mismatch")
        for key, value in expected_claims.items():
            if not isinstance(recorded, Mapping) or recorded.get(key) is not value:
                errors.append(key + "_mismatch")
    return errors
