"""Unit tests for ``quant_fund.mc_engine.engine_config``.

The fingerprint is the checkpoint identity for one scenario-risk run. Its
whole point is that it covers **every input that changes scenario numbers**
and nothing else. Two failure modes matter and they pull in opposite
directions:

* a number input that changes the numbers but not the fingerprint would let a
  checkpoint be resumed under a different scenario (silent wrong answer);
* a *control-plane* setting (worker count, backend, checkpoint directory, the
  early-stop hook) that leaked into the fingerprint would make two numerically
  identical runs look different, so the receipt chain could never match.

These tests pin both directions.
"""

from __future__ import annotations

import dataclasses

import pytest

from quant_fund.mc_engine.engine_config import (
    BACKENDS,
    MC_ENGINE_VERSION,
    MEMORY_MODES,
    SHOCK_MODES,
    EngineConfig,
    config_fingerprint,
    config_from_payload,
    fingerprint_payload,
    resolved_importance_shift,
)

SPEC = {"generator": "garch", "n_factors": 3}


def _config(**overrides: object) -> EngineConfig:
    base: dict[str, object] = {"n_paths": 1024, "seed": 11}
    base.update(overrides)
    return EngineConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Fingerprint coverage
# --------------------------------------------------------------------------- #


def test_payload_contains_every_scenario_affecting_input() -> None:
    payload = fingerprint_payload(SPEC, _config())
    for key in (
        "n_paths",
        "chunk_size",
        "seed",
        "shock_mode",
        "importance_shift",
        "qmc_scramble",
        "n_scrambles",
        "control_variate",
        "pilot_every",
        "memory_mode",
        "ruin_level",
        "es_levels",
        "ci_level",
        "tdigest_compression",
        "measure_variance_reduction",
        "evt_threshold",
        "max_exceedances_per_chunk",
        "generator",
        "version",
    ):
        assert key in payload, f"{key} must be covered by the fingerprint"


def test_payload_excludes_the_control_plane_settings() -> None:
    """Worker count, backend, checkpoint dir and the stop hook are not inputs."""
    payload = fingerprint_payload(SPEC, _config())
    for key in ("workers", "backend", "checkpoint_dir", "stop_after_new_chunks"):
        assert key not in payload


def test_control_plane_changes_do_not_move_the_fingerprint() -> None:
    base = _config()
    moved = dataclasses.replace(
        base,
        workers=8,
        backend="ray",
        checkpoint_dir="/tmp/checkpoints",
        stop_after_new_chunks=3,
    )
    assert config_fingerprint(SPEC, base) == config_fingerprint(SPEC, moved)


@pytest.mark.parametrize(
    "field, value",
    [
        ("n_paths", 2048),
        ("chunk_size", 512),
        ("seed", 12),
        ("shock_mode", "antithetic"),
        ("importance_shift", 0.25),
        ("qmc_scramble", False),
        ("n_scrambles", 4),
        ("control_variate", True),
        ("pilot_every", 3),
        ("memory_mode", "sketch"),
        ("ruin_level", 0.25),
        ("es_levels", (0.9, 0.99)),
        ("ci_level", 0.9),
        ("tdigest_compression", 50.0),
        ("measure_variance_reduction", False),
        ("evt_threshold", 0.2),
        ("max_exceedances_per_chunk", 10),
    ],
)
def test_every_scenario_input_moves_the_fingerprint(field: str, value: object) -> None:
    base = _config()
    changed = dataclasses.replace(base, **{field: value})
    assert config_fingerprint(SPEC, base) != config_fingerprint(SPEC, changed)


def test_the_generator_spec_is_covered_by_the_fingerprint() -> None:
    base = _config()
    other = {**SPEC, "n_factors": 4}
    assert config_fingerprint(SPEC, base) != config_fingerprint(other, base)


def test_fingerprint_is_deterministic_across_calls() -> None:
    config = _config()
    assert config_fingerprint(SPEC, config) == config_fingerprint(SPEC, config)


def test_fingerprint_is_a_sha256_hex_digest() -> None:
    digest = config_fingerprint(SPEC, _config())
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_fingerprint_rejects_a_non_finite_input() -> None:
    """A NaN parameter cannot be silently serialised into a checkpoint id."""
    with pytest.raises(ValueError):
        config_fingerprint(SPEC, _config(importance_shift=float("nan")))


# --------------------------------------------------------------------------- #
# Payload round-trip
# --------------------------------------------------------------------------- #


def test_payload_round_trips_into_an_equivalent_config() -> None:
    config = _config()
    payload = fingerprint_payload(SPEC, config)
    restored = config_from_payload(payload, workers=4, backend="serial")
    assert restored == dataclasses.replace(config, workers=4, backend="serial")


def test_round_trip_preserves_the_optional_thresholds() -> None:
    config = _config(importance_shift=0.25, evt_threshold=0.1)
    payload = fingerprint_payload(SPEC, config)
    restored = config_from_payload(payload, workers=1, backend="serial")
    assert restored.importance_shift == pytest.approx(0.25)
    assert restored.evt_threshold == pytest.approx(0.1)


def test_round_trip_keeps_a_none_threshold_none() -> None:
    config = _config(evt_threshold=None)
    payload = fingerprint_payload(SPEC, config)
    restored = config_from_payload(payload, workers=1, backend="serial")
    assert restored.evt_threshold is None


def test_round_trip_restores_the_es_levels_as_a_tuple() -> None:
    payload = fingerprint_payload(SPEC, _config(es_levels=(0.9, 0.95, 0.99)))
    assert payload["es_levels"] == [0.9, 0.95, 0.99]
    restored = config_from_payload(payload, workers=1, backend="serial")
    assert restored.es_levels == (0.9, 0.95, 0.99)


def test_round_trip_takes_the_worker_and_backend_from_the_caller() -> None:
    """Those two are control plane — the caller supplies them, not the payload."""
    payload = fingerprint_payload(SPEC, _config())
    restored = config_from_payload(payload, workers=16, backend="process")
    assert restored.workers == 16
    assert restored.backend == "process"


# --------------------------------------------------------------------------- #
# Importance shift
# --------------------------------------------------------------------------- #


def test_unset_importance_shift_resolves_to_a_unit_norm_shift() -> None:
    """``None`` means -1/sqrt(n_steps): the squared norm of the shift is 1."""
    shift = resolved_importance_shift(_config(importance_shift=None), 10_000)
    assert shift == pytest.approx(-1.0 / 100.0)
    assert shift**2 * 10_000 == pytest.approx(1.0)


def test_an_explicit_importance_shift_is_used_verbatim() -> None:
    config = _config(importance_shift=0.25)
    assert resolved_importance_shift(config, 10_000) == pytest.approx(0.25)


def test_an_explicit_zero_shift_stays_zero() -> None:
    config = _config(importance_shift=0.0)
    assert resolved_importance_shift(config, 10_000) == 0.0


# --------------------------------------------------------------------------- #
# Declared vocabularies
# --------------------------------------------------------------------------- #


def test_the_declared_vocabularies_are_pinned() -> None:
    assert MC_ENGINE_VERSION == 1
    assert frozenset({"crude", "antithetic", "qmc_sobol", "importance"}) == SHOCK_MODES
    assert frozenset({"serial", "process", "ray"}) == BACKENDS
    assert frozenset({"exact", "sketch"}) == MEMORY_MODES
