"""Dynamic workflow schedule for /superpower execution."""

from __future__ import annotations

from fx1.interactive.superpower import CAPABILITIES, KIND_HARNESS
from fx1.interactive.workflow import StageKind, build_superpower_schedule, iter_stage_keys


def test_schedule_preserves_spine_and_harness_expansion() -> None:
    selected = [
        "flash_retrieve",
        "research",
        "doctor",
        "web_research",
        "flash_save",
        "synthesize",
    ]
    schedule = build_superpower_schedule(selected, CAPABILITIES)
    keys = list(iter_stage_keys(schedule))
    assert keys[0] == StageKind.FLASH_RETRIEVE.value
    assert keys[1] == StageKind.WEB_RESEARCH.value
    harness_keys = [k for k in keys if k.startswith("harness:")]
    assert harness_keys == ["harness:research", "harness:doctor"]
    assert keys[-2] == StageKind.FLASH_SAVE.value
    assert keys[-1] == StageKind.SYNTHESIZE.value


def test_schedule_skips_absent_front_door_stages() -> None:
    harness_only = [c for c, m in CAPABILITIES.items() if m["kind"] == KIND_HARNESS][:1]
    schedule = build_superpower_schedule(harness_only, CAPABILITIES)
    kinds = [node.kind for node in schedule]
    assert StageKind.FLASH_RETRIEVE not in kinds
    assert kinds == [StageKind.HARNESS]
