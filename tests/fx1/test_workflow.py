"""Dynamic workflow schedule for /superpower execution."""

from __future__ import annotations

import threading
import time

from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.superpower import (
    CAPABILITIES,
    KIND_HARNESS,
    PlannedCapability,
    SuperpowerContext,
    SuperpowerPlan,
    run_plan,
)
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
    harness_keys = [k for k in keys if k.startswith("harness")]
    assert harness_keys == ["harness:research", "harness:doctor"]
    assert keys[-2] == StageKind.FLASH_SAVE.value
    assert keys[-1] == StageKind.SYNTHESIZE.value


def test_schedule_skips_absent_front_door_stages() -> None:
    harness_only = [c for c, m in CAPABILITIES.items() if m["kind"] == KIND_HARNESS][:1]
    schedule = build_superpower_schedule(harness_only, CAPABILITIES)
    kinds = [node.kind for node in schedule]
    assert StageKind.FLASH_RETRIEVE not in kinds
    assert kinds == [StageKind.HARNESS]


def test_schedule_coalesces_verification_harness_wave() -> None:
    selected = ["doctor", "verify-research", "research", "synthesize"]
    keys = list(iter_stage_keys(build_superpower_schedule(selected, CAPABILITIES)))
    assert "harness-wave:doctor+verify-research" in keys
    assert "harness:research" in keys


class _HarnessResult:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def test_run_plan_overlaps_verification_harnesses() -> None:
    class SlowHarness:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self._lock = threading.Lock()

        def run(self, name: str, extra_args: list[str] | None = None) -> _HarnessResult:
            with self._lock:
                self.calls.append(name)
            time.sleep(0.05)
            return _HarnessResult(0, "DATA_LABEL=SYNTHETIC\nok\n", "")

    harness = SlowHarness()
    plan = SuperpowerPlan(
        goal="verify",
        capabilities=[
            PlannedCapability("doctor", "a"),
            PlannedCapability("verify-research", "b"),
            PlannedCapability("synthesize", "c"),
        ],
    )
    start = time.monotonic()
    run_plan(plan, SuperpowerContext(model_fn=None, harness=harness, approvals=ApprovalGate()))
    elapsed = time.monotonic() - start
    assert set(harness.calls) == {"doctor", "verify-research"}
    assert elapsed < 0.09
