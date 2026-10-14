"""Dynamic workflow loop for dipcatcher interactive orchestration.

/superpower plans are static lists, but execution is a *dynamic* loop: each
stage consults live cancellation, wall budgets, and approval gates before
mutating shared context. Harness capabilities expand into one node per planned
id while preserving the dependency spine (memory → web → harness* → save →
synthesis).

The schedule builder is pure; :func:`drive_workflow` is the only runtime loop.
That split keeps the concierge shell and tests able to reason about ordering
without re-implementing admission control in every stage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

WORKFLOW_SCHEMA = "dipcatcher.workflow.v2"


class StageKind(StrEnum):
    FLASH_RETRIEVE = "flash_retrieve"
    WEB_RESEARCH = "web_research"
    HARNESS = "harness"
    FLASH_SAVE = "flash_save"
    SYNTHESIZE = "synthesize"


@dataclass(frozen=True)
class StageNode:
    """One executable step in a superpower run."""

    kind: StageKind
    cap_id: str | None = None

    @property
    def key(self) -> str:
        if self.kind is StageKind.HARNESS:
            assert self.cap_id is not None
            return f"harness:{self.cap_id}"
        return self.kind.value


class StageHandler(Protocol):
    def __call__(
        self,
        node: StageNode,
        *,
        plan: Any,
        ctx: Any,
        result: Any,
        parts: list[str],
        selected: list[str],
    ) -> None: ...


def build_superpower_schedule(
    selected_ids: Iterable[str],
    capabilities: Mapping[str, Mapping[str, Any]],
) -> tuple[StageNode, ...]:
    """Expand a deduplicated plan id list into ordered stage nodes."""
    selected = list(dict.fromkeys(selected_ids))
    nodes: list[StageNode] = []
    if "flash_retrieve" in selected:
        nodes.append(StageNode(StageKind.FLASH_RETRIEVE))
    if "web_research" in selected:
        nodes.append(StageNode(StageKind.WEB_RESEARCH))
    for cap_id in selected:
        meta = capabilities.get(cap_id)
        if meta and meta.get("kind") == "harness":
            nodes.append(StageNode(StageKind.HARNESS, cap_id=cap_id))
    if "flash_save" in selected:
        nodes.append(StageNode(StageKind.FLASH_SAVE))
    if "synthesize" in selected:
        nodes.append(StageNode(StageKind.SYNTHESIZE))
    return tuple(nodes)


def drive_workflow(
    schedule: Iterable[StageNode],
    *,
    plan: Any,
    ctx: Any,
    result: Any,
    parts: list[str],
    selected: list[str],
    handlers: Mapping[StageKind, StageHandler],
    should_abort: Callable[[], bool],
) -> None:
    """Run *schedule* until completion or *should_abort* returns True."""
    for node in schedule:
        if should_abort():
            return
        handler = handlers[node.kind]
        handler(
            node,
            plan=plan,
            ctx=ctx,
            result=result,
            parts=parts,
            selected=selected,
        )


def iter_stage_keys(schedule: Iterable[StageNode]) -> Iterator[str]:
    for node in schedule:
        yield node.key
