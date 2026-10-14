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

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from fx1.harness import HarnessRole

WORKFLOW_SCHEMA = "dipcatcher.workflow.v3"
_PARALLEL_VERIFICATION_ROLES = frozenset({HarnessRole.VERIFICATION.value})
_MAX_PARALLEL_VERIFICATION = 3


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
    harness_wave: tuple[str, ...] | None = None

    @property
    def key(self) -> str:
        if self.kind is StageKind.HARNESS:
            if self.harness_wave and len(self.harness_wave) > 1:
                return "harness-wave:" + "+".join(self.harness_wave)
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
    harness_nodes: list[StageNode] = []
    for cap_id in selected:
        meta = capabilities.get(cap_id)
        if meta and meta.get("kind") == "harness":
            harness_nodes.append(StageNode(StageKind.HARNESS, cap_id=cap_id))
    nodes.extend(coalesce_parallel_harness_waves(harness_nodes, capabilities))
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


def _parallel_harness_eligible(meta: Mapping[str, Any]) -> bool:
    return (
        meta.get("kind") == "harness"
        and not meta.get("consequential")
        and meta.get("role") in _PARALLEL_VERIFICATION_ROLES
    )


def coalesce_parallel_harness_waves(
    nodes: list[StageNode],
    capabilities: Mapping[str, Mapping[str, Any]],
) -> tuple[StageNode, ...]:
    """Merge consecutive verification harness nodes into one parallel wave."""
    out: list[StageNode] = []
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if node.kind is not StageKind.HARNESS or node.cap_id is None:
            out.append(node)
            index += 1
            continue
        meta = capabilities.get(node.cap_id, {})
        if not _parallel_harness_eligible(meta):
            out.append(node)
            index += 1
            continue
        wave = [node.cap_id]
        role = meta.get("role")
        scan = index + 1
        while scan < len(nodes) and len(wave) < _MAX_PARALLEL_VERIFICATION:
            nxt = nodes[scan]
            if nxt.kind is not StageKind.HARNESS or nxt.cap_id is None:
                break
            nmeta = capabilities.get(nxt.cap_id, {})
            if _parallel_harness_eligible(nmeta) and nmeta.get("role") == role:
                wave.append(nxt.cap_id)
                scan += 1
            else:
                break
        if len(wave) == 1:
            out.append(node)
        else:
            out.append(
                StageNode(
                    StageKind.HARNESS,
                    cap_id=wave[0],
                    harness_wave=tuple(wave),
                )
            )
        index = scan
    return tuple(out)


def run_parallel_harness_wave(
    wave: Sequence[str],
    *,
    run_one: Callable[[str], None],
    cancel_requested: Callable[[], bool],
) -> None:
    """Execute *wave* with bounded parallelism when len>1."""
    if len(wave) <= 1:
        if not cancel_requested():
            run_one(wave[0])
        return
    if cancel_requested():
        return
    with ThreadPoolExecutor(max_workers=min(len(wave), _MAX_PARALLEL_VERIFICATION)) as pool:
        futures = [pool.submit(run_one, cap_id) for cap_id in wave]
        for future in as_completed(futures):
            future.result()


def workflow_parallel_policy() -> dict[str, Any]:
    """Serialised parallel-harness policy for surface seals."""
    return {
        "roles": sorted(_PARALLEL_VERIFICATION_ROLES),
        "max_parallel": _MAX_PARALLEL_VERIFICATION,
    }
