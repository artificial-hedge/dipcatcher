"""fx-1 evaluation harness — built before any training runs.

Heavy capability-battery imports (calibration / ts-reasoning pull sklearn
and the lab stack) stay lazy so ``fx1 doctor`` and other light consumers of
``DEFAULT_BANK`` stay under the cold-start budget.
"""

from __future__ import annotations

from typing import Any

from fx1.eval.bank import DEFAULT_BANK, DOMAIN_TASKS, GENERAL_TASKS, HONESTY_BAITS
from fx1.eval.compare import ComparisonResult, compare_runs
from fx1.eval.contamination import ContaminationReport, run_contamination_audit
from fx1.eval.masking import MemoryGapReport, mask_task, masked_twins, memory_gap_report
from fx1.eval.redteam import REDTEAM_TASKS
from fx1.eval.rephrased import rephrased_twins, run_rephrased_gap
from fx1.eval.suite import EvalResult, EvalTask, run_suite
from fx1.eval.timepart import TimePartition, partition_tasks, post_cutoff_pass_rate

__all__ = [
    "DEFAULT_BANK",
    "DOMAIN_TASKS",
    "GENERAL_TASKS",
    "HONESTY_BAITS",
    "REDTEAM_TASKS",
    "CapabilityEvalReport",
    "ComparisonResult",
    "ContaminationReport",
    "EvalResult",
    "EvalTask",
    "MemoryGapReport",
    "TimePartition",
    "compare_runs",
    "eval_prompt_surface",
    "mask_task",
    "masked_twins",
    "memory_gap_report",
    "partition_tasks",
    "post_cutoff_pass_rate",
    "rephrased_twins",
    "run_capability_eval",
    "run_contamination_audit",
    "run_rephrased_gap",
    "run_suite",
]


def eval_prompt_surface() -> list[str]:
    """Every user prompt the eval surface presents to a model.

    The decontamination target: canonical bank prompts, red-team tasks,
    rephrased twins, and masked twins. A corpus may copy an eval item in
    *any* of those surface forms — the quality gate and the contamination
    audit must screen against all of them, not only the canonical bank.
    """
    tasks = (
        list(DEFAULT_BANK)
        + list(REDTEAM_TASKS)
        + rephrased_twins([*HONESTY_BAITS, *DOMAIN_TASKS])
        + masked_twins(list(DEFAULT_BANK) + list(REDTEAM_TASKS))
    )
    prompts: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        for message in task.messages:
            if message["role"] == "user" and message["content"] not in seen:
                seen.add(message["content"])
                prompts.append(message["content"])
    return prompts


_LAZY_CAPABILITY = frozenset({"CapabilityEvalReport", "run_capability_eval"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_CAPABILITY:
        from fx1.eval.capability import CapabilityEvalReport, run_capability_eval

        exports = {
            "CapabilityEvalReport": CapabilityEvalReport,
            "run_capability_eval": run_capability_eval,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
