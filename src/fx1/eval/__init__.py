"""fx-1 evaluation harness — built before any training runs.

PEP 562 lazy facade: submodule symbols resolve on first attribute access
so light consumers (``fx1 doctor``, ``--help``) never pay the
capability/calibration import chain. ``fx1.eval.capability`` and friends
import each other directly — that path stays eager by design.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fx1.eval.bank import (
        DEFAULT_BANK as DEFAULT_BANK,
    )
    from fx1.eval.bank import (
        DOMAIN_TASKS as DOMAIN_TASKS,
    )
    from fx1.eval.bank import (
        GENERAL_TASKS as GENERAL_TASKS,
    )
    from fx1.eval.bank import (
        HONESTY_BAITS as HONESTY_BAITS,
    )
    from fx1.eval.capability import (
        CapabilityEvalReport as CapabilityEvalReport,
    )
    from fx1.eval.capability import (
        run_capability_eval as run_capability_eval,
    )
    from fx1.eval.compare import (
        ComparisonResult as ComparisonResult,
    )
    from fx1.eval.compare import (
        compare_runs as compare_runs,
    )
    from fx1.eval.contamination import (
        ContaminationReport as ContaminationReport,
    )
    from fx1.eval.contamination import (
        run_contamination_audit as run_contamination_audit,
    )
    from fx1.eval.masking import (
        MemoryGapReport as MemoryGapReport,
    )
    from fx1.eval.masking import (
        mask_task as mask_task,
    )
    from fx1.eval.masking import (
        masked_twins as masked_twins,
    )
    from fx1.eval.masking import (
        memory_gap_report as memory_gap_report,
    )
    from fx1.eval.redteam import REDTEAM_TASKS as REDTEAM_TASKS
    from fx1.eval.rephrased import (
        rephrased_twins as rephrased_twins,
    )
    from fx1.eval.rephrased import (
        run_rephrased_gap as run_rephrased_gap,
    )
    from fx1.eval.suite import (
        EvalResult as EvalResult,
    )
    from fx1.eval.suite import (
        EvalTask as EvalTask,
    )
    from fx1.eval.suite import (
        run_suite as run_suite,
    )
    from fx1.eval.timepart import (
        TimePartition as TimePartition,
    )
    from fx1.eval.timepart import (
        partition_tasks as partition_tasks,
    )
    from fx1.eval.timepart import (
        post_cutoff_pass_rate as post_cutoff_pass_rate,
    )

_EXPORTS = {
    "bank": ("DEFAULT_BANK", "DOMAIN_TASKS", "GENERAL_TASKS", "HONESTY_BAITS"),
    "capability": ("CapabilityEvalReport", "run_capability_eval"),
    "compare": ("ComparisonResult", "compare_runs"),
    "contamination": ("ContaminationReport", "run_contamination_audit"),
    "masking": ("MemoryGapReport", "mask_task", "masked_twins", "memory_gap_report"),
    "redteam": ("REDTEAM_TASKS",),
    "rephrased": ("rephrased_twins", "run_rephrased_gap"),
    "suite": ("EvalResult", "EvalTask", "run_suite"),
    "timepart": ("TimePartition", "partition_tasks", "post_cutoff_pass_rate"),
}
_ATTR_TO_MODULE = {attr: mod for mod, attrs in _EXPORTS.items() for attr in attrs}
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

__all__ = sorted(set(_ATTR_TO_MODULE) | {"eval_prompt_surface"})


def eval_prompt_surface() -> list[str]:
    """Every message content the eval surface presents to a model.
    The decontamination target: canonical bank prompts, red-team tasks,
    rephrased twins, and masked twins — instruction text AND seeded
    completions, any role. A corpus may copy an eval item in *any* of
    those surface forms — the quality gate and the contamination audit
    must screen against all of them, not only the canonical bank.
    """
    # Resolve the lazy-facade names explicitly: the module's PEP 562
    # __getattr__ fires only on external attribute access, never on bare
    # global lookups inside this module — referencing DEFAULT_BANK &co.
    # directly raised NameError unless another import had materialized them.
    from fx1.eval.bank import DEFAULT_BANK, DOMAIN_TASKS, HONESTY_BAITS
    from fx1.eval.masking import masked_twins
    from fx1.eval.redteam import REDTEAM_TASKS
    from fx1.eval.rephrased import rephrased_twins

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
            if message["content"] not in seen:
                seen.add(message["content"])
                prompts.append(message["content"])
    return prompts


_LAZY_CAPABILITY = frozenset({"CapabilityEvalReport", "run_capability_eval"})


def __getattr__(name: str) -> Any:
    module = _ATTR_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"fx1.eval.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
