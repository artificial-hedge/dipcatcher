"""fx-1 evaluation harness — built before any training runs."""

from fx1.eval.bank import DEFAULT_BANK, DOMAIN_TASKS, GENERAL_TASKS, HONESTY_BAITS
from fx1.eval.compare import (
    ComparisonResult,
    PromotionGate,
    PromotionGateError,
    compare_runs,
    evaluate_promotion,
    is_refusal,
    refusal_rate,
)
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
    "ComparisonResult",
    "ContaminationReport",
    "EvalResult",
    "EvalTask",
    "MemoryGapReport",
    "PromotionGate",
    "PromotionGateError",
    "TimePartition",
    "compare_runs",
    "evaluate_promotion",
    "is_refusal",
    "mask_task",
    "masked_twins",
    "memory_gap_report",
    "partition_tasks",
    "post_cutoff_pass_rate",
    "refusal_rate",
    "rephrased_twins",
    "run_contamination_audit",
    "run_rephrased_gap",
    "run_suite",
]
