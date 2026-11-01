"""Unit tests for ``fx1.eval.timepart``.

Post-cutoff partitioning is the contamination control for the ship gate: a
task whose source artifact was created *after* the base model's knowledge
cutoff cannot have been memorised from pretraining. The module's honesty
property is that **undated** tasks are excluded from both headline slices
rather than being counted as either pre- or post-cutoff evidence.
"""

from __future__ import annotations

import pytest

from fx1.eval.suite import EvalTask
from fx1.eval.timepart import TimePartition, partition_tasks, post_cutoff_pass_rate

CUTOFF = "2023-09-01"


def _task(name: str) -> EvalTask:
    return EvalTask(name=name, kind="domain", messages=[{"role": "user", "content": "q"}])


def test_tasks_are_split_by_creation_date_against_the_cutoff() -> None:
    tasks = [_task("early"), _task("late"), _task("same_day")]
    created = {
        "early": "2023-08-31",
        "late": "2023-09-02",
        "same_day": CUTOFF,
    }
    partition = partition_tasks(tasks, created, CUTOFF)
    assert partition.cutoff == CUTOFF
    assert partition.pre_cutoff == ["early", "same_day"]
    assert partition.post_cutoff == ["late"]
    assert partition.undated == []


def test_undated_tasks_are_excluded_from_both_slices() -> None:
    """Undated evidence is not time-evidence — it belongs to neither headline."""
    tasks = [_task("dated"), _task("undated")]
    partition = partition_tasks(tasks, {"dated": "2024-01-01"}, CUTOFF)
    assert partition.post_cutoff == ["dated"]
    assert partition.pre_cutoff == []
    assert partition.undated == ["undated"]
    assert set(partition.pre_cutoff) & set(partition.post_cutoff) == set()


def test_missing_creation_dates_land_in_undated() -> None:
    partition = partition_tasks([_task("a")], {}, CUTOFF)
    assert partition.undated == ["a"]
    assert partition.pre_cutoff == []
    assert partition.post_cutoff == []


def test_post_cutoff_pass_rate_is_none_for_an_empty_slice() -> None:
    partition = partition_tasks([_task("a")], {"a": "2020-01-01"}, CUTOFF)
    assert partition.post_cutoff == []
    assert post_cutoff_pass_rate(partition, {}) is None


def test_post_cutoff_pass_rate_counts_only_the_post_cutoff_slice() -> None:
    partition = partition_tasks(
        [_task("pre"), _task("post1"), _task("post2")],
        {"pre": "2020-01-01", "post1": "2024-01-01", "post2": "2024-01-02"},
        CUTOFF,
    )
    rate = post_cutoff_pass_rate(partition, {"pre": False, "post1": True, "post2": False})
    assert rate == pytest.approx(0.5)


def test_post_cutoff_pass_rate_treats_a_missing_result_as_a_failure() -> None:
    """An unrecorded result is not a pass — the rate must not silently improve."""
    partition = partition_tasks(
        [_task("post1"), _task("post2")],
        {"post1": "2024-01-01", "post2": "2024-01-02"},
        CUTOFF,
    )
    assert post_cutoff_pass_rate(partition, {"post1": True}) == pytest.approx(0.5)


def test_post_cutoff_pass_rate_is_one_when_every_post_task_passes() -> None:
    partition = partition_tasks([_task("post1")], {"post1": "2024-01-01"}, CUTOFF)
    assert post_cutoff_pass_rate(partition, {"post1": True}) == pytest.approx(1.0)


def test_time_partition_defaults_are_empty_and_not_shared_between_instances() -> None:
    """Two partitions must not share the same default list objects."""
    first = TimePartition(cutoff=CUTOFF)
    second = TimePartition(cutoff=CUTOFF)
    first.pre_cutoff.append("x")
    assert second.pre_cutoff == []
    assert first.post_cutoff == []
    assert first.undated == []
