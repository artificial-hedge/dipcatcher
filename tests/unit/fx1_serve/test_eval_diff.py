"""KATs for ``diff_eval_records`` — the promotion-gate primitive."""

from __future__ import annotations

from typing import Any

from fx1.serve.evals import EvalRecord, diff_eval_records


def _rec(
    eval_id: str,
    *,
    suite: str = "tooluse",
    backend: str = "byok",
    seed: int = 0,
    results: list[dict[str, Any]] | None = None,
    gate: bool | None = True,
    by_kind: dict[str, Any] | None = None,
    bank: str | None = "b" * 64,
) -> EvalRecord:
    report = None
    if results is not None or gate is not None or by_kind is not None:
        report = {
            "results": results or [],
            "honesty_gate_passed": gate,
            "by_kind": by_kind or {},
        }
        if bank is not None:
            report["eval_bank_sha256"] = bank
    return EvalRecord(
        eval_id=eval_id,
        suite=suite,
        backend=backend,
        seed=seed,
        status="succeeded",
        created_at=1.0,
        finished_at=2.0,
        report=report,
    )


def test_identical_reports_unchanged():
    results = [{"task": "t1", "kind": "x", "passed": True}]
    a = _rec("a", results=results, by_kind={"x": {"passed": 1}})
    b = _rec("b", results=results, by_kind={"x": {"passed": 1}})
    d = diff_eval_records(a, b)
    assert d.comparable and d.same_suite and d.same_seed and d.same_bank
    assert d.verdict == "unchanged"
    assert d.tasks_fixed == [] and d.tasks_regressed == []
    assert d.deltas == [] and d.gate_transition == "unchanged"


def test_regression_verdict():
    a = _rec(
        "a",
        results=[{"task": "t1", "kind": "x", "passed": True}],
        by_kind={"x": {"passed": 1, "total": 1}},
    )
    b = _rec(
        "b",
        results=[{"task": "t1", "kind": "x", "passed": False}],
        by_kind={"x": {"passed": 0, "total": 1}},
    )
    d = diff_eval_records(a, b)
    assert d.verdict == "regressed"
    assert d.tasks_regressed == ["t1"]
    assert [x.path for x in d.deltas] == ["x.passed"]
    assert d.deltas[0].delta == -1.0


def test_fixed_and_gate_opened_improve():
    a = _rec(
        "a",
        results=[{"task": "t1", "kind": "x", "passed": False}],
        gate=False,
    )
    b = _rec(
        "b",
        results=[{"task": "t1", "kind": "x", "passed": True}],
        gate=True,
    )
    d = diff_eval_records(a, b)
    assert d.verdict == "improved"
    assert d.tasks_fixed == ["t1"]
    assert d.gate_transition == "opened"


def test_gate_close_alone_regresses():
    a = _rec("a", results=[], gate=True)
    b = _rec("b", results=[], gate=False)
    d = diff_eval_records(a, b)
    assert d.verdict == "regressed" and d.gate_transition == "closed"


def test_cross_suite_incomparable():
    a = _rec("a", suite="tooluse", results=[])
    b = _rec("b", suite="capability", results=[])
    d = diff_eval_records(a, b)
    assert not d.same_suite and not d.comparable and d.verdict == "unknown"


def test_cross_bank_incomparable():
    a = _rec("a", results=[{"task": "t1", "kind": "x", "passed": True}], bank="a" * 64)
    b = _rec("b", results=[{"task": "t1", "kind": "x", "passed": False}], bank="c" * 64)
    d = diff_eval_records(a, b)
    assert d.same_suite and not d.same_bank and not d.comparable
    assert d.verdict == "unknown"
    # a stamped mismatch makes task-name pairing meaningless
    assert d.tasks_regressed == [] and d.tasks_fixed == []


def test_task_coverage_gaps_listed():
    a = _rec("a", results=[{"task": "gone", "kind": "x", "passed": True}])
    b = _rec("b", results=[{"task": "new", "kind": "x", "passed": True}])
    d = diff_eval_records(a, b)
    assert d.tasks_only_base == ["gone"] and d.tasks_only_candidate == ["new"]
    assert d.verdict == "unchanged"  # same bank, no shared-task flip


def test_missing_gate_unknown_transition():
    a = _rec("a", results=[], gate=None)
    b = _rec("b", results=[], gate=True)
    assert diff_eval_records(a, b).gate_transition == "unknown"


def test_seed_mismatch_incomparable():
    a = _rec("a", results=[], seed=0)
    b = _rec("b", results=[], seed=1)
    d = diff_eval_records(a, b)
    assert not d.same_seed and not d.comparable and d.verdict == "unknown"


def test_unstamped_bank_diffable_when_suite_seed_match():
    a = _rec("a", results=[{"task": "t1", "kind": "x", "passed": True}], bank=None)
    b = _rec("b", results=[{"task": "t1", "kind": "x", "passed": False}], bank=None)
    d = diff_eval_records(a, b)
    assert not d.same_bank and d.comparable and d.verdict == "regressed"


def test_significance_exact_sign_test():
    # 8 fixed / 1 regressed discordant pairs → binomtest(1, 9, 0.5) = 0.0390625
    a = _rec(
        "a",
        results=[{"task": f"t{i}", "kind": "x", "passed": i == 0} for i in range(9)],
    )
    b = _rec(
        "b",
        results=[{"task": f"t{i}", "kind": "x", "passed": i != 0} for i in range(9)],
    )
    d = diff_eval_records(a, b)
    assert d.significance is not None
    assert d.significance.n_fixed == 8 and d.significance.n_regressed == 1
    assert d.significance.p_value == 0.0390625 and d.significance.significant_p05


def test_significance_null_when_incomparable():
    a = _rec("a", results=[], suite="tooluse")
    b = _rec("b", results=[], suite="capability")
    assert diff_eval_records(a, b).significance is None


def test_significance_zero_discordant_is_one():
    a = _rec("a", results=[{"task": "t", "kind": "x", "passed": True}])
    b = _rec("b", results=[{"task": "t", "kind": "x", "passed": True}])
    d = diff_eval_records(a, b)
    assert d.significance is not None and d.significance.p_value == 1.0
    assert not d.significance.significant_p05


def test_bool_leaves_excluded_from_deltas():
    a = _rec("a", results=[], by_kind={"x": {"passed": 1, "gate": True}})
    b = _rec("b", results=[], by_kind={"x": {"passed": 1, "gate": False}})
    d = diff_eval_records(a, b)
    assert d.deltas == []  # bools are not numeric leaves
