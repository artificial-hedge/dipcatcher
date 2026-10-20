"""/superpower: capability registry, planning, coordination, approvals, honesty.

Offline: harness, researcher, model, and store seams are injected. The
registry tests assert the fail-closed property that matters most — every
harness capability maps to a registered command, and consequential commands
are exactly the ones gated behind approval.
"""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fx1.flash.store import FlashStore
from fx1.harness import HARNESS_REGISTRY
from fx1.interactive.approvals import ApprovalGate, console_ask_fn
from fx1.interactive.budget import BudgetLedger, RunBudget
from fx1.interactive.families import FamilyCatalog, goal_sha256
from fx1.interactive.superpower import (
    _REQUIRED_HARNESS_INPUTS,
    CAPABILITIES,
    KIND_HARNESS,
    PlannedCapability,
    SuperpowerContext,
    SuperpowerPlan,
    SuperpowerResult,
    describe_capabilities,
    parse_family_selection,
    plan_goal,
    run_plan,
)
from fx1.webresearch.engine import ResearchReport, SourceDoc

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeHarnessResult:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeHarness:
    def __init__(self, stdout: str = "DATA_LABEL=SYNTHETIC\nSYNTHETIC\nran ok\n") -> None:
        self.calls: list[tuple[str, list[str] | None]] = []
        self._stdout = stdout

    def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
        self.calls.append((name, extra_args))
        return FakeHarnessResult(0, self._stdout, "")


class FakeResearcher:
    def __init__(self, report: ResearchReport | None = None) -> None:
        self.calls: list[str] = []
        self._report = report or ResearchReport(
            goal="g",
            sources=[SourceDoc(url="https://a.example/x", title="A", status=200, text="t")],
        )

    def run(self, goal, *, budget=None, queries=None, cancel_event=None, on_progress=None):  # noqa: ANN001
        self.calls.append(goal)
        if on_progress is not None:
            from fx1.webresearch.engine import ResearchProgress

            on_progress(ResearchProgress(event="done", detail="fake"))
        return self._report


@pytest.fixture()
def store(tmp_path: Path) -> FlashStore:
    return FlashStore(tmp_path / "flash" / "entries.jsonl")


def _plan(goal: str, ids: list[str]) -> SuperpowerPlan:
    return SuperpowerPlan(
        goal=goal,
        capabilities=[PlannedCapability(i, "test") for i in ids],
        source="test",
    )


# ---------------------------------------------------------------------------
# capability registry
# ---------------------------------------------------------------------------


def test_every_registered_harness_command_is_a_capability() -> None:
    registered = {c.name for c in HARNESS_REGISTRY}
    harness_caps = {k for k, v in CAPABILITIES.items() if v["kind"] == KIND_HARNESS}
    assert registered == harness_caps


def test_manual_only_map_matches_required_click_inputs() -> None:
    from typer.main import get_command

    from quant_fund.cli.main import app

    cli = get_command(app)
    required = {
        command.name: tuple(
            param.name for param in cli.commands[command.name].params if param.required
        )
        for command in HARNESS_REGISTRY
    }
    required = {name: names for name, names in required.items() if names}
    assert required == {
        name: (input_name,) for name, (input_name, _) in _REQUIRED_HARNESS_INPUTS.items()
    }


def test_consequential_capabilities_are_training_surfaces() -> None:
    consequential = {k for k, v in CAPABILITIES.items() if v.get("consequential")}
    assert consequential == {"train", "optimize", "paper"}


def test_front_door_capabilities_present() -> None:
    for cap in ("web_research", "flash_retrieve", "flash_save", "synthesize"):
        assert cap in CAPABILITIES


def test_describe_capabilities_lists_kinds_and_flags() -> None:
    text = describe_capabilities()
    assert "web_research" in text
    assert "approval-required" in text
    assert "[harness" in text
    assert "validate [harness; manual-only" in text
    assert "/harness run validate MODEL_ID" in text


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def test_plan_selects_research_for_benchmark_goal() -> None:
    plan = plan_goal("run the conformal volatility benchmark family")
    ids = plan.ids()
    assert "research" in ids
    assert "paper" not in ids and "train" not in ids
    assert plan.source == "deterministic"
    # mandatory spine
    assert ids[0] == "flash_retrieve"
    assert ids[-1] == "synthesize"
    assert "flash_save" in ids


def test_plan_selects_web_for_current_events_goal() -> None:
    plan = plan_goal("what is the latest news on the Fed rate decision")
    assert "web_research" in plan.ids()


def test_plan_falls_back_to_web_when_no_harness_match() -> None:
    plan = plan_goal("summarize the history of the dutch tulip market")
    ids = plan.ids()
    assert "web_research" in ids
    assert not any(CAPABILITIES[i]["kind"] == KIND_HARNESS for i in ids)


@pytest.mark.parametrize(
    ("goal", "command", "usage"),
    [
        ("validate a model", "validate", "/harness run validate MODEL_ID"),
        ("collect a public feed", "collect", "/harness run collect --source SOURCE"),
        ("generate a tearsheet", "tearsheet", "/harness run tearsheet --equity PATH"),
    ],
)
def test_required_input_harness_commands_are_excluded_from_plan(
    goal: str, command: str, usage: str
) -> None:
    plan = plan_goal(goal)
    assert command not in plan.ids()
    assert "web_research" not in plan.ids()
    assert f"excluded: {command}: requires" in plan.describe()
    assert usage in plan.describe()


def test_model_cannot_select_required_input_harness_commands() -> None:
    def model_fn(prompt: str) -> str:
        for command in _REQUIRED_HARNESS_INPUTS:
            assert f"- {command}:" not in prompt
        return json.dumps({"capabilities": [*list(_REQUIRED_HARNESS_INPUTS), "doctor"]})

    plan = plan_goal("check readiness", model_fn=model_fn)
    assert plan.ids() == ["flash_retrieve", "doctor", "synthesize"]


def test_model_cannot_add_research_without_loaded_catalog_or_deterministic_match() -> None:
    def model_fn(prompt: str) -> str:
        assert "- research:" not in prompt
        return json.dumps({"capabilities": ["research", "doctor"]})

    plan = plan_goal("check readiness", model_fn=model_fn)
    assert plan.ids() == ["flash_retrieve", "doctor", "synthesize"]


def test_plan_caps_harness_selections() -> None:
    goal = (
        "ingest data, build features and labels, run the research benchmark "
        "family, backtest it, then train and optimize a model"
    )
    plan = plan_goal(goal, max_harness=3)
    harness_ids = [i for i in plan.ids() if CAPABILITIES[i]["kind"] == KIND_HARNESS]
    assert len(harness_ids) <= 3


def test_plan_rejects_negative_harness_limit() -> None:
    with pytest.raises(ValueError, match="max_harness must be nonnegative"):
        plan_goal("ingest, benchmark, backtest, train and optimize", max_harness=-1)


def test_plan_describe_explains_choices() -> None:
    plan = plan_goal("check the northset order book identities")
    text = plan.describe()
    assert "northset" in text
    assert "planner: deterministic" in text


def test_model_refinement_validates_ids() -> None:
    def model_fn(prompt: str) -> str:
        assert "northset" in prompt  # registry listing reached the model
        return (
            '{"capabilities": ["northset", "totally-made-up"], '
            '"rationale": "goal names order book identities"}'
        )

    plan = plan_goal("check the order book identities", model_fn=model_fn)
    assert plan.source == "model+validated"
    ids = plan.ids()
    assert "northset" in ids
    assert "totally-made-up" not in ids  # unknown ids dropped, fail-closed
    assert ids[0] == "flash_retrieve"  # spine re-added
    assert ids[-1] == "synthesize"


@pytest.mark.parametrize("max_harness", [0, 1, 3])
def test_model_refinement_cannot_expand_past_harness_limit(max_harness: int) -> None:
    response = json.dumps({"capabilities": list(CAPABILITIES)})
    plan = plan_goal("check a benchmark", model_fn=lambda _: response, max_harness=max_harness)
    harness_ids = [i for i in plan.ids() if CAPABILITIES[i]["kind"] == KIND_HARNESS]
    expected = [
        i
        for i, meta in CAPABILITIES.items()
        if meta["kind"] == KIND_HARNESS and i not in _REQUIRED_HARNESS_INPUTS
    ]
    assert plan.source == "model+validated"
    assert harness_ids == expected[:max_harness]
    assert len(plan.ids()) == len(set(plan.ids()))
    assert plan.ids()[0] == "flash_retrieve"
    assert plan.ids()[-1] == "synthesize"


def test_model_refinement_deduplicates_before_applying_limit() -> None:
    response = json.dumps(
        {
            "capabilities": [
                "synthesize",
                "doctor",
                "doctor",
                "flash_retrieve",
                "research",
                "totally-made-up",
                "train",
                "synthesize",
                "flash_retrieve",
            ]
        }
    )
    plan = plan_goal("check a benchmark", model_fn=lambda _: response, max_harness=2)
    assert plan.ids() == ["flash_retrieve", "doctor", "research", "synthesize"]


def test_model_refinement_can_reselect_registered_capabilities() -> None:
    goal = "check the order book identities"
    assert "doctor" not in plan_goal(goal).ids()
    plan = plan_goal(goal, model_fn=lambda _: '{"capabilities": ["doctor"]}')
    assert plan.source == "model+validated"
    assert plan.ids() == ["flash_retrieve", "doctor", "synthesize"]


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"rationale": "no selection supplied"}',
        '{"capabilities": "research"}',
        '{"capabilities": ["research", 1]}',
        '{"capabilities": ["research"], "rationale": {"execute": "train"}}',
    ],
)
def test_malformed_refinement_preserves_bounded_fallback(response: str) -> None:
    goal = "ingest data, benchmark, backtest, then train and optimize"
    expected = plan_goal(goal, max_harness=1)
    actual = plan_goal(goal, model_fn=lambda _: response, max_harness=1)
    assert actual == expected


def test_model_unknown_selection_cannot_become_executable() -> None:
    plan = plan_goal(
        "check a benchmark",
        model_fn=lambda _: '{"capabilities": ["unknown-command", "!rm -rf /"]}',
    )
    assert plan.ids() == ["flash_retrieve", "synthesize"]


def test_model_garbage_falls_back_to_deterministic() -> None:
    plan = plan_goal(
        "run the conformal volatility benchmark",
        model_fn=lambda prompt: "I cannot output JSON, sorry",
    )
    assert plan.source == "deterministic"
    assert "research" in plan.ids()


def test_model_crash_falls_back_to_deterministic() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("endpoint down")

    plan = plan_goal("run the conformal benchmark", model_fn=boom)
    assert plan.source == "deterministic"
    assert "research" in plan.ids()


# ---------------------------------------------------------------------------
# coordination
# ---------------------------------------------------------------------------


def test_synthesis_prompt_excludes_private_flash_findings(store: FlashStore) -> None:
    for i in range(4):
        store.add(text=f"inflation PRIVATE_SENTINEL_{i}", task="inflation", private=True)
    public = store.add(text="inflation public summary", task="inflation")
    prompts: list[str] = []

    def model_fn(prompt: str) -> str:
        prompts.append(prompt)
        return "Public context summarized."

    result = run_plan(
        _plan("inflation", ["flash_retrieve", "synthesize"]),
        SuperpowerContext(model_fn=model_fn, flash_store=store),
    )
    assert prompts
    assert "PRIVATE_SENTINEL" not in str(prompts)
    assert "inflation public summary" in prompts[0]
    assert store.get(public.id).uses == 1
    assert result.runs[0].detail == "1 relevant memory entries retrieved"


@pytest.mark.parametrize(
    ("stage", "store_method"),
    [("flash_retrieve", "all"), ("flash_save", "add")],
)
def test_flash_storage_errors_do_not_reach_synthesis_model(
    store: FlashStore, monkeypatch: pytest.MonkeyPatch, stage: str, store_method: str
) -> None:
    sentinel = "PRIVATE_STORAGE_PATH_AND_FINDING"

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError(sentinel)

    monkeypatch.setattr(store, store_method, fail)
    prompts: list[str] = []

    def model_fn(prompt: str) -> str:
        prompts.append(prompt)
        return "This run is partial because flash storage failed."

    result = run_plan(
        _plan("check data", [stage, "synthesize"]),
        SuperpowerContext(model_fn=model_fn, flash_store=store),
    )

    assert prompts
    assert result.partial
    assert {run.id: run for run in result.runs}[stage].status == "failed"
    assert sentinel not in prompts[0]
    assert sentinel not in result.explain()
    assert sentinel not in str(result.errors)
    assert sentinel not in result.summary


@pytest.mark.parametrize("command", list(_REQUIRED_HARNESS_INPUTS))
def test_handcrafted_plan_skips_harness_command_without_required_input(command: str) -> None:
    harness = FakeHarness()
    result = run_plan(
        _plan("g", [command, "doctor", "synthesize"]),
        SuperpowerContext(harness=harness, approvals=ApprovalGate()),
    )
    runs = {run.id: run for run in result.runs}
    assert runs[command].status == "skipped"
    assert _REQUIRED_HARNESS_INPUTS[command][1] in runs[command].detail
    assert result.partial
    assert harness.calls == [("doctor", None)]


def test_handcrafted_research_plan_skips_unavailable_family_catalog() -> None:
    harness = FakeHarness()
    result = run_plan(
        _plan("benchmark volatility", ["research", "doctor", "synthesize"]),
        SuperpowerContext(
            harness=harness,
            approvals=ApprovalGate(),
            catalog=FamilyCatalog.unavailable("catalog command failed"),
        ),
    )
    runs = {run.id: run for run in result.runs}
    assert runs["research"].status == "skipped"
    assert "family eligibility cannot be checked" in runs["research"].detail
    assert result.partial
    assert harness.calls == [("doctor", None)]


def test_handcrafted_named_family_plan_cannot_bypass_selector_gate() -> None:
    harness = FakeHarness()
    catalog = FamilyCatalog(
        schema_version=2,
        catalog_version=1,
        receipt_schema_version=1,
        required=frozenset({"conformal"}),
        live_optional=frozenset({"quantile_bandit"}),
        live_optional_total=1,
        exploratory_count=0,
        retired_count=1,
        retired_sample=frozenset(),
        order=("conformal", "quantile_bandit"),
        generated_at="2026-10-16T00:00:00Z",
        execution_constraint=True,
    )
    for goal, reason in (
        ("run the conformal benchmark", "named-family execution unavailable"),
        ("run the bandit benchmark", "named-family execution unavailable"),
        ("benchmark family:legacy_template_family", "no suitable family"),
    ):
        result = run_plan(
            _plan(goal, ["research", "synthesize"]),
            SuperpowerContext(
                harness=harness,
                catalog=replace(catalog, lookup_goal_sha256=goal_sha256(goal)),
            ),
        )
        assert result.runs[0].status == "skipped"
        assert reason in result.runs[0].detail
        assert result.partial
    assert harness.calls == []


def test_handcrafted_unscoped_research_requires_exact_goal_lookup() -> None:
    goal = "run the research benchmark"
    catalog = FamilyCatalog(
        schema_version=2,
        catalog_version=2,
        receipt_schema_version=2,
        required=frozenset({"conformal"}),
        live_optional=frozenset(),
        retired_count=0,
        retired_sample=frozenset(),
        order=("conformal",),
        generated_at="2026-10-17T00:00:00Z",
    )
    for candidate in (catalog, replace(catalog, lookup_goal_sha256=goal_sha256("other goal"))):
        harness = FakeHarness()
        result = run_plan(
            _plan(goal, ["research", "synthesize"]),
            SuperpowerContext(harness=harness, catalog=candidate),
        )
        assert result.runs[0].status == "skipped"
        assert "lookup missing or stale" in result.runs[0].detail
        assert result.partial
        assert harness.calls == []
    plan = plan_goal(goal, catalog=catalog)
    assert "research" not in plan.ids()
    assert "lookup missing or stale" in plan.describe()


def test_model_cannot_add_research_when_bare_name_did_not_load_catalog() -> None:
    def model_fn(prompt: str) -> str:
        assert "- research:" not in prompt
        return json.dumps({"capabilities": ["research", "web_research"]})

    plan = plan_goal("a1 homotopy", model_fn=model_fn)
    assert "research" not in plan.ids()
    assert "web_research" in plan.ids()


def test_planned_exclusion_makes_result_partial_and_visible_in_summary() -> None:
    harness = FakeHarness()
    plan = plan_goal("validate a model")
    result = run_plan(plan, SuperpowerContext(harness=harness, approvals=ApprovalGate()))
    assert harness.calls == []
    assert result.partial
    assert "result: partial" in result.explain()
    assert "[excluded] validate: requires model_id" in result.explain()
    assert "excluded: validate: requires model_id" in result.summary
    assert "this run is PARTIAL" in result.summary


def test_fully_served_plan_remains_complete() -> None:
    result = run_plan(
        _plan("check readiness", ["doctor", "synthesize"]),
        SuperpowerContext(harness=FakeHarness(), approvals=ApprovalGate()),
    )
    assert result.partial is False
    assert "result: complete" in result.explain()


def test_run_plan_coordinates_memory_web_harness_synthesis(store: FlashStore) -> None:
    store.add(
        text="prior finding: conformal coverage held at 90% on synthetic panels",
        task="research benchmark",
        tags=["conformal"],
    )
    harness = FakeHarness()
    researcher = FakeResearcher()
    prompts: list[str] = []

    def model_fn(prompt: str) -> str:
        prompts.append(prompt)
        return "Coverage held at the nominal 90% level (SYNTHETIC panel). Sources listed above."

    plan = _plan(
        "run the research benchmark",
        ["flash_retrieve", "research", "web_research", "flash_save", "synthesize"],
    )
    ctx = SuperpowerContext(
        model_fn=model_fn,
        harness=harness,
        approvals=ApprovalGate(),
        researcher=researcher,
        flash_store=store,
        allow_web=True,
    )
    result = run_plan(plan, ctx)

    assert harness.calls == [("research", None)]
    assert researcher.calls == [plan.goal]
    by_id = {r.id: r for r in result.runs}
    assert by_id["flash_retrieve"].status == "ran"
    assert "1 relevant memory entries" in by_id["flash_retrieve"].detail
    assert by_id["web_research"].status == "ran"
    assert by_id["research"].status == "ran"
    assert "DATA_LABEL=SYNTHETIC" in by_id["research"].detail
    assert by_id["flash_save"].status == "ran"
    assert by_id["synthesize"].status == "ran"
    assert result.honesty_violation is None
    assert "Coverage held" in result.summary
    # synthesis prompt carried goal + evidence + capability explanations
    assert "research benchmark" in prompts[0]
    assert "HEURISTIC WEB VERIFICATION" in prompts[0]
    assert "prior finding" in prompts[0]
    # the run was persisted to flash with provenance tags
    saved = [e for e in store.all() if "superpower" in e.tags]
    assert len(saved) == 1
    assert "research benchmark" in saved[0].task
    # explain() lists what ran and why
    explanation = result.explain()
    assert "capabilities used and why" in explanation
    assert "[ran] research" in explanation


def test_web_research_without_opt_in_does_not_call_the_search_client(store: FlashStore) -> None:
    from fx1.webresearch.engine import Researcher

    class _SpySearcher:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search(self, query: str, **kwargs: object) -> None:
            self.calls.append(query)
            raise AssertionError("search client must not be called without --web")

    spy = _SpySearcher()
    researcher = Researcher(searcher=spy)  # type: ignore[arg-type]
    plan = _plan(
        "search the latest pinball literature",
        ["flash_retrieve", "web_research", "synthesize"],
    )
    result = run_plan(
        plan,
        SuperpowerContext(researcher=researcher, flash_store=store, approvals=ApprovalGate()),
    )
    assert spy.calls == []
    web = {run.id: run for run in result.runs}["web_research"]
    assert web.status == "skipped"
    assert "opt-in" in web.detail


def test_run_plan_denies_consequential_without_approval(store: FlashStore) -> None:
    harness = FakeHarness()
    gate = ApprovalGate(ask_fn=lambda question: False)
    plan = _plan("run a paper trading shadow run", ["paper", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=lambda prompt: "Paper run skipped: not approved.",
        harness=harness,
        approvals=gate,
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    by_id = {r.id: r for r in result.runs}
    assert by_id["paper"].status == "approval-denied"
    assert result.partial is True
    assert "result: partial" in result.explain()
    assert harness.calls == []  # never executed
    assert gate.denied() == 1


def test_web_research_errors_make_superpower_partial() -> None:
    report = ResearchReport(
        goal="check public sources",
        sources=[SourceDoc(url="https://a.example/x", title="A", status=200, text="t")],
        errors=["search request timed out"],
    )
    result = run_plan(
        _plan("check public sources", ["web_research", "synthesize"]),
        SuperpowerContext(researcher=FakeResearcher(report)),
    )
    web = next(run for run in result.runs if run.id == "web_research")
    assert web.status == "failed"
    assert result.partial is True
    assert "web_research: search request timed out" in result.errors
    assert "PARTIAL" in result.summary


def test_harness_is_not_started_without_one_second_for_its_minimum_timeout() -> None:
    clock = [0.0]
    ledger = BudgetLedger(RunBudget(max_wall_s=1.5), clock=lambda: clock[0])
    clock[0] = 1.0
    harness = FakeHarness()
    result = run_plan(
        _plan("check health", ["doctor", "synthesize"]),
        SuperpowerContext(harness=harness, budget_ledger=ledger),
    )
    assert harness.calls == []
    assert result.partial is True
    assert next(run for run in result.runs if run.id == "doctor").status == "failed"


def test_refined_consequential_selection_still_requires_approval(store: FlashStore) -> None:
    harness = FakeHarness()
    gate = ApprovalGate(ask_fn=lambda question: False)
    plan = plan_goal(
        "check a benchmark",
        model_fn=lambda _: '{"capabilities": ["train", "train", "doctor", "unknown"]}',
        max_harness=2,
    )
    result = run_plan(
        plan,
        SuperpowerContext(harness=harness, approvals=gate, flash_store=store),
    )
    assert harness.calls == [("doctor", None)]
    assert {r.id: r.status for r in result.runs}["train"] == "approval-denied"
    assert gate.denied() == 1


def test_run_plan_does_not_repeat_duplicate_harness_steps(store: FlashStore) -> None:
    harness = FakeHarness()
    plan = _plan("check data", ["doctor", "doctor", "synthesize"])
    run_plan(plan, SuperpowerContext(harness=harness, flash_store=store))
    assert harness.calls == [("doctor", None)]


def test_run_plan_allows_consequential_with_approval(store: FlashStore) -> None:
    harness = FakeHarness()
    questions: list[str] = []

    def approve(question: str) -> bool:
        questions.append(question)
        return True

    plan = _plan("train the ranking family", ["train", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=lambda prompt: "Training dispatched.",
        harness=harness,
        approvals=ApprovalGate(ask_fn=approve),
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    assert harness.calls == [("train", None)]
    assert {r.id: r for r in result.runs}["train"].status == "ran"
    assert any("consequential" in q for q in questions)


def test_run_plan_honesty_gate_withholds_violating_summary(store: FlashStore) -> None:
    plan = _plan("summarize performance", ["synthesize"])
    ctx = SuperpowerContext(
        model_fn=lambda prompt: "Great news: Sharpe 2.1 achieved live.",
        harness=FakeHarness(),
        approvals=ApprovalGate(),
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    assert result.honesty_violation is not None
    assert "forbidden" in result.honesty_violation
    assert "withheld" in result.summary
    assert "Sharpe 2.1" not in result.summary  # violating text never emitted
    assert {r.id: r for r in result.runs}["synthesize"].status == "failed"


def test_run_plan_without_model_gives_offline_summary(store: FlashStore) -> None:
    researcher = FakeResearcher()
    plan = _plan("check inflation", ["flash_retrieve", "web_research", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=None,
        harness=FakeHarness(),
        approvals=ApprovalGate(),
        researcher=researcher,
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    assert "no model endpoint" in result.summary
    assert "web_research" in result.summary
    assert result.honesty_violation is None


def test_run_plan_admits_direct_model_call_against_budget() -> None:
    ledger = BudgetLedger(RunBudget(max_model_calls=1))
    calls: list[str] = []

    def model_fn(prompt: str) -> str:
        calls.append(prompt)
        return "Answer from model."

    ctx = SuperpowerContext(model_fn=model_fn, budget_ledger=ledger)
    first = run_plan(_plan("question", ["synthesize"]), ctx)
    second = run_plan(_plan("another question", ["synthesize"]), ctx)

    assert len(calls) == 1
    assert ledger.snapshot()["model_calls"] == {"used": 1, "limit": 1}
    assert first.runs[-1].status == "ran"
    assert second.runs[-1].status == "budget-exceeded"


@pytest.mark.parametrize("catalog", [None, FamilyCatalog.unavailable("offline")])
def test_direct_run_plan_blocks_research_without_live_catalog(
    catalog: FamilyCatalog | None,
) -> None:
    harness = FakeHarness()
    result = run_plan(
        _plan("conformal benchmark", ["research", "synthesize"]),
        SuperpowerContext(harness=harness, catalog=catalog),
    )

    assert harness.calls == []
    assert result.runs[0].status == "skipped"
    assert "family catalog unavailable" in result.runs[0].detail
    assert result.partial is True


def test_direct_model_synthesis_returns_on_cancel() -> None:
    started = threading.Event()
    release = threading.Event()
    cancel = threading.Event()
    outcome: list[SuperpowerResult] = []

    def model_fn(_prompt: str) -> str:
        started.set()
        release.wait(5)
        return "late answer"

    def run() -> None:
        outcome.append(
            run_plan(
                _plan("question", ["synthesize"]),
                SuperpowerContext(model_fn=model_fn, cancel_event=cancel),
            )
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert started.wait(1)
        cancel.set()
        worker.join(1)
        assert not worker.is_alive()
        assert outcome[0].cancelled is True
        assert outcome[0].runs[-1].status == "cancelled"
        assert "late answer" not in outcome[0].summary
    finally:
        release.set()


def test_direct_model_synthesis_returns_at_wall_deadline() -> None:
    release = threading.Event()
    ledger = BudgetLedger(RunBudget(max_wall_s=0.5))

    def model_fn(_prompt: str) -> str:
        release.wait(5)
        return "late answer"

    try:
        started = time.monotonic()
        result = run_plan(
            _plan("question", ["synthesize"]),
            SuperpowerContext(model_fn=model_fn, budget_ledger=ledger),
        )
        assert time.monotonic() - started < 2.0
        assert result.runs[-1].status == "budget-exceeded"
        assert "late answer" not in result.summary
    finally:
        release.set()


def test_direct_web_research_returns_on_cancel() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    cancel = threading.Event()
    outcome: list[SuperpowerResult] = []
    transport_events: list[threading.Event] = []

    class StalledResearcher:
        def run(
            self, goal: str, *, cancel_event: threading.Event, **kwargs: object
        ) -> ResearchReport:
            transport_events.append(cancel_event)
            started.set()
            release.wait(5)
            finished.set()
            return ResearchReport(
                goal=goal, sources=[SourceDoc(url="https://late.example", status=200)]
            )

    def run() -> None:
        outcome.append(
            run_plan(
                _plan("question", ["web_research", "synthesize"]),
                SuperpowerContext(researcher=StalledResearcher(), cancel_event=cancel),
            )
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert started.wait(1)
        cancel.set()
        worker.join(1)
        assert not worker.is_alive()
        assert transport_events[0].is_set()
        assert outcome[0].cancelled is True
        assert outcome[0].runs[0].status == "cancelled"
        assert "late.example" not in outcome[0].summary
    finally:
        release.set()
        assert finished.wait(1)


def test_direct_web_research_returns_at_wall_deadline() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    transport_events: list[threading.Event] = []

    class StalledResearcher:
        def run(
            self, goal: str, *, cancel_event: threading.Event, **kwargs: object
        ) -> ResearchReport:
            transport_events.append(cancel_event)
            started.set()
            release.wait(5)
            finished.set()
            return ResearchReport(
                goal=goal, sources=[SourceDoc(url="https://late.example", status=200)]
            )

    try:
        ledger = BudgetLedger(RunBudget(max_wall_s=0.2))
        before = time.monotonic()
        result = run_plan(
            _plan("question", ["web_research", "synthesize"]),
            SuperpowerContext(researcher=StalledResearcher(), budget_ledger=ledger),
        )
        assert started.is_set()
        assert time.monotonic() - before < 2
        assert transport_events[0].is_set()
        assert result.runs[0].status == "budget-exceeded"
        assert "late.example" not in result.summary
    finally:
        release.set()
        assert finished.wait(1)


def test_run_plan_cancel_skips_remaining_work(store: FlashStore) -> None:
    cancel = threading.Event()
    cancel.set()
    harness = FakeHarness()
    researcher = FakeResearcher()
    plan = _plan("g", ["flash_retrieve", "web_research", "research", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=None,
        harness=harness,
        approvals=ApprovalGate(),
        researcher=researcher,
        flash_store=store,
        cancel_event=cancel,
    )
    result = run_plan(plan, ctx)
    assert researcher.calls == []  # web research never started
    assert harness.calls == []  # harness never started
    statuses = {r.id: r.status for r in result.runs}
    assert statuses["research"] == "cancelled"
    assert statuses["web_research"] == "cancelled"
    assert result.cancelled is True
    assert result.partial is True
    assert "cancelled by operator" in result.explain()
    assert "PARTIAL" in result.summary


def test_run_plan_harness_failure_recorded_not_fatal(store: FlashStore) -> None:
    class FailingHarness:
        def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
            return FakeHarnessResult(2, "", "boom")

    plan = _plan("verify receipts", ["verify-research", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=None,
        harness=FailingHarness(),
        approvals=ApprovalGate(),
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    by_id = {r.id: r for r in result.runs}
    assert by_id["verify-research"].status == "failed"
    assert by_id["synthesize"].status == "ran"  # synthesis still reports honestly


def test_run_plan_unknown_capability_id_skipped(store: FlashStore) -> None:
    plan = _plan("g", ["definitely-not-registered", "synthesize"])
    ctx = SuperpowerContext(model_fn=None, harness=FakeHarness(), flash_store=store)
    result = run_plan(plan, ctx)
    assert all(r.id != "definitely-not-registered" for r in result.runs)


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------


def test_approval_gate_defaults_to_deny() -> None:
    gate = ApprovalGate()
    assert gate.ask("do the thing?") is False
    assert gate.denied() == 1
    assert gate.records[0].question == "do the thing?"


def test_approval_gate_records_decisions() -> None:
    gate = ApprovalGate(ask_fn=lambda q: True)
    assert gate.ask("a?", kind="consequential") is True
    assert gate.granted() == 1
    assert gate.records[0].kind == "consequential"
    assert gate.records[0].timestamp


def test_console_ask_fn_yes_no_eof() -> None:
    assert console_ask_fn(lambda prompt: "y")("q?") is True
    assert console_ask_fn(lambda prompt: "yes")("q?") is True
    assert console_ask_fn(lambda prompt: "n")("q?") is False
    assert console_ask_fn(lambda prompt: "")("q?") is False

    def eof(prompt: str) -> str:
        raise EOFError

    assert console_ask_fn(eof)("q?") is False


def test_family_selector_constrains_research_to_eligible_ids(store: FlashStore) -> None:
    harness = FakeHarness()

    def selector(goal: str) -> dict[str, object]:
        assert "volatility" in goal
        return {
            "schema": "dipcatcher.family_surface.v1",
            "selected": [
                {"id": "volatility", "executable": True, "qualification": "required"},
            ],
            "exclusions": [{"id": "a1_degrees", "reason": "retired"}],
        }

    plan = plan_goal("score volatility with qlike", family_selector=selector)
    assert plan.research_constrained is True
    assert plan.families == ["volatility"]
    result = run_plan(plan, SuperpowerContext(harness=harness, flash_store=store))
    assert harness.calls == [("research", ["--only", "volatility"])]
    assert "excluded a1_degrees: retired" in plan.describe()
    assert result.summary
    # The harness evidence block reports a zero count ("0 verified receipts").
    # The prose above that block must not claim a verified receipt.
    prose, block, _rest = result.summary.partition("Evidence (harness-verified")
    assert block
    assert "verified receipt" not in prose


def test_family_surface_failure_does_not_run_the_full_catalog(store: FlashStore) -> None:
    harness = FakeHarness()

    def selector(_goal: str) -> dict[str, object]:
        raise RuntimeError("catalog unavailable")

    plan = plan_goal("run the benchmark family", family_selector=selector)
    result = run_plan(plan, SuperpowerContext(harness=harness, flash_store=store))
    assert plan.families == []
    assert harness.calls == []
    assert any(run.id == "research" and run.status == "skipped" for run in result.runs)


def test_ineligible_family_payload_is_refused() -> None:
    with pytest.raises(ValueError, match="ineligible"):
        parse_family_selection(
            {
                "schema": "dipcatcher.family_surface.v1",
                "selected": [{"id": "volatility", "executable": False, "qualification": "retired"}],
                "exclusions": [],
            }
        )
