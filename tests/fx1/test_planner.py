"""Qualified planning, validation, model refinement, and honesty of descriptions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS, validate_fx1_output
from fx1.interactive.contracts import CapabilityCatalog
from fx1.interactive.planner import (
    SYNTHETIC_NOTE,
    DatasetRef,
    ExecutionPlan,
    PlanStep,
    order_steps,
    plan_goal,
    validate_plan,
)
from tests.fx1.capabilities_fixture import capability_entry, sample_catalog, sample_catalog_document

_PARAPHRASES: list[tuple[str, list[str]]] = [
    ("run doctor", ["doctor"]),
    ("please run the health check", ["doctor"]),
    ("check readiness of the environment", ["doctor"]),
    ("environment health status", ["doctor"]),
    ("run the health checks", ["doctor"]),
    ("verify research provenance please", ["verify-research"]),
    ("I need a research provenance audit", ["verify-research"]),
    ("verify receipt for the last job", ["verify-receipt"]),
    ("check the receipt digest", ["verify-receipt"]),
    ("run vol bench on realized variance", ["vol-bench"]),
    ("compare garch forecasts", ["vol-bench"]),
    ("compare the garchs", ["vol-bench"]),
    ("evaluate the har model", ["vol-bench"]),
    ("qlike scoring for variance", ["vol-bench"]),
    ("volatility forecast evaluation", ["vol-bench"]),
    ("run calibration eval", ["calibration-eval"]),
    ("compute the probability integral transform", ["calibration-eval"]),
    ("report the pit score", ["calibration-eval"]),
    ("run the full scorecard", ["doctor", "research"]),
    ("execute the benchmark suite", ["doctor", "research"]),
    ("start the research scorecard", ["doctor", "research"]),
    ("hyperparameter search for the model", ["optimize"]),
    ("use optuna to tune", ["optimize"]),
    ("run doctor and verify receipt", ["verify-receipt", "doctor"]),
    ("garch plus calibration eval", ["calibration-eval", "vol-bench"]),
    ("health check then the full scorecard", ["doctor", "research"]),
    ("realized variance and pit score", ["calibration-eval", "vol-bench"]),
    ("optuna and a health check", ["doctor", "optimize"]),
    ("receipt digest and research provenance", ["verify-research", "verify-receipt"]),
    ("har model and qlike scoring", ["vol-bench"]),
    ("environment health and benchmark suite", ["doctor", "research"]),
    ("probability integral and realized variance", ["calibration-eval", "vol-bench"]),
    ("readiness and optuna", ["doctor", "optimize"]),
    ("vol bench with the har model", ["vol-bench"]),
]

_NO_MATCH = [
    "what is the weather in lisbon tomorrow",
    "summarize the meeting notes from monday",
    "translate this paragraph into french",
    "book a flight to oslo",
    "explain photosynthesis to a child",
    "write a poem about the sea",
    "how do I cook pasta carbonara",
    "set an alarm for seven",
    "recommend a science fiction movie",
    "draft an email to the team",
    "resize the image to thumbnail",
    "who won the chess tournament",
    "define the word serendipity",
    "help me plan a vacation itinerary",
    "sort my todo list by priority",
    "convert kilometers to miles",
    "what is the capital of canada",
    "the chart looks fine",
    "doctorate program overview",
]


def _ids(plan: ExecutionPlan) -> list[str]:
    return [step.capability_id for step in plan.steps]


def _codes(plan: ExecutionPlan) -> list[str]:
    return [item.code for item in plan.exclusions]


@pytest.fixture
def catalog() -> CapabilityCatalog:
    return sample_catalog()


@pytest.mark.parametrize(("goal", "expected"), _PARAPHRASES)
def test_paraphrase_selects_expected_capabilities(
    catalog: CapabilityCatalog, goal: str, expected: list[str]
) -> None:
    plan = plan_goal(goal, catalog)
    assert _ids(plan) == expected
    assert plan.source == "deterministic"
    assert validate_plan(plan, catalog, max_steps=4, budget_s=None) == []
    text = plan.describe()
    assert validate_fx1_output(text) == text
    _assert_honest(text)


@pytest.mark.parametrize("goal", _NO_MATCH)
def test_unrelated_goals_have_no_suitable_capability(catalog: CapabilityCatalog, goal: str) -> None:
    plan = plan_goal(goal, catalog)
    assert plan.steps == []
    assert "no_match" in _codes(plan)
    assert "No suitable capability" in plan.describe()
    assert validate_fx1_output(plan.describe()) == plan.describe()


def test_retired_and_unqualified_are_explained(catalog: CapabilityCatalog) -> None:
    retired = plan_goal("measure alpha decay", catalog)
    assert retired.steps == []
    assert any(
        item.code == "retired" and item.capability_id == "alpha-decay"
        for item in retired.exclusions
    )
    assert "historical receipts" in retired.describe()
    train = plan_goal("fine tune the model", catalog)
    assert train.steps == []
    assert any(
        item.code == "unqualified" and item.capability_id == "train" for item in train.exclusions
    )
    assert "No suitable capability" in train.describe()


def test_experimental_status_uses_unqualified_code() -> None:
    doc = sample_catalog_document()
    entry = capability_entry(doc, "train")
    entry["qualification"]["status"] = "experimental"
    catalog = CapabilityCatalog.from_json(doc)
    plan = plan_goal("lora train the weights", catalog)
    assert any(
        item.code == "unqualified" and item.capability_id == "train" for item in plan.exclusions
    )


def test_dependency_order_post_checks_and_scores(catalog: CapabilityCatalog) -> None:
    plan = plan_goal("run the full scorecard", catalog)
    assert _ids(plan) == ["doctor", "research"]
    assert plan.steps[1].depends_on == ("doctor",)
    assert plan.steps[1].evidence_class == "SYNTHETIC"
    assert "pinball" in plan.steps[1].scores
    vol = plan_goal("run vol bench", catalog)
    assert _ids(vol) == ["vol-bench"]
    assert vol.steps[0].post_checks[0]["kind"] == "verify_receipt"
    assert "verify-receipt" not in _ids(vol)
    text = vol.describe()
    assert "QLIKE" in text
    assert "SYNTHETIC" in text
    assert "Time limit: 60s" in text


def test_missing_or_cyclic_dependency_is_excluded() -> None:
    doc = sample_catalog_document()
    capability_entry(doc, "vol-bench")["dependencies"] = ["train"]
    catalog = CapabilityCatalog.from_json(doc)
    plan = plan_goal("run vol bench", catalog)
    assert plan.steps == []
    assert any(
        item.code == "dependency_missing" and item.capability_id == "vol-bench"
        for item in plan.exclusions
    )

    cycled = sample_catalog_document()
    capability_entry(cycled, "doctor")["dependencies"] = ["verify-receipt"]
    capability_entry(cycled, "verify-receipt")["dependencies"] = ["doctor"]
    cycle_catalog = CapabilityCatalog.from_json(cycled)
    cycle = plan_goal("run doctor", cycle_catalog)
    assert cycle.steps == []
    assert "dependency_missing" in _codes(cycle)


def test_max_steps_counts_dependencies_and_does_not_pad(catalog: CapabilityCatalog) -> None:
    blocked = plan_goal("run the full scorecard", catalog, max_steps=1)
    assert blocked.steps == []
    assert "max_steps" in _codes(blocked)
    mixed = plan_goal("health check then the full scorecard", catalog, max_steps=1)
    assert _ids(mixed) == ["doctor"]
    assert any(
        item.code == "max_steps" and item.capability_id == "research" for item in mixed.exclusions
    )
    assert validate_plan(mixed, catalog, max_steps=1, budget_s=None) == []


def test_budget_trims_a_suffix_without_reordering(catalog: CapabilityCatalog) -> None:
    fitted = plan_goal("run vol bench", catalog, budget_s=60)
    assert _ids(fitted) == ["vol-bench"]
    short = plan_goal("run vol bench", catalog, budget_s=59)
    assert short.steps == []
    assert "budget" in _codes(short)
    scorecard = plan_goal("run the full scorecard", catalog, budget_s=100)
    assert _ids(scorecard) == ["doctor"]
    assert any(
        item.code == "budget" and item.capability_id == "research" for item in scorecard.exclusions
    )
    heavy = plan_goal("optuna hyperparameter search and a health check", catalog, budget_s=10)
    assert heavy.steps == []
    assert [item.capability_id for item in heavy.exclusions if item.code == "budget"] == [
        "optimize",
        "doctor",
    ]
    kept = plan_goal("optuna hyperparameter search and a health check", catalog, budget_s=304)
    assert _ids(kept) == ["optimize"]
    assert any(item.code == "budget" and item.capability_id == "doctor" for item in kept.exclusions)
    text = kept.describe()
    assert "consequential: yes" in text
    assert "Budget: 304s" in text
    assert validate_plan(kept, catalog, max_steps=4, budget_s=304) == []


def test_user_data_synthetic_note_and_needs_dataset(catalog: CapabilityCatalog) -> None:
    supplied = DatasetRef(path="bars.csv", kind="bars", label="USER_SUPPLIED")
    plan = plan_goal("compare garch forecasts", catalog, datasets=(supplied,))
    assert _ids(plan) == ["vol-bench"]
    assert SYNTHETIC_NOTE in plan.notes
    assert any(item.code == "needs_dataset" for item in plan.exclusions)
    assert "no qualified dataset-bound runner matched" in plan.describe()
    assert validate_fx1_output(plan.describe()) == plan.describe()

    attached = plan_goal("the attached series needs a garch check", catalog)
    assert _ids(attached) == ["vol-bench"]
    assert SYNTHETIC_NOTE in attached.notes

    path_goal = plan_goal("evaluate variance.csv with garch", catalog)
    assert _ids(path_goal) == ["vol-bench"]
    assert "needs_dataset" in _codes(path_goal)

    doctor = plan_goal("run doctor", catalog, datasets=(supplied,))
    assert _ids(doctor) == ["doctor"]
    assert SYNTHETIC_NOTE not in doctor.notes
    assert "needs_dataset" in _codes(doctor)

    missing = plan_goal("please inspect my data", catalog)
    assert missing.steps == []
    assert "no_match" in _codes(missing)
    assert "needs_dataset" in _codes(missing)
    assert SYNTHETIC_NOTE not in missing.notes


def test_dataset_bound_runner_is_preferred_when_arguments_exist(
    catalog: CapabilityCatalog, tmp_path: Path
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    target = root / "mine.csv"
    blocked = plan_goal("score my forecasts", catalog, roots={"input": [root]})
    assert blocked.steps == []
    assert any(
        item.code == "invalid_args" and item.capability_id == "forecast-eval"
        for item in blocked.exclusions
    )
    assert "needs_dataset" in _codes(blocked)

    def model(_prompt: str) -> str:
        return json.dumps(
            {
                "rationale": "use the attached column",
                "steps": [
                    {
                        "args": {"--dataset": str(target)},
                        "capability": "forecast-eval",
                        "why": "score the user column",
                    }
                ],
            }
        )

    planned = plan_goal("score my forecasts", catalog, model_fn=model, roots={"input": [root]})
    assert _ids(planned) == ["forecast-eval"]
    assert planned.source == "model+validated"
    assert Path(str(planned.steps[0].args["--dataset"])) == target.resolve()
    assert "needs_dataset" not in _codes(planned)
    assert SYNTHETIC_NOTE not in planned.notes
    assert planned.steps[0].evidence_class == "RESEARCH"
    assert (
        validate_plan(planned, catalog, max_steps=4, budget_s=None, roots={"input": [root]}) == []
    )


def test_model_refinement_accepts_and_ignores_privilege_fields(catalog: CapabilityCatalog) -> None:
    seen: list[str] = []

    def model(prompt: str) -> str:
        seen.append(prompt)
        return "Sure: " + json.dumps(
            {
                "rationale": "volatility only",
                "steps": [
                    {
                        "args": {"--n-origins": 2},
                        "capability": "vol-bench",
                        "evidence_class": "RESEARCH",
                        "receipt_flag": ["--nope"],
                        "why": "variance benchmark",
                    }
                ],
            }
        )

    plan = plan_goal("something unrelated xyz", catalog, model_fn=model)
    assert plan.source == "model+validated"
    assert _ids(plan) == ["vol-bench"]
    assert plan.steps[0].args["--n-origins"] == 2
    assert plan.steps[0].evidence_class == "SYNTHETIC"
    assert "--receipt-version" not in plan.steps[0].args
    prompt = seen[0]
    for cap_id in ("doctor", "vol-bench", "optimize", "forecast-eval", "research"):
        assert f"- {cap_id}:" in prompt
    assert "- train:" not in prompt
    assert "- alpha-decay:" not in prompt
    assert "--horizons" in prompt
    assert "--epochs" not in prompt
    assert "model rationale: volatility only" in plan.notes


def test_model_rejections_drop_bad_steps(catalog: CapabilityCatalog, tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)

    def unknown(_prompt: str) -> str:
        return json.dumps(
            {"rationale": "", "steps": [{"args": {}, "capability": "nope", "why": "x"}]}
        )

    unknown_plan = plan_goal("zz", catalog, model_fn=unknown)
    assert unknown_plan.steps == []
    assert "unknown_id" in _codes(unknown_plan)
    assert unknown_plan.source == "model+validated"

    def bad_arg(_prompt: str) -> str:
        return json.dumps(
            {
                "rationale": "",
                "steps": [
                    {"args": {"--n-origins": 100}, "capability": "vol-bench", "why": "too many"}
                ],
            }
        )

    assert "invalid_args" in _codes(plan_goal("zz", catalog, model_fn=bad_arg))

    def duplicate(_prompt: str) -> str:
        step = {"args": {}, "capability": "doctor", "why": "health"}
        return json.dumps({"rationale": "", "steps": [step, step]})

    duplicated = plan_goal("zz", catalog, model_fn=duplicate)
    assert _ids(duplicated) == ["doctor"]
    assert "duplicate" in _codes(duplicated)

    def over_cap(_prompt: str) -> str:
        steps = [
            {"args": {}, "capability": cap_id, "why": "keep"}
            for cap_id in ("doctor", "verify-receipt", "vol-bench")
        ]
        return json.dumps({"rationale": "", "steps": steps})

    capped = plan_goal("zz", catalog, model_fn=over_cap, max_steps=1)
    assert _ids(capped) == ["doctor"]
    assert "max_steps" in _codes(capped)

    def shell(_prompt: str) -> str:
        return json.dumps(
            {
                "rationale": "",
                "steps": [
                    {"args": {"--models": "har;rm"}, "capability": "vol-bench", "why": "bad"}
                ],
            }
        )

    shelled = plan_goal("zz", catalog, model_fn=shell)
    assert shelled.steps == []
    assert "invalid_args" in _codes(shelled)

    def blocked(_prompt: str) -> str:
        return json.dumps(
            {
                "rationale": "",
                "steps": [
                    {"args": {}, "capability": "train", "why": "train it"},
                    {"args": {}, "capability": "alpha-decay", "why": "old"},
                    {"args": {}, "capability": "doctor", "why": "health"},
                ],
            }
        )

    mixed = plan_goal("zz", catalog, model_fn=blocked)
    assert _ids(mixed) == ["doctor"]
    assert "unqualified" in _codes(mixed)
    assert "retired" in _codes(mixed)

    def outside_path(_prompt: str) -> str:
        return json.dumps(
            {
                "rationale": "",
                "steps": [
                    {
                        "args": {"--dataset": str(link / "secret.csv")},
                        "capability": "forecast-eval",
                        "why": "escape",
                    }
                ],
            }
        )

    escaped = plan_goal("zz", catalog, model_fn=outside_path, roots={"input": [root]})
    assert escaped.steps == []
    assert "invalid_args" in _codes(escaped)

    def receipt(_prompt: str) -> str:
        return json.dumps(
            {
                "rationale": "",
                "steps": [
                    {"args": {"--receipt-version": "2"}, "capability": "vol-bench", "why": "flag"}
                ],
            }
        )

    assert "invalid_args" in _codes(plan_goal("zz", catalog, model_fn=receipt))


def test_structural_model_failures_keep_the_deterministic_plan(catalog: CapabilityCatalog) -> None:
    def malformed(_prompt: str) -> str:
        return '{"steps": [}'

    def prose(_prompt: str) -> str:
        return "I think you should run vol-bench"

    def boom(_prompt: str) -> str:
        raise RuntimeError("backend down")

    def not_text(_prompt: str) -> str:
        return None  # type: ignore[return-value]

    for model in (malformed, prose, boom, not_text):
        plan = plan_goal("run doctor", catalog, model_fn=model)
        assert plan.source == "deterministic"
        assert _ids(plan) == ["doctor"]
        assert any(note.startswith("model refinement rejected:") for note in plan.notes)


def test_empty_model_plan_is_not_padded(catalog: CapabilityCatalog) -> None:
    def model(_prompt: str) -> str:
        return json.dumps({"rationale": "nothing applies", "steps": []})

    plan = plan_goal("run doctor", catalog, model_fn=model)
    assert plan.source == "model+validated"
    assert plan.steps == []
    assert "no_match" in _codes(plan)


def test_hostile_model_text_is_not_copied_into_the_plan(catalog: CapabilityCatalog) -> None:
    def model(_prompt: str) -> str:
        return json.dumps(
            {
                "rationale": "sharpe is 2.4",
                "steps": [{"args": {}, "capability": "doctor", "why": "sharpe is 2.4"}],
            }
        )

    plan = plan_goal("zz", catalog, model_fn=model)
    assert "sharpe" not in plan.steps[0].rationale.lower()
    assert all("sharpe" not in note.lower() for note in plan.notes)
    assert validate_fx1_output(plan.describe()) == plan.describe()


def test_plan_is_deterministic(catalog: CapabilityCatalog) -> None:
    kwargs = {"budget_s": 1000.0, "datasets": (DatasetRef("bars.csv", "bars", "USER_SUPPLIED"),)}
    first = plan_goal("compare garch forecasts", catalog, **kwargs)
    second = plan_goal("compare garch forecasts", catalog, **kwargs)
    assert first.to_dict() == second.to_dict()
    assert first.catalog_digest == catalog.digest


def test_describe_wraps_and_lists_exclusions(catalog: CapabilityCatalog) -> None:
    plan = plan_goal("run vol bench", catalog, budget_s=30)
    text = plan.describe(width=40)
    assert "No suitable capability" in text
    assert "budget" in text
    for line in text.splitlines():
        assert len(line) <= 40 or " " not in line.strip()
    assert validate_fx1_output(text) == text


def test_order_steps_is_stable_and_cycle_safe(catalog: CapabilityCatalog) -> None:
    plan = plan_goal("run the full scorecard", catalog)
    swapped = order_steps([plan.steps[1], plan.steps[0]], catalog)
    assert [step.capability_id for step in swapped] == ["doctor", "research"]
    doctor = plan.steps[0]
    receipt = plan_goal("check the receipt digest", catalog).steps[0]
    cycled = [
        _replace_depends(doctor, ("verify-receipt",)),
        _replace_depends(receipt, ("doctor",)),
    ]
    ordered = order_steps(cycled, catalog)
    assert [step.capability_id for step in ordered] == ["doctor", "verify-receipt"]


def test_validate_plan_reports_structural_problems(catalog: CapabilityCatalog) -> None:
    good = plan_goal("run the full scorecard", catalog)
    assert validate_plan(good, catalog, max_steps=4, budget_s=None) == []
    reversed_plan = ExecutionPlan(
        goal=good.goal,
        steps=[good.steps[1], good.steps[0]],
        exclusions=[],
        notes=[],
        source=good.source,
        catalog_digest=good.catalog_digest,
        max_steps=4,
    )
    assert any(
        item.code == "dependency_missing"
        for item in validate_plan(reversed_plan, catalog, max_steps=4, budget_s=None)
    )
    duplicated = ExecutionPlan(
        goal=good.goal,
        steps=[good.steps[0], good.steps[0]],
        exclusions=[],
        notes=[],
        source=good.source,
        catalog_digest=good.catalog_digest,
        max_steps=4,
    )
    assert any(
        item.code == "duplicate"
        for item in validate_plan(duplicated, catalog, max_steps=4, budget_s=None)
    )
    vol = plan_goal("run vol bench", catalog)
    assert any(
        item.code == "budget" for item in validate_plan(vol, catalog, max_steps=4, budget_s=10)
    )
    assert any(
        item.code == "max_steps"
        for item in validate_plan(good, catalog, max_steps=1, budget_s=None)
    )
    unknown = ExecutionPlan(
        goal="x",
        steps=[_replace_id(vol.steps[0], "nope")],
        exclusions=[],
        notes=[],
        source="deterministic",
        catalog_digest=catalog.digest,
        max_steps=4,
    )
    assert any(
        item.code == "unknown_id"
        for item in validate_plan(unknown, catalog, max_steps=4, budget_s=None)
    )
    lied = ExecutionPlan(
        goal=good.goal,
        steps=[_replace_flag(good.steps[0])],
        exclusions=[],
        notes=[],
        source=good.source,
        catalog_digest=good.catalog_digest,
        max_steps=4,
    )
    problems = validate_plan(lied, catalog, max_steps=4, budget_s=None)
    assert any(item.code == "invalid_args" and "consequential" in item.detail for item in problems)


def test_negative_limits_are_rejected(catalog: CapabilityCatalog) -> None:
    with pytest.raises(ValueError):
        plan_goal("run doctor", catalog, max_steps=-1)
    with pytest.raises(ValueError):
        plan_goal("run doctor", catalog, budget_s=-5)


def _replace_depends(step: PlanStep, depends_on: tuple[str, ...]) -> PlanStep:
    return PlanStep(
        capability_id=step.capability_id,
        args=step.args,
        rationale=step.rationale,
        depends_on=depends_on,
        evidence_class=step.evidence_class,
        consequential=step.consequential,
        est_runtime_s=step.est_runtime_s,
        step_key=step.step_key,
        scores=step.scores,
        post_checks=step.post_checks,
        max_output_chars=step.max_output_chars,
    )


def _replace_id(step: PlanStep, capability_id: str) -> PlanStep:
    return PlanStep(
        capability_id=capability_id,
        args=step.args,
        rationale=step.rationale,
        depends_on=step.depends_on,
        evidence_class=step.evidence_class,
        consequential=step.consequential,
        est_runtime_s=step.est_runtime_s,
        step_key="not-the-real-key",
        scores=step.scores,
        post_checks=step.post_checks,
        max_output_chars=step.max_output_chars,
    )


def _replace_flag(step: PlanStep) -> PlanStep:
    return PlanStep(
        capability_id=step.capability_id,
        args=step.args,
        rationale=step.rationale,
        depends_on=step.depends_on,
        evidence_class=step.evidence_class,
        consequential=not step.consequential,
        est_runtime_s=step.est_runtime_s,
        step_key=step.step_key,
        scores=step.scores,
        post_checks=step.post_checks,
        max_output_chars=step.max_output_chars,
    )


def _assert_honest(text: str) -> None:
    lowered = text.lower()
    for token in FORBIDDEN_HEADLINE_TOKENS:
        assert f" {token} " not in f" {lowered} "
