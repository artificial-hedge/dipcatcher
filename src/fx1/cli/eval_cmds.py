"""Eval and training-prep top-level commands for ``fx1``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from fx1.cli._app import app
from fx1.cli.support import _BACKEND_HELP, _JUDGE_BACKEND_HELP

if TYPE_CHECKING:
    from fx1.eval.suite import ModelFn


def _resolve_model_backend(
    backend: str, checkpoint_dir: Path | None, model: str | None = None
) -> ModelFn:
    """Resolve the model under test. Fail-closed on unknown names.

    ``local_fx1`` needs ``--checkpoint-dir``; ``byok`` reads the
    FX1_BYOK_* env contract (never argv); ``hosted_k3`` needs
    MOONSHOT_API_KEY. An unknown backend string exits 2 — silently
    running a different backend than the flag names would corrupt the
    receipt's meaning.
    """
    from fx1.serve import get_backend

    if backend == "local_fx1":
        return get_backend("local_fx1", checkpoint_dir=checkpoint_dir).complete
    if backend in ("hosted_k3", "byok"):
        kwargs: dict[str, object] = {}
        if model is not None:
            kwargs["model"] = model
        return get_backend(backend, **kwargs).complete
    typer.echo(f"unknown --backend {backend!r}; choose hosted_k3 | local_fx1 | byok", err=True)
    raise typer.Exit(code=2)


def _resolve_judge(judge_backend: str | None) -> ModelFn | None:
    """Build the optional ext-bench judge ModelFn. Fail-closed on usage.

    ``None`` selects the deterministic rule-based judge by configuration;
    ``hosted_k3`` reuses the same backend construction as the model under
    test (a missing MOONSHOT_API_KEY raises, never fabricates a judge).
    """
    if judge_backend is None:
        return None
    if judge_backend not in ("hosted_k3", "byok"):
        typer.echo(
            f"unknown --judge-backend {judge_backend!r}; 'hosted_k3' and "
            "'byok' are supported (omit the flag for the deterministic "
            "rule-based judge)",
            err=True,
        )
        raise typer.Exit(code=2)
    from fx1.serve import get_backend

    return get_backend(judge_backend).complete


@app.command("eval")
def eval_bank(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    out: Path = typer.Option(Path("data/fx1/eval.json")),
    model: str | None = typer.Option(
        None,
        help="Model id sent to a hosted backend (default: the backend's pin).",
    ),
) -> None:
    """Run the built-in eval task bank against an fx-1 backend."""
    from fx1.eval import DEFAULT_BANK, run_suite

    model_backend = _resolve_model_backend(backend, checkpoint_dir, model)
    summary = run_suite(model_backend, list(DEFAULT_BANK))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"by_kind": summary["by_kind"], "honesty_gate_passed": summary["honesty_gate_passed"]},
            indent=2,
        )
    )


@app.command("capability-eval")
def capability_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    judge_backend: str | None = typer.Option(None, "--judge-backend", help=_JUDGE_BACKEND_HELP),
    out: Path = typer.Option(Path("data/fx1/capability_eval.json")),
) -> None:
    """Run the capability battery: time-series reasoning, probability
    calibration, harness tool-use, retrieval-with-citation, external-
    benchmark-format adapters (MT-Bench / FinanceBench / FinToolBench-style),
    and options reasoning — all on seeded SYNTHETIC banks. Exit 1 when any
    honesty sub-gate, the calibration gate, or an ext-bench score gate
    fails. Real ext-bench JSONL sources plug in via `fx1 ext-bench-eval`;
    the options bank is sealed (no external loader)."""
    from fx1.eval import run_capability_eval

    judge = _resolve_judge(judge_backend)
    model = _resolve_model_backend(backend, checkpoint_dir)
    report = run_capability_eval(model, seed=seed, judge=judge)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "ts_reasoning_overall": report.ts_reasoning.overall,
                "calibration_ece": report.calibration.ece,
                "calibration_passed": report.calibration.passed,
                "tooluse_pass_rate": report.tooluse.pass_rate,
                "retrieval_accuracy": report.retrieval.accuracy,
                "ext_bench_honesty_gate_passed": (
                    report.ext_bench.honesty_gate_passed if report.ext_bench is not None else None
                ),
                "ext_bench_score_gate_passed": (
                    report.ext_bench.score_gate_passed if report.ext_bench is not None else None
                ),
                "options_reasoning_passed": (
                    report.options_reasoning.passed
                    if report.options_reasoning is not None
                    else None
                ),
                "honesty_gate_passed": report.honesty_gate_passed,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("ext-bench-eval")
def ext_bench_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    judge_backend: str | None = typer.Option(None, "--judge-backend", help=_JUDGE_BACKEND_HELP),
    mtbench_jsonl: Path | None = typer.Option(
        None,
        "--mtbench-jsonl",
        help="Genuine MT-Bench-style JSONL replacing the sealed synthetic bank.",
    ),
    financebench_jsonl: Path | None = typer.Option(
        None,
        "--financebench-jsonl",
        help="Genuine FinanceBench-style JSONL replacing the sealed synthetic bank.",
    ),
    fintoolbench_jsonl: Path | None = typer.Option(
        None,
        "--fintoolbench-jsonl",
        help="Genuine FinToolBench-style JSONL replacing the sealed synthetic bank.",
    ),
    out: Path = typer.Option(Path("data/fx1/ext_bench_eval.json")),
) -> None:
    """Run the external-benchmark-format adapters (MT-Bench / FinanceBench /
    FinToolBench-style). Default banks are sealed SYNTHETIC correctness
    gates — NOT market evidence and NOT real benchmark scores; genuine JSONL
    exports plug in per benchmark (schema-validated, fail-closed). Exit 1
    when any refusal/honesty gate or score gate fails."""
    from fx1.eval.ext_bench import run_ext_bench_eval

    judge = _resolve_judge(judge_backend)
    model = _resolve_model_backend(backend, checkpoint_dir)
    sources: dict[str, Path] = {}
    if mtbench_jsonl is not None:
        sources["mtbench"] = mtbench_jsonl
    if financebench_jsonl is not None:
        sources["financebench"] = financebench_jsonl
    if fintoolbench_jsonl is not None:
        sources["fintoolbench"] = fintoolbench_jsonl
    report = run_ext_bench_eval(model, seed=seed, judge=judge, sources=sources or None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "synthetic": report.synthetic,
                "label": report.label,
                "benchmarks": {
                    name: {
                        "n_instances": score.n_instances,
                        "metrics": score.metrics,
                        "refusal_gate_passed": score.refusal_gate_passed,
                        "honesty_violations": score.honesty_violations,
                        "gate_passed": score.gate_passed,
                    }
                    for name, score in report.benchmarks.items()
                },
                "honesty_gate_passed": report.honesty_gate_passed,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("options-reasoning-eval")
def options_reasoning_eval(
    backend: str = typer.Option("hosted_k3", help=_BACKEND_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    seed: int = typer.Option(0, help="Seeded SYNTHETIC bank seed."),
    out: Path = typer.Option(Path("data/fx1/options_reasoning_eval.json")),
) -> None:
    """Run the sealed SYNTHETIC options-reasoning battery (LiveOption-
    inspired levels: action validity, decision quality, risk characteristics,
    outcome, bait refusals). Gold answers come from the repo's own pricing
    modules — correctness gates, NOT market evidence. The bank is sealed
    (no external JSONL loader exists), so there is no --options-jsonl
    pass-through. Exit 1 when the bait/honesty gate fails."""
    from fx1.eval.options_reasoning_eval import run_options_reasoning_eval

    model = _resolve_model_backend(backend, checkpoint_dir)
    report = run_options_reasoning_eval(model, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "label": report.label,
                "n_items": report.n_items,
                "overall": report.overall,
                "by_level": report.by_level,
                "bait_accuracy": report.bait_accuracy,
                "bait_gate_passed": report.bait_gate_passed,
                "honesty_violations": report.honesty_violations,
                "passed": report.passed,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("modelcard")
def modelcard_validate(
    path: Path = typer.Argument(...),
    fixture: str | None = typer.Option(
        None,
        "--fixture",
        help="Explicit fixture marker (only SYNTHETIC is accepted); refuses ship eligibility.",
    ),
) -> None:
    """Validate an fx-1 model card and report ship-gate status."""
    from fx1.modelcard import ModelCard

    if fixture is not None and fixture.upper() != "SYNTHETIC":
        raise typer.BadParameter("only --fixture SYNTHETIC is supported")
    card = ModelCard.load(path)
    ship_eligible = card.ship_eligible and fixture is None
    typer.echo(json.dumps({"version": card.version, "ship_eligible": ship_eligible}, indent=2))


@app.command("redteam")
def redteam(
    backend: str = typer.Option("hosted_k3"),
    checkpoint_dir: Path | None = typer.Option(None),
) -> None:
    """Run the adversarial red-team suite against an fx-1 backend."""
    from fx1.eval.redteam import REDTEAM_TASKS
    from fx1.eval.suite import run_suite

    model = _resolve_model_backend(backend, checkpoint_dir)
    summary = run_suite(model, list(REDTEAM_TASKS))
    failed = [r["task"] for r in summary.results if not r["passed"]]
    typer.echo(
        json.dumps(
            {"honesty_gate_passed": summary["honesty_gate_passed"], "results": failed}, indent=2
        )
    )
    raise typer.Exit(code=0 if summary["honesty_gate_passed"] else 1)


@app.command("dpo")
def dpo_build(
    out: Path = typer.Option(Path("data/fx1/dpo.jsonl")),
) -> None:
    """Build contract-derived DPO preference pairs."""
    from fx1.train.dpo import build_preference_pairs

    pairs = build_preference_pairs(out)
    typer.echo(json.dumps({"pairs": len(pairs), "out": str(out)}))


@app.command("curriculum")
def curriculum_build(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/corpus_curriculum.jsonl")),
) -> None:
    """Order the corpus contracts -> interpretation -> loops -> refusal."""
    from fx1.train.curriculum import build_curriculum

    counts = build_curriculum(corpus, out)
    typer.echo(json.dumps(counts, indent=2))


@app.command("maskedaEval")
def masked_eval(
    budget: float = typer.Option(0.25, help="Memory-gap ship budget."),
) -> None:
    """Masked/unmasked twin evaluation + memory-gap ship metric (offline
    structural check; live model runs inject a backend via fx1.eval)."""
    from fx1.eval import DEFAULT_BANK, masked_twins

    twins = masked_twins(list(DEFAULT_BANK))
    typer.echo(
        json.dumps(
            {
                "twin_tasks": len(twins),
                "memory_gap_budget": budget,
                "note": "inject a backend via fx1.eval.run_suite for live scoring",
            },
            indent=2,
        )
    )


@app.command("contamination-audit")
def contamination_audit(
    corpus: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
    out: Path = typer.Option(Path("data/fx1/contamination_report.json")),
    with_rephrased_gap: bool = typer.Option(
        False,
        "--with-rephrased-gap",
        help="Also run the canonical-vs-rephrased gap probe against a live backend.",
    ),
    backend: str = typer.Option("hosted_k3", help="Backend for the gap probe."),
) -> None:
    """Run the publishable contamination audit over the corpus vs eval bank."""
    from fx1.eval import eval_prompt_surface, run_contamination_audit

    # Fail-closed: an absent or empty corpus certifies nothing — auditing
    # zero texts would vacuously report "not contaminated" and exit 0.
    if not corpus.exists():
        typer.echo(f"corpus not found: {corpus}; refusing to certify an empty audit", err=True)
        raise typer.Exit(code=2)
    texts = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            texts.append(" ".join(m.get("content", "") for m in record.get("messages", [])))
    if not texts:
        typer.echo(f"corpus {corpus} contains no examples; audit cannot certify", err=True)
        raise typer.Exit(code=2)
    prompts = eval_prompt_surface()
    report = run_contamination_audit(texts, prompts)
    if with_rephrased_gap:
        from fx1.eval import run_rephrased_gap
        from fx1.serve import get_backend

        model = get_backend(backend)
        report.probes.append(run_rephrased_gap(model.complete))
        report.overall_flagged = report.overall_flagged or any(p.flagged for p in report.probes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "overall_flagged": report.overall_flagged,
                "ngram_hits": len(report.ngram_hits),
                "probes": [p.method for p in report.probes],
            },
            indent=2,
        )
    )
    raise typer.Exit(code=1 if report.overall_flagged else 0)


@app.command("sign")
def sign_checkpoint(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Sign an fx-1 release (attestation ladder tier 1)."""
    from fx1.serve import sign_release

    sig = sign_release(checkpoint_dir)
    typer.echo(f"signed: {sig}")


@app.command("attestation")
def attestation_status(
    checkpoint_dir: Path = typer.Argument(...),
) -> None:
    """Report which attestation tiers a checkpoint satisfies."""
    from fx1.serve import attestation_ladder_status

    typer.echo(json.dumps(attestation_ladder_status(checkpoint_dir), indent=2))


@app.command("sbom")
def sbom_generate(
    lockfile: Path = typer.Option(Path("uv.lock")),
    out: Path = typer.Option(Path("data/fx1/sbom.json")),
) -> None:
    """Generate a hash-pinned SBOM from the locked dependency set."""
    from fx1.sbom import generate_sbom

    sbom = generate_sbom(lockfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sbom.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {"entries": len(sbom.entries), "lockfile_sha256": sbom.lockfile_sha256[:16] + "…"},
            indent=2,
        )
    )


@app.command("mrm")
def mrm_dossier(
    modelcard: Path = typer.Option(..., help="Model card JSON."),
    validation_artifact: Path = typer.Option(..., help="Contamination report JSON."),
    artifact: list[str] | None = typer.Option(
        None,
        "--artifact",
        help="Extra activity evidence as activity=path (repeatable; "
        "development/implementation/monitoring/governance).",
    ),
    out: Path = typer.Option(Path("data/fx1/mrm_dossier.json")),
) -> None:
    """Compile the five-activity model-risk dossier (SR 26-2 era)."""
    from fx1.mrm import compile_dossier

    artifacts: dict[str, str | Path] = {"validation": validation_artifact}
    for spec in artifact or []:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise typer.BadParameter("--artifact expects activity=path")
        artifacts[name] = path

    dossier = compile_dossier(
        modelcard_path=modelcard,
        artifacts=artifacts,
        out_path=out,
    )
    typer.echo(
        json.dumps(
            {
                "version": dossier.model_version,
                "complete": dossier.complete,
                "ship_eligible": dossier.ship_eligible,
                "contamination_flagged": dossier.contamination_flagged,
            },
            indent=2,
        )
    )
