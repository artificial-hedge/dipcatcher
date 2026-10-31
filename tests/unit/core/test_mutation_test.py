"""Contract tests for the AST mutation-testing harness (scripts/mutation_test.py).

Operator tests run on tiny fixture sources; runner bookkeeping uses injected
fake runners (no pytest subprocesses) except for one real-subprocess test that
proves the sitecustomize overlay actually redirects an import. All fast.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import mutation_test as mt

# ---------------------------------------------------------------------------
# AST operator tests on tiny fixture sources
# ---------------------------------------------------------------------------


def _mutants(source: str) -> list[mt.Mutant]:
    return mt.generate_mutants(source, filename="fixture.py")


def test_comparison_swap_lte_generates_both_replacements() -> None:
    mutants = _mutants("if a <= b:\n    pass\n")
    comparisons = [m for m in mutants if m.operator == "comparison-swap"]
    assert len(comparisons) == 2
    descs = {m.description.split(" ")[0] for m in comparisons}
    assert descs == {"LtE->Lt", "LtE->GtE"}
    sources = {m.source for m in comparisons}
    assert any("if a < b:" in s for s in sources)
    assert any("if a >= b:" in s for s in sources)


def test_comparison_swap_eq_and_chained() -> None:
    eq = [m for m in _mutants("x = a == b\n") if m.operator == "comparison-swap"]
    assert len(eq) == 1 and eq[0].source == "x = a != b"
    chained = [
        m for m in _mutants("if 0.0 <= v <= 1.0:\n    pass\n") if m.operator == "comparison-swap"
    ]
    assert len(chained) == 4  # two LtE ops x two replacements each


def test_boolop_swap_and_or() -> None:
    mutants = [m for m in _mutants("z = a and b or c\n") if m.operator == "boolop-swap"]
    assert len(mutants) == 2
    sources = {m.source for m in mutants}
    assert "z = (a or b) or c" in sources
    assert "z = (a and b) and c" in sources


def test_numeric_perturbation_int_and_float() -> None:
    mutants = [m for m in _mutants("n = 3\nx = 2.5\n") if m.operator == "numeric-perturb"]
    assert len(mutants) == 2
    by_desc = {m.description.split(" ")[1] for m in mutants}
    assert by_desc == {"3->4", "2.5->3.5"}
    assert any(m.source == "n = 4\nx = 2.5" for m in mutants)
    assert any(m.source == "n = 3\nx = 3.5" for m in mutants)


def test_const_bool_flip() -> None:
    mutants = [
        m for m in _mutants("flag = True\nother = False\n") if m.operator == "const-bool-flip"
    ]
    assert len(mutants) == 2
    sources = {m.source for m in mutants}
    assert "flag = False\nother = False" in sources
    assert "flag = True\nother = True" in sources


def test_bool_constant_is_not_numerically_perturbed() -> None:
    mutants = _mutants("flag = True\n")
    assert [m.operator for m in mutants] == ["const-bool-flip"]


def test_negation_insertion_and_removal() -> None:
    inserted = _mutants("if a:\n    pass\n")
    assert len(inserted) == 1
    assert inserted[0].operator == "negation-insertion"
    assert inserted[0].source == "if not a:\n    pass"
    removed = _mutants("if not a:\n    pass\n")
    negations = [m for m in removed if m.operator.startswith("negation")]
    # Removal only: inserting `not (not a)` would duplicate the removal mutant.
    assert len(negations) == 1
    assert negations[0].operator == "negation-removal"
    assert negations[0].source == "if a:\n    pass"


def test_negation_covers_while_ternary_assert() -> None:
    mutants = [m for m in _mutants("while a:\n    pass\nb = 1 if a else 2\nassert a\n")]
    ops = [m.operator for m in mutants]
    assert ops.count("negation-insertion") == 3


def test_mutants_deterministic_unique_and_compile() -> None:
    source = (
        "def f(a, b, keep):\n"
        "    if a < b and keep:\n"
        "        return 0.05\n"
        "    while not keep:\n"
        "        a += 1\n"
        "    return True\n"
    )
    first = _mutants(source)
    second = _mutants(source)
    assert [m.mutant_id for m in first] == [m.mutant_id for m in second]
    assert [m.source for m in first] == [m.source for m in second]
    assert len({m.mutant_id for m in first}) == len(first)
    assert first  # non-trivial fixture
    for mutant in first:
        compile(mutant.source, mutant.mutant_id, "exec")
    # Each mutant differs from the (unparsed) original.
    original = ast.unparse(ast.parse(source))
    assert all(m.source != original for m in first)


def test_generate_mutants_rejects_syntax_error() -> None:
    with pytest.raises(mt.MutationTestError, match="does not parse"):
        mt.generate_mutants("def f(:\n", filename="broken.py")


def test_no_mutants_for_flat_source() -> None:
    assert _mutants("x = 'text'\n") == []


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _fake_mutants(n: int) -> list[mt.Mutant]:
    return [
        mt.Mutant(f"M{i:04d}", "numeric-perturb", i, f"int {i}->{i + 1}", f"x = {i + 1}")
        for i in range(n)
    ]


def test_sample_mutants_cap_is_seeded_and_order_preserving() -> None:
    mutants = _fake_mutants(50)
    a = mt.sample_mutants(mutants, 10, seed=7)
    b = mt.sample_mutants(mutants, 10, seed=7)
    assert [m.mutant_id for m in a] == [m.mutant_id for m in b]
    assert len(a) == 10
    indices = [int(m.mutant_id[1:]) for m in a]
    assert indices == sorted(indices)


def test_sample_mutants_no_cap_below_limit() -> None:
    mutants = _fake_mutants(5)
    assert mt.sample_mutants(mutants, 10, seed=0) == mutants
    with pytest.raises(mt.MutationTestError):
        mt.sample_mutants(mutants, 0, seed=0)


# ---------------------------------------------------------------------------
# Verdict classification + runner bookkeeping with fake runners
# ---------------------------------------------------------------------------


def test_classify_exit_codes() -> None:
    assert mt.classify(None) == "timeout"
    assert mt.classify(0) == "survived"
    assert mt.classify(1) == "killed"
    assert mt.classify(2) == "killed"
    # rc=4 after a green baseline = mutant crashed conftest imports (this
    # repo's tests/conftest.py eagerly imports the research layer).
    assert mt.classify(4) == "killed"
    assert mt.classify(3) == "error"
    assert mt.classify(5) == "error"


def _campaign(tmp_path: Path, runner: mt.Runner, mutants: list[mt.Mutant]) -> list[mt.MutantResult]:
    return mt.run_campaign(
        mutants,
        ("tests/test_fake.py",),
        repo=tmp_path,
        overlay_dir=tmp_path / "overlay",
        module_name="pkg.mod",
        timeout_s=1.0,
        runner=runner,
        preflight=False,
    )


def test_run_campaign_always_fail_kills_everything(tmp_path: Path) -> None:
    def runner(command, cwd, env, timeout_s):
        assert "-x" in command and "tests/test_fake.py" in command
        return mt.RunOutcome(returncode=1, duration_s=0.01)

    results = _campaign(tmp_path, runner, _fake_mutants(4))
    assert [r.status for r in results] == ["killed"] * 4
    assert all(r.returncode == 1 for r in results)


def test_run_campaign_always_pass_survives_everything(tmp_path: Path) -> None:
    results = _campaign(
        tmp_path, lambda *a: mt.RunOutcome(returncode=0, duration_s=0.01), _fake_mutants(3)
    )
    assert [r.status for r in results] == ["survived"] * 3


def test_run_campaign_timeout_and_error_verdicts(tmp_path: Path) -> None:
    outcomes = iter(
        [
            mt.RunOutcome(returncode=None, duration_s=1.0),
            mt.RunOutcome(returncode=5, duration_s=0.01),
            mt.RunOutcome(returncode=2, duration_s=0.01),
        ]
    )
    results = _campaign(tmp_path, lambda *a: next(outcomes), _fake_mutants(3))
    assert [r.status for r in results] == ["timeout", "error", "killed"]


def test_run_campaign_writes_overlay_per_mutant(tmp_path: Path) -> None:
    seen: list[str] = []

    def runner(command, cwd, env, timeout_s):
        path = Path(env["MUTATION_TARGET_FILE"])
        seen.append(path.read_text(encoding="utf-8"))
        assert env["MUTATION_TARGET_MODULE"] == "pkg.mod"
        assert (path.parent / "sitecustomize.py").is_file()
        return mt.RunOutcome(returncode=1, duration_s=0.0)

    mutants = _fake_mutants(2)
    _campaign(tmp_path, runner, mutants)
    assert seen == [m.source for m in mutants]


def test_overlay_env_pythonpath_first_and_coverage_stripped(tmp_path: Path) -> None:
    env = mt.overlay_env(
        tmp_path,
        "pkg.mod",
        tmp_path / "mutant_module.py",
        base_env={"PYTHONPATH": "/elsewhere", "COVERAGE_PROCESS_START": "1"},
    )
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)
    assert "/elsewhere" in env["PYTHONPATH"].split(os.pathsep)
    assert "COVERAGE_PROCESS_START" not in env


# ---------------------------------------------------------------------------
# Report scoring
# ---------------------------------------------------------------------------


def _result(status: str, i: int) -> mt.MutantResult:
    return mt.MutantResult(f"M{i:04d}", "numeric-perturb", i, f"int {i}->{i + 1}", status, 1, 0.01)


def test_build_report_score_excludes_errors_counts_timeouts() -> None:
    results = [
        _result("killed", 0),
        _result("killed", 1),
        _result("timeout", 2),
        _result("survived", 3),
        _result("error", 4),
    ]
    report = mt.build_report(Path("t.py"), "pkg.t", ("tests/test_t.py",), 0, 1.0, 99, results)
    # (2 killed + 1 timeout) / (5 sampled - 1 error) = 0.75
    assert report.score == 0.75
    assert report.counts == {"killed": 2, "survived": 1, "timeout": 1, "error": 1}
    assert report.total_mutants == 99 and report.sampled == 5
    assert [row["mutant_id"] for row in report.survivors] == ["M0003"]
    assert report.watermark == mt.WATERMARK
    payload = mt.report_to_dict(report)
    assert json.loads(json.dumps(payload))["score"] == 0.75


def test_format_summary_lists_survivors() -> None:
    report = mt.build_report(
        Path("t.py"), "pkg.t", ("tests/test_t.py",), 0, 1.0, 1, [_result("survived", 0)]
    )
    text = mt.format_summary(report)
    assert "mutation score: 0.00%" in text
    assert "M0000" in text and "survivors:" in text


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def _repo_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "pkg" / "mod.py").write_text(
        "def lt(a, b):\n    return a < b\n", encoding="utf-8"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_mod.py").write_text(
        "def test_noop():\n    assert True\n", encoding="utf-8"
    )
    return repo


def test_resolve_target_rejects_outside_repo(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(mt.MutationTestError, match="outside the repo"):
        mt.resolve_target(outside, repo)


def test_resolve_target_rejects_missing_nonpy_and_nonpackage(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    with pytest.raises(mt.MutationTestError, match="does not exist"):
        mt.resolve_target("src/pkg/nope.py", repo)
    (repo / "src" / "data.txt").write_text("x", encoding="utf-8")
    with pytest.raises(mt.MutationTestError, match=r"\.py file"):
        mt.resolve_target("src/data.txt", repo)
    (repo / "src" / "loose.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(mt.MutationTestError, match="importable package"):
        mt.resolve_target("src/loose.py", repo)


def test_resolve_target_accepts_relative_and_absolute(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    resolved = mt.resolve_target("src/pkg/mod.py", repo)
    assert resolved == mt.resolve_target(repo / "src" / "pkg" / "mod.py", repo)
    assert mt.module_name_for(resolved) == "pkg.mod"


def test_validate_selectors_fail_closed(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    with pytest.raises(mt.MutationTestError, match="empty"):
        mt.validate_selectors((), repo)
    with pytest.raises(mt.MutationTestError, match="non-empty"):
        mt.validate_selectors(("   ",), repo)
    with pytest.raises(mt.MutationTestError, match="option"):
        mt.validate_selectors(("-k", "everything"), repo)
    with pytest.raises(mt.MutationTestError, match="outside the repo"):
        mt.validate_selectors((str(tmp_path / "other_test.py"),), repo)


def test_validate_selectors_accepts_nodeid_forms(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    assert mt.validate_selectors(("tests/test_mod.py",), repo) == ("tests/test_mod.py",)
    nodeid = "tests/test_mod.py::test_noop"
    assert mt.validate_selectors((nodeid,), repo) == (nodeid,)


def test_preflight_rejects_red_baseline(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)

    def runner(command, cwd, env, timeout_s):
        return mt.RunOutcome(returncode=1, duration_s=0.0, output_tail="boom")

    with pytest.raises(mt.MutationTestError, match="baseline pytest run did not pass"):
        mt.run_campaign(
            _fake_mutants(1),
            ("tests/test_mod.py",),
            repo=repo,
            overlay_dir=tmp_path / "overlay",
            module_name="pkg.mod",
            timeout_s=1.0,
            runner=runner,
            original_source="x = 1\n",
        )


def test_preflight_rejects_surviving_sentinel(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)

    def runner(command, cwd, env, timeout_s):
        return mt.RunOutcome(returncode=0, duration_s=0.0)

    with pytest.raises(mt.MutationTestError, match="sentinel overlay was not killed"):
        mt.run_campaign(
            _fake_mutants(1),
            ("tests/test_mod.py",),
            repo=repo,
            overlay_dir=tmp_path / "overlay",
            module_name="pkg.mod",
            timeout_s=1.0,
            runner=runner,
            original_source="x = 1\n",
        )


def test_preflight_rejects_missing_original_source(tmp_path: Path) -> None:
    with pytest.raises(mt.MutationTestError, match="original_source"):
        mt.run_campaign(
            _fake_mutants(1),
            ("tests/test_mod.py",),
            repo=tmp_path,
            overlay_dir=tmp_path / "overlay",
            module_name="pkg.mod",
            timeout_s=1.0,
            runner=lambda *a: mt.RunOutcome(returncode=0, duration_s=0.0),
        )


# ---------------------------------------------------------------------------
# Overlay mechanism, proven with one real subprocess (no pytest involved)
# ---------------------------------------------------------------------------


def test_overlay_redirects_import_in_real_subprocess(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    overlay_dir = tmp_path / "overlay"
    mutant_file = mt.write_overlay(overlay_dir, "VALUE = 2\n")
    env = mt.overlay_env(overlay_dir, "pkg.mod", mutant_file, base_env=dict())
    env["PYTHONPATH"] = f"{overlay_dir}{':'}{repo / 'src'}"
    env["MUTATION_TARGET_MODULE"] = "pkg.mod"
    env["MUTATION_TARGET_FILE"] = str(mutant_file)
    completed = subprocess.run(
        [sys.executable, "-c", "import pkg.mod; print(pkg.mod.VALUE); print(pkg.mod.__file__)"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value, module_file = completed.stdout.split()
    assert value == "2"
    assert module_file == str(mutant_file)


# ---------------------------------------------------------------------------
# main() bookkeeping end to end with a stubbed subprocess runner
# ---------------------------------------------------------------------------


def _stub_runner_factory(repo: Path, kill: str = "any") -> mt.Runner:
    """Fake runner that reads the overlay file: kill per policy, else pass."""
    original = (repo / "src" / "pkg" / "mod.py").read_text(encoding="utf-8")

    def runner(command, cwd, env, timeout_s):
        content = Path(env["MUTATION_TARGET_FILE"]).read_text(encoding="utf-8")
        assert cwd == repo
        if content == original:
            return mt.RunOutcome(returncode=0, duration_s=0.0)  # baseline passes
        if kill == "any" or mt.SENTINEL_MARKER in content:
            return mt.RunOutcome(returncode=1, duration_s=0.0)  # killed
        return mt.RunOutcome(returncode=0, duration_s=0.0)  # survived

    return runner


def test_main_end_to_end_all_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo_fixture(tmp_path)
    monkeypatch.setattr(mt, "default_runner", _stub_runner_factory(repo))
    json_out = tmp_path / "report.json"
    code = mt.main(
        [
            "--target",
            "src/pkg/mod.py",
            "--pytest",
            "tests/test_mod.py",
            "--seed",
            "3",
            "--max-mutants",
            "5",
            "--timeout",
            "5",
            "--json-out",
            str(json_out),
        ],
        repo=repo,
    )
    assert code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["module"] == "pkg.mod"
    assert payload["score"] == 1.0
    assert payload["counts"]["killed"] == payload["sampled"] >= 1
    assert payload["watermark"] == mt.WATERMARK
    assert payload["survivors"] == []


def test_main_fails_closed_when_sentinel_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_fixture(tmp_path)
    monkeypatch.setattr(
        mt, "default_runner", lambda *a: mt.RunOutcome(returncode=0, duration_s=0.0)
    )
    code = mt.main(
        ["--target", "src/pkg/mod.py", "--pytest", "tests/test_mod.py"],
        repo=repo,
    )
    assert code == 2


def test_main_min_score_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo_fixture(tmp_path)
    # Only the sentinel is killed; every real mutant survives.
    monkeypatch.setattr(mt, "default_runner", _stub_runner_factory(repo, kill="sentinel"))
    code = mt.main(
        [
            "--target",
            "src/pkg/mod.py",
            "--pytest",
            "tests/test_mod.py",
            "--min-score",
            "0.5",
        ],
        repo=repo,
    )
    assert code == 1


def test_main_rejects_bad_arguments(tmp_path: Path) -> None:
    repo = _repo_fixture(tmp_path)
    assert (
        mt.main(
            ["--target", "src/pkg/mod.py", "--pytest", "tests/test_mod.py", "--max-mutants", "0"],
            repo=repo,
        )
        == 2
    )
    assert (
        mt.main(
            ["--target", "src/pkg/mod.py", "--pytest", "tests/test_mod.py", "--timeout", "-1"],
            repo=repo,
        )
        == 2
    )
    assert mt.main(["--target", "src/pkg/mod.py"], repo=repo) == 2  # no selectors
    assert mt.main(["--target", "/etc/passwd", "--pytest", "tests/test_mod.py"], repo=repo) == 2


def test_main_list_mutants_does_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo_fixture(tmp_path)

    def explode(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("runner invoked despite --list-mutants")

    monkeypatch.setattr(mt, "default_runner", explode)
    code = mt.main(
        ["--target", "src/pkg/mod.py", "--list-mutants"],
        repo=repo,
    )
    assert code == 0
