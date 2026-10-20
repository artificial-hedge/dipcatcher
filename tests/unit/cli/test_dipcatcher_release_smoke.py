"""Offline installed-wheel release smoke: record shape, and one real wheel."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts.dipcatcher_release_smoke import (
    BUILD_ISOLATED,
    MODE_LOCKFILE,
    MODE_PTH,
    MODE_UV_PIP,
    NOT_COVERED,
    OWNED,
    RETRY_DELAY_S,
    SCHEMA,
    SCOPE,
    STRATEGY,
    Attempt,
    CheckSpec,
    CommandResult,
    assemble_record,
    atomic_write_text,
    build_failure_status,
    evaluate_check,
    format_sha256sums,
    import_retry_reason,
    main,
    modules_match,
    owned_package_dirs,
    sha256_file,
    wheel_version,
    write_checksum_file,
    write_third_party_pth,
)

ROOT = Path(__file__).resolve().parents[3]
_FIXED = datetime(2026, 10, 12, 5, 30, tzinfo=UTC)
_KEYS = ("sk-smoke-fx1-aaa111", "sk-smoke-lite-bbb222")


def _clock() -> datetime:
    return _FIXED


def _flag(args: Sequence[str], name: str) -> str:
    items = list(args)
    if name not in items:
        return ""
    index = items.index(name)
    if index + 1 >= len(items):
        return ""
    return items[index + 1]


def _ok(text: str) -> CommandResult:
    if not text.endswith("\n"):
        text += "\n"
    return CommandResult(0, text, "", False)


def _probe_payload(pure: str) -> str:
    modules = {name: {"file": f"{pure}/{name}/__init__.py"} for name in OWNED}
    return json.dumps({"modules": modules, "purelib": pure, "prefix": pure})


class _Scripted:
    """In-process stand-in for uv and the installed console scripts."""

    def __init__(
        self, sites: Mapping[str, str], *, sync_code: int, help_failures: int, help_kind: str
    ) -> None:
        self.sites = dict(sites)
        self.sync_code = sync_code
        self.help_failures = help_failures
        self.help_kind = help_kind
        self.builds = 0

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: float,
        stdin: bytes = b"",
    ) -> CommandResult:
        del cwd, env, timeout_s, stdin
        tokens = [str(part) for part in args]
        head = tokens[1] if len(tokens) > 1 else ""
        if head == "--version":
            return _ok("uv 0.13.0")
        if "build" in tokens and "--wheel" in tokens:
            return self._build(tokens)
        if head == "venv":
            return _ok("created")
        if head == "sync":
            if self.sync_code != 0:
                return CommandResult(self.sync_code, "", "sync failed\n", False)
            return _ok("synced")
        if tokens[1:3] == ["pip", "install"]:
            return self._pip(tokens)
        if "uninstall" in tokens:
            return _ok("uninstalled")
        if "-c" in tokens:
            return self._python(tokens)
        return self._cli(tokens)

    def _build(self, tokens: list[str]) -> CommandResult:
        out = Path(_flag(tokens, "--out-dir"))
        out.mkdir(parents=True, exist_ok=True)
        wheel = out / "fx_1-0.4.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel-bytes-v1")
        self.builds += 1
        return _ok(f"Successfully built {wheel}")

    def _pip(self, tokens: list[str]) -> CommandResult:
        if "--no-deps" in tokens:
            return _ok("installed")
        return CommandResult(
            1,
            "",
            "Because example was not found in the cache and the network was disabled\n",
            False,
        )

    def _python(self, tokens: list[str]) -> CommandResult:
        code = tokens[tokens.index("-c") + 1]
        if "dipcatcher.release_smoke.probe" in code:
            pure = self._site(tokens[0])
            return CommandResult(0, _probe_payload(pure), "", False)
        if "dipcatcher.release_smoke.purelib" in code:
            return _ok(self._site(tokens[0]))
        if "import dipcatcher_cli" in code:
            return CommandResult(1, "", "ModuleNotFoundError: dipcatcher_cli\n", False)
        return CommandResult(97, "", f"unhandled python: {code}\n", False)

    def _site(self, executable: str) -> str:
        if "venv-pth" in executable:
            return self.sites["pth"]
        if "venv-sync" in executable:
            return self.sites["sync"]
        return self.sites["project"]

    def _cli(self, tokens: list[str]) -> CommandResult:
        if "exec" in tokens and "--help" in tokens:
            return _ok("usage: dipcatcher exec")
        if "--model" in tokens:
            model = tokens[tokens.index("--model") + 1]
            return _ok(f"stub {model} reply")
        if tokens[-1] == "version":
            return _ok("dipcatcher 0.4.0")
        if "models" in tokens:
            return _ok("fx1          unset\nfx1-lite     unset")
        if "doctor" in tokens:
            return _ok("Usage: dipcatcher doctor")
        if "-m" in tokens:
            return _ok("Usage: dipcatcher")
        if self._plain_help(tokens) and self.help_failures > 0:
            self.help_failures -= 1
            if self.help_kind == "import":
                return CommandResult(1, "", "ImportError: half-written module\n", False)
            return CommandResult(2, "", "usage error\n", False)
        if "--help" in tokens or len(tokens) == 1:
            return _ok("Start a conversation\nUsage: dipcatcher\nfxi")
        return CommandResult(97, "", "unhandled: " + " ".join(tokens) + "\n", False)

    def _plain_help(self, tokens: list[str]) -> bool:
        if "--help" not in tokens or "-m" in tokens:
            return False
        if "exec" in tokens or "doctor" in tokens:
            return False
        return Path(tokens[0]).name == "dipcatcher"


def _sites(tmp_path: Path) -> dict[str, str]:
    return {
        "sync": str(tmp_path / "sync-site"),
        "pth": str(tmp_path / "pth-site"),
        "project": str(tmp_path / "project-site"),
    }


def _run(
    tmp_path: Path,
    scripted: _Scripted,
    sleeps: list[float],
    *,
    checksums: bool = False,
) -> tuple[int, dict[str, object]]:
    out = tmp_path / "evidence.json"
    argv = [
        "--repo",
        str(tmp_path / "repo"),
        "--work-dir",
        str(tmp_path / "work"),
        "--out",
        str(out),
        "--uv",
        "uv",
        "--python",
        "python",
    ]
    if checksums:
        argv.extend(["--checksums", str(tmp_path / "SHA256SUMS")])
    code = main(
        argv,
        runner=scripted,
        sleep=sleeps.append,
        clock=_clock,
        stub_url="http://127.0.0.1:9",
        keys=_KEYS,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    return code, payload


def _scripted(
    tmp_path: Path, *, sync_code: int = 0, help_failures: int = 0, help_kind: str = "usage"
) -> _Scripted:
    return _Scripted(
        _sites(tmp_path), sync_code=sync_code, help_failures=help_failures, help_kind=help_kind
    )


def test_record_schema_and_secret_scrub() -> None:
    attempt = Attempt(
        wheel_name="fx_1-0.4.0-py3-none-any.whl",
        wheel_sha256="ab" * 32,
        build_status="ok",
        build_mode=BUILD_ISOLATED,
        install_mode=MODE_LOCKFILE,
        notes=["resolver missed the cache", "leaked sk-smoke-fx1-aaa111"],
        modules={name: f"/v/{name}/__init__.py" for name in OWNED},
        modules_from_install=True,
        checks=[
            {
                "name": "dipcatcher_help",
                "command": ["dipcatcher", "--help"],
                "exit": 0,
                "passed": True,
                "detail": "",
                "import_failure": False,
            }
        ],
    )
    record = assemble_record(
        attempt,
        started_at="2026-10-12T05:30:00Z",
        finished_at="2026-10-12T05:31:00Z",
        retry={"attempted": False, "reason": ""},
        uv_version="uv 0.13.0",
        base_python="/usr/bin/python",
        secrets=_KEYS,
    )
    assert record["schema"] == SCHEMA
    assert record["passed"] is True
    assert record["not_covered"] == list(NOT_COVERED)
    assert "Signing" in SCOPE
    assert "rollback" in str(record["scope"])
    assert "macOS/Windows" in str(record["scope"])
    assert record["strategy"] == list(STRATEGY)
    assert "import_failure" not in record["checks"][0]
    encoded = json.dumps(record)
    assert "sk-smoke-fx1-aaa111" not in encoded
    assert "[redacted]" in encoded


def test_failed_check_fails_the_record() -> None:
    attempt = Attempt(
        wheel_name="fx_1-0.4.0-py3-none-any.whl",
        wheel_sha256="cd" * 32,
        build_status="ok",
        build_mode=BUILD_ISOLATED,
        install_mode=MODE_LOCKFILE,
        modules_from_install=True,
        checks=[
            {
                "name": "quant_help",
                "command": ["quant", "--help"],
                "exit": 2,
                "passed": False,
                "detail": "exit 2",
                "import_failure": False,
            }
        ],
    )
    record = assemble_record(
        attempt,
        started_at="2026-10-12T05:30:00Z",
        finished_at="2026-10-12T05:30:01Z",
        retry={"attempted": False, "reason": ""},
        uv_version="",
        base_python="python",
        secrets=(),
    )
    assert record["passed"] is False


def test_modules_outside_the_install_fail_the_record() -> None:
    attempt = Attempt(
        build_status="ok",
        modules_from_install=False,
        checks=[
            {
                "name": "modules_from_wheel",
                "command": ["python"],
                "exit": 1,
                "passed": False,
                "detail": "outside",
                "import_failure": False,
            }
        ],
    )
    record = assemble_record(
        attempt,
        started_at="t",
        finished_at="t",
        retry={"attempted": False, "reason": ""},
        uv_version="",
        base_python="python",
        secrets=(),
    )
    assert record["passed"] is False


def test_checksum_file_matches_bytes_and_mode(tmp_path: Path) -> None:
    wheel = tmp_path / "fx_1-0.4.0-py3-none-any.whl"
    evidence = tmp_path / "evidence.json"
    wheel.write_bytes(b"wheel-bytes-v1")
    evidence.write_text("{}\n", encoding="utf-8")
    sums = tmp_path / "SHA256SUMS"
    entries = write_checksum_file(sums, [(wheel.name, wheel), (evidence.name, evidence)])
    assert entries[wheel.name] == sha256_file(wheel)
    text = sums.read_text(encoding="utf-8")
    assert text == format_sha256sums(entries)
    assert text.endswith("\n")
    assert f"{entries[evidence.name]}  {evidence.name}" in text
    assert stat.S_IMODE(sums.stat().st_mode) == 0o600
    again = tmp_path / "note.txt"
    atomic_write_text(again, "hello\n")
    assert again.read_text(encoding="utf-8") == "hello\n"
    assert stat.S_IMODE(again.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".note.txt.*")) == []


def test_checksum_rejects_duplicate_names(tmp_path: Path) -> None:
    artifact = tmp_path / "a.txt"
    artifact.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        write_checksum_file(tmp_path / "SHA256SUMS", [("a.txt", artifact), ("a.txt", artifact)])


def test_evaluate_check_timeout_import_and_scrub() -> None:
    spec = CheckSpec("dipcatcher_help", ("dipcatcher", "--help"), stdout_contains=("Usage:",))
    timed = evaluate_check(spec, CommandResult(-1, "", "", True), ())
    assert timed["passed"] is False
    assert timed["detail"] == "timed out"
    assert timed["import_failure"] is False
    broken = evaluate_check(spec, CommandResult(1, "", "ImportError: half-written\n", False), ())
    assert broken["import_failure"] is True
    assert import_retry_reason([broken]) == "dipcatcher_help"
    leaked = evaluate_check(
        CheckSpec("n", ("dipcatcher",), stdout_exact="ok"),
        CommandResult(1, "", "saw sk-smoke-fx1-aaa111", False),
        _KEYS,
    )
    assert "sk-smoke" not in str(leaked["detail"])
    assert import_retry_reason([timed]) == ""


def test_wheel_version_and_module_roots(tmp_path: Path) -> None:
    assert wheel_version("fx_1-0.4.0-py3-none-any.whl") == "0.4.0"
    assert wheel_version("not-a-wheel.txt") == ""
    root = tmp_path / "site"
    files = {name: str(root / name / "__init__.py") for name in OWNED}
    assert modules_match(files, root) is True
    files["fx1"] = str(tmp_path / "src" / "fx1" / "__init__.py")
    assert modules_match(files, root) is False
    assert build_failure_status("the network was disabled") == "offline-unavailable"
    assert build_failure_status("executable not found: uv") == "offline-unavailable"
    assert build_failure_status("SyntaxError: invalid syntax") == "failed"


def test_owned_package_dirs_ignore_editable_and_quant_tree(tmp_path: Path) -> None:
    site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (site / "_editable_impl_fx_1.pth").write_text("/src\n", encoding="utf-8")
    (site / "quant_fund").mkdir()
    assert owned_package_dirs(tmp_path) == set()
    package = site / "fx1"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '0'\n", encoding="utf-8")
    found = owned_package_dirs(tmp_path)
    assert len(found) == 1
    assert found.pop().endswith(f"fx1{os.sep}__init__.py")


def test_third_party_pth_points_at_the_project_site(tmp_path: Path) -> None:
    site = tmp_path / "venv-site"
    project = tmp_path / "project-site"
    target = write_third_party_pth(site, project)
    text = target.read_text(encoding="utf-8")
    assert str(project) in text
    assert "fx1, quant_fund, and dipcatcher_cli must resolve from this install" in text


def test_scripted_runner_uses_lockfile_install_and_checksums(tmp_path: Path) -> None:
    scripted = _scripted(tmp_path)
    code, record = _run(tmp_path, scripted, [], checksums=True)
    assert code == 0
    assert record["passed"] is True
    assert record["schema"] == SCHEMA
    assert record["install_mode"] == MODE_LOCKFILE
    assert record["build_mode"] == BUILD_ISOLATED
    assert record["build_status"] == "ok"
    assert scripted.builds == 1
    notes = " ".join(str(note) for note in record["install_notes"])
    assert "network was disabled" in notes
    wheel = tmp_path / "work" / "wheel" / "fx_1-0.4.0-py3-none-any.whl"
    assert record["wheel"] == {"name": wheel.name, "sha256": sha256_file(wheel)}
    sums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert sha256_file(wheel) in sums
    assert sha256_file(tmp_path / "evidence.json") in sums
    assert stat.S_IMODE((tmp_path / "evidence.json").stat().st_mode) == 0o600
    encoded = json.dumps(record)
    assert _KEYS[0] not in encoded
    assert _KEYS[1] not in encoded
    assert all(row["passed"] is True for row in record["checks"])
    assert not (tmp_path / "work" / "venv-sync").exists()


def test_import_failure_retries_the_build_once(tmp_path: Path) -> None:
    scripted = _scripted(tmp_path, help_failures=1, help_kind="import")
    sleeps: list[float] = []
    code, record = _run(tmp_path, scripted, sleeps)
    assert code == 0
    assert sleeps == [RETRY_DELAY_S]
    assert scripted.builds == 2
    retry = record["retry"]
    assert retry == {"attempted": True, "reason": "dipcatcher_help"}
    assert record["passed"] is True


def test_usage_failure_does_not_retry_and_exits_nonzero(tmp_path: Path) -> None:
    scripted = _scripted(tmp_path, help_failures=1, help_kind="usage")
    sleeps: list[float] = []
    code, record = _run(tmp_path, scripted, sleeps)
    assert code == 1
    assert sleeps == []
    assert scripted.builds == 1
    assert record["passed"] is False
    help_row = next(row for row in record["checks"] if row["name"] == "dipcatcher_help")
    assert help_row["exit"] == 2
    assert help_row["passed"] is False
    assert record["retry"] == {"attempted": False, "reason": ""}


def test_pth_fallback_when_lockfile_install_fails(tmp_path: Path) -> None:
    scripted = _scripted(tmp_path, sync_code=1)
    code, record = _run(tmp_path, scripted, [])
    assert code == 0
    assert record["install_mode"] == MODE_PTH
    assert record["modules_from_install"] is True
    pth = Path(str(_sites(tmp_path)["pth"])) / "zz-third-party.pth"
    assert _sites(tmp_path)["project"] in pth.read_text(encoding="utf-8")
    for path in record["modules"].values():
        assert _sites(tmp_path)["pth"] in str(path)


@pytest.mark.slow
def test_real_wheel_installs_offline_and_console_scripts_work(tmp_path: Path) -> None:
    """Build the working tree and exercise the installed scripts. Skips if offline build is impossible."""
    out = tmp_path / "evidence.json"
    sums = tmp_path / "SHA256SUMS"
    work = tmp_path / "work"
    before = owned_package_dirs(ROOT)
    code = main(
        [
            "--repo",
            str(ROOT),
            "--work-dir",
            str(work),
            "--out",
            str(out),
            "--checksums",
            str(sums),
        ],
        keys=_KEYS,
    )
    record = json.loads(out.read_text(encoding="utf-8"))
    if code != 0 and record.get("build_status") == "offline-unavailable":
        notes = record.get("install_notes", [])
        detail = "; ".join(str(note) for note in notes[-3:])
        pytest.skip(f"wheel cannot be built offline here: {detail}")
    failed = [row for row in record["checks"] if row.get("passed") is not True]
    assert code == 0, json.dumps(
        {
            "build_status": record.get("build_status"),
            "install_mode": record.get("install_mode"),
            "failed": failed,
            "notes": record.get("install_notes"),
        },
        indent=2,
    )
    assert record["schema"] == SCHEMA
    assert record["passed"] is True
    assert record["install_mode"] in {MODE_UV_PIP, MODE_LOCKFILE, MODE_PTH}
    assert record["not_covered"] == list(NOT_COVERED)
    assert "Signing" in str(record["scope"])
    assert str(record.get("uv_version", "")).startswith("uv ")
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert stat.S_IMODE(sums.stat().st_mode) == 0o600
    wheels = list((work / "wheel").glob("*.whl"))
    assert len(wheels) == 1
    assert record["wheel"]["sha256"] == sha256_file(wheels[0])
    assert record["wheel"]["name"] == wheels[0].name
    sum_text = sums.read_text(encoding="utf-8")
    assert record["wheel"]["sha256"] in sum_text
    assert sha256_file(out) in sum_text
    names = [row["name"] for row in record["checks"]]
    for required in (
        "modules_from_wheel",
        "dipcatcher_help",
        "dipcatcher_version",
        "dipcatcher_exec_help",
        "dipcatcher_models_no_endpoint",
        "dipcatcher_exec_fx1",
        "dipcatcher_exec_fx1_lite",
        "quant_help",
        "fxi_help",
        "dipcatcher_doctor_help",
        "dipcatcher_bare_nontty",
        "python_m_dipcatcher_cli_help",
        "console_scripts_removed",
        "owned_import_removed",
        "project_env_unchanged",
    ):
        assert required in names
    for row in record["checks"]:
        assert row["passed"] is True
        if row["name"] == "owned_import_removed":
            assert row["exit"] == 1
        else:
            assert row["exit"] == 0
    folder = {MODE_UV_PIP: "venv-pip", MODE_LOCKFILE: "venv-sync", MODE_PTH: "venv-pth"}
    marker = folder[str(record["install_mode"])]
    for path in record["modules"].values():
        assert marker in str(path)
        assert "site-packages" in str(path)
        assert str(ROOT / "src") not in str(path)
    assert not (work / marker).exists()
    assert owned_package_dirs(ROOT) == before
    encoded = out.read_text(encoding="utf-8")
    assert _KEYS[0] not in encoded
    assert _KEYS[1] not in encoded
