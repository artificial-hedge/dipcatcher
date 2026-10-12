#!/usr/bin/env python3
"""Smoke an installed dipcatcher wheel from an empty working directory.

Builds a wheel under a temp directory, installs it ``--no-deps`` into a fresh
venv, and satisfies third-party imports from the checkout ``.venv`` via a
``site.addsitedir`` bridge. The installed ``fx1`` / ``quant_fund`` /
``dipcatcher_cli`` must shadow the checkout. Prints one JSON report on stdout.

Exit 0 when every probe matches the locked audit (including known defects).
Exit 1 on a regression. Exit 2 when ``uv build`` cannot produce a wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

HEAVY_MODULES: tuple[str, ...] = (
    "numpy",
    "polars",
    "pandas",
    "torch",
    "scipy",
    "sklearn",
    "duckdb",
    "pyarrow",
)
WHEEL_PACKAGES: tuple[str, ...] = ("src/fx1", "src/quant_fund", "src/dipcatcher_cli")
PACKAGED_FILES: tuple[str, ...] = (
    "src/fx1/prompts/fx1_system.md",
    "src/quant_fund/research/receipt_v2.schema.json",
    "src/quant_fund/research/research100.json",
    "src/quant_fund/py.typed",
    "src/quant_fund/stress/public_domain/fred_h10_h15.json",
    "src/quant_fund/market_sim/_lob_core.c",
)
LAYOUT_FILES: tuple[str, ...] = (
    "configs/research.yaml",
    "configs/base.yaml",
    "configs/backtest.yaml",
    "configs/paper.yaml",
    "configs/sim_live.yaml",
    "configs/net_tournament.json",
)
WHEEL_MEMBERS: tuple[str, ...] = (
    "fx1/prompts/fx1_system.md",
    "quant_fund/research/receipt_v2.schema.json",
    "quant_fund/research/research100.json",
    "quant_fund/py.typed",
    "quant_fund/stress/public_domain/fred_h10_h15.json",
    "quant_fund/market_sim/_lob_core.c",
)
BUNDLED_WHEEL: tuple[str, ...] = tuple(
    f"quant_fund/_bundled/configs/{name}"
    for name in (
        "base.yaml",
        "research.yaml",
        "backtest.yaml",
        "paper.yaml",
        "sim_live.yaml",
        "net_tournament.json",
    )
)
SOURCE_ANCHORS: tuple[tuple[str, str], ...] = (
    ("src/quant_fund/cli/data_cmds.py", 'Path("configs/research.yaml")'),
    ("src/fx1/harness.py", "def _configs_dir"),
    ("src/fx1/interactive/system_prompt.py", "fx1_system.md"),
    ("src/quant_fund/schemas/receipt_v2_leaf.py", 'files("quant_fund.research")'),
    ("src/quant_fund/cli/capability_cmds.py", '"src" / "quant_fund"'),
    ("src/quant_fund/cli/score_cmds.py", 'Path("receipts")'),
)
RECEIPT_RELATIVE = "receipts/corpus_proof_exemplar.json"


class BuildBackendError(RuntimeError):
    """``uv`` could not build or install the wheel."""


@dataclass(frozen=True)
class Expect:
    name: str
    kind: str
    args: tuple[str, ...]
    expect_exit: int
    classification: str
    stdout_has: tuple[str, ...] = ()
    stderr_has: tuple[str, ...] = ()
    stderr_lacks: tuple[str, ...] = ()
    defect: str | None = None
    timeout_s: float = 60.0


@dataclass(frozen=True)
class CmdCapture:
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None


GENERIC: tuple[Expect, ...] = (
    Expect("dipcatcher --version", "script", ("dipcatcher", "--version"), 0, "pass"),
    Expect(
        "python -m dipcatcher_cli --version",
        "module",
        ("dipcatcher_cli", "--version"),
        0,
        "pass",
    ),
    Expect("quant --version", "script", ("quant", "--version"), 0, "pass"),
    Expect(
        "dipcatcher --help",
        "script",
        ("dipcatcher", "--help"),
        0,
        "pass",
        stdout_has=("Start a conversation: dipcatcher", "Usage:"),
    ),
    Expect(
        "dipcatcher chat",
        "script",
        ("dipcatcher", "chat"),
        2,
        "pass",
        stderr_has=("requires an interactive terminal",),
    ),
    Expect(
        "dipcatcher",
        "script",
        ("dipcatcher",),
        0,
        "pass",
        stdout_has=("Start a conversation: dipcatcher", "Usage:"),
    ),
    Expect(
        "capabilities list --json --executable-only",
        "script",
        ("dipcatcher", "capabilities", "list", "--json", "--executable-only"),
        0,
        "pass",
        stdout_has=('"schema": "dipcatcher.capabilities.v1"',),
    ),
    Expect(
        "families",
        "script",
        ("dipcatcher", "families"),
        0,
        "pass",
        stdout_has=('"schema": "dipcatcher.family_surface.v1"',),
    ),
    Expect(
        "score-forecasts --inspect",
        "script",
        ("dipcatcher", "score-forecasts", "--task", "volatility", "--data", "{csv}", "--inspect"),
        0,
        "pass",
        stdout_has=('"schema": "dataset_profile.v1"',),
    ),
    Expect(
        "verify-receipt",
        "script",
        ("dipcatcher", "verify-receipt", "{receipt}"),
        0,
        "pass",
        stdout_has=('"valid": true', '"schema": "receipt.v2"'),
    ),
    Expect(
        "python -m dipcatcher_cli doctor",
        "module",
        ("dipcatcher_cli", "doctor"),
        1,
        "pass",
        stdout_has=("data_manifest: missing", "research_receipt: missing", "dir_raw: missing"),
        stderr_lacks=("Traceback", "FileNotFoundError"),
    ),
    Expect(
        "doctor missing explicit config",
        "module",
        ("dipcatcher_cli", "doctor", "--config", "{missing}"),
        2,
        "pass",
        stderr_has=("config not found:",),
        stderr_lacks=("Traceback",),
    ),
    Expect(
        "fxi --version",
        "script",
        ("fxi", "--version"),
        0,
        "pass",
        stdout_has=("fxi ",),
    ),
    Expect(
        "fx1 --version",
        "script",
        ("fx1", "--version"),
        0,
        "pass",
        stdout_has=("fx1 ",),
    ),
    Expect(
        "quant --help",
        "script",
        ("quant", "--help"),
        0,
        "pass",
        stdout_has=("Usage:",),
    ),
    Expect(
        "verify-ledger --help",
        "script",
        ("verify-ledger", "--help"),
        0,
        "pass",
        stdout_has=("usage:",),
    ),
    Expect(
        "mc-engine --help",
        "script",
        ("mc-engine", "--help"),
        0,
        "pass",
        stdout_has=("Usage:",),
        timeout_s=90.0,
    ),
)


DEFECTS: dict[str, dict[str, str]] = {
    "no-dipcatcher-version": {
        "class": "d",
        "severity": "medium",
        "module": "quant_fund.cli._app.app",
        "symptom": "dipcatcher, python -m dipcatcher_cli, and quant reject --version with exit 2",
        "fix": (
            "Add an eager --version/-V callback on the root Typer app in "
            "src/quant_fund/cli/_app.py that prints quant_fund.__version__ and "
            "raises typer.Exit(). dipcatcher_cli.app.main already forwards "
            "unknown args, so both console scripts pick it up. Do not handle "
            "it only in dipcatcher_cli.app.main."
        ),
    },
    "configs-not-in-wheel": {
        "class": "a",
        "severity": "high",
        "module": "pyproject.toml [tool.hatch.build.targets.wheel]",
        "symptom": "wheel has no configs/ tree; sdist only-include lists configs but pip installs the wheel",
        "fix": (
            "force-include the configs inherit closure (at least configs/base.yaml "
            "and configs/research.yaml; also backtest.yaml, paper.yaml, "
            "sim_live.yaml, net_tournament.json) into the wheel, and resolve CLI "
            "defaults from that package data when Path.cwd()/configs/<file> is absent. "
            "install.sh never copies configs; Dockerfile only works because it "
            "COPY configs into WORKDIR /app."
        ),
    },
    "doctor-missing-config-traceback": {
        "class": "d",
        "severity": "medium",
        "module": "quant_fund.cli.data_cmds.doctor",
        "symptom": "empty cwd: uncaught FileNotFoundError for configs/research.yaml, exit 1, rich traceback",
        "fix": (
            "Catch FileNotFoundError/OSError around load_config in doctor and "
            "exit 2 with 'config not found: <path>' on stderr. Default the path "
            "to the bundled configs/research.yaml instead of Path('configs/research.yaml')."
        ),
    },
    "harness-cwd-configs-allowlist": {
        "class": "a",
        "severity": "high",
        "module": "fx1.harness._contained_config",
        "symptom": "Harness.run(config=...) raises ValueError unless the path is under Path.cwd()/configs",
        "fix": (
            "Allow the bundled configs directory in addition to Path.cwd()/configs. "
            "Keep rejecting paths outside those roots. An installed process whose "
            "cwd is not a checkout currently cannot pass any --config."
        ),
    },
    "doctor-data-lake": {
        "class": "a",
        "severity": "medium",
        "module": "quant_fund.pipeline.doctor.doctor",
        "symptom": "with configs present and no lake: exit 1, data_manifest missing, research_receipt missing",
        "fix": (
            "Do not package data/ or receipts/. After config resolution is fixed, "
            "keep this fail-closed status. Stop mkdir of data/{raw,bronze,silver,gold,metadata} "
            "and stop quant_fund.cli._app._cfg dump_resolved from writing "
            "data/metadata/resolved_config.json on the failing doctor path "
            "(data_cmds.doctor calls _cfg unconditionally)."
        ),
    },
}


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def load_pyproject(repo: Path) -> dict[str, Any]:
    import tomllib

    loaded: dict[str, Any] = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    return loaded


def wheel_target(project: Mapping[str, Any]) -> Mapping[str, Any]:
    tool = cast(Mapping[str, Any], project["tool"])
    hatch = cast(Mapping[str, Any], tool["hatch"])
    build = cast(Mapping[str, Any], hatch["build"])
    targets = cast(Mapping[str, Any], build["targets"])
    return cast(Mapping[str, Any], targets["wheel"])


def sdist_target(project: Mapping[str, Any]) -> Mapping[str, Any]:
    tool = cast(Mapping[str, Any], project["tool"])
    hatch = cast(Mapping[str, Any], tool["hatch"])
    build = cast(Mapping[str, Any], hatch["build"])
    targets = cast(Mapping[str, Any], build["targets"])
    return cast(Mapping[str, Any], targets["sdist"])


def wheel_covers(repo_rel: str, project: Mapping[str, Any]) -> bool:
    wheel = wheel_target(project)
    packages = tuple(wheel.get("packages", ()))
    force = cast(Mapping[str, str], wheel.get("force-include", {}))
    if repo_rel in force:
        return True
    for package in packages:
        prefix = str(package).rstrip("/")
        if repo_rel == prefix or repo_rel.startswith(prefix + "/"):
            return True
    return False


def layout_gaps(repo: Path) -> dict[str, Any]:
    project = load_pyproject(repo)
    sdist = sdist_target(project)
    only = tuple(sdist.get("only-include", ()))
    return {
        "packages": list(wheel_target(project).get("packages", ())),
        "force_include": dict(wheel_target(project).get("force-include", {})),
        "sdist_includes_configs": "configs" in only,
        "not_in_wheel": [item for item in LAYOUT_FILES if not wheel_covers(item, project)],
        "in_wheel": [
            item
            for item in PACKAGED_FILES
            if wheel_covers(item, project) and (repo / item).is_file()
        ],
        "scripts": dict(cast(Mapping[str, str], project["project"]["scripts"])),
    }


def source_anchors(repo: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel, needle in SOURCE_ANCHORS:
        text = (repo / rel).read_text(encoding="utf-8")
        rows.append({"path": rel, "needle": needle, "present": needle in text})
    return rows


def importlib_resource_targets(repo: Path) -> list[dict[str, object]]:
    import re

    literal = re.compile(
        r"""files\(\s*['"]([A-Za-z0-9_.]+)['"]\s*\)\.joinpath\(\s*['"]([^'"]+)['"]\s*\)"""
    )
    named = re.compile(
        r"""files\(\s*['"]([A-Za-z0-9_.]+)['"]\s*\)\.joinpath\(\s*([A-Z0-9_]+)\s*\)"""
    )
    assign = re.compile(r"""^([A-Z0-9_]+)\s*=\s*['"]([^'"]+)['"]""", re.MULTILINE)
    rows: list[dict[str, object]] = []
    for path in (repo / "src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        constants = dict(assign.findall(text))
        pairs = list(literal.findall(text))
        pairs.extend(
            (pkg, constants[const]) for pkg, const in named.findall(text) if const in constants
        )
        for package, filename in pairs:
            rel = Path("src") / Path(*package.split(".")) / filename
            rows.append(
                {
                    "source": path.relative_to(repo).as_posix(),
                    "package": package,
                    "file": filename,
                    "repo_path": rel.as_posix(),
                    "exists": (repo / rel).is_file(),
                    "in_wheel_config": wheel_covers(rel.as_posix(), load_pyproject(repo)),
                }
            )
    return rows


def launcher_import_probe_source() -> str:
    heavy = json.dumps(list(HEAVY_MODULES))
    return f"""
import contextlib, io, json, sys
from dipcatcher_cli.app import main
heavy = {heavy}
rows = []
for args in (["--help"], ["--version"]):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            main(args)
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
    names = set(sys.modules)
    rows.append({{
        "args": args,
        "exit": code,
        "heavy": [name for name in heavy if name in names],
        "interactive": any(
            name == "fx1.interactive" or name.startswith("fx1.interactive.") for name in names
        ),
        "stdout_has_usage": "Usage:" in out.getvalue(),
        "stdout": out.getvalue().strip(),
    }})
print(json.dumps(rows))
"""


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _script(venv: Path, name: str) -> Path:
    folder = venv / ("Scripts" if sys.platform == "win32" else "bin")
    if sys.platform == "win32":
        exe = folder / f"{name}.exe"
        return exe if exe.is_file() else folder / name
    return folder / name


def _tool_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    return env


def _run(
    args: Sequence[str], *, cwd: Path | None, timeout: float, env: Mapping[str, str] | None
) -> CmdCapture:
    try:
        proc = subprocess.run(
            list(args),
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CmdCapture(None, _text(exc.stdout), _text(exc.stderr), "timeout")
    return CmdCapture(proc.returncode, proc.stdout, proc.stderr, None)


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _checked(args: Sequence[str], *, cwd: Path, timeout: float) -> None:
    proc = _run(args, cwd=cwd, timeout=timeout, env=_tool_env())
    if proc.exit_code != 0:
        detail = (proc.stderr or proc.stdout)[-800:]
        raise BuildBackendError(f"{args[0]} exited {proc.exit_code}: {detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_wheel(uv: str, repo: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _checked([uv, "build", "--wheel", "--out-dir", str(out_dir), str(repo)], cwd=repo, timeout=180)
    wheels = sorted(out_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise BuildBackendError(f"expected one wheel in {out_dir}, found {wheels}")
    return wheels[0]


def _site_packages(python: Path) -> Path:
    proc = _run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        cwd=None,
        timeout=30,
        env=_tool_env(),
    )
    if proc.exit_code != 0 or not proc.stdout.strip():
        raise BuildBackendError(proc.stderr[-400:] or "could not resolve site-packages")
    return Path(proc.stdout.strip())


def install_wheel(uv: str, base_python: Path, venv: Path, wheel: Path) -> Path:
    _checked([uv, "venv", str(venv), "--python", str(base_python)], cwd=venv.parent, timeout=120)
    python = _venv_python(venv)
    _checked(
        [uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        cwd=venv.parent,
        timeout=180,
    )
    workspace = _site_packages(base_python)
    bridge = _site_packages(python) / "_workspace_deps.pth"
    bridge.write_text(f"import site; site.addsitedir({str(workspace)!r})\n", encoding="utf-8")
    return python


def scrubbed_env(venv: Path, home: Path, config: Path) -> dict[str, str]:
    bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    system = os.environ.get("PATH", "/usr/bin:/bin")
    session = config / "sessions"
    session.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": os.pathsep.join((str(bin_dir), system)),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "FX1_CONFIG_DIR": str(config),
        "FX1_SESSION_DIR": str(session),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _clip(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:180] + "\n...\n" + text[-280:]


def _argv(spec: Expect, python: Path, venv: Path, inputs: Mapping[str, str]) -> list[str]:
    rendered = tuple(part.format(**inputs) for part in spec.args)
    if spec.kind == "script":
        return [str(_script(venv, rendered[0])), *rendered[1:]]
    if spec.kind == "module":
        return [str(python), "-m", *rendered]
    return [str(python), *rendered]


def _probe_dict(spec: Expect, capture: CmdCapture) -> dict[str, Any]:
    stdout_ok = all(item in capture.stdout for item in spec.stdout_has)
    stderr_ok = all(item in capture.stderr for item in spec.stderr_has)
    stderr_clean = all(item not in capture.stderr for item in spec.stderr_lacks)
    matched = (
        capture.error is None
        and capture.exit_code == spec.expect_exit
        and stdout_ok
        and stderr_ok
        and stderr_clean
    )
    return {
        "name": spec.name,
        "classification": spec.classification,
        "defect": spec.defect,
        "expected_exit": spec.expect_exit,
        "exit_code": capture.exit_code,
        "matched": matched,
        "error": capture.error,
        "stdout_head": _clip(capture.stdout),
        "stderr_head": _clip(capture.stderr),
    }


def run_generic(
    specs: Sequence[Expect],
    *,
    python: Path,
    venv: Path,
    cwd: Path,
    env: Mapping[str, str],
    inputs: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        capture = _run(_argv(spec, python, venv, inputs), cwd=cwd, timeout=spec.timeout_s, env=env)
        rows.append(_probe_dict(spec, capture))
    return rows


def _wheel_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _entry_points(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        name = next(
            item for item in archive.namelist() if item.endswith(".dist-info/entry_points.txt")
        )
        text = archive.read(name).decode("utf-8")
    found: dict[str, str] = {}
    in_scripts = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_scripts = line == "[console_scripts]"
            continue
        if in_scripts and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            found[key] = value
    return found


def _bundled_ok(names: set[str]) -> bool:
    leaked = [
        item for item in names if item.startswith("corpus_epoch") or "arch_boundaries.toml" in item
    ]
    bundled = [item for item in BUNDLED_WHEEL if item in names]
    return bundled == list(BUNDLED_WHEEL) and leaked == []


def inspect_wheel(wheel: Path, scripts: Mapping[str, str]) -> dict[str, Any]:
    names = _wheel_names(wheel)
    present = [item for item in WHEEL_MEMBERS if item in names]
    bundled = [item for item in BUNDLED_WHEEL if item in names]
    points = _entry_points(wheel)
    return {
        "path": str(wheel),
        "sha256": _sha256(wheel),
        "present": present,
        "present_ok": present == list(WHEEL_MEMBERS),
        "bundled": bundled,
        "bundled_ok": _bundled_ok(names),
        "entry_points": points,
        "entry_points_match": points == dict(scripts),
    }


def _shadow_imports(python: Path) -> dict[str, str]:
    code = (
        "import dipcatcher_cli, fx1, quant_fund\n"
        "print(fx1.__file__)\n"
        "print(quant_fund.__file__)\n"
        "print(dipcatcher_cli.__file__)\n"
        "print(fx1.__version__)\n"
    )
    capture = _run([str(python), "-c", code], cwd=None, timeout=30, env=_tool_env())
    if capture.exit_code != 0:
        raise BuildBackendError(capture.stderr[-500:] or "shadow import failed")
    fx1_file, quant_file, cli_file, version = capture.stdout.splitlines()[:4]
    return {
        "fx1": fx1_file,
        "quant_fund": quant_file,
        "dipcatcher_cli": cli_file,
        "version": version,
    }


def _under_site(paths: Mapping[str, str], site: Path) -> bool:
    return all(
        Path(paths[name]).resolve().is_relative_to(site)
        for name in ("fx1", "quant_fund", "dipcatcher_cli")
    )


def _import_probe(python: Path, cwd: Path, env: Mapping[str, str]) -> dict[str, Any]:
    capture = _run(
        [str(python), "-c", launcher_import_probe_source()],
        cwd=cwd,
        timeout=60,
        env=env,
    )
    matched = False
    rows: list[dict[str, Any]] = []
    if capture.exit_code == 0 and capture.stdout.strip():
        rows = cast(list[dict[str, Any]], json.loads(capture.stdout))
        help_row = next(item for item in rows if item["args"] == ["--help"])
        version_row = next(item for item in rows if item["args"] == ["--version"])
        matched = (
            help_row["exit"] == 0
            and help_row["heavy"] == []
            and help_row["interactive"] is False
            and version_row["exit"] == 0
            and version_row["heavy"] == []
            and version_row["interactive"] is False
            and version_row["stdout"] != ""
            and "Usage:" not in version_row["stdout"]
        )
    return {
        "name": "import weight --help/--version",
        "classification": "pass",
        "defect": None,
        "expected_exit": 0,
        "exit_code": capture.exit_code,
        "matched": matched,
        "error": capture.error,
        "stdout_head": _clip(capture.stdout),
        "stderr_head": _clip(capture.stderr),
        "rows": rows,
    }


def _doctor_with_configs(
    python: Path, env: Mapping[str, str], repo: Path, work: Path
) -> dict[str, Any]:
    cwd = work / "with-configs"
    (cwd / "configs").mkdir(parents=True)
    for name in ("base.yaml", "research.yaml"):
        shutil.copyfile(repo / "configs" / name, cwd / "configs" / name)
    capture = _run([str(python), "-m", "dipcatcher_cli", "doctor"], cwd=cwd, timeout=60, env=env)
    created = [
        rel
        for rel in (
            "data/raw",
            "data/bronze",
            "data/silver",
            "data/gold",
            "data/metadata",
            "data/metadata/resolved_config.json",
        )
        if (cwd / rel).exists()
    ]
    needles = ("data_manifest: missing", "research_receipt: missing", "dir_raw: missing")
    matched = (
        capture.error is None
        and capture.exit_code == 1
        and all(item in capture.stdout for item in needles)
        and created == []
        and "Traceback" not in capture.stderr
    )
    return {
        "name": "doctor with configs and no lake",
        "classification": "pass",
        "defect": None,
        "expected_exit": 1,
        "exit_code": capture.exit_code,
        "matched": matched,
        "error": capture.error,
        "stdout_head": _clip(capture.stdout),
        "stderr_head": _clip(capture.stderr),
        "created": created,
    }


INSTALLED_FACTS = """
import json, os
from pathlib import Path
from fx1.harness import Harness
from fx1.interactive.procs import scrub_env
from fx1.interactive.system_prompt import _rules_path
from importlib.resources import files
from quant_fund.cli.capability_cmds import find_repo_root
import dipcatcher_cli
import fx1
import quant_fund

site = Path(os.environ["DIPCATCHER_SMOKE_SITE"])
rules = _rules_path().resolve()
schema = files("quant_fund.research").joinpath("receipt_v2.schema.json")
catalog = files("quant_fund.research").joinpath("research100.json")
scrubbed = scrub_env({
    "PATH": "/usr/bin",
    "HOME": "/tmp/unused",
    "FX1_CONFIG_DIR": "/tmp/unused",
    "MOONSHOT_API_KEY": "installed-smoke-canary",
    "AWS_SECRET_ACCESS_KEY": "installed-smoke-canary",
})
secret_blob = json.dumps(scrubbed)
harness = Harness()
rows = []
for name, extra in (
    ("families", None),
    ("capabilities", ["list", "--json", "--executable-only"]),
    ("verify-receipt", [os.environ["DIPCATCHER_SMOKE_RECEIPT"]]),
    ("score-forecasts", ["--inspect", "--task", "volatility", "--data", os.environ["DIPCATCHER_SMOKE_CSV"]]),
    ("doctor", None),
):
    result = harness.run(name, extra)
    stderr = result.stderr or ""
    combined = (result.stdout or "") + stderr
    needles = {
        "families": "dipcatcher.family_surface.v1",
        "capabilities": "dipcatcher.capabilities.v1",
        "verify-receipt": '"valid": true',
        "score-forecasts": "dataset_profile.v1",
        "doctor": "configs/research.yaml",
    }
    rows.append({
        "name": "harness " + name,
        "exit_code": result.exit_code,
        "stdout_head": (result.stdout or "")[:240],
        "stderr_tail": stderr[-300:],
        "needle_ok": needles[name] in combined,
        "clean_doctor": (
            result.exit_code == 1
            and "data_manifest: missing" in (result.stdout or "")
            and "Traceback" not in stderr
            and "FileNotFoundError" not in stderr
        ),
    })
contained = {"raised": False, "detail": ""}
try:
    harness.run("families", config=Path(os.environ["DIPCATCHER_SMOKE_OUTSIDE"]))
except ValueError as exc:
    contained = {"raised": True, "detail": str(exc)[:300]}
print(json.dumps({
    "fx1": fx1.__file__,
    "quant_fund": quant_fund.__file__,
    "dipcatcher_cli": dipcatcher_cli.__file__,
    "under_site": all(
        Path(path).resolve().is_relative_to(site)
        for path in (fx1.__file__, quant_fund.__file__, dipcatcher_cli.__file__)
    ),
    "rules": str(rules),
    "rules_ok": rules.is_file() and rules.is_relative_to(site) and "Hard rules" in rules.read_text(encoding="utf-8"),
    "schema_ok": schema.is_file(),
    "catalog_ok": catalog.is_file(),
    "repo_root": None if find_repo_root() is None else str(find_repo_root()),
    "secret_scrubbed": "installed-smoke-canary" not in secret_blob,
    "probes": rows,
    "containment": contained,
}))
"""


def _fact_probes(blob: Mapping[str, Any]) -> list[dict[str, Any]]:
    probes = cast(list[dict[str, Any]], blob["probes"])
    by_name = {str(item["name"]): item for item in probes}
    rows: list[dict[str, Any]] = []
    for name in (
        "harness families",
        "harness capabilities",
        "harness verify-receipt",
        "harness score-forecasts",
    ):
        item = by_name[name]
        matched = item["exit_code"] == 0 and bool(item["needle_ok"])
        rows.append(_fact_row(name, "pass", None, 0, matched, item))
    doctor = by_name["harness doctor"]
    doctor_ok = bool(doctor["clean_doctor"])
    rows.append(_fact_row("harness doctor", "pass", None, 1, doctor_ok, doctor))
    containment = cast(Mapping[str, Any], blob["containment"])
    contained_ok = bool(containment["raised"]) and "escapes the configs/ allowlist" in str(
        containment["detail"]
    )
    rows.append(
        {
            "name": "harness config containment",
            "classification": "pass",
            "defect": None,
            "expected_exit": None,
            "exit_code": None,
            "matched": contained_ok,
            "error": None,
            "stdout_head": str(containment["detail"]),
            "stderr_head": "",
        }
    )
    resource_ok = bool(blob["rules_ok"]) and bool(blob["schema_ok"]) and bool(blob["catalog_ok"])
    rows.append(
        {
            "name": "packaged prompt and schemas",
            "classification": "pass",
            "defect": None,
            "expected_exit": None,
            "exit_code": None,
            "matched": resource_ok and blob["repo_root"] is None and bool(blob["secret_scrubbed"]),
            "error": None,
            "stdout_head": json.dumps(
                {
                    "rules_ok": blob["rules_ok"],
                    "schema_ok": blob["schema_ok"],
                    "catalog_ok": blob["catalog_ok"],
                    "repo_root": blob["repo_root"],
                    "secret_scrubbed": blob["secret_scrubbed"],
                }
            ),
            "stderr_head": "",
        }
    )
    return rows


def _fact_row(
    name: str,
    classification: str,
    defect: str | None,
    expected: int,
    matched: bool,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "classification": classification,
        "defect": defect,
        "expected_exit": expected,
        "exit_code": item["exit_code"],
        "matched": matched,
        "error": None,
        "stdout_head": str(item["stdout_head"]),
        "stderr_head": str(item["stderr_tail"]),
    }


def run_installed_facts(
    python: Path, cwd: Path, env: dict[str, str], site: Path, inputs: Mapping[str, str]
) -> list[dict[str, Any]]:
    child_env = dict(env)
    child_env["DIPCATCHER_SMOKE_SITE"] = str(site)
    child_env["DIPCATCHER_SMOKE_RECEIPT"] = inputs["receipt"]
    child_env["DIPCATCHER_SMOKE_CSV"] = inputs["csv"]
    child_env["DIPCATCHER_SMOKE_OUTSIDE"] = inputs["outside"]
    script = cwd.parent / "installed_facts.py"
    script.write_text(INSTALLED_FACTS, encoding="utf-8")
    capture = _run([str(python), str(script)], cwd=cwd, timeout=120, env=child_env)
    if capture.exit_code != 0 or not capture.stdout.strip():
        return [
            {
                "name": "installed facts",
                "classification": "d",
                "defect": None,
                "expected_exit": 0,
                "exit_code": capture.exit_code,
                "matched": False,
                "error": capture.error or "facts command failed",
                "stdout_head": _clip(capture.stdout),
                "stderr_head": _clip(capture.stderr),
            }
        ]
    return _fact_probes(cast(Mapping[str, Any], json.loads(capture.stdout)))


def _defects(probes: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for probe in probes:
        ident = probe.get("defect")
        if not isinstance(ident, str) or ident in seen or not probe.get("matched"):
            continue
        seen.add(ident)
        found.append({"id": ident, **DEFECTS[ident]})
    return found


def _static_probe(
    name: str, classification: str, defect: str | None, matched: bool, detail: str
) -> dict[str, Any]:
    return {
        "name": name,
        "classification": classification,
        "defect": defect,
        "expected_exit": None,
        "exit_code": None,
        "matched": matched,
        "error": None,
        "stdout_head": detail,
        "stderr_head": "",
    }


def _wheel_probes(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _static_probe(
            "wheel ships package data",
            "pass",
            None,
            bool(info["present_ok"]),
            json.dumps(info["present"]),
        ),
        _static_probe(
            "wheel entry points",
            "pass",
            None,
            bool(info["entry_points_match"]),
            json.dumps(info["entry_points"]),
        ),
        _static_probe(
            "wheel ships bundled configs",
            "pass",
            None,
            bool(info.get("bundled_ok")),
            json.dumps(info.get("bundled", [])),
        ),
    ]


def _prepare_inputs(repo: Path, work: Path) -> dict[str, str]:
    inputs = work / "inputs"
    inputs.mkdir(parents=True)
    receipt = inputs / "receipt.json"
    shutil.copyfile(repo / RECEIPT_RELATIVE, receipt)
    csv_path = inputs / "tiny.csv"
    csv_path.write_text(
        "timestamp,realized,forecast_var\n"
        "2024-01-01T00:00:00Z,0.1,0.2\n"
        "2024-01-02T00:00:00Z,0.2,0.2\n",
        encoding="utf-8",
    )
    outside = inputs / "outside.yaml"
    outside.write_text("runtime: {}\n", encoding="utf-8")
    missing = inputs / "missing.yaml"
    return {
        "csv": str(csv_path),
        "receipt": str(receipt),
        "outside": str(outside),
        "missing": str(missing),
    }


def _base_python(repo: Path) -> Path:
    folder = repo / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    name = "python.exe" if sys.platform == "win32" else "python"
    candidate = folder / name
    if candidate.is_file():
        return candidate
    return Path(sys.executable)


def execute(repo: Path, wheel_arg: Path | None, work: Path) -> dict[str, Any]:
    uv = shutil.which("uv")
    if uv is None:
        raise BuildBackendError("uv is not on PATH")
    project = load_pyproject(repo)
    scripts = cast(Mapping[str, str], project["project"]["scripts"])
    wheel = wheel_arg if wheel_arg is not None else build_wheel(uv, repo, work / "wheel")
    base = _base_python(repo)
    before = _shadow_imports(base)
    venv = work / "venv"
    python = install_wheel(uv, base, venv, wheel)
    after = _shadow_imports(base)
    if before["fx1"] != after["fx1"]:
        raise BuildBackendError("install changed the checkout fx1 import")
    site = _site_packages(python)
    installed = _shadow_imports(python)
    info = inspect_wheel(wheel, scripts)
    empty = work / "empty"
    empty.mkdir()
    env = scrubbed_env(venv, work / "home", work / "fx1-config")
    inputs = _prepare_inputs(repo, work)
    probes = _wheel_probes(info)
    probes.extend(run_generic(GENERIC, python=python, venv=venv, cwd=empty, env=env, inputs=inputs))
    probes.append(_import_probe(python, empty, env))
    probes.extend(run_installed_facts(python, empty, env, site, inputs))
    probes.append(_doctor_with_configs(python, env, repo, work))
    version = installed["version"]
    _lock_versions(probes, version)
    defects = _defects(probes)
    shadowed = _under_site(installed, site)
    untouched = not any(empty.iterdir())
    ok = all(bool(probe["matched"]) for probe in probes) and shadowed and untouched
    return {
        "ok": ok,
        "version": version,
        "checkout_fx1": before["fx1"],
        "installed": installed,
        "shadows_checkout": shadowed,
        "empty_cwd_untouched": untouched,
        "wheel": info,
        "layout": layout_gaps(repo),
        "probes": probes,
        "defects": defects,
    }


def _lock_versions(probes: list[dict[str, Any]], version: str) -> None:
    wanted = {
        "fxi --version": f"fxi {version}",
        "fx1 --version": f"fx1 {version}",
        "quant --version": version,
        "dipcatcher --version": version,
        "python -m dipcatcher_cli --version": version,
    }
    for probe in probes:
        needle = wanted.get(str(probe["name"]))
        if needle is None:
            continue
        head = str(probe["stdout_head"])
        probe["matched"] = bool(probe["matched"]) and needle in head


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test dipcatcher away from the checkout.")
    parser.add_argument("--repo", type=Path, default=repo_root_from_here())
    parser.add_argument("--wheel", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(prefix="dipcatcher-install-smoke-") as raw:
            report = execute(Path(args.repo), args.wheel, Path(raw))
    except BuildBackendError as exc:
        payload = {"ok": False, "error": "build-backend-unavailable", "detail": str(exc)}
        _emit(payload, args.report)
        return 2
    _emit(report, args.report)
    return 0 if report["ok"] else 1


def _emit(payload: Mapping[str, Any], report: Path | None) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    sys.stdout.write(text)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
