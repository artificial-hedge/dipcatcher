"""Front-door truthfulness helpers — receipt census and env documentation parity."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from dipcatcher_cli.paths import repo_root

_ENV_LINE = re.compile(r"^([A-Z][A-Z0-9_]+)\s*=")
_ENV_MENTION = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
_TEST_ONLY_SECTION = "Test / audit-harness internals"


def receipt_census(root: Path | None = None) -> dict[str, int]:
    checkout = root or repo_root()
    proc = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "receipts/"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    json_files = [p for p in lines if p.endswith(".json")]
    top_level = [p for p in json_files if p.count("/") == 1]
    legacy_unsealed = [p for p in json_files if p.startswith("receipts/legacy-unsealed/")]
    readme = sum(1 for p in lines if p.endswith("README.md"))
    return {
        "json_total": len(json_files),
        "json_top_level": len(top_level),
        "legacy_unsealed_json": len(legacy_unsealed),
        "readme_files": readme,
        "tracked_under_receipts": len(lines),
    }


def env_example_variables(path: Path | None = None) -> frozenset[str]:
    example = path or (repo_root() / ".env.example")
    names: set[str] = set()
    in_test_section = False
    for line in example.read_text(encoding="utf-8").splitlines():
        if _TEST_ONLY_SECTION in line:
            in_test_section = True
        if in_test_section:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        body = stripped.lstrip("#").strip() if stripped.startswith("#") else stripped
        match = _ENV_LINE.match(body)
        if match:
            names.add(match.group(1))
    return frozenset(names)


def env_example_all_variables(path: Path | None = None) -> frozenset[str]:
    """Every ``NAME=`` in ``.env.example``, including the test-only section."""
    example = path or (repo_root() / ".env.example")
    names: set[str] = set()
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        body = stripped.lstrip("#").strip() if stripped.startswith("#") else stripped
        match = _ENV_LINE.match(body)
        if match:
            names.add(match.group(1))
    return frozenset(names)


RUNTIME_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "SHELL",
        "PWD",
        "OLDPWD",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "COLORTERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "CLICOLOR",
        "CLICOLOR_FORCE",
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_WORKSPACE",
        "GITHUB_OUTPUT",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_STEP_SUMMARY",
        "RUNNER_OS",
        "RUNNER_TEMP",
        "RUNNER_TOOL_CACHE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "WINDIR",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "UV_PROJECT_ENVIRONMENT",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUNBUFFERED",
        "PYTHONHASHSEED",
        "PYTEST_CURRENT_TEST",
        "PYTEST_ADDOPTS",
        "COVERAGE_FILE",
        "COLUMNS",
        "LINES",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
        "SSH_AUTH_SOCK",
        "EDITOR",
        "VISUAL",
        "PAGER",
        "HOSTNAME",
        "HOST",
        "LOGNAME",
        "OS",
        "SHLVL",
        "TERM_PROGRAM",
        "CURSOR_AGENT",
        "FXI_ORBS",
        "COLORFGBG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    }
)


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def source_environ_names(root: Path | None = None) -> frozenset[str]:
    """AST-scan ``src/`` for ``os.environ`` / ``os.getenv`` constant keys."""
    checkout = root or repo_root()
    names: set[str] = set()
    src = checkout / "src"
    for path in src.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                attr = func.attr if isinstance(func, ast.Attribute) else None
                name = func.id if isinstance(func, ast.Name) else None
                key = _const_str(node.args[0]) if node.args else None
                if not key or not key.isupper() or not key.replace("_", "").isalnum():
                    continue
                os_getenv = name == "getenv"
                environ_get = (
                    attr == "get"
                    and isinstance(func, ast.Attribute)
                    and (
                        (isinstance(func.value, ast.Name) and func.value.id in {"os", "environ"})
                        or (
                            isinstance(func.value, ast.Attribute)
                            and isinstance(func.value.value, ast.Name)
                            and func.value.value.id == "os"
                            and func.value.attr == "environ"
                        )
                    )
                )
                if os_getenv or environ_get:
                    names.add(key)
            elif isinstance(node, ast.Subscript):
                key = _const_str(node.slice)
                if not key or not key.isupper():
                    continue
                target = node.value
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "os"
                    and target.attr == "environ"
                ):
                    names.add(key)
                elif isinstance(target, ast.Name) and target.id == "environ":
                    names.add(key)
    return frozenset(names)


def env_documentation_surfaces(root: Path | None = None) -> frozenset[str]:
    checkout = root or repo_root()
    surfaces = (checkout / ".env.example", checkout / "README.md", checkout / "AGENTS.md")
    mentioned: set[str] = set()
    for path in surfaces:
        if not path.is_file():
            continue
        for token in _ENV_MENTION.findall(path.read_text(encoding="utf-8")):
            mentioned.add(token)
    return frozenset(mentioned)
