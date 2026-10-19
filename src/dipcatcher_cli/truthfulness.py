"""Front-door truthfulness helpers — receipt census and env documentation parity."""

from __future__ import annotations

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
