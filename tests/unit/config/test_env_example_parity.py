"""Literal environment reads in src/ are named in .env.example."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
EXAMPLE = ROOT / ".env.example"

# os.environ.get/pop/setdefault("NAME"), os.getenv("NAME"), os.environ["NAME"].
_READ = re.compile(
    r"""(?:os\.environ\.(?:get|pop|setdefault)|os\.getenv)\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]"""
    r"""|os\.environ\[\s*['\"]([A-Z][A-Z0-9_]+)['\"]"""
)
_EXAMPLE_NAME = re.compile(r"^#?\s*([A-Z][A-Z0-9_]+)=", re.MULTILINE)

# Process and OS variables. They are not operator configuration for this repo.
_OS_VARS = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOCALAPPDATA",
        "LOGNAME",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PYTHONPATH",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "TZ",
        "USER",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
        "GITHUB_REPOSITORY",
    }
)


def _literal_env_reads() -> set[str]:
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _READ.finditer(text):
            names.add(next(group for group in match.groups() if group))
    return names


def test_env_example_names_every_literal_src_read() -> None:
    documented = set(_EXAMPLE_NAME.findall(EXAMPLE.read_text(encoding="utf-8")))
    missing = sorted(_literal_env_reads() - documented - _OS_VARS)
    assert missing == [], f".env.example omits {missing}"
