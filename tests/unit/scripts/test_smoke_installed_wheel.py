"""The release smoke covers the wheel's declared public command surface."""

from __future__ import annotations

import tomllib
from pathlib import Path

from scripts.smoke_installed_wheel import _SCRIPTS

_ROOT = Path(__file__).resolve().parents[3]


def test_installed_wheel_smoke_covers_every_console_script() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(project["project"]["scripts"])
    exercised = {name for name, _, _ in _SCRIPTS}
    assert exercised == declared
    assert (
        "dipcatcher",
        ("superpower", "--capabilities"),
        "available /superpower capabilities",
    ) in _SCRIPTS


def test_ci_and_release_smoke_the_installed_wheel() -> None:
    for workflow in ("ci.yml", "release.yml"):
        text = (_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        install = text.index("uv pip install --python .venv/bin/python --no-deps dist/*.whl")
        smoke = text.index(".venv/bin/python scripts/smoke_installed_wheel.py")
        assert install < smoke
