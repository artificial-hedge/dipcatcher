"""Every GitHub Action ``uses:`` is a full commit SHA plus a version comment.

A floating tag (``@v1``) or a SHA with no version comment is unpinned. The
comment is the release tag ``vMAJOR.MINOR.PATCH`` that points at that commit.
``dtolnay/rust-toolchain`` publishes no semver tag on the pinned commit (its
only semver tag, ``v1``, is a later commit), so that action's comment is the
committer date ``YYYY-MM-DD`` instead of a channel name like ``stable``.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_USES_LINE = re.compile(
    r"^(?P<indent> *)(?:- )?uses:\s+(?P<quote>['\"]?)(?P<value>[^\s'\"]+)(?P=quote)"
    r"(?:\s+#\s*(?P<comment>\S+))?\s*$"
)
_ACTION = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)@(?P<sha>[0-9a-f]{40})$"
)
_SEMVER = re.compile(r"^v\d+\.\d+\.\d+$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUST_TOOLCHAIN = "dtolnay/rust-toolchain"


def _workflow_texts() -> list[tuple[str, str]]:
    paths = sorted(_WORKFLOWS.glob("*.yml")) + sorted(_WORKFLOWS.glob("*.yaml"))
    assert paths, "no workflows to pin"
    return [(path.name, path.read_text(encoding="utf-8")) for path in paths]


def _version_comment(action: str, comment: str | None) -> bool:
    if comment is None:
        return False
    if _SEMVER.fullmatch(comment):
        return True
    return action == _RUST_TOOLCHAIN and _DATE.fullmatch(comment) is not None


def test_every_action_use_is_sha_pinned_with_a_version_comment() -> None:
    failures: list[str] = []
    seen = 0
    for name, text in _workflow_texts():
        for number, line in enumerate(text.splitlines(), start=1):
            match = _USES_LINE.match(line)
            if match is None:
                continue
            seen += 1
            value = match.group("value")
            action = _ACTION.fullmatch(value)
            comment = match.group("comment")
            where = f"{name}:{number}: {line.strip()}"
            if action is None:
                failures.append(f"unpinned uses: {where}")
                continue
            if not _version_comment(action.group("name"), comment):
                failures.append(f"missing version comment: {where}")
    assert seen > 0
    assert failures == []
