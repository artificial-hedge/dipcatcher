"""Fail collection when a formal or property test module yields no tests.

Hypothesis 6.168 runs ``run_state_machine_as_test`` at import and returns
``None``. A module that only does that collects zero items and drops the
machines from the suite. ``tests/formal/conftest.py`` and
``tests/property/conftest.py`` call these hooks. The root ``tests/conftest.py``
stays untouched.

Both directory conftests can be loaded in one session. Noting a module and
counting it toward ``testsfailed`` happen once.
"""

from __future__ import annotations

from typing import Any

_ROOTS = ("tests/formal/", "tests/property/")
_SEEN: set[str] = set()
_EMPTY: list[str] = []
_COUNTED = False


def zero_collected_module(nodeid: str, outcome: str, result_count: int) -> str | None:
    """Return an error when a watched test module collected nothing.

    Non-test helpers, failed imports, and modules outside ``tests/formal``
    and ``tests/property`` are left alone. ``result_count`` is the number of
    collectors the module produced, before ``-k`` deselection.
    """
    if outcome != "passed" or result_count:
        return None
    path = nodeid.replace("\\", "/")
    if not path.endswith(".py"):
        return None
    name = path.rsplit("/", 1)[-1]
    if not name.startswith("test_"):
        return None
    if path.startswith(_ROOTS):
        return f"{path} collected 0 tests"
    return None


def reset_collection() -> None:
    """Drop notes from an earlier in-process session."""
    global _COUNTED
    _SEEN.clear()
    _EMPTY.clear()
    _COUNTED = False


def note_collect_report(report: Any) -> None:
    """Mark an empty formal or property module as a collection failure."""
    result = report.result or []
    message = zero_collected_module(report.nodeid, report.outcome, len(result))
    if message is None or message in _SEEN:
        return
    _SEEN.add(message)
    _EMPTY.append(message)
    report.outcome = "failed"
    report.longrepr = message


def finish_collection(session: Any) -> None:
    """Count each empty module once.

    The session hook can read the report before this one marks it failed, so
    ``testsfailed`` is what makes ``--collect-only`` exit non-zero.
    """
    global _COUNTED
    if _COUNTED or not _EMPTY:
        return
    _COUNTED = True
    session.testsfailed += len(_EMPTY)
