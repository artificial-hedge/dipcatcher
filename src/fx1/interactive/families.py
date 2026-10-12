"""fx-1 client for the read-only family surface.

The lab is reached only through an injected runner
``(argv: list[str]) -> (exit_code, stdout, stderr)`` matching harness stdout.
This module does not import the lab and does not spawn a process. A response
that is not the qualified executable contract raises :class:`FamilyClientError`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any, cast


class FamilyClientError(ValueError):
    """The family surface response broke the read-only contract."""


SURFACE_SCHEMA = "dipcatcher.family_surface.v1"

_FAMILY_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STDERR_LIMIT = 500
_MAX_GOAL = 4096
_MAX_FAMILIES = 23
_MISSING = object()

Runner = Callable[[list[str]], tuple[int, str, str]]


def _stderr_snippet(stderr: str) -> str:
    if len(stderr) <= _STDERR_LIMIT:
        return stderr
    return stderr[:_STDERR_LIMIT]


def _parse_json(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FamilyClientError(f"family surface returned invalid JSON: {exc}") from exc
    except (ValueError, RecursionError, OverflowError) as exc:
        raise FamilyClientError("family surface returned invalid JSON") from exc


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FamilyClientError("family surface payload must be a JSON object")
    schema = payload.get("schema")
    if schema != SURFACE_SCHEMA:
        raise FamilyClientError(
            f"family surface schema must be {SURFACE_SCHEMA!r} (got {schema!r})"
        )
    return payload


def _require_executable_required(item: Any, *, where: str) -> None:
    """Reject a record the harness should already have filtered out."""
    if not isinstance(item, dict):
        raise FamilyClientError(f"{where} must be an object")
    _reject_item_schema(item, where)
    if item.get("executable") is not True:
        raise FamilyClientError(f"{where} must have executable true")
    qualification = item.get("qualification")
    if qualification != "required":
        raise FamilyClientError(f"{where} qualification must be 'required' (got {qualification!r})")
    family_id = item.get("id")
    if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
        raise FamilyClientError(f"{where} id must match {_FAMILY_ID.pattern}")


def _reject_item_schema(item: Mapping[str, Any], where: str) -> None:
    if "schema" not in item:
        return
    schema = item.get("schema")
    if schema != SURFACE_SCHEMA:
        raise FamilyClientError(f"{where} schema must be {SURFACE_SCHEMA!r} (got {schema!r})")


def _require_select_shape(payload: dict[str, Any]) -> None:
    missing = [key for key in ("schema", "selected", "exclusions") if key not in payload]
    if missing:
        raise FamilyClientError(f"select payload missing keys: {', '.join(missing)}")
    selected = payload["selected"]
    if not isinstance(selected, list):
        raise FamilyClientError("select payload 'selected' must be a list")
    _require_exclusions(payload["exclusions"])
    for index, item in enumerate(selected):
        _require_executable_required(item, where=f"selected[{index}]")


def _require_exclusions(exclusions: Any) -> None:
    if not isinstance(exclusions, list):
        raise FamilyClientError("select payload 'exclusions' must be a list")
    for index, item in enumerate(exclusions):
        _require_exclusion(item, index)


def _require_exclusion(item: Any, index: int) -> None:
    if not isinstance(item, dict):
        raise FamilyClientError(f"exclusions[{index}] must be an object")
    family_id = item.get("id")
    reason = item.get("reason")
    if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
        raise FamilyClientError(f"exclusions[{index}] id is invalid")
    if not isinstance(reason, str) or not reason.strip():
        raise FamilyClientError(f"exclusions[{index}] reason must be non-empty text")


def _reject_present_schema(payload: Any) -> None:
    """A missing schema stays tolerated; a present wrong schema does not."""
    if not isinstance(payload, dict) or "schema" not in payload:
        return
    schema = payload.get("schema")
    if schema != SURFACE_SCHEMA:
        raise FamilyClientError(
            f"family surface schema must be {SURFACE_SCHEMA!r} (got {schema!r})"
        )


def _preview(text: object) -> str:
    if not isinstance(text, str):
        return repr(type(text).__name__)
    if len(text) <= 80:
        return repr(text)
    return repr(f"{text[:32]}...({len(text)} chars)")


def _require_goal_arg(goal: str) -> None:
    if not isinstance(goal, str) or "\x00" in goal or not goal.strip():
        raise FamilyClientError("goal must be non-empty text without NUL")
    if len(goal) > _MAX_GOAL:
        raise FamilyClientError(f"goal exceeds {_MAX_GOAL} characters")


def _require_max_arg(max_families: int) -> None:
    if isinstance(max_families, bool) or not isinstance(max_families, int):
        raise FamilyClientError("max_families must be an integer")
    if max_families < 0 or max_families > _MAX_FAMILIES:
        raise FamilyClientError(f"max_families must be an integer in 0..{_MAX_FAMILIES}")


def _require_family_arg(family_id: str) -> None:
    if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
        shown = _preview(family_id)
        pattern = _FAMILY_ID.pattern
        raise FamilyClientError(f"family id must match {pattern} (got {shown})")


def _family_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        families = payload.get("families")
        if isinstance(families, list):
            return families
    raise FamilyClientError("list payload must be a list or an object with a families list")


def _runner_triple(result: object) -> tuple[int, str, str]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise FamilyClientError("family runner must return (exit_code, stdout, stderr)")
    code: object = result[0]
    stdout: object = result[1]
    stderr: object = result[2]
    if isinstance(code, bool) or not isinstance(code, int):
        raise FamilyClientError("family runner exit_code must be an int")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise FamilyClientError("family runner stdout and stderr must be strings")
    return code, stdout, stderr


class FamilyClient:
    """Call the read-only family surface through an injected runner."""

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    def select(self, goal: str, *, max_families: int = 4) -> dict[str, Any]:
        """Choose qualified families for ``goal``.

        The parsed object is returned unchanged when it matches the contract.
        A selected record that is not executable and required is a contract
        break: it is rejected, not dropped.
        """
        _require_goal_arg(goal)
        _require_max_arg(max_families)
        argv = ["--select", goal, "--max", str(max_families)]
        payload = _require_object(self._payload(argv))
        _require_select_shape(payload)
        return payload

    def status(self, family_id: str) -> dict[str, Any]:
        """Return the surface record for one family id."""
        _require_family_arg(family_id)
        payload = _require_object(self._payload(["--id", family_id]))
        got = payload.get("id")
        if "id" not in payload or not isinstance(got, str):
            raise FamilyClientError("status payload must contain schema and id")
        if _FAMILY_ID.fullmatch(got) is None or got != family_id:
            raise FamilyClientError(f"status id {got!r} does not match requested {family_id!r}")
        return payload

    def list_executable(self) -> list[dict[str, Any]]:
        """Return families the surface marks executable and required."""
        payload = self._payload([])
        _reject_present_schema(payload)
        items = _family_items(payload)
        for index, item in enumerate(items):
            _require_executable_required(item, where=f"families[{index}]")
        return cast(list[dict[str, Any]], items)

    def _payload(self, argv: list[str]) -> Any:
        _code, stdout, _stderr = self._run(argv)
        return _parse_json(stdout)

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        code, stdout, stderr = _runner_triple(self._runner(argv))
        if code != 0:
            raise FamilyClientError(f"family surface exited {code}: {_stderr_snippet(stderr)}")
        return code, stdout, stderr


def selected_ids(payload: dict[str, Any]) -> list[str]:
    """Ids from ``payload['selected']`` in order, keeping the first of each id."""
    if not isinstance(payload, dict):
        raise FamilyClientError("selected_ids payload must be an object")
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise FamilyClientError("selected_ids payload must contain a selected list")
    ids: list[str] = []
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise FamilyClientError(f"selected[{index}] must be an object")
        family_id = item.get("id")
        if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
            raise FamilyClientError(f"selected[{index}] id is invalid")
        ids.append(family_id)
    return list(dict.fromkeys(ids))


def harness_family_runner(harness: Any) -> Runner:
    """Return a runner that calls ``harness.run('families', extra_args)``."""

    def runner(argv: list[str]) -> tuple[int, str, str]:
        result = harness.run("families", argv)
        code = getattr(result, "exit_code", _MISSING)
        if code is _MISSING:
            raise FamilyClientError("harness families result has no exit_code")
        stdout = getattr(result, "stdout", _MISSING)
        stderr = getattr(result, "stderr", _MISSING)
        if isinstance(code, bool) or not isinstance(code, int):
            raise FamilyClientError("harness families exit_code must be an int")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise FamilyClientError("harness families stdout and stderr must be strings")
        return code, stdout, stderr

    return runner
