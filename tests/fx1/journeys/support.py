"""Per-journey workspace: env, stub, files, and help probe."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tests.fx1.journeys.textutil import FX1_KEY, LITE_KEY, substitute

CORRUPT_ID = "deadbeefdeadbeefdeadbeefdeadbeef"
_ISOLATED = (
    "FX1_CONFIG_DIR",
    "FX1_SESSION_DIR",
    "MOONSHOT_API_KEY",
    "FX1_BASE_URL",
    "MOONSHOT_API_URL",
    "DIPCATCHER_FX1_API_KEY",
    "DIPCATCHER_FX1_BASE_URL",
    "DIPCATCHER_FX1_LITE_API_KEY",
    "DIPCATCHER_FX1_LITE_BASE_URL",
    "NO_COLOR",
    "FORCE_COLOR",
)
_HELP: str | None = None


@contextmanager
def isolated_env(config: Path, sessions: Path) -> Iterator[None]:
    """Point fx1 stores at the journey dirs and drop inherited credentials."""
    saved = {key: os.environ.get(key) for key in _ISOLATED}
    os.environ["FX1_CONFIG_DIR"] = str(config)
    os.environ["FX1_SESSION_DIR"] = str(sessions)
    os.environ["NO_COLOR"] = "1"
    for key in _ISOLATED:
        if key in {"FX1_CONFIG_DIR", "FX1_SESSION_DIR", "NO_COLOR"}:
            continue
        os.environ.pop(key, None)
    try:
        yield
    finally:
        _restore_env(saved)


def help_text(scratch: Path) -> str:
    """Live ``/help`` from this process, cached for the run."""
    global _HELP
    if _HELP is None:
        _HELP = _probe_help(scratch)
    return _HELP


def reset_help_cache() -> None:
    """Test hook. Production runs keep the cache for one process."""
    global _HELP
    _HELP = None


def missing_features(requires: list[str], shown: str) -> list[str]:
    """Feature names whose commands are absent from help text."""
    import re

    patterns = {
        "data": re.compile(r"(?m)^[ \t]*/data\b"),
        "runs": re.compile(r"(?m)^[ \t]*/runs\b"),
        "models-check": re.compile(r"/models\b[^\n]*\bcheck\b"),
    }
    return [name for name in requires if patterns[name].search(shown) is None]


def prepare_tree(root: Path) -> dict[str, Path]:
    """Create the journey directories and return their paths."""
    paths = {
        "work": root / "work",
        "config": root / "cfg",
        "sessions": root / "sessions",
        "home": root / "home",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def apply_setup(journey: dict[str, Any], paths: dict[str, Path], stub_url: str) -> None:
    """Write declared user files, a corrupt session, and endpoint presets."""
    setup = _setup(journey)
    _write_files(paths["work"], setup.get("files", []), stub_url)
    if setup.get("corrupt_session"):
        target = paths["sessions"] / f"{CORRUPT_ID}.json"
        target.write_text("{", encoding="utf-8")
    models = setup.get("endpoints", [])
    if models:
        _store_endpoints(list(models), stub_url)


def queue_replies(server: Any, specs: list[dict[str, Any]], *, stub_url: str, work: str) -> None:
    """Append scripted stub replies. ``repeat`` copies are independent objects."""
    for spec in specs:
        repeat = int(spec.get("repeat", 1))
        for _ in range(repeat):
            server.queue(_reply(spec, stub_url=stub_url, work=work))


def needs_stub(journey: dict[str, Any]) -> bool:
    setup = _setup(journey)
    return bool(setup.get("stub") or setup.get("endpoints") or setup.get("replies"))


def _setup(journey: dict[str, Any]) -> dict[str, Any]:
    setup = journey.get("setup", {})
    if not isinstance(setup, dict):
        raise ValueError(f"{journey['id']} setup must be an object")
    return setup


def _write_files(work: Path, specs: object, stub_url: str) -> None:
    if not isinstance(specs, list):
        raise ValueError("setup.files must be a list")
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("setup file must be an object")
        rel = Path(str(spec.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("setup file path escapes the work directory")
        text = substitute(str(spec.get("text", "")), stub_url=stub_url, work=str(work))
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def _store_endpoints(models: list[str], stub_url: str) -> None:
    from fx1.interactive import profiles

    keys = {"fx1": FX1_KEY, "fx1-lite": LITE_KEY}
    for model in models:
        profiles.set_endpoint(model, keys[model], stub_url)


def _reply(spec: dict[str, Any], *, stub_url: str, work: str) -> Any:
    from tests.fx1.stub_model_server import Reply

    if "delay_s" in spec:
        inner = dict(spec)
        inner.pop("delay_s")
        inner.pop("repeat", None)
        return Reply.delay(float(spec["delay_s"]), _reply(inner, stub_url=stub_url, work=work))
    if "status" in spec:
        return _error_reply(spec, stub_url=stub_url, work=work)
    text = substitute(str(spec.get("text", "stub reply")), stub_url=stub_url, work=work)
    served = spec.get("served_model")
    served_model = str(served) if isinstance(served, str) else None
    return Reply.ok(text, served_model=served_model)


def _error_reply(spec: dict[str, Any], *, stub_url: str, work: str) -> Any:
    from tests.fx1.stub_model_server import Reply

    message = substitute(str(spec.get("message", "error")), stub_url=stub_url, work=work)
    body = {
        "error": {
            "message": message,
            "type": str(spec.get("err_type", "server_error")),
            "code": str(spec.get("code", "error")),
        }
    }
    headers: dict[str, str] = {}
    if "retry_after" in spec:
        headers["Retry-After"] = str(spec["retry_after"])
    return Reply.error(int(spec["status"]), headers=headers, body=body)


def _probe_help(scratch: Path) -> str:
    from fx1.interactive.concierge import Concierge

    config = scratch / "probe-cfg"
    sessions = scratch / "probe-sessions"
    config.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with isolated_env(config, sessions):
        concierge = Concierge(
            output_fn=lines.append,
            animated=False,
            input_fn=_no_input,
            session_store=None,
        )
        concierge.handle_line("/help")
    return "\n".join(str(line) for line in lines)


def _no_input(_prompt: str) -> str:
    raise EOFError


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
