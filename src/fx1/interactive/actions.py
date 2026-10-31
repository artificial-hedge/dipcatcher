"""Operations behind ``fxi`` commands — shared by the REPL and the Typer app.

The one rule of this module: fx-1 credentials travel as environment values
on child processes, never as argv. Subprocess calls apply the active model
profile to ``os.environ`` (so the session remembers it) and inherit the
environment.
"""

from __future__ import annotations

import json
import shutil
import socket
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer

from fx1.doctor import collect_status
from fx1.harness import HARNESS_REGISTRY, Harness
from fx1.interactive import profiles
from fx1.interactive.orb import OrbAnimator, OrbState
from fx1.interactive.profiles import DEFAULT_MODEL
from fx1.serve.backends import HostedK3Backend

# Console scripts shipped in the fx-1 distribution that fxi may launch.
TOOLS: tuple[str, ...] = ("fxi", "fx1", "dipcatcher", "quant", "verify-ledger", "mc-engine")

BackendFactory = Callable[..., HostedK3Backend]


def _apply(model: str) -> None:
    """Inject the model's endpoint into os.environ; fail-closed when missing."""
    profiles.apply_profile(model)


def _which(argv0: str) -> str:
    exe = shutil.which(argv0)
    if exe is None:
        raise RuntimeError(f"{argv0!r} not found on PATH; is fx-1 installed?")
    return exe


def run_eval(out: Path | None, model: str = DEFAULT_MODEL) -> int:
    """Run the fx-1 eval bank through the same CLI path as ``make fx1-eval``.

    The child ``fx1 eval`` reads ``MOONSHOT_API_KEY``/``FX1_BASE_URL`` from
    its environment (applied here) and is told which model id to send.
    """
    _apply(model)
    argv = [_which("fx1"), "eval", "--backend", "hosted_k3", "--model", model]
    if out is not None:
        argv += ["--out", str(out)]
    proc = subprocess.run(argv, check=False)  # noqa: S603 — argv built from literals
    return proc.returncode


def run_tool(tool: str, args: list[str], model: str = DEFAULT_MODEL) -> int:
    """Run a shipped console script with the model's key in its environment."""
    if tool not in TOOLS:
        raise KeyError(f"unknown tool {tool!r}; choose from {sorted(TOOLS)}")
    _apply(model)
    proc = subprocess.run([_which(tool), *args], check=False)  # noqa: S603
    return proc.returncode


def _complete_with_orb(
    backend: HostedK3Backend, messages: list[dict[str, str]], *, animated: bool = True
) -> str:
    """Run one model turn; a thinking orb animates while the API thinks."""
    if not animated:
        return backend.complete(messages)
    with OrbAnimator(OrbState.COMPOSING, preset="inline"):
        return backend.complete(messages)


def chat(
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    *,
    backend_factory: BackendFactory | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = typer.echo,
    animated: bool | None = None,
) -> int:
    """One-shot (``prompt`` given) or multi-turn chat with a hosted fx-1 model.

    Fail-closed when no endpoint resolves for *model*; network errors inside
    the loop are reported and the turn is dropped, never fatal to the shell.
    """
    factory = backend_factory or HostedK3Backend
    resolved = profiles.resolve_endpoint(model)
    if not resolved:
        raise RuntimeError(
            f"no endpoint for model {model!r}; run `dipcatcher` or `fxi setup` "
            f"to enter your API key and base URL, or export {profiles.ENV_API_KEY}"
        )
    api_key, base_url = resolved
    backend = factory(api_key=api_key, model=model, api_url=base_url)
    animate = True if animated is None else animated  # OrbAnimator no-ops off-TTY
    messages: list[dict[str, str]] = []
    if prompt is not None:
        text = prompt.strip()
        if not text:
            return 0
        messages.append({"role": "user", "content": text})
        reply = _complete_with_orb(backend, messages, animated=animate)
        messages.append({"role": "assistant", "content": reply})
        output_fn(reply)
        return 0
    host = profiles.host_of(base_url)
    output_fn(f"fxi chat — model {model} @ {host}; 'exit' or Ctrl-D to leave")
    while True:
        try:
            line = input_fn("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            return 0
        messages.append({"role": "user", "content": line})
        try:
            reply = _complete_with_orb(backend, messages, animated=animate)
        except Exception as exc:  # noqa: BLE001 — a dead network must not kill the shell
            output_fn(f"error: {exc}")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        output_fn(reply)


def harness_rows() -> list[dict[str, str]]:
    """Metadata for every registered harness command (pure, no subprocess)."""
    return [
        {
            "name": c.name,
            "role": c.role.value,
            "timeout_s": str(c.timeout_s),
            "description": c.description,
        }
        for c in HARNESS_REGISTRY
    ]


def harness_get(name: str) -> Mapping[str, Any]:
    command = Harness().get(name)
    return {
        "name": command.name,
        "role": command.role.value,
        "argv": command.argv,
        "timeout_s": command.timeout_s,
        "description": command.description,
    }


def harness_run(name: str, extra_args: list[str] | None = None, model: str = DEFAULT_MODEL) -> int:
    """Execute a registered harness command with the session env (keys included)."""
    _apply(model)
    result = Harness().run(name, extra_args)
    if result.stdout:
        typer.echo(result.stdout.rstrip())
    if result.stderr:
        typer.secho(result.stderr.rstrip(), err=True, fg=typer.colors.RED)
    return result.exit_code


def verify(path: Path, model: str = DEFAULT_MODEL) -> int:
    return run_tool("dipcatcher", ["verify-research", str(path)], model)


def _probe(base_url: str, timeout: float = 3.0) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "reachable"
    except OSError:
        return "unreachable"


def doctor(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """fx-1 readiness signals plus fxi endpoint state. Presence-only."""
    status = collect_status()
    store = profiles.store_path()
    status["fxi_store"] = str(store)
    status["fxi_store_mode"] = (
        oct(stat.S_IMODE(store.stat().st_mode)) if store.is_file() else "missing"
    )
    status["active_model"] = model
    endpoint = profiles.resolve_endpoint(model)
    base_url = endpoint[1] if endpoint else profiles.default_base_url()
    status["api_host"] = profiles.host_of(base_url)
    status["api_reachable"] = _probe(base_url)
    status["models"] = profiles.list_models()
    return status


def render_table(rows: Sequence[Mapping[str, str]]) -> str:
    """Plain fixed-width table for terminal output (no rich dependency)."""
    if not rows:
        return "(no rows)"
    columns = list(rows[0])
    widths = {col: max(len(col), *(len(str(row[col])) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    rule = "  ".join("-" * widths[col] for col in columns)
    body = ["  ".join(str(row[col]).ljust(widths[col]) for col in columns) for row in rows]
    return "\n".join([header, rule, *body])


def status_json(status: Mapping[str, Any]) -> str:
    return json.dumps(status, indent=2, default=str)
