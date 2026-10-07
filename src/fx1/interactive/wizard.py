"""First-run onboarding: enter your API key and base URL, then drop into the
concierge console (chat, /superpower, web research, flash context).

Launched by ``fxi setup``. Every I/O handle is injectable so tests can drive
the wizard without a TTY; set ``animations=False`` for fully static output.
"""

from __future__ import annotations

import getpass
import importlib
from collections.abc import Callable
from typing import Any

import typer

from fx1.interactive import actions, profiles
from fx1.interactive.orb import OrbAnimator, OrbState
from fx1.interactive.orb import banner as orb_banner
from fx1.interactive.profiles import DEFAULT_MODEL


def _default_shell_factory() -> Any:
    """The concierge console entry point, resolved lazily.

    ``fx1.interactive.concierge`` ships in a later commit; resolving at
    call time keeps the wizard module importable while it is staged.
    """
    return importlib.import_module("fx1.interactive.concierge").run_console


def _prompt_endpoint(
    model: str,
    *,
    input_fn: Callable[[str], str],
    password_fn: Callable[[str], str],
    output_fn: Callable[..., None],
) -> tuple[str, str] | None:
    """Ask for one model's API key (not echoed) and base URL. None = skipped."""
    default_host = profiles.host_of(profiles.default_base_url())
    attempts = 0
    while True:
        try:
            api_key = password_fn(f"› {model} API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return None
        if api_key:
            break
        attempts += 1
        if attempts >= 3:
            output_fn(f"skipped {model}: no API key entered")
            return None
    while True:
        try:
            raw_url = input_fn(f"› {model} base URL [{default_host}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return None
        base_url = raw_url or profiles.default_base_url()
        try:
            profiles.set_endpoint(model, api_key, base_url)
            return api_key, base_url
        except ValueError as exc:
            output_fn(f"error: {exc}")


def enter(
    *,
    force: bool = False,
    input_fn: Callable[[str], str] = input,
    password_fn: Callable[[str], str] = getpass.getpass,
    output_fn: Callable[..., None] = typer.echo,
    animations: bool = True,
    shell_factory: Callable[..., None] | None = None,
    prober: Callable[[str], str] = actions._probe,
) -> None:
    """Onboard the operator, then hand off to the concierge console."""
    if shell_factory is None:
        shell_factory = _default_shell_factory()
    resolved = profiles.resolve_endpoint(DEFAULT_MODEL)
    if not force and resolved:
        shell_factory(DEFAULT_MODEL)
        return

    model = DEFAULT_MODEL
    output_fn(orb_banner(model, profiles.host_of(profiles.default_base_url())))
    output_fn("Welcome to dipcatcher — the fx-1 harness. Set up your hosted model endpoint.")
    output_fn("(keys are stored in ~/.fx1/credentials.json, mode 0600, shown never — only ...wxyz)")
    output_fn("")

    try:
        raw_model = input_fn(f"› model to configure [{DEFAULT_MODEL}] (fx1 | fx1-lite): ").strip()
    except (EOFError, KeyboardInterrupt):
        output_fn("")
        return
    if raw_model:
        try:
            profiles.check_model(raw_model)
            model = raw_model
        except KeyError as exc:
            output_fn(f"error: {exc}; keeping {DEFAULT_MODEL}")
            model = DEFAULT_MODEL

    endpoint = _prompt_endpoint(
        model, input_fn=input_fn, password_fn=password_fn, output_fn=output_fn
    )
    if endpoint is None:
        return
    _, base_url = endpoint

    try:
        other = "fx1-lite" if model == "fx1" else "fx1"
        also = input_fn(f"› also configure {other}? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        also = ""
    if also in {"y", "yes"}:
        _prompt_endpoint(
            "fx1-lite" if model == "fx1" else "fx1",
            input_fn=input_fn,
            password_fn=password_fn,
            output_fn=output_fn,
        )

    host = profiles.host_of(base_url)
    if animations:
        with OrbAnimator(OrbState.SEARCHING, preset="inline"):
            reachability = prober(base_url)
    else:
        output_fn(f"checking {host} …")
        reachability = prober(base_url)
    if reachability == "reachable":
        output_fn(f"{host} is reachable — endpoint looks good.")
    else:
        output_fn(f"warning: could not reach {host} (TCP connect). Saved anyway — re-check later.")

    output_fn("")
    output_fn(f"Entering the harness — model {model}. Type 'help' once inside.")
    shell_factory(model)
