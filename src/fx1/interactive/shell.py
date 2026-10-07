"""Interactive REPL for ``fxi`` — stdlib ``cmd.Cmd``, no new dependencies.

Session state is just the active model: ``use fx1-lite`` injects that
model's endpoint (``MOONSHOT_API_KEY`` + ``FX1_BASE_URL``) into
``os.environ``, so every harness command launched from the shell
(directly or via ``run``/``!``) inherits the credential. The prompt shows
``model @ host``; a static listening-orb banner opens the session.
"""

from __future__ import annotations

import cmd
import getpass
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

import typer

from fx1.interactive import actions, profiles
from fx1.interactive.orb import banner as orb_banner
from fx1.interactive.profiles import DEFAULT_MODEL

_HELP = """\
keys                              list model endpoints (presence-only)
keys set <model>                  store an API key + base URL (prompted, not echoed)
keys remove <model>               delete a stored endpoint
use <model>                       activate a model and inject its endpoint
model                             show the active model and host
doctor                            readiness: endpoints, store, API reachability
eval [--out PATH]                 run the fx-1 eval bank via the active model
chat [opening words]              multi-turn chat with the active model
harness list                      registered dipcatcher harness commands
harness describe <name>           one harness command's metadata
harness run <name> [args]         execute a registered harness command
verify <path>                     dipcatcher verify-research on a receipt/artifact
run <tool> <args...>              launch fx1 | dipcatcher | quant with the session env
!<shell command>                  run a raw shell command
exit                              leave the shell
"""


def _endpoint_host(model: str) -> str:
    resolved = profiles.resolve_endpoint(model)
    base_url = resolved[1] if resolved else profiles.default_base_url()
    return profiles.host_of(base_url)


class FxiShell(cmd.Cmd):
    intro = ""
    ruler = "-"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        super().__init__()
        self.model = model
        self._refresh_prompt()

    # -- session ------------------------------------------------------------

    def _refresh_prompt(self) -> None:
        self.prompt = f"fxi · {self.model}@{_endpoint_host(self.model)} › "

    def _say_error(self, exc: Exception) -> None:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)

    def _split(self, arg: str) -> list[str]:
        try:
            return shlex.split(arg)
        except ValueError as exc:
            raise ValueError(f"could not parse arguments: {exc}") from None

    # -- key management -----------------------------------------------------

    def do_keys(self, arg: str) -> None:
        try:
            tokens = self._split(arg)
        except ValueError as exc:
            self._say_error(exc)
            return
        if not tokens:
            typer.echo(actions.render_table(profiles.list_models()))
            return
        sub, rest = tokens[0], tokens[1:]
        if sub == "set":
            self._keys_set(rest)
        elif sub == "remove":
            self._keys_remove(rest)
        else:
            self._say_error(ValueError(f"unknown keys subcommand {sub!r}; use set|remove"))

    def complete_keys(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        subcommands = ["set", "remove"]
        if line[:begidx].strip() == "keys":
            return [s for s in subcommands if s.startswith(text)]
        return [m for m in profiles.MODELS if m.startswith(text)]

    def _keys_set(self, rest: list[str]) -> None:
        if len(rest) > 1:
            self._say_error(ValueError("usage: keys set <model>"))
            return
        model = rest[0] if rest else DEFAULT_MODEL
        try:
            profiles.check_model(model)
        except KeyError as exc:
            self._say_error(exc)
            return
        default_host = profiles.host_of(profiles.default_base_url())
        api_key = getpass.getpass(f"› {model} API key: ")
        raw_url = input(f"› {model} base URL [{default_host}]: ").strip()
        try:
            path = profiles.set_endpoint(model, api_key, raw_url or None)
        except ValueError as exc:
            self._say_error(exc)
            return
        typer.echo(f"stored endpoint for {model!r} in {path} (mode 0600)")

    def _keys_remove(self, rest: list[str]) -> None:
        if len(rest) != 1:
            self._say_error(ValueError("usage: keys remove <model>"))
            return
        try:
            removed = profiles.remove_endpoint(rest[0])
        except KeyError as exc:
            self._say_error(exc)
            return
        typer.echo(
            f"removed endpoint {rest[0]!r}" if removed else f"endpoint {rest[0]!r} was not stored"
        )

    def do_use(self, arg: str) -> None:
        try:
            model = self._split(arg)[0]
        except (ValueError, IndexError):
            self._say_error(ValueError("usage: use <model>"))
            return
        try:
            profiles.apply_profile(model)
        except (KeyError, RuntimeError) as exc:
            self._say_error(exc)
            return
        self.model = model
        self._refresh_prompt()
        typer.echo(
            f"active model {model!r}; {profiles.ENV_API_KEY} + {profiles.ENV_BASE_URL} "
            "injected into this session"
        )

    def complete_use(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        return [m for m in profiles.MODELS if m.startswith(text)]

    def do_model(self, arg: str) -> None:
        typer.echo(f"active model: {self.model} @ {_endpoint_host(self.model)}")

    # -- harness actions ----------------------------------------------------

    def do_doctor(self, arg: str) -> None:
        typer.echo(actions.status_json(actions.doctor(self.model)))

    def do_eval(self, arg: str) -> None:
        try:
            tokens = self._split(arg)
        except ValueError as exc:
            self._say_error(exc)
            return
        out: Path | None = None
        i = 0
        while i < len(tokens):
            if tokens[i] == "--out" and i + 1 < len(tokens):
                out = Path(tokens[i + 1])
                i += 2
            else:
                self._say_error(ValueError(f"unknown eval argument {tokens[i]!r}"))
                return
        try:
            code = actions.run_eval(out, self.model)
        except (KeyError, RuntimeError) as exc:
            self._say_error(exc)
            return
        if code:
            typer.secho(f"eval exited {code}", err=True, fg=typer.colors.RED)

    def do_chat(self, arg: str) -> None:
        opening = arg.strip() or None
        try:
            actions.chat(self.model, opening)
        except RuntimeError as exc:
            self._say_error(exc)

    def do_harness(self, arg: str) -> None:
        try:
            tokens = self._split(arg)
        except ValueError as exc:
            self._say_error(exc)
            return
        if not tokens or tokens[0] == "list":
            typer.echo(actions.render_table(actions.harness_rows()))
            return
        sub = tokens[0]
        if sub == "describe" and len(tokens) == 2:
            try:
                meta = actions.harness_get(tokens[1])
            except KeyError as exc:
                self._say_error(exc)
                return
            typer.echo(actions.status_json(meta))
            return
        if sub == "run" and len(tokens) >= 2:
            try:
                code = actions.harness_run(tokens[1], tokens[2:], self.model)
            except (KeyError, RuntimeError) as exc:
                self._say_error(exc)
                return
            if code:
                typer.secho(f"harness command exited {code}", err=True, fg=typer.colors.RED)
            return
        self._say_error(
            ValueError("usage: harness [list | describe <name> | run <name> [args...]]")
        )

    def complete_harness(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        prefix = line[:begidx].strip()
        if prefix == "harness":
            return [s for s in ("list", "describe", "run") if s.startswith(text)]
        names = [row["name"] for row in actions.harness_rows()]
        return [n for n in names if n.startswith(text)]

    def do_verify(self, arg: str) -> None:
        try:
            tokens = self._split(arg)
        except ValueError as exc:
            self._say_error(exc)
            return
        if len(tokens) != 1:
            self._say_error(ValueError("usage: verify <path>"))
            return
        try:
            code = actions.verify(Path(tokens[0]), self.model)
        except (KeyError, RuntimeError) as exc:
            self._say_error(exc)
            return
        if code:
            typer.secho(f"verify exited {code}", err=True, fg=typer.colors.RED)

    def do_run(self, arg: str) -> None:
        try:
            tokens = self._split(arg)
        except ValueError as exc:
            self._say_error(exc)
            return
        if not tokens:
            self._say_error(ValueError("usage: run <tool> <args...>"))
            return
        try:
            code = actions.run_tool(tokens[0], tokens[1:], self.model)
        except (KeyError, RuntimeError) as exc:
            self._say_error(exc)
            return
        if code:
            typer.secho(f"{tokens[0]} exited {code}", err=True, fg=typer.colors.RED)

    def complete_run(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        if line[:begidx].strip() == "run":
            return [t for t in actions.TOOLS if t.startswith(text)]
        return []

    # -- shell passthrough --------------------------------------------------

    def do_shell(self, arg: str) -> None:
        if not arg.strip():
            self._say_error(ValueError("usage: !<shell command>"))
            return
        subprocess.run(arg, shell=True, check=False)  # noqa: S602  # nosec B602 — explicit operator escape hatch

    do_bang = do_shell

    # -- lifecycle ----------------------------------------------------------

    def do_exit(self, arg: str) -> bool:
        return True

    do_quit = do_exit

    def do_EOF(self, arg: str) -> bool:  # noqa: N802 — cmd.Cmd hook name
        typer.echo("")
        return True

    def do_help(self, arg: str) -> None:  # noqa: A003 — cmd.Cmd hook name
        typer.echo(_HELP)

    def default(self, line: str) -> None:
        typer.secho(
            f"unknown command {line.split()[0]!r}; type 'help'",
            err=True,
            fg=typer.colors.YELLOW,
        )


def run_shell(model: str = DEFAULT_MODEL, *, output_fn: Callable[..., None] = typer.echo) -> None:
    output_fn(orb_banner(model, _endpoint_host(model)))
    FxiShell(model=model).cmdloop()
