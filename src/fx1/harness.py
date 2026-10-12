"""dipcatcher as the fx-1 harness.

The relationship, inverted: **fx-1 is the model; dipcatcher is the harness
that builds, evaluates, and verifies it.** This module is the typed bridge —
every lab capability fx-1 is allowed to touch is registered here as a
:class:`HarnessCommand` and executed through an injectable runner, so tests
never spawn real lab processes and fx-1 can never reach an unregistered
command.

Harness roles:
- *Data engine* — ingest/collect/features/labels feed ``fx1.data``.
- *Evaluation* — benches and books score fx-1's domain competence;
  ``fx1.eval`` scores its behavior.
- *Verification* — doctor / verify-research / validate / monitor gate every
  artifact the model is trained on or claims about.
- *Model training* — the lab's own train/optimize/paper surfaces, usable as
  curriculum and tool-use trace sources for fx-1.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess  # noqa: S404 - runner is injectable; see Harness.run
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fx1.interactive.procs import ProcessRegistry, ProcResult


class HarnessRole(StrEnum):
    DATA_ENGINE = "data_engine"
    EVALUATION = "evaluation"
    VERIFICATION = "verification"
    MODEL_TRAINING = "model_training"


class HarnessCommand(BaseModel):
    """A registered lab command fx-1 may invoke through the harness."""

    name: str
    role: HarnessRole
    argv: list[str] = Field(description="dipcatcher CLI argv (no executable)")
    timeout_s: int = Field(default=300, ge=1, le=7200)
    description: str


def _cmd(name: str, role: HarnessRole, description: str, timeout_s: int = 300) -> HarnessCommand:
    return HarnessCommand(
        name=name, role=role, argv=[name], timeout_s=timeout_s, description=description
    )


# The complete registry of lab surfaces fx-1 can touch — mirrors
# ``quant_fund.cli.main``. Anything not listed here is unreachable by the
# model: fail-closed by construction. The hidden ``lab`` command and the
# network-serving ``api`` command are deliberately excluded.
HARNESS_REGISTRY: tuple[HarnessCommand, ...] = (
    # --- Verification ---
    _cmd(
        "doctor",
        HarnessRole.VERIFICATION,
        "Verify data manifest and latest research receipt before trusting state.",
    ),
    _cmd(
        "verify-research",
        HarnessRole.VERIFICATION,
        "Verify immutable provenance, scorecards, and research artifacts.",
    ),
    _cmd(
        "validate",
        HarnessRole.VERIFICATION,
        "Fail-closed causal / walk-forward / promotion gates for a model id.",
    ),
    _cmd(
        "monitor",
        HarnessRole.VERIFICATION,
        "Operational monitoring surface for the harness runtime.",
    ),
    _cmd(
        "capabilities",
        HarnessRole.VERIFICATION,
        "Read-only capability contract catalog (dipcatcher.capabilities.v1). "
        "Does not execute benches.",
        timeout_s=120,
    ),
    _cmd(
        "verify-receipt",
        HarnessRole.VERIFICATION,
        "Verify one sealed receipt: structure plus hash consistency.",
        timeout_s=120,
    ),
    # --- Data engine ---
    _cmd(
        "ingest",
        HarnessRole.DATA_ENGINE,
        "Bronze/silver lake ingest; writes data_manifest.json with SHA-256 hashes.",
        timeout_s=1800,
    ),
    _cmd(
        "collect",
        HarnessRole.DATA_ENGINE,
        "Explicit opt-in public-feed collection (binance, fred) with receipts.",
    ),
    _cmd(
        "build-features",
        HarnessRole.DATA_ENGINE,
        "Feature construction on the PIT lake.",
        timeout_s=1800,
    ),
    _cmd(
        "build-labels",
        HarnessRole.DATA_ENGINE,
        "Label construction on the PIT lake.",
        timeout_s=1800,
    ),
    # --- Evaluation ---
    _cmd(
        "research",
        HarnessRole.EVALUATION,
        "Scientific benches (proper scores; SYNTHETIC labeled).",
        timeout_s=3600,
    ),
    _cmd(
        "families",
        HarnessRole.EVALUATION,
        "Read-only qualified research-family surface. Does not execute benches.",
        timeout_s=120,
    ),
    _cmd(
        "vol-bench",
        HarnessRole.EVALUATION,
        "SYNTHETIC volatility bench: QLIKE + MSE with a sealed receipt.",
        timeout_s=900,
    ),
    _cmd(
        "calibration-eval",
        HarnessRole.EVALUATION,
        "SYNTHETIC calibration lane: PIT, coverage, reliability with a sealed receipt.",
        timeout_s=900,
    ),
    _cmd(
        "rankic",
        HarnessRole.EVALUATION,
        "SYNTHETIC cross-sectional rank-IC bench with a sealed receipt.",
        timeout_s=900,
    ),
    _cmd(
        "serial-watch",
        HarnessRole.EVALUATION,
        "Serial-dependence watch over supplied PIT series with a sealed receipt.",
        timeout_s=600,
    ),
    _cmd(
        "lane-power",
        HarnessRole.EVALUATION,
        "Detection-power lane for monitors on SYNTHETIC streams with a sealed receipt.",
        timeout_s=900,
    ),
    _cmd(
        "score-forecasts",
        HarnessRole.EVALUATION,
        "Score user-supplied volatility, quantile, or probability forecasts with proper "
        "scores and seal a receipt.",
        timeout_s=300,
    ),
    _cmd(
        "northset",
        HarnessRole.EVALUATION,
        "Order-book/candlestick slice: identities, OHLC vol, Kyle/Roll/OFI/VPIN.",
        timeout_s=1800,
    ),
    _cmd(
        "backtest",
        HarnessRole.EVALUATION,
        "Next-open fills, costs, participation; tamper-evident artifacts.",
        timeout_s=1800,
    ),
    _cmd("forecast", HarnessRole.EVALUATION, "Forecast surface for validated models."),
    _cmd(
        "kronos-forecast",
        HarnessRole.EVALUATION,
        "Research-only local Kronos candlestick adapter (no network).",
    ),
    _cmd(
        "candle-book",
        HarnessRole.EVALUATION,
        "Candle/order-book reconstruction benches.",
        timeout_s=900,
    ),
    _cmd(
        "kyle-ofi",
        HarnessRole.EVALUATION,
        "Kyle lambda / OFI microstructure benches.",
        timeout_s=900,
    ),
    _cmd(
        "session-book", HarnessRole.EVALUATION, "Session-level order-book benches.", timeout_s=900
    ),
    _cmd("vendor-book-map", HarnessRole.EVALUATION, "Vendor book mapping diagnostics."),
    _cmd("book-panel", HarnessRole.EVALUATION, "Book panel aggregation benches.", timeout_s=900),
    _cmd("report", HarnessRole.EVALUATION, "Research notebook/report generation."),
    _cmd(
        "tearsheet",
        HarnessRole.EVALUATION,
        "Backtest tearsheet diagnostics (analytics export, live_pnl_claim=false).",
    ),
    # --- Model training (lab surfaces; curriculum + tool-use traces) ---
    _cmd(
        "train",
        HarnessRole.MODEL_TRAINING,
        "Lab model training families (ranking/distribution/calibration/"
        "volatility/alpha/covariance/regime/tail/reinforcement/liquidity).",
        timeout_s=7200,
    ),
    _cmd(
        "optimize",
        HarnessRole.MODEL_TRAINING,
        "Hyperparameter optimization under fail-closed gates.",
        timeout_s=7200,
    ),
    _cmd(
        "paper",
        HarnessRole.MODEL_TRAINING,
        "Simulated broker shadow runs (no live fills).",
        timeout_s=1800,
    ),
)

# Captured at import so a later mutation of a registry object cannot raise
# the deadline the runner is allowed to use.
_TIMEOUT_CEILING: dict[str, int] = {command.name: command.timeout_s for command in HARNESS_REGISTRY}


class HarnessResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# Runner signature: (argv, timeout_s) -> (exit_code, stdout, stderr).
# Extended runners set supports_extended and accept the keyword arguments
# passed by Harness.run; they return ProcResult.
Runner = Callable[[list[str], int], tuple[int, str, str]]
_Emit = Callable[[str, Mapping[str, Any]], None]


class ExtendedRunner(Protocol):
    """Runner that accepts cancel, env, and step and returns a ProcResult."""

    supports_extended: bool

    def __call__(
        self,
        argv: list[str],
        timeout_s: int,
        *,
        cancel_event: threading.Event | None = None,
        env_extra: Mapping[str, str] | None = None,
        step: str | None = None,
    ) -> ProcResult: ...


class OwnedRunner:
    """Default runner: ``python -m dipcatcher_cli`` with a scrubbed environment.

    The process runner is imported on the first call so ``import fx1.harness``
    stays limited to this module.
    """

    supports_extended: bool = True

    def __init__(
        self,
        *,
        registry: ProcessRegistry | None = None,
        emit: _Emit | None = None,
    ) -> None:
        self._registry = registry
        self._emit = emit

    def __call__(
        self,
        argv: list[str],
        timeout_s: int,
        *,
        cancel_event: threading.Event | None = None,
        env_extra: Mapping[str, str] | None = None,
        step: str | None = None,
    ) -> ProcResult:
        from fx1.interactive.procs import REGISTRY, ProcLimits, run_owned, scrub_env

        registry = REGISTRY if self._registry is None else self._registry
        registry.install_atexit()
        full = [sys.executable, "-m", "dipcatcher_cli", *argv]
        return run_owned(
            full,
            ProcLimits(timeout_s=float(timeout_s)),
            env=scrub_env(os.environ, extra=env_extra),
            cancel_event=cancel_event,
            registry=registry,
            emit=self._emit,
            step=step,
        )


def popen_cancellable(
    argv: list[str],
    timeout_s: int,
    cancel_event: threading.Event | None = None,
) -> tuple[int, str, str]:
    """Run a full argv until exit, timeout, or cancel (compatibility wrapper).

    Backed by the owned runner: own process group, bounded output, whole-tree
    termination. Cancel returns ``(130, stdout, stderr + "cancelled")``; a
    deadline raises ``subprocess.TimeoutExpired`` like ``subprocess.run``.
    """
    from fx1.interactive.procs import ProcLimits, run_owned, scrub_env

    result = run_owned(
        [str(part) for part in argv],
        ProcLimits(timeout_s=float(timeout_s)),
        env=scrub_env(os.environ),
        cancel_event=cancel_event,
    )
    if result.timed_out:
        raise subprocess.TimeoutExpired(argv, timeout_s, output=result.stdout, stderr=result.stderr)
    if result.cancelled:
        return 130, result.stdout, f"{result.stderr}\ncancelled".strip()
    return result.exit_code, result.stdout, result.stderr


class Harness:
    """The fx-1 side of the dipcatcher harness."""

    def __init__(self, runner: Runner | ExtendedRunner | None = None) -> None:
        self._runner: Runner | ExtendedRunner = runner if runner is not None else OwnedRunner()
        self._registry = {command.name: command for command in HARNESS_REGISTRY}

    def list_commands(self, role: HarnessRole | None = None) -> list[HarnessCommand]:
        commands = list(self._registry.values())
        if role is not None:
            commands = [command for command in commands if command.role == role]
        return commands

    def get(self, name: str) -> HarnessCommand:
        if name not in self._registry:
            raise KeyError(
                f"{name!r} is not a registered harness command; fx-1 may only "
                f"use: {sorted(self._registry)}"
            )
        return self._registry[name]

    def run(
        self,
        name: str,
        extra_args: list[str] | None = None,
        *,
        config: Path | None = None,
        cancel_event: threading.Event | None = None,
        env_extra: Mapping[str, str] | None = None,
        step: str | None = None,
        timeout_s: int | None = None,
    ) -> HarnessResult:
        """Execute a registered harness command, fail-closed on unknown names.

        Config paths are contained to the repo ``configs/`` directory, matching
        the lab API's allowlist posture. Runners without ``supports_extended``
        are called as ``runner(argv, timeout_s)``. Extended runners also receive
        cancellation, extra environment, and step. A deadline is a result with
        exit code 124, not an exception. ``timeout_s`` may shorten the registry
        ceiling; a larger request is clamped down to that ceiling.
        """
        command = self.get(name)
        limit = _clamp_timeout(command.name, command.timeout_s, timeout_s)
        argv = _assemble_argv(command, extra_args, config)
        outcome = self._dispatch(argv, limit, cancel_event, env_extra, step)
        return _coerce(name, outcome)

    def _dispatch(
        self,
        argv: list[str],
        timeout_s: int,
        cancel_event: threading.Event | None,
        env_extra: Mapping[str, str] | None,
        step: str | None,
    ) -> ProcResult | tuple[int, str, str]:
        try:
            return self._call_runner(argv, timeout_s, cancel_event, env_extra, step)
        except subprocess.TimeoutExpired as exc:
            return _timeout_result(exc)

    def _call_runner(
        self,
        argv: list[str],
        timeout_s: int,
        cancel_event: threading.Event | None,
        env_extra: Mapping[str, str] | None,
        step: str | None,
    ) -> ProcResult | tuple[int, str, str]:
        runner = self._runner
        if getattr(runner, "supports_extended", False):
            extended = cast(ExtendedRunner, runner)
            return extended(
                argv,
                timeout_s,
                cancel_event=cancel_event,
                env_extra=env_extra,
                step=step,
            )
        basic = cast(Runner, runner)
        return basic(argv, timeout_s)


def _assemble_argv(
    command: HarnessCommand,
    extra_args: list[str] | None,
    config: Path | None,
) -> list[str]:
    argv = list(command.argv)
    if config is not None:
        argv += ["--config", str(_contained_config(config))]
    if extra_args:
        _reject_extra_config(extra_args)
        argv += list(extra_args)
    return argv


def _clamp_timeout(name: str, registered: int, requested: int | None) -> int:
    ceiling = _TIMEOUT_CEILING.get(name, registered)
    chosen = _requested_timeout(registered, requested)
    if chosen > ceiling:
        return ceiling
    return chosen


def _requested_timeout(registered: int, requested: int | None) -> int:
    chosen = registered if requested is None else requested
    if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen < 1:
        raise ValueError(f"timeout_s must be a positive int, got {chosen!r}")
    return chosen


def _contained_config(raw: Path | str) -> Path:
    path = Path(raw)
    if path.is_symlink():
        raise ValueError(f"config path {path} escapes the configs/ allowlist")
    resolved = path.resolve()
    if _path_allowed(resolved):
        return resolved
    raise ValueError(f"config path {resolved} escapes the configs/ allowlist")


def _path_allowed(resolved: Path) -> bool:
    return any(_within_root(resolved, root) for root in _config_roots())


def _config_roots() -> tuple[Path, ...]:
    bundled = _bundled_configs_dir()
    if bundled is None:
        return (_configs_dir(),)
    return (_configs_dir(), bundled)


def _within_root(resolved: Path, root: Path) -> bool:
    return resolved == root or root in resolved.parents


def _bundled_configs_dir() -> Path | None:
    """Installed ``quant_fund/_bundled/configs``, without importing quant_fund."""
    origin = _quant_fund_origin()
    if origin is None:
        return None
    packaged = origin.parent / "_bundled" / "configs"
    if packaged.is_dir() and not packaged.is_symlink():
        return packaged.resolve()
    return _ancestor_configs(origin.parent)


def _quant_fund_origin() -> Path | None:
    spec = importlib.util.find_spec("quant_fund")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _ancestor_configs(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        configs = candidate / "configs"
        marker = (candidate / "pyproject.toml").is_file()
        if marker and configs.is_dir() and not configs.is_symlink():
            return configs.resolve()
    return None


def _configs_dir() -> Path:
    """Nearest directory that has both ``pyproject.toml`` and ``configs/``.

    A scratch working directory with only ``configs/`` (no project marker)
    keeps that directory as the allowlist. A symlink standing in for
    ``configs`` is rejected: resolving it would move the allowlist.
    """
    here = Path.cwd().resolve()
    found = _marked_configs(here)
    if found is not None:
        return found
    return _plain_configs(here)


def _plain_configs(here: Path) -> Path:
    local = here / "configs"
    if local.is_symlink():
        raise ValueError(f"config path {local} escapes the configs/ allowlist")
    return local.resolve()


def _marked_configs(here: Path) -> Path | None:
    for candidate in (here, *here.parents):
        configs = candidate / "configs"
        if (candidate / "pyproject.toml").is_file() and configs.is_dir():
            if configs.is_symlink():
                raise ValueError(f"config path {configs} escapes the configs/ allowlist")
            return configs.resolve()
    return None


def _reject_extra_config(extra_args: list[str]) -> None:
    # ``--config`` in extra_args must obey the same containment as the keyword.
    for index, arg in enumerate(extra_args):
        _check_config_arg(extra_args, index, arg)


def _check_config_arg(extra_args: list[str], index: int, arg: str) -> None:
    if arg == "--config":
        _require_config_value(extra_args, index)
        return
    if arg.startswith("--config="):
        _require_config_text(arg.split("=", 1)[1])


def _require_config_value(extra_args: list[str], index: int) -> None:
    if index + 1 >= len(extra_args):
        raise ValueError("missing --config value escapes the configs/ allowlist")
    _require_config_text(extra_args[index + 1])


def _require_config_text(value: str) -> None:
    if value == "":
        raise ValueError("missing --config value escapes the configs/ allowlist")
    _contained_config(value)


def _coerce(name: str, outcome: ProcResult | tuple[int, str, str]) -> HarnessResult:
    if isinstance(outcome, tuple):
        code, out, err = outcome
        return HarnessResult(command=name, exit_code=code, stdout=out, stderr=err)
    return HarnessResult(
        command=name,
        exit_code=outcome.exit_code,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        timed_out=outcome.timed_out,
        cancelled=outcome.cancelled,
        truncated=outcome.truncated,
        duration_s=outcome.duration_s,
    )


def _timeout_result(exc: subprocess.TimeoutExpired) -> ProcResult:
    from fx1.interactive.procs import ProcResult as ProcResultCls

    note = "\n[timeout] process exceeded its deadline; terminated\n"
    return ProcResultCls(
        argv_summary=_timeout_argv(exc.cmd),
        exit_code=124,
        stdout=_as_text(exc.stdout),
        stderr=_as_text(exc.stderr) + note,
        timed_out=True,
        cancelled=False,
        truncated=False,
        spawn_error=None,
        duration_s=0.0,
        pid=None,
        pgid=None,
    )


def _timeout_argv(cmd: str | bytes | Sequence[object]) -> list[str]:
    from fx1.interactive.redact import redact_argv

    return redact_argv(_command_parts(cmd))


def _command_parts(cmd: str | bytes | Sequence[object]) -> list[str]:
    if isinstance(cmd, (str, bytes)):
        return [os.fsdecode(cmd)]
    parts: list[str] = []
    for part in cmd:
        parts.append(_part_text(part))
    return parts


def _part_text(part: object) -> str:
    if isinstance(part, (str, bytes)):
        return os.fsdecode(part)
    return str(part)


def _as_text(payload: str | bytes | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload
