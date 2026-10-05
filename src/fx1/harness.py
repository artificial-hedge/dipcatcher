"""dipcatcher as the fx-1 harness.

The relationship, inverted: **fx-1 is the model; dipcatcher is the harness
that builds, evaluates, and verifies it.** This module is the typed bridge —
lab CLI surfaces are registered here as :class:`HarnessCommand` objects and
executed through an injectable runner, so tests never spawn real lab processes.
Independent feature, skill, and plugin implementations use the explicit
``fx1.operations`` registry and a host-controlled workspace root. Neither
execution path accepts unregistered commands or arbitrary module names.

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

import subprocess  # noqa: S404 - runner is injectable; see Harness.run
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field


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


class HarnessResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# Runner signature: (argv, timeout_s) -> (exit_code, stdout, stderr)
Runner = Callable[[list[str], int], tuple[int, str, str]]


def _subprocess_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603 - argv is registry-constrained
        ["dipcatcher", *argv],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


class Harness:
    """The fx-1 side of the dipcatcher harness."""

    def __init__(self, runner: Runner | None = None, *, workspace_root: Path | None = None) -> None:
        self._runner = runner or _subprocess_runner
        self._registry = {c.name: c for c in HARNESS_REGISTRY}
        self._workspace_root = (
            workspace_root if workspace_root is not None else Path.cwd()
        ).resolve()

    def list_commands(self, role: HarnessRole | None = None) -> list[HarnessCommand]:
        commands = list(self._registry.values())
        if role is not None:
            commands = [c for c in commands if c.role == role]
        return commands

    def search_capabilities(
        self,
        query: str = "",
        *,
        kind: str | None = None,
        command: str | None = None,
        source: str | None = None,
        feature: str | None = None,
        market: str | None = None,
        asset: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, object]:
        """Search the skills, datasource plugins, and feature recipes catalog.

        This is a read-only discovery surface. Every returned command or data
        source is checked against the existing fail-closed registries; catalog
        entries do not add executable commands.
        """
        from fx1.capabilities import CapabilityKind, search_capabilities

        return search_capabilities(
            query,
            kind=cast(CapabilityKind | None, kind),
            command=command,
            source=source,
            feature=feature,
            market=market,
            asset=asset,
            offset=offset,
            limit=limit,
        )

    def discovery_tool_specs(self) -> list[dict[str, object]]:
        """Return AI discovery and operation tools separate from lab subprocesses."""
        from fx1.capabilities import CAPABILITY_SEARCH_TOOL_SPEC, EXTENSION_MANIFEST_TOOL_SPEC
        from fx1.operations.registry import operation_tool_specs

        return [
            CAPABILITY_SEARCH_TOOL_SPEC,
            EXTENSION_MANIFEST_TOOL_SPEC,
            *operation_tool_specs(),
        ]

    def execute_operation(
        self, operation_id: str, arguments: dict[str, object]
    ) -> dict[str, object]:
        """Run a registered implementation inside the host-selected workspace."""
        from fx1.operations.registry import execute_operation

        return execute_operation(operation_id, arguments, workspace_root=self._workspace_root)

    def extension_manifest(self, kind: str, owner: str) -> dict[str, object]:
        """Load one approved extension manifest through the fixed package registry."""
        from fx1.extensions.naming import ExtensionKind
        from fx1.extensions.registry import extension_manifest

        return extension_manifest(cast(ExtensionKind, kind), owner)

    def invoke_discovery_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Invoke a schema-validated discovery or operation tool by its fixed name."""
        if name in ("list_operations", "describe_operation", "execute_operation"):
            from fx1.operations.registry import invoke_operation_tool

            return invoke_operation_tool(name, arguments, workspace_root=self._workspace_root)
        if name == "get_extension_manifest":
            from fx1.capabilities import ExtensionManifestArguments

            parsed_manifest = ExtensionManifestArguments.model_validate(arguments)
            return self.extension_manifest(parsed_manifest.kind, parsed_manifest.owner)
        if name != "search_capabilities":
            raise KeyError(
                "unknown discovery tool "
                f"{name!r}; known: ['search_capabilities', 'get_extension_manifest', "
                "'list_operations', 'describe_operation', 'execute_operation']"
            )
        from fx1.capabilities import CapabilitySearchArguments, search_capabilities

        parsed = CapabilitySearchArguments.model_validate(arguments)
        return search_capabilities(
            parsed.query,
            kind=parsed.kind,
            command=parsed.command,
            source=parsed.source,
            feature=parsed.feature,
            market=parsed.market,
            asset=parsed.asset,
            offset=parsed.offset,
            limit=parsed.limit,
        )

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
    ) -> HarnessResult:
        """Execute a registered harness command, fail-closed on unknown names.

        Config paths are contained to the repo ``configs/`` directory, matching
        the lab API's allowlist posture.
        """
        command = self.get(name)
        argv = list(command.argv)
        if config is not None:
            resolved = config.resolve()
            configs_dir = (Path.cwd() / "configs").resolve()
            if configs_dir not in resolved.parents and resolved != configs_dir:
                raise ValueError(f"config path {resolved} escapes the configs/ allowlist")
            argv += ["--config", str(resolved)]
        if extra_args:
            argv += list(extra_args)
        code, out, err = self._runner(argv, command.timeout_s)
        return HarnessResult(command=name, exit_code=code, stdout=out, stderr=err)
