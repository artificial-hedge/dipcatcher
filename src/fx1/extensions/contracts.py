"""Typed, fail-closed contracts for individual fx-1 extension modules."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fx1.extensions.naming import ExtensionKind, module_path

if TYPE_CHECKING:
    from fx1.data.sources import DataSourceAdapter, FetchRequest, FetchResult, SourceProbe
    from fx1.extensions.feature_catalog import FeatureDefinition
    from fx1.harness import Harness, HarnessCommand, HarnessResult


@dataclass(frozen=True)
class ExtensionModule:
    """A separately loadable artifact bound to a real harness surface."""

    kind: ExtensionKind
    owner: str
    references: Sequence[int]
    module: str

    def __post_init__(self) -> None:
        if self.module != module_path(self.kind, self.owner):
            raise ValueError(
                f"extension {self.kind}/{self.owner} must live at "
                f"{module_path(self.kind, self.owner)!r}"
            )
        if not self.references:
            raise ValueError(f"extension {self.kind}/{self.owner} has no capability records")

    @property
    def card_count(self) -> int:
        """Number of generated discovery cards assigned to this module."""
        return len(self.references)

    def manifest(self) -> dict[str, object]:
        """Return a read-only, directly addressable artifact manifest."""
        return {
            "schema": "fx1.extension-module/v1",
            "kind": self.kind,
            "owner": self.owner,
            "module": self.module,
            "card_count": self.card_count,
            "market_evidence": False,
        }

    def records(self, *, offset: int = 0, limit: int = 20) -> Iterator[dict[str, object]]:
        """Resolve cards owned by this artifact without scanning other modules."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        from fx1.capabilities import resolve_seed_id

        for entry_id in self.references[offset : offset + limit]:
            entry = resolve_seed_id(int(entry_id))
            if entry["kind"] != self.kind or self._record_owner(entry) != self.owner:
                raise ValueError(
                    f"extension {self.kind}/{self.owner} contains a foreign capability record"
                )
            yield entry

    def verify(self) -> None:
        """Check every generated registration in this module against the seed layout."""
        from fx1.capabilities import resolve_seed_id

        for entry_id in self.references:
            entry = resolve_seed_id(int(entry_id))
            if entry["kind"] != self.kind or self._record_owner(entry) != self.owner:
                raise ValueError(
                    f"extension {self.kind}/{self.owner} contains a foreign capability record"
                )

    def _record_owner(self, entry: dict[str, object]) -> str | None:
        raise NotImplementedError


@dataclass(frozen=True)
class SkillExtension(ExtensionModule):
    """One skill module bound to a single registered harness command."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "skill":
            raise ValueError("SkillExtension requires kind='skill'")

    def _record_owner(self, entry: dict[str, object]) -> str | None:
        v = entry.get("command")
        return v if isinstance(v, str) else None

    def command(self) -> HarnessCommand:
        """Return the fail-closed command this skill may invoke."""
        from fx1.harness import Harness

        return Harness().get(self.owner)

    def run(
        self,
        *,
        config: Path | None = None,
        harness: Harness | None = None,
    ) -> HarnessResult:
        """Run only this module's registered harness command."""
        from fx1.harness import Harness

        return (harness or Harness()).run(self.owner, config=config)


@dataclass(frozen=True)
class PluginExtension(ExtensionModule):
    """One datasource plugin module using the typed source adapter boundary."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "plugin":
            raise ValueError("PluginExtension requires kind='plugin'")

    def _record_owner(self, entry: dict[str, object]) -> str | None:
        v = entry.get("source")
        return v if isinstance(v, str) else None

    def adapter(self) -> DataSourceAdapter:
        """Build the registered adapter; availability remains fail-closed."""
        from fx1.data.sources import build_adapter, get_spec

        return build_adapter(get_spec(self.owner))

    def probe(self) -> SourceProbe:
        """Report honest local availability without reading credential values."""
        return self.adapter().probe()

    def describe(self) -> FetchResult:
        """Ask the installed source implementation for its documented operations."""
        return self.adapter().describe()

    def fetch(self, request: FetchRequest) -> FetchResult:
        """Route a typed request through the existing source adapter."""
        return self.adapter().fetch(request)


@dataclass(frozen=True)
class FeatureExtension(ExtensionModule):
    """One feature module bound to a named output of the PIT feature builder."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "feature":
            raise ValueError("FeatureExtension requires kind='feature'")

    def _record_owner(self, entry: dict[str, object]) -> str | None:
        v = entry.get("feature")
        return v if isinstance(v, str) else None

    def metadata(self) -> FeatureDefinition:
        """Return the declared metadata for this public feature column."""
        from fx1.extensions.feature_catalog import get_feature_definition

        return get_feature_definition(self.owner)

    def build(
        self,
        *,
        config: Path | None = None,
        harness: Harness | None = None,
    ) -> HarnessResult:
        """Run the sole registered PIT feature-build surface for this feature."""
        from fx1.harness import Harness

        return (harness or Harness()).run("build-features", config=config)
