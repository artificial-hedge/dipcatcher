"""SYNTHETIC adversarial tests for the generated fx-1 extension leaf modules.

Every file under ``fx1.extensions.{features,skills,plugins}`` is a generated
single-owner wrapper: it must re-declare exactly the identity its path claims
(kind, owner, module), carry only its own registered capability records, and
never assert market evidence. These probes enumerate the real tree — a file
that deviates from the wrapper shape, a duplicated owner, or a foreign
capability record fails closed.

All fixtures are synthetic; results are correctness checks, never market
evidence.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from fx1.extensions.contracts import (
    ExtensionModule,
    FeatureExtension,
    PluginExtension,
    SkillExtension,
)
from fx1.extensions.naming import module_basename, module_path

ROOT = Path(__file__).resolve().parents[2]
LEAF_ROOT = ROOT / "src" / "fx1" / "extensions"

_DIR_SPEC = {
    "features": ("feature", FeatureExtension),
    "skills": ("skill", SkillExtension),
    "plugins": ("plugin", PluginExtension),
}

_WRAPPER = re.compile(
    r'^"""Generated (?P<kind>feature|skill|plugin) wrapper for'
    r" '(?P<owner>[a-z][a-z0-9_-]*)'; binds to the existing registry\.\"\"\"\n"
    r"\nfrom fx1\.capabilities import owner_references\n"
    r"from fx1\.extensions\.contracts import (?P<cls>\w+)\n"
    r"\nMODULE = (?P=cls)\(\n"
    r'    kind="(?P=kind)",\n'
    r'    owner="(?P=owner)",\n'
    r'    references=owner_references\("(?P=kind)", "(?P=owner)"\),\n'
    r"    module=__name__,\n\)\n$"
)


def _leaf_files() -> list[tuple[str, Path]]:
    leaves: list[tuple[str, Path]] = []
    for directory in _DIR_SPEC:
        for path in sorted((LEAF_ROOT / directory).glob("*.py")):
            if path.name != "__init__.py":
                leaves.append((directory, path))
    return leaves


_LEAVES = _leaf_files()


def test_leaf_census_is_nonempty() -> None:
    # Pin the generated-tree shape so the parametrize sweep below can never
    # silently degenerate to zero cases if the leaves move.
    assert len(_LEAVES) >= 80


@pytest.mark.parametrize("directory,path", _LEAVES, ids=lambda p: getattr(p, "stem", str(p)))
def test_leaf_matches_canonical_wrapper_shape(directory: str, path: Path) -> None:
    """A leaf that is not byte-shaped like the generator output is a defect."""
    match = _WRAPPER.match(path.read_text(encoding="utf-8"))
    assert match is not None, f"{path.relative_to(ROOT)} deviates from wrapper shape"
    kind, expected_cls = _DIR_SPEC[directory]
    assert match["kind"] == kind
    assert match["cls"] == expected_cls.__name__
    # Filename stem is the only permitted basename for the declared owner.
    assert path.stem == module_basename(match["owner"])


@pytest.mark.parametrize("directory,path", _LEAVES, ids=lambda p: getattr(p, "stem", str(p)))
def test_leaf_identity_and_records(directory: str, path: Path) -> None:
    kind, expected_cls = _DIR_SPEC[directory]
    module = importlib.import_module(f"fx1.extensions.{directory}.{path.stem}")
    leaf = module.MODULE
    assert isinstance(leaf, expected_cls)
    assert leaf.kind == kind
    assert leaf.module == module_path(leaf.kind, leaf.owner)
    assert leaf.references  # fail-closed: no empty capability record list
    # Every referenced card resolves back to this owner only.
    leaf.verify()
    # The manifest is a read-only artifact claim; it can never assert
    # market evidence (honesty contract).
    manifest = leaf.manifest()
    assert manifest["schema"] == "fx1.extension-module/v1"
    assert manifest["market_evidence"] is False


def test_leaf_owners_are_unique() -> None:
    """No two generated leaves may resolve to the same (kind, owner)."""
    seen: set[tuple[str, str]] = set()
    for directory, path in _LEAVES:
        module = importlib.import_module(f"fx1.extensions.{directory}.{path.stem}")
        key = (module.MODULE.kind, module.MODULE.owner)
        assert key not in seen, f"owner collision at {path.name}"
        seen.add(key)


def test_leaf_references_do_not_overlap() -> None:
    """A capability card may be owned by exactly one generated leaf."""
    seen: set[int] = set()
    for directory, path in _LEAVES:
        module = importlib.import_module(f"fx1.extensions.{directory}.{path.stem}")
        for entry_id in module.MODULE.references:
            assert entry_id not in seen, f"card {entry_id} claimed by two leaves"
            seen.add(entry_id)


def test_skill_owners_are_kebab_commands() -> None:
    """Skill owners name harness commands (kebab); leaves stay snake files."""
    for path in sorted((LEAF_ROOT / "skills").glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = importlib.import_module(f"fx1.extensions.skills.{path.stem}")
        owner = module.MODULE.owner
        assert re.fullmatch(r"[a-z][a-z0-9-]*", owner), owner
        assert path.stem == owner.replace("-", "_")


def test_verify_rejects_foreign_capability_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A leaf whose registration resolves to another owner must fail closed."""
    leaf = importlib.import_module("fx1.extensions.features.log_ret_1").MODULE
    from fx1 import capabilities

    foreign = {"kind": "feature", "owner": "other-owner"}

    def _foreign(seed_id: int) -> dict[str, object]:
        return dict(foreign)

    monkeypatch.setattr(capabilities, "resolve_seed_id", _foreign)
    with pytest.raises(ValueError, match="foreign capability record"):
        leaf.verify()


def test_records_pagination_bounds() -> None:
    leaf = importlib.import_module("fx1.extensions.features.log_ret_1").MODULE
    with pytest.raises(ValueError, match="offset must be non-negative"):
        list(leaf.records(offset=-1))
    with pytest.raises(ValueError, match="limit must be between"):
        list(leaf.records(limit=0))
    with pytest.raises(ValueError, match="limit must be between"):
        list(leaf.records(limit=101))


def test_every_leaf_is_an_extension_module() -> None:
    for directory, path in _LEAVES:
        module = importlib.import_module(f"fx1.extensions.{directory}.{path.stem}")
        assert isinstance(module.MODULE, ExtensionModule)
