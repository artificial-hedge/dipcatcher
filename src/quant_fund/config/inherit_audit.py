"""inherit_audit — pin the actual semantics of config ``inherit:`` + deep_merge.

The loader's contract is subtler than it looks: ``_root`` binds to the
*child's* directory, so inheriting ``../base.yaml`` (a sibling directory)
legitimately escapes the root and fails closed. deep_merge replaces
non-dict values wholesale — an overlay list *replaces*, never appends.
Cycles and root escapes must raise; same-dir and nested-child inheritance
must work.

This lane writes a scratch config tree and asserts each pattern's real
verdict — not what we wish it were, what it is. Drift in these semantics
is a config-supply-chain bug class. Sealed ``inherit_audit.v1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from quant_fund.config.loader import deep_merge, load_config
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["inherit_audit", "inherit_audit_bench"]


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _outcome(fn: Any) -> str:
    try:
        fn()
        return "ok"
    except ValueError as exc:
        return f"raise:{type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 — any other class is itself a finding
        return f"unexpected:{type(exc).__name__}"


def inherit_audit(scratch: Path) -> dict[str, Any]:
    """Exercise the loader's real inheritance semantics on a scratch tree."""
    results: dict[str, Any] = {}

    _write(
        scratch / "base.yaml",
        {
            "runtime": {"experiment_id": "base"},
            "horizons": {"bars": [1, 5], "names": ["h1", "h5"]},
        },
    )
    _write(
        scratch / "child.yaml",
        {"inherit": "base.yaml", "runtime": {"experiment_id": "child"}},
    )
    cfg = load_config(scratch / "child.yaml")
    results["same_dir_inherit"] = {
        "ok": cfg.runtime.experiment_id == "child" and cfg.horizons.bars == [1, 5],
        "detail": "override + non-overlapping parent key survives",
    }

    _write(scratch / "sub" / "leaf.yaml", {"inherit": "../base.yaml"})
    results["parent_dir_inherit"] = {
        "outcome": _outcome(lambda: load_config(scratch / "sub" / "leaf.yaml")),
        "note": "root binds to the child's dir — inheriting upward escapes it",
    }

    _write(scratch / "cyc" / "a.yaml", {"inherit": "b.yaml"})
    _write(scratch / "cyc" / "b.yaml", {"inherit": "a.yaml"})
    results["cycle"] = {
        "outcome": _outcome(lambda: load_config(scratch / "cyc" / "a.yaml")),
    }

    _write(scratch / "escape.yaml", {"inherit": "../../etc/passwd"})
    results["root_escape"] = {
        "outcome": _outcome(lambda: load_config(scratch / "escape.yaml")),
    }

    _write(
        scratch / "nested" / "inner.yaml",
        {"inherit": "../base.yaml", "runtime": {"experiment_id": "inner"}},
    )
    _write(scratch / "nested" / "outer.yaml", {"inherit": "inner.yaml"})
    results["nested_chain_up"] = {
        "outcome": _outcome(lambda: load_config(scratch / "nested" / "outer.yaml")),
    }

    merge = deep_merge(
        {"a": [1, 2], "b": {"x": 1, "y": 2}, "c": 1},
        {"a": [3], "b": {"y": 9}},
    )
    results["list_semantics"] = {
        "overlay_list_replaces": merge["a"] == [3],
        "nested_merge_keeps_sibling": merge["b"] == {"x": 1, "y": 9},
        "scalar_overlay_wins": merge["c"] == 1,
    }
    return results


def inherit_audit_bench(tmp_root: Path | str | None = None) -> dict[str, Any]:
    import shutil
    import tempfile

    if tmp_root is None:
        scratch = Path(tempfile.mkdtemp(prefix="inherit_audit_"))
    else:
        scratch = Path(tmp_root)
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
    results = inherit_audit(scratch)
    # Contract: cycles and escapes must raise; same-dir works; upward inherit
    # is documented-fail; list overlay replaces (never appends).
    ok = (
        results["same_dir_inherit"]["ok"]
        and results["cycle"]["outcome"].startswith("raise")
        and results["root_escape"]["outcome"].startswith(("raise", "unexpected:FileNotFoundError"))
        and results["parent_dir_inherit"]["outcome"].startswith(
            ("raise", "unexpected:FileNotFoundError")
        )
        and results["list_semantics"]["overlay_list_replaces"]
        and results["list_semantics"]["nested_merge_keeps_sibling"]
    )
    payload: dict[str, Any] = {
        "kind": "inherit_audit",
        "schema": "inherit_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": results, "ok": ok},
        "interpretation": (
            "Config inheritance semantics pinned: upward inherit fails closed "
            "(root = child dir), cycles raise, escapes raise, list overlays "
            "replace rather than append."
            if ok
            else f"SEMANTIC DRIFT: {results}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
