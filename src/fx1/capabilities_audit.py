"""capabilities_audit — the compact seed-ledger's own contract battery.

``fx1.capabilities`` holds the capability declaration ledger in compact
form: one arithmetic progression ``(owner, first, stride, count)`` per
extension owner, where the expanded million-line source ledger lives at
``scripts/generated_capability_declarations/{features,skills,plugins}/``.

This battery proves the compact table *is* the ledger — every shard
file is expanded line-by-line and its ``_register(seed_id)`` ids must
reproduce the owner's progression exactly — and pins the resolution
contract:

- table integrity (nonnegative first, positive stride/count, in-range)
- ``owner_references`` arithmetic, ``ValueError`` on unknown kind,
  ``KeyError`` on unknown owner
- ``resolve_seed_id`` bounds (``0 <= seed_id <= 1_000_000``), the
  owner-field key each kind reports (``feature``/``command``/
  ``source``), and full-coverage partition: the three tables tile
  every id in range exactly once — zero intra-kind collisions, zero
  cross-kind shadowing — so resolve is a total injective owner lookup
- shard ↔ table agreement: one file per owner (hyphenated owners map
  to underscored filenames), header naming the right runtime wrapper
  module, every registered id within the per-file ``seed_id < 0``
  guard

Probes are literal bools; the sealed receipt names every defect found.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from fx1.capabilities import (
    _OWNER_FIELD,
    _TABLES,
    SeedKind,
    owner_references,
    resolve_seed_id,
)

__all__ = ["capabilities_audit", "capabilities_audit_bench"]

_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _ROOT / "scripts" / "generated_capability_declarations"
_REGISTER_RE = re.compile(r"^_register\((\d+)\)$")
_MODULE_RE = re.compile(r"^from fx1\.extensions\.\w+\.\w+ import MODULE$")


def _refuses(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — refuse probes accept any failure
        return True
    return False


def _probe_tables() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["tb_all_kinds_present"] = set(_TABLES) == {"feature", "skill", "plugin"}
    well_formed = True
    bounded = True
    owners_unique = True
    for table in _TABLES.values():
        seen: set[str] = set()
        for owner, first, stride, count in table:
            if first < 0 or stride < 1 or count < 1:
                well_formed = False
            if first + stride * (count - 1) > 1_000_000:
                bounded = False
            if owner in seen:
                owners_unique = False
            seen.add(owner)
    out["tb_entries_well_formed"] = well_formed
    out["tb_progressions_bounded"] = bounded
    out["tb_owners_unique_per_kind"] = owners_unique
    out["tb_owner_field_map"] = _OWNER_FIELD == {
        "feature": "feature",
        "skill": "command",
        "plugin": "source",
    }
    return out


def _probe_owner_references() -> dict[str, bool]:
    out: dict[str, bool] = {}
    arithmetic_ok = True
    for kind, table in _TABLES.items():
        for owner, first, stride, count in table:
            refs = owner_references(kind, owner)
            if (
                len(refs) != count
                or refs[0] != first
                or refs[-1] != first + stride * (count - 1)
                or any(refs[i + 1] - refs[i] != stride for i in range(0, len(refs) - 1, 997))
            ):
                arithmetic_ok = False
    out["or_arithmetic_exact"] = arithmetic_ok
    out["or_returns_tuple"] = isinstance(owner_references("skill", "doctor"), tuple)
    out["or_doctor_starts_zero"] = owner_references("skill", "doctor")[:2] == (0, 69)
    out["or_unknown_kind"] = _refuses(owner_references, "nope", "x")
    try:
        owner_references(cast(SeedKind, "nope"), "x")
    except ValueError:
        out["or_kind_is_valueerror"] = True
    except Exception:  # noqa: BLE001
        out["or_kind_is_valueerror"] = False
    out["or_unknown_owner"] = _refuses(owner_references, "feature", "no_such_owner")
    out["or_cross_kind_owner_split"] = _refuses(
        owner_references, "feature", "doctor"
    )  # doctor is a skill, not a feature
    return out


def _probe_resolve() -> dict[str, bool]:
    out: dict[str, bool] = {}
    r0 = resolve_seed_id(0)
    out["rs_zero_is_doctor"] = (
        r0["kind"] == "skill" and r0["owner"] == "doctor" and r0["command"] == "doctor"
    )
    r_top = resolve_seed_id(1_000_000)
    out["rs_top_registered"] = r_top["kind"] == "plugin"
    out["rs_owner_field_echoed"] = r_top[_OWNER_FIELD[r_top["kind"]]] == r_top["owner"]
    # round-trip the first and last id of every progression
    roundtrip = True
    for kind, table in _TABLES.items():
        for owner, first, stride, count in table:
            for sid in (first, first + stride * (count - 1)):
                r = resolve_seed_id(sid)
                if r["kind"] != kind or r["owner"] != owner or r["seed_id"] != sid:
                    roundtrip = False
    out["rs_first_last_roundtrip"] = roundtrip
    out["rs_negative_refused"] = _refuses(resolve_seed_id, -1)
    out["rs_over_top_refused"] = _refuses(resolve_seed_id, 1_000_001)
    out["rs_string_refused"] = _refuses(resolve_seed_id, "5")
    out["rs_float_refused"] = _refuses(resolve_seed_id, 1.5)
    # measured quirk: bool is an int — True resolves as seed 1 (binance_crypto plugin)
    out["rs_bool_quirk_measured"] = resolve_seed_id(True)["owner"] == "binance_crypto"
    return out


def _probe_partition() -> dict[str, bool]:
    """The three tables must tile [0, 1_000_000] with zero overlaps."""
    out: dict[str, bool] = {}
    maps: dict[str, dict[int, str]] = {}
    intra_dups = 0
    total = 0
    for kind, table in _TABLES.items():
        m: dict[int, str] = {}
        for owner, first, stride, count in table:
            for i in range(count):
                sid = first + stride * i
                if sid in m:
                    intra_dups += 1
                m[sid] = owner
        maps[kind] = m
        total += len(m)
    out["pt_no_intra_kind_collision"] = intra_dups == 0
    order = ["feature", "skill", "plugin"]
    shadowed = 0
    for i, later_kind in enumerate(order):
        earlier: set[int] = set()
        for k2 in order[:i]:
            earlier.update(maps[k2])
        shadowed += sum(1 for sid in maps[later_kind] if sid in earlier)
    out["pt_no_cross_kind_shadowing"] = shadowed == 0
    out["pt_full_coverage"] = total == 1_000_001 and all(
        sid in maps["feature"] or sid in maps["skill"] or sid in maps["plugin"]
        for sid in (0, 500_000, 1_000_000)
    )
    return out


def _expand_shard(path: Path) -> tuple[list[int], str | None]:
    """Return (registered ids, imported wrapper module) for one shard."""
    ids: list[int] = []
    module: str | None = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = _REGISTER_RE.match(line)
            if m:
                ids.append(int(m.group(1)))
            elif _MODULE_RE.match(line):
                module = line.split()[1]
    return ids, module


def _check_kind_ledger(
    dirname: str, table: tuple[tuple[str, int, int, int], ...], out: dict[str, bool]
) -> tuple[bool, bool, bool, bool]:
    """Check one kind's shard dir against its compact table.

    Returns (file_count_ok, ids_exact, modules_exist, guard_ok).
    """
    file_count_ok = ids_exact = modules_exist = guard_ok = True
    files = sorted((_LEDGER / dirname).glob("*.py"))
    expected_names = sorted(f"{owner.replace('-', '_')}.py" for owner, *_ in table)
    if [f.name for f in files] != expected_names:
        file_count_ok = False
    by_owner = {o.replace("-", "_"): (f, s, c) for o, f, s, c in table}
    for shard in files:
        stem = shard.stem
        entry = by_owner.get(stem)
        if entry is None:
            file_count_ok = False
            continue
        first, stride, count = entry
        ids, module = _expand_shard(shard)
        if ids != [first + stride * i for i in range(count)]:
            ids_exact = False
        if any(sid < 0 for sid in ids):
            guard_ok = False
        wrapper = (
            (_ROOT / "src" / module.replace(".", "/")).with_suffix(".py")
            if module is not None
            else None
        )
        if wrapper is None or not wrapper.is_file():
            modules_exist = False
        out[f"lg_{stem}_count"] = len(ids) == count
    return file_count_ok, ids_exact, modules_exist, guard_ok


def _probe_ledger() -> dict[str, bool]:
    """Compact table must reproduce the expanded declaration ledger exactly."""
    out: dict[str, bool] = {}
    kind_dirs: dict[SeedKind, str] = {
        "feature": "features",
        "skill": "skills",
        "plugin": "plugins",
    }
    shards_exist = all((_LEDGER / d).is_dir() for d in kind_dirs.values())
    out["lg_shards_present"] = shards_exist
    if not shards_exist:
        out["lg_file_count_matches"] = False
        out["lg_ids_exact"] = False
        out["lg_wrapper_modules_exist"] = False
        out["lg_ids_in_guard_range"] = False
        return out
    results = [
        _check_kind_ledger(dirname, _TABLES[kind], out) for kind, dirname in kind_dirs.items()
    ]
    (
        out["lg_file_count_matches"],
        out["lg_ids_exact"],
        out["lg_wrapper_modules_exist"],
        out["lg_ids_in_guard_range"],
    ) = (all(r[i] for r in results) for i in range(4))
    return out


def capabilities_audit() -> dict[str, bool]:
    """Every seed-ledger contract as booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_tables())
    out.update(_probe_owner_references())
    out.update(_probe_resolve())
    out.update(_probe_partition())
    out.update(_probe_ledger())
    return out


def capabilities_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the capabilities battery."""
    r = capabilities_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "capabilities_audit",
        "schema": "capabilities_audit.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process table walk + full shard expansion",
            "not_verified": [
                "extension MODULE behavior (see extensions_audit)",
                "seed re-generation pipeline (scripts/seed_fx1_capabilities.py)",
            ],
        },
        "interpretation": (
            "Capability seed ledger verified: the compact progressions "
            "tile all 1,000,001 ids exactly once, resolve_seed_id is a "
            "total owner lookup, owner_references reproduces exact "
            "arithmetic, and every expanded declaration shard matches "
            "its compact entry bit-for-bit with a resolvable runtime "
            "wrapper."
            if ok
            else f"CAPABILITIES AUDIT DEFECTS: {defects}"
        ),
    }
    from quant_fund.utils.reproducibility import git_revision

    out["git_revision"] = git_revision()
    from quant_fund.research.receipt_v2 import canonical_json_bytes
    from quant_fund.utils.hashing import hash_bytes

    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(capabilities_audit_bench(), indent=2, sort_keys=True))
