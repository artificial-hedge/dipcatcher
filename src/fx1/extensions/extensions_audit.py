"""extensions_audit — fx-1 extension machinery + seed ledger battery.

The `extensions` tree is generated-card machinery: 86 first-party
modules (45 features, 23 skills, 18 plugins) each exporting a
``MODULE`` extension bound to a reviewed surface. This battery pins:

- *naming* — owner regex, dash→underscore mapping, kind dispatch and
  fail-closed unknown kinds.
- *contracts* — ``ExtensionModule`` frozen post-init checks (module
  path must equal ``module_path(kind, owner)``, nonempty references),
  ``card_count``/``manifest()`` shape, bounded ``records()`` paging
  with foreign-record refusal, ``verify()`` owner enforcement, and the
  per-kind subclass invariants (skill→command, plugin→source,
  feature→feature metadata).
- *feature_catalog* — 45 reviewed definitions, unique names, every
  name backed by an on-disk module file, ``synthetic_only`` iff family
  ``synthetic_oracle``, unknown names fail closed.
- *capabilities* (the root seed ledger this tree resolves through) —
  the three arithmetic progressions: ``owner_references`` returns
  ``count`` ordered ids per owner, ``resolve_seed_id`` reverses every
  owner's first and last seed, bounds are [0, 1_000_000], and unmapped
  ids fail closed.
- *registry* — ``_owners`` per kind matches the backing registries,
  ``list_extensions`` enumerates all 86 with the v1 manifest shape,
  ``get_extension`` loads only registered identities, and
  ``extension_manifest`` layers the concrete binding per kind.
- *conformance census* — every on-disk module file exports a ``MODULE``
  whose (kind, owner, module) matches its registered identity.

Probes are literal bools; the sealed receipt names every defect found.
"""

from __future__ import annotations

import json
import re
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from fx1 import capabilities as caps
from fx1.extensions import contracts, feature_catalog, naming, registry
from quant_fund.research.receipt_v2 import canonical_json_bytes
from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_revision

_RET1_MODULE = "fx1.extensions.features.ret_1"

__all__ = ["extensions_audit", "extensions_audit_bench"]

_SKILLS_N = 23
_PLUGINS_N = 18
_FEATURES_N = 45


def _refuses(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — refuse probes accept any failure
        return True
    return False


def _probe_naming() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["nam_plain"] = naming.module_basename("doctor") == "doctor"
    out["nam_dash_to_us"] = naming.module_basename("book-panel") == "book_panel"
    out["nam_digit_ok"] = naming.module_basename("ret_1") == "ret_1"
    out["nam_hyphen_tail"] = naming.module_basename("verify-research") == ("verify_research")
    out["nam_upper"] = _refuses(naming.module_basename, "Doctor")
    out["nam_empty"] = _refuses(naming.module_basename, "")
    out["nam_dot"] = _refuses(naming.module_basename, "a.b")
    out["nam_space"] = _refuses(naming.module_basename, "a b")
    out["nam_slash"] = _refuses(naming.module_basename, "a/b")
    out["nam_lead_digit"] = _refuses(naming.module_basename, "1a")
    out["nam_lead_us"] = _refuses(naming.module_basename, "_a")
    out["nam_lead_dash"] = _refuses(naming.module_basename, "-a")
    out["nam_path_skill"] = naming.module_path("skill", "book-panel") == (
        "fx1.extensions.skills.book_panel"
    )
    out["nam_path_plugin"] = naming.module_path("plugin", "imf") == ("fx1.extensions.plugins.imf")
    out["nam_path_feature"] = naming.module_path("feature", "ret_1") == (_RET1_MODULE)
    out["nam_path_bad_kind"] = _refuses(naming.module_path, "bogus", "x")
    out["nam_dir_skill"] = naming.module_directory("skill") == "skills"
    out["nam_dir_plugin"] = naming.module_directory("plugin") == "plugins"
    out["nam_dir_feature"] = naming.module_directory("feature") == "features"
    out["nam_dir_bad_kind"] = _refuses(naming.module_directory, "bogus")
    return out


def _probe_contracts() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ext = registry.get_extension("feature", "ret_1")
    out["ct_type_feature"] = isinstance(ext, contracts.FeatureExtension)
    out["ct_card_count"] = ext.card_count == 7408
    out["ct_manifest"] = (
        ext.manifest()["schema"] == "fx1.extension-module/v1"
        and ext.manifest()["market_evidence"] is False
        and ext.manifest()["card_count"] == 7408
    )
    out["ct_frozen"] = _refuses(setattr, ext, "owner", "other")
    out["ct_wrong_module"] = _refuses(
        contracts.FeatureExtension,
        kind="feature",
        owner="ret_1",
        references=(2,),
        module="fx1.extensions.features.ret_5",
    )
    out["ct_empty_refs"] = _refuses(
        contracts.FeatureExtension,
        kind="feature",
        owner="ret_1",
        references=(),
        module=_RET1_MODULE,
    )
    out["ct_kind_enforced"] = _refuses(
        contracts.SkillExtension,
        kind="feature",
        owner="doctor",
        references=(0,),
        module="fx1.extensions.skills.doctor",
    )
    out["ct_base_owner_abstract"] = _refuses(
        contracts.ExtensionModule(
            kind="feature",
            owner="ret_1",
            references=(2,),
            module=_RET1_MODULE,
        ).verify
    )
    # records(): bounded paging + foreign-record refusal
    recs = list(ext.records(limit=2))
    out["ct_records_page"] = (
        len(recs) == 2
        and recs[0]["seed_id"] == 2
        and recs[0]["feature"] == "ret_1"
        and recs[0]["kind"] == "feature"
    )
    recs_p2 = list(ext.records(offset=1, limit=1))
    out["ct_records_offset"] = recs_p2[0]["seed_id"] == 2 + 135
    out["ct_records_offset_neg"] = _refuses(lambda: list(ext.records(offset=-1)))
    out["ct_records_limit_zero"] = _refuses(lambda: list(ext.records(limit=0)))
    out["ct_records_limit_over"] = _refuses(lambda: list(ext.records(limit=101)))
    foreign = contracts.FeatureExtension(
        kind="feature",
        owner="ret_1",
        references=(0,),  # seed 0 belongs to skill 'doctor'
        module=_RET1_MODULE,
    )
    out["ct_records_foreign"] = _refuses(lambda: list(foreign.records()))
    out["ct_verify_foreign"] = _refuses(foreign.verify)
    out["ct_verify_clean"] = not _refuses(registry.get_extension("skill", "doctor").verify)
    # concrete bindings
    feat = contracts.FeatureExtension(
        kind="feature",
        owner="ret_1",
        references=(2,),
        module=_RET1_MODULE,
    )
    out["ct_feature_metadata"] = feat.metadata().name == "ret_1"
    skill = cast(contracts.SkillExtension, registry.get_extension("skill", "doctor"))
    out["ct_skill_command"] = skill.command().name == "doctor"
    plug = cast(contracts.PluginExtension, registry.get_extension("plugin", "imf"))
    out["ct_plugin_adapter"] = plug.adapter().spec.name == "imf"
    avail = plug.probe()
    out["ct_plugin_probe_honest"] = (
        isinstance(avail.status, str)
        and avail.status in {"ready", "no_script", "missing_credentials", "unavailable"}
        and isinstance(avail.credentials, list)
    )
    return out


def _probe_feature_catalog() -> dict[str, bool]:
    out: dict[str, bool] = {}
    defs = feature_catalog.list_feature_definitions()
    names = [d.name for d in defs]
    out["fc_count"] = len(defs) == _FEATURES_N
    out["fc_unique"] = len(set(names)) == _FEATURES_N
    out["fc_order_stable"] = names[0] == "ret_1" and names[-1] == ("cs_z_planted_signal")
    out["fc_pit_default"] = all(d.point_in_time_safe for d in defs)
    synth = {d.name for d in defs if d.synthetic_only}
    out["fc_synthetic_only_oracle"] = synth == {"planted_signal", "cs_z_planted_signal"} and all(
        d.synthetic_only == (d.family == "synthetic_oracle") for d in defs
    )
    out["fc_lookbacks"] = (
        feature_catalog.get_feature_definition("mom_252").lookback == 252
        and feature_catalog.get_feature_definition("ret_1").lookback == 1
        and feature_catalog.get_feature_definition("planted_signal").lookback == 0
    )
    out["fc_unknown"] = _refuses(feature_catalog.get_feature_definition, "bogus_feature")
    out["fc_immutable_type"] = isinstance(defs, tuple)
    out["fc_def_frozen"] = _refuses(setattr, defs[0], "name", "bogus")
    # every catalog name is backed by an on-disk module file
    feats_dir = Path(__file__).resolve().parent / "features"
    stems = {p.stem for p in feats_dir.glob("*.py")} - {"__init__"}
    out["fc_disk_backing"] = stems == set(names)
    return out


def _resolve_probes() -> dict[str, bool]:
    """resolve_seed_id entry surface: bounds + per-kind exemplars."""
    out: dict[str, bool] = {}
    r = caps.resolve_seed_id(0)
    out["cap_resolve_zero"] = (
        r["kind"] == "skill" and r["owner"] == "doctor" and r["command"] == "doctor"
    )
    r2 = caps.resolve_seed_id(2)
    out["cap_resolve_feature"] = (
        r2["kind"] == "feature"
        and r2["owner"] == "ret_1"
        and r2["feature"] == "ret_1"
        and r2["seed_id"] == 2
    )
    r1 = caps.resolve_seed_id(1)
    out["cap_resolve_plugin"] = (
        r1["kind"] == "plugin"
        and r1["owner"] == "binance_crypto"
        and r1["source"] == "binance_crypto"
    )
    out["cap_neg"] = _refuses(caps.resolve_seed_id, -1)
    out["cap_over"] = _refuses(caps.resolve_seed_id, 1_000_001)
    return out


def _owner_roundtrip_ok() -> bool:
    """Every owner's first and last seed resolve back to that owner."""
    for kind, table in (
        ("feature", caps._FEATURES),  # noqa: SLF001
        ("skill", caps._SKILLS),  # noqa: SLF001
        ("plugin", caps._PLUGINS),  # noqa: SLF001
    ):
        for owner, first, stride, count in table:
            for seed in (first, first + stride * (count - 1)):
                entry = caps.resolve_seed_id(seed)
                if entry["owner"] != owner or entry["kind"] != kind:
                    return False
    return True


def _probe_capability_ledger() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["cap_tables"] = (
        len(caps._FEATURES) == _FEATURES_N  # noqa: SLF001
        and len(caps._SKILLS) == _SKILLS_N  # noqa: SLF001
        and len(caps._PLUGINS) == _PLUGINS_N  # noqa: SLF001
    )
    refs = caps.owner_references("feature", "ret_1")
    out["cap_refs_shape"] = len(refs) == 7408 and refs[0] == 2 and refs[-1] == 2 + 135 * 7407
    out["cap_refs_ordered"] = all(b - a == 135 for a, b in zip(refs[:49], refs[1:50], strict=True))
    out["cap_unknown_owner"] = _refuses(caps.owner_references, "feature", "bogus")
    out["cap_unknown_kind"] = _refuses(caps.owner_references, "bogus", "ret_1")
    out.update(_resolve_probes())
    # the layout is total: strides are all ≡0 mod 3 and first seeds
    # partition the residues, so every in-range id resolves to exactly
    # one kind — skills ≡0, plugins ≡1, features ≡2
    residue_kind = {0: "skill", 1: "plugin", 2: "feature"}
    out["cap_total_layout"] = all(
        caps.resolve_seed_id(seed)["kind"] == residue_kind[seed % 3] for seed in range(0, 999, 7)
    )
    out["cap_roundtrip_all"] = _owner_roundtrip_ok()
    # no seed may resolve to a different owner's field key
    out["cap_owner_field"] = all(
        "command" in caps.resolve_seed_id(first)
        for _, first, _, _ in caps._SKILLS  # noqa: SLF001
    ) and all(
        "source" in caps.resolve_seed_id(first)
        for _, first, _, _ in caps._PLUGINS  # noqa: SLF001
    )
    return out


def _probe_registry() -> dict[str, bool]:
    out: dict[str, bool] = {}
    owners = (
        registry._owners("skill"),  # noqa: SLF001
        registry._owners("plugin"),  # noqa: SLF001
        registry._owners("feature"),  # noqa: SLF001
    )
    out["rg_owners_counts"] = (
        len(owners[0]) == _SKILLS_N
        and len(owners[1]) == _PLUGINS_N
        and len(owners[2]) == _FEATURES_N
    )
    out["rg_owners_bad_kind"] = _refuses(registry._owners, "bogus")  # noqa: SLF001
    listed = registry.list_extensions()
    out["rg_list_all"] = len(listed) == 86
    out["rg_list_shape"] = all(
        e["schema"] == "fx1.extension-module/v1"
        and e["market_evidence"] is False
        and re.fullmatch(
            r"fx1\.extensions\.(skills|plugins|features)\.[a-z0-9_]+",
            str(e["module"]),
        )
        for e in listed
    )
    feats_only = registry.list_extensions("feature")
    out["rg_list_kind_filter"] = len(feats_only) == _FEATURES_N and all(
        e["kind"] == "feature" for e in feats_only
    )
    plugs_only = registry.list_extensions("plugin")
    out["rg_list_plugin_filter"] = len(plugs_only) == _PLUGINS_N
    out["rg_get_unknown"] = _refuses(registry.get_extension, "feature", "bogus")
    out["rg_get_wrong_kind"] = _refuses(registry.get_extension, "skill", "ret_1")
    ext = registry.get_extension("feature", "ret_1")
    out["rg_get_identity"] = (
        ext.kind == "feature" and ext.owner == "ret_1" and ext.module == _RET1_MODULE
    )
    mf = registry.extension_manifest("feature", "ret_1")
    out["rg_manifest_feature"] = (
        mf["feature"] == "ret_1"
        and mf["feature_family"] == "returns"
        and mf["lookback"] == 1
        and mf["point_in_time_safe"] is True
        and mf["synthetic_only"] is False
        and mf["operations"] == ["metadata", "build"]
    )
    ms = registry.extension_manifest("skill", "doctor")
    out["rg_manifest_skill"] = (
        ms["command"] == "doctor"
        and "role" in ms
        and "description" in ms
        and ms["operations"] == ["run"]
    )
    mp = registry.extension_manifest("plugin", "imf")
    avail_dict = mp["availability"]
    out["rg_manifest_plugin"] = (
        mp["source"] == "imf"
        and isinstance(mp["markets"], list)
        and isinstance(mp["assets"], list)
        and isinstance(avail_dict, dict)
        and isinstance(avail_dict["status"], str)
        and mp["operations"] == ["probe", "describe", "fetch"]
    )
    return out


def _probe_module_conformance() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ext_root = Path(__file__).resolve().parent
    expected: list[tuple[naming.ExtensionKind, str, str]] = []
    kind_of: dict[str, naming.ExtensionKind] = {
        "features": "feature",
        "plugins": "plugin",
        "skills": "skill",
    }
    for subdir, kind in kind_of.items():
        for p in sorted((ext_root / subdir).glob("*.py")):
            if p.stem == "__init__":
                continue
            owner = registry._owners(kind)  # noqa: SLF001
            owner_stem = next(
                (o for o in owner if o.replace("-", "_") == p.stem),
                None,
            )
            expected.append((kind, owner_stem or p.stem, p.stem))
    out["mc_file_count"] = len(expected) == 86
    loaded = 0
    identity_ok = True
    for kind, owner_name, stem in expected:
        mod = import_module(f"fx1.extensions.{subdir_of(kind)}.{stem}")
        m = getattr(mod, "MODULE", None)
        if (
            isinstance(m, contracts.ExtensionModule)
            and m.kind == kind
            and m.owner == owner_name
            and m.module == f"fx1.extensions.{subdir_of(kind)}.{stem}"
        ):
            loaded += 1
        else:
            identity_ok = False
    out["mc_all_load"] = loaded == 86
    out["mc_identity"] = identity_ok
    # every module verifies its own registered references
    verif = all(
        not _refuses(import_module(f"fx1.extensions.{subdir_of(k)}.{s}").MODULE.verify)
        for k, _, s in expected[:9]  # sample: first 9 stems
    )
    out["mc_verify_sample"] = verif
    # card counts sum to the ledger total
    total = sum(len(caps.owner_references(k, o)) for k, o, _ in expected)
    out["mc_card_total"] = total == sum(
        entry[3]  # noqa: SLF001
        for table in (caps._FEATURES, caps._SKILLS, caps._PLUGINS)  # noqa: SLF001
        for entry in table
    )
    return out


def subdir_of(kind: naming.ExtensionKind) -> str:
    return naming.module_directory(kind)


def extensions_audit() -> dict[str, bool]:
    """Every extension-machinery contract as booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_naming())
    out.update(_probe_contracts())
    out.update(_probe_feature_catalog())
    out.update(_probe_capability_ledger())
    out.update(_probe_registry())
    out.update(_probe_module_conformance())
    return out


def extensions_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the extensions battery."""
    r = extensions_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "extensions_audit",
        "schema": "extensions_audit.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process imports + contracts; no adapter network I/O",
            "not_verified": [
                "plugin describe()/fetch() live calls (credential-gated by design)",
                "skill run()/build() end-to-end (harness command surface covered by serve audits)",
            ],
        },
        "interpretation": (
            "Extension machinery holds: owner names are namespaced and "
            "safe, every module binds its declared kind/owner/path, "
            "records paging is bounded with foreign-record refusal, the "
            "feature catalog stays reviewed-only with synthetic_oracle "
            "honesty, the seed ledger round-trips every owner, and the "
            "registry resolves exactly the 86 generated modules with "
            "manifest-level binding metadata."
            if ok
            else f"EXTENSIONS AUDIT DEFECTS: {defects}"
        ),
    }
    out["git_revision"] = git_revision()
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(extensions_audit_bench(), indent=2, sort_keys=True))
