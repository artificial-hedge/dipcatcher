"""Catalog loading, argument validation, argv building, and cache behavior."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive.atomicstore import atomic_write_json
from fx1.interactive.contracts import (
    SCHEMA_ID,
    CapabilityCatalog,
    CatalogCache,
    CatalogUnavailable,
    ContractError,
    ContractViolation,
    build_argv,
    catalog_digest,
    fetch_catalog,
    parse_cli_tokens,
    validate_args,
)
from tests.fx1.capabilities_fixture import (
    capability_entry,
    sample_catalog,
    sample_catalog_document,
)


def _cap(catalog: CapabilityCatalog, cap_id: str) -> Any:
    found = catalog.get(cap_id)
    assert found is not None
    return found


def test_fixture_loads_and_digest_is_stable() -> None:
    doc = sample_catalog_document()
    first = CapabilityCatalog.from_json(doc)
    second = CapabilityCatalog.from_json(json.dumps(doc, indent=2))
    reversed_doc = {key: doc[key] for key in reversed(list(doc))}
    third = CapabilityCatalog.from_json(json.dumps(reversed_doc))
    assert first.digest == second.digest == third.digest
    assert len(first.digest) == 64
    assert first.digest == catalog_digest(first.document)
    changed = sample_catalog_document()
    changed["capabilities"][0]["purpose"] += " Extra."
    assert CapabilityCatalog.from_json(changed).digest != first.digest


def test_unknown_optional_fields_are_ignored() -> None:
    doc = sample_catalog_document()
    doc["comment"] = "extension"
    doc["capabilities"][0]["future_hook"] = {"enabled": True}
    doc["capabilities"][0]["command"]["extra"] = True
    catalog = CapabilityCatalog.from_json(doc)
    doctor = _cap(catalog, "doctor")
    assert doctor.id == "doctor"
    assert not hasattr(doctor, "future_hook")
    assert catalog.digest == catalog_digest(doc)


def test_executable_list_is_sorted_and_closed() -> None:
    catalog = sample_catalog()
    ids = [cap.id for cap in catalog.executable()]
    assert ids == sorted(ids)
    assert "train" not in ids
    assert "alpha-decay" not in ids
    assert "vol-bench" in ids
    assert catalog.get("missing") is None


@pytest.mark.parametrize(
    "schema",
    ["dipcatcher.capabilities.v2", "dipcatcher.capabilities.v0", "other", None, 1],
)
def test_wrong_schema_is_rejected(schema: object) -> None:
    doc = sample_catalog_document()
    if schema is None:
        del doc["schema"]
    else:
        doc["schema"] = schema
    with pytest.raises(ContractError, match="schema"):
        CapabilityCatalog.from_json(doc)


def test_schema_constant_matches_fixture() -> None:
    assert sample_catalog_document()["schema"] == SCHEMA_ID


@pytest.mark.parametrize(
    "mutator",
    [
        lambda doc: doc["capabilities"][0].pop("purpose"),
        lambda doc: doc.pop("generated_by"),
        lambda doc: doc.pop("capabilities"),
        lambda doc: doc["capabilities"][0].pop("command"),
        lambda doc: doc["capabilities"][0]["inputs"].pop("requires_dataset"),
        lambda doc: doc["capabilities"][0]["outputs"].pop("scores"),
        lambda doc: doc["capabilities"][0]["limits"].pop("max_runtime_s"),
        lambda doc: doc["capabilities"][0]["qualification"].pop("status"),
        lambda doc: doc["capabilities"][0].pop("executable"),
        lambda doc: doc["capabilities"][0].pop("tasks"),
        lambda doc: doc["capabilities"][0].pop("post_checks"),
        lambda doc: doc["generated_by"].__setitem__("overlay_sha256", "abcd"),
        lambda doc: doc["generated_by"].__setitem__("n_lab_commands", True),
        lambda doc: doc.__setitem__("capabilities", {}),
    ],
)
def test_missing_or_mistyped_required_fields(mutator: Any) -> None:
    doc = sample_catalog_document()
    mutator(doc)
    with pytest.raises(ContractError):
        CapabilityCatalog.from_json(doc)


@pytest.mark.parametrize(
    ("cap_id", "status"),
    [
        ("doctor", "unqualified"),
        ("doctor", "experimental"),
        ("doctor", "retired"),
        ("alpha-decay", "retired"),
    ],
)
def test_executable_requires_qualified_status(cap_id: str, status: str) -> None:
    doc = sample_catalog_document()
    entry = capability_entry(doc, cap_id)
    entry["executable"] = True
    entry["qualification"]["status"] = status
    entry["qualification"]["evidence"] = ["tests/fx1/capabilities_fixture.py"]
    with pytest.raises(ContractError, match="executable"):
        CapabilityCatalog.from_json(doc)


def test_executable_without_evidence_is_rejected() -> None:
    doc = sample_catalog_document()
    entry = capability_entry(doc, "doctor")
    entry["qualification"]["evidence"] = []
    with pytest.raises(ContractError, match="evidence"):
        CapabilityCatalog.from_json(doc)


def test_duplicate_ids_and_bad_dependencies_are_rejected() -> None:
    doc = sample_catalog_document()
    doc["capabilities"].append(json.loads(json.dumps(doc["capabilities"][0])))
    with pytest.raises(ContractError, match="duplicate"):
        CapabilityCatalog.from_json(doc)
    again = sample_catalog_document()
    capability_entry(again, "doctor")["dependencies"] = ["missing-cap"]
    with pytest.raises(ContractError, match="unknown id"):
        CapabilityCatalog.from_json(again)
    cycled = sample_catalog_document()
    capability_entry(cycled, "doctor")["dependencies"] = ["doctor"]
    with pytest.raises(ContractError, match="itself"):
        CapabilityCatalog.from_json(cycled)


def test_non_executable_without_reason_is_rejected() -> None:
    doc = sample_catalog_document()
    capability_entry(doc, "train")["executable_reason"] = "  "
    with pytest.raises(ContractError, match="executable_reason"):
        CapabilityCatalog.from_json(doc)


def test_invalid_json_text_and_non_object() -> None:
    with pytest.raises(ContractError, match="JSON"):
        CapabilityCatalog.from_json("{")
    with pytest.raises(ContractError, match="object"):
        CapabilityCatalog.from_json("[]")


def test_absolute_path_default_is_rejected() -> None:
    doc = sample_catalog_document()
    capability_entry(doc, "doctor")["args"].append(
        {
            "default": "/tmp/out",
            "help": "Output.",
            "kind": "option",
            "name": "--out",
            "path_role": "output_dir",
            "required": False,
            "type": "path",
        }
    )
    with pytest.raises(ContractError, match="default"):
        CapabilityCatalog.from_json(doc)


def test_widening_choices_is_rejected() -> None:
    doc = sample_catalog_document()
    capability_entry(doc, "vol-bench")["allowed_args"]["--models"]["choices"] = ["har", "nope"]
    with pytest.raises(ContractError, match="choices"):
        CapabilityCatalog.from_json(doc)


def _violation(cap: Any, args: dict[str, Any], **kwargs: Any) -> ContractViolation:
    with pytest.raises(ContractViolation) as caught:
        validate_args(cap, args, **kwargs)
    return caught.value


def test_flag_int_float_enum_and_str_validation() -> None:
    catalog = sample_catalog()
    doctor = _cap(catalog, "doctor")
    vol = _cap(catalog, "vol-bench")
    calibration = _cap(catalog, "calibration-eval")
    optimize = _cap(catalog, "optimize")
    assert validate_args(doctor, {"--strict": True})["--strict"] is True
    assert _violation(doctor, {"--strict": "true"}).code == "bad_type"
    assert validate_args(vol, {"--n-origins": 8})["--n-origins"] == 8
    assert _violation(vol, {"--n-origins": True}).code == "bad_type"
    assert _violation(vol, {"--n-origins": 1.5}).code == "bad_type"
    assert _violation(vol, {"--n-origins": "3"}).code == "bad_type"
    assert _violation(vol, {"--n-origins": 9}).code == "out_of_range"
    assert _violation(vol, {"--n-origins": 0}).code == "out_of_range"
    assert validate_args(calibration, {"--bandwidth": 0.25})["--bandwidth"] == 0.25
    assert _violation(calibration, {"--bandwidth": True}).code == "bad_type"
    assert _violation(calibration, {"--bandwidth": float("inf")}).code == "bad_type"
    assert _violation(calibration, {"--bandwidth": 3.0}).code == "out_of_range"
    assert validate_args(calibration, {"--method": "kernel"})["--method"] == "kernel"
    assert _violation(calibration, {"--method": "spline"}).code == "bad_choice"
    long_text = "a" * 257
    assert _violation(optimize, {"--study": long_text}).code == "too_long"
    assert _violation(optimize, {"--study": "ok;rm"}).code == "bad_type"
    assert _violation(optimize, {"--study": "bad\\path"}).code == "bad_type"
    assert _violation(optimize, {"--study": "line\nbreak"}).code == "bad_type"
    assert validate_args(optimize, {"--study": "lab-study"})["--study"] == "lab-study"


def test_csv_choices_tightening_and_length() -> None:
    vol = _cap(sample_catalog(), "vol-bench")
    parsed = validate_args(vol, {"--horizons": "1, 5", "--models": ["har", "garch"]})
    assert parsed["--horizons"] == [1, 5]
    assert parsed["--models"] == ["har", "garch"]
    assert _violation(vol, {"--models": "ewma"}).code == "bad_choice"
    assert _violation(vol, {"--horizons": "1,,5"}).code == "bad_type"
    assert _violation(vol, {"--horizons": "1,x"}).code == "bad_type"
    too_many = ",".join(str(index) for index in range(1, 34))
    assert _violation(vol, {"--horizons": too_many}).code == "too_long"
    assert _violation(vol, {"--models": "har;rm"}).code == "bad_type"


def test_looser_numeric_tightening_cannot_widen_the_spec() -> None:
    doc = sample_catalog_document()
    capability_entry(doc, "vol-bench")["allowed_args"]["--n-origins"]["max"] = 100
    cap = _cap(CapabilityCatalog.from_json(doc), "vol-bench")
    assert validate_args(cap, {"--n-origins": 40})["--n-origins"] == 40
    assert _violation(cap, {"--n-origins": 60}).code == "out_of_range"


def test_whitelist_rejects_args_outside_allowed_args() -> None:
    doc = sample_catalog_document()
    capability_entry(doc, "vol-bench")["allowed_args"] = ["--horizons", "--models"]
    cap = _cap(CapabilityCatalog.from_json(doc), "vol-bench")
    violation = _violation(cap, {"--n-origins": 2})
    assert violation.code == "unknown_arg"
    assert violation.arg == "--n-origins"
    parsed = validate_args(cap, {})
    assert "--n-origins" not in parsed
    assert parsed["--models"] == ["har"]


def test_unknown_and_missing_and_not_executable() -> None:
    catalog = sample_catalog()
    vol = _cap(catalog, "vol-bench")
    forecast = _cap(catalog, "forecast-eval")
    train = _cap(catalog, "train")
    unknown = _violation(vol, {"--nope": 1})
    assert unknown.code == "unknown_arg"
    missing = _violation(forecast, {}, roots={"input": [Path("/tmp")]})
    assert missing.code == "missing_required"
    assert missing.arg == "--dataset"
    blocked = _violation(train, {"--epochs": 1})
    assert blocked.code == "not_executable"
    assert "qualification" in str(blocked)


def test_path_roots_symlink_and_escape(tmp_path: Path) -> None:
    catalog = sample_catalog()
    notes = _cap(catalog, "verify-research")
    forecast = _cap(catalog, "forecast-eval")
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text("nope", encoding="utf-8")
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    ok = validate_args(notes, {"--note-file": "ok.csv"}, roots={"input": [root]})
    assert Path(ok["--note-file"]) == (root / "ok.csv").resolve()
    escaped = _violation(notes, {"--note-file": str(link / "secret.csv")}, roots={"any": [root]})
    assert escaped.code == "path_escape"
    assert (
        _violation(notes, {"--note-file": "../secret.csv"}, roots={"input": [root]}).code
        == "path_escape"
    )
    assert (
        _violation(notes, {"--note-file": "/etc/passwd"}, roots={"input": [root]}).code
        == "path_escape"
    )
    assert (
        _violation(notes, {"--note-file": "bad\x00.csv"}, roots={"input": [root]}).code
        == "path_escape"
    )
    windows_name = validate_args(notes, {"--note-file": "foo\\bar"}, roots={"input": [root]})
    assert Path(windows_name["--note-file"]).is_relative_to(root.resolve())
    placed = root / "mine.csv"
    accepted = validate_args(forecast, {"--dataset": str(placed)}, roots={"input": [root]})
    assert accepted["--dataset"] == str(placed.resolve())
    wrong_role = _violation(forecast, {"--dataset": str(placed)}, roots={"output_dir": [root]})
    assert wrong_role.code == "path_escape"


def test_parse_cli_tokens_match_validate_args() -> None:
    catalog = sample_catalog()
    vol = _cap(catalog, "vol-bench")
    parsed = parse_cli_tokens(
        vol, ["--horizons", "1,5", "--models=garch", "--n-origins", "3", "--verbose=true"]
    )
    direct = validate_args(
        vol, {"--horizons": "1,5", "--models": "garch", "--n-origins": 3, "--verbose": True}
    )
    assert parsed == direct
    assert parse_cli_tokens(vol, ["--horizons=1,5"])["--horizons"] == [1, 5]
    negative = _violation_cli(vol, ["--n-origins", "-1"])
    assert negative.code == "out_of_range"
    missing = _violation_cli(vol, ["--n-origins"])
    assert missing.code == "missing_required"
    unknown = _violation_cli(vol, ["--nope", "1"])
    assert unknown.code == "unknown_arg"
    duplicate = _violation_cli(vol, ["--models", "har", "--models", "garch"])
    assert duplicate.code == "unknown_arg"


def _violation_cli(cap: Any, tokens: list[str]) -> ContractViolation:
    with pytest.raises(ContractViolation) as caught:
        parse_cli_tokens(cap, tokens)
    return caught.value


def test_positional_repeatable_and_end_of_options() -> None:
    catalog = sample_catalog()
    doctor = _cap(catalog, "doctor")
    verify = _cap(catalog, "verify-research")
    parsed = parse_cli_tokens(doctor, ["--strict", "lab"])
    assert parsed["--strict"] is True
    assert parsed["subject"] == "lab"
    assert parse_cli_tokens(doctor, ["--", "lab"])["subject"] == "lab"
    assert parse_cli_tokens(doctor, ["--strict=false"])["--strict"] is False
    extra = _violation_cli(doctor, ["lab", "more"])
    assert extra.code == "unknown_arg"
    tags = parse_cli_tokens(verify, ["--tag", "a", "--tag", "b"])
    assert tags["--tag"] == ["a", "b"]
    single = validate_args(verify, {"--tag": "alpha"})
    assert single["--tag"] == ["alpha"]
    too_many = _violation(verify, {"--tag": [f"t{index}" for index in range(33)]})
    assert too_many.code == "too_long"


def test_build_argv_is_deterministic_and_adds_receipt_flag() -> None:
    vol = _cap(sample_catalog(), "vol-bench")
    validated = validate_args(vol, {"--verbose": True, "--n-origins": 2, "--models": "har"})
    first = build_argv(vol, validated)
    second = build_argv(vol, {"--n-origins": 2, "--verbose": True})
    assert first == second
    assert first == [
        "vol-bench",
        "--horizons",
        "1,5",
        "--models",
        "har",
        "--n-origins",
        "2",
        "--verbose",
        "--receipt-version",
        "2",
    ]
    quiet = build_argv(vol, {"--verbose": False})
    assert "--verbose" not in quiet
    assert quiet[-2:] == ["--receipt-version", "2"]
    doctor = _cap(sample_catalog(), "doctor")
    assert build_argv(doctor, {}) == ["doctor"]
    assert build_argv(doctor, {"subject": "lab", "--strict": True}) == ["doctor", "--strict", "lab"]
    verify = _cap(sample_catalog(), "verify-research")
    tagged = build_argv(verify, {"--tag": ["a", "b"]})
    assert tagged == ["verify-research", "--tag", "a", "--tag", "b"]
    for token in first:
        assert not any(char in token for char in ";|&`$()<>\\\n")


def test_receipt_flag_is_not_duplicated_and_respects_writes() -> None:
    doc = sample_catalog_document()
    entry = capability_entry(doc, "calibration-eval")
    entry["outputs"]["receipt_flag"] = ["--bins", "10"]
    cap = _cap(CapabilityCatalog.from_json(doc), "calibration-eval")
    argv = build_argv(cap, {})
    assert argv.count("--bins") == 1
    muted = sample_catalog_document()
    capability_entry(muted, "vol-bench")["command"]["writes"] = []
    capability_entry(muted, "vol-bench")["outputs"]["receipt_flag"] = []
    capability_entry(muted, "vol-bench")["outputs"]["receipt_schema"] = ""
    capability_entry(muted, "vol-bench")["outputs"]["receipt_stdout_marker"] = ""
    capability_entry(muted, "vol-bench")["post_checks"] = []
    plain = _cap(CapabilityCatalog.from_json(muted), "vol-bench")
    assert "--receipt-version" not in build_argv(plain, {})


def test_shell_metacharacters_in_catalog_tokens_are_rejected() -> None:
    doc = sample_catalog_document()
    capability_entry(doc, "doctor")["command"]["argv"] = ["doctor;rm"]
    with pytest.raises(ContractError, match="shell"):
        CapabilityCatalog.from_json(doc)


def test_fetch_catalog_success_and_failures() -> None:
    doc = sample_catalog_document()
    seen: list[tuple[str, list[str]]] = []

    def ok(name: str, args: list[str]) -> tuple[int, str]:
        seen.append((name, args))
        return 0, json.dumps(doc)

    catalog = fetch_catalog(ok)
    assert seen == [("capabilities", ["list", "--json"])]
    assert catalog.digest == sample_catalog().digest
    assert isinstance(fetch_catalog(ok), CapabilityCatalog)

    def boom(name: str, args: list[str]) -> tuple[int, str]:
        raise RuntimeError("bearer sk-abcdefghijklmnopqrstuvwxyz1234")

    with pytest.raises(CatalogUnavailable, match="REDACTED") as caught:
        fetch_catalog(boom)
    assert isinstance(caught.value, ContractError)
    assert "sk-" not in str(caught.value)

    def bad_exit(name: str, args: list[str]) -> tuple[int, str]:
        return 2, json.dumps(doc)

    with pytest.raises(CatalogUnavailable, match="exited 2"):
        fetch_catalog(bad_exit)

    def bad_json(name: str, args: list[str]) -> tuple[int, str]:
        return 0, "{not-json"

    with pytest.raises(CatalogUnavailable, match="JSON"):
        fetch_catalog(bad_json)

    def bad_schema(name: str, args: list[str]) -> tuple[int, str]:
        payload = sample_catalog_document()
        payload["schema"] = "dipcatcher.capabilities.v2"
        return 0, json.dumps(payload)

    with pytest.raises(CatalogUnavailable, match="schema"):
        fetch_catalog(bad_schema)

    def bad_shape(name: str, args: list[str]) -> tuple[int, str]:
        return 0, b"{}"  # type: ignore[return-value]

    with pytest.raises(CatalogUnavailable, match="non-text"):
        fetch_catalog(bad_shape)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def test_cache_ttl_hit_expiry_and_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    clock = _Clock()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    def runner(name: str, args: list[str]) -> tuple[int, str]:
        calls["n"] += 1
        return 0, json.dumps(sample_catalog_document())

    cache = CatalogCache(runner, config_dir=tmp_path / "cfg", clock=clock, ttl_s=300)
    first = cache.get()
    second = cache.get()
    assert calls["n"] == 1
    assert first.digest == second.digest
    assert cache.path().is_file()
    if os.name == "posix":
        assert stat.S_IMODE(cache.path().stat().st_mode) == 0o600
    clock.now += 300
    third = cache.get()
    assert calls["n"] == 2
    assert third.digest == first.digest
    stored = json.loads(cache.path().read_text(encoding="utf-8"))
    assert stored["fetched_at"] == clock.now
    assert not (home / ".fx1").exists()


def test_cache_uses_fx1_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path))
    calls = {"n": 0}

    def runner(name: str, args: list[str]) -> tuple[int, str]:
        calls["n"] += 1
        return 0, json.dumps(sample_catalog_document())

    cache = CatalogCache(runner, clock=_Clock(), ttl_s=300)
    cache.get()
    cache.get()
    assert calls["n"] == 1
    assert (tmp_path / "cache" / "capabilities.json").is_file()


def test_cache_refetches_on_mismatch_corruption_and_failure(tmp_path: Path) -> None:
    docs = {"current": sample_catalog_document()}
    calls = {"n": 0}
    clock = _Clock()

    def runner(name: str, args: list[str]) -> tuple[int, str]:
        calls["n"] += 1
        return 0, json.dumps(docs["current"])

    cache = CatalogCache(runner, config_dir=tmp_path, clock=clock, ttl_s=300, version="0.4.0")
    original = cache.get()
    payload = json.loads(cache.path().read_text(encoding="utf-8"))
    payload["digest"] = "0" * 64
    atomic_write_json(cache.path(), payload)
    assert cache.get().digest == original.digest
    assert calls["n"] == 2

    payload = json.loads(cache.path().read_text(encoding="utf-8"))
    payload["fx1_version"] = "9.9.9"
    atomic_write_json(cache.path(), payload)
    assert cache.get().digest == original.digest
    assert calls["n"] == 3

    cache.path().write_text("{", encoding="utf-8")
    assert cache.get().digest == original.digest
    assert calls["n"] == 4

    payload = json.loads(cache.path().read_text(encoding="utf-8"))
    payload["document"]["schema"] = "dipcatcher.capabilities.v2"
    payload["digest"] = catalog_digest(payload["document"])
    atomic_write_json(cache.path(), payload)
    clock.now += 1
    assert cache.get().digest == original.digest
    assert calls["n"] == 5

    updated = sample_catalog_document()
    updated["capabilities"][0]["purpose"] = "Check environment health and the manifest again."
    docs["current"] = updated
    clock.now += 500
    refreshed = cache.get()
    assert refreshed.digest == CapabilityCatalog.from_json(updated).digest
    assert refreshed.digest != original.digest
    stored = json.loads(cache.path().read_text(encoding="utf-8"))
    assert stored["digest"] == refreshed.digest

    clock.now += 500

    def fail(name: str, args: list[str]) -> tuple[int, str]:
        return 1, ""

    failing = CatalogCache(fail, config_dir=tmp_path, clock=clock, ttl_s=300, version="0.4.0")
    with pytest.raises(CatalogUnavailable):
        failing.get()


def test_negative_ttl_is_rejected(tmp_path: Path) -> None:
    def runner(name: str, args: list[str]) -> tuple[int, str]:
        return 0, "{}"

    with pytest.raises(ValueError):
        CatalogCache(runner, config_dir=tmp_path, ttl_s=-1)
