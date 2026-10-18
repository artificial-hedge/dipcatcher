"""Adversarial probes of the evidence pipeline: seals, quarantine, tapes, epoch chains.

Every case drives the production functions — ``verify_receipt_bytes`` /
``verify_receipt_file`` / ``verify_receipt_payload``, ``load_quarantine`` /
``classify_receipt`` / ``suite_health``, ``tape_binding_errors``,
``check_epoch_chain`` / ``load_epoch_acks`` / ``update_heads_pin`` — and the
real CLI through ``typer.testing.CliRunner``, over scratch corpora under
``tmp_path``. Nothing under test is mocked, monkeypatched or re-implemented.

Each test names the invariant it pins. Negative controls MUST fail closed and
positive controls MUST pass on the same fixture, so a probe that silently
stops reaching the verifier cannot stay green.

Regression tests for six production defects this suite exposed (all fixed in
the same change): an overflowing number literal (``1e400``) sealing like
``null``; receipt.v2 envelopes skipping the e-value-family contracts (which also
let a body renamed to ``suite_health`` dodge the tape ratchet); non-string tape
bindings read as "unbound"; hostile files (deep nesting, huge integers, lone
surrogates) crashing the corpus sweep, chain check, lattice and admission
instead of being reported; and duplicate JSON keys in the quarantine registry
and epoch ledger resolving last-wins.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import unicodedata
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner, Result

import quant_fund.research.suite_health as suite_health_mod
from quant_fund.cli.main import app
from quant_fund.research.admission import admission_check
from quant_fund.research.corpus_epoch import (
    EpochStampLocked,
    _acquire_stamp_lock,
    check_epoch_chain,
    corpus_epoch,
    epoch_contract_errors,
    epoch_heads_key,
    load_epoch_acks,
    load_heads_pin,
    member_digests,
    update_heads_pin,
    write_epoch_receipt,
)
from quant_fund.research.legacy_unsealed import (
    EMPTY_QUARANTINE,
    Quarantine,
    classify_receipt,
    load_quarantine,
)
from quant_fund.research.receipt_v2 import (
    seal_receipt,
    verify_receipt_bytes,
    verify_receipt_file,
    verify_receipt_payload,
    wrap_receipt_v2,
)
from quant_fund.research.suite_health import suite_health
from quant_fund.research.tape_registry import (
    TAPE_MANIFEST_SCHEMA,
    load_registry,
    pin_tape,
    tape_binding_errors,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Hypothesis budget: modest, deterministic-enough, no on-disk example DB.
_PROPERTY = settings(
    max_examples=60,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_SEAL_ERRORS = frozenset({"receipt_sha256_mismatch", "receipt_sha256_missing_or_invalid"})


# --- shared fixtures ----------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _body(**extra: Any) -> dict[str, Any]:
    """A SYNTHETIC v1 body that carries the identity fields the repair cared about."""
    body: dict[str, Any] = {
        "schema": "adversarial_fixture.v1",
        "kind": "adversarial_fixture",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "git_revision": "0123456789abcdef0123456789abcdef01234567",
        "params": {"seed": 7, "taus": [0.1, 0.5, 0.9]},
        "results": [{"head": "gaussian", "pinball": 0.125, "crps": 0.25}],
        "note": None,
    }
    body.update(extra)
    return body


def _v1(**extra: Any) -> dict[str, Any]:
    return seal_receipt(_body(**extra))


@cache
def _v2_frozen() -> str:
    envelope = seal_receipt(
        wrap_receipt_v2(_body(), code_files=(Path(suite_health_mod.__file__),), verdict="pass")
    )
    return json.dumps(envelope)


def _v2() -> dict[str, Any]:
    """A sealed receipt.v2 envelope (built once; a fresh deep copy per call)."""
    loaded: dict[str, Any] = json.loads(_v2_frozen())
    return loaded


def _reseal(payload: dict[str, Any]) -> dict[str, Any]:
    """An honest fresh seal over edited content — consistency, never provenance."""
    return seal_receipt({k: v for k, v in payload.items() if k != "receipt_sha256"})


def _bytes(payload: object, **dumps: Any) -> bytes:
    return json.dumps(payload, **dumps).encode("utf-8")


def _verdict(data: bytes) -> tuple[bool, list[str], str | None]:
    result = verify_receipt_bytes(data)
    return bool(result["valid"]), list(result["errors"]), result["digest_convention"]


def _invoke(args: Sequence[str]) -> Result:
    return CliRunner().invoke(app, [str(a) for a in args])


def _sealed_file(path: Path, **extra: Any) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(_v1(**extra), indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return data


def test_fixture_positive_controls() -> None:
    """Positive control: the fixtures every negative control mutates verify clean."""
    for payload in (_v1(), _v2()):
        assert verify_receipt_payload(payload)["valid"], verify_receipt_payload(payload)["errors"]
    assert verify_receipt_payload(_v1())["digest_convention"] == "canonical_json"
    assert verify_receipt_payload(_v2())["schema"] == "receipt.v2"


# --- 1. altered fields --------------------------------------------------------

_JSON_LEAF = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**18), max_value=10**18)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=12)
)
_JSON = st.recursive(
    _JSON_LEAF,
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=6), children, max_size=3)
    ),
    max_leaves=6,
)


def _mutate(
    sealed: dict[str, Any], op: str, pick: int, new_key: str, value: object
) -> dict[str, Any] | None:
    """One top-level add / remove / value change; ``None`` when it would be a no-op."""
    keys = sorted(sealed)
    mutated = dict(sealed)
    if op == "add":
        if new_key in sealed:
            return None
        mutated[new_key] = value
        return mutated
    key = keys[pick % len(keys)]
    if op == "remove":
        del mutated[key]
        return mutated
    # A JSON-distinct value (1 vs 1.0 vs true, -0.0 vs 0.0 all count as changes).
    if canonical_json_bytes(value) == canonical_json_bytes(sealed[key]):
        return None
    mutated[key] = value
    return mutated


@_PROPERTY
@given(
    generation=st.sampled_from(["v1", "v2"]),
    op=st.sampled_from(["add", "remove", "change"]),
    pick=st.integers(min_value=0, max_value=10**6),
    new_key=st.text(min_size=1, max_size=10),
    value=_JSON,
)
def test_any_single_top_level_mutation_breaks_verification(
    generation: str, op: str, pick: int, new_key: str, value: object
) -> None:
    """Invariant: no single top-level field add/remove/value change survives the seal.

    Covers every key, ``receipt_sha256`` itself included, on both the v1 body
    and the receipt.v2 envelope; the mutated bytes go through the byte-level
    production entry point."""
    sealed = _v1() if generation == "v1" else _v2()
    mutated = _mutate(sealed, op, pick, new_key, value)
    assume(mutated is not None)
    valid, errors, _ = _verdict(_bytes(mutated))
    assert not valid
    assert errors
    # The seal check fired, or the edit broke the envelope schema first, or a
    # wrongly-typed dispatch field (e.g. ``kind: []``) failed closed as undigestable.
    assert any(
        e in _SEAL_ERRORS or e.startswith(("receipt_v2_schema:", "receipt_undigestable:"))
        for e in errors
    ), errors


@pytest.mark.parametrize("generation", ["v1", "v2"])
@pytest.mark.parametrize("field", ["schema", "git_revision", "data_label"])
def test_identity_fields_are_sealed(generation: str, field: str) -> None:
    """Invariant: ``schema``, ``git_revision`` and ``data_label`` are inside the seal.

    Negative controls: edit, delete, or insert-into-a-body-lacking-it — all
    fail. The 87ff0c06 / 677b01a3 edits (schema inserted, git_revision swapped)
    are exactly these mutations. Resealing makes the bytes self-consistent
    again, which is why the epoch chain, not the seal, owns provenance."""
    sealed = _v1() if generation == "v1" else _v2()
    edited = dict(sealed)
    edited[field] = f"{sealed[field]}-edited"
    assert not verify_receipt_payload(edited)["valid"]
    removed = {k: v for k, v in sealed.items() if k != field}
    assert not verify_receipt_payload(removed)["valid"]
    if generation == "v1":
        lacking = seal_receipt({k: v for k, v in _body().items() if k != field})
        assert verify_receipt_payload(lacking)["valid"]  # positive control
        inserted = {**lacking, field: sealed[field]}
        assert not verify_receipt_payload(inserted)["valid"]
    if field == "git_revision":
        # Threat model: an honest reseal over a swapped revision verifies.
        assert verify_receipt_payload(_reseal(edited))["valid"]


def _shuffled(value: object, rnd: random.Random) -> object:
    """Randomize object key order recursively; list order is content, never moved."""
    if isinstance(value, dict):
        items = list(value.items())
        rnd.shuffle(items)
        return {key: _shuffled(item, rnd) for key, item in items}
    if isinstance(value, list):
        return [_shuffled(item, rnd) for item in value]
    return value


@_PROPERTY
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    indent=st.sampled_from([None, 0, 1, 2, 4, "\t"]),
    separators=st.sampled_from([None, (",", ":"), (", ", ": "), (" , ", " : ")]),
    ensure_ascii=st.booleans(),
    padding=st.sampled_from(["", "\n", "\r\n", " \t\n"]),
)
def test_formatting_never_changes_the_verdict(
    seed: int,
    indent: int | str | None,
    separators: tuple[str, str] | None,
    ensure_ascii: bool,
    padding: str,
) -> None:
    """Invariant: key order, indentation, separators, escaping and surrounding
    whitespace are encoding, not content — the verdict (valid, errors, digest
    convention) is identical to the compact canonical bytes', for a valid v1,
    a valid v2 and a tampered v1 alike (deterministic canonicalization)."""
    rnd = random.Random(seed)
    tampered = {**_v1(note="sealed"), "note": "edited after sealing"}
    for payload in (_v1(note="unicodé ✓"), _v2(), tampered):
        baseline = _verdict(canonical_json_bytes(payload))
        text = json.dumps(
            _shuffled(payload, rnd),
            indent=indent,
            separators=separators,
            ensure_ascii=ensure_ascii,
        )
        assert _verdict((padding + text + padding).encode("utf-8")) == baseline


def test_unicode_seal_conventions_are_exactly_the_documented_two() -> None:
    """Invariant: a non-ASCII body verifies under the canonical (``ensure_ascii=False``)
    and the strict (``ensure_ascii=True``) seal conventions — each reported by
    name — whatever escaping the file uses; any other serialization's digest
    is a mismatch, and NFC/NFD-different text is different content."""
    body = _body(note='café — ✓ 漢字 "quoted" \\ back sep')
    canonical = seal_receipt(body)
    strict_seal = _sha(_bytes(body, sort_keys=True, separators=(",", ":"), allow_nan=False))
    strict = {**body, "receipt_sha256": strict_seal}
    assert canonical["receipt_sha256"] != strict_seal  # the conventions really differ here
    for payload, convention in ((canonical, "canonical_json"), (strict, "strict_json")):
        for ensure_ascii in (True, False):
            valid, errors, got = _verdict(_bytes(payload, ensure_ascii=ensure_ascii))
            assert (valid, errors, got) == (True, [], convention)
    for dumps in (
        {"sort_keys": True},  # default separators
        {"separators": (",", ":"), "ensure_ascii": False},  # insertion order
        {"sort_keys": True, "indent": 2},
        {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False, "indent": 1},
    ):
        forged = {**body, "receipt_sha256": _sha(_bytes(body, **dumps))}
        assert verify_receipt_payload(forged)["errors"] == ["receipt_sha256_mismatch"]
    # JSON escapes are transparent: c IS "c".
    plain = seal_receipt(_body(note="cafe"))
    raw = _bytes(plain, sort_keys=True)
    escaped = raw.replace(b'"note": "cafe"', b'"note": "\\u0063afe"')
    assert escaped != raw and _verdict(escaped)[0]
    # Canonically-equivalent but distinct code points are distinct content.
    nfc = seal_receipt(_body(note=unicodedata.normalize("NFC", "café")))
    nfd = {**nfc, "note": unicodedata.normalize("NFD", "café")}
    assert verify_receipt_payload(nfc)["valid"]
    assert verify_receipt_payload(nfd)["errors"] == ["receipt_sha256_mismatch"]


@pytest.mark.parametrize(
    ("variant", "key"),
    [
        ("same_value", "kind"),
        ("shadowed_first", "kind"),
        ("nested", "seed"),
        ("duplicate_seal", "receipt_sha256"),
    ],
)
def test_duplicate_json_keys_are_rejected(tmp_path: Path, variant: str, key: str) -> None:
    """Invariant: the bytes must determine one payload — any duplicate object key
    (top level, nested, or the seal itself) is rejected, even when Python's
    last-wins parse would reproduce exactly the sealed payload."""
    sealed = _v1()
    raw = canonical_json_bytes(sealed)
    assert _verdict(raw)[0]  # positive control
    if variant == "same_value":
        forged = raw.replace(b"{", b'{"kind":"adversarial_fixture",', 1)
    elif variant == "shadowed_first":
        forged = raw.replace(b"{", b'{"kind":"a_claim_nobody_sealed",', 1)
    elif variant == "nested":
        forged = raw.replace(b'"params":{', b'"params":{"seed":99,', 1)
    else:
        forged = raw.replace(b"{", b'{"receipt_sha256":"' + b"0" * 64 + b'",', 1)
    assert forged != raw
    valid, errors, _ = _verdict(forged)
    assert not valid and errors == [f"duplicate_json_key:{key}"]
    path = tmp_path / "dup.json"
    path.write_bytes(forged)
    assert verify_receipt_file(path)["errors"] == [f"duplicate_json_key:{key}"]


_OVERFLOW_LITERALS = [b"1e400", b"-1e400", b"1E999"]


def _null_swapped_for(literal: bytes) -> tuple[bytes, bytes]:
    raw = _bytes(_v1(note=None), sort_keys=True)
    return raw, raw.replace(b'"note": null', b'"note": ' + literal)


@pytest.mark.parametrize("literal", _OVERFLOW_LITERALS)
def test_overflow_literal_swap_changes_the_payload(literal: bytes) -> None:
    """Control for the regression test below: the swap really changes content (``null`` →
    ±inf), the original verifies, and the spelled-out ``Infinity`` token —
    the same in-memory value — is already rejected."""
    raw, forged = _null_swapped_for(literal)
    assert _verdict(raw)[0]
    assert forged != raw
    assert json.loads(forged)["note"] in (float("inf"), float("-inf"))
    spelled = raw.replace(b'"note": null', b'"note": Infinity')
    assert _verdict(spelled)[1] == ["receipt_unreadable:ValueError"]


@pytest.mark.parametrize("literal", _OVERFLOW_LITERALS)
def test_overflowing_number_literal_cannot_alias_a_sealed_null(literal: bytes) -> None:
    """Invariant: a seal binds one payload. ``null`` and a literal that decodes to
    ±inf are different payloads, so swapping one for the other must not verify."""
    _raw, forged = _null_swapped_for(literal)
    assert not _verdict(forged)[0]


# --- 2. malformed receipts ----------------------------------------------------

_MALFORMED: dict[str, bytes] = {
    "array": b"[]",
    "string": b'"receipt"',
    "number": b"42",
    "null": b"null",
    "true": b"true",
    "empty": b"",
    "whitespace": b"  \n\t",
    "truncated_object": b'{"kind": "x", "receipt_sha256": "ab',
    "trailing_garbage": b'{"kind": "x"} trailing',
    "two_documents": b"{}{}",
    "nan_literal": b'{"kind": "x", "v": NaN}',
    "infinity_literal": b'{"kind": "x", "v": Infinity}',
    "neg_infinity_literal": b'{"kind": "x", "v": -Infinity}',
    "nested_nan": b'{"kind": "x", "v": {"w": [1, NaN]}}',
    "invalid_utf8": b'{"kind": "\xff\xfe\xfa"}',
    "lone_surrogate_in_label": b'{"data_label": "\\ud800", "receipt_sha256": "' + b"a" * 64 + b'"}',
    "huge_int": b'{"kind": "x", "v": ' + b"9" * 5000 + b"}",
    "deep_array": b"[" * 100_000 + b"]" * 100_000,
    "deep_object": b'{"a":' * 50_000 + b"1" + b"}" * 50_000,
    "deep_but_parseable": b'{"receipt_sha256": "'
    + b"a" * 64
    + b'", "x": '
    + b"[" * 2_000
    + b"]" * 2_000
    + b"}",
}


@pytest.mark.parametrize("name", sorted(_MALFORMED))
def test_malformed_receipt_is_invalid_and_never_raises(tmp_path: Path, name: str) -> None:
    """Invariant: hostile bytes degrade to an ``invalid`` verdict with a reason
    through the byte and file verifier entry points — never an exception."""
    data = _MALFORMED[name]
    valid, errors, _ = _verdict(data)
    assert not valid and errors
    path = tmp_path / f"{name}.json"
    path.write_bytes(data)
    result = verify_receipt_file(path)
    assert not result["valid"] and result["errors"] == errors


@pytest.mark.parametrize("name", sorted(_MALFORMED))
def test_malformed_receipt_is_reported_invalid_by_suite_health(tmp_path: Path, name: str) -> None:
    """Invariant: in a corpus sweep every hostile file is counted ``invalid`` and
    named — one corrupt artifact must never abort the audit of the rest."""
    corpus = tmp_path / "corpus"
    _sealed_file(corpus / "good.json")
    (corpus / f"{name}.json").write_bytes(_MALFORMED[name])
    _frame, receipt = suite_health(corpus)
    assert (receipt["n_ok"], receipt["n_invalid"]) == (1, 1)
    assert receipt["invalid"][f"{name}.json"]


@pytest.mark.parametrize("gate", ["check_epoch_chain", "admission_check"])
def test_deeply_nested_file_does_not_crash_chain_or_admission(tmp_path: Path, gate: str) -> None:
    """Invariant: the chain check reports a hostile corpus file (it is just an
    unstamped member) and admission rejects a hostile candidate — neither raises."""
    corpus = tmp_path / "receipts"
    _sealed_file(corpus / "good.json")
    _stamp(corpus)
    if gate == "check_epoch_chain":
        (corpus / "hostile.json").write_bytes(_MALFORMED["deep_array"])
        assert check_epoch_chain(corpus)["unstamped"] == ["hostile.json"]
    else:
        candidate = tmp_path / "hostile.json"
        candidate.write_bytes(_MALFORMED["deep_array"])
        assert admission_check(candidate, corpus)["verdict"] == "reject"


def test_malformed_corpus_fails_strict_cli(tmp_path: Path) -> None:
    """Invariant: the strict CLI gate turns a corpus of malformed files into a
    clean exit 1 listing each file — not a traceback."""
    names = ("array", "nan_literal", "truncated_object", "invalid_utf8", "deep_but_parseable")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in names:
        (corpus / f"{name}.json").write_bytes(_MALFORMED[name])
    res = _invoke(
        [
            "suite-health",
            "--receipts-dir",
            corpus,
            "--out-dir",
            tmp_path / "out",
            "--strict",
            "--quarantine",
            tmp_path / "absent_registry.json",
        ]
    )
    assert res.exit_code == 1, res.output
    assert isinstance(res.exception, SystemExit)
    assert "STRICT FAILURE" in res.output
    for name in names:
        assert f"{name}.json" in res.output


_SURROGATE_KIND = b'{"kind": "\\ud800", "receipt_sha256": "' + b"a" * 64 + b'"}'


def test_lone_surrogate_kind_is_invalid_at_the_receipt_level() -> None:
    """Invariant: a JSON-escaped lone surrogate (valid JSON, unencodable as
    UTF-8) degrades to an invalid verdict in the verifier itself."""
    valid, errors, _ = _verdict(_SURROGATE_KIND)
    assert not valid and errors == ["receipt_sha256_mismatch"]


def test_lone_surrogate_kind_is_reported_invalid_by_suite_health(tmp_path: Path) -> None:
    """Invariant: a corrupt receipt shows up as ``n_invalid``, it never raises."""
    corpus = tmp_path / "receipts"
    _sealed_file(corpus / "good.json")
    (corpus / "hostile.json").write_bytes(_SURROGATE_KIND)
    _frame, receipt = suite_health(corpus)
    assert receipt["n_invalid"] == 1 and "hostile.json" in receipt["invalid"]


@pytest.mark.parametrize(
    "seal",
    [
        None,
        1,
        1.5,
        True,
        [],
        {"sha256": "a" * 64},
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        " " + "a" * 63,
    ],
    ids=repr,
)
def test_wrongly_typed_seal_is_invalid(seal: object) -> None:
    """Invariant: ``receipt_sha256`` must be 64 lowercase hex; any other type or
    shape is ``receipt_sha256_missing_or_invalid`` (absence too)."""
    body = _body()
    assert verify_receipt_payload({**body, "receipt_sha256": seal})["errors"] == [
        "receipt_sha256_missing_or_invalid"
    ]
    assert verify_receipt_payload(body)["errors"] == ["receipt_sha256_missing_or_invalid"]


def test_every_truncation_of_a_valid_receipt_is_invalid() -> None:
    """Invariant: a partially-written receipt (any strict prefix of valid bytes)
    never verifies and never raises."""
    data = canonical_json_bytes(_v1())
    assert _verdict(data)[0]  # positive control: the whole file verifies
    for cut in range(len(data)):
        valid, errors, _ = _verdict(data[:cut])
        assert not valid and errors, cut


@_PROPERTY
@given(data=st.binary(max_size=512))
def test_arbitrary_bytes_never_raise_or_verify(data: bytes) -> None:
    """Invariant: the byte-level verifier is total over arbitrary input."""
    result = verify_receipt_bytes(data)
    assert result["valid"] is False and result["errors"]


@_PROPERTY
@given(payload=_JSON)
def test_arbitrary_json_never_raises_or_verifies(payload: object) -> None:
    """Invariant: any parsed JSON value without a matching seal is rejected, never raises."""
    result = verify_receipt_payload(payload)
    assert result["valid"] is False and result["errors"]


# --- 3. unknown tapes ---------------------------------------------------------

#: Certainly unknown to the repo's committed tape registry (asserted below).
_UNKNOWN_DIGEST = _sha(b"evidence-pipeline-adversarial: no tape manifest attests this digest")


def _repo_registry() -> dict[str, dict[str, Any]]:
    registry = load_registry(REPO_ROOT)
    assert _UNKNOWN_DIGEST not in registry
    return registry


@pytest.mark.parametrize("binding", ["dataset_sha256", "tape_manifest_sha256"])
@pytest.mark.parametrize("generation", ["v1", "v2"])
def test_unknown_tape_binding_fails_closed_on_the_production_path(
    tmp_path: Path, binding: str, generation: str
) -> None:
    """Invariant: a non-SYNTHETIC receipt binding a digest no committed
    ``tape_manifest.v1`` attests is ``tape_manifest_unknown`` — on the real
    verifier path (repo-root registry), for v1 bodies and v2 inner payloads."""
    _repo_registry()
    body = _body(data_label="yahoo_eod", **{binding: _UNKNOWN_DIGEST})
    if generation == "v1":
        payload = seal_receipt(body)
    else:
        payload = seal_receipt(
            wrap_receipt_v2(body, code_files=(Path(suite_health_mod.__file__),), verdict="pass")
        )
    path = tmp_path / "bound.json"
    path.write_bytes(_bytes(payload, indent=2, sort_keys=True))
    result = verify_receipt_file(path)
    assert not result["valid"]
    assert "tape_manifest_unknown" in result["errors"]


def test_known_tape_binding_resolves_on_the_production_path() -> None:
    """Positive control: a digest the committed registry attests is not flagged."""
    registry = _repo_registry()
    if not registry:
        pytest.skip("no committed tape manifests in this checkout")
    known = sorted(registry)[0]
    payload = seal_receipt(_body(data_label="yahoo_eod", dataset_sha256=known))
    assert "tape_manifest_unknown" not in verify_receipt_payload(payload)["errors"]


@pytest.mark.parametrize(
    ("label", "exempt"),
    [
        ("SYNTHETIC", True),
        ("synthetic", True),
        ("CORPUS", True),
        ("yahoo_eod", False),
        ("SYNTHETIC ", False),
        ("SYNTHETIC_ISH", False),
        ("", False),
        (None, False),
        (["SYNTHETIC"], False),
    ],
    ids=repr,
)
def test_only_synthetic_labels_are_exempt(label: object, exempt: bool) -> None:
    """Invariant: the ratchet exempts exactly the SYNTHETIC/CORPUS labels; a
    missing, empty, padded, look-alike or non-string label is not exempt."""
    _repo_registry()
    body = {k: v for k, v in _body(dataset_sha256=_UNKNOWN_DIGEST).items() if k != "data_label"}
    if label is not None:
        body["data_label"] = label
    errors = verify_receipt_payload(seal_receipt(body))["errors"]
    assert ("tape_manifest_unknown" not in errors) is exempt, errors


def _tape_root(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "repo"
    tape = root / "data" / "raw" / "bars.parquet"
    tape.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "security_id": ["AAA", "AAA", "BBB"],
            "event_time": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "close": [101.0, 102.5, 7.0],
        }
    ).write_parquet(tape)
    pinned = pin_tape("adv_tape", tape, out_dir=root / "data" / "manifests", root=root)
    return root, pinned


def test_pinned_manifest_attests_and_a_broken_one_attests_nothing(tmp_path: Path) -> None:
    """Invariant: digests resolve only through a seal-valid ``tape_manifest.v1``
    under the given root. Positive control: a freshly pinned manifest attests
    its seal, tape bytes and CSV digest. Negative controls: the same digests
    under another root, after the manifest is edited without reseal, with its
    seal stripped, or relabelled to another schema (honestly resealed) — all
    ``tape_manifest_unknown``."""
    root, pinned = _tape_root(tmp_path)
    manifest = pinned["manifest"]
    digests = [
        manifest["receipt_sha256"],
        manifest["tape_files"][0]["sha256"],
        manifest["frame_csv_sha256"],
    ]
    bodies = [_body(data_label="yahoo_eod", dataset_sha256=d) for d in digests]
    for body in bodies:
        assert tape_binding_errors(body, root=root) == []
    elsewhere = tmp_path / "other_root"
    elsewhere.mkdir()
    for body in bodies:
        assert tape_binding_errors(body, root=elsewhere) == ["tape_manifest_unknown"]

    path: Path = pinned["path"]
    original = path.read_bytes()
    edited = {**manifest, "n_rows": manifest["n_rows"] + 1}
    unsealed = {k: v for k, v in manifest.items() if k != "receipt_sha256"}
    wrong_schema = dict(unsealed, schema="tape_manifest.v2")
    wrong_schema["receipt_sha256"] = hash_bytes(canonical_json_bytes(wrong_schema))
    for broken in (edited, unsealed, wrong_schema):
        path.write_bytes(_bytes(broken, indent=2, sort_keys=True))
        for body in bodies:
            assert tape_binding_errors(body, root=root) == ["tape_manifest_unknown"]
    path.write_bytes(b"{not json")
    assert tape_binding_errors(bodies[0], root=root) == ["tape_manifest_unknown"]
    path.write_bytes(original)
    assert tape_binding_errors(bodies[0], root=root) == []  # restored: attests again
    assert manifest["schema"] == TAPE_MANIFEST_SCHEMA


@pytest.mark.parametrize(
    "binding", [[_UNKNOWN_DIGEST], {"bars": _UNKNOWN_DIGEST}, 7], ids=["list", "dict", "int"]
)
def test_non_string_tape_binding_fails_closed(binding: object) -> None:
    """Invariant: a declared tape binding that no manifest attests fails closed,
    whatever JSON type the declaration uses."""
    _repo_registry()
    payload = seal_receipt(_body(data_label="yahoo_eod", dataset_sha256=binding))
    assert not verify_receipt_payload(payload)["valid"]


def test_kind_rename_cannot_launder_an_unknown_tape_inside_v2() -> None:
    """Invariant: renaming the sealed inner ``kind`` must not shed the tape ratchet
    (control: the un-renamed envelope is ``tape_manifest_unknown`` — see
    ``test_unknown_tape_binding_fails_closed_on_the_production_path``)."""
    _repo_registry()
    renamed = _body(data_label="yahoo_eod", dataset_sha256=_UNKNOWN_DIGEST, kind="suite_health")
    envelope = seal_receipt(
        wrap_receipt_v2(renamed, code_files=(Path(suite_health_mod.__file__),), verdict="pass")
    )
    assert not verify_receipt_payload(envelope)["valid"]


# --- 4. quarantine registry ---------------------------------------------------

_LEGACY = b'{"kind": "legacy_fixture", "note": "written before the seal convention"}\n'
_LEGACY_OTHER = b'{"kind": "legacy_fixture", "note": "same error set, different bytes"}\n'
_LEGACY_ERRORS = ["receipt_sha256_missing_or_invalid"]


def _entry(sha: str, errors: list[str] | None = None, **over: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "sha256": sha,
        "errors": list(_LEGACY_ERRORS if errors is None else errors),
        "category": "pre_seal_era",
        "reason": "adversarial fixture",
    }
    entry.update(over)
    return entry


def _registry(path: Path, files: dict[str, Any], **top: Any) -> Path:
    path.write_text(json.dumps({"schema": "legacy_quarantine.v2", "files": files, **top}))
    return path


def _health(corpus: Path, quarantine: Quarantine) -> tuple[dict[str, str], dict[str, Any]]:
    frame, receipt = suite_health(corpus, quarantine=quarantine)
    statuses = {row["file"]: row["status"] for row in frame.to_dicts()}
    return statuses, receipt


@pytest.fixture()
def legacy_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "receipts"
    _sealed_file(corpus / "good.json")
    (corpus / "legacy-unsealed").mkdir()
    (corpus / "legacy-unsealed" / "old.json").write_bytes(_LEGACY)
    assert sorted(verify_receipt_file(corpus / "legacy-unsealed" / "old.json")["errors"]) == (
        _LEGACY_ERRORS
    )
    return corpus


def test_exact_registry_entry_quarantines(legacy_corpus: Path, tmp_path: Path) -> None:
    """Positive control: path + bytes + exact error set → ``quarantined``,
    counted apart from verified, never invalid, never pooled."""
    reg = _registry(tmp_path / "q.json", {"legacy-unsealed/old.json": _entry(_sha(_LEGACY))})
    statuses, receipt = _health(legacy_corpus, load_quarantine(reg))
    assert statuses == {"good.json": "verified", "legacy-unsealed/old.json": "quarantined"}
    assert (receipt["n_ok"], receipt["n_quarantined"], receipt["n_invalid"]) == (1, 1, 0)
    assert receipt["invalid"] == {} and receipt["quarantine"]["stale_entries"] == []
    assert receipt["quarantine"]["by_category"] == {"pre_seal_era": 1}
    assert receipt["pooled_evalue"] is None  # withheld: quarantined is not verified


def test_no_registry_means_failures_are_invalid(legacy_corpus: Path) -> None:
    """Invariant: the empty registry (absent file) quarantines nothing."""
    statuses, receipt = _health(legacy_corpus, EMPTY_QUARANTINE)
    assert statuses["legacy-unsealed/old.json"] == "invalid"
    assert receipt["invalid"]["legacy-unsealed/old.json"][0] == "quarantine_unregistered"


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("bytes_drift", "quarantine_bytes_drift"),
        ("errors_superset", "quarantine_errors_drift"),
        ("errors_different", "quarantine_errors_drift"),
        ("entry_now_verifies", "quarantine_entry_verifies"),
        ("unregistered_valid_in_subtree", "quarantine_unregistered"),
        ("moved_within_subtree", "quarantine_unregistered"),
    ],
)
def test_registry_drift_fails_closed(
    legacy_corpus: Path, tmp_path: Path, scenario: str, expected: str
) -> None:
    """Invariant: anything short of an exact (path, bytes, error-set) match is
    ``invalid`` with the drift named — bytes swapped for others with the same
    error set, a widened or different error set, an entry whose file now
    verifies, any unregistered file under ``legacy-unsealed/`` (even a valid
    one), and a quarantined file moved to another path."""
    old = legacy_corpus / "legacy-unsealed" / "old.json"
    rel = "legacy-unsealed/old.json"
    files: dict[str, Any] = {rel: _entry(_sha(_LEGACY))}
    if scenario == "bytes_drift":
        old.write_bytes(_LEGACY_OTHER)
    elif scenario == "errors_superset":
        files[rel] = _entry(_sha(_LEGACY), ["forbidden_metric_keys", *_LEGACY_ERRORS])
    elif scenario == "errors_different":
        files[rel] = _entry(_sha(_LEGACY), ["receipt_sha256_mismatch"])
    elif scenario == "entry_now_verifies":
        data = _sealed_file(old)
        files[rel] = _entry(_sha(data))
    elif scenario == "unregistered_valid_in_subtree":
        _sealed_file(legacy_corpus / "legacy-unsealed" / "fresh.json")
        rel = "legacy-unsealed/fresh.json"
    else:
        old.rename(old.with_name("moved.json"))
        rel = "legacy-unsealed/moved.json"
    statuses, receipt = _health(
        legacy_corpus, load_quarantine(_registry(tmp_path / "q.json", files))
    )
    assert statuses[rel] == "invalid"
    assert receipt["invalid"][rel][0] == expected
    assert receipt["n_invalid"] == len(receipt["invalid"]) >= 1
    if scenario == "unregistered_valid_in_subtree":
        # Control: the registered sibling still quarantines exactly.
        assert statuses["legacy-unsealed/old.json"] == "quarantined"
    else:
        assert receipt["n_quarantined"] == 0
    if scenario == "moved_within_subtree":
        assert receipt["quarantine"]["stale_entries"] == [
            "quarantine_entry_missing:'legacy-unsealed/old.json'"
        ]


@pytest.mark.parametrize(
    "key",
    ["./legacy-unsealed/old.json", "legacy-unsealed/./old.json", "legacy-unsealed//old.json"],
)
def test_noncanonical_registry_path_never_quarantines(
    legacy_corpus: Path, tmp_path: Path, key: str
) -> None:
    """Invariant: an entry keyed by a non-canonical spelling of the path fails
    closed — the loader rejects it, or it matches nothing and the file it
    names stays ``invalid``; it never quarantines."""
    reg = _registry(tmp_path / "q.json", {key: _entry(_sha(_LEGACY))})
    try:
        quarantine = load_quarantine(reg)
    except ValueError:
        return  # rejected at load: fail closed
    statuses, receipt = _health(legacy_corpus, quarantine)
    assert statuses["legacy-unsealed/old.json"] == "invalid"
    assert receipt["n_quarantined"] == 0


def test_stale_entry_fails_the_strict_gate(legacy_corpus: Path, tmp_path: Path) -> None:
    """Invariant: a registry entry whose file is gone is stale and fails
    ``suite-health --strict`` even when every remaining receipt is fine."""
    reg = _registry(
        tmp_path / "q.json",
        {
            "legacy-unsealed/old.json": _entry(_sha(_LEGACY)),
            "legacy-unsealed/gone.json": _entry("ab" * 32),
        },
    )
    args = ["suite-health", "--receipts-dir", legacy_corpus, "--out-dir", tmp_path / "out"]
    res = _invoke([*args, "--strict", "--quarantine", reg])
    assert res.exit_code == 1, res.output
    assert "quarantine_entry_missing:'legacy-unsealed/gone.json'" in res.output
    # Positive control: drop the stale entry and the same corpus passes.
    _registry(reg, {"legacy-unsealed/old.json": _entry(_sha(_LEGACY))})
    res = _invoke([*args, "--strict", "--quarantine", reg])
    assert res.exit_code == 0, res.output
    assert "verified=1 quarantined=1 invalid=0" in res.output


@pytest.mark.parametrize(
    ("verified", "file_sha", "errors", "expected"),
    [
        (False, "L", _LEGACY_ERRORS, ("quarantined", [])),
        (False, "L", [*_LEGACY_ERRORS, *_LEGACY_ERRORS], ("quarantined", [])),
        (True, "L", [], ("invalid", ["quarantine_entry_verifies"])),
        (False, "X", _LEGACY_ERRORS, ("invalid", ["quarantine_bytes_drift", *_LEGACY_ERRORS])),
        (False, "L", ["other"], ("invalid", ["quarantine_errors_drift", "other"])),
    ],
)
def test_classify_receipt_truth_table(
    tmp_path: Path,
    verified: bool,
    file_sha: str,
    errors: list[str],
    expected: tuple[str, list[str]],
) -> None:
    """Invariant: the classifier's verdict per (validity, bytes, error set);
    duplicate verifier errors de-duplicate before the exact-set comparison."""
    registry = load_quarantine(_registry(tmp_path / "q.json", {"x.json": _entry(_sha(_LEGACY))}))
    sha = _sha(_LEGACY) if file_sha == "L" else _sha(_LEGACY_OTHER)
    got = classify_receipt("x.json", sha, valid=verified, errors=errors, quarantine=registry)
    assert got == expected


_GOOD = _entry("ab" * 32)
_MALFORMED_REGISTRIES: dict[str, bytes] = {
    "not_json": b"{not json",
    "top_level_array": b"[]",
    "schema_v1": _bytes({"schema": "legacy_quarantine.v1", "files": {}}),
    "schema_missing": _bytes({"files": {}}),
    "unknown_top_level_key": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {}, "approved_by": "x"}
    ),
    "files_not_object": _bytes({"schema": "legacy_quarantine.v2", "files": []}),
    "unknown_category": _bytes(
        {
            "schema": "legacy_quarantine.v2",
            "files": {"a.json": {**_GOOD, "category": "grandfathered"}},
        }
    ),
    "unsorted_errors": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "errors": ["z", "a"]}}}
    ),
    "duplicate_errors": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "errors": ["a", "a"]}}}
    ),
    "empty_errors": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "errors": []}}}
    ),
    "non_string_error": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "errors": [1]}}}
    ),
    "extra_entry_key": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "approved": True}}}
    ),
    "missing_entry_key": _bytes(
        {
            "schema": "legacy_quarantine.v2",
            "files": {"a.json": {k: v for k, v in _GOOD.items() if k != "reason"}},
        }
    ),
    "blank_reason": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "reason": "  "}}}
    ),
    "sha_uppercase": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "sha256": "AB" * 32}}}
    ),
    "sha_short": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"a.json": {**_GOOD, "sha256": "ab" * 31}}}
    ),
    "path_traversal": _bytes({"schema": "legacy_quarantine.v2", "files": {"../x.json": _GOOD}}),
    "nested_traversal": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"legacy-unsealed/../../x.json": _GOOD}}
    ),
    "absolute_path": _bytes({"schema": "legacy_quarantine.v2", "files": {"/etc/x.json": _GOOD}}),
    "backslash_path": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {"legacy-unsealed\\x.json": _GOOD}}
    ),
    "empty_path": _bytes({"schema": "legacy_quarantine.v2", "files": {"": _GOOD}}),
    "drift_ack_single_member": _bytes(
        {
            "schema": "legacy_quarantine.v2",
            "files": {},
            "schema_drift_acks": {"t.v1": {"members": {"a.json": "ab" * 32}, "reason": "x"}},
        }
    ),
    "drift_ack_bad_digest": _bytes(
        {
            "schema": "legacy_quarantine.v2",
            "files": {},
            "schema_drift_acks": {
                "t.v1": {"members": {"a.json": "ab" * 32, "b.json": "zz" * 32}, "reason": "x"}
            },
        }
    ),
    "obsolete_provenance_gaps_key": _bytes(
        {"schema": "legacy_quarantine.v2", "files": {}, "provenance_gaps": {}}
    ),
}


@pytest.mark.parametrize("variant", sorted(_MALFORMED_REGISTRIES))
def test_malformed_registry_fails_closed(tmp_path: Path, variant: str) -> None:
    """Invariant: every malformed registry raises in ``load_quarantine`` and makes
    the strict CLI refuse to run (BadParameter, exit 2, no receipt written)
    instead of silently widening or narrowing what counts as quarantined."""
    reg = tmp_path / "q.json"
    reg.write_bytes(_MALFORMED_REGISTRIES[variant])
    with pytest.raises(ValueError):
        load_quarantine(reg)
    corpus = tmp_path / "receipts"
    _sealed_file(corpus / "a.json")
    out = tmp_path / "out"
    res = _invoke(
        [
            "suite-health",
            "--receipts-dir",
            corpus,
            "--out-dir",
            out,
            "--strict",
            "--quarantine",
            reg,
        ]
    )
    assert res.exit_code == 2, res.output
    assert not out.exists() or not any(out.iterdir())


def test_registry_with_duplicate_json_keys_is_rejected(tmp_path: Path) -> None:
    """Invariant: registry bytes determine one registry — duplicate keys fail closed."""
    first = json.dumps(_entry("ab" * 32, reason="the entry a reviewer reads"))
    second = json.dumps(_entry("cd" * 32, reason="the entry the parser keeps"))
    reg = tmp_path / "q.json"
    reg.write_text(
        '{"schema": "legacy_quarantine.v2", "files": {'
        + f'"a.json": {first}, "a.json": {second}'
        + "}}"
    )
    with pytest.raises(ValueError):
        load_quarantine(reg)


# --- 5. epoch chain -----------------------------------------------------------


def _put(corpus: Path, name: str, marker: str) -> bytes:
    return _sealed_file(corpus / name, note=marker)


def _stamp(corpus: Path) -> Path:
    return write_epoch_receipt(corpus_epoch(corpus), corpus)


def _pin(pin: Path, corpus: Path, head: Path) -> dict[str, Any]:
    update_heads_pin(pin, corpus, "*.json", head)
    return load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]


def _ack(
    corpus: Path, member: str, transition: str, frm: str, to: str | None, epoch: str
) -> dict[str, Any]:
    return {
        "corpus": epoch_heads_key(corpus, "*.json"),
        "member": member,
        "transition": transition,
        "from_sha256": frm,
        "to_sha256": to,
        "epoch": epoch,
        "reason": "adversarial fixture",
    }


def _chain(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(corpus, pin file, genesis epoch): two receipts stamped and pinned."""
    corpus = tmp_path / "receipts"
    _put(corpus, "a.json", "a1")
    _put(corpus, "b.json", "b1")
    genesis = _stamp(corpus)
    pin = tmp_path / "quality" / "epoch_heads.json"
    pin.parent.mkdir()
    exp = _pin(pin, corpus, genesis)
    assert check_epoch_chain(corpus, expected_head=exp)["errors"] == []  # positive control
    return corpus, pin, genesis


def _expected(pin: Path, corpus: Path) -> dict[str, Any]:
    return load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]


@pytest.mark.parametrize("variant", ["identical_duplicate", "shadowing_duplicate"])
def test_duplicate_member_names_in_an_epoch_fail_closed(tmp_path: Path, variant: str) -> None:
    """Invariant: an epoch payload listing a member name twice violates the
    ``corpus_epoch.v1`` contract — the writer refuses it, and a hand-sealed
    one never verifies, so it cannot become a chain head."""
    corpus = tmp_path / "receipts"
    _put(corpus, "a.json", "a1")
    _put(corpus, "b.json", "b1")
    epoch = corpus_epoch(corpus)
    dup = dict(epoch["members"][-1])
    if variant == "shadowing_duplicate":
        dup["sha256"] = "f" * 64
    forged = {**epoch, "members": [*epoch["members"], dup]}
    forged["n_members"] = len(forged["members"])
    assert "members_malformed_or_dup_names" in epoch_contract_errors(forged)
    with pytest.raises(ValueError, match="contract"):
        write_epoch_receipt(forged, corpus)
    sealed = seal_receipt(forged)
    path = corpus / f"corpus_epoch_{sealed['receipt_sha256'][:16]}.json"
    path.write_bytes(_bytes(sealed, indent=2, sort_keys=True))
    assert "members_malformed_or_dup_names" in verify_receipt_file(path)["errors"]
    errors = check_epoch_chain(corpus)["errors"]
    assert f"epoch_receipt_invalid:{path.name!a}" in errors
    assert "no_valid_epoch_receipts" in errors


def test_deleted_pinned_head_is_missing_and_rollback(tmp_path: Path) -> None:
    """Invariant: deleting the pinned (newest) epoch is ``epoch_head_missing`` and
    ``epoch_head_rollback`` — the chain cannot silently rewind to genesis."""
    corpus, pin, _genesis = _chain(tmp_path)
    _put(corpus, "c.json", "c1")
    head = _stamp(corpus)
    exp = _pin(pin, corpus, head)
    assert check_epoch_chain(corpus, expected_head=exp)["errors"] == []  # positive control
    head.unlink()
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert f"epoch_head_missing:{head.name!a}" in errors
    assert f"epoch_head_rollback:{head.name!a}" in errors


def test_deleting_every_epoch_is_still_caught_by_the_pin(tmp_path: Path) -> None:
    """Invariant: with a pin, an epoch-free corpus is a rollback, not 'unstamped'."""
    corpus, pin, genesis = _chain(tmp_path)
    genesis.unlink()
    errors = check_epoch_chain(corpus, expected_head=_expected(pin, corpus))["errors"]
    assert f"epoch_head_missing:{genesis.name!a}" in errors
    assert f"epoch_head_rollback:{genesis.name!a}" in errors
    assert "no_epoch_receipts" in errors
    _frame, receipt = suite_health(corpus, expected_head=_expected(pin, corpus))
    assert receipt["corpus_epoch"]["chain_ok"] is False


@pytest.mark.parametrize("field", ["sha256", "tree_root", "chain_root", "n_epochs"])
def test_pin_field_lies_are_detected(tmp_path: Path, field: str) -> None:
    """Invariant: every bound pin field is re-derived — a pin naming the live
    head but lying about its bytes, member tree, chain root or chain length
    fails closed."""
    corpus, pin, genesis = _chain(tmp_path)
    exp = dict(_expected(pin, corpus))
    exp[field] = 2 if field == "n_epochs" else "e" * 64
    expected_error = {
        "sha256": "epoch_head_mutated",
        "tree_root": "epoch_head_tree_root_mismatch",
        "chain_root": "epoch_head_chain_root_mismatch",
        "n_epochs": "epoch_pin_n_epochs_mismatch",
    }[field]
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert errors == [f"{expected_error}:{genesis.name!a}"]


def test_reserialized_head_epoch_is_mutated(tmp_path: Path) -> None:
    """Invariant: the pin binds the head's *bytes* — a whitespace-only rewrite
    still verifies as a receipt, yet ``epoch_head_mutated`` fires."""
    corpus, pin, genesis = _chain(tmp_path)
    genesis.write_bytes(canonical_json_bytes(json.loads(genesis.read_bytes())))
    assert verify_receipt_file(genesis)["valid"]
    errors = check_epoch_chain(corpus, expected_head=_expected(pin, corpus))["errors"]
    # Both the byte pin and the position commitment (leaves bind file sha) fire.
    assert errors == [
        f"epoch_head_mutated:{genesis.name!a}",
        f"epoch_head_chain_root_mismatch:{genesis.name!a}",
    ]


@pytest.mark.parametrize(
    "pin_doc",
    [
        {"schema": "epoch_heads.v1", "heads": []},
        {"schema": "epoch_heads.v1"},
        {"schema": "epoch_heads.v1", "heads": {"receipts/*.json": {"receipt": "x"}}},
        {"schema": "epoch_heads.v1", "heads": {"k": {"receipt": "x", "sha256": "a" * 63}}},
        {
            "schema": "epoch_heads.v1",
            "heads": {"k": {"receipt": "x", "sha256": "a" * 64, "n_epochs": 0}},
        },
        {
            "schema": "epoch_heads.v1",
            "heads": {"k": {"receipt": "x", "sha256": "a" * 64, "tree_root": "b"}},
        },
    ],
    ids=["heads_list", "heads_missing", "no_sha", "short_sha", "zero_epochs", "short_tree"],
)
def test_malformed_heads_pin_fails_closed(tmp_path: Path, pin_doc: dict[str, Any]) -> None:
    """Invariant: a malformed pin raises in ``load_heads_pin`` and the strict
    CLI refuses it (exit 2) instead of running unpinned."""
    corpus, pin, _genesis = _chain(tmp_path)
    pin.write_text(json.dumps(pin_doc))
    with pytest.raises(ValueError):
        load_heads_pin(pin)
    res = _invoke(
        [
            "suite-health",
            "--receipts-dir",
            corpus,
            "--out-dir",
            tmp_path / "out",
            "--strict",
            "--quarantine",
            tmp_path / "absent.json",
            "--heads-pin",
            pin,
        ]
    )
    assert res.exit_code == 2, res.output


def test_pin_without_this_corpus_key_fails_closed(tmp_path: Path) -> None:
    """Invariant: a pin that does not name this chain is an error, never 'unpinned'."""
    corpus, pin, _genesis = _chain(tmp_path)
    doc = json.loads(pin.read_text())
    doc["heads"] = {"elsewhere/*.json": next(iter(doc["heads"].values()))}
    pin.write_text(json.dumps(doc))
    for command in (
        [
            "suite-health",
            "--receipts-dir",
            corpus,
            "--out-dir",
            tmp_path / "out",
            "--strict",
            "--quarantine",
            tmp_path / "absent.json",
        ],
        ["corpus-epoch", "--corpus-dir", corpus, "--check"],
    ):
        res = _invoke([*command, "--heads-pin", pin])
        assert res.exit_code == 2, res.output
        assert "has no entry" in res.output


def test_fork_is_detected(tmp_path: Path) -> None:
    """Invariant: two epochs naming the same ``prev`` are a fork with two heads —
    neither is trusted as the head, and the pin check fails too."""
    corpus, pin, genesis = _chain(tmp_path)
    _put(corpus, "c.json", "c1")
    left = corpus_epoch(corpus)
    _put(corpus, "d.json", "d1")
    right = corpus_epoch(corpus)
    assert left["prev_epoch_receipt"] == right["prev_epoch_receipt"] == genesis.name
    left_path, right_path = write_epoch_receipt(left, corpus), write_epoch_receipt(right, corpus)
    result = check_epoch_chain(corpus, expected_head=_expected(pin, corpus))
    names = sorted([left_path.name, right_path.name])
    assert f"epoch_fork:{genesis.name!a}->{names[0]!a},{names[1]!a}" in result["errors"] or (
        f"epoch_fork:{genesis.name!a}->{names[1]!a},{names[0]!a}" in result["errors"]
    )
    assert f"epoch_multiple_heads:{names[0]!a},{names[1]!a}" in result["errors"]
    assert result["head"] is None


def _recorded_removal(tmp_path: Path) -> tuple[Path, str, Path]:
    """(corpus, removed member's stamped digest, recording epoch)."""
    corpus, _pin_path, _genesis = _chain(tmp_path)
    digest = member_digests(corpus)["a.json"]
    (corpus / "a.json").unlink()
    recording = _stamp(corpus)
    errors = check_epoch_chain(corpus)["errors"]
    assert errors == [f"member_removed:'a.json'@{recording.name!a}"]  # unacknowledged
    return corpus, digest, recording


def _recorded_mutation(tmp_path: Path) -> tuple[Path, str, str, Path]:
    """(corpus, stamped digest, new digest, recording epoch)."""
    corpus, _pin_path, _genesis = _chain(tmp_path)
    before = member_digests(corpus)["a.json"]
    _put(corpus, "a.json", "a2-rewritten")
    after = member_digests(corpus)["a.json"]
    recording = _stamp(corpus)
    errors = check_epoch_chain(corpus)["errors"]
    assert errors == [f"member_mutated:'a.json'@{recording.name!a}"]  # append-only: tamper
    return corpus, before, after, recording


def test_exact_acks_move_recorded_transitions_to_acknowledged(tmp_path: Path) -> None:
    """Positive control: an ack naming the exact member, epoch and digests turns
    the recorded error into a printed acknowledgment."""
    corpus, digest, recording = _recorded_removal(tmp_path / "removal")
    ack = _ack(corpus, "a.json", "removed", digest, None, recording.name)
    result = check_epoch_chain(corpus, acks=[ack])
    assert result["errors"] == []
    assert result["acknowledged"] == [f"member_removed:'a.json'@{recording.name!a}"]
    corpus, before, after, recording = _recorded_mutation(tmp_path / "mutation")
    ack = _ack(corpus, "a.json", "mutated", before, after, recording.name)
    result = check_epoch_chain(corpus, acks=[ack])
    assert result["errors"] == []
    assert result["acknowledged"] == [f"member_mutated:'a.json'@{recording.name!a}"]


@pytest.mark.parametrize(
    "lie", ["wrong_from", "wrong_epoch", "nonexistent_epoch", "wrong_transition", "wrong_member"]
)
def test_inexact_removal_ack_explains_nothing(tmp_path: Path, lie: str) -> None:
    """Invariant: an ack off by any field is ``epoch_ack_unmatched`` and the
    original ``member_removed`` error stays."""
    corpus, digest, recording = _recorded_removal(tmp_path)
    genesis_name = sorted(p.name for p in corpus.glob("corpus_epoch_*.json") if p != recording)[0]
    ack = _ack(corpus, "a.json", "removed", digest, None, recording.name)
    if lie == "wrong_from":
        ack["from_sha256"] = "0" * 64
    elif lie == "wrong_epoch":
        ack["epoch"] = genesis_name
    elif lie == "nonexistent_epoch":
        ack["epoch"] = "corpus_epoch_0123456789abcdef.json"
    elif lie == "wrong_transition":
        ack.update(transition="mutated", to_sha256="1" * 64)
    else:
        ack["member"] = "b.json"
    errors = check_epoch_chain(corpus, acks=[ack])["errors"]
    assert f"member_removed:'a.json'@{recording.name!a}" in errors
    assert f"epoch_ack_unmatched:{ack['member']!a}@{ack['epoch']!a}" in errors


@pytest.mark.parametrize("lie", ["wrong_to", "swapped", "removal_shape"])
def test_inexact_mutation_ack_explains_nothing(tmp_path: Path, lie: str) -> None:
    """Invariant: a mutation ack must bind both digests in order; anything else
    is unmatched and ``member_mutated`` stays."""
    corpus, before, after, recording = _recorded_mutation(tmp_path)
    ack = _ack(corpus, "a.json", "mutated", before, after, recording.name)
    if lie == "wrong_to":
        ack["to_sha256"] = "2" * 64
    elif lie == "swapped":
        ack.update(from_sha256=after, to_sha256=before)
    else:
        ack.update(transition="removed", to_sha256=None)
    errors = check_epoch_chain(corpus, acks=[ack])["errors"]
    assert f"member_mutated:'a.json'@{recording.name!a}" in errors
    assert f"epoch_ack_unmatched:'a.json'@{recording.name!a}" in errors


@pytest.mark.parametrize("transition", ["removed", "mutated"])
def test_live_drift_is_never_acknowledgeable(tmp_path: Path, transition: str) -> None:
    """Invariant: an ack can only explain sealed history. Live removal / rewrite
    after the head stamp keeps ``head_member_missing_live`` /
    ``head_member_digest_drift`` and the ack naming the head epoch is
    ``epoch_ack_unmatched``."""
    corpus, pin, genesis = _chain(tmp_path)
    before = member_digests(corpus)["a.json"]
    if transition == "removed":
        (corpus / "a.json").unlink()
        ack = _ack(corpus, "a.json", "removed", before, None, genesis.name)
        live_error = "head_member_missing_live:'a.json'"
    else:
        _put(corpus, "a.json", "a2-live")
        ack = _ack(
            corpus, "a.json", "mutated", before, member_digests(corpus)["a.json"], genesis.name
        )
        live_error = "head_member_digest_drift:'a.json'"
    errors = check_epoch_chain(corpus, acks=[ack], expected_head=_expected(pin, corpus))["errors"]
    assert errors == [live_error, f"epoch_ack_unmatched:'a.json'@{genesis.name!a}"]


def test_append_only_mutation_without_ack_is_member_mutated(tmp_path: Path) -> None:
    """Invariant: in an append-only corpus a stamped member rewritten and then
    re-stamped is tamper evidence (``member_mutated``) on the CLI chain check —
    re-stamping does not launder it; only the mutable-corpus policy relaxes it."""
    corpus, _before, _after, recording = _recorded_mutation(tmp_path)
    check = ["corpus-epoch", "--corpus-dir", corpus, "--check"]
    res = _invoke(check)
    assert res.exit_code == 1, res.output
    assert f"epoch-chain error: member_mutated:'a.json'@{recording.name!a}" in res.output
    relaxed = _invoke([*check, "--allow-member-updates"])
    assert relaxed.exit_code == 0, relaxed.output


def test_mutation_ack_in_a_mutable_corpus_is_unmatched(tmp_path: Path) -> None:
    """Invariant: under ``allow_member_updates`` a mutation is not an error, so an
    ack for it explains nothing — it is ``epoch_ack_unmatched``, never a
    silent no-op that could later pre-authorize tamper."""
    corpus, before, after, recording = _recorded_mutation(tmp_path)
    assert check_epoch_chain(corpus, allow_member_updates=True)["errors"] == []  # control
    ack = _ack(corpus, "a.json", "mutated", before, after, recording.name)
    result = check_epoch_chain(corpus, acks=[ack], allow_member_updates=True)
    assert result["errors"] == [f"epoch_ack_unmatched:'a.json'@{recording.name!a}"]
    assert result["acknowledged"] == []


def test_ack_scoped_to_another_corpus_cannot_explain_this_one(tmp_path: Path) -> None:
    """Invariant: on the CLI path the ledger is filtered by corpus key — an ack
    for another chain cannot explain this chain's removal. Positive control:
    the same ack scoped correctly is printed as acknowledged."""
    corpus, digest, recording = _recorded_removal(tmp_path)
    ledger = tmp_path / "acks.json"
    ack = _ack(corpus, "a.json", "removed", digest, None, recording.name)
    check = ["corpus-epoch", "--corpus-dir", corpus, "--check", "--epoch-acks", ledger]
    ledger.write_text(json.dumps({"schema": "epoch_acknowledgments.v1", "acks": [ack]}))
    res = _invoke(check)
    assert res.exit_code == 0, res.output
    assert "epoch-chain acknowledged: member_removed:'a.json'" in res.output
    ledger.write_text(
        json.dumps(
            {"schema": "epoch_acknowledgments.v1", "acks": [{**ack, "corpus": "quality/*.json"}]}
        )
    )
    res = _invoke(check)
    assert res.exit_code == 1, res.output
    assert "epoch-chain error: member_removed:'a.json'" in res.output


_ACK = {
    "corpus": "receipts/*.json",
    "member": "a.json",
    "transition": "removed",
    "from_sha256": "a" * 64,
    "to_sha256": None,
    "epoch": "corpus_epoch_0123456789abcdef.json",
    "reason": "adversarial fixture",
}
_BAD_LEDGERS: dict[str, object] = {
    "legacy_flat_map": {"a.json": "a" * 64},
    "schema_missing": {"acks": [_ACK]},
    "schema_wrong": {"schema": "epoch_allowed_removals.v0", "acks": [_ACK]},
    "extra_top_level": {"schema": "epoch_acknowledgments.v1", "acks": [], "note": "x"},
    "acks_not_list": {"schema": "epoch_acknowledgments.v1", "acks": {"a.json": _ACK}},
    "duplicate_ack": {"schema": "epoch_acknowledgments.v1", "acks": [_ACK, dict(_ACK)]},
    "duplicate_ack_other_reason": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [_ACK, {**_ACK, "reason": "a second story"}],
    },
    "extra_field": {"schema": "epoch_acknowledgments.v1", "acks": [{**_ACK, "by": "me"}]},
    "missing_field": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{k: v for k, v in _ACK.items() if k != "reason"}],
    },
    "corpus_without_pattern": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "corpus": "receipts"}],
    },
    "empty_member": {"schema": "epoch_acknowledgments.v1", "acks": [{**_ACK, "member": ""}]},
    "unknown_transition": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "transition": "deleted"}],
    },
    "epoch_not_receipt_name": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "epoch": "corpus_epoch_short.json"}],
    },
    "epoch_path_traversal": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "epoch": "../corpus_epoch_0123456789abcdef.json"}],
    },
    "removal_with_to": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "to_sha256": "b" * 64}],
    },
    "mutation_without_to": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "transition": "mutated"}],
    },
    "mutation_to_equals_from": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "transition": "mutated", "to_sha256": "a" * 64}],
    },
    "uppercase_digest": {
        "schema": "epoch_acknowledgments.v1",
        "acks": [{**_ACK, "from_sha256": "A" * 64}],
    },
    "blank_reason": {"schema": "epoch_acknowledgments.v1", "acks": [{**_ACK, "reason": " "}]},
}


@pytest.mark.parametrize("variant", sorted(_BAD_LEDGERS))
def test_malformed_ack_ledger_fails_closed(tmp_path: Path, variant: str) -> None:
    """Invariant: ``load_epoch_acks`` raises on duplicates, the legacy flat
    ``{name: sha}`` map, unknown/missing fields and every malformed entry,
    and the CLI chain check refuses the ledger (exit 2)."""
    ledger = tmp_path / "acks.json"
    ledger.write_text(json.dumps(_BAD_LEDGERS[variant]))
    with pytest.raises(ValueError):
        load_epoch_acks(ledger)
    corpus, _pin_path, _genesis = _chain(tmp_path)
    res = _invoke(["corpus-epoch", "--corpus-dir", corpus, "--check", "--epoch-acks", ledger])
    assert res.exit_code == 2, res.output


def test_ack_entry_with_duplicate_json_keys_is_rejected(tmp_path: Path) -> None:
    """Invariant: ledger bytes determine one ledger — duplicate keys fail closed."""
    entry = json.dumps(_ACK)[:-1] + ', "member": "a_different_member.json"}'
    ledger = tmp_path / "acks.json"
    ledger.write_text('{"schema": "epoch_acknowledgments.v1", "acks": [' + entry + "]}")
    with pytest.raises(ValueError):
        load_epoch_acks(ledger)


# --- 6. interrupted update ----------------------------------------------------


def test_stamp_interrupted_before_pin_update_is_an_accepted_descendant(tmp_path: Path) -> None:
    """Documents the crash window *after* ``write_epoch_receipt`` but *before*
    ``update_heads_pin``: the old pin still names the previous head, the new
    epoch is its descendant, and the checker accepts it (pinned bound fields
    are re-derived on the chain truncated at the pinned head). Re-running the
    pin update heals the pin; nothing is lost or forged."""
    corpus, pin, genesis = _chain(tmp_path)
    old = _expected(pin, corpus)
    _put(corpus, "c.json", "c1")
    newer = _stamp(corpus)  # ... crash here: pin not updated
    result = check_epoch_chain(corpus, expected_head=old, require_stamped=True)
    assert result["errors"] == [] and result["head"] == newer.name
    _frame, receipt = suite_health(corpus, expected_head=old, require_stamped=True)
    assert receipt["corpus_epoch"]["chain_ok"] is True
    healed = _pin(pin, corpus, newer)
    assert healed["receipt"] == newer.name and healed["n_epochs"] == 2
    assert check_epoch_chain(corpus, expected_head=healed)["errors"] == []
    assert old["receipt"] == genesis.name


def test_pin_naming_an_unwritten_head_is_missing(tmp_path: Path) -> None:
    """Invariant: the reverse crash — the pin advanced to an epoch whose write
    never landed in the corpus — is ``epoch_head_missing`` (+ rollback)."""
    corpus, pin, _genesis = _chain(tmp_path)
    _put(corpus, "c.json", "c1")
    staging = tmp_path / "staging"  # where the lost atomic write lived
    lost = write_epoch_receipt(corpus_epoch(corpus), staging)
    # The writer cannot be pointed at a head the chain does not contain...
    with pytest.raises(ValueError, match="refusing to pin"):
        update_heads_pin(pin, corpus, "*.json", lost)
    # ...so model the crash order: the head landed, the pin advanced, then
    # the head write was lost.
    landed = corpus / lost.name
    landed.write_bytes(lost.read_bytes())
    exp = _pin(pin, corpus, landed)
    landed.unlink()
    assert exp["receipt"] == lost.name and not (corpus / lost.name).exists()
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert f"epoch_head_missing:{lost.name!a}" in errors
    assert f"epoch_head_rollback:{lost.name!a}" in errors


@pytest.mark.parametrize("damage", ["truncated", "empty", "bitflip"])
def test_partially_written_epoch_fails_closed(tmp_path: Path, damage: str) -> None:
    """Invariant: a torn or corrupted epoch file never extends the chain: torn
    bytes squat the reserved prefix (and block further stamping), a bit-flip
    fails the epoch's own seal; as the pinned head either is also mutated."""
    corpus, pin, _genesis = _chain(tmp_path)
    _put(corpus, "c.json", "c1")
    head = _stamp(corpus)
    exp = _pin(pin, corpus, head)
    data = head.read_bytes()
    if damage == "truncated":
        head.write_bytes(data[: len(data) // 2])
    elif damage == "empty":
        head.write_bytes(b"")
    else:
        flipped = json.loads(data)
        flipped["members"][0]["sha256"] = "0" * 64
        head.write_bytes(_bytes(flipped, indent=2, sort_keys=True))
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert f"epoch_head_mutated:{head.name!a}" in errors
    if damage == "bitflip":
        assert f"epoch_receipt_invalid:{head.name!a}" in errors
    else:
        assert f"epoch_prefix_squat:{head.name!a}" in errors
        with pytest.raises(ValueError, match="squat"):
            corpus_epoch(corpus)
        before, pin_bytes = sorted(corpus.glob("*.json")), pin.read_bytes()
        res = _invoke(
            ["corpus-epoch", "--corpus-dir", corpus, "--out-dir", corpus, "--heads-pin", pin]
        )
        assert res.exit_code == 2, res.output
        assert sorted(corpus.glob("*.json")) == before and pin.read_bytes() == pin_bytes


# --- 7. concurrent writers ----------------------------------------------------


def test_stamp_lock_excludes_a_second_process(tmp_path: Path) -> None:
    """Invariant: the stamp lock is exclusive across processes — while another
    process holds it, acquisition raises ``EpochStampLocked`` and the CLI
    stamp refuses (exit 2) without writing an epoch or touching the pin;
    once released, stamping proceeds."""
    corpus, pin, _genesis = _chain(tmp_path)
    _put(corpus, "c.json", "c1")
    holder = (
        "import sys\n"
        "from pathlib import Path\n"
        "from quant_fund.research.corpus_epoch import _acquire_stamp_lock\n"
        "fd = _acquire_stamp_lock(Path(sys.argv[1]))\n"
        "print('locked', flush=True)\n"
        "sys.stdin.read()\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-c", holder, str(corpus)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "locked"
        with pytest.raises(EpochStampLocked):
            _acquire_stamp_lock(corpus)
        before, pin_bytes = sorted(corpus.glob("*.json")), pin.read_bytes()
        res = _invoke(
            ["corpus-epoch", "--corpus-dir", corpus, "--out-dir", corpus, "--heads-pin", pin]
        )
        assert res.exit_code == 2, res.output
        assert "in flight" in res.output
        assert sorted(corpus.glob("*.json")) == before
        assert pin.read_bytes() == pin_bytes
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=60)
    fd = _acquire_stamp_lock(corpus)  # released with the holder
    fd.close()


def test_in_process_lock_is_exclusive(tmp_path: Path) -> None:
    """Invariant: a second acquisition in the same process also fails closed."""
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    held = _acquire_stamp_lock(corpus)
    try:
        with pytest.raises(EpochStampLocked):
            _acquire_stamp_lock(corpus)
    finally:
        held.close()


def test_two_unlocked_stamps_from_one_head_fork_the_chain(tmp_path: Path) -> None:
    """Invariant: two writers that both read the same head (bypassing the lock)
    produce sibling epochs, and every gate sees the fork — the checker, the
    suite-health epoch state, and admission's epoch check."""
    corpus, pin, genesis = _chain(tmp_path)
    _put(corpus, "c.json", "c1")
    first = corpus_epoch(corpus)  # writer 1 reads head = genesis
    _put(corpus, "d.json", "d1")
    second = corpus_epoch(corpus)  # writer 2 reads head = genesis too
    write_epoch_receipt(first, corpus)
    write_epoch_receipt(second, corpus)
    exp = _expected(pin, corpus)
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert any(e.startswith(f"epoch_fork:{genesis.name!a}->") for e in errors)
    assert any(e.startswith("epoch_multiple_heads:") for e in errors)
    _frame, receipt = suite_health(corpus, expected_head=exp)
    assert receipt["corpus_epoch"]["chain_ok"] is False


# --- 8. schema transitions ----------------------------------------------------


def test_honest_v1_and_v2_both_verify(tmp_path: Path) -> None:
    """Positive control: the same body verifies sealed as v1 and wrapped as v2."""
    for name, payload in (("v1.json", _v1()), ("v2.json", _v2())):
        path = tmp_path / name
        path.write_bytes(_bytes(payload, indent=2, sort_keys=True))
        result = verify_receipt_file(path)
        assert result["valid"], result["errors"]
    assert verify_receipt_file(tmp_path / "v2.json")["schema"] == "receipt.v2"
    assert verify_receipt_file(tmp_path / "v1.json")["schema"] == "adversarial_fixture.v1"


@pytest.mark.parametrize(
    ("stripped", "expected"),
    [
        (("schema", "schema_version"), "receipt_v2_schema:schema:Field required"),
        (("schema", "schema_version", "environment"), "possible_v2_downgrade"),
    ],
    ids=["strip_tags", "strip_tags_and_environment"],
)
def test_v2_downgrade_is_caught_even_when_resealed(
    stripped: tuple[str, ...], expected: str
) -> None:
    """Invariant: stripping the envelope tags (and the environment block that
    keys v2 dispatch) and resealing honestly cannot pass as a weaker v1
    receipt — it is either still validated as a broken envelope or flagged
    ``possible_v2_downgrade``."""
    downgraded = seal_receipt(
        {k: v for k, v in _v2().items() if k not in (*stripped, "receipt_sha256")}
    )
    result = verify_receipt_payload(downgraded)
    assert not result["valid"]
    assert expected in result["errors"]


def _suite_receipt(tmp_path: Path) -> dict[str, Any]:
    """A real suite_health body over a corpus with one verified, one
    quarantined and one invalid receipt."""
    corpus = tmp_path / "receipts"
    _sealed_file(corpus / "good.json")
    (corpus / "legacy-unsealed").mkdir()
    (corpus / "legacy-unsealed" / "old.json").write_bytes(_LEGACY)
    (corpus / "broken.json").write_bytes(b'{"kind": "broken"}')
    reg = _registry(tmp_path / "q.json", {"legacy-unsealed/old.json": _entry(_sha(_LEGACY))})
    _frame, receipt = suite_health(corpus, quarantine=load_quarantine(reg))
    assert (receipt["n_ok"], receipt["n_quarantined"], receipt["n_invalid"]) == (1, 1, 1)
    return receipt


def _v1_sealed(body: dict[str, Any]) -> dict[str, Any]:
    """Seal exactly as ``write_suite_health_receipt(receipt_version=1)`` does."""
    canonical = json.loads(canonical_json_bytes(body))
    return {**canonical, "receipt_sha256": hash_bytes(canonical_json_bytes(canonical))}


def test_suite_health_body_without_quarantine_fields_still_verifies(tmp_path: Path) -> None:
    """Invariant (backward compat): a ``suite_health.v1`` body sealed before the
    registry existed — no ``n_quarantined``/``n_invalid`` — still verifies."""
    receipt = _suite_receipt(tmp_path)
    assert verify_receipt_payload(_v1_sealed(receipt))["valid"]  # positive control
    legacy = {k: v for k, v in receipt.items() if k not in ("n_quarantined", "n_invalid")}
    result = verify_receipt_payload(_v1_sealed(legacy))
    assert result["valid"], result["errors"]


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        ({"n_quarantined": 2}, "quarantined_plus_invalid_neq_failed"),
        ({"n_invalid": 0}, "quarantined_plus_invalid_neq_failed"),
        ({"n_quarantined": 0, "n_invalid": 2}, "invalid_map_count_mismatch"),
        ({"n_quarantined": -1}, "n_quarantined_not_nonnegative_int"),
        ({"n_invalid": True}, "n_invalid_not_nonnegative_int"),
        ({"n_invalid": None}, "n_invalid_not_nonnegative_int"),
        ({"invalid": {}}, "invalid_map_count_mismatch"),
        ({"invalid": None}, "invalid_map_missing"),
    ],
    ids=repr,
)
def test_suite_health_quarantine_accounting_lies_are_rejected(
    tmp_path: Path, edit: dict[str, Any], expected: str
) -> None:
    """Invariant: a suite_health.v1 body whose quarantine accounting does not add
    up fails the production verifier even under a fresh, honest seal."""
    forged = {**_suite_receipt(tmp_path), **edit}
    result = verify_receipt_payload(_v1_sealed(forged))
    assert not result["valid"]
    assert expected in result["errors"], result["errors"]


def test_suite_health_chain_ok_lie_is_rejected_in_v1(tmp_path: Path) -> None:
    """Invariant: a v1 suite_health body claiming ``chain_ok`` while listing chain
    errors is rejected under a fresh seal (control for the v2 regression test below)."""
    forged = dict(_suite_receipt(tmp_path))
    forged["corpus_epoch"] = {
        **forged["corpus_epoch"],
        "errors": ["head_member_digest_drift:'good.json'"],
        "chain_ok": True,
    }
    result = verify_receipt_payload(_v1_sealed(forged))
    assert "corpus_epoch_chain_ok_lies" in result["errors"]


def test_honest_suite_health_v2_envelope_verifies(tmp_path: Path) -> None:
    """Positive control: the production v2 writer's honest output verifies."""
    receipt = _suite_receipt(tmp_path)
    path = suite_health_mod.write_suite_health_receipt(receipt, tmp_path / "out", receipt_version=2)
    assert verify_receipt_file(path)["valid"], verify_receipt_file(path)["errors"]


@pytest.mark.parametrize(
    "edit",
    [{"n_quarantined": 2}, {"corpus_epoch_chain_ok_lie": True}],
    ids=["quarantine_count", "chain_ok_lie"],
)
def test_suite_health_v2_envelope_enforces_the_same_contract(
    tmp_path: Path, edit: dict[str, Any]
) -> None:
    """Invariant: the suite_health contract binds the body in either generation."""
    forged = dict(_suite_receipt(tmp_path))
    if "corpus_epoch_chain_ok_lie" in edit:
        forged["corpus_epoch"] = {
            **forged["corpus_epoch"],
            "errors": ["head_member_digest_drift:'good.json'"],
            "chain_ok": True,
        }
    else:
        forged.update(edit)
    envelope = seal_receipt(
        wrap_receipt_v2(forged, code_files=(Path(suite_health_mod.__file__),), verdict="fail")
    )
    assert not verify_receipt_payload(envelope)["valid"]
