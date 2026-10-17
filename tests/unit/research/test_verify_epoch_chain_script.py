"""Differential pin: scripts/verify_epoch_chain.py ↔ check_epoch_chain.

The standalone script is the third-party oracle for the whole chain — an
auditor verifies seals, prev links, topology, delta honesty, and
head-vs-live consistency without importing the repo. A divergence between
its verdict and the library's is itself a finding.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_epoch_chain.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _member(dirpath: Path, name: str, content: str = "{}") -> Path:
    p = dirpath / name
    p.write_text(content)
    return p


def _chain(corpus: Path, n_stamps: int = 2) -> list[Path]:
    """Stamp ``n_stamps`` epochs, mutating one member between stamps."""
    from quant_fund.research.corpus_epoch import corpus_epoch, write_epoch_receipt

    out: list[Path] = []
    for i in range(n_stamps):
        if i:
            _member(corpus, f"m{i}.json", f'{{"i": {i}}}')
        out.append(write_epoch_receipt(corpus_epoch(corpus), corpus))
    return out


def _errors_of(proc: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        ln[2:]
        for ln in proc.stdout.splitlines()
        if ln.startswith("  ") and not ln.startswith("  unstamped:")
    ]


def _script_errors(corpus: Path, *extra: str) -> list[str]:
    return _errors_of(_run("--corpus-dir", str(corpus), *extra))


def _lib_errors(corpus: Path, **kw) -> list[str]:
    from quant_fund.research.corpus_epoch import check_epoch_chain

    return check_epoch_chain(corpus, **kw)["errors"]


def test_clean_chain_agrees(tmp_path: Path) -> None:
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _member(corpus, "a.json", '{"a": 1}')
    _chain(corpus)
    lib = _lib_errors(corpus)
    script = _script_errors(corpus)
    assert lib == [], lib
    assert script == [], script


def test_mutation_classes_agree(tmp_path: Path) -> None:
    """Every tamper class must be flagged identically by both verifiers."""

    cases: dict[str, object] = {}

    def build(mutate) -> Path:
        corpus = tmp_path / f"c{len(cases)}"
        corpus.mkdir()
        _member(corpus, "a.json", '{"a": 1}')
        _chain(corpus)
        mutate(corpus)
        return corpus

    cases["byte_flip"] = build(lambda c: (c / "a.json").write_text('{"a": 2}'))
    cases["member_drop"] = build(lambda c: (c / "m1.json").unlink())
    cases["squatter"] = build(lambda c: (c / "corpus_epoch_evil.json").write_text('{"x": 1}'))
    cases["unparseable_squatter"] = build(
        lambda c: (c / "corpus_epoch_evil.json").write_bytes(b"{oops")
    )

    def forged(c: Path) -> None:
        body = {
            "kind": "corpus_epoch.v1",
            "schema": "corpus_epoch.v1",
            "members": [],
            "n_members": 0,
            "members_added": [],
            "members_removed": [],
            "prev_epoch_receipt": None,
            "prev_epoch_sha256": "0" * 64,
            "epoch_root_sha256": "0" * 64,
            "verdict": "genesis",
        }
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
        body["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
        (c / "corpus_epoch_forged.json").write_text(json.dumps(body))

    cases["forged_epoch"] = build(forged)

    def fork(c: Path) -> None:
        epochs = sorted(c.glob("corpus_epoch_*.json"))
        first = json.loads(epochs[0].read_text())
        dup = dict(first)
        dup["members"] = []
        canon = json.dumps(
            {k: v for k, v in dup.items() if k != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        )
        dup["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
        (c / "corpus_epoch_twin.json").write_text(json.dumps(dup))

    cases["forked_genesis"] = build(fork)

    def orphan(c: Path) -> None:
        epochs = sorted(c.glob("corpus_epoch_*.json"))
        head = json.loads(epochs[-1].read_text())
        head["prev_epoch_receipt"] = "corpus_epoch_missing.json"
        canon = json.dumps(
            {k: v for k, v in head.items() if k != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        )
        head["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
        epochs[-1].unlink()
        (c / "corpus_epoch_orphaned.json").write_text(json.dumps(head))

    cases["orphan_link"] = build(orphan)

    def mid_chain_drop(c: Path) -> None:
        # Receipt names are content-derived, so sorted() order != chain
        # order — resolve a receipt that is someone's ``prev`` and drop it.
        receipts = {p.name: p for p in c.glob("corpus_epoch_*.json")}
        prevs = {json.loads(p.read_text()).get("prev_epoch_receipt") for p in receipts.values()}
        referenced = [n for n in receipts if n in prevs]
        assert referenced, "expected a multi-link chain"
        receipts[sorted(referenced)[0]].unlink()

    cases["mid_chain_drop"] = build(mid_chain_drop)

    for name, corpus in cases.items():
        lib = _lib_errors(corpus)
        script = _script_errors(corpus)
        assert lib, f"{name}: library saw nothing"
        assert script, f"{name}: script saw nothing"
        # Same label vocabulary — script errors are a subset that must agree
        # on the defect class even when the library carries extra context.
        lib_labels = {e.split(":", 1)[0] for e in lib}
        script_labels = {e.split(":", 1)[0] for e in script}
        assert script_labels <= lib_labels, (name, script_labels - lib_labels)
        assert script_labels & lib_labels, (name, script, lib)


def test_symlinked_corpus_root_agrees(tmp_path: Path) -> None:
    """A re-rooted corpus is flagged by the library and the standalone oracle."""
    real = tmp_path / "zz_corpus"
    real.mkdir()
    _member(real, "a.json", '{"a": 1}')
    _chain(real)
    root = tmp_path / "receipts"
    root.symlink_to(real, target_is_directory=True)

    assert _lib_errors(real) == [] == _script_errors(real)
    lib = _lib_errors(root)
    script = _script_errors(root)
    assert "corpus_root_is_symlink:'receipts'" in lib, lib
    assert "corpus_root_is_symlink:'receipts'" in script, script


def test_pin_rollback_and_missing_agree(tmp_path: Path) -> None:
    """The committed-head pin binds both verifiers to the same head."""
    from quant_fund.research.corpus_epoch import (
        epoch_heads_key,
    )

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _member(corpus, "a.json", '{"a": 1}')
    epochs = _chain(corpus)

    pin_file = tmp_path / "epoch_heads.json"
    pin_file.write_text(
        json.dumps(
            {
                "heads": {
                    epoch_heads_key("receipts", "*.json"): {
                        "receipt": epochs[-1].name,
                        "sha256": hashlib.sha256(epochs[-1].read_bytes()).hexdigest(),
                    }
                }
            }
        )
    )
    expected = {
        "receipt": epochs[-1].name,
        "sha256": hashlib.sha256(epochs[-1].read_bytes()).hexdigest(),
    }
    lib = _lib_errors(corpus, expected_head=expected)
    script = _script_errors(corpus, "--pin", str(pin_file), "--key", "receipts/*.json")
    assert lib == script == []

    # Rollback: delete the pinned head → both must flag.
    epochs[-1].unlink()
    lib = _lib_errors(corpus, expected_head=expected)
    script = _script_errors(corpus, "--pin", str(pin_file), "--key", "receipts/*.json")
    assert "epoch_head_missing" in {e.split(":")[0] for e in lib}
    assert "epoch_head_missing" in {e.split(":")[0] for e in script}


def test_require_stamped_and_member_updates_flags(tmp_path: Path) -> None:
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _member(corpus, "a.json", '{"a": 1}')
    _chain(corpus, n_stamps=1)
    _member(corpus, "late.json", '{"late": true}')  # arrives post-stamp

    # Strict corpus: the arrival must be an error in both verifiers.
    lib = _lib_errors(corpus, require_stamped=True)
    script = _script_errors(corpus, "--require-stamped")
    assert {e.split(":")[0] for e in lib} == {e.split(":")[0] for e in script}
    assert "unstamped_member" in {e.split(":")[0] for e in lib}

    # Mutable corpus: post-stamp member edit is bookkeeping, not tamper.
    (corpus / "a.json").write_text('{"a": 2}')
    lib = _lib_errors(corpus, allow_member_updates=True)
    script = _script_errors(corpus, "--allow-member-updates")
    assert {e.split(":")[0] for e in lib} == {e.split(":")[0] for e in script}
    assert not any("member_mutated" in e for e in lib)


def test_usage_errors(tmp_path: Path) -> None:
    assert _run().returncode == 2
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    assert _run("--corpus-dir", str(corpus), "--checkpoint", "x").returncode == 2


@pytest.mark.parametrize(
    ("epoch_bytes", "label"),
    [
        (b"[1, 2, 3]", "no_epoch_receipts"),  # squats the prefix, yields no epoch
        (
            json.dumps(
                {
                    "kind": "corpus_epoch.v1",
                    "schema": "corpus_epoch.v1",
                    "members": [],
                    "receipt_sha256": "0" * 64,
                }
            ).encode(),
            "no_valid_epoch_receipts",  # an epoch payload whose seal fails
        ),
    ],
)
def test_corpus_without_a_valid_epoch_fails_without_crashing(
    tmp_path: Path, epoch_bytes: bytes, label: str
) -> None:
    """The early-return paths (nothing to walk) must report, not raise: a
    traceback exits non-zero too, but prints no verdict an auditor can read."""
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _member(corpus, "a.json", '{"a": 1}')
    (corpus / "corpus_epoch_0000000000000000.json").write_bytes(epoch_bytes)
    proc = _run("--corpus-dir", str(corpus))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "FAIL" in proc.stdout
    script = _script_errors(corpus)
    assert label in script
    assert label in _lib_errors(corpus)


# --- pinned-head position commitments (tree_root / chain_root / n_epochs) ---
#
# A pin that names the right head (receipt + sha256) but lies about its
# position fields must fail the standalone auditor exactly as it fails the
# library. Offline ``epoch_position.v1`` proofs anchor on these values, so an
# auditor that only compared the head's sha256 would bless a forged anchor.

POS_TREE = "epoch_head_tree_root_mismatch"
POS_CHAIN = "epoch_head_chain_root_mismatch"
POS_N = "epoch_pin_n_epochs_mismatch"
POS_MALFORMED = "epoch_pin_field_malformed"
POS_UNVERIFIABLE = "epoch_pin_chain_unverifiable"
POSITION_LABELS = (POS_TREE, POS_CHAIN, POS_N, POS_MALFORMED, POS_UNVERIFIABLE)


def _pinned_corpus(tmp_path: Path, n_stamps: int = 3) -> tuple[Path, list[Path], Path, str]:
    """A stamped corpus plus the honest pin the production writer emits."""
    from quant_fund.research.corpus_epoch import epoch_heads_key, update_heads_pin

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _member(corpus, "a.json", '{"a": 1}')
    epochs = _chain(corpus, n_stamps=n_stamps)
    pin = tmp_path / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", epochs[-1])
    key = epoch_heads_key(corpus, "*.json")
    entry = json.loads(pin.read_text())["heads"][key]
    # The honest pin must carry every position field, else nothing below tests anything.
    assert {"tree_root", "chain_root", "n_epochs"} <= set(entry), entry
    assert entry["n_epochs"] == n_stamps
    return corpus, epochs, pin, key


def _variant_pin(pin: Path, key: str, name: str, **fields: object) -> Path:
    """Copy ``pin`` with ``fields`` overwritten on ``key`` (a field set to the
    sentinel ``DROP`` is removed). The head's receipt and sha256 are untouched."""
    doc = json.loads(pin.read_text())
    for fname, value in fields.items():
        if value is DROP:
            doc["heads"][key].pop(fname, None)
        else:
            doc["heads"][key][fname] = value
    out = pin.with_name(name)
    out.write_text(json.dumps(doc))
    return out


DROP = object()


def _both_verdicts(
    corpus: Path, pin: Path, key: str
) -> tuple[list[str], list[str], subprocess.CompletedProcess[str]]:
    """(library errors, script errors, script proc) for one pin — the library
    side goes through ``load_heads_pin`` exactly as ``verify-repo`` does."""
    from quant_fund.research.corpus_epoch import load_heads_pin

    lib = _lib_errors(corpus, expected_head=load_heads_pin(pin)[key])
    proc = _run("--corpus-dir", str(corpus), "--pin", str(pin), "--key", key)
    return lib, _errors_of(proc), proc


def test_honest_full_pin_agrees(tmp_path: Path) -> None:
    """No false positive: the pin ``update_heads_pin`` writes is clean in both."""
    corpus, _, pin, key = _pinned_corpus(tmp_path)
    lib, script, proc = _both_verdicts(corpus, pin, key)
    assert lib == [], lib
    assert script == [], script
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    ("lies", "labels"),
    [
        (lambda honest: {"chain_root": "f" * 64}, {POS_CHAIN}),
        (lambda honest: {"tree_root": "e" * 64}, {POS_TREE}),
        (lambda honest: {"n_epochs": honest["n_epochs"] + 7}, {POS_N}),
        (lambda honest: {"n_epochs": honest["n_epochs"] - 1}, {POS_N}),
        (
            lambda honest: {
                "chain_root": "f" * 64,
                "tree_root": "e" * 64,
                "n_epochs": 999,
            },
            {POS_TREE, POS_CHAIN, POS_N},
        ),
    ],
    ids=["chain_root", "tree_root", "n_epochs_high", "n_epochs_low", "all_three"],
)
def test_lying_position_fields_agree(tmp_path: Path, lies, labels: set[str]) -> None:
    """Right head, lying position fields: library and standalone reject with
    the identical errors. (Before the fix the standalone auditor exited 0.)"""
    corpus, epochs, pin, key = _pinned_corpus(tmp_path)
    honest = json.loads(pin.read_text())["heads"][key]
    bad = _variant_pin(pin, key, "lying_pin.json", **lies(honest))
    lib, script, proc = _both_verdicts(corpus, bad, key)
    expected = sorted(f"{label}:{epochs[-1].name!a}" for label in labels)
    assert sorted(lib) == expected, lib
    assert sorted(script) == expected, script
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout


@pytest.mark.parametrize("honest_prefix", [0, 1], ids=["genesis", "second_epoch"])
def test_ancestor_pin_is_checked_against_the_truncated_chain(
    tmp_path: Path, honest_prefix: int
) -> None:
    """A pin at a non-tip ancestor commits to the chain *up to that head*:
    ``chain_root`` / ``n_epochs`` over the prefix are clean in both
    implementations, and the whole-chain values (a pin lying about position)
    are flagged in both. The auditor must truncate at the pinned head."""
    from quant_fund.research.epoch_merkle import chain_tree_root, ordered_epoch_chain

    corpus, epochs, pin, key = _pinned_corpus(tmp_path)
    full = json.loads(pin.read_text())["heads"][key]
    ordered, order_errors = ordered_epoch_chain(corpus, pattern="*.json")
    assert not order_errors, order_errors
    assert [e[0] for e in ordered] == [e.name for e in epochs]
    anc = epochs[honest_prefix]
    prefix = [(e[0], e[1]) for e in ordered[: honest_prefix + 1]]
    anc_body = json.loads(anc.read_text())
    honest = {
        "receipt": anc.name,
        "sha256": hashlib.sha256(anc.read_bytes()).hexdigest(),
        "tree_root": anc_body["member_tree_root"],
        "chain_root": chain_tree_root(prefix),
        "n_epochs": honest_prefix + 1,
    }
    assert honest["chain_root"] != full["chain_root"]

    def pin_with(name: str, **over: object) -> Path:
        out = tmp_path / name
        out.write_text(json.dumps({"heads": {key: {**honest, **over}}}))
        return out

    lib, script, proc = _both_verdicts(corpus, pin_with("anc_ok.json"), key)
    assert lib == script == [], (lib, script)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    for field in ("chain_root", "n_epochs", "tree_root"):
        bad = pin_with(f"anc_bad_{field}.json", **{field: full[field]})
        lib, script, proc = _both_verdicts(corpus, bad, key)
        label = {"chain_root": POS_CHAIN, "n_epochs": POS_N, "tree_root": POS_TREE}[field]
        assert lib == script == [f"{label}:{anc.name!a}"], (field, lib, script)
        assert proc.returncode == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chain_root", 123),
        ("chain_root", "abc"),
        ("chain_root", ["f" * 64]),
        ("chain_root", "F" * 64),  # 64 chars but not the lowercase-hex form
        ("tree_root", ["x"]),
        ("tree_root", "e" * 63),
        ("n_epochs", "3"),
        ("n_epochs", 0),
        ("n_epochs", -1),
        ("n_epochs", 3.0),
        ("n_epochs", True),  # bool is an int; it must not pass as a count of 1
    ],
)
def test_malformed_pin_fields_fail_closed(tmp_path: Path, field: str, value: object) -> None:
    """A present-but-malformed position field is an error, never skipped."""
    from quant_fund.research.corpus_epoch import load_heads_pin

    corpus, _, pin, key = _pinned_corpus(tmp_path)
    bad = _variant_pin(pin, key, "malformed_pin.json", **{field: value})

    proc = _run("--corpus-dir", str(corpus), "--pin", str(bad), "--key", key)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"{POS_MALFORMED}:{field!a}" in _errors_of(proc), proc.stdout
    assert "Traceback" not in proc.stderr, proc.stderr

    # The library rejects the same pin too — at load (ValueError) or by check.
    try:
        lib = _lib_errors(corpus, expected_head=load_heads_pin(bad)[key])
    except ValueError:
        lib = ["rejected_at_load"]
    assert lib, "library accepted a pin the standalone auditor rejects"


@pytest.mark.parametrize("dropped", [("tree_root",), ("chain_root", "n_epochs")])
def test_partially_present_position_fields_are_each_checked(
    tmp_path: Path, dropped: tuple[str, ...]
) -> None:
    """Each field present is verified on its own; absent fields are not
    invented (legacy pin shape), exactly as the library treats them."""
    corpus, epochs, pin, key = _pinned_corpus(tmp_path)
    all_lies = {"tree_root": "e" * 64, "chain_root": "f" * 64, "n_epochs": 999}
    labels = {"tree_root": POS_TREE, "chain_root": POS_CHAIN, "n_epochs": POS_N}
    over: dict[str, object] = {f: DROP for f in dropped}
    over.update({f: v for f, v in all_lies.items() if f not in dropped})
    bad = _variant_pin(pin, key, "partial_pin.json", **over)
    lib, script, proc = _both_verdicts(corpus, bad, key)
    expected = sorted(f"{labels[f]}:{epochs[-1].name!a}" for f in all_lies if f not in dropped)
    assert sorted(lib) == sorted(script) == expected, (lib, script)
    assert proc.returncode == 1


def _reseal(path: Path, **changes: object) -> None:
    """Rewrite a flat epoch receipt with ``changes`` and a fresh valid seal."""
    doc = json.loads(path.read_text())
    doc.update(changes)
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    doc["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize("shape", ["cycle", "orphan"])
def test_pin_over_unwalkable_chain_is_unverifiable(tmp_path: Path, shape: str) -> None:
    """Position fields cannot be recomputed over a chain whose prev links do
    not reach genesis (a cycle, or a link to a receipt that is not there):
    the auditor says so instead of skipping the check."""
    corpus, epochs, _, _ = _pinned_corpus(tmp_path, n_stamps=2)
    genesis, head = epochs
    if shape == "cycle":
        _reseal(genesis, verdict="advancing", prev_epoch_receipt=head.name)
    else:
        _reseal(head, prev_epoch_receipt="corpus_epoch_missing.json")
    pin = tmp_path / "unwalkable_pin.json"
    key = "receipts/*.json"
    pin.write_text(
        json.dumps(
            {
                "heads": {
                    key: {
                        "receipt": head.name,
                        "sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
                        "chain_root": "f" * 64,
                    }
                }
            }
        )
    )
    proc = _run("--corpus-dir", str(corpus), "--pin", str(pin), "--key", key)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"{POS_UNVERIFIABLE}:{head.name!a}" in _errors_of(proc), proc.stdout
    assert "Traceback" not in proc.stderr, proc.stderr


def _load_script_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_epoch_chain_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chain_tree_root_mirror_matches_library() -> None:
    """The auditor carries its own copy of the chain-tree construction (it may
    not import the repo). This pins the copy to the library's root for every
    shape of tree, including odd tails and non-ASCII receipt names."""
    from quant_fund.research.epoch_merkle import chain_tree_root

    script = _load_script_module()
    names = ["corpus_epoch_é中.json", "corpus_epoch_ascii.json", 'corpus_epoch_q"\\.json']
    for n in range(1, 41):
        chain = [
            (f"{names[i % 3]}{i:03d}", hashlib.sha256(f"epoch-{i}".encode()).hexdigest())
            for i in range(n)
        ]
        assert script._chain_tree_root(chain) == chain_tree_root(chain), n
    # position is bound into the leaf: a reordered chain has a different root
    chain = [(f"r{i}.json", hashlib.sha256(bytes([i])).hexdigest()) for i in range(5)]
    assert script._chain_tree_root(chain) != script._chain_tree_root(chain[::-1])
    with pytest.raises(ValueError):
        script._chain_tree_root([])
    with pytest.raises(ValueError):
        chain_tree_root([])


_CHEAP_REAL_KEYS = (
    ".cursor/*",
    "clients/*",
    "deploy/*",
    "docker/*",
    "spec/*",
    "typings/*",
    "verifier/*.md",
)


def _real_pin_position_errors(keys) -> dict[str, list[str]]:
    pin = REPO_ROOT / "quality" / "epoch_heads.json"
    heads = json.loads(pin.read_text())["heads"]
    out: dict[str, list[str]] = {}
    for key in keys:
        if key not in heads:
            continue
        # A pin that dropped a position field would make this check vacuous.
        assert {"tree_root", "chain_root", "n_epochs"} <= set(heads[key]), key
        corpus_dir, pattern = key.rsplit("/", 1)
        proc = _run(
            "--corpus-dir",
            corpus_dir,
            "--pattern",
            pattern,
            "--pin",
            str(pin),
            "--key",
            key,
            "--allow-member-updates",  # member drift is not what this asserts
        )
        out[key] = [e for e in _errors_of(proc) if e.split(":", 1)[0] in POSITION_LABELS]
    return out


def test_committed_pin_position_fields_verify_standalone_cheap_corpora() -> None:
    """The repo's own committed pin satisfies the standalone position leg."""
    found = _real_pin_position_errors(_CHEAP_REAL_KEYS)
    assert found, "none of the cheap corpora are pinned; update _CHEAP_REAL_KEYS"
    assert {k: v for k, v in found.items() if v} == {}


@pytest.mark.slow
def test_committed_pin_position_fields_verify_standalone_every_key() -> None:
    heads = json.loads((REPO_ROOT / "quality" / "epoch_heads.json").read_text())["heads"]
    found = _real_pin_position_errors(sorted(heads))
    assert set(found) == set(heads)
    assert {k: v for k, v in found.items() if v} == {}


# --- v2 quorum checkpoints --------------------------------------------------


def _v2_tree(tmp_path: Path, *, threshold: int = 1, n_keys: int = 1):
    """A corpus stamped + pinned, a quorum registry, and a v2 checkpoint."""
    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.gate_signatures import generate_keypair, init_quorum
    from quant_fund.research.integrity_checkpoint import write_checkpoint

    root = tmp_path / "tree"
    (root / "quality").mkdir(parents=True)
    corpus = root / "receipts"
    corpus.mkdir()
    _member(corpus, "a.json", '{"a": 1}')
    pin = root / "quality" / "epoch_heads.json"
    ep = write_epoch_receipt(corpus_epoch(corpus), corpus)
    update_heads_pin(pin, "receipts", "*.json", ep)
    pairs = [generate_keypair() for _ in range(n_keys)]
    init_quorum(root, [pub for _, pub in pairs], threshold=threshold)
    (root / "quality" / "gate_signing.pub").write_text(pairs[0][1] + "\n")
    write_checkpoint(root, pairs[:threshold])
    return root, pairs


def _script_checkpoint_run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run(
        "--corpus-dir",
        str(root / "receipts"),
        "--checkpoint",
        str(root / "quality" / "checkpoint.json"),
        "--pubkey",
        str(root / "quality" / "gate_signing.pub"),
        *extra,
    )


def test_checkpoint_v2_quorum_baseline(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    root, _ = _v2_tree(tmp_path)
    proc = _script_checkpoint_run(root, "--quorum", str(root / "quality" / "gate_quorum.json"))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_checkpoint_v2_without_registry_single_key(tmp_path: Path) -> None:
    """No --quorum: v2 falls back to a 1-of-1 registry built from --pubkey."""
    pytest.importorskip("cryptography")
    root, _ = _v2_tree(tmp_path)
    proc = _script_checkpoint_run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_checkpoint_v2_below_threshold_fails(tmp_path: Path) -> None:
    """Raising the quorum threshold after signing invalidates the
    checkpoint — the payload-bound registry digest no longer matches and
    the signature count is below the new threshold."""
    pytest.importorskip("cryptography")
    from quant_fund.research.gate_signatures import init_quorum

    root, pairs = _v2_tree(tmp_path, threshold=1, n_keys=2)
    init_quorum(root, [pub for _, pub in pairs], threshold=2)
    proc = _script_checkpoint_run(root, "--quorum", str(root / "quality" / "gate_quorum.json"))
    assert proc.returncode == 1
    assert "quorum_registry_drift" in proc.stdout or "checkpoint_quorum_not_met" in proc.stdout


def test_checkpoint_v2_foreign_pubkey_fails(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from quant_fund.research.gate_signatures import generate_keypair

    root, _ = _v2_tree(tmp_path)
    _, foreign_pub = generate_keypair()
    (root / "quality" / "gate_signing.pub").write_text(foreign_pub + "\n")
    proc = _script_checkpoint_run(root, "--quorum", str(root / "quality" / "gate_quorum.json"))
    assert proc.returncode == 1
    assert "pubkey_not_in_quorum" in proc.stdout


def test_checkpoint_v2_tampered_payload_fails(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    root, _ = _v2_tree(tmp_path)
    cp = root / "quality" / "checkpoint.json"
    body = json.loads(cp.read_text())
    body["payload"]["heads"]["receipts/*.json"] = "0" * 64
    cp.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    proc = _script_checkpoint_run(root, "--quorum", str(root / "quality" / "gate_quorum.json"))
    assert proc.returncode == 1


def test_checkpoint_v2_registry_swap_fails(tmp_path: Path) -> None:
    """A different quorum file listing the same keys must not satisfy the
    checkpoint — the payload binds the registry's canonical digest."""
    pytest.importorskip("cryptography")
    from quant_fund.research.gate_signatures import registry_file_bytes

    root, pairs = _v2_tree(tmp_path)
    # Same keys, same threshold — but a byte-different registry (labels added).
    swapped = tmp_path / "alt_quorum.json"
    reg = json.loads((root / "quality" / "gate_quorum.json").read_text())
    reg["keys"][0]["label"] = "attacker-controlled"
    swapped.write_bytes(registry_file_bytes(reg))
    proc = _script_checkpoint_run(root, "--quorum", str(swapped))
    assert proc.returncode == 1
    assert "quorum_registry_drift" in proc.stdout


def test_checkpoint_v2_signed_heads_with_lying_position_fields_fail(tmp_path: Path) -> None:
    """The signed checkpoint's ``heads`` entry feeds the same position leg:
    a signature over a pin that lies about its chain position is not enough."""
    pytest.importorskip("cryptography")
    from quant_fund.research.epoch_merkle import chain_tree_root
    from quant_fund.research.integrity_checkpoint import write_checkpoint

    root, pairs = _v2_tree(tmp_path)
    pin_path = root / "quality" / "epoch_heads.json"
    (head_receipt,) = sorted((root / "receipts").glob("corpus_epoch_*.json"))
    honest_chain_root = chain_tree_root(
        [(head_receipt.name, hashlib.sha256(head_receipt.read_bytes()).hexdigest())]
    )
    quorum = str(root / "quality" / "gate_quorum.json")

    def sign_pin_with(**fields: object) -> subprocess.CompletedProcess[str]:
        doc = json.loads(pin_path.read_text())
        doc["heads"]["receipts/*.json"].update(fields)
        pin_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        write_checkpoint(root, pairs[:1])
        return _script_checkpoint_run(root, "--quorum", quorum)

    proc = sign_pin_with(chain_root=honest_chain_root, n_epochs=1)
    assert proc.returncode == 0, proc.stdout + proc.stderr  # positive control

    for lie, label in (
        ({"chain_root": "f" * 64}, POS_CHAIN),
        ({"n_epochs": 5}, POS_N),
        ({"tree_root": "e" * 64}, POS_TREE),
    ):
        proc = sign_pin_with(**{"chain_root": honest_chain_root, "n_epochs": 1, **lie})
        assert proc.returncode == 1, (lie, proc.stdout + proc.stderr)
        assert f"{label}:{head_receipt.name!a}" in _errors_of(proc), (lie, proc.stdout)
