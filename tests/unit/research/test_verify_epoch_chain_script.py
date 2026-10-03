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


def _script_errors(corpus: Path, *extra: str) -> list[str]:
    proc = _run("--corpus-dir", str(corpus), *extra)
    return [
        ln[2:]
        for ln in proc.stdout.splitlines()
        if ln.startswith("  ") and not ln.startswith("  unstamped:")
    ]


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
