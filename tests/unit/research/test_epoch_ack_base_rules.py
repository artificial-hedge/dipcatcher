"""Second-review rules for append-only epoch chains.

* Resurrection: a stamped name removed in one epoch and re-added with
  different bytes later is a mutation spread over two epochs — an
  acknowledged removal must not launder a rewrite.
* Ledger vs base: a new acknowledgment for an append-only chain may only
  explain history the BASE chain already recorded, for every transition
  type (``ack_ledger_gate``, run on every pull request and push to ``main``
  by ``make admission-gate``).
* ``chain_ancestry`` is authenticated against the pin's ``chain_root``, so
  the set of "epochs the base already held" cannot be shaped by editing
  older epoch files.

Library and standalone script must agree on every verdict here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.admission import admit_batch
from quant_fund.research.corpus_epoch import (
    ack_ledger_base_errors,
    chain_ancestry,
    check_epoch_chain,
    corpus_epoch,
    load_heads_pin,
    member_digests,
    pinned_head_members,
    update_heads_pin,
    write_epoch_receipt,
)
from quant_fund.research.repo_integrity import ack_ledger_gate, append_only_chain_keys
from quant_fund.utils.receipt import seal_receipt

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_epoch_chain.py"


def _write(path: Path, marker: str) -> Path:
    body = {
        "kind": "synthetic_fixture.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "marker": marker,
    }
    path.write_text(json.dumps(seal_receipt(body), indent=2, sort_keys=True) + "\n")
    return path


def _stamp(corpus: Path) -> Path:
    return write_epoch_receipt(corpus_epoch(corpus), corpus)


def _ack(
    corpus_key: str, member: str, epoch: str, transition: str, frm: str, to: str | None
) -> dict[str, Any]:
    return {
        "corpus": corpus_key,
        "member": member,
        "transition": transition,
        "from_sha256": frm,
        "to_sha256": to,
        "epoch": epoch,
        "reason": "test fixture",
    }


def _ledger(path: Path, acks: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"schema": "epoch_acknowledgments.v1", "acks": acks}, indent=2))
    return path


def _script(corpus: Path, ledger: Path | None, *extra: str) -> tuple[list[str], list[str]]:
    args = [sys.executable, str(SCRIPT), "--corpus-dir", str(corpus), *extra]
    if ledger is not None:
        args += ["--epoch-acks", str(ledger)]
    proc = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)
    lines = proc.stdout.splitlines()
    errors = [ln[2:] for ln in lines if ln.startswith("  ") and not ln.startswith("  unstamped:")]
    errors = [e for e in errors if not e.startswith("acknowledged:")]
    acked = [ln.split("acknowledged:", 1)[1].strip() for ln in lines if "acknowledged:" in ln]
    return errors, acked


# ---------------------------------------------------------------- resurrection


def _resurrected_corpus(tmp_path: Path, *, same_bytes: bool) -> tuple[Path, dict[str, str]]:
    """genesis {a, b} → epoch removing a → epoch re-adding a (new or same bytes)."""
    corpus = tmp_path / "c"
    corpus.mkdir()
    _write(corpus / "a.json", "v1")
    _write(corpus / "b.json", "keep")
    first = _stamp(corpus)
    v1 = member_digests(corpus)["a.json"]
    saved = (corpus / "a.json").read_bytes()
    (corpus / "a.json").unlink()
    removal = _stamp(corpus)
    if same_bytes:
        (corpus / "a.json").write_bytes(saved)
    else:
        _write(corpus / "a.json", "v2-rewritten")
    readd = _stamp(corpus)
    return corpus, {
        "first": first.name,
        "removal": removal.name,
        "readd": readd.name,
        "v1": v1,
        "v2": member_digests(corpus)["a.json"],
    }


def test_removal_then_readd_with_new_bytes_is_resurrection(tmp_path: Path) -> None:
    corpus, ep = _resurrected_corpus(tmp_path, same_bytes=False)
    key = f"{corpus.as_posix()}/*.json"
    removal_ack = _ack(key, "a.json", ep["removal"], "removed", ep["v1"], None)
    result = check_epoch_chain(corpus, acks=[removal_ack])
    label = f"member_resurrected:'a.json'@{ep['readd']!a}"
    assert label in result["errors"]
    # The acknowledged removal is still acknowledged — it is the re-add
    # that launders, and it is what fails.
    assert f"member_removed:'a.json'@{ep['removal']!a}" in result["acknowledged"]
    lib_errors = sorted(result["errors"])
    script_errors, _ = _script(corpus, _ledger(tmp_path / "acks.json", [removal_ack]))
    assert sorted(script_errors) == lib_errors


def test_resurrection_is_acknowledgeable_only_as_exact_mutation(tmp_path: Path) -> None:
    corpus, ep = _resurrected_corpus(tmp_path, same_bytes=False)
    key = f"{corpus.as_posix()}/*.json"
    acks = [
        _ack(key, "a.json", ep["removal"], "removed", ep["v1"], None),
        _ack(key, "a.json", ep["readd"], "mutated", ep["v1"], ep["v2"]),
    ]
    result = check_epoch_chain(corpus, acks=acks)
    assert result["errors"] == []
    assert f"member_resurrected:'a.json'@{ep['readd']!a}" in result["acknowledged"]
    script_errors, script_acked = _script(corpus, _ledger(tmp_path / "acks.json", acks))
    assert script_errors == []
    assert sorted(script_acked) == sorted(result["acknowledged"])
    # Wrong digests explain nothing: the resurrection stays an error and the
    # ack is reported unmatched.
    wrong = [acks[0], _ack(key, "a.json", ep["readd"], "mutated", ep["v1"], "e" * 64)]
    errors = check_epoch_chain(corpus, acks=wrong)["errors"]
    assert f"member_resurrected:'a.json'@{ep['readd']!a}" in errors
    assert f"epoch_ack_unmatched:'a.json'@{ep['readd']!a}" in errors


def test_readd_of_identical_bytes_is_not_resurrection(tmp_path: Path) -> None:
    corpus, ep = _resurrected_corpus(tmp_path, same_bytes=True)
    key = f"{corpus.as_posix()}/*.json"
    acks = [_ack(key, "a.json", ep["removal"], "removed", ep["v1"], None)]
    result = check_epoch_chain(corpus, acks=acks)
    assert result["errors"] == []
    script_errors, _ = _script(corpus, _ledger(tmp_path / "acks.json", acks))
    assert script_errors == []


def test_mutable_chain_is_outside_the_resurrection_rule(tmp_path: Path) -> None:
    corpus, ep = _resurrected_corpus(tmp_path, same_bytes=False)
    key = f"{corpus.as_posix()}/*.json"
    acks = [_ack(key, "a.json", ep["removal"], "removed", ep["v1"], None)]
    result = check_epoch_chain(corpus, acks=acks, allow_member_updates=True)
    assert not any("resurrected" in e for e in result["errors"])
    script_errors, _ = _script(
        corpus, _ledger(tmp_path / "acks.json", acks), "--allow-member-updates"
    )
    assert not any("resurrected" in e for e in script_errors)


def test_readd_under_another_case_is_resurrection(tmp_path: Path) -> None:
    """On the Windows fleet host ``A.json`` opens what ``a.json`` named."""
    corpus = tmp_path / "c"
    corpus.mkdir()
    _write(corpus / "a.json", "v1")
    _write(corpus / "b.json", "keep")
    _stamp(corpus)
    v1 = member_digests(corpus)["a.json"]
    (corpus / "a.json").unlink()
    removal = _stamp(corpus).name
    _write(corpus / "A.json", "v2-rewritten")
    readd = _stamp(corpus).name
    key = f"{corpus.as_posix()}/*.json"
    acks = [_ack(key, "a.json", removal, "removed", v1, None)]
    errors = check_epoch_chain(corpus, acks=acks)["errors"]
    assert f"member_resurrected:'A.json'@{readd!a}" in errors
    script_errors, _ = _script(corpus, _ledger(tmp_path / "acks.json", acks))
    assert sorted(script_errors) == sorted(errors)


def test_unstamped_resurrection_is_caught_before_the_next_stamp(tmp_path: Path) -> None:
    """The PR re-adds a removed name with new bytes and does not stamp it.
    The chain flags it live (not acknowledgeable), and admission refuses it."""
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _write(corpus / "a.json", "v1")
    _write(corpus / "b.json", "keep")
    _stamp(corpus)
    v1 = member_digests(corpus)["a.json"]
    (corpus / "a.json").unlink()
    removal = _stamp(corpus)
    pin = tmp_path / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", removal)
    entry = load_heads_pin(pin)[f"{corpus.as_posix()}/*.json"]
    key = f"{corpus.as_posix()}/*.json"
    acks = [_ack(key, "a.json", removal.name, "removed", v1, None)]
    _write(corpus / "a.json", "v2-rewritten")
    result = check_epoch_chain(corpus, acks=acks, expected_head=entry)
    assert result["errors"] == ["member_resurrected_live:'a.json'"]
    script_errors, _ = _script(corpus, _ledger(tmp_path / "acks.json", acks))
    assert script_errors == result["errors"]
    batch = admit_batch([corpus / "a.json"], corpus, expected_head=entry, epoch_acks=acks)
    assert batch["n_gated"] == 1 and batch["verdict"] != "admit"
    # Negative control: the same bytes back is a restoration, not flagged.
    (corpus / "a.json").unlink()
    _write(corpus / "a.json", "v1")
    assert check_epoch_chain(corpus, acks=acks, expected_head=entry)["errors"] == []


def test_duplicate_keys_in_an_epoch_fail_both_verifiers(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    _write(corpus / "a.json", "1")
    epoch = _stamp(corpus)
    raw = epoch.read_text()
    # A second "members" key: last-wins JSON would read an empty membership.
    doctored = raw.replace('"members":', '"members": [], "members":', 1)
    assert doctored != raw
    epoch.write_text(doctored)
    lib = check_epoch_chain(corpus)["errors"]
    script_errors, _ = _script(corpus, None)
    assert f"epoch_receipt_invalid:{epoch.name!a}" in lib
    assert f"epoch_receipt_invalid:{epoch.name!a}" in script_errors


# ------------------------------------------------- authenticated chain_ancestry


def _pinned_chain(tmp_path: Path, n: int = 3) -> tuple[Path, dict[str, Any], list[Path]]:
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    epochs = []
    for i in range(n):
        _write(corpus / f"r{i}.json", str(i))
        epochs.append(_stamp(corpus))
    pin = tmp_path / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", epochs[-1])
    return corpus, load_heads_pin(pin)[f"{corpus.as_posix()}/*.json"], epochs


def test_chain_ancestry_returns_the_pinned_chain(tmp_path: Path) -> None:
    corpus, entry, epochs = _pinned_chain(tmp_path)
    assert chain_ancestry(corpus, entry) == frozenset(e.name for e in epochs)


def test_chain_ancestry_rejects_an_edited_older_epoch(tmp_path: Path) -> None:
    """Same JSON, different bytes: the file digest feeds the chain_root."""
    corpus, entry, epochs = _pinned_chain(tmp_path)
    older = epochs[0]
    older.write_text(json.dumps(json.loads(older.read_text()), indent=4))
    with pytest.raises(ValueError, match="chain_root"):
        chain_ancestry(corpus, entry)


def test_chain_ancestry_fails_closed(tmp_path: Path) -> None:
    corpus, entry, epochs = _pinned_chain(tmp_path)
    with pytest.raises(ValueError, match="chain_root"):
        chain_ancestry(corpus, {k: v for k, v in entry.items() if k != "chain_root"})
    with pytest.raises(ValueError, match="pinned digest"):
        chain_ancestry(corpus, {**entry, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="pin says"):
        chain_ancestry(corpus, {**entry, "n_epochs": 99})
    with pytest.raises(ValueError, match="missing"):
        chain_ancestry(corpus, {**entry, "receipt": "../epoch_heads.json"})
    epochs[1].write_bytes(b"[" * 100_000 + b"]" * 100_000)
    with pytest.raises(ValueError, match="parseable"):
        chain_ancestry(corpus, entry)
    epochs[1].unlink()
    with pytest.raises(ValueError, match="missing"):
        chain_ancestry(corpus, entry)


# --------------------------------------------------------- ledger vs base chain


def test_ack_ledger_base_errors_covers_every_transition() -> None:
    key = "receipts/*.json"
    base_chain = {key: frozenset({"corpus_epoch_" + "a" * 16 + ".json"})}
    old_epoch = "corpus_epoch_" + "a" * 16 + ".json"
    new_epoch = "corpus_epoch_" + "b" * 16 + ".json"
    removal = _ack(key, "x.json", new_epoch, "removed", "1" * 64, None)
    mutation = _ack(key, "y.json", new_epoch, "mutated", "1" * 64, "2" * 64)
    historical = _ack(key, "z.json", old_epoch, "removed", "1" * 64, None)
    errors = ack_ledger_base_errors(
        [removal, mutation, historical], [], base_chain, frozenset({key})
    )
    assert errors == [
        f"epoch_ack_postdates_base:{key!a}:'x.json'@{new_epoch!a}",
        f"epoch_ack_postdates_base:{key!a}:'y.json'@{new_epoch!a}",
    ]
    # An entry the base ledger already held is reviewed history, unchanged.
    assert ack_ledger_base_errors([removal], [removal], {}, frozenset({key})) == []
    # Editing it (here: its reason) makes it new again.
    edited = {**removal, "reason": "rewritten"}
    assert ack_ledger_base_errors([edited], [removal], {}, frozenset({key}))
    # Mutable chains are outside the rule.
    assert ack_ledger_base_errors([removal], [], {}, frozenset()) == []


def test_append_only_keys_match_the_corpus_policy() -> None:
    keys = append_only_chain_keys()
    assert {"receipts/*.json", "verifier/*.md"} <= keys
    assert "quality/*.json" not in keys and "src/*" not in keys


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A repo root with a stamped receipts chain; returns (root, base pin, corpus)."""
    root = tmp_path / "repo"
    (root / "quality").mkdir(parents=True)
    corpus = root / "receipts"
    corpus.mkdir()
    monkeypatch.chdir(root)
    _write(corpus / "stamped.json", "evidence")
    _write(corpus / "other.json", "other")
    head = _stamp(Path("receipts"))
    update_heads_pin(Path("quality/epoch_heads.json"), Path("receipts"), "*.json", head)
    base_pin = tmp_path / "base_epoch_heads.json"
    shutil.copy(root / "quality" / "epoch_heads.json", base_pin)
    return root, base_pin, corpus


def test_removal_laundering_is_rejected_by_the_pr_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PR deletes a stamped receipt, stamps the removal in a NEW epoch and
    acknowledges it in the same diff. The chain verifier alone accepts that
    (the ack matches exactly); the base rule does not."""
    root, base_pin, corpus = _repo(tmp_path, monkeypatch)
    sha = member_digests(corpus)["stamped.json"]
    (corpus / "stamped.json").unlink()
    new_epoch = _stamp(Path("receipts")).name
    ledger = _ledger(
        root / "quality" / "epoch_allowed_removals.json",
        [_ack("receipts/*.json", "stamped.json", new_epoch, "removed", sha, None)],
    )
    assert (
        check_epoch_chain(Path("receipts"), acks=json.loads(ledger.read_text())["acks"])["errors"]
        == []
    )
    result = ack_ledger_gate(root, base_heads_pin=base_pin, base_ledger=None, ledger=ledger)
    assert result["ok"] is False
    assert result["errors"] == [
        f"epoch_ack_postdates_base:'receipts/*.json':'stamped.json'@{new_epoch!a}"
    ]
    # The CLI is the PR gate's entry point and fails the same way.
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    out = CliRunner().invoke(
        cli,
        ["epoch-ack-gate", "--base-heads-pin", str(base_pin), "--epoch-acks", str(ledger)],
    )
    assert out.exit_code == 1, out.output
    assert "epoch_ack_postdates_base" in out.output


def test_ack_for_base_history_passes_the_pr_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: the base already recorded the removal; the PR only
    adds the acknowledgment."""
    root, _, corpus = _repo(tmp_path, monkeypatch)
    sha = member_digests(corpus)["stamped.json"]
    (corpus / "stamped.json").unlink()
    removal_epoch = _stamp(Path("receipts"))
    update_heads_pin(Path("quality/epoch_heads.json"), Path("receipts"), "*.json", removal_epoch)
    base_pin = tmp_path / "base_after_removal.json"
    shutil.copy(root / "quality" / "epoch_heads.json", base_pin)
    ledger = _ledger(
        root / "quality" / "epoch_allowed_removals.json",
        [_ack("receipts/*.json", "stamped.json", removal_epoch.name, "removed", sha, None)],
    )
    # A legacy (pre-v1) base ledger reads as empty — stricter, still passes.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"stamped.json": sha}))
    result = ack_ledger_gate(root, base_heads_pin=base_pin, base_ledger=legacy, ledger=ledger)
    assert result == {
        "ok": True,
        "errors": [],
        "n_acks": 1,
        "n_base_acks": 0,
        "chains_checked": ["receipts/*.json"],
    }


def test_pr_gate_rejects_a_forged_base_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing an epoch the base pin covers makes the base chain
    unauthenticated — the gate fails instead of trusting the walk."""
    root, base_pin, corpus = _repo(tmp_path, monkeypatch)
    sha = member_digests(corpus)["stamped.json"]
    head = sorted(corpus.glob("corpus_epoch_*.json"))[0]
    head.write_text(json.dumps(json.loads(head.read_text()), indent=4))
    ledger = _ledger(
        root / "quality" / "epoch_allowed_removals.json",
        [_ack("receipts/*.json", "stamped.json", head.name, "removed", sha, None)],
    )
    result = ack_ledger_gate(root, base_heads_pin=base_pin, base_ledger=None, ledger=ledger)
    assert result["ok"] is False
    assert any(e.startswith("epoch_ack_base_chain_unauthenticated:") for e in result["errors"])


def test_pr_gate_rejects_unknown_corpus_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absolute spelling of the receipts dir is not a policy key: the gate
    could never check it against a base, yet a standalone auditor run with
    that --corpus-dir would honour it."""
    root, base_pin, corpus = _repo(tmp_path, monkeypatch)
    sha = member_digests(corpus)["stamped.json"]
    (corpus / "stamped.json").unlink()
    new_epoch = _stamp(Path("receipts")).name
    alias = f"{corpus.as_posix()}/*.json"
    ledger = _ledger(
        root / "quality" / "epoch_allowed_removals.json",
        [_ack(alias, "stamped.json", new_epoch, "removed", sha, None)],
    )
    result = ack_ledger_gate(root, base_heads_pin=base_pin, base_ledger=None, ledger=ledger)
    assert result["ok"] is False
    assert result["errors"] == [f"epoch_ack_unknown_corpus:{alias!a}"]


def test_base_held_acks_need_no_reauthentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewed history for a chain this checkout cannot authenticate (its
    epochs are local-only, like data/metadata) must not fail every PR."""
    root, base_pin, _ = _repo(tmp_path, monkeypatch)
    pin = json.loads(base_pin.read_text())
    local_only = "corpus_epoch_" + "c" * 16 + ".json"  # not in this checkout
    pin["heads"]["verifier/*.md"] = {
        "receipt": local_only,
        "sha256": "0" * 64,
        "chain_root": "0" * 64,
        "n_epochs": 1,
    }
    base_pin.write_text(json.dumps(pin))
    held = _ack(
        "verifier/*.md", "x.md", "corpus_epoch_" + "c" * 16 + ".json", "removed", "1" * 64, None
    )
    ledger = _ledger(root / "quality" / "epoch_allowed_removals.json", [held])
    base_ledger = _ledger(tmp_path / "base_acks.json", [held])
    result = ack_ledger_gate(root, base_heads_pin=base_pin, base_ledger=base_ledger, ledger=ledger)
    assert result["ok"] is True and result["chains_checked"] == []
    # The same entry, new in this PR, is checked — and that chain cannot be
    # authenticated here, so the gate fails closed.
    fresh = ack_ledger_gate(root, base_heads_pin=base_pin, base_ledger=None, ledger=ledger)
    assert fresh["ok"] is False
    assert any(
        e.startswith("epoch_ack_base_chain_unauthenticated:'verifier/*.md'")
        for e in fresh["errors"]
    )


def test_malformed_ledger_fails_the_pr_gate_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, base_pin, _ = _repo(tmp_path, monkeypatch)
    ledger = root / "quality" / "epoch_allowed_removals.json"
    ledger.write_text('{"schema": "epoch_acknowledgments.v1", "acks": [], "acks": []}')
    with pytest.raises(ValueError, match="duplicate_json_key"):
        ack_ledger_gate(root, base_heads_pin=base_pin, base_ledger=None, ledger=ledger)


def test_script_rejects_duplicate_keys_in_the_ledger(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    _write(corpus / "a.json", "1")
    _stamp(corpus)
    ledger = tmp_path / "acks.json"
    ledger.write_text('{"schema": "epoch_acknowledgments.v1", "acks": [], "acks": []}')
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus-dir", str(corpus), "--epoch-acks", str(ledger)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode != 0
    assert "duplicate_json_key" in proc.stdout + proc.stderr


# ------------------------------------------- admission: nothing gated, chain bad


def test_restoration_only_batch_still_checks_the_chain(tmp_path: Path) -> None:
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    _write(corpus / "old.json", "v1")
    _write(corpus / "gone.json", "stamped")
    head = _stamp(corpus)
    pin = tmp_path / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", head)
    entry = load_heads_pin(pin)[f"{corpus.as_posix()}/*.json"]
    stamped = pinned_head_members(corpus, entry)
    clean = admit_batch([corpus / "old.json"], corpus, base_stamped=stamped, expected_head=entry)
    assert clean["n_gated"] == 0 and clean["chain_errors"] == [] and clean["verdict"] == "admit"
    (corpus / "gone.json").unlink()  # live drift alongside a restoration-only diff
    batch = admit_batch([corpus / "old.json"], corpus, base_stamped=stamped, expected_head=entry)
    assert batch["n_gated"] == 0
    assert batch["chain_errors"] == ["head_member_missing_live:'gone.json'"]
    assert batch["verdict"] == "quarantine"


# --------------------------------------------------- preserved newer evidence


def test_preserved_receipts_match_the_migration_record() -> None:
    """Evidence kept as a new member (not overwritten onto a stamped name)
    must be the exact bytes the migration record names, and must verify."""
    from quant_fund.research.receipt_v2 import verify_receipt_file
    from quant_fund.utils.hashing import hash_bytes

    record = json.loads(
        (
            REPO_ROOT / "quality" / "evidence_migrations" / "evidence_migration_20261013.json"
        ).read_text()
    )
    preserved = record["preserved_receipts"]
    assert preserved
    for row in preserved:
        path = REPO_ROOT / row["path"]
        assert hash_bytes(path.read_bytes()) == row["sha256"]
        assert json.loads(path.read_text())["receipt_sha256"] == row["receipt_sha256"]
        assert verify_receipt_file(path)["valid"]


def test_readd_with_ntfs_trailing_dot_is_resurrection(tmp_path: Path) -> None:
    """NTFS drops trailing dots and spaces: ``a.json.`` opens ``a.json``."""
    corpus = tmp_path / "c"
    corpus.mkdir()
    _write(corpus / "a.json", "v1")
    _write(corpus / "b.json", "keep")

    def stamp() -> str:
        return write_epoch_receipt(corpus_epoch(corpus, pattern="*"), corpus).name

    stamp()
    v1 = member_digests(corpus, pattern="*")["a.json"]
    (corpus / "a.json").unlink()
    removal = stamp()
    key = f"{corpus.as_posix()}/*"
    acks = [_ack(key, "a.json", removal, "removed", v1, None)]
    _write(corpus / "a.json.", "v2-rewritten")
    errors = check_epoch_chain(corpus, acks=acks, pattern="*")["errors"]
    # Linux keeps the trailing dot as its own name. NTFS strips it, so the
    # rewrite lands on a.json and the live resurrection is reported there.
    live = set(member_digests(corpus, pattern="*"))
    resurrected = "a.json." if "a.json." in live else "a.json"
    assert errors == [f"member_resurrected_live:{resurrected!a}"]
    script_errors, _ = _script(corpus, _ledger(tmp_path / "acks.json", acks), "--pattern", "*")
    assert script_errors == errors


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_quarantine_gate_refuses_a_receipt_registered_in_its_own_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through git and the CLI, as `make admission-gate` runs it."""
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    root, base_pin, corpus = _repo(tmp_path, monkeypatch)
    _git(root, "init", "-q")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    registry = root / "quality" / "legacy_quarantine.json"

    def register(rel: str, category: str, errors: list[str]) -> None:
        sha = hashlib.sha256((corpus / rel).read_bytes()).hexdigest()
        doc = {
            "schema": "legacy_quarantine.v2",
            "files": {
                rel: {"sha256": sha, "errors": errors, "category": category, "reason": "fixture"}
            },
            "schema_drift_acks": {},
        }
        registry.write_text(json.dumps(doc))

    def gate() -> Any:
        return CliRunner().invoke(
            cli,
            ["quarantine-gate", "--base-ref", "HEAD", "--base-heads-pin", str(base_pin)],
        )

    # New receipt, new registry entry, same change: refused.
    (corpus / "new.json").write_text('{"kind": "x"}')
    register("new.json", "claim_absent", ["receipt_sha256_missing_or_invalid"])
    out = gate()
    assert out.exit_code == 1, out.output
    assert "quarantine_entry_postdates_base:'new.json'" in out.output
    # Demoting a receipt the base stamped as an honesty gap: refused.
    register("stamped.json", "admission_honesty_gap", ["data_label_missing"])
    out = gate()
    assert out.exit_code == 1, out.output
    assert "quarantine_honesty_gap_not_unadmitted_at_base:'stamped.json'" in out.output
    # Positive control: a file the base already carried, unstamped.
    (corpus / "new.json").unlink()
    (corpus / "late.json").write_text('{"kind": "y"}')
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "late arrival")
    register("late.json", "admission_honesty_gap", ["data_label_missing"])
    out = gate()
    assert out.exit_code == 0, out.output


@pytest.mark.parametrize(
    "pin",
    [None, b"not json", b'{"heads": []}', b'{"heads": {"receipts/*.json": {"receipt": null}}}'],
    ids=["absent", "not_json", "heads_not_a_mapping", "head_not_a_name"],
)
def test_epoch_consistency_fails_closed_on_an_unreadable_base_pin(
    tmp_path: Path, pin: bytes | None
) -> None:
    """`make epoch-consistency` once skipped every chain, and passed, when the
    base's heads pin was absent or malformed. The probe base is HEAD with the
    pin removed or replaced, written to a scratch object store so the repo's
    own is untouched."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    objects = tmp_path / "objects"
    objects.mkdir()
    store = {
        **os.environ,
        "GIT_OBJECT_DIRECTORY": str(objects),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": f"{common}/objects",
    }
    probe = {
        **store,
        "GIT_INDEX_FILE": str(tmp_path / "index"),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def git(*args: str, stdin: bytes | None = None) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, env=probe, input=stdin, capture_output=True, check=True
        )
        return proc.stdout.decode().strip()

    git("read-tree", "HEAD")
    if pin is None:
        git("update-index", "--force-remove", "quality/epoch_heads.json")
    else:
        blob = git("hash-object", "-w", "--stdin", stdin=pin)
        git("update-index", "--cacheinfo", f"100644,{blob},quality/epoch_heads.json")
    base = git("commit-tree", git("write-tree"), "-p", "HEAD", "-m", "probe base")
    proc = subprocess.run(
        ["make", "epoch-consistency"],
        cwd=REPO_ROOT,
        env={**store, "EPOCH_BASE": base, "RUNNER_TEMP": str(tmp_path / "tmp")},
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "skip" not in proc.stdout, out
    assert "epoch-consistency: " in proc.stdout, out
