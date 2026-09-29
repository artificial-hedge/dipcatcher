from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.repo_integrity import (
    REPO_INTEGRITY_SCHEMA,
    verify_repo,
    write_repo_integrity_receipt,
)


def _git_repo(tmp_path: Path) -> Path:
    """Minimal tree satisfying every gate: 9 jewels + 4 stamped corpora."""
    import subprocess

    from quant_fund.research.corpus_epoch import (
        HEADS_PIN_BASENAME,
        corpus_epoch,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.crown_jewels import (
        DEFAULT_JEWELS,
        write_crown_jewels_pin,
    )

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for jewel in DEFAULT_JEWELS:
        p = root / jewel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {jewel}\n")
    pin = root / "quality" / HEADS_PIN_BASENAME
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps({"schema": "epoch_heads.v1", "heads": {}}))
    for corpus_dir, pattern in (
        ("receipts", "*.json"),
        ("verifier", "*.md"),
        ("quality", "*.json"),
        (".github/workflows", "*.yml"),
        ("configs", "*"),
    ):
        d = root / corpus_dir
        d.mkdir(parents=True, exist_ok=True)
        seed = f"seed.{pattern[2:]}" if len(pattern) > 1 else "seed"
        (d / seed).write_text("seed")
        ep = write_epoch_receipt(corpus_epoch(d, pattern=pattern), d)
        update_heads_pin(pin, corpus_dir, pattern, ep)
    write_crown_jewels_pin(root)
    # Re-stamp quality: the pin and jewels file now exist as members.
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    # The committed pubkey is a crown jewel — sign the pins for real so the
    # gate isn't testing a stub. Priv key is test-local only.
    from quant_fund.research.gate_signatures import generate_keypair, sign_pins

    priv, pub = generate_keypair()
    sign_pins(root, priv, pub)
    write_crown_jewels_pin(root)
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    sign_pins(root, priv, pub)
    return root


def test_verify_repo_all_gates_green(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    res = verify_repo(root)
    assert res["ok"], res
    # Fixture signs the pins for real (the pubkey is a jewel, so an unsigned
    # fixture would leave a stub pubkey failing the gate).
    assert res["gates"]["pin_signatures"] == {"ok": True, "signed": True, "errors": []}
    # No timestamp anchors committed — neutral gate state, same contract as
    # unsigned pins: absence is fine, a malformed one must fail closed.
    assert res["gates"]["timestamp_anchors"]["anchored"] is False
    assert res["gates"]["timestamp_anchors"]["ok"] is True
    # No checkpoint committed — same neutral contract as unsigned pins.
    assert res["gates"]["checkpoint"]["signed"] is False
    assert res["gates"]["checkpoint"]["ok"] is True
    assert set(res["gates"]) == {
        "crown_jewels",
        "pin_signatures",
        "timestamp_anchors",
        "checkpoint",
        "epoch:receipts",
        "epoch:verifier",
        "epoch:quality",
        "epoch:.github/workflows",
        "epoch:configs",
    }


def test_verify_repo_fails_on_any_gate(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    (root / "pyproject.toml").write_text("tampered")
    res = verify_repo(root)
    assert not res["ok"]
    assert res["gates"]["crown_jewels"]["errors"] == ["jewel_mutated:pyproject.toml"]
    # Epochs untouched -> other gates still report clean.
    assert res["gates"]["epoch:receipts"]["ok"]


def test_repo_integrity_receipt_seals_and_verifies(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    out = root / "quality" / "repo_integrity.json"
    write_repo_integrity_receipt(out, root)
    verdict = verify_receipt_file(out)
    assert verdict["valid"], verdict["errors"]
    body = json.loads(out.read_text())
    assert body["schema"] == REPO_INTEGRITY_SCHEMA
    assert body["ok"] is True
    assert body["pins"]["epoch_heads_sha256"]
    assert body["pins"]["crown_jewels_sha256"]


def test_verify_repo_missing_pin_fails_closed(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    (root / "quality" / "epoch_heads.json").unlink()
    res = verify_repo(root)
    assert not res["ok"]
    # Quality chain loses its member + the pin is gone: jewel gate fine,
    # epoch gates fail closed.
    assert res["gates"]["crown_jewels"]["ok"]
    assert not res["gates"]["epoch:receipts"]["ok"]
    assert not res["gates"]["epoch:quality"]["ok"]


def test_repo_integrity_contract_clean_fixture(tmp_path: Path) -> None:
    """The contract accepts a fresh attestation — verify-receipt calls it."""
    from quant_fund.research.repo_integrity import (
        repo_integrity_contract_errors,
        repo_integrity_receipt,
    )

    root = _git_repo(tmp_path)
    body = repo_integrity_receipt(root)
    assert repo_integrity_contract_errors(body) == []
    # verify-receipt must dispatch to the contract — this is the bound that
    # makes a forged verdict catchable instead of structure-only.
    out = root / "quality" / "repo_integrity.json"
    write_repo_integrity_receipt(out, root)
    assert verify_receipt_file(out)["valid"]


def test_repo_integrity_contract_catches_forged_ok(tmp_path: Path) -> None:
    from quant_fund.research.repo_integrity import (
        repo_integrity_contract_errors,
        repo_integrity_receipt,
    )

    root = _git_repo(tmp_path)
    body = repo_integrity_receipt(root)
    # Forge: claim ok while a gate lists its failures.
    forged = dict(body)
    gates = dict(body["gates"])
    gates["crown_jewels"] = {"ok": False, "errors": ["jewel_mutated:x"]}
    forged["gates"] = gates
    forged["ok"] = True  # the lie: top-level ok despite a failed gate
    errs = repo_integrity_contract_errors(forged)
    assert "ok_incoherent" in errs
    # Also incoherent: gate ok=True while listing errors.
    forged2 = dict(body)
    g2 = dict(body["gates"])
    g2["epoch:receipts"] = {"ok": True, "errors": ["member_removed:x.json"]}
    forged2["gates"] = g2
    assert "gate_ok_incoherent:epoch:receipts" in repo_integrity_contract_errors(forged2)
    # Hidden surface: drop a required gate, keep ok=true.
    forged3 = dict(body)
    g3 = dict(body["gates"])
    del g3["pin_signatures"]
    forged3["gates"] = g3
    errs3 = repo_integrity_contract_errors(forged3)
    assert "gate_missing:pin_signatures" in errs3
    # Pin coherence: attestation can't disagree with its own gate.
    forged4 = dict(body)
    forged4["pins"] = dict(body["pins"], gate_pins_signed=not body["pins"]["gate_pins_signed"])
    assert "pins_sig_incoherent" in repo_integrity_contract_errors(forged4)
