"""End to end: seal, stamp, gate, admit, re-stamp and replay in a scratch repo.

The layout under ``tmp_path`` mirrors the repository: ``receipts/`` is the
corpus, ``quality/`` holds the heads pin, the acknowledgment ledger and the
quarantine registry, ``data/manifests/replay/`` holds replay carriers, and a
lane script sits at the root. Every step runs the production path: receipts
are sealed by ``quant_fund.utils.receipt.seal_receipt``, chains are stamped
by ``write_epoch_receipt``/``update_heads_pin`` and by the real
``corpus-epoch`` CLI, the gates are the real ``suite-health --strict``,
``corpus-epoch --check``, ``admit`` and ``admit-batch --strict`` commands
(``typer.testing.CliRunner``), and replay is ``run_replay_sweep`` executing
a real subprocess. Nothing is mocked; nothing touches the repo's own
``receipts/`` or ``quality/``.

The happy path is the positive control. Each negative control starts from
that same, already-passing fixture and changes exactly one thing.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner, Result

import quant_fund.research.replay_sweep as replay_sweep_mod
from quant_fund.cli.main import app
from quant_fund.research.admission import admission_check, admit_batch
from quant_fund.research.corpus_epoch import (
    acks_for_corpus,
    check_epoch_chain,
    corpus_epoch,
    epoch_heads_key,
    load_epoch_acks,
    load_heads_pin,
    member_digests,
    update_heads_pin,
    write_epoch_receipt,
)
from quant_fund.research.legacy_unsealed import load_quarantine
from quant_fund.research.receipt_v2 import seal_receipt as seal_v2
from quant_fund.research.receipt_v2 import (
    verify_receipt_file,
    verify_receipt_payload,
    wrap_receipt_v2,
)
from quant_fund.research.replay_sweep import replay_coverage_contract_errors, run_replay_sweep
from quant_fund.research.suite_health import suite_health
from quant_fund.utils.receipt import seal_receipt

_LEGACY = b'{"kind": "legacy_fixture", "note": "written before the seal convention"}\n'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lane_body(name: str, claim: str = "original") -> dict[str, Any]:
    return {
        "schema": "pipeline_fixture.v1",
        "kind": "pipeline_fixture",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "name": name,
        "claim": claim,
        "scores": {"pinball": 0.125, "crps": 0.25},
    }


def _receipt_bytes(body: dict[str, Any]) -> bytes:
    """The exact bytes a lane writes — the replay script reproduces them."""
    return (json.dumps(seal_receipt(body), indent=2, sort_keys=True) + "\n").encode("utf-8")


@dataclass(frozen=True)
class Lab:
    """A scratch repository root with the evidence layout."""

    root: Path

    @property
    def receipts(self) -> Path:
        return self.root / "receipts"

    @property
    def quality(self) -> Path:
        return self.root / "quality"

    @property
    def pin(self) -> Path:
        return self.quality / "epoch_heads.json"

    @property
    def acks(self) -> Path:
        return self.quality / "epoch_allowed_removals.json"

    @property
    def registry(self) -> Path:
        return self.quality / "legacy_quarantine.json"

    @property
    def out(self) -> Path:
        return self.root / "gate_out"

    @property
    def carriers(self) -> Path:
        return self.root / "data" / "manifests" / "replay"

    def put(self, name: str, claim: str = "original") -> bytes:
        data = _receipt_bytes(_lane_body(name, claim))
        path = self.receipts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def policy(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """(pinned head, this chain's acks) loaded the way every CLI gate loads them."""
        key = epoch_heads_key(self.receipts, "*.json")
        return load_heads_pin(self.pin)[key], acks_for_corpus(load_epoch_acks(self.acks), key)

    def write_ledger(self, acks: list[dict[str, Any]]) -> None:
        self.acks.write_text(json.dumps({"schema": "epoch_acknowledgments.v1", "acks": acks}))

    def write_registry(self, files: dict[str, Any]) -> None:
        self.registry.write_text(json.dumps({"schema": "legacy_quarantine.v2", "files": files}))


# --- the real CLI -------------------------------------------------------------


def _invoke(args: Sequence[object]) -> Result:
    return CliRunner().invoke(app, [str(a) for a in args])


def _gate_args(lab: Lab) -> list[object]:
    return ["--heads-pin", lab.pin, "--epoch-acks", lab.acks]


def suite_health_cli(lab: Lab, *extra: object) -> Result:
    return _invoke(
        [
            "suite-health",
            "--receipts-dir",
            lab.receipts,
            "--out-dir",
            lab.out,
            "--strict",
            "--quarantine",
            lab.registry,
            *_gate_args(lab),
            *extra,
        ]
    )


def check_cli(lab: Lab, *extra: object) -> Result:
    return _invoke(
        ["corpus-epoch", "--corpus-dir", lab.receipts, "--check", *_gate_args(lab), *extra]
    )


def stamp_cli(lab: Lab) -> Result:
    return _invoke(
        [
            "corpus-epoch",
            "--corpus-dir",
            lab.receipts,
            "--out-dir",
            lab.receipts,
            "--heads-pin",
            lab.pin,
        ]
    )


def admit_batch_cli(lab: Lab, *candidates: Path) -> Result:
    return _invoke(
        [
            "admit-batch",
            *candidates,
            "--corpus-dir",
            lab.receipts,
            "--out-dir",
            lab.out,
            "--strict",
            "--quarantine",
            lab.registry,
            *_gate_args(lab),
        ]
    )


def admit_cli(lab: Lab, candidate: Path) -> Result:
    return _invoke(
        [
            "admit",
            candidate,
            "--corpus-dir",
            lab.receipts,
            "--out-dir",
            lab.out,
            "--strict",
            *_gate_args(lab),
        ]
    )


def _written(result: Result) -> list[Path]:
    """Receipt paths a command reported writing (``receipt=<path>`` lines)."""
    return [Path(p) for p in re.findall(r"^receipt=(.+)$", result.output, flags=re.MULTILINE)]


def _written_body(result: Result) -> dict[str, Any]:
    """The single receipt a gate wrote — which must itself verify."""
    (path,) = _written(result)
    verification = verify_receipt_file(path)
    assert verification["valid"], verification["errors"]
    body: dict[str, Any] = json.loads(path.read_text())
    return body


# --- fixture construction -----------------------------------------------------

_EMIT_TEMPLATE = (
    "import json\n"
    "import pathlib\n"
    "\n"
    "from quant_fund.utils.receipt import seal_receipt\n"
    "\n"
    "body = json.loads(BODY)\n"
    "out = pathlib.Path('replay_out')\n"
    "out.mkdir(parents=True, exist_ok=True)\n"
    "text = json.dumps(seal_receipt(body), indent=2, sort_keys=True) + '\\n'\n"
    "(out / 'r3.json').write_bytes(text.encode('utf-8'))\n"
)


def _write_lane(lab: Lab, data: bytes) -> Path:
    """A deterministic lane script regenerating r3 + a sealed carrier binding it."""
    script = _EMIT_TEMPLATE.replace("BODY", repr(json.dumps(_lane_body("r3.json"))))
    (lab.root / "emit_r3.py").write_text(script, encoding="utf-8")
    # Replay runs tracked code only: index the lane script as a checkout would.
    if not (lab.root / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=lab.root, check=True)
    subprocess.run(["git", "add", "--", "emit_r3.py"], cwd=lab.root, check=True)
    carrier = seal_receipt(
        {
            "schema": "replay_manifest.v1",
            "kind": "replay_manifest",
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
            "replay": {
                "argv": ["emit_r3.py"],  # *.py resolves to this interpreter
                "artifacts": [{"path": "replay_out/r3.json", "sha256": _sha(data)}],
            },
            "reproduces": [
                {
                    "receipt": "receipts/r3.json",
                    "committed_sha256": _sha(data),
                    "claims_equal": True,
                }
            ],
        }
    )
    path = lab.carriers / "r3_lane.json"
    path.write_text(json.dumps(carrier, indent=2, sort_keys=True) + "\n")
    assert verify_receipt_file(path)["valid"]
    return path


def _seed(tmp_path: Path) -> tuple[Lab, Path]:
    """Two sealed receipts, a genesis epoch stamped by the library, the pin."""
    lab = Lab(tmp_path / "repo")
    lab.receipts.mkdir(parents=True)
    lab.quality.mkdir()
    lab.carriers.mkdir(parents=True)
    lab.write_ledger([])
    lab.write_registry({})
    lab.put("r1.json")
    lab.put("r2.json")
    genesis = write_epoch_receipt(corpus_epoch(lab.receipts), lab.receipts)
    update_heads_pin(lab.pin, lab.receipts, "*.json", genesis)
    return lab, genesis


def _sweep(lab: Lab) -> dict[str, Any]:
    return run_replay_sweep(lab.carriers, root=lab.root, timeout_s=120)


def _sealed_coverage(body: dict[str, Any]) -> dict[str, Any]:
    """Seal a coverage body exactly as ``dipcatcher replay-all`` does."""
    return seal_v2(
        wrap_receipt_v2(
            body,
            code_files=(Path(replay_sweep_mod.__file__),),
            verdict=body["verdict"],
            dataset={"carriers_dir": body["carriers_dir"]},
            params={"timeout_s": body["timeout_s"]},
        )
    )


@dataclass(frozen=True)
class Stamped:
    lab: Lab
    genesis: Path
    head: Path
    r3: bytes


def _advance(tmp_path: Path) -> Stamped:
    """The happy path to a stamped, pinned, replay-declared corpus (asserted
    minimally here; ``test_full_chain_end_to_end`` asserts every step)."""
    lab, genesis = _seed(tmp_path)
    r3 = lab.put("r3.json")
    admitted = admit_batch_cli(lab, lab.receipts / "r3.json")
    assert admitted.exit_code == 0, admitted.output
    stamped = stamp_cli(lab)
    assert stamped.exit_code == 0, stamped.output
    (head,) = _written(stamped)
    _write_lane(lab, r3)
    return Stamped(lab=lab, genesis=genesis, head=head, r3=r3)


@pytest.fixture()
def stamped(tmp_path: Path) -> Stamped:
    """The passing fixture every negative control mutates (positive control included)."""
    state = _advance(tmp_path)
    passing = suite_health_cli(state.lab, "--require-stamped")
    assert passing.exit_code == 0, passing.output
    assert check_cli(state.lab, "--require-stamped").exit_code == 0
    return state


# --- 9. full chain + negative controls ---------------------------------------


def test_full_chain_end_to_end(tmp_path: Path) -> None:
    """Positive control, every step asserted: sealed receipts → genesis epoch +
    heads pin → strict gates pass → a new receipt is admitted while unstamped
    → CLI re-stamp advances the chain and the pin → strict gates pass with
    ``--require-stamped`` → the replay carrier re-executes the lane, reproduces
    the receipt byte for byte, and binds to the live bytes."""
    lab, genesis = _seed(tmp_path)

    res = suite_health_cli(lab)
    assert res.exit_code == 0, res.output
    assert "receipts=3 verified=3 quarantined=0 invalid=0" in res.output
    body = _written_body(res)
    assert body["corpus_epoch"]["chain_ok"] is True and body["corpus_epoch"]["pinned"] is True
    assert body["corpus_epoch"]["head"] == genesis.name
    check = check_cli(lab, "--require-stamped")
    assert check.exit_code == 0 and "epoch-chain intact" in check.output

    # A new receipt arrives: informational unless every member must be stamped.
    r3 = lab.put("r3.json")
    assert suite_health_cli(lab).exit_code == 0
    unstamped = suite_health_cli(lab, "--require-stamped")
    assert unstamped.exit_code == 1
    assert "unstamped_member:'r3.json'" in unstamped.output

    # Admission does not require stamping: the arrival is admitted, recorded unstamped.
    admitted = admit_batch_cli(lab, lab.receipts / "r3.json")
    assert admitted.exit_code == 0, admitted.output
    assert "admission candidate=r3.json" in admitted.output and "verdict=admit" in admitted.output
    admission = _written_body(admitted)
    epoch_check = next(c for c in admission["checks"] if c["name"] == "epoch_chain")
    assert epoch_check["ok"] is True and epoch_check["unstamped"] == ["r3.json"]
    assert admission["corpus_epoch_receipt"] == genesis.name

    # Re-stamp through the CLI: the chain advances and the pin follows.
    stamp = stamp_cli(lab)
    assert stamp.exit_code == 0, stamp.output
    assert "verdict=advancing" in stamp.output
    (head,) = _written(stamp)
    pinned, _acks = lab.policy()
    assert pinned["receipt"] == head.name and pinned["n_epochs"] == 2
    assert pinned["sha256"] == _sha(head.read_bytes())
    for result in (suite_health_cli(lab, "--require-stamped"), check_cli(lab, "--require-stamped")):
        assert result.exit_code == 0, result.output
    epoch_state = _written_body(suite_health_cli(lab))["corpus_epoch"]
    assert epoch_state["head"] == head.name and epoch_state["errors"] == []

    # Replay: the carrier re-executes the lane and binds the reproduced receipt.
    _write_lane(lab, r3)
    sweep = _sweep(lab)
    (row,) = sweep["rows"]
    assert (row["verdict"], row["execution_verdict"], row["binding_errors"]) == ("pass", "pass", [])
    (record,) = row["reproduces"]
    assert record["bound"] is True and record["receipt_valid"] is True
    assert record["live_sha256"] == record["committed_sha256"] == _sha(r3)
    assert (lab.root / "replay_out" / "r3.json").read_bytes() == r3  # argv really reproduced it
    assert (sweep["verdict"], sweep["n_pass"], sweep["n_reproduces_bound"]) == ("pass", 1, 1)
    assert replay_coverage_contract_errors(sweep) == []
    sealed = _sealed_coverage(sweep)
    assert verify_receipt_payload(sealed)["valid"], verify_receipt_payload(sealed)["errors"]


def test_honestly_resealed_tamper_of_a_stamped_receipt_fails_every_gate(
    stamped: Stamped,
) -> None:
    """Negative control: rewrite the stamped r3 and re-seal it honestly. The
    receipt still verifies (a seal proves consistency, not provenance), so
    receipt validity is intact — but corpus validity is not: suite-health
    strict and the chain check fail on ``head_member_digest_drift``, the
    admission gate quarantines it, and the replay row fails binding with
    ``reproduced_receipt_drift`` even though the lane itself still reproduces
    the original bytes."""
    lab = stamped.lab
    tampered = lab.put("r3.json", claim="rewritten after stamping")
    assert tampered != stamped.r3
    assert verify_receipt_file(lab.receipts / "r3.json")["valid"]

    res = suite_health_cli(lab)
    assert res.exit_code == 1, res.output
    assert "corpus_epoch: head_member_digest_drift:'r3.json'" in res.output
    assert "quarantined=0 invalid=0" in res.output  # every receipt still verifies
    body = _written_body(res)
    assert body["n_invalid"] == 0 and body["corpus_epoch"]["chain_ok"] is False

    check = check_cli(lab)
    assert check.exit_code == 1
    assert "epoch-chain error: head_member_digest_drift:'r3.json'" in check.output

    admitted = admit_batch_cli(lab, lab.receipts / "r3.json")
    assert admitted.exit_code == 1, admitted.output
    assert "verdict=quarantine" in admitted.output
    head, acks = lab.policy()
    batch = admit_batch(
        [lab.receipts / "r3.json"],
        lab.receipts,
        quarantine=load_quarantine(lab.registry),
        expected_head=head,
        epoch_acks=acks,
    )
    (result,) = batch["results"]
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["seal"]["ok"] is True
    assert checks["epoch_chain"]["ok"] is False
    assert "head_member_digest_drift:'r3.json'" in checks["epoch_chain"]["errors"]
    assert result["verdict"] == "quarantine" and batch["failures"] == ["r3.json"]

    sweep = _sweep(lab)
    (row,) = sweep["rows"]
    assert row["execution_verdict"] == "pass" and row["all_match"] is True
    assert row["verdict"] == "fail"
    assert row["binding_errors"] == ["reproduced_receipt_drift:receipts/r3.json"]
    assert row["reproduces"][0]["live_sha256"] == _sha(tampered)
    assert sweep["verdict"] == "fail" and sweep["n_reproduces_bound"] == 0
    assert replay_coverage_contract_errors(sweep) == []


def test_unregistered_invalid_receipt_fails_strict(stamped: Stamped) -> None:
    """Negative control: a receipt whose seal no longer binds its body, and that
    no registry entry pins, is ``invalid`` — strict fails and admission
    rejects, while the chain itself stays intact (an unstamped arrival)."""
    lab = stamped.lab
    forged = json.loads(_receipt_bytes(_lane_body("bogus.json")))
    forged["claim"] = "edited after sealing"
    (lab.receipts / "bogus.json").write_text(json.dumps(forged, indent=2, sort_keys=True))

    res = suite_health_cli(lab)
    assert res.exit_code == 1, res.output
    assert "STRICT FAILURE" in res.output
    assert "bogus.json: ['receipt_sha256_mismatch']" in res.output
    body = _written_body(res)
    assert body["n_invalid"] == 1 and list(body["invalid"]) == ["bogus.json"]
    assert body["corpus_epoch"]["chain_ok"] is True  # receipt vs corpus validity stay apart
    rejected = admit_batch_cli(lab, lab.receipts / "bogus.json")
    assert rejected.exit_code == 1 and "verdict=reject" in rejected.output


def test_registered_quarantined_receipt_is_retained_never_admitted(stamped: Stamped) -> None:
    """A retained pre-seal receipt pinned in the registry (path + bytes + exact
    error set) passes strict as ``quarantined`` — excluded from ``n_ok`` and
    from pooling — and admit-batch lists it as ``retained_quarantined``,
    never as an admitted candidate. Negative control: the same bytes without
    the registry entry fail strict and are rejected at admission."""
    lab = stamped.lab
    rel = "legacy-unsealed/pre_seal.json"
    legacy = lab.receipts / rel
    legacy.parent.mkdir()
    legacy.write_bytes(_LEGACY)
    errors = sorted(set(verify_receipt_file(legacy)["errors"]))
    assert errors == ["receipt_sha256_missing_or_invalid"]
    r4 = lab.put("r4.json")
    assert r4

    # Without a registry entry: invalid, fails strict, rejected at admission.
    res = suite_health_cli(lab)
    assert res.exit_code == 1 and "quarantine_unregistered" in res.output
    rejected = admit_batch_cli(lab, legacy)
    assert rejected.exit_code == 1 and "verdict=reject" in rejected.output

    lab.write_registry(
        {
            rel: {
                "sha256": _sha(_LEGACY),
                "errors": errors,
                "category": "pre_seal_era",
                "reason": "integration fixture: retained historical record",
            }
        }
    )
    res = suite_health_cli(lab)
    assert res.exit_code == 0, res.output
    assert "quarantined=1 invalid=0" in res.output
    assert "quarantined[pre_seal_era]=1" in res.output
    body = _written_body(res)
    assert body["n_ok"] == body["n_receipts"] - 1 and body["n_quarantined"] == 1
    assert body["pooled_evalue"] is None
    frame, _receipt = suite_health(lab.receipts, quarantine=load_quarantine(lab.registry))
    statuses = {row["file"]: row["status"] for row in frame.to_dicts()}
    assert statuses[rel] == "quarantined"
    assert {s for f, s in statuses.items() if f != rel} == {"verified"}

    admitted = admit_batch_cli(lab, legacy, lab.receipts / "r4.json")
    assert admitted.exit_code == 0, admitted.output
    assert f"retained-quarantined (not admitted) candidate={rel}" in admitted.output
    assert "admission candidate=r4.json" in admitted.output
    assert "admission candidate=pre_seal.json" not in admitted.output
    assert len(_written(admitted)) == 1  # one admission receipt: r4's
    head, acks = lab.policy()
    batch = admit_batch(
        [legacy, lab.receipts / "r4.json"],
        lab.receipts,
        quarantine=load_quarantine(lab.registry),
        expected_head=head,
        epoch_acks=acks,
    )
    assert batch["retained_quarantined"] == [rel]
    assert [r["candidate"] for r in batch["results"]] == ["r4.json"]
    assert batch["results"][0]["verdict"] == "admit" and batch["failures"] == []


def test_deleting_the_newest_epoch_is_a_rollback_for_every_gate(stamped: Stamped) -> None:
    """Negative control: delete the pinned head epoch. The older chain is still
    internally consistent, but the pin turns the rewind into
    ``epoch_head_missing`` + ``epoch_head_rollback`` for suite-health, the
    chain check, single admission and batch admission alike."""
    lab, head = stamped.lab, stamped.head
    head.unlink()
    missing, rollback = f"epoch_head_missing:{head.name!a}", f"epoch_head_rollback:{head.name!a}"

    res = suite_health_cli(lab)
    assert res.exit_code == 1, res.output
    assert missing in res.output and rollback in res.output
    assert _written_body(res)["corpus_epoch"]["chain_ok"] is False
    check = check_cli(lab)
    assert check.exit_code == 1 and rollback in check.output

    incoming = lab.root / "incoming" / "r4.json"
    incoming.parent.mkdir()
    incoming.write_bytes(_receipt_bytes(_lane_body("r4.json")))
    single = admit_cli(lab, incoming)
    assert single.exit_code == 1, single.output
    assert (
        "verdict=quarantine" in single.output and "failed_checks=['epoch_chain']" in single.output
    )
    lab.put("r4.json")
    batch = admit_batch_cli(lab, lab.receipts / "r4.json")
    assert batch.exit_code == 1 and "verdict=quarantine" in batch.output

    pinned, acks = lab.policy()
    result = admission_check(incoming, lab.receipts, expected_head=pinned, epoch_acks=acks)
    epoch_check = next(c for c in result["checks"] if c["name"] == "epoch_chain")
    assert missing in epoch_check["errors"] and rollback in epoch_check["errors"]


def test_recorded_removal_needs_an_exact_ledger_entry(stamped: Stamped) -> None:
    """A removal stamped into the chain fails every gate until the ledger names
    it exactly; then each gate prints it as acknowledged and passes. An entry
    with the wrong digest explains nothing and keeps the gates red."""
    lab = stamped.lab
    stamped_digest = member_digests(lab.receipts)["r1.json"]
    (lab.receipts / "r1.json").unlink()
    restamp = stamp_cli(lab)
    assert restamp.exit_code == 0 and "verdict=shrinking" in restamp.output
    (recording,) = _written(restamp)
    removed = f"member_removed:'r1.json'@{recording.name!a}"
    assert removed in suite_health_cli(lab).output
    assert suite_health_cli(lab).exit_code == 1

    entry = {
        "corpus": epoch_heads_key(lab.receipts, "*.json"),
        "member": "r1.json",
        "transition": "removed",
        "from_sha256": "0" * 64,
        "to_sha256": None,
        "epoch": recording.name,
        "reason": "integration fixture: reviewed removal",
    }
    lab.write_ledger([entry])  # wrong digest
    wrong = suite_health_cli(lab)
    assert wrong.exit_code == 1
    assert removed in wrong.output and "epoch_ack_unmatched:'r1.json'" in wrong.output

    lab.write_ledger([{**entry, "from_sha256": stamped_digest}])
    for result in (suite_health_cli(lab), check_cli(lab)):
        assert result.exit_code == 0, result.output
        assert removed in result.output and "acknowledged" in result.output
    assert _written_body(suite_health_cli(lab))["corpus_epoch"]["acknowledged"] == [removed]


# --- 10. determinism -----------------------------------------------------------


def test_suite_health_is_deterministic(stamped: Stamped) -> None:
    """Invariant: re-auditing an unchanged corpus — verified, quarantined and
    invalid receipts plus a stamped chain — yields the identical receipt
    (dataset_sha256, per-file statuses, epoch state), and the CLI writes the
    identical receipt file twice."""
    lab = stamped.lab
    legacy = lab.receipts / "legacy-unsealed" / "pre_seal.json"
    legacy.parent.mkdir()
    legacy.write_bytes(_LEGACY)
    lab.write_registry(
        {
            "legacy-unsealed/pre_seal.json": {
                "sha256": _sha(_LEGACY),
                "errors": ["receipt_sha256_missing_or_invalid"],
                "category": "pre_seal_era",
                "reason": "integration fixture",
            }
        }
    )
    (lab.receipts / "bogus.json").write_text('{"kind": "bogus"}')
    head, acks = lab.policy()
    runs = [
        suite_health(
            lab.receipts,
            quarantine=load_quarantine(lab.registry),
            expected_head=head,
            epoch_acks=acks,
        )
        for _ in range(2)
    ]
    (frame_a, receipt_a), (frame_b, receipt_b) = runs
    assert frame_a.to_dicts() == frame_b.to_dicts()
    assert receipt_a == receipt_b
    assert (receipt_a["n_ok"], receipt_a["n_quarantined"], receipt_a["n_invalid"]) == (5, 1, 1)
    first, second = suite_health_cli(lab), suite_health_cli(lab)
    assert first.exit_code == second.exit_code == 1  # bogus.json is invalid
    (path_a,), (path_b,) = _written(first), _written(second)
    assert path_a == path_b
    assert check_epoch_chain(lab.receipts, expected_head=head, acks=acks) == check_epoch_chain(
        lab.receipts, expected_head=head, acks=acks
    )


_NAMES = [
    "a.json",
    "b.json",
    "z.json",
    "nested/c.json",
    "nested/deeper/d.json",
    "legacy-unsealed/e.json",
]


def _populate(root: Path, order: Sequence[str]) -> None:
    for name in order:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_receipt_bytes(_lane_body(name)))


@settings(max_examples=12, deadline=None, database=None)
@given(order_a=st.permutations(_NAMES), order_b=st.permutations(_NAMES))
def test_chain_roots_are_independent_of_creation_order(
    order_a: list[str], order_b: list[str]
) -> None:
    """Invariant: two corpora with the same names and bytes, created in different
    orders in different directories, have the same member map, epoch root,
    member tree root and suite-health dataset digest; each stamps a chain
    that checks clean."""
    with tempfile.TemporaryDirectory() as scratch:
        dir_a, dir_b = Path(scratch) / "a" / "receipts", Path(scratch) / "b" / "receipts"
        _populate(dir_a, order_a)
        _populate(dir_b, order_b)
        assert member_digests(dir_a) == member_digests(dir_b)
        assert list(member_digests(dir_a)) == sorted(_NAMES)
        _frame_a, health_a = suite_health(dir_a)
        _frame_b, health_b = suite_health(dir_b)
        assert health_a["dataset_sha256"] == health_b["dataset_sha256"]
        epoch_a, epoch_b = corpus_epoch(dir_a), corpus_epoch(dir_b)
        for field in ("epoch_root_sha256", "member_tree_root", "members", "n_members"):
            assert epoch_a[field] == epoch_b[field], field
        for corpus, epoch in ((dir_a, epoch_a), (dir_b, epoch_b)):
            write_epoch_receipt(epoch, corpus)
            assert check_epoch_chain(corpus, require_stamped=True)["errors"] == []
