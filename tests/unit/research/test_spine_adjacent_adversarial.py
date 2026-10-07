"""Adversarial probes for the spine-adjacent audit lane.

Each test pins one defect found during the line-level pass over the
spine-adjacent research modules — verdict envelope mapping, e-process
input bounds, lattice count re-derivation, admission delta keying,
witness-scan fail-closedness, fuzz-drill clone hygiene, monitor alarm
accounting, evalue-contract semantics, and verifier path containment.
Seeded, SYNTHETIC, self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def test_envelope_verdict_maps_confirmed_to_pass(tmp_path: Path) -> None:
    """A ``confirmed`` honest_verdict must not wrap as envelope 'blocked'."""
    from quant_fund.research.verdict_run import write_verdict_receipt

    body: dict[str, Any] = {
        "kind": "honest_verdict.v1",
        "schema": "honest_verdict.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "verdict": "confirmed",
        "winner": "gmm",
        "alpha": 0.05,
        "n_obs": 128,
        "n_heads": 2,
        "inputs_sha256": "a" * 64,
        "run": {"params": {"heads": ["a", "b"]}},
    }
    out = write_verdict_receipt(body, tmp_path, receipt_version=2)
    envelope = json.loads(out.read_text())
    assert envelope["payload"]["verdict"] == "confirmed"
    assert envelope["verdict"] == "pass"


def test_short_baseline_predictions_do_not_crash() -> None:
    """A baseline emitting fewer rows than the head must bound, not crash."""
    import numpy as np

    from quant_fund.research.mean_eprocess import audit_mean_eprocess

    class _Head:
        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            return None

        def predict(self, x: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            taus = np.linspace(0.05, 0.95, 9)
            return np.tile(taus - 0.5, (n, 1))

    class _ShortBase(_Head):
        def predict(self, x: np.ndarray) -> np.ndarray:
            # returns fewer rows than requested — the hostile shape
            full = super().predict(x)
            return full[: max(1, full.shape[0] - 3)]

    from quant_fund.research.fleet_eval import SyntheticShard

    def shard(n: int, seed: int) -> SyntheticShard:
        rng = np.random.default_rng(seed)
        return SyntheticShard(
            name="s",
            x=rng.normal(size=(n, 1)),
            y=rng.normal(size=n),
            config={"data_label": "SYNTHETIC"},
        )

    frame, receipt = audit_mean_eprocess(
        {"base": _ShortBase(), "head": _Head()},
        {"s": shard},
        n_train=64,
        n_eval=48,
        seed=0,
        taus=list(np.linspace(0.05, 0.95, 9)),
    )
    assert frame.height > 0  # the audit ran to completion
    assert receipt["schema"] == "mean_eprocess.v1"


def _lattice_payload() -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": "receipt_lattice.v1",
        "schema": "receipt_lattice.v1",
        "data_label": "CORPUS",
        "research_only": True,
        "live_pnl_claim": False,
        "verdict": "consistent",
        "n_receipts": 1,
        "n_parse_errors": 0,
        "parse_errors": [],
        "file_digests": {"a.json": "b" * 64},
        "n_inputs_unspecified": 0,
        "inputs_unspecified": [],
        "n_dataset_unspecified": 0,
        "dataset_unspecified": [],
        "n_stale_code": 0,
        "stale_code": [],
        "n_claim_groups": 0,
        "n_consistent_groups": 0,
        "n_drift_groups": 0,
        "n_inconsistent_groups": 0,
        "n_known_inconsistent_groups": 0,
        "n_singleton_claims": 0,
        "n_retracted": 0,
        "retracted": {},
        "groups": [],
        "params": {"glob": "*.json", "n_known_inconsistent_pins": 0},
    }
    # inputs_sha256 binds digests+params — re-derived by the contract.
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    body["inputs_sha256"] = hash_bytes(
        canonical_json_bytes({"digests": body["file_digests"], "params": body["params"]})
    )
    return body


def test_lattice_counts_are_rederived_from_lists() -> None:
    """Counts that disagree with their lists must fail the contract."""
    from quant_fund.research.receipt_lattice import lattice_contract_errors

    assert lattice_contract_errors(_lattice_payload()) == []

    bad = _lattice_payload()
    bad["parse_errors"] = ["a.json:bad_json"]
    # claims 0 errors while carrying one
    assert "n_parse_errors" in lattice_contract_errors(bad)
    # and once the count is honest the verdict must flip with it
    bad["n_parse_errors"] = 1
    assert "verdict" in lattice_contract_errors(bad)

    bad2 = _lattice_payload()
    bad2["n_stale_code"] = 3  # list is empty — the count lies
    assert "n_stale_code" in lattice_contract_errors(bad2)

    bad3 = _lattice_payload()
    bad3["retracted"] = {"x.json": {"tombstone": "t.json", "scope": "all"}}
    assert "n_retracted" in lattice_contract_errors(bad3)


def test_admission_second_contradiction_in_inconsistent_group() -> None:
    """A new contradiction inside an already-inconsistent group must delta."""
    from quant_fund.research.admission import _inconsistent_groups

    fp = "fp" * 32
    before = {
        "groups": [
            {
                "fingerprint": fp,
                "verdict": "inconsistent",
                "disagreements": [{"path": "a", "values": [1, 2]}],
            }
        ]
    }
    after = {
        "groups": [
            {
                "fingerprint": fp,
                "verdict": "inconsistent",
                "disagreements": [
                    {"path": "a", "values": [1, 2]},
                    {"path": "b", "values": [3, 4]},  # the candidate's new hit
                ],
            }
        ]
    }
    delta = _inconsistent_groups(after) - _inconsistent_groups(before)
    assert delta, "second contradiction under a known fingerprint escaped the delta"
    # identical disagreement sets produce no delta
    assert not (_inconsistent_groups(before) - _inconsistent_groups(before))


def test_witness_scan_unreachable_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    """ok=False when the Rekor index can't be reached (blind ≠ clean)."""
    import quant_fund.research.witness_scan as mod

    root = tmp_path
    (root / "quality").mkdir()
    (root / "quality/witness_signing.pub").write_bytes(b"pub")

    def _dead(*args: Any, **kwargs: Any) -> bytes:
        raise OSError("offline")

    monkeypatch.setattr(mod, "_entry_uuids", _dead)
    res = mod.scan_witness_log(root)
    assert res["ok"] is False and res["online"] is False
    assert res["errors"][0].startswith("index_unreachable")


def test_fuzz_drill_failed_mutation_restores_clone(tmp_path: Path, monkeypatch) -> None:
    """A mutation that throws after partial writes must not poison the next grade."""
    import quant_fund.research.fuzz_drill as fd

    sentinel = "CONTAMINATED.txt"

    def _mutations(clone: Path, rng: Any) -> list[tuple[str, str, Any]]:
        def dirty_then_raise() -> str:
            (clone / sentinel).write_text("x")
            raise OSError("partial write")

        return [
            ("dirty_then_raise", "fail", dirty_then_raise),
            ("benign_uncovered", "ok", lambda: "noop"),
        ]

    seen_dirty: list[bool] = []

    def _verify(_clone: Path) -> dict[str, Any]:
        seen_dirty.append((_clone / sentinel).exists())
        return {"ok": True, "gates": {}}

    # NOTE: the baseline verify (before any mutation) also records — clean.

    monkeypatch.setattr(fd, "_mutations", _mutations)
    import quant_fund.research.repo_integrity as ri

    monkeypatch.setattr(ri, "verify_repo", _verify)
    res = fd.fuzz_drill(tmp_path, seed=0)
    benign = next(m for m in res["mutations"] if m["mutation"] == "benign_uncovered")
    assert benign["outcome"] == "correct"
    # every verify_repo call — baseline AND the benign grade — saw a clean
    # clone: the dirty mutation's writes were rolled back by the finally.
    assert len(seen_dirty) == 2 and not any(seen_dirty)


def test_anytime_p_one_sided_bound() -> None:
    """Post-peak dips are legal; exceeding 1/current_e never is."""
    from quant_fund.research.evalue_contracts import evalue_family_contract_errors

    base: dict[str, Any] = {
        "kind": "evalue_promotion.v1",
        "alpha": 0.05,
        "lam": 0.5,
        "n_origins": 64,
        "final_evalue": 5.0,
        "anytime_p": 0.04,  # peak was 25 — dipped to 5: 1/25 <= bound 1/5
        "promoted": True,
        "promotion_origin": 12,
    }
    assert evalue_family_contract_errors(base) == []
    base["anytime_p"] = 0.5  # > 1/5 — impossible under running-max semantics
    assert "anytime_p_exceeds_reciprocal_of_evalue" in evalue_family_contract_errors(base)


def test_honest_verdict_any_unavailable_lane_inconclusive() -> None:
    """Even a non-core unavailable lane must force inconclusive."""
    from quant_fund.research.evalue_contracts import evalue_family_contract_errors

    base: dict[str, Any] = {
        "kind": "honest_verdict.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "verdict": "supported_with_caveats",
        "winner": "gmm",
        "alpha": 0.05,
        "n_obs": 128,
        "n_heads": 2,
        "inputs_sha256": "a" * 64,
        "components": {"magnitude": {}},
        "unavailable_lanes": ["magnitude"],  # non-core lane down
    }
    assert "unavailable_lanes_but_verdict_not_inconclusive" in evalue_family_contract_errors(base)
    base["verdict"] = "inconclusive"
    assert evalue_family_contract_errors(base) == []


def test_serial_watch_alarm_partition() -> None:
    """alarmed_lags must equal exactly the per-lag rows that alarmed."""
    from quant_fund.research.evalue_contracts import evalue_family_contract_errors

    base: dict[str, Any] = {
        "kind": "serial_watch",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "alpha": 0.05,
        "lam": 0.5,
        "n_lags": 3,
        "n_origins": 64,
        "per_lag": {str(k): {"pos": 1.1, "neg": 0.9, "alarmed": k == 1} for k in range(1, 4)},
        "alarmed_lags": [1],
        "alarm_origins": {},
        "any_lag_alarmed": True,
        "pooled_alarmed": False,
        "pooled_evalue": 2.0,
        "evidence": ["anytime_valid"],
    }
    assert evalue_family_contract_errors(base) == []
    base["alarmed_lags"] = [1, 2]  # per_lag says only lag 1 alarmed
    assert "alarmed_lags_not_per_lag_partition" in evalue_family_contract_errors(base)


def test_tombstone_target_must_stay_in_corpus(tmp_path: Path) -> None:
    """A tombstone naming a path outside the corpus is invalid, not evaluated."""
    from quant_fund.research.receipt_tombstone import load_tombstones, tombstone_body
    from quant_fund.research.receipt_v2 import seal_receipt

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    import hashlib

    body = tombstone_body(
        target_name="../outside.txt",
        target_sha256=hashlib.sha256(b"x").hexdigest(),
        reason="probe",
    )
    sealed = seal_receipt(body)
    (corpus / f"tombstone_{body['target_sha256'][:16]}.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True)
    )
    res = load_tombstones(corpus)
    assert not res["active"]
    assert any("tombstone_target_uncontained" in e for e in res["invalid"])


def test_timestamp_anchor_target_must_stay_in_root(tmp_path: Path) -> None:
    """A manifest anchor pointing outside the tree is a hard error."""
    import hashlib

    from quant_fund.research.timestamp_anchor import (
        SHA256_ALG_OID,
        _seq,
        _tlv,
        verify_timestamps,
    )

    root = tmp_path / "root"
    ts_dir = root / "quality" / "timestamps"
    ts_dir.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret bytes")
    digest = hashlib.sha256(b"secret bytes").hexdigest()

    # Minimal DER TSR whose TSTInfo messageImprint commits to `digest` —
    # enough for extract_imprint to bind; chain check stays 'unchecked'.
    alg = _seq(SHA256_ALG_OID + _tlv(0x05, b""))
    mi = _seq(alg + _tlv(0x04, bytes.fromhex(digest)))
    tst = _seq(_tlv(0x02, b"\x01") + _tlv(0x06, b"\x2a") + mi)
    eci = _seq(_tlv(0x06, b"\x01\x02") + _tlv(0xA0, _tlv(0x04, tst)))
    sd = _seq(_tlv(0x02, b"\x01") + _tlv(0x31, _seq(b"")) + eci)
    tsr = _seq(_seq(_tlv(0x05, b"")) + _seq(_tlv(0x06, b"\x01\x02") + _tlv(0xA0, sd)))
    (ts_dir / "evil.tsr").write_bytes(tsr)
    (ts_dir / "anchors.json").write_text(
        json.dumps(
            {
                "schema": "timestamp_anchors.v1",
                "anchors": {"evil.tsr": {"target": "../../outside.bin", "sha256": digest}},
            }
        )
    )
    res = verify_timestamps(root)
    assert res["ok"] is False
    assert any(e.startswith("anchor_target_uncontained") for e in res["errors"])


def test_monitor_alarm_rows_count_calibration(tmp_path: Path, monkeypatch) -> None:
    """calibration_alarmed rows must count toward n_alarm_rows."""
    import numpy as np
    import polars as pl

    import quant_fund.research.monitor_run as mr

    class _Under:
        """Always predicts far below y → PITs collapse → calib alarms."""

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            return None

        def predict(self, x: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            taus = np.linspace(0.05, 0.95, 9)
            return np.tile(-50.0 - np.asarray(taus), (n, 1))

    class _Other:
        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            return None

        def predict(self, x: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            taus = np.linspace(0.05, 0.95, 9)
            return np.tile(np.asarray(taus) - 0.5, (n, 1))

    # Drive via the public runner with only the calibration lane live.
    monkeypatch.setattr(
        mr,
        "_lazy",
        lambda name: (
            __import__("quant_fund.research.calibration_eprocess", fromlist=["x"])
            if name == "calibration_eprocess"
            else None
        ),
    )

    from quant_fund.research.fleet_eval import SyntheticShard

    def shard(n: int, seed: int) -> SyntheticShard:
        rng = np.random.default_rng(seed)
        return SyntheticShard(
            name="s",
            x=rng.normal(size=(n, 1)),
            y=rng.normal(size=n),
            config={"data_label": "SYNTHETIC"},
        )

    frame, receipt = mr.monitor_fleet(
        {"under": _Under(), "other": _Other()},
        {"s": shard},
        n_train=64,
        n_eval=64,
        seed=0,
        taus=list(np.linspace(0.05, 0.95, 9)),
    )
    calib_rows = frame.filter(pl.col("calibration_alarmed") == True).height  # noqa: E712
    if calib_rows:
        assert receipt["n_alarm_rows"] >= calib_rows
