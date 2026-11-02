"""Fail-closed error handling on the fx-1 reporting and monitoring surfaces.

The fx-1 package has no ``reporting/`` or ``monitoring/`` package; the
closest equivalents are the readiness doctor, the harness runner, and the
usage report. These tests pin their failure classes:

- ``fx1.doctor.collect_status`` — an unreadable or corrupt data file must be
  *reported* (``unreadable``), not crash the doctor before it can say
  anything else, and not be mistaken for a missing file.
- ``fx1.harness.Harness.run`` — a command that outlives its registered
  timeout budget must surface as a typed result, not leak a raw
  ``subprocess.TimeoutExpired`` through the harness contract.
- ``fx1.serve.usage_report.aggregate_usage`` — usage counters must only sum
  claims the backend actually reported and that are real integers.

Every fixture is a ``tmp_path`` file or an injected runner. No credential
path, receipt store, or real data directory is read or written.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from fx1.doctor import collect_status
from fx1.harness import Harness, HarnessRole
from fx1.serve.usage_report import aggregate_usage

# --------------------------------------------------------------------------- #
# fx1.doctor — unreadable files are reported, not raised
# --------------------------------------------------------------------------- #


def _fx1_dir(root: Path) -> Path:
    path = root / "data" / "fx1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_doctor_reports_a_corrupt_corpus_as_unreadable(tmp_path: Path) -> None:
    """A present-but-undecodable file is a different defect from an absent one."""
    (_fx1_dir(tmp_path) / "corpus.jsonl").write_bytes(b"\xff\xfe\x00binary")
    status = collect_status(tmp_path)
    assert status["data_fx1_corpus"] == "unreadable"
    assert status["data_fx1_eval"] == "missing"


def test_doctor_does_not_crash_on_a_corrupt_corpus(tmp_path: Path) -> None:
    """The doctor reports; it must not hard-fail on one bad file."""
    (_fx1_dir(tmp_path) / "corpus.jsonl").write_bytes(b"\xff\xfe\x00binary")
    status = collect_status(tmp_path)
    # The rest of the board is still readable.
    assert status["fx1_version"]
    assert status["harness_package"] == "ok"
    assert status["eval_bank_tasks"] != "unavailable"


def test_doctor_reports_a_corrupt_manifest_as_unreadable(tmp_path: Path) -> None:
    (_fx1_dir(tmp_path) / "manifest.json").write_bytes(b"\xff\xfe\x00binary")
    assert collect_status(tmp_path)["data_fx1_manifest"] == "unreadable"


def test_doctor_reports_missing_files_as_missing(tmp_path: Path) -> None:
    status = collect_status(tmp_path)
    assert status["data_fx1_corpus"] == "missing"
    assert status["data_fx1_manifest"] == "missing"
    assert status["receipts"] == "missing"


def test_doctor_reports_an_unreadable_receipts_dir_as_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipts dir that cannot be walked is unreadable, not missing."""
    receipts = tmp_path / "receipts"
    receipts.mkdir()

    def _boom(*_args: Any, **_kwargs: Any):
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "rglob", _boom)
    assert collect_status(tmp_path)["receipts"] == "unreadable"


def test_doctor_still_counts_readable_files(tmp_path: Path) -> None:
    corpus = _fx1_dir(tmp_path) / "corpus.jsonl"
    corpus.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert collect_status(tmp_path)["data_fx1_corpus"] == 2


def test_doctor_reports_a_corrupt_ledger_as_unverifiable(tmp_path: Path) -> None:
    (_fx1_dir(tmp_path) / "corpus_ledger.jsonl").write_text(
        '{"event": "corrupt_entry"}\n', encoding="utf-8"
    )
    assert collect_status(tmp_path)["corpus_ledger_chain"] in {"unverifiable", "BROKEN"}


def test_doctor_never_echoes_secret_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-secret")
    monkeypatch.setenv("FX1_SIGNING_KEY", "sig-test-secret")
    status = collect_status(tmp_path)
    assert status["moonshot_key"] == "set"
    assert status["signing_key"] == "set"
    blob = repr(status)
    assert "sk-test-secret" not in blob
    assert "sig-test-secret" not in blob


# --------------------------------------------------------------------------- #
# fx1.harness — a timeout is a result, not a crash
# --------------------------------------------------------------------------- #


def _ok_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, f"ran: {' '.join(argv)}", ""


def _timeout_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    raise subprocess.TimeoutExpired(argv, timeout_s)


def test_harness_timeout_becomes_a_typed_result() -> None:
    result = Harness(runner=_timeout_runner).run("verify-research")
    assert result.exit_code == 124
    assert result.ok is False
    assert result.command == "verify-research"
    assert result.stdout == ""


def test_harness_timeout_names_the_command_and_budget() -> None:
    result = Harness(runner=_timeout_runner).run("train")
    assert "timed out" in result.stderr
    assert "train" in result.stderr


def test_harness_timeout_carries_the_registered_budget() -> None:
    result = Harness(runner=_timeout_runner).run("train")
    budget = Harness().get("train").timeout_s
    assert str(budget) in result.stderr


def test_harness_success_is_unaffected() -> None:
    result = Harness(runner=_ok_runner).run("verify-research")
    assert result.ok is True
    assert result.exit_code == 0


def test_harness_unknown_command_still_fails_closed() -> None:
    with pytest.raises(KeyError, match="not a registered harness command"):
        Harness(runner=_ok_runner).run("not-a-command")


def test_harness_config_containment_still_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowlist"):
        Harness(runner=_ok_runner).run("research", config=tmp_path / "evil.yaml")


def test_harness_runner_receives_the_registered_timeout() -> None:
    seen: list[int] = []

    def _recording(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        seen.append(timeout_s)
        return 0, "", ""

    Harness(runner=_recording).run("train")
    assert seen == [Harness().get("train").timeout_s]


def test_harness_registry_covers_the_verification_and_reporting_surfaces() -> None:
    names = {c.name for c in Harness().list_commands()}
    assert {"monitor", "report", "tearsheet", "doctor", "verify-research"} <= names


def test_harness_monitoring_command_is_a_verification_surface() -> None:
    command = Harness().get("monitor")
    assert command.role is HarnessRole.VERIFICATION


# --------------------------------------------------------------------------- #
# fx1.serve.usage_report — only reported, integer usage is summed
# --------------------------------------------------------------------------- #


class _Record:
    """Minimal stand-in for the SDK's ``CompletionRecord``."""

    def __init__(
        self,
        *,
        backend: str = "local",
        model: str | None = "m",
        ok: bool = True,
        latency_ms: float = 10.0,
        at: float = 100.0,
        usage: dict[str, Any] | None = None,
        key_id: str | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.ok = ok
        self.latency_ms = latency_ms
        self.at = at
        self.usage = usage
        self.key_id = key_id


def test_usage_over_an_empty_log_is_zeroed_not_missing() -> None:
    report = aggregate_usage([], cap=10, dropped=0)
    assert report.records_seen == 0
    assert report.totals.requests == 0
    assert report.totals.usage_reported == 0
    assert report.totals.mean_latency_ms is None


def test_usage_counts_only_reported_usage() -> None:
    records = [
        _Record(usage={"prompt_tokens": 10, "completion_tokens": 5}),
        _Record(usage=None),
    ]
    report = aggregate_usage(records, cap=10, dropped=0)
    assert report.totals.requests == 2
    assert report.totals.usage_reported == 1
    assert report.totals.prompt_tokens == 10
    assert report.totals.completion_tokens == 5


def test_usage_skips_non_integer_claims() -> None:
    """A float, string, or bool token count is never summed."""
    records = [
        _Record(usage={"prompt_tokens": 10.5}),
        _Record(usage={"prompt_tokens": "12"}),
        _Record(usage={"prompt_tokens": True}),
    ]
    report = aggregate_usage(records, cap=10, dropped=0)
    assert report.totals.usage_reported == 3
    assert report.totals.prompt_tokens == 0


def test_usage_keeps_provider_specific_counters_verbatim() -> None:
    records = [_Record(usage={"cached_tokens": 7})]
    report = aggregate_usage(records, cap=10, dropped=0)
    assert report.totals.other_usage == {"cached_tokens": 7}


def test_usage_reports_ring_evictions_explicitly() -> None:
    report = aggregate_usage([_Record()], cap=10, dropped=3)
    assert report.records_dropped == 3
    assert report.ring_cap == 10


def test_usage_is_never_a_live_pnl_claim() -> None:
    report = aggregate_usage([_Record()], cap=10, dropped=0)
    assert report.totals.errors == 0
    assert report.totals.ok == 1
