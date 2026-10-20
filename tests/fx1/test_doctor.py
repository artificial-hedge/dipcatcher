"""fx1 doctor: readiness signals, presence-flag hygiene, graceful missing dirs."""

import json
from pathlib import Path

from typer.testing import CliRunner

from fx1.cli import app
from fx1.doctor import collect_status


def test_doctor_reports_version_and_base_model(tmp_path: Path) -> None:
    status = collect_status(tmp_path)
    assert status["fx1_version"]
    assert status["base_model"] == "moonshotai/Kimi-K3"
    assert status["harness_package"] == "ok"
    extras = status["optional_extras"]
    assert isinstance(extras, dict)
    assert set(extras) >= {"torch", "safetensors"}
    assert extras["torch"] in {"present", "missing"}
    assert extras["safetensors"] in {"present", "missing"}
    assert "distribution_lineage" in status


def test_doctor_missing_dirs_reported_not_raised(tmp_path: Path) -> None:
    status = collect_status(tmp_path)
    assert status["receipts"] == "missing"
    assert status["data_fx1_corpus"] == "missing"


def test_doctor_counts_corpus_lines(tmp_path: Path) -> None:
    corpus = tmp_path / "data" / "fx1" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (tmp_path / "receipts").mkdir()
    (tmp_path / "receipts" / "r.json").write_text("{}", encoding="utf-8")
    status = collect_status(tmp_path)
    assert status["data_fx1_corpus"] == 2
    assert status["receipts"] == 1


def test_doctor_never_echoes_secret_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-super-secret-value")
    monkeypatch.setenv("FX1_SIGNING_KEY", "sig-super-secret-value")
    status = collect_status(tmp_path)
    assert status["moonshot_key"] == "set"
    assert status["signing_key"] == "set"
    blob = json.dumps(status)
    assert "sk-super-secret-value" not in blob
    assert "sig-super-secret-value" not in blob


def test_doctor_cli_json(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["doctor", "--root", str(tmp_path)])
    assert result.exit_code == 0
    status = json.loads(result.stdout)
    assert status["fx1_version"]


def test_doctor_reports_ledger_chain_state(tmp_path: Path) -> None:
    assert collect_status(tmp_path)["corpus_ledger_chain"] == "missing"
    (tmp_path / "data" / "fx1").mkdir(parents=True)
    (tmp_path / "data" / "fx1" / "corpus_ledger.jsonl").write_text(
        '{"event": "corrupt_entry"}\n', encoding="utf-8"
    )
    assert collect_status(tmp_path)["corpus_ledger_chain"] in {"unverifiable", "BROKEN"}


def test_doctor_distribution_lineage_matches_crown_jewels_library(tmp_path: Path) -> None:
    from quant_fund.research.crown_jewels import (
        DEFAULT_JEWELS,
        distribution_lineage_digest,
        write_crown_jewels_pin,
    )

    for rel in DEFAULT_JEWELS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n", encoding="utf-8")
    (tmp_path / "quality" / "epoch_heads.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "quality" / "repo_integrity.json").write_text("{}\n", encoding="utf-8")
    write_crown_jewels_pin(tmp_path, tmp_path / "quality" / "crown_jewels.json")
    lineage = distribution_lineage_digest(tmp_path)
    status = collect_status(tmp_path)
    assert status["distribution_lineage"] == lineage["digest"]
    assert status["distribution_jewels_intact"] is True

    (tmp_path / "pyproject.toml").write_text("# mutated\n", encoding="utf-8")
    broken = distribution_lineage_digest(tmp_path)
    assert broken["jewels_intact"] is False
    status_broken = collect_status(tmp_path)
    assert status_broken["distribution_jewels_intact"] is False
    assert status_broken["distribution_lineage"] == broken["digest"]
    assert status_broken["distribution_lineage"] != lineage["digest"]
