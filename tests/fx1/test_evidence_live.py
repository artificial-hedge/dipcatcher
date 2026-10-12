"""Live evidence: real lab receipts, tamper, discovery, staleness, grounding.

Pipeline under test: ``EvidenceLedger.verified_receipts()`` (current bytes
only) supplies ``ground_summary(verified_receipts=...)``. ``check_citations``
reads the same ledger and treats a mutated file as unverified. A forbidden
headline still raises ``EvidenceError`` / withholds the composed answer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive.evidence import EvidenceError, ground_summary
from fx1.interactive.evidence_ledger import (
    EvidenceLedger,
    Verdict,
    check_citations,
    compose_answer,
    extract_receipt_paths,
    verify_receipt,
)
from quant_fund.schemas.receipt_v2_leaf import seal_receipt

QUANT = "/workspace/.venv/bin/quant"
COMMITTED = Path("/workspace/receipts/kill_audit.json")


@pytest.fixture(autouse=True)
def _dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))


def _quant(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/src"
    env["FX1_CONFIG_DIR"] = str(cwd / "fx1-config")
    env["FX1_SESSION_DIR"] = str(cwd / "sessions")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        [QUANT, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )


def _ts(index: int) -> str:
    return f"2020-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00+00:00"


def _write_vol(path: Path, forecast: float, *, synthetic: bool) -> None:
    rows = [f"{_ts(i)},1.0,{forecast}" for i in range(30)]
    header = "timestamp,realized,forecast_var\n"
    preamble = "# DATA_LABEL=SYNTHETIC\n" if synthetic else ""
    path.write_text(preamble + header + "\n".join(rows) + "\n", encoding="utf-8")


def _write_probability(path: Path) -> None:
    rows = [f"{_ts(i)},{0 if i % 3 else 1},{0.3 + (i % 4) * 0.15}" for i in range(30)]
    path.write_text("timestamp,realized,forecast_prob\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _write_quantile(path: Path) -> None:
    rows = [f"{_ts(i)},{(i % 20) / 20},0.10,0.50,0.90" for i in range(30)]
    path.write_text("timestamp,realized,q05,q50,q95\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _receipt_path(stdout: str) -> Path:
    lines = [line for line in stdout.splitlines() if line.startswith("receipt=")]
    assert lines, stdout
    return Path(lines[-1].split("=", 1)[1])


def _score(task: str, data: Path, out: Path) -> Path:
    proc = _quant(
        ["score-forecasts", "--task", task, "--data", str(data), "--out-dir", str(out)],
        out,
    )
    assert proc.returncode == 0, proc.stderr
    return _receipt_path(proc.stdout)


def _qlike(path: Path, root: Path) -> float:
    item = verify_receipt(path, roots=[root])
    assert item.verdict is Verdict.VERIFIED, item.detail
    return dict(item.scores)["qlike"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("live-evidence")
    data = root / "data"
    out = root / "out"
    data.mkdir()
    out.mkdir()
    _write_vol(data / "a.csv", 2.0, synthetic=False)
    _write_vol(data / "b.csv", 8.0, synthetic=False)
    _write_vol(data / "syn.csv", 2.0, synthetic=True)
    _write_probability(data / "p.csv")
    _write_quantile(data / "q.csv")
    scored = {
        "a": _score("volatility", data / "a.csv", out),
        "b": _score("volatility", data / "b.csv", out),
        "synthetic": _score("volatility", data / "syn.csv", out),
        "probability": _score("probability", data / "p.csv", out),
        "quantile": _score("quantile", data / "q.csv", out),
    }
    proc = _quant(
        [
            "vol-bench",
            "--config",
            "/workspace/configs/research.yaml",
            "--models",
            "rv_roll",
            "--shards",
            "garch_vol",
            "--horizons",
            "1",
            "--min-history",
            "30",
            "--n-origins",
            "10",
            "--n-bars",
            "50",
            "--out-dir",
            str(out),
            "--receipt-version",
            "2",
        ],
        out,
    )
    assert proc.returncode == 0, proc.stderr
    scored["vol"] = _receipt_path(proc.stdout)
    shutil.copy(COMMITTED, root / "kill_audit.json")
    return {"root": root, "out": out, **scored}


def _plant(tmp_path: Path, data: bytes, name: str = "r.json") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _refuse(path: Path, roots: list[Path]) -> Any:
    item = verify_receipt(path, roots=roots)
    assert item.verdict is not Verdict.VERIFIED
    assert item.detail
    return item


def _reseal(payload: dict[str, Any], tmp_path: Path, name: str) -> Path:
    return _plant(tmp_path, json.dumps(seal_receipt(payload)).encode(), name)


def ordinary_answers() -> list[str]:
    tickers = ["AAPL", "MSFT", "EURUSD", "GBPJPY", "SPX"]
    answers: list[str] = []
    for year in range(2000, 2020):
        for ticker in tickers:
            count = (year + len(ticker)) % 17 + 1
            answers.append(
                f"In {year} {ticker} accounted for {count}% across {count + 2} observations, "
                f"build v2.{year % 10}.{count}, dated {year}-03-{(count % 27) + 1:02d}, "
                f"id s-20261012T024500Z-{year % 256:02x}{count:06x}."
            )
    answers.append("navigation of the 2020 canvas stayed partial.")
    answers.append("an unproven estimate for 2019 used 4 names.")
    answers.append("The result is unverified and partial. Count was 12 on 2024-01-02.")
    return answers


def _pipeline(summary: str, ledger: EvidenceLedger) -> str:
    refs = {item.ref for item in ledger.verified_receipts()}
    return ground_summary(summary, verified_receipts=refs)


def _caught(text: str, ledger: EvidenceLedger) -> bool:
    problems = check_citations(text, ledger)
    try:
        grounded = _pipeline(text, ledger)
    except EvidenceError:
        return True
    return bool(problems) or grounded != text


def fabricated_claims(score_a: float, score_b: float, name_a: str) -> list[str]:
    wrong = score_a + 3.5
    claims = [f"qlike={wrong + index * 0.17}" for index in range(12)]
    claims.extend(
        f"verified receipt missing-{index}.json records qlike={score_a}" for index in range(10)
    )
    claims.extend(f"sealed by {index:064x}" for index in range(10))
    claims.extend(
        [
            "independently verified",
            "this result is proven",
            "confirmed by receipt",
            "verification passed",
            "the lane is verified",
            "v\u200berified receipt of the run",
            "independently\u200b verified",
            "v e r i f i e d receipt",
            "p r o v e n by the run",
            "v\u0435rified receipt",
        ]
    )
    claims.extend(f"qlike={score_b} is the score in {name_a} (copy {index})" for index in range(10))
    claims.extend([f"pinball=9.{index} on {name_a}" for index in range(8)])
    return claims


def test_lab_receipts_verify_and_synthetic_is_not_research(corpus: dict[str, Any]) -> None:
    root = corpus["out"]
    ledger = EvidenceLedger()
    for key in ("a", "b", "probability", "quantile", "synthetic", "vol"):
        item = ledger.add_receipt(corpus[key], roots=[root])
        assert item.verdict is Verdict.VERIFIED, (key, item.detail)
        assert "VERIFIED_RESEARCH" not in item.data_label
    synthetic = next(item for item in ledger.items() if item.ref == Path(corpus["synthetic"]).name)
    vol = next(item for item in ledger.items() if item.ref == Path(corpus["vol"]).name)
    assert synthetic.data_label == "SYNTHETIC"
    assert vol.data_label == "SYNTHETIC"
    block = ledger.render_block(width=200)
    assert "VERIFIED_RESEARCH" not in block
    assert "SYNTHETIC correctness evidence only — not market evidence" in block
    copied = verify_receipt(corpus["root"] / "kill_audit.json", roots=[corpus["root"]])
    assert copied.verdict is Verdict.VERIFIED
    assert copied.data_label == "SYNTHETIC"


def test_tamper_fails_closed(corpus: dict[str, Any], tmp_path: Path) -> None:
    raw = Path(corpus["a"]).read_bytes()
    body = json.loads(raw)
    root = [tmp_path]
    cases: dict[str, bytes] = {
        "flip": raw[:80] + bytes([raw[80] ^ 0x01]) + raw[81:],
        "trunc": raw[:40],
        "nonutf": b"\xff" + raw[:30],
        "bom": b"\xef\xbb\xbf" + raw,
        "nan": b'{"a": NaN}',
        "inf": b'{"a": Infinity}',
        "dup": raw.replace(b"{", b'{"schema":"injected",', 1),
    }
    text = raw.decode("utf-8")
    cases["score"] = text.replace(str(body["payload"]["scores"]["qlike"]), "9.5", 1).encode()
    cases["dataset"] = text.replace(body["dataset_hash"], "ab" * 32, 1).encode()
    cases["kind"] = text.replace(body["kind"], "forged.kind", 1).encode()
    cases["live"] = text.replace('"live_pnl_claim": false', '"live_pnl_claim": true', 1).encode()
    cases["research"] = text.replace('"research_only": true', '"research_only": false', 1).encode()
    for name, data in cases.items():
        item = _refuse(_plant(tmp_path, data, f"{name}.json"), root)
        assert item.verdict in {Verdict.INVALID, Verdict.UNVERIFIABLE}, name
    huge = tmp_path / "huge.json"
    with huge.open("wb") as handle:
        handle.truncate(33 * 1024 * 1024)
    huge_item = _refuse(huge, root)
    assert "size" in huge_item.detail


def test_resealed_fingerprint_and_honesty_flags_fail(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    body = json.loads(Path(corpus["a"]).read_bytes())
    env = json.loads(json.dumps(body))
    env["environment"]["python"] = "0.0.1-tampered"
    env_item = _refuse(_reseal(env, tmp_path, "env.json"), [tmp_path])
    assert "fingerprint" in env_item.detail
    flags = json.loads(json.dumps(body))
    flags["payload"]["research_only"] = False
    research = _refuse(_reseal(flags, tmp_path, "research.json"), [tmp_path])
    assert "research_only" in research.detail
    live = json.loads(json.dumps(body))
    live["payload"]["live_pnl_claim"] = True
    live_item = _refuse(_reseal(live, tmp_path, "live.json"), [tmp_path])
    assert "live_pnl_claim" in live_item.detail


def test_discovery_hygiene(corpus: dict[str, Any], tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    good = Path(corpus["a"]).read_bytes()
    nested = root / "sub"
    nested.mkdir()
    (nested / "a.json").write_bytes(good)
    (root / "résumé.json").write_bytes(good)
    outside = tmp_path / "outside.json"
    outside.write_bytes(good)
    link_out = root / "escape.json"
    link_out.symlink_to(outside)
    passwd = root / "passwd.json"
    passwd.symlink_to("/etc/passwd")
    started = time.perf_counter()
    junk = ["noise"] * 100_000
    junk.append('receipt="sub/a.json"')
    junk.append("receipt='résumé.json'")
    junk.extend(f"receipt=junk-{index}.json" for index in range(1000))
    junk.append("receipt=sub/a.json")
    junk.append("receipt=../outside.json")
    stdout = "\r\n".join(junk) + "\r\n"
    found = extract_receipt_paths(stdout)
    assert time.perf_counter() - started < 2
    assert found[0] == "sub/a.json"
    assert "résumé.json" in found
    assert len(found) == 20
    assert "../outside.json" not in found
    assert found.count("sub/a.json") == 1
    relative = verify_receipt("sub/a.json", roots=[root])
    assert relative.verdict is Verdict.VERIFIED
    unicode = verify_receipt("résumé.json", roots=[root])
    assert unicode.verdict is Verdict.VERIFIED
    escaped = _refuse(link_out, [root])
    assert "outside" not in escaped.ref
    leaked = _refuse(passwd, [root])
    assert "root:" not in leaked.detail + leaked.ref
    missing = verify_receipt(root / "absent.json", roots=[root])
    assert missing.verdict is Verdict.MISSING
    absolute = _refuse(Path("/etc/passwd"), [root])
    assert "root:" not in absolute.detail
    dotdot = _refuse(root / ".." / "outside.json", [root])
    assert dotdot.verdict is not Verdict.VERIFIED
    directory = _refuse(root, [root])
    assert directory.detail == "not a regular file"


def test_fifo_and_device_do_not_hang(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    holder: dict[str, Any] = {}

    def read_fifo() -> None:
        holder["fifo"] = verify_receipt(fifo, roots=[tmp_path])

    def read_null() -> None:
        holder["null"] = verify_receipt("/dev/null", roots=[Path("/")])

    threads = [
        threading.Thread(target=read_fifo, daemon=True),
        threading.Thread(target=read_null, daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert holder["fifo"].verdict is not Verdict.VERIFIED
    assert holder["null"].detail == "not a regular file"


def test_stale_receipt_drops_out_of_the_verified_set(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "a.json"
    shutil.copy(corpus["a"], path)
    ledger = EvidenceLedger()
    item = ledger.add_receipt(path, roots=[tmp_path])
    assert item.verdict is Verdict.VERIFIED
    assert ledger.is_current(item) is True
    name = path.name
    kept = _pipeline(f"verified receipt {name} records qlike={dict(item.scores)['qlike']}", ledger)
    assert kept.startswith(f"verified receipt {name}")
    path.write_bytes(path.read_bytes() + b"\n")
    assert ledger.is_current(item) is False
    assert ledger.verified_receipts() == []
    assert ledger.has_verified() is False
    rewritten = _pipeline(f"verified receipt {name} records the run", ledger)
    assert "verified receipt" not in rewritten.lower()
    composed = compose_answer(f"the result is verified via {name}", ledger)
    assert composed.status != "verified"
    assert any(problem.code == "verified_language_without_receipt" for problem in composed.problems)


def test_ordinary_prose_is_not_blocked(corpus: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    shutil.copy(corpus["a"], path)
    ledger = EvidenceLedger()
    ledger.add_receipt(path, roots=[tmp_path])
    answers = ordinary_answers()
    assert len(answers) >= 100
    for text in answers:
        assert _pipeline(text, ledger) == text
        assert check_citations(text, ledger) == []
        assert compose_answer(text, ledger).status != "withheld"


def test_fabricated_claims_are_caught(corpus: dict[str, Any], tmp_path: Path) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    shutil.copy(corpus["a"], path_a)
    shutil.copy(corpus["b"], path_b)
    score_a = _qlike(path_a, tmp_path)
    score_b = _qlike(path_b, tmp_path)
    assert abs(score_a - score_b) > 0.05
    ledger = EvidenceLedger()
    ledger.add_receipt(path_a, roots=[tmp_path])
    ledger.add_receipt(path_b, roots=[tmp_path])
    honest = f"verified receipt a.json records qlike={score_a}"
    assert check_citations(honest, ledger) == []
    assert _pipeline(honest, ledger) == honest
    claims = fabricated_claims(score_a, score_b, "a.json")
    assert len(claims) >= 60
    missed = [text for text in claims if not _caught(text, ledger)]
    assert missed == []


def test_forbidden_headline_still_withholds(corpus: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    shutil.copy(corpus["a"], path)
    ledger = EvidenceLedger()
    item = ledger.add_receipt(path, roots=[tmp_path])
    summary = f"verified receipt a.json qlike={dict(item.scores)['qlike']} and Sharpe 1.4"
    with pytest.raises(EvidenceError):
        _pipeline(summary, ledger)
    composed = compose_answer(summary, ledger)
    assert composed.status == "withheld"
    assert "1.4" not in composed.text
    assert "verified receipt a.json" not in composed.text
