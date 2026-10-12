"""Live and adversarial checks for the selective family surface.

These tests drive the real selector, the real bench callables' module
attributes, real panel frames, and the research verifier. Injected runners
are used only to prove a refusal happens before any bench.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_fund.research.catalog import registry
from quant_fund.research.family_surface import (
    RUNNERS,
    FamilySurfaceError,
    PanelBundle,
    family_contract,
    parse_only,
    run_families,
    select_families,
)
from quant_fund.research.verify import verify_research_artifact
from tests.unit.research.test_catalog_retired_families import RETIRED_SAMPLE, _write_receipt

_ROOT = Path(__file__).resolve().parents[3]
_EXCLUSION_CODES = frozenset({"over_cap", "retired", "unknown", "no_runner", "not_qualified"})


def _config(source: str, root: Path | None = None) -> SimpleNamespace:
    data = SimpleNamespace(source=source, benchmark_id="SEC_MKT")
    if root is not None:
        data.root = root
    return SimpleNamespace(
        data=data,
        train=SimpleNamespace(random_seed=1, ranking_target="future_return_1"),
    )


def _frame(source: str):
    import polars as pl

    return pl.DataFrame({"source": [source], "future_return_1": [0.01]})


def _patch_benches(monkeypatch: pytest.MonkeyPatch, sink: list[str]) -> None:
    """Replace every selective runner with a callable that records and raises."""

    def _boom(*_args: object, _family: str, **_kwargs: object) -> None:
        sink.append(_family)
        raise AssertionError(f"{_family} bench was called")

    for spec in RUNNERS.values():
        module_name, _, attr = spec.target.partition(":")
        module = importlib.import_module(module_name)
        monkeypatch.setattr(
            module,
            attr,
            lambda *args, _family=spec.family_id, **kwargs: _boom(*args, _family=_family, **kwargs),
        )


def test_every_documented_trigger_selects_its_family() -> None:
    for spec in RUNNERS.values():
        for term in spec.trigger_terms:
            payload = select_families(term, max_families=8)
            ids = [row["id"] for row in payload["selected"]]
            assert spec.family_id in ids, term


def test_cv_plus_trigger_is_not_dropped() -> None:
    payload = select_families("cv+", max_families=4)
    assert [row["id"] for row in payload["selected"]] == ["cv_plus"]


def test_select_is_deterministic_for_adversarial_goals() -> None:
    goals = [
        "volatility; rm -rf /",
        "$(touch /tmp/not-created)",
        "波动率 volatility",
        "VOLATILITY forecast",
        "top-k conformal",
        "a" * 4000,
    ]
    for goal in goals:
        first = select_families(goal, max_families=4)
        second = select_families(goal, max_families=4)
        assert first == second
        assert all(row["executable"] is True for row in first["selected"])


def test_empty_nul_and_huge_goals_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FamilySurfaceError, match="non-empty"):
        select_families("   ")
    with pytest.raises(FamilySurfaceError, match="NUL"):
        select_families("volatility\x00ranking")
    with pytest.raises(FamilySurfaceError, match="4096"):
        select_families("volatility " * 5000)
    marker = tmp_path / "pwned"
    payload = select_families(f"volatility; touch {marker}", max_families=2)
    assert [row["id"] for row in payload["selected"]] == ["volatility"]
    assert not marker.exists()


def test_max_families_rejects_non_integers_and_zero_is_a_cap() -> None:
    with pytest.raises(FamilySurfaceError, match="integer"):
        select_families("volatility", max_families=1.5)  # type: ignore[arg-type]
    with pytest.raises(FamilySurfaceError, match="integer"):
        select_families("volatility", max_families=True)  # type: ignore[arg-type]
    payload = select_families("volatility ranking alpha", max_families=0)
    assert payload["selected"] == []
    assert payload["exclusions"]
    assert {row["code"] for row in payload["exclusions"]} == {"over_cap"}


def test_exclusion_codes_match_the_real_contract() -> None:
    retired = next(iter(registry.RETIRED_BENCHMARK_FAMILIES))
    live = sorted(registry.LIVE_OPTIONAL_BENCHMARK_FAMILIES - set(RUNNERS))[0]
    payload = select_families(
        f"volatility conformal jackknife ranking alpha drawdown {retired} {live}",
        max_families=1,
    )
    selected = {row["id"] for row in payload["selected"]}
    assert selected.isdisjoint({row["id"] for row in payload["exclusions"]})
    for row in payload["exclusions"]:
        assert row["code"] in _EXCLUSION_CODES
        assert row["reason"]
        if row["code"] == "over_cap":
            assert family_contract(row["id"])["executable"] is True
            assert "max_families=1" in row["reason"]
        elif row["code"] == "retired":
            assert family_contract(row["id"])["qualification"] == "retired"
            assert row["id"] == retired
        elif row["code"] == "no_runner":
            assert family_contract(row["id"])["qualification"] == "live_optional"
            assert row["id"] == live


def test_unsafe_ids_fail_closed_before_any_bench(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: list[str] = []
    _patch_benches(monkeypatch, sink)
    samples = [
        "not_a_real_family_zzz",
        next(iter(registry.RETIRED_BENCHMARK_FAMILIES)),
        sorted(registry.LIVE_OPTIONAL_BENCHMARK_FAMILIES - set(RUNNERS))[0],
        "Volatility",
        "volatility;rm",
        "$(id)",
        "../etc",
        "a\x00b",
        "x" * 10_000,
        "volatility,,alpha",
    ]
    for raw in samples:
        sink.clear()
        with pytest.raises(FamilySurfaceError):
            if "," in raw or raw.strip() != raw:
                parse_only(raw)
            run_families(_config("binance_public_data"), [raw], write_artifact=False)
        assert sink == []


def test_duplicates_and_whitespace_only_are_refused() -> None:
    with pytest.raises(FamilySurfaceError, match="duplicate"):
        parse_only("volatility, volatility")
    with pytest.raises(FamilySurfaceError, match="at least one"):
        parse_only("   ")
    assert parse_only(" volatility , alpha ") == ["volatility", "alpha"]


def test_only_the_named_bench_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: list[str] = []
    _patch_benches(monkeypatch, sink)
    monkeypatch.setattr(
        importlib.import_module("quant_fund.research.benches"),
        "bench_cpcv_audit",
        lambda: {"fold_separation": 1.0},
    )
    payload = run_families(_config("synthetic"), ["cpcv"], write_artifact=False)
    assert sink == []
    assert payload["family_ids"] == ["cpcv"]
    assert payload["partial"] is True
    assert payload["verified"] is False
    assert payload["evidence_class"] == "SYNTHETIC"
    assert payload["data_source"] == "SYNTHETIC"
    assert set(payload["results"]) == {"cpcv"}


def test_synthetic_frame_cannot_be_labeled_as_a_real_source() -> None:
    called: list[str] = []

    def loader(_config: object) -> PanelBundle:
        frame = _frame("synthetic")
        return PanelBundle(frame=frame, label="future_return_1", northset=frame)

    def runner(*_args: object) -> dict[str, float]:
        called.append("ran")
        return {"qlike": 1.0}

    with pytest.raises(FamilySurfaceError, match="synthetic"):
        run_families(
            _config("binance_public_data"),
            ["volatility"],
            runners={"volatility": runner},
            load_panel=loader,
            write_artifact=False,
        )
    assert called == []


def test_market_frame_cannot_be_labeled_synthetic() -> None:
    def loader(_config: object) -> PanelBundle:
        frame = _frame("binance_public_data")
        return PanelBundle(frame=frame, label="future_return_1", northset=frame)

    def runner(*_args: object) -> dict[str, float]:
        raise AssertionError("runner must not start on a mislabeled panel")

    with pytest.raises(FamilySurfaceError, match="does not match panel source"):
        run_families(
            _config("synthetic"),
            ["volatility"],
            runners={"volatility": runner},
            load_panel=loader,
            write_artifact=False,
        )


def test_unmarked_panel_cannot_be_called_market_data() -> None:
    import polars as pl

    frame = pl.DataFrame({"future_return_1": [0.01]})

    def loader(_config: object) -> PanelBundle:
        return PanelBundle(frame=frame, label="future_return_1", northset=frame)

    with pytest.raises(FamilySurfaceError, match="without a panel source"):
        run_families(
            _config("file"),
            ["liquidity"],
            runners={"liquidity": lambda *_a: {"liquidity score": 1.0}},
            load_panel=loader,
            write_artifact=False,
        )


def test_panel_less_bench_is_not_market_evidence() -> None:
    payload = run_families(
        _config("binance_public_data"),
        ["cpcv"],
        runners={"cpcv": lambda: {"fold_separation": 1.0}},
        write_artifact=False,
    )
    assert payload["evidence_class"] == "SYNTHETIC"
    assert payload["data_source"] == "SYNTHETIC"
    assert payload["verified"] is False


def test_stale_synthetic_manifest_is_refused_before_the_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame("synthetic")
    monkeypatch.setattr("quant_fund.pipeline.dataset.build_gold", lambda *_a, **_k: (frame, frame))
    monkeypatch.setattr("quant_fund.pipeline.dataset.panel", lambda *_a, **_k: frame)
    monkeypatch.setattr("quant_fund.pipeline.dataset.ensure_silver", lambda *_a, **_k: frame)
    manifest = tmp_path / "metadata" / "data_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"source": "synthetic"}), encoding="utf-8")
    called: list[str] = []

    def runner(*_args: object) -> dict[str, float]:
        called.append("ran")
        return {"qlike": 1.0}

    with pytest.raises(FamilySurfaceError, match="manifest source"):
        run_families(
            _config("file", tmp_path),
            ["volatility"],
            runners={"volatility": runner},
        )
    assert called == []
    assert list(tmp_path.glob("family_runs/*")) == []


def test_matching_manifest_seals_an_unverified_partial_record(tmp_path: Path) -> None:
    frame = _frame("file")

    def loader(_config: object) -> PanelBundle:
        return PanelBundle(frame=frame, label="future_return_1", northset=frame)

    payload = run_families(
        _config("file", tmp_path),
        ["volatility"],
        runners={"volatility": lambda *_a: {"qlike": 0.2}},
        load_panel=loader,
    )
    assert payload["evidence_class"] == "file"
    assert payload["data_source"] == "file"
    assert payload["verified"] is False
    assert payload["partial"] is True
    artifact = Path(payload["artifact"])
    result = verify_research_artifact(artifact)
    assert result["valid"] is False
    assert result["errors"]


def test_retired_lookup_keeps_the_reason_and_old_receipts_verify(tmp_path: Path) -> None:
    contract = family_contract(RETIRED_SAMPLE)
    assert contract["qualification"] == "retired"
    assert contract["executable"] is False
    reason = contract["retirement_reason"]
    assert isinstance(reason, str) and "template" in reason.lower()
    with pytest.raises(FamilySurfaceError, match="retired"):
        run_families(_config("synthetic"), [RETIRED_SAMPLE], write_artifact=False)
    path = _write_receipt(tmp_path, 2, RETIRED_SAMPLE)
    result = verify_research_artifact(path)
    assert result["valid"] is True, result["errors"]


def test_research_only_cli_refuses_injection_and_seals_only_cpcv(tmp_path: Path) -> None:
    config = tmp_path / "research.yaml"
    root = tmp_path / "lake"
    config.write_text(
        "runtime:\n  mode: research\ndata:\n  source: synthetic\n"
        f"  root: {root}\ntrain:\n  ranking_target: future_return_1\n  random_seed: 1\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(_ROOT / "src"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "FX1_CONFIG_DIR": str(tmp_path / "fx1"),
        "FX1_SESSION_DIR": str(tmp_path / "sessions"),
        "OMP_NUM_THREADS": "1",
    }
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "dipcatcher_cli",
            "research",
            "--config",
            str(config),
            "--only",
            "volatility;rm",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert refused.returncode == 1
    assert "PARTIAL_FAMILY_RUN" not in refused.stdout
    ran = subprocess.run(
        [
            sys.executable,
            "-m",
            "dipcatcher_cli",
            "research",
            "--config",
            str(config),
            "--only",
            "cpcv",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert "DATA_LABEL=SYNTHETIC" in ran.stdout
    assert "SYNTHETIC" in ran.stdout
    assert "PARTIAL_FAMILY_RUN verified=false" in ran.stdout
    assert "families=cpcv" in ran.stdout
    artifact = Path(
        next(line.split("=", 1)[1] for line in ran.stdout.splitlines() if line.startswith("json="))
    )
    record = json.loads(artifact.read_text(encoding="utf-8"))
    assert record["schema"] == "dipcatcher.family_run.v1"
    assert record["verified"] is False
    assert record["family_ids"] == ["cpcv"]
    checked = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", "verify-research", str(artifact)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert checked.returncode == 1
    body = json.loads(checked.stdout)
    assert body["valid"] is False
