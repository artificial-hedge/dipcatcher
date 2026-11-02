"""Production-readiness scorecard. Passes are observations, never defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.paper.trading_session import module_network_imports
from quant_fund.simtest.prod_readiness import (
    ProdReadinessError,
    _anchors,
    _epochs,
    _fx1_weights,
    _guard,
    _row,
    _tail,
    main,
)

pytestmark = pytest.mark.prod_readiness


def test_a_pass_requires_an_observation() -> None:
    with pytest.raises(ProdReadinessError, match="disagrees"):
        _row("demo", "pass", "looks fine", ok=False)
    with pytest.raises(ProdReadinessError, match="observation"):
        _row("demo", "pass", "looks fine", observed=False)
    row = _row("demo", "needs-owner", "owner must sign")
    assert row["status"] == "needs-owner"
    assert row["evidence"]["ok"] is False
    assert row["evidence"]["observed"] is True


def test_epoch_and_anchor_staleness_is_a_failure() -> None:
    stale = _epochs(
        {
            "epoch:src": {"ok": False, "errors": ["unstamped_member:'new.py'"]},
            "epoch:docs": {"ok": True, "errors": []},
        }
    )
    assert stale["status"] == "fail"
    assert stale["evidence"]["ok"] is False
    fresh = _anchors(
        {
            "ok": True,
            "anchored": True,
            "errors": [],
            "fresh": {"quality/epoch_heads.json": False},
        }
    )
    assert fresh["status"] == "fail"
    absent = _anchors({"ok": True, "anchored": False, "errors": [], "fresh": {}})
    assert absent["status"] == "needs-owner"


def test_api_guard_and_weights_do_not_treat_fixtures_as_a_release(tmp_path: Path) -> None:
    source = Path("src/quant_fund/simtest/prod_readiness.py").read_text(encoding="utf-8")
    assert module_network_imports(source) == []
    bare = tmp_path / "api.py"
    bare.write_text("def app():\n    return None\n", encoding="utf-8")
    assert (
        _guard(tmp_path, "api.py", ("hmac.compare_digest",), "api_auth_guard")["status"] == "fail"
    )
    real = _guard(
        Path("."),
        "src/fx1/serve/api.py",
        ("hmac.compare_digest", "X-API-Key", "non-loopback binding requires FX1_API_KEY"),
        "api_auth_guard",
    )
    assert real["status"] == "pass"
    card = tmp_path / "artifacts" / "fx1_tiny_lm"
    card.mkdir(parents=True)
    (card / "weights.safetensors").write_bytes(b"fixture")
    (card / "modelcard.json").write_text(
        json.dumps({"version": "fx-1.v0", "known_limits": ["not an fx-1 release candidate"]}),
        encoding="utf-8",
    )
    assert _fx1_weights(tmp_path)["status"] == "fail"
    published = tmp_path / "models" / "fx1"
    published.mkdir(parents=True)
    (published / "weights.safetensors").write_bytes(b"weights")
    (published / "modelcard.json").write_text(
        json.dumps({"version": "fx-1.1", "base_model": "moonshotai/Kimi-K3", "known_limits": []}),
        encoding="utf-8",
    )
    assert _fx1_weights(tmp_path)["status"] == "pass"


def test_failure_tail_quotes_the_finding() -> None:
    body = '{\n  "errors": [\n    "scorecard_invalid:extra_conformal"\n  ]\n}\n'
    assert _tail(body) == '"scorecard_invalid:extra_conformal"'
    assert _tail("research_receipt: invalid\nshape floors ok\n") == "research_receipt: invalid"


def test_scorecard_on_this_tree_exits_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "readiness.json"
    code = main(["--root", ".", "--out", str(out), "--work", str(tmp_path / "work")])
    receipt = json.loads(out.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in receipt["checks"]}
    assert code == 1
    assert receipt["ready"] is False
    assert receipt["fail_count"] >= 1
    assert receipt["live_pnl_claim"] is False
    assert by_id["fx1_weights"]["status"] == "fail"
    assert by_id["release_tag"]["status"] == "fail"
    assert by_id["pin_signatures"]["status"] == "fail"
    assert by_id["epoch_chains"]["status"] == "fail"
    assert by_id["timestamp_anchors"]["status"] == "fail"
    assert by_id["checkpoint"]["status"] == "fail"
    assert by_id["api_auth_guard"]["status"] == "pass"
    assert by_id["kill_switch"]["status"] == "pass"
    assert by_id["e2e_smoke"]["status"] == "pass"
    assert by_id["paper_session"]["status"] == "pass"
    assert by_id["paper_soak"]["status"] == "pass"
    assert by_id["release_dry_run"]["status"] == "pass"
    assert by_id["live_block"]["status"] == "pass"
    assert by_id["live_condition_1"]["status"] == "needs-owner"
    assert by_id["qualifying_feed_closed"]["status"] == "pass"
    assert all(row["evidence"]["observed"] is True for row in receipt["checks"])
    assert all(row["evidence"]["ok"] is (row["status"] == "pass") for row in receipt["checks"])
