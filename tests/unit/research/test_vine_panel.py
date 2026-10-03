"""Vine panel bench — joint-tail claims on the committed cross-section."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.vine_panel import (
    PANEL_PATH,
    _empirical_pit,
    _load_panel_returns,
    vine_panel_bench,
    write_vine_panel_receipt,
)

pytestmark = pytest.mark.skipif(not PANEL_PATH.is_file(), reason="silver panel not checked out")


@pytest.fixture(scope="module")
def payload() -> dict:
    return vine_panel_bench(n_sim=2000, max_symbols=8)


def test_panel_loads_deepest_series() -> None:
    rets, syms = _load_panel_returns(PANEL_PATH, max_symbols=6, min_obs=400)
    assert rets.shape[1] == 6
    assert len(syms) == 6
    assert np.isfinite(rets).all()


def test_empirical_pit_bounds_and_monotone() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, 4))
    obs = rng.normal(size=(50, 4))
    u = _empirical_pit(train, obs)
    assert u.shape == (50, 4)
    assert np.all(u > 0) and np.all(u < 1)
    # monotone: sorted obs map to sorted PITs
    order = np.argsort(obs[:, 0])
    assert np.all(np.diff(u[order, 0]) >= -1e-12)


def test_empirical_pit_no_refit_leakage() -> None:
    """Observations beyond the train range clip to (eps, 1-eps), never NaN."""
    train = np.linspace(-1, 1, 200).reshape(200, 1)
    obs = np.array([[5.0], [-5.0]])
    u = _empirical_pit(train, obs)
    assert np.isfinite(u).all()
    assert u[0, 0] > 0.99 and u[1, 0] < 0.01


def test_bench_runs_and_probes(payload: dict) -> None:
    assert payload["kind"] == "vine_panel.v1"
    assert payload["claim"]["ok"], {k: v for k, v in payload["claim"]["results"].items() if not v}
    assert payload["claim"]["n_probes"] >= 8


def test_vines_beat_independent_oos(payload: dict) -> None:
    ll = payload["claim"]["oos_mean_loglik"]
    assert ll["independent"] == pytest.approx(0.0)
    for s in ("cvine", "dvine", "rvine"):
        assert ll[s] > ll["independent"]


def test_joint_crash_estimates_bounded(payload: dict) -> None:
    for name, est in payload["claim"]["joint_crash"]["estimates"].items():
        assert 0 <= est <= 1, name
    # every fitted copula understates the test fold's joint-tail frequency —
    # the pinned honest divergence, not a bug claim
    emp = payload["claim"]["joint_crash"]["empirical"]
    assert emp > 0


def test_determinism() -> None:
    a = vine_panel_bench(n_sim=500, max_symbols=6)
    b = vine_panel_bench(n_sim=500, max_symbols=6)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_committed_receipt_verifies() -> None:
    path = Path("receipts/vine_panel.json")
    if not path.is_file():
        pytest.skip("receipt not generated yet")
    v = verify_receipt_file(path)
    assert v["valid"], v["errors"]


def test_writer_seals_and_verifies(payload: dict, tmp_path: Path) -> None:
    path = write_vine_panel_receipt(payload, tmp_path)
    assert path.is_file()
    v = verify_receipt_file(path)
    assert v["valid"], v["errors"]


def test_writer_refuses_foreign_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vine_panel"):
        write_vine_panel_receipt({"kind": "other.v1"}, tmp_path)


def test_no_forbidden_metric_keys(payload: dict) -> None:
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}

    def walk(node) -> list[str]:
        hits: list[str] = []
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower() in forbidden:
                    hits.append(str(k))
                hits.extend(walk(v))
        elif isinstance(node, list):
            for v in node:
                hits.extend(walk(v))
        return hits

    assert walk(payload) == []


def test_git_revision_matches_head() -> None:
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    p = vine_panel_bench(n_sim=200, max_symbols=4, min_obs=300)
    assert p["git_revision"] == head


def test_receipt_sha256_resides(payload: dict) -> None:
    doc = json.loads(json.dumps(payload))
    assert doc["receipt_sha256"]
    stripped = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    assert hash_bytes(canonical_json_bytes(stripped)) == doc["receipt_sha256"]


def test_committed_receipt_dispatches_contract() -> None:
    from quant_fund.research.lane_contracts import lane_contract_errors

    committed = json.loads(Path("receipts/vine_panel.json").read_text())
    assert committed["schema"] == "vine_panel.v1"
    assert lane_contract_errors(committed) == []


def test_contract_catches_incoherent_claim(payload: dict) -> None:
    from quant_fund.research.vine_panel import vine_panel_contract_errors

    forged = json.loads(json.dumps(payload))
    forged["claim"]["n_passed"] = 0  # incoherent: ok still True
    errors = vine_panel_contract_errors(forged)
    assert "n_passed_mismatch" in errors

    forged2 = json.loads(json.dumps(payload))
    forged2["claim"]["joint_crash"]["estimates"]["rvine"] = 1.7
    assert "joint_crash_out_of_range" in vine_panel_contract_errors(forged2)
