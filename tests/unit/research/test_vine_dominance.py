"""Pins for the vine_dominance OOS copula-dominance bench."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from quant_fund.research.vine_dominance import (
    VINE_DOMINANCE_SCHEMA,
    block_corr,
    mvt_copula_sample,
    vine_dominance_bench,
    write_vine_dominance_receipt,
)

_REPO = Path(__file__).resolve().parents[3]


def test_mvt_copula_planted_tail_dependence() -> None:
    """t-copula has real joint tail mass a Gaussian copula lacks."""
    rng = np.random.default_rng(7)
    corr = block_corr([3, 3], 0.7, 0.25)
    u = mvt_copula_sample(60_000, corr, nu=5.0, rng=rng)
    joint_emp = float(np.all(u < 0.05, axis=1).mean())
    assert joint_emp > 1e-3  # Gaussian copula tail dep = 0 -> ~q^d ~ 3e-8


def test_block_corr_is_correlation() -> None:
    c = block_corr([4, 4], 0.7, 0.25)
    assert np.allclose(np.diag(c), 1.0)
    assert np.linalg.eigvalsh(c).min() > 0
    assert c[0, 1] == pytest.approx(0.7) and c[0, 7] == pytest.approx(0.25)


def test_vine_dominance_bench_runs_and_beats_independent() -> None:
    p = vine_dominance_bench(n_train=1200, n_test=600, n_sim=4000, seed=11)
    claim = p["claim"]
    assert p["schema"] == VINE_DOMINANCE_SCHEMA
    assert p["data_label"] == "SYNTHETIC"
    assert claim["mean_loglik_oos"]["rvine"] > 0.0
    assert claim["mean_loglik_oos"]["rvine"] > claim["mean_loglik_oos"]["gaussian"]
    # empirical joint tail is heavy; gaussian copula must underestimate it
    assert claim["tail_empirical"]["joint_crash_prob"] > 0.0
    assert (
        claim["tail_model"]["gaussian"]["joint_crash_prob"]
        < claim["tail_empirical"]["joint_crash_prob"]
    )
    for v in claim["dominance"].values():
        assert v["cs_lo"] <= v["cs_hi"]


def test_vine_dominance_deterministic() -> None:
    a = vine_dominance_bench(n_train=800, n_test=400, n_sim=3000, seed=5)
    b = vine_dominance_bench(n_train=800, n_test=400, n_sim=3000, seed=5)
    assert a["claim"]["mean_loglik_oos"] == b["claim"]["mean_loglik_oos"]
    assert a["claim"]["dominance"] == b["claim"]["dominance"]


def test_committed_receipt_verifies() -> None:
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    path = _REPO / "receipts" / "vine_dominance.json"
    if not path.exists():
        pytest.skip("committed receipt not present yet")
    v = verify_receipt_payload(json.loads(path.read_text()))
    assert v["valid"], v["errors"]


def test_receipt_writer_seals_and_verifies(tmp_path: Path) -> None:
    p = vine_dominance_bench(n_train=600, n_test=300, n_sim=2000, seed=3)
    out = write_vine_dominance_receipt(p, tmp_path)
    payload = json.loads(out.read_text())
    assert payload["receipt_sha256"] and len(payload["receipt_sha256"]) == 64
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    assert verify_receipt_payload(payload)["valid"]


def test_receipt_writer_refuses_foreign_kind(tmp_path: Path) -> None:
    p = vine_dominance_bench(n_train=600, n_test=300, n_sim=2000, seed=3)
    p["kind"] = "other"
    with pytest.raises(ValueError):
        write_vine_dominance_receipt(p, tmp_path)


def test_forbidden_metric_keys_absent() -> None:
    """Receipt payload must not carry headline-metric keys anywhere."""
    p = vine_dominance_bench(n_train=500, n_test=250, n_sim=1500, seed=9)

    def _walk(o: object) -> list[str]:
        hits: list[str] = []
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() in {"sharpe", "sortino", "calmar", "pnl", "nav"}:
                    hits.append(str(k))
                hits += _walk(v)
        elif isinstance(o, list):
            for v in o:
                hits += _walk(v)
        return hits

    assert _walk(p) == []


def test_git_revision_present() -> None:
    p = vine_dominance_bench(n_train=400, n_test=200, n_sim=1200, seed=21)
    assert (
        p["git_revision"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def test_committed_receipt_dispatches_contract() -> None:
    from quant_fund.research.lane_contracts import lane_contract_errors

    committed = json.loads(Path("receipts/vine_dominance.json").read_text())
    assert committed["schema"] == "vine_dominance.v1"
    assert lane_contract_errors(committed) == []


def test_contract_catches_incoherent_claim() -> None:
    from quant_fund.research.vine_dominance import vine_dominance_contract_errors

    p = json.loads(Path("receipts/vine_dominance.json").read_text())
    p["claim"]["n_passed"] = 0
    assert "n_passed_mismatch" in vine_dominance_contract_errors(p)
    p2 = json.loads(Path("receipts/vine_dominance.json").read_text())
    p2["claim"]["dominance"]["rvine_vs_cvine"]["verdict"] = "totally_dominates"
    assert "dominance_rvine_vs_cvine_verdict" in vine_dominance_contract_errors(p2)
    p3 = json.loads(Path("receipts/vine_dominance.json").read_text())
    p3["claim"]["tail_model"]["gaussian"]["joint_crash_prob"] = 2.0
    assert "tail_model_shape" in vine_dominance_contract_errors(p3)
