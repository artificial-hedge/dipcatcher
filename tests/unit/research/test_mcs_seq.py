"""Anytime-valid MCS — coverage, elimination dynamics, fail-closed edges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.research.mcs_seq import AnytimeMCS, mcs_report


def _iid_streams(heads: list[str], n: int, seed: int, means: dict[str, float] | None = None):
    rng = np.random.default_rng(seed)
    means = means or {}
    return {h: rng.normal(means.get(h, 0.0), 1.0, n) for h in heads}


def test_survivor_set_covers_under_null() -> None:
    """All-equal losses: a fixed head is eliminated in <= alpha of runs.

    Under the all-optimal null every head is optimal, so each head's
    elimination frequency bounds the coverage error directly.
    """
    heads = ["a", "b", "c", "d"]
    n_runs, alpha, n = 400, 0.10, 60
    elim_count = {h: 0 for h in heads}
    for s in range(n_runs):
        streams = _iid_streams(heads, n, seed=1000 + s)
        mcs = AnytimeMCS(tuple(heads), alpha=alpha, lam=0.5)
        for t in range(n):
            mcs.update({h: float(streams[h][t]) for h in heads})
        for h in mcs.eliminated:
            elim_count[h] += 1
    mc_slack = 3.0 * np.sqrt(alpha * (1 - alpha) / n_runs)
    for h in heads:
        rate = elim_count[h] / n_runs
        assert rate <= alpha + mc_slack, f"{h} eliminated at {rate:.3f} > {alpha} + slack"


def test_dominant_head_survives_and_eliminates_others() -> None:
    heads = ["champ", "x", "y", "z"]
    streams = _iid_streams(heads, 400, seed=0, means={"champ": -0.5})
    rep = mcs_report(streams, alpha=0.05)
    assert rep["champion"] == "champ"
    assert "champ" in rep["survivors"]
    assert set(rep["eliminated"]) == {"x", "y", "z"}


def test_two_equal_best_both_survive() -> None:
    heads = ["g1", "g2", "bad1", "bad2"]
    streams = _iid_streams(
        heads, 300, seed=7, means={"g1": -0.4, "g2": -0.4, "bad1": 0.0, "bad2": 0.0}
    )
    rep = mcs_report(streams, alpha=0.05)
    assert set(rep["survivors"]) == {"g1", "g2"}
    assert rep["champion"] is None  # 2 survivors — no unique champion


def test_elimination_is_permanent() -> None:
    """Once crossed, a later dip below the threshold must not resurrect."""
    heads = ["a", "b"]
    mcs = AnytimeMCS(tuple(heads), alpha=0.5, lam=0.9)
    rng = np.random.default_rng(3)
    saw_elim = False
    for _ in range(500):
        mcs.update({"a": float(rng.normal(0.3, 1.0)), "b": float(rng.normal(0.0, 1.0))})
        if "a" in mcs.eliminated:
            saw_elim = True
            elim_at = mcs.eliminated["a"]
        if saw_elim:
            # a stays eliminated regardless of subsequent evidence
            assert "a" not in mcs.states[-1].survivors
    assert saw_elim, "expected a to be eliminated at some origin"
    assert elim_at < 500


def test_fail_closed_edges() -> None:
    mcs = AnytimeMCS(("a", "b"))
    with pytest.raises(ValueError, match="cover exactly"):
        mcs.update({"a": 1.0})  # missing head
    with pytest.raises(ValueError, match="cover exactly"):
        mcs.update({"a": 1.0, "b": 2.0, "c": 3.0})  # unknown head
    with pytest.raises(ValueError, match="non-finite"):
        mcs.update({"a": float("nan"), "b": 1.0})
    with pytest.raises(ValueError):
        AnytimeMCS(("solo",))
    with pytest.raises(ValueError):
        AnytimeMCS(("a", "a"))
    with pytest.raises(ValueError):
        mcs_report({"a": np.ones(10)})  # single head
    with pytest.raises(ValueError):
        mcs_report({"a": np.ones(10), "b": np.ones(11)})  # length mismatch
    with pytest.raises(ValueError):
        mcs_report({"a": np.ones(10), "b": np.full(10, np.inf)})


def test_receipt_shape_and_monotone_shrink() -> None:
    heads = ["a", "b", "c"]
    streams = _iid_streams(heads, 50, seed=1)
    rep = mcs_report(streams, alpha=0.1)
    assert rep["kind"] == "mcs_seq.v1"
    assert rep["n_heads"] == 3
    assert rep["n_origins"] == 50
    assert rep["n_eliminated"] == len(rep["eliminated"])
    assert len(rep["survivors"]) + rep["n_eliminated"] == 3
    assert rep["coverage_guarantee"].startswith("P(")
    # survivor set is non-increasing over time
    mcs = AnytimeMCS(tuple(heads), alpha=0.1)
    prev = set(heads)
    for t in range(50):
        st = mcs.update({h: float(streams[h][t]) for h in heads})
        assert set(st.survivors) <= prev
        prev = set(st.survivors)


def test_causality_future_cannot_rewrite_past() -> None:
    heads = ["a", "b"]
    rng = np.random.default_rng(5)
    base = [{"a": rng.normal(0, 1), "b": rng.normal(0, 1)} for _ in range(30)]

    mcs1 = AnytimeMCS(tuple(heads), alpha=0.05)
    for d in base:
        mcs1.update(d)
    snap = [(s.origin, s.survivors, s.n_eliminated) for s in mcs1.states]

    mcs2 = AnytimeMCS(tuple(heads), alpha=0.05)
    for d in base:
        mcs2.update(d)
    for _ in range(30):  # radically different future
        mcs2.update({"a": rng.normal(10, 1), "b": rng.normal(-10, 1)})
    snap2 = [(s.origin, s.survivors, s.n_eliminated) for s in mcs2.states[:30]]
    assert snap == snap2


def test_mcs_cli_writes_sealed_receipt(tmp_path: Path) -> None:
    """`dipcatcher mcs` loads streams, seals an mcs_seq.v1 receipt."""
    import json

    import numpy as np
    from typer.testing import CliRunner

    from quant_fund.cli.main import app
    from quant_fund.research.receipt_v2 import verify_receipt_file

    rng = np.random.default_rng(5)
    streams = {
        "best": (rng.standard_normal(300) * 0.01).tolist(),
        "worse": (rng.standard_normal(300) * 0.01 + 0.02).tolist(),
        "worst": (rng.standard_normal(300) * 0.01 + 0.05).tolist(),
    }
    src = tmp_path / "streams.json"
    src.write_text(json.dumps(streams))
    out = tmp_path / "receipts"

    result = CliRunner().invoke(
        app,
        ["mcs", str(src), "--alpha", "0.05", "--data-label", "SYNTHETIC", "--out-dir", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    assert "champion=best" in result.output
    receipts = list(out.glob("mcs_seq_*.json"))
    assert len(receipts) == 1
    assert verify_receipt_file(receipts[0])["valid"]


def test_mcs_cli_fail_closed_on_bad_input(tmp_path: Path) -> None:
    """Non-map JSON / missing file / <2 heads all exit non-zero."""
    import json

    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"only": [1.0, 2.0]}))
    assert CliRunner().invoke(app, ["mcs", str(bad)]).exit_code != 0
    assert CliRunner().invoke(app, ["mcs", str(tmp_path / "gone.json")]).exit_code != 0
    arr = tmp_path / "arr.json"
    arr.write_text(json.dumps([1.0, 2.0]))
    assert CliRunner().invoke(app, ["mcs", str(arr)]).exit_code != 0
