"""Tests for microstructure/price_clustering.py — round-number clustering.

Mod-m residue clustering of limit-order placement, reachable-normalized so it
works on integer tick levels (the ZI-LOB sim) and on real prices snapped to a
tick grid. Everything is **labeled SYNTHETIC** correctness validation —
never market evidence, no live-trading claim.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quant_fund.microstructure.price_clustering import (
    cluster_stats,
    levels_from_prices,
    price_clustering_bench,
    sim_cluster_stats,
)
from quant_fund.microstructure.zi_lob_simulator import santa_fe_config

# ---------------------------------------------------------------------------
# cluster_stats — uniform null, planted signal, KATs, fail-closed edges
# ---------------------------------------------------------------------------


def test_uniform_stream_excess_near_zero() -> None:
    """Level-uniform submits over a contiguous span: no clustering."""
    rng = np.random.default_rng(0)
    levels = rng.integers(-100, 101, size=200_000)
    out = cluster_stats(levels, 10)
    assert abs(out["excess_round_share"]) < 0.01
    assert out["chi2_p_value"] is not None and out["chi2_p_value"] > 0.05


def test_planted_residue0_preference_detected() -> None:
    """A 2x weight on residue-0 levels must light the detector up."""
    rng = np.random.default_rng(1)
    universe = np.arange(200, dtype=np.int64)
    weights = np.where(np.mod(universe, 10) == 0, 2.0, 1.0)
    levels = rng.choice(universe, size=100_000, p=weights / weights.sum())
    out = cluster_stats(levels, 10)
    # Baseline share of residue 0 is 20/200 = 0.10; planted share ≈ 40/220.
    assert out["excess_round_share"] > 0.05
    assert out["chi2_p_value"] is not None and out["chi2_p_value"] < 1e-6


def test_cluster_stats_uniform_span_kat() -> None:
    """Hand-computed: levels 0..9 each once, m=5 — uniform span universe."""
    out = cluster_stats(np.arange(10), 5)
    # Span [0, 9]: two reachable levels per residue → baseline 0.2 everywhere;
    # observed counts are 2 per residue → share 0.2, chi-square vanishes.
    assert out["residue_shares"] == pytest.approx([0.2] * 5)
    assert out["uniform_baseline_shares"] == pytest.approx([0.2] * 5)
    assert out["excess_round_share"] == pytest.approx(0.0)
    assert out["chi2_stat"] == pytest.approx(0.0)
    assert out["chi2_dof"] == 4


def test_cluster_stats_concentrated_kat() -> None:
    """Hand-computed: all submits at level 0, reachable [0..4], m=2."""
    out = cluster_stats(np.zeros(10, dtype=np.int64), 2, reachable_levels=np.arange(5))
    # Reachable {0..4}: residues {0:3, 1:2} → baseline (0.6, 0.4); observed
    # (10, 0) → expected (6, 4) → chi2 = 16/6 + 16/4 = 20/3; excess = 0.4.
    assert out["uniform_baseline_shares"] == pytest.approx([0.6, 0.4])
    assert out["excess_round_share"] == pytest.approx(0.4)
    assert out["chi2_stat"] == pytest.approx(20.0 / 3.0)
    assert out["chi2_dof"] == 1
    assert out["chi2_p_value"] is not None and out["chi2_p_value"] < 0.01


def test_cluster_stats_matched_baseline_kat() -> None:
    """Observed shares equal to the non-uniform baseline → chi2 = 0."""
    out = cluster_stats(np.array([0, 0, 0, 1, 1], dtype=np.int64), 2, reachable_levels=np.arange(5))
    assert out["excess_round_share"] == pytest.approx(0.0)
    assert out["chi2_stat"] == pytest.approx(0.0)


def test_cluster_stats_negative_levels_residue() -> None:
    """Negative levels use mathematical residues (-1 mod 10 == 9)."""
    out = cluster_stats(np.array([-1, -2, -3, -4, -5], dtype=np.int64), 10)
    for r in (5, 6, 7, 8, 9):
        assert out["residue_shares"][r] == pytest.approx(0.2)


def test_cluster_stats_empty_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cluster_stats(np.array([], dtype=np.int64), 10)


@pytest.mark.parametrize("bad_m", [0, 1, -3])
def test_cluster_stats_bad_modulus_fail_closed(bad_m: int) -> None:
    with pytest.raises(ValueError, match=">= 2"):
        cluster_stats(np.arange(10), bad_m)


def test_cluster_stats_nonint_modulus_fail_closed() -> None:
    with pytest.raises((TypeError, ValueError)):
        cluster_stats(np.arange(10), 2.5)  # type: ignore[arg-type]


def test_cluster_stats_noninteger_levels_fail_closed() -> None:
    with pytest.raises(ValueError, match="integer-valued"):
        cluster_stats(np.array([0.5, 1.5, 2.5]), 10)


def test_cluster_stats_nan_levels_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        cluster_stats(np.array([0.0, np.nan, 2.0]), 10)


def test_cluster_stats_out_of_band_fail_closed() -> None:
    """A submit outside the declared reachable universe is a contract break."""
    with pytest.raises(ValueError, match="outside the reachable universe"):
        cluster_stats(np.array([0, 1, 7], dtype=np.int64), 2, reachable_levels=np.arange(3))


def test_cluster_stats_single_residue_universe() -> None:
    """Universe inside one residue class: no clustering test is defined."""
    out = cluster_stats(
        np.array([0, 10, 20], dtype=np.int64),
        10,
        reachable_levels=np.array([0, 10, 20, 30], dtype=np.int64),
    )
    assert out["chi2_dof"] == 0
    assert out["chi2_p_value"] is None
    assert out["excess_round_share"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# levels_from_prices — real-price path
# ---------------------------------------------------------------------------


def test_levels_from_prices_roundtrip() -> None:
    prices = np.array([100.00, 100.01, 99.99, 100.10])
    levels = levels_from_prices(prices, tick=0.01, ref_price=100.0)
    assert levels.tolist() == [0, 1, -1, 10]
    out = cluster_stats(levels, 10, reachable_levels=np.arange(-10, 11))
    assert out["residue_counts"][0] == 2


def test_levels_from_prices_off_grid_fail_closed() -> None:
    with pytest.raises(ValueError, match="off the tick grid"):
        levels_from_prices(np.array([100.005]), tick=0.01)


def test_levels_from_prices_bad_tick_fail_closed() -> None:
    with pytest.raises(ValueError, match="tick"):
        levels_from_prices(np.array([100.0]), tick=0.0)


# ---------------------------------------------------------------------------
# sim_cluster_stats — oid-diff attribution + reachable windows (SYNTHETIC)
# ---------------------------------------------------------------------------


def test_sim_cluster_stats_deterministic_and_exact() -> None:
    a = sim_cluster_stats(horizon=2_000.0, seed=7, m=10)
    b = sim_cluster_stats(horizon=2_000.0, seed=7, m=10)
    assert a["submit_levels"] == b["submit_levels"]
    assert a["reachable_levels"] == b["reachable_levels"]
    assert a["n_submits"] > 0
    # Every attributed submit lies inside the reachable union — by
    # construction of the oid-diff + window tracking, not by assertion luck.
    assert set(a["submit_levels"]) <= set(a["reachable_levels"])
    assert a["cluster"]["n_submits"] == a["n_submits"]
    # The real-price path measures the identical residues.
    assert a["price_domain_residues_match"] is True
    # ZI flow plants no clustering: small excess at this horizon.
    assert abs(a["cluster"]["excess_round_share"]) < 0.05


def test_sim_cluster_stats_ref_anchor_reachable_excludes_residue0() -> None:
    """Frozen ref anchor: submits only at ±1..±band → residue 0 unreachable."""
    cfg = replace(santa_fe_config(seed=3, band=5), anchor="ref")
    out = sim_cluster_stats(horizon=2_000.0, m=10, config=cfg)
    reach = set(out["reachable_levels"])
    assert reach <= set(range(-5, 0)) | set(range(1, 6))
    assert reach
    assert out["cluster"]["reachable_counts"][0] == 0
    # With residue 0 unreachable the normalized excess stays ~0 — a naive
    # 1/m baseline would read -0.1 here and fabricate anti-clustering.
    assert out["cluster"]["excess_round_share"] == pytest.approx(0.0)


def test_sim_cluster_stats_markov_regime_arm() -> None:
    from quant_fund.microstructure.zi_lob_simulator import (
        MarkovRegimeFlow,
        RegimeState,
    )

    flow = MarkovRegimeFlow(
        states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
        stay_probs=(0.995, 0.985),
        seed=5,
    )
    out = sim_cluster_stats(flow=flow, horizon=2_000.0, seed=5, m=10)
    assert out["n_submits"] > 0
    assert abs(out["cluster"]["excess_round_share"]) < 0.05


def test_sim_cluster_stats_empty_horizon_fail_closed() -> None:
    with pytest.raises(ValueError, match="horizon"):
        sim_cluster_stats(horizon=0.0, seed=0)


# ---------------------------------------------------------------------------
# price_clustering_bench — sealed receipt
# ---------------------------------------------------------------------------


def test_bench_receipt_verifies_clean() -> None:
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    receipt = price_clustering_bench(horizon=1_000.0, seed=11, m=10, band=5)
    assert receipt["schema"] == "price_clustering.v1"
    assert receipt["kind"] == "price_clustering"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["research_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert set(receipt["arms"]) == {"touch_zi", "markov_regime", "ref_band"}
    result = verify_receipt_payload(receipt)
    assert result["errors"] == [], result
    assert result["digest_convention"] == "canonical_json"


def test_bench_receipt_seal_is_canonical() -> None:
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    receipt = price_clustering_bench(horizon=500.0, seed=13, m=10)
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert receipt["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))


def test_detector_probe_fires_on_planted_flow() -> None:
    receipt = price_clustering_bench(horizon=300.0, seed=17, m=10)
    probe = receipt["detector_probe"]
    assert probe["detected"] is True
    assert probe["excess_round_share"] > 0.0
