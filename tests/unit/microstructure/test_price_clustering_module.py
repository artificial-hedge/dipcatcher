"""Unit tests for microstructure/price_clustering.py — SYNTHETIC round-number clustering.

Covers the statistic on hand-built level vectors (residue counts, reachable-set
normalisation and the chi-square computed by hand), the price→level snapping
contract, a small seeded ZI-LOB run through ``sim_cluster_stats``, and the
sealed ``price_clustering.v1`` receipt written to ``tmp_path``.

Everything here is a SYNTHETIC correctness diagnostic on simulated placement —
never market evidence, never a headline performance ratio, no live-trading
claim. The receipt assertions mirror the neighbouring receipt tests in this
directory (honesty fields present, forbidden headline tokens absent, and the
seal hash reproducible from the payload body).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.price_clustering import (
    PRICE_CLUSTERING_KIND,
    PRICE_CLUSTERING_RECEIPT,
    PRICE_CLUSTERING_SCHEMA,
    cluster_stats,
    levels_from_prices,
    price_clustering_bench,
    sim_cluster_stats,
    write_price_clustering_receipt,
)
from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    santa_fe_config,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

#: Mirrors tests/unit/microstructure/test_zi_lob_simulator.py — "pnl" is
#: permitted only inside the simulator-internal diagnostic namespace.
FORBIDDEN_HEADLINE_TOKENS = ("sharpe", "sortino", "calmar", "nav")

_BENCH = {"horizon": 250.0, "seed": 5, "m": 10, "band": 5}


# ---------------------------------------------------------------------------
# cluster_stats — hand-computed residue arithmetic
# ---------------------------------------------------------------------------


def test_cluster_stats_matches_hand_computed_residue_counts() -> None:
    """levels = [−10, 0, 10, 20, 1, 2] at m = 10.

    Observed: residue 0 gets four hits (−10, 0, 10, 20), residues 1 and 2 one
    each. Default reachable universe is the observed span −10..20 → 31 levels,
    carrying residue 0 four times (−10, 0, 10, 20) and residues 1..9 three
    times each (4 + 9·3 = 31). So the baseline share of residue 0 is 4/31, not
    1/10, and the excess is 4/6 − 4/31.
    """
    levels = np.array([-10, 0, 10, 20, 1, 2], dtype=np.int64)
    stats = cluster_stats(levels, 10)
    assert stats["label"] == "SYNTHETIC"
    assert stats["m"] == 10
    assert stats["n_submits"] == 6
    assert stats["n_reachable_levels"] == 31
    assert stats["residue_counts"] == [4, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert stats["reachable_counts"] == [4, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    assert stats["residue_shares"][0] == pytest.approx(4.0 / 6.0)
    assert stats["uniform_baseline_shares"][0] == pytest.approx(4.0 / 31.0)
    assert stats["excess_round_share"] == pytest.approx(4.0 / 6.0 - 4.0 / 31.0)
    assert sum(stats["residue_shares"]) == pytest.approx(1.0)
    assert sum(stats["uniform_baseline_shares"]) == pytest.approx(1.0)


def test_cluster_stats_chi2_matches_hand_computation() -> None:
    """Pearson chi-square over the residues with a reachable count > 0."""
    levels = np.array([-10, 0, 10, 20, 1, 2], dtype=np.int64)
    stats = cluster_stats(levels, 10)
    observed = [4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    reachable = [4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    n, total = 6.0, 31.0
    expected = [r / total * n for r in reachable]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected, strict=True) if e > 0.0)
    assert stats["chi2_stat"] == pytest.approx(chi2)
    assert stats["chi2_dof"] == 9  # ten reachable residues − 1
    p_value = stats["chi2_p_value"]
    assert p_value is not None and 0.0 <= p_value < 0.05, "planted clustering must fire"


def test_cluster_stats_residue_neutral_placement_has_zero_excess() -> None:
    """One submit per level of a full 0..9 span: observed == baseline exactly."""
    levels = np.tile(np.arange(10, dtype=np.int64), 5)
    stats = cluster_stats(levels, 10)
    assert stats["residue_counts"] == [5] * 10
    assert stats["uniform_baseline_shares"] == [0.1] * 10
    assert stats["excess_round_share"] == pytest.approx(0.0)
    assert stats["chi2_stat"] == pytest.approx(0.0)
    assert stats["chi2_p_value"] == pytest.approx(1.0)


def test_cluster_stats_reachable_normalisation_avoids_a_fake_deficit() -> None:
    """The reachability point of the module docstring, pinned.

    With a universe of ±1..±band (residue 0 literally unreachable) the baseline
    share of residue 0 is 0, so ``excess_round_share`` is 0 — where a naive
    1/m baseline would fabricate a −0.1 anti-clustering deficit.
    """
    universe = np.arange(1, 10, dtype=np.int64)
    stats = cluster_stats(universe, 10, reachable_levels=universe)
    assert stats["reachable_counts"][0] == 0
    assert stats["uniform_baseline_shares"][0] == pytest.approx(0.0)
    assert stats["excess_round_share"] == pytest.approx(0.0)
    assert stats["chi2_stat"] == pytest.approx(0.0)
    assert stats["chi2_dof"] == 8  # nine reachable residues − 1
    assert -0.1 < stats["excess_round_share"] < 0.0 + 1e-12


def test_cluster_stats_detects_a_planted_round_level_preference() -> None:
    """Half the submits on multiples of 10 → a strongly positive excess."""
    rng = np.random.default_rng(4)
    universe = np.arange(-30, 31, dtype=np.int64)
    weights = np.where(np.mod(universe, 10) == 0, 4.0, 1.0)
    levels = rng.choice(universe, size=4000, p=weights / weights.sum())
    stats = cluster_stats(levels, 10)
    baseline0 = float(stats["uniform_baseline_shares"][0])
    assert stats["excess_round_share"] > baseline0
    assert stats["chi2_p_value"] is not None
    assert stats["chi2_p_value"] < 1e-6


def test_cluster_stats_single_residue_universe_has_no_p_value() -> None:
    """A universe spanning one residue class leaves dof = 0 → p-value is None
    rather than a fabricated number."""
    levels = np.array([10, 20, 30], dtype=np.int64)
    stats = cluster_stats(levels, 10, reachable_levels=np.array([10, 20, 30], dtype=np.int64))
    assert stats["chi2_dof"] == 0
    assert stats["chi2_p_value"] is None
    assert stats["chi2_stat"] == pytest.approx(0.0)


def test_cluster_stats_accepts_integral_float_levels() -> None:
    a = cluster_stats(np.array([1.0, 2.0, 10.0]), 10)
    b = cluster_stats(np.array([1, 2, 10], dtype=np.int64), 10)
    assert a["residue_counts"] == b["residue_counts"]
    assert a["chi2_stat"] == pytest.approx(float(b["chi2_stat"]))


def test_cluster_stats_counts_are_ints_and_shares_are_floats() -> None:
    stats = cluster_stats(np.array([0, 1, 2], dtype=np.int64), 3)
    assert all(isinstance(c, int) for c in stats["residue_counts"])
    assert all(isinstance(c, int) for c in stats["reachable_counts"])
    assert all(isinstance(s, float) for s in stats["residue_shares"])
    assert isinstance(stats["n_submits"], int)
    assert isinstance(stats["n_reachable_levels"], int)


# ---------------------------------------------------------------------------
# cluster_stats — fail-closed validation
# ---------------------------------------------------------------------------


def test_cluster_stats_rejects_a_modulus_below_two() -> None:
    levels = np.array([1, 2, 3], dtype=np.int64)
    for bad in (1, 0, -10):
        with pytest.raises(ValueError, match="m must be >= 2"):
            cluster_stats(levels, bad)


def test_cluster_stats_rejects_non_int_modulus() -> None:
    levels = np.array([1, 2, 3], dtype=np.int64)
    for bad in (True, False, 10.0, "10"):
        with pytest.raises(TypeError, match="m must be an int"):
            cluster_stats(levels, bad)


def test_cluster_stats_rejects_empty_and_shaped_level_vectors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cluster_stats(np.empty(0, dtype=np.int64), 10)
    with pytest.raises(ValueError, match="1-D array"):
        cluster_stats(np.zeros((3, 2), dtype=np.int64), 10)


def test_cluster_stats_rejects_non_integer_levels() -> None:
    with pytest.raises(ValueError, match="integer-valued levels"):
        cluster_stats(np.array([1.0, 1.5, 2.0]), 10)
    with pytest.raises(ValueError, match="must be finite"):
        cluster_stats(np.array([1.0, np.nan]), 10)
    with pytest.raises(TypeError, match="integer-valued"):
        cluster_stats(np.array([True, False]), 10)
    with pytest.raises(TypeError, match="integer-valued"):
        cluster_stats(np.array(["1", "2"]), 10)


def test_cluster_stats_rejects_levels_outside_the_declared_universe() -> None:
    universe = np.arange(0, 10, dtype=np.int64)
    with pytest.raises(ValueError, match="outside the reachable universe"):
        cluster_stats(np.array([0, 1, 42], dtype=np.int64), 10, reachable_levels=universe)


def test_cluster_stats_rejects_an_empty_reachable_universe() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cluster_stats(
            np.array([1], dtype=np.int64), 10, reachable_levels=np.empty(0, dtype=np.int64)
        )


# ---------------------------------------------------------------------------
# levels_from_prices
# ---------------------------------------------------------------------------


def test_levels_from_prices_snaps_onto_the_tick_grid() -> None:
    prices = np.array([99.99, 100.0, 100.01, 100.05])
    out = levels_from_prices(prices, tick=0.01, ref_price=100.0)
    assert out.dtype == np.int64
    assert out.tolist() == [-1, 0, 1, 5]


def test_levels_from_prices_default_reference_is_zero() -> None:
    out = levels_from_prices(np.array([0.0, 0.02, -0.03]), tick=0.01)
    assert out.tolist() == [0, 2, -3]


def test_levels_from_prices_round_trips_the_simulator_price_map() -> None:
    """level → price → level must be the identity at the sim's own tick/ref."""
    cfg = santa_fe_config(seed=2)
    levels = np.arange(-25, 26, dtype=np.int64)
    prices = np.asarray([cfg.s0 + int(x) * cfg.tick for x in levels])
    assert levels_from_prices(prices, tick=cfg.tick, ref_price=cfg.s0).tolist() == levels.tolist()


def test_levels_from_prices_rejects_an_off_grid_stream() -> None:
    """A half-tick price is a contract violation, not a rounding opportunity."""
    with pytest.raises(ValueError, match="off the tick grid"):
        levels_from_prices(np.array([100.0, 100.005]), tick=0.01, ref_price=100.0)


def test_levels_from_prices_rejects_bad_arguments() -> None:
    for bad_tick in (0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tick must be positive and finite"):
            levels_from_prices(np.array([100.0]), tick=bad_tick)
    for bad_ref in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ref_price must be finite"):
            levels_from_prices(np.array([100.0]), tick=0.01, ref_price=bad_ref)
    with pytest.raises(ValueError, match="non-empty 1-D array"):
        levels_from_prices(np.empty(0), tick=0.01)
    with pytest.raises(ValueError, match="non-empty 1-D array"):
        levels_from_prices(np.zeros((2, 2)), tick=0.01)
    with pytest.raises(ValueError, match="prices must be finite"):
        levels_from_prices(np.array([100.0, np.inf]), tick=0.01)


# ---------------------------------------------------------------------------
# sim_cluster_stats — a small seeded ZI-LOB run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_run() -> dict[str, object]:
    return sim_cluster_stats(horizon=400.0, seed=3, m=10)


def test_sim_cluster_stats_reports_the_run_contract(sim_run: dict[str, object]) -> None:
    assert sim_run["label"] == "SYNTHETIC"
    assert sim_run["data_source"] == ZI_LOB_REVISION
    assert sim_run["seed"] == 3
    assert sim_run["horizon"] == pytest.approx(400.0)
    assert sim_run["m"] == 10
    assert sim_run["anchor"] == "touch"
    assert sim_run["band"] == 5
    assert int(sim_run["n_events"]) > 0  # type: ignore[arg-type]
    assert int(sim_run["n_submits"]) > 0  # type: ignore[arg-type]


def test_sim_cluster_stats_submit_attribution_is_consistent(sim_run: dict[str, object]) -> None:
    submits = sim_run["submit_levels"]
    reachable = sim_run["reachable_levels"]
    assert isinstance(submits, list) and isinstance(reachable, list)
    assert len(submits) == int(sim_run["n_submits"])  # type: ignore[arg-type]
    cluster = sim_run["cluster"]
    assert isinstance(cluster, dict)
    assert cluster["n_submits"] == len(submits)
    assert cluster["m"] == 10
    # Every submitted level must sit inside the reconstructed placement window.
    assert set(submits) <= set(reachable)
    assert cluster["n_reachable_levels"] == len(reachable)


def test_sim_cluster_stats_price_domain_cross_check_matches(sim_run: dict[str, object]) -> None:
    """The same statistic on real prices must give identical residue counts."""
    assert sim_run["price_domain_residues_match"] is True


def test_sim_cluster_stats_event_counts_conserve_orders(sim_run: dict[str, object]) -> None:
    """``event_counts`` is the simulator's conservation ledger: every dispatched
    event is an LO arrival, an MO arrival or a cancellation, and unit orders
    balance as created = fills + cancels + resting. The order-id-diff
    attribution in ``sim_cluster_stats`` must equal the created-order count,
    which is what makes the submit vector exact rather than approximate."""
    counts = sim_run["event_counts"]
    assert isinstance(counts, dict)
    assert counts["n_events"] == sim_run["n_events"]
    for key in (
        "n_lo_arrivals",
        "n_mo_arrivals",
        "n_fills",
        "n_cancellations",
        "n_orders_created",
        "resting",
    ):
        assert int(counts[key]) >= 0, key  # type: ignore[index]
    assert (
        int(counts["n_lo_arrivals"])  # type: ignore[index]
        + int(counts["n_mo_arrivals"])  # type: ignore[index]
        + int(counts["n_cancellations"])  # type: ignore[index]
        == int(counts["n_events"])  # type: ignore[index]
    )
    created = int(counts["n_orders_created"])  # type: ignore[index]
    assert created == (
        int(counts["n_fills"])  # type: ignore[index]
        + int(counts["n_cancellations"])  # type: ignore[index]
        + int(counts["resting"])  # type: ignore[index]
    )
    assert int(sim_run["n_submits"]) == created  # type: ignore[arg-type]


def test_sim_cluster_stats_is_seed_deterministic(sim_run: dict[str, object]) -> None:
    again = sim_cluster_stats(horizon=400.0, seed=3, m=10)
    assert again["submit_levels"] == sim_run["submit_levels"]
    assert again["reachable_levels"] == sim_run["reachable_levels"]
    assert again["n_events"] == sim_run["n_events"]
    other = sim_cluster_stats(horizon=400.0, seed=4, m=10)
    assert other["submit_levels"] != sim_run["submit_levels"]


def test_sim_cluster_stats_config_overrides_the_seed_argument() -> None:
    """When ``config`` is given its own seed governs, per the docstring."""
    run = sim_cluster_stats(horizon=250.0, seed=99, m=5, config=santa_fe_config(seed=9, band=3))
    assert run["seed"] == 9
    assert run["band"] == 3
    assert run["m"] == 5
    assert run["anchor"] == "touch"


def test_sim_cluster_stats_touch_arm_is_residue_neutral() -> None:
    """The documented touch-anchor result: no round-level preference, so the
    excess at residue 0 stays small even though the omnibus chi-square may fire
    on level *density* (placement is not uniform over the reachable hull)."""
    run = sim_cluster_stats(horizon=2000.0, seed=6, m=10)
    cluster = run["cluster"]
    assert isinstance(cluster, dict)
    assert abs(float(cluster["excess_round_share"])) < 0.05


def test_sim_cluster_stats_rejects_bad_arguments() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="horizon must be positive and finite"):
            sim_cluster_stats(horizon=bad, seed=0)
    with pytest.raises(ValueError, match="m must be >= 2"):
        sim_cluster_stats(horizon=50.0, seed=0, m=1)
    with pytest.raises(TypeError, match="config must be a ZILobConfig"):
        sim_cluster_stats(horizon=50.0, seed=0, config=object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# price_clustering_bench — sealed payload
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench() -> dict[str, object]:
    return price_clustering_bench(**_BENCH)


def test_bench_payload_carries_the_honesty_fields(bench: dict[str, object]) -> None:
    assert bench["schema"] == PRICE_CLUSTERING_SCHEMA
    assert bench["kind"] == PRICE_CLUSTERING_KIND
    assert bench["data_label"] == "SYNTHETIC"
    assert bench["label"] == "SYNTHETIC"
    assert bench["research_only"] is True
    assert bench["live_pnl_claim"] is False
    assert bench["simulated_only"] is True
    assert bench["claim"] == "synthetic_placement_diagnostic_only"
    assert bench["params"] == {"horizon": 250.0, "seed": 5, "m": 10, "band": 5}
    assert isinstance(bench["git_revision"], str)


def test_bench_arms_are_the_three_documented_flows(bench: dict[str, object]) -> None:
    arms = bench["arms"]
    assert isinstance(arms, dict)
    assert set(arms) == {"touch_zi", "markov_regime", "ref_band"}
    for name, block in arms.items():
        assert isinstance(block, dict), name
        for key in (
            "n_events",
            "n_submits",
            "n_reachable_levels",
            "residue_shares",
            "uniform_baseline_shares",
            "excess_round_share",
            "chi2_stat",
            "chi2_dof",
            "chi2_p_value",
            "residue0_reachable",
            "price_domain_residues_match",
        ):
            assert key in block, f"{name} missing {key}"
        assert block["price_domain_residues_match"] is True, name
        assert int(block["n_submits"]) > 0, name  # type: ignore[arg-type]


def test_bench_ref_anchor_arm_exercises_reachability(bench: dict[str, object]) -> None:
    """The frozen reference anchor leaves residue 0 unreachable, and the
    reachable-set normalisation must not manufacture a deficit there."""
    arms = bench["arms"]
    assert isinstance(arms, dict)
    ref = arms["ref_band"]
    assert isinstance(ref, dict)
    assert ref["residue0_reachable"] is False
    assert float(ref["excess_round_share"]) > -0.02  # type: ignore[arg-type]
    assert float(ref["uniform_baseline_shares"][0]) == pytest.approx(0.0)  # type: ignore[index]


def test_bench_detector_probe_recovers_the_planted_preference(
    bench: dict[str, object],
) -> None:
    """A measure that cannot flag a planted 2x round-level preference detects
    nothing — the probe is bound into the receipt for exactly that reason."""
    probe = bench["detector_probe"]
    assert isinstance(probe, dict)
    assert probe["planted_residue0_multiplier"] == pytest.approx(2.0)
    assert probe["detected"] is True
    assert float(probe["excess_round_share"]) > 0.0  # type: ignore[arg-type]
    p_value = probe["chi2_p_value"]
    assert p_value is not None and float(p_value) < 1e-6


def test_bench_seal_hash_is_reproducible_from_the_body(bench: dict[str, object]) -> None:
    digest = bench["receipt_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    body = {k: v for k, v in bench.items() if k != "receipt_sha256"}
    canonical = json.loads(canonical_json_bytes(body))
    assert hash_bytes(canonical_json_bytes(canonical)) == digest


def test_bench_payload_has_no_forbidden_headline_metric(bench: dict[str, object]) -> None:
    assert family_blob_forbidden_metrics_absent(bench)
    text = json.dumps(bench, sort_keys=True, default=str).lower()
    for token in FORBIDDEN_HEADLINE_TOKENS:
        assert token not in text, token


def test_bench_interpretation_is_labeled_synthetic(bench: dict[str, object]) -> None:
    text = bench["interpretation"]
    assert isinstance(text, str) and text
    assert "SYNTHETIC" in text
    assert "not market evidence" in text


def test_bench_is_deterministic_and_rejects_bad_arguments(
    bench: dict[str, object],
) -> None:
    assert price_clustering_bench(**_BENCH)["receipt_sha256"] == bench["receipt_sha256"]
    for bad_horizon in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="horizon must be positive and finite"):
            price_clustering_bench(horizon=bad_horizon, seed=1)
    for bad_band in (0, -1, True):
        with pytest.raises(ValueError, match="band must be an int >= 1"):
            price_clustering_bench(horizon=50.0, seed=1, band=bad_band)
    with pytest.raises(ValueError, match="m must be >= 2"):
        price_clustering_bench(horizon=50.0, seed=1, m=1)
    with pytest.raises(ValueError, match="seed must be an int"):
        price_clustering_bench(horizon=50.0, seed="5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# write_price_clustering_receipt — tmp_path only, never the repo receipts/ dir
# ---------------------------------------------------------------------------


def test_write_receipt_lands_in_the_given_directory(tmp_path: Path) -> None:
    path = write_price_clustering_receipt(tmp_path, **_BENCH)
    assert path == tmp_path / PRICE_CLUSTERING_RECEIPT
    assert path.is_file()
    # The test writes under tmp_path only — the repo receipts/ tree is untouched.
    assert tmp_path in path.parents


def test_write_receipt_creates_missing_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "dir"
    path = write_price_clustering_receipt(nested, **_BENCH)
    assert path.parent == nested
    assert path.is_file()


def test_write_receipt_accepts_a_string_directory(tmp_path: Path) -> None:
    path = write_price_clustering_receipt(str(tmp_path), **_BENCH)
    assert Path(path).is_file()
    assert Path(path).name == PRICE_CLUSTERING_RECEIPT


def test_receipt_fields_are_sealed_honest_and_hash_stable(tmp_path: Path) -> None:
    path = write_price_clustering_receipt(tmp_path, **_BENCH)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert isinstance(payload, dict)
    # Written with sort_keys=True — the seal convention the repo checks.
    assert list(payload) == sorted(payload)
    assert payload["schema"] == PRICE_CLUSTERING_SCHEMA
    assert payload["kind"] == PRICE_CLUSTERING_KIND
    assert payload["research_only"] is True
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["simulated_only"] is True
    assert payload["live_pnl_claim"] is False
    assert set(payload["arms"]) == {"touch_zi", "markov_regime", "ref_band"}
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.loads(canonical_json_bytes(body))
    assert hash_bytes(canonical_json_bytes(canonical)) == payload["receipt_sha256"]


def test_receipt_text_carries_no_forbidden_headline_token(tmp_path: Path) -> None:
    path = write_price_clustering_receipt(tmp_path, **_BENCH)
    text = path.read_text(encoding="utf-8").lower()
    for token in FORBIDDEN_HEADLINE_TOKENS:
        assert token not in text, token
    assert "synthetic" in text


def test_receipt_matches_the_in_memory_bench(tmp_path: Path) -> None:
    path = write_price_clustering_receipt(tmp_path, **_BENCH)
    written = json.loads(path.read_text(encoding="utf-8"))
    memory = price_clustering_bench(**_BENCH)
    assert written == json.loads(canonical_json_bytes(memory))
