"""Tests for repost_frontier_bench."""

from quant_fund.microstructure.repost_frontier_bench import (
    repost_frontier_bench,
)


def test_stats_shape() -> None:
    out = repost_frontier_bench(tape_dir=None, horizon=8000, seed=7)
    assert out["schema"] == "repost_frontier.v1"
    assert out["data_label"] == "MIXED"
    assert out["tape"] is None
    assert len(out["cells"]) == 10
    assert set(out["claims"]) == {
        "frontier_cells_measured",
        "composable_cell_exists",
        "fill_trigger_composes_too",
        "delay_sets_latency",
        "joint_undeerseeds",
        "tape_reseeds_majority",
        "tape_reseed_returns_to_touch",
    }
    for c in out["cells"]:
        assert c["n_emptied"] >= 0
        assert isinstance(c["deltas"], dict)
    assert len(out["receipt_sha256"]) == 64


def test_repost_cell_params_recorded() -> None:
    out = repost_frontier_bench(tape_dir=None, horizon=4000, seed=3)
    by_name = {c["regime"]: c for c in out["cells"]}
    assert by_name["joint"]["deltas"] == {}
    assert by_name["arr_f60_b3"]["deltas"]["repost_band"] == 3
    assert by_name["evt_f55_m320"]["deltas"]["fill_repost_delay"] == 320
    assert by_name["evt_f55_m160_d8"]["deltas"]["repost_depth"] == 8
