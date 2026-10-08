"""Adversarial probes for clark_scarf echelon-inventory sim."""

from __future__ import annotations

from quant_fund.models.clark_scarf import _echelon_levels, bench_clark_scarf


def test_upstream_orders_fully_delivered() -> None:
    # Every upstream replenishment must arrive in full — the old code
    # silently dropped half of each order (`inv2 += q2 * 0.5`), starving
    # the downstream arm so the myopic policy's fill rate collapsed to
    # ~0.71 while the echelon arm reached ~0.98. With complete delivery
    # both arms see the same upstream availability and their fill rates
    # agree within a few points.
    out = bench_clark_scarf()
    assert abs(out["synthetic_myopic_fill"] - out["synthetic_cs_fill"]) < 0.05


def test_echelon_levels_monotone() -> None:
    s1, s2 = _echelon_levels(20.0, 0.5, 0.2, 5.0)
    assert s2 >= s1 >= 0.0


def test_bench_deterministic() -> None:
    assert bench_clark_scarf() == bench_clark_scarf()
