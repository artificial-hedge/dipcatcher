"""engine_fuzz — randomized invariant testing of the ZI matching engine.

Drives ``ZILobSimulator`` with random legal operations — limit submissions
inside the band, cancels of live and dead order ids, market orders, exogenous
clock steps — and after EVERY step asserts the book's invariants:

- uncrossed book: ``best_bid_level < best_ask_level`` when both exist;
- registry consistency: every order id lives in exactly one deque position at
  its recorded (side, level); every deque member is registered;
- fail-closed edges: cancel of a missing id returns ``False`` (never raises);
  a marketable limit submission raises ``ValueError``; ``qty < 1`` market
  orders raise.

Any violation means the matching engine's state machine is unsound — the
exact class of bug a stress test must catch before any calibration claim on
the sim matters. Verdict ``ok`` iff zero violations across all seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.microstructure.zi_lob_simulator import (
    Side,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["check_book_invariants", "engine_fuzz", "engine_fuzz_bench"]


def check_book_invariants(sim: ZILobSimulator) -> list[str]:
    """Uncrossed book + registry↔deque consistency. ``[]`` means sound."""
    errors: list[str] = []
    bids = sim._bids  # noqa: SLF001
    asks = sim._asks  # noqa: SLF001
    orders = sim._orders  # noqa: SLF001
    bb, ba = sim.best_bid_level, sim.best_ask_level
    if bb is not None and ba is not None and bb >= ba:
        errors.append(f"crossed_book:{bb}>={ba}")
    seen: set[int] = set()
    for side, book in (("buy", bids), ("sell", asks)):
        for level, dq in book.items():
            if not dq:
                errors.append(f"empty_level:{side}:{level}")
            for oid in dq:
                if oid in seen:
                    errors.append(f"dup_order:{oid}")
                seen.add(oid)
                order = orders.get(oid)
                if order is None:
                    errors.append(f"orphan_deque_entry:{oid}")
                elif order.side != side or order.level != level:
                    errors.append(f"misplaced:{oid}:recorded({order.side},{order.level})")
    for oid in orders:
        if oid not in seen:
            errors.append(f"unbooked_order:{oid}")
    return errors


def engine_fuzz(
    seed: int = 0, n_steps: int = 500, config: ZILobConfig | None = None
) -> dict[str, Any]:
    """Random legal ops; per-step invariants; fail-closed edge probes."""
    rng = np.random.default_rng(seed)
    sim = ZILobSimulator(config or santa_fe_config())
    violations: list[str] = []
    counters = {
        "submits": 0,
        "cancels_live": 0,
        "cancels_dead": 0,
        "market_orders": 0,
        "steps": 0,
        "rejections_marketable": 0,
        "rejections_qty": 0,
    }
    t_prev = sim.t
    for i in range(n_steps):
        op = int(rng.integers(0, 5))
        try:
            if op == 0:
                side: Side = "buy" if rng.random() < 0.5 else "sell"
                anchor = sim.best_ask_level or sim.best_bid_level or sim._ref_level  # noqa: SLF001
                dist = int(rng.integers(1, sim.cfg.band + 1))
                level = anchor - dist if side == "buy" else anchor + dist
                sim.submit_limit_order(side, sim.level_to_price(level))
                counters["submits"] += 1
            elif op == 1:
                live = list(sim._orders)  # noqa: SLF001
                if live:
                    oid = int(live[int(rng.integers(0, len(live)))])
                    if sim.cancel_order(oid):
                        counters["cancels_live"] += 1
                    else:
                        violations.append(f"step{i}: live cancel returned False:{oid}")
                else:
                    if sim.cancel_order(int(rng.integers(0, 10_000))):
                        violations.append(f"step{i}: cancel of absent id returned True")
                    else:
                        counters["cancels_dead"] += 1
            elif op == 2:
                qty = int(rng.integers(1, 4))
                side2: Side = "buy" if rng.random() < 0.5 else "sell"
                sim.inject_market_order(side2, qty)
                counters["market_orders"] += 1
            elif op == 3:
                sim.step()
                counters["steps"] += 1
            else:
                # Edge probes: marketable submit must raise, qty=0 must raise.
                ba = sim.best_ask_level
                if ba is not None:
                    try:
                        sim.submit_limit_order("buy", sim.level_to_price(ba))
                        violations.append(f"step{i}: marketable buy accepted at ask {ba}")
                    except ValueError:
                        counters["rejections_marketable"] += 1
                try:
                    sim.inject_market_order("buy", 0)
                    violations.append(f"step{i}: qty=0 market order accepted")
                except ValueError:
                    counters["rejections_qty"] += 1
        except (ValueError, RuntimeError) as exc:
            # Only the two documented rejections may raise inside the loop.
            if isinstance(exc, RuntimeError):
                violations.append(f"step{i}: RuntimeError {exc}")
        errs = check_book_invariants(sim)
        violations.extend(f"step{i}:{e}" for e in errs)
        if sim.t < t_prev:
            violations.append(f"step{i}: clock moved backward {t_prev}->{sim.t}")
        t_prev = sim.t
    return {
        "seed": seed,
        "n_steps": n_steps,
        "violations": violations[:50],
        "n_violations": len(violations),
        "counters": counters,
        "final_depth": sim.total_depth,
        "n_fills": sim.n_fills,
    }


def engine_fuzz_bench(seeds: tuple[int, ...] = (0, 1, 2, 3), n_steps: int = 500) -> dict[str, Any]:
    """Seeded fuzz arms; ``ok`` iff no invariant violations anywhere."""
    runs = [engine_fuzz(seed, n_steps) for seed in seeds]
    total_viol = sum(r["n_violations"] for r in runs)
    ok = total_viol == 0
    payload: dict[str, Any] = {
        "kind": "engine_fuzz",
        "schema": "engine_fuzz.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "book uncrossed + registry consistent + edges fail closed, every step",
            "seeds": list(seeds),
            "steps_per_seed": n_steps,
            "total_violations": total_viol,
            "ok": ok,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "violations": r["n_violations"],
                    "fills": r["n_fills"],
                    "depth": r["final_depth"],
                    "counters": r["counters"],
                }
                for r in runs
            ],
            "first_violations": [v for r in runs for v in r["violations"]][:10],
        },
        "interpretation": (
            f"{len(runs)} seeds × {n_steps} ops: {total_viol} invariant violations. "
            + (
                "Engine state machine sound under random legal ops."
                if ok
                else "INVARIANT VIOLATION — matching engine unsound; see first_violations"
            )
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
