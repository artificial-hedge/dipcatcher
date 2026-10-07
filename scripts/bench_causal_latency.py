#!/usr/bin/env python3
"""SYNTHETIC latency rebenchmark for the causal-forecast scoring path (task 8).

Times the causal forecast -> scoring primitives that sit on the decision path:
Student-t fit, CRPS scoring, and the conformal quantile, plus their composed
interval. Seeded SYNTHETIC inputs only: this measures compute cost, never market
evidence and never a live-trading or profitability claim (live_pnl_claim=False).

Reproduce:

    uv run python scripts/bench_causal_latency.py --reps 200 --n 5000

Prints a markdown latency table (mean/median/max over ``--reps`` repetitions,
milliseconds per call at ``--n`` rows).
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.risk_parametric import fit_student_t
from quant_fund.metrics.scoring import crps_student_t


def _workload(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded SYNTHETIC target/forecast/residual arrays (deterministic)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_t(df=5, size=n)  # synthetic heavy-tailed target
    mu = rng.normal(0.0, 0.1, size=n)  # synthetic location forecast
    sigma = np.abs(rng.normal(1.0, 0.05, size=n)) + 1e-3  # synthetic scale forecast
    losses = np.abs(y - mu)  # synthetic conformal residual scores
    return y, mu, sigma, losses


def _time(fn: object, reps: int) -> list[float]:
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()  # type: ignore[operator]
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _row(name: str, n: int, samples: list[float]) -> str:
    mean = statistics.fmean(samples)
    med = statistics.median(samples)
    mx = max(samples)
    return f"| {name} | {n} | {mean:.2f} | {med:.2f} | {mx:.2f} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()

    y, mu, sigma, losses = _workload(args.n, args.seed)

    def _composed() -> None:
        fit = fit_student_t(losses)
        conformal_quantile(losses, 0.1)
        crps_student_t(y, mu, sigma, nu=fit.get("nu", 5.0))

    rows = [
        _row(
            "`fit_student_t` (Student-t fit)",
            args.n,
            _time(lambda: fit_student_t(losses), args.reps),
        ),
        _row(
            "`crps_student_t` (CRPS score)",
            args.n,
            _time(lambda: crps_student_t(y, mu, sigma, nu=5.0), args.reps),
        ),
        _row(
            "`conformal_quantile`",
            args.n,
            _time(lambda: conformal_quantile(losses, 0.1), args.reps),
        ),
        _row("composed interval", args.n, _time(_composed, args.reps)),
    ]

    print(f"# Causal-forecast latency (n={args.n}, reps={args.reps}, seed={args.seed})")
    print()
    print("| call | n | mean ms | median ms | max ms |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(r)
    print()
    print("SYNTHETIC workload only; not market evidence. live_pnl_claim=False.")


if __name__ == "__main__":
    main()
