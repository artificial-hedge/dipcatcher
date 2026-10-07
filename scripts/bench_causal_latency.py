#!/usr/bin/env python
"""Wall-clock latency rebenchmark of the causal-forecast conformal/Student-t path.

`docs/TOP10_EXECUTION_PLAN.md` task 8 asks for a fresh WALL-CLOCK latency
rebenchmark of the causal forecast path on a seeded workload, with the
conformal Student-t fit suspected as the dominant cost. This script times the
dominant building blocks — the Student-t fit (``metrics.fit_student_t``), the
Student-t CRPS (``metrics.crps_student_t``, which evaluates ``student_t.cdf`` /
``student_t.pdf``), and the split-conformal quantile
(``metrics.conformal.conformal_quantile``) — on a seeded SYNTHETIC workload and
prints an honest latency table.

SYNTHETIC workload only (a correctness/performance measurement, never market
evidence). ``live_pnl_claim=False``; no Sharpe/P&L/NAV is computed.

Run::

    uv run python scripts/bench_causal_latency.py --reps 200 --n 5000

Scope note: this measures the conformal/Student-t building blocks that dominate
the causal-forecast latency. The end-to-end ``pipeline.forecast`` causal path
and any optimization FIX live in ``src/quant_fund/metrics/**`` /
``src/quant_fund/pipeline/**``, which this lane does not own; a fix there must
be routed to the metrics owner (see the note in ``docs/PERF.md``).
"""

from __future__ import annotations

import argparse
import time
from statistics import median

import numpy as np

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.risk_parametric import fit_student_t
from quant_fund.metrics.scoring import crps_student_t


def _workload(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.standard_t(df=5, size=n) * 0.01
    mu = rng.normal(0.0, 0.001, size=n)
    sigma = np.abs(rng.normal(0.01, 0.002, size=n)) + 1e-6
    return {"y": y, "mu": mu, "sigma": sigma, "losses": np.abs(y)}


def _time(fn, reps: int) -> list[float]:
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _row(name: str, samples: list[float], calls: int) -> str:
    mean = sum(samples) / len(samples)
    return f"| {name} | {calls} | {mean:.4f} | {median(samples):.4f} | {max(samples):.4f} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()

    w = _workload(args.n, args.seed)
    y, mu, sigma, losses = w["y"], w["mu"], w["sigma"], w["losses"]
    nu = 5.0

    def _fit() -> None:
        fit_student_t(losses)

    def _crps() -> None:
        crps_student_t(y, mu, sigma, nu)

    def _conf() -> None:
        conformal_quantile(np.abs(y - mu), 0.1)

    def _conformal_student_t_interval() -> None:
        fit = fit_student_t(losses)
        crps_student_t(y, mu, sigma, fit.get("nu", nu))
        conformal_quantile(np.abs(y - mu), 0.1)

    rows = [
        _row("fit_student_t (Student-t fit)", _time(_fit, args.reps), args.n),
        _row("crps_student_t (student_t.cdf/pdf)", _time(_crps, args.reps), args.n),
        _row("conformal_quantile (split conformal)", _time(_conf, args.reps), args.n),
        _row(
            "conformal+Student-t interval (composed)",
            _time(_conformal_student_t_interval, args.reps),
            args.n,
        ),
    ]

    print(f"# Causal-forecast conformal/Student-t latency (n={args.n}, reps={args.reps})")
    print()
    print("SYNTHETIC seeded workload; correctness/performance measurement only.")
    print("`live_pnl_claim=false`. No Sharpe/P&L/NAV computed.")
    print()
    print("| Stage | n per call | mean ms | median ms | max ms |")
    print("|---|---:|---:|---:|---:|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
