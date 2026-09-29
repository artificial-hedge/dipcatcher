"""Promotion-time bench: fixed Kelly plug-in vs online lambda selection.

Compares ``LossEProcess`` under ``lambda_policy="kelly"`` (running
win-rate plug-in, memory ~ 1/n) and ``lambda_policy="online"``
(GRAPA-style exponential-gradient ascent, constant-step tracker) on
three synthetic differential streams:

- ``iid``: iid Gaussian diffs with a fixed challenger edge.
- ``ar1``: AR(1)-correlated diffs with the same unconditional edge —
  wins and losses cluster, so the conditional edge oscillates around
  the flat average the Kelly plug-in tracks.
- ``regime``: abrupt regime change — incumbent-better edge for the
  first ``REGIME_SPLIT`` origins, then challenger-better edge. The
  flat-average bet must unwind the first regime before it can bet.

Streams are SYNTHETIC correctness benchmarks, not market evidence.
Prints a promotion-time table (median crossing origin and promote
rate over ``N_SIMS`` seeds) plus one logged e-value trajectory per
policy for the fixed probe seed.

Run: ``PYTHONPATH=src python scripts/bench_lambda_policies.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quant_fund.research.evalues import LossEProcess  # noqa: E402

N_SIMS = 400
N_ORIGINS = 400
ALPHA = 0.05
EDGE = -0.008  # negative diff = challenger better
SCALE = 0.02
RHO = 0.6
REGIME_SPLIT = 120


def stream_iid(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(EDGE, SCALE, n)


def stream_ar1(rng: np.random.Generator, n: int) -> np.ndarray:
    # Stationary AR(1): same unconditional mean/sd as the iid stream.
    eps = rng.normal(0.0, SCALE * np.sqrt(1.0 - RHO * RHO), n)
    d = np.empty(n)
    d[0] = EDGE / (1.0 - RHO) + eps[0]
    for i in range(1, n):
        d[i] = EDGE + RHO * (d[i - 1] - EDGE / (1.0 - RHO)) + eps[i]
    return d


def stream_regime(rng: np.random.Generator, n: int) -> np.ndarray:
    # Incumbent-better edge until REGIME_SPLIT, then challenger-better.
    d = rng.normal(-EDGE, SCALE, n)
    d[REGIME_SPLIT:] = rng.normal(EDGE, SCALE, n - REGIME_SPLIT)
    return d


STREAMS = {"iid": stream_iid, "ar1": stream_ar1, "regime": stream_regime}
POLICIES = ("kelly", "online")


def run(make_stream, policy: str, n_sims: int = N_SIMS) -> dict[str, float]:
    crossings: list[int] = []
    for seed in range(n_sims):
        rng = np.random.default_rng(seed * 7919 + 17)
        proc = LossEProcess(alpha=ALPHA, lambda_policy=policy)
        for di in make_stream(rng, N_ORIGINS):
            proc.update(float(di), 0.0)
        if proc.promotion_origin is not None:
            crossings.append(proc.promotion_origin)
    arr = np.asarray(crossings, dtype=float)
    return {
        "promoted": len(crossings) / n_sims,
        "median_cross": float(np.median(arr)) if arr.size else float("nan"),
        "mean_cross": float(np.mean(arr)) if arr.size else float("nan"),
    }


def log_trajectory(make_stream, policy: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    proc = LossEProcess(alpha=ALPHA, lambda_policy=policy)
    for di in make_stream(rng, N_ORIGINS):
        proc.update(float(di), 0.0)
    states = proc.states
    idx = np.linspace(0, len(states) - 1, 12).astype(int)
    traj = ", ".join(f"{i}:{states[i].evalue:.3g}" for i in idx)
    lam = ", ".join(f"{i}:{states[i].lam:.2f}" for i in idx)
    print(f"  {policy:>6}  e[{traj}]")
    print(f"  {policy:>6}  λ[{lam}]")


def main() -> None:
    print(f"n_sims={N_SIMS}  n_origins={N_ORIGINS}  alpha={ALPHA}  lam_max=0.5")
    print(f"edge={EDGE}  scale={SCALE}  rho={RHO}  regime_split={REGIME_SPLIT}")
    header = f"{'stream':<8}{'policy':<8}{'promote%':>9}{'median_x':>10}{'mean_x':>9}"
    print(header)
    print("-" * len(header))
    for name, make_stream in STREAMS.items():
        for policy in POLICIES:
            r = run(make_stream, policy)
            print(
                f"{name:<8}{policy:<8}{r['promoted']:>9.3f}"
                f"{r['median_cross']:>10.1f}{r['mean_cross']:>9.1f}"
            )
        print()
    print("e-value trajectories (origin:evalue, sampled), seed=0:")
    for name, make_stream in STREAMS.items():
        print(f" {name}:")
        for policy in POLICIES:
            log_trajectory(make_stream, policy)


if __name__ == "__main__":
    main()
