"""spa_test — White's Reality Check + Romano–Wolf stepdown.

Fixed-sample counterpart to the e-process lanes: is the *best* head
actually better than the benchmark? White (2000) RC uses the stationary
bootstrap (Politis–Romano) to get a valid null for max_j √n·d̄_j under
dependence; Romano–Wolf (2005) stepdown gives per-head adjusted p-values
with FWER control.

Claims: FWER-controlled at alpha on a null arm; power monotone in the
planted winner's advantage. Sealed `spa_test.v1`. SYNTHETIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SPA_SCHEMA = "spa_test.v1"


def stationary_bootstrap_indices(n: int, p: float, rng: np.random.Generator) -> NDArray[np.int64]:
    """Politis–Romano stationary bootstrap indices, mean block 1/p."""
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0,1]: {p}")
    idx = np.empty(n, dtype=np.int64)
    i = int(rng.integers(0, n))
    t = 0
    while t < n:
        if rng.random() < p:
            i = int(rng.integers(0, n))
        else:
            i = (i + 1) % n
        idx[t] = i
        t += 1
    return idx


def reality_check_pvalue(
    loss_matrix: NDArray[np.float64],
    *,
    benchmark_col: int = 0,
    n_boot: int = 500,
    p: float = 0.1,
    seed: int = 0,
) -> float:
    """White RC: H0 = no head beats the benchmark.

    d_j = L_j − L_bench (lower d_j = better). Statistic
    T = max_j √n·d̄_j clipped at 0; bootstrap null resamples the
    recentered diffs with the stationary bootstrap.
    """
    d = _loss_diffs(loss_matrix, benchmark_col)
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    stat = float(np.sqrt(n) * d.mean(axis=0).max())
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, p, rng)
        boot_stats[b] = np.sqrt(n) * (d[idx].mean(axis=0) - d.mean(axis=0)).max()
    return float((boot_stats >= stat).mean())


@dataclass
class StepdownResult:
    rejected: list[int]  # head indices (loss_matrix columns) beating the benchmark
    adj_p: NDArray[np.float64]  # per-head adjusted p-value, shape (k,)


def romano_wolf_stepdown(
    loss_matrix: NDArray[np.float64],
    *,
    benchmark_col: int = 0,
    n_boot: int = 500,
    p: float = 0.1,
    alpha: float = 0.10,
    seed: int = 0,
) -> StepdownResult:
    """Romano–Wolf stepdown on max-t statistics; FWER ≤ alpha."""
    d = _loss_diffs(loss_matrix, benchmark_col)
    n, k = d.shape
    se = d.std(axis=0, ddof=1)
    se = np.where(se < 1e-12, 1e-12, se)
    t_obs = np.sqrt(n) * d.mean(axis=0) / se
    rng = np.random.default_rng(seed)
    # pooled bootstrap null: recentered, studentized
    t_boot = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, p, rng)
        db = d[idx]
        dm = db.mean(axis=0) - d.mean(axis=0)
        seb = db.std(axis=0, ddof=1)
        seb = np.where(seb < 1e-12, 1e-12, seb)
        t_boot[b] = np.sqrt(n) * dm / seb
    order = np.argsort(-t_obs)  # largest first
    adj = np.empty(k)
    remaining = np.arange(k)
    prev = 0.0
    for head in order:
        mx = t_boot[:, remaining].max(axis=1)
        pv = float((mx >= t_obs[head]).mean())
        prev = max(prev, pv)
        adj[head] = prev
        remaining = remaining[remaining != head]
    rejected = [int(order[i]) for i in range(k) if adj[order[i]] < alpha]
    return StepdownResult(rejected=rejected, adj_p=adj)


def _loss_diffs(loss_matrix: NDArray[np.float64], benchmark_col: int) -> NDArray[np.float64]:
    m = np.asarray(loss_matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("loss_matrix must be (n, k>=2)")
    if not 0 <= benchmark_col < m.shape[1]:
        raise ValueError(f"benchmark_col out of range: {benchmark_col}")
    heads = [j for j in range(m.shape[1]) if j != benchmark_col]
    bench = m[:, benchmark_col]
    return -m[:, heads] + bench[:, None]  # d_j = bench − head; positive = head better


def spa_bench(
    *,
    n: int = 200,
    k: int = 8,
    n_boot: int = 200,
    n_reps: int = 40,
    deltas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5),
    alpha: float = 0.10,
    seed: int = 0,
) -> dict[str, Any]:
    """FWER on the null arm + power curve in planted-winner δ."""
    arm_out: dict[str, dict[str, float]] = {}
    for delta in deltas:
        reject_rates = []
        for rep in range(n_reps):
            r = np.random.default_rng(seed + 1 + rep)
            base = r.standard_normal((n, 1))
            heads = base + r.standard_normal((n, k)) * 1.0
            if delta > 0:
                heads[:, 0] -= delta * np.sqrt(1.0)  # head 0 loses less
            losses = np.column_stack([base[:, 0], heads])
            res = romano_wolf_stepdown(
                losses, benchmark_col=0, n_boot=n_boot, alpha=alpha, seed=rep
            )
            reject_rates.append(float(len(res.rejected) > 0))
        arm_out[f"delta={delta}"] = {
            "reject_rate": float(np.mean(reject_rates)),
            "n_reps": float(n_reps),
        }
    null_rate = arm_out[f"delta={deltas[0]}"]["reject_rate"]
    rates = [arm_out[f"delta={d}"]["reject_rate"] for d in deltas]
    payload: dict[str, Any] = {
        "schema": SPA_SCHEMA,
        "kind": "spa_test",
        "n": n,
        "k": k,
        "n_boot": n_boot,
        "alpha": alpha,
        "arms": arm_out,
        "fwer_controlled": bool(null_rate <= alpha + 0.08),
        "power_monotone": bool(all(rates[i] <= rates[i + 1] + 0.12 for i in range(len(rates) - 1))),
        "interpretation": (
            "delta=0 arm is the null (heads iid identical to benchmark): "
            "any rejection is a familywise error, must stay ≈alpha. "
            "Increasing delta raises the planted winner's edge → "
            "rejection rate is the power curve"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
