"""em_audit — pin the Baum–Welch contract empirically.

The EM guarantee: likelihood is non-decreasing along ``baum_welch``'s
history on a fixed sequence. ``log_space=True`` must produce the same
fixed point — it is the same M-step on a log trellis, not a different
algorithm. Same-seed restarts must be bitwise identical. A state never
visited by the posterior must keep its prior ``A`` row (the M-step has
no evidence for it — the code relies on this to not emit NaNs).

Sealed ``em_audit.v1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.hmm.discrete import DiscreteHMM, baum_welch
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["em_audit", "em_audit_bench"]

_TOL = 1e-10


def _sample_obs(rng: np.random.Generator, n: int, n_obs: int) -> list[int]:
    return [int(x) for x in rng.integers(0, n_obs, size=n)]


def em_audit() -> dict[str, Any]:
    results: dict[str, Any] = {}

    rng = np.random.default_rng(11)
    # Linear trellis underflows once EM sharpens the model (~n>=30 here);
    # the boundary is pinned explicitly by the underflow_boundary probe.
    obs = _sample_obs(rng, 20, 3)
    _, history = baum_welch(obs, n_states=2, n_obs=3, n_iter=15, seed=7)
    diffs = np.diff(np.asarray(history, dtype=float))
    results["monotone_linear"] = {
        "ok": bool(np.all(diffs > -_TOL)),
        "min_delta": float(diffs.min()),
        "n_iter": len(history),
    }

    _, log_history = baum_welch(obs, n_states=2, n_obs=3, n_iter=15, seed=7, log_space=True)
    # Both tracks target the same likelihood; log history is in log units
    # while linear history is a probability in (0,1].
    log_diffs = np.diff(np.asarray(log_history, dtype=float))
    results["monotone_log"] = {
        "ok": bool(np.all(log_diffs > -_TOL)),
        "min_delta": float(log_diffs.min()),
    }
    lin_final = float(history[-1])
    log_final = float(log_history[-1])
    results["log_linear_consistent"] = {
        "ok": bool(abs(np.log(lin_final) - log_final) < 1e-6),
        "linear_final": lin_final,
        "log_final": log_final,
    }

    _, h_a = baum_welch(obs, n_states=2, n_obs=3, n_iter=8, seed=42)
    _, h_b = baum_welch(obs, n_states=2, n_obs=3, n_iter=8, seed=42)
    results["seed_bitwise"] = {
        "ok": h_a == h_b,
        "n_iter": len(h_a),
    }

    # Dead-state contract: init a 3-state model on a 1-symbol tape — the
    # unreachable state's prior A row must be carried through untouched.
    dead_obs = [0] * 60
    model0 = DiscreteHMM(
        A=np.array([[0.9, 0.05, 0.05], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6]]),
        B=np.array([[1.0], [1.0], [1.0]]),
        pi=np.array([1.0, 0.0, 0.0]),
    )
    fitted, _ = baum_welch(dead_obs, n_states=3, n_obs=1, n_iter=5, model=model0)
    dead_rows_ok = bool(
        np.allclose(fitted.A[1:, :], model0.A[1:, :], atol=1e-12)
        and np.all(np.isfinite(fitted.A))
        and np.all(np.isfinite(fitted.B))
    )
    results["dead_state_row_preserved"] = {
        "ok": dead_rows_ok,
        "note": "states with zero posterior mass keep their prior A row",
    }

    # Underflow boundary: linear-space forward fails closed (ValueError, not
    # NaN posteriors) once the tape is long enough; log-space keeps training.
    from quant_fund.hmm.discrete import gamma_xi

    long_obs = _sample_obs(np.random.default_rng(13), 2000, 3)
    lin_outcome = "ok"
    try:
        gamma_xi(model0, np.asarray(long_obs, dtype=int))
    except ValueError:
        lin_outcome = "raise:ValueError"
    _, log_hist = baum_welch(
        long_obs,
        n_states=3,
        n_obs=3,
        n_iter=3,
        model=DiscreteHMM(
            A=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
            B=np.array(
                [[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]],
                dtype=float,
            ),
            pi=np.array([1.0 / 3.0] * 3),
        ),
        log_space=True,
    )
    results["underflow_boundary"] = {
        "linear": lin_outcome,
        "log_space_survives": bool(len(log_hist) == 4 and np.isfinite(log_hist).all()),
        "ok": lin_outcome == "raise:ValueError"
        and len(log_hist) == 4
        and bool(np.isfinite(log_hist).all()),
        "note": "linear trellis underflows on long tapes; log-space is the contract",
    }
    return results


def em_audit_bench() -> dict[str, Any]:
    results = em_audit()
    ok = all(v["ok"] for v in results.values())
    payload: dict[str, Any] = {
        "kind": "em_audit",
        "schema": "em_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": results, "ok": ok},
        "interpretation": (
            "Baum–Welch contract holds: EM likelihood non-decreasing in both "
            "spaces, log-space history is ln(linear) to 1e-6, same-seed "
            "restarts bitwise, dead states keep prior rows."
            if ok
            else f"EM CONTRACT VIOLATION: {results}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
