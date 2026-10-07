"""stack_watch — anytime-valid audit of the fusion claim itself.

``quantile_stack`` reports mean OOF pinball per tau — a fixed-sample
verdict. The live question is different: as origins stream in, *is the
stack actually beating what it's made of?* ``stack_watch`` feeds the
per-origin pinball diff (stack vs the per-origin best member — the
oracle baseline, so promotion means beating *whichever member was right
that day*) into a :class:`LossEProcess`, plus one process per member.

Claims per stream: ``promoted`` (anytime-valid, Ville-guaranteed),
``evalue``, ``anytime_p``, ``win_share``. The oracle baseline is the
strongest honest comparator: promoting against it is rare; a stack that
merely averages members will lose to it. That makes ``stack_vs_oracle``
the discriminating claim — promoting there means the fusion is
*learning combination structure*, not just diversifying.

Sealed ``stack_watch.v1`` receipts; SYNTHETIC demos only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from quant_fund.fusion.quantile_stack import QuantileStackResult, _pinball
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = npt.NDArray[np.float64]


def _per_origin_pinball(oof: Array, member_q: Array, y: Array, taus: Array) -> tuple[Array, Array]:
    """Per-origin mean-pinball for the stack and each member.

    ``oof``: (n, k) stacked quantiles; ``member_q``: (n, m, k); ``y``: (n,).
    Returns stack losses (n,) and member losses (n, m).
    """
    n, m, k = member_q.shape
    stack_l = np.full(n, np.nan)
    memb_l = np.full((n, m), np.nan)
    for i in range(n):
        if not np.isfinite(oof[i]).all():
            continue
        stack_l[i] = float(
            np.mean(
                [_pinball(float(t), np.asarray([y[i] - oof[i, j]]))[0] for j, t in enumerate(taus)]
            )
        )
        for mi in range(m):
            memb_l[i, mi] = float(
                np.mean(
                    [
                        _pinball(float(t), np.asarray([y[i] - member_q[i, mi, j]]))[0]
                        for j, t in enumerate(taus)
                    ]
                )
            )
    return stack_l, memb_l


def stack_watch(
    result: QuantileStackResult,
    member_quantiles: Array,
    target: Array,
    *,
    alpha: float = 0.05,
    lam: float = 0.5,
) -> dict[str, Any]:
    """Stream the OOF rows through per-member + oracle e-processes."""
    from quant_fund.research.evalues import LossEProcess, promotion_report

    y = np.asarray(target, dtype=float).reshape(-1)
    q = np.asarray(member_quantiles, dtype=float)
    elig = result.fold_ids >= 0
    if np.count_nonzero(elig) < 2:
        raise ValueError("need >= 2 eligible OOF rows to audit")
    idx = np.where(elig)[0]
    stack_l, memb_l = _per_origin_pinball(result.oof_quantiles, q, y, result.taus)
    stack_l, memb_l = stack_l[idx], memb_l[idx]
    if not (np.isfinite(stack_l).all() and np.isfinite(memb_l).all()):
        raise ValueError("non-finite OOF losses")

    n, m = memb_l.shape
    oracle_l = memb_l.min(axis=1)
    oracle_proc = LossEProcess(alpha=alpha, lam=lam)
    for i in range(n):
        oracle_proc.update(stack_l[i], oracle_l[i])
    oracle_rep = promotion_report(
        stack_l, oracle_l, alpha=alpha, lam=lam, challenger="stack", incumbent="oracle_best_member"
    )

    per_member: dict[str, Any] = {}
    for mi in range(m):
        proc = LossEProcess(alpha=alpha, lam=lam)
        for i in range(n):
            proc.update(stack_l[i], memb_l[i, mi])
        per_member[f"member_{mi}"] = {
            "promoted": proc.promotion_origin is not None,
            "promotion_origin": proc.promotion_origin,
            "final_evalue": proc.states[-1].evalue,
            "win_share": float(np.mean(stack_l < memb_l[:, mi])),
        }

    payload: dict[str, Any] = {
        "stack_vs_oracle": {
            "promoted": oracle_rep["promoted"],
            "promotion_origin": oracle_rep.get("promotion_origin"),
            "final_evalue": oracle_rep["final_evalue"],
            "anytime_p": oracle_rep["anytime_p"],
            "win_share": float(np.mean(stack_l < oracle_l)),
        },
        "per_member": per_member,
        "n_origins": int(n),
        "n_members": int(m),
        "stack_mean_pinball": float(stack_l.mean()),
        "oracle_mean_pinball": float(oracle_l.mean()),
        "alpha": alpha,
    }
    return payload


def stack_watch_demo(seed: int = 0, n: int = 400) -> dict[str, Any]:
    """Synthetic demo: complementary members -> stack should beat the oracle."""
    from quant_fund.fusion.quantile_stack import cross_fitted_quantile_stack

    rng = np.random.default_rng(seed)
    taus = np.array([0.1, 0.5, 0.9])
    y = rng.normal(0, 0.5, n)
    # each member sees the true quantile grid plus independent per-row
    # noise — the stack's average halves the noise, so it should beat
    # EACH member on nearly every origin while the per-origin oracle
    # (whichever member happened to be closer) stays hard to beat
    q = np.zeros((n, 2, taus.size))
    for j, t in enumerate(taus):
        z = float(np.quantile(rng.standard_normal(5000), t))
        q[:, 0, j] = 0.5 * z + 0.25 * rng.normal(0, 1, n)
        q[:, 1, j] = 0.5 * z + 0.25 * rng.normal(0, 1, n)
    idx = np.arange(n)
    folds = [
        (idx[::2], idx[1::2]),
        (idx[1::2], idx[::2]),
    ]
    res = cross_fitted_quantile_stack(q, y, taus, folds, alpha=1.0)
    claim = stack_watch(res, q, y)
    payload: dict[str, Any] = {
        "kind": "stack_watch",
        "schema": "stack_watch.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": claim,
        "interpretation": {
            "promoted_vs_oracle": "stack beats the per-origin best member — fusion learns combination structure",
            "per_member": "win share and e-process vs each constituent",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["stack_watch", "stack_watch_demo"]
