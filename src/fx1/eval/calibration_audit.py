"""calibration_audit — adversarial probes on the calibration eval.

Pinned contract:

- ``extract_probability``: last-number wins, ``%`` scales, out-of-range
  or unparseable → ``None`` (never raises).
- ``build_calibration_bank``: ``n_questions`` must be a positive multiple
  of 4; banks are seed-deterministic; every prompt passes
  ``validate_fx1_output`` at build; every truth sits inside
  ``(_P_MIN, _P_MAX)``.
- ``synthetic_oracle``: unknown mode refuses; unknown question ids answer
  ``""``; ``"true"`` echoes the closed form.
- ``run_calibration_eval``: the true oracle passes; an exploding model
  and a degenerate constant model both fail closed (NaN stats); bad
  ``n_bins`` refuses. Flagged surface: last-number semantics mean a
  hedged "0.4 … but maybe 0.5" scores the *second* number.

Sealed ``calibration_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import math
from typing import Any, cast

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["calibration_audit", "calibration_audit_bench"]


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def calibration_audit() -> dict[str, Any]:
    import os

    env_before = dict(os.environ)
    from fx1.eval.calibration_eval import (
        build_calibration_bank,
        extract_probability,
        run_calibration_eval,
        synthetic_oracle,
    )

    out: dict[str, Any] = {}
    out["extract_last_wins"] = extract_probability("0.4 but maybe 0.5") == 0.5
    out["extract_pct_scales"] = extract_probability("about 40%") == 0.4
    out["extract_oob_none"] = extract_probability("probability is 1.4") is None
    out["extract_unparseable_none"] = extract_probability("no numbers") is None
    out["extract_exp"] = extract_probability("2.5e-1") == 0.25

    out["bad_n_refuses"] = (
        _raises(lambda: build_calibration_bank(n_questions=6)) == "ValueError"
        and _raises(lambda: build_calibration_bank(n_questions=0)) == "ValueError"
    )
    b1 = build_calibration_bank(seed=3, n_questions=8)
    b2 = build_calibration_bank(seed=3, n_questions=8)
    b3 = build_calibration_bank(seed=4, n_questions=8)
    out["bank_deterministic"] = [q.prompt for q in b1] == [q.prompt for q in b2]
    out["seed_changes_bank"] = [q.prompt for q in b1] != [q.prompt for q in b3]
    out["truths_bounded"] = all(0.0 < q.true_probability < 1.0 for q in b1)
    out["prompts_carry_label"] = all("SYNTHETIC" in q.prompt for q in b1)

    out["bad_mode_refuses"] = (
        _raises(lambda: synthetic_oracle(cast("Any", "bogus"))) == "ValueError"
    )
    oracle = synthetic_oracle("true", seed=3)
    q = b1[0]
    out["oracle_true_exact"] = (
        abs(float(oracle([{"role": "user", "content": q.prompt}])) - q.true_probability) <= 0.0005
    )
    out["oracle_unknown_empty"] = oracle([{"role": "user", "content": "[question_id: nope]"}]) == ""

    rep_true = run_calibration_eval(oracle, seed=3)
    out["true_oracle_passes"] = rep_true.passed is True and rep_true.n_unparseable == 0

    def explode(m: Any) -> str:
        raise RuntimeError("boom")

    rep_boom = run_calibration_eval(explode, seed=3)
    out["exploding_fails_closed"] = (
        rep_boom.passed is False
        and rep_boom.n_unparseable == rep_boom.n_questions
        and math.isnan(rep_boom.ece)
    )
    rep_const = run_calibration_eval(lambda m: "0.5", seed=3)
    out["constant_fails_closed"] = rep_const.passed is False
    out["bad_bins_refuses"] = (
        _raises(lambda: run_calibration_eval(oracle, seed=3, n_bins=0)) == "ValueError"
    )
    # BLAS/numexpr init inside the eval import mutates os.environ — restore
    os.environ.clear()
    os.environ.update(env_before)
    return out


def calibration_audit_bench() -> dict[str, Any]:
    r = calibration_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "calibration_audit",
        "schema": "calibration_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Calibration eval contract holds: extraction is last-number "
            "with %-scaling and fail-soft; the bank is seed-deterministic "
            "with labeled prompts and bounded truths; the true oracle "
            "passes while exploding and constant models fail closed."
            if ok
            else f"CALIBRATION AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
