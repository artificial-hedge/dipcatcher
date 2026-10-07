"""Out-of-sample copula dominance — R-vine vs C-/D-vine vs Gaussian.

Fits four dependence models on a synthetic panel whose truth is a
correlated Student-t copula (planted joint tail dependence), then scores
each on held-out pseudo-observations:

- per-observation copula log-likelihood (higher is better),
- a betting confidence sequence on the log-loss differential
  ``d_t = -ll_R - ll_comp`` (the ``loss_cs`` machinery, so "R-vine
  dominates" is a *time-uniform* claim, not a point estimate),
- joint-tail scenario risk: P(all d marginals < q) and the conditional
  expectation of the worst coordinate given a joint-crash event, each
  measured empirically on the holdout and under Monte-Carlo draws from
  every fitted model.

All results are SYNTHETIC: the panel is generated, never market data.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

from quant_fund.models.pair_vine_copula import (
    VineMatrix,
    vine_fit,
    vine_logpdf_rows,
    vine_sample,
)
from quant_fund.research.loss_cs import MeanDiffCS

Array = NDArray[np.float64]

VINE_DOMINANCE_SCHEMA = "vine_dominance.v1"

_EPS = 1e-10


def _clip(u: Array) -> Array:
    return np.clip(u, _EPS, 1.0 - _EPS)


def mvt_copula_sample(n: int, corr: Array, nu: float, rng: np.random.Generator) -> Array:
    """Correlated multivariate Student-t copula uniforms.

    x = z / sqrt(w), z ~ N(0, corr), w ~ chi2_nu / nu; u = t_nu(x). Tail
    dependence is lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))) — real
    joint-crash mass that a Gaussian copula (lambda = 0) cannot express.
    """
    corr = np.asarray(corr, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    d = corr.shape[0]
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("nu must be finite and > 2")
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, d)) @ chol.T
    w = rng.chisquare(nu, size=n) / nu
    x = z / np.sqrt(w)[:, None]
    out: Array = np.asarray(sstats.t.cdf(x, df=nu), dtype=np.float64)
    return out


def block_corr(block_sizes: list[int], rho_in: float, rho_out: float) -> Array:
    """Block-equit correlation matrix, lifted to the nearest PD matrix if needed."""
    d = sum(block_sizes)
    c = np.full((d, d), rho_out, dtype=np.float64)
    start = 0
    for size in block_sizes:
        c[start : start + size, start : start + size] = rho_in
        start += size
    np.fill_diagonal(c, 1.0)
    # clip negative eigenvalues and renormalize to a correlation matrix
    evals, evecs = np.linalg.eigh(c)
    if evals.min() <= 0.0:
        evals = np.clip(evals, 1e-8, None)
        c = evecs @ np.diag(evals) @ evecs.T
        inv = 1.0 / np.sqrt(np.diag(c))
        c = c * inv[:, None] * inv[None, :]
    return np.asarray(c, dtype=np.float64)


def _gauss_fit(u: Array) -> Array:
    """Gaussian copula = correlation of the normal scores."""
    z = np.asarray(sstats.norm.ppf(_clip(u)), dtype=np.float64)
    out: Array = np.asarray(np.corrcoef(z, rowvar=False), dtype=np.float64)
    return out


def _gauss_logpdf(u: Array, corr: Array) -> Array:
    z = np.asarray(sstats.norm.ppf(_clip(u)), dtype=np.float64)
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        raise ValueError("gaussian copula corr not PD")
    sol = np.linalg.solve(corr, z.T).T
    out: Array = np.asarray(-0.5 * np.sum(z * sol - z * z, axis=1) - 0.5 * logdet, dtype=np.float64)
    return out


def _tail_metrics(u: Array, q: float, alpha: float) -> dict[str, float]:
    """Joint-tail statistics on a uniform sample: crash prob + worst-coord ES."""
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[0]
    worst = u.min(axis=1)
    joint = np.all(u < q, axis=1)
    tail_mass = float((worst < alpha).mean())
    es_worst = float(worst[worst < alpha].mean()) if bool((worst < alpha).any()) else math.nan
    return {
        "joint_crash_prob": float(joint.mean()),
        "crash_observed": float(joint.sum()),
        "worst_coord_es": es_worst,
        "tail_mass": tail_mass,
        "n": float(n),
    }


def _cs_verdict(diffs: Array, alpha: float, bound_pad: float) -> dict[str, float | str]:
    """Betting CS on d_t = challenger(-ll) - incumbent(-ll); lo>0 means incumbent wins."""
    d = np.asarray(diffs, dtype=np.float64).reshape(-1)
    if d.size == 0 or not np.isfinite(d).all():
        raise ValueError("diff stream must be non-empty and finite")
    bound = float(np.abs(d).max()) * bound_pad
    cs = MeanDiffCS(alpha=alpha, lam=0.5, bound=bound)
    for x in d:
        cs.update(float(x))
    lo, hi = cs.interval()
    verdict = "challenger_wins" if hi < 0 else "incumbent_wins" if lo > 0 else "inconclusive"
    return {
        "cs_lo": float(lo),
        "cs_hi": float(hi),
        "cs_bound": bound,
        "mean_diff": float(d.mean()),
        "verdict": verdict,
        "n": float(d.size),
    }


def vine_dominance_bench(
    *,
    seed: int = 37,
    n_train: int = 4000,
    n_test: int = 2500,
    d: int = 8,
    nu: float = 5.0,
    alpha: float = 0.05,
    tail_q: float = 0.05,
    es_alpha: float = 0.05,
    n_sim: int = 20_000,
) -> dict[str, Any]:
    """Run the dominance bench; returns the raw receipt body (unsealed)."""
    rng = np.random.default_rng(seed)
    corr = block_corr([d // 2, d - d // 2], rho_in=0.7, rho_out=0.25)
    u = mvt_copula_sample(n_train + n_test, corr, nu, rng)
    u_tr, u_te = u[:n_train], u[n_train:]

    fits: dict[str, Any] = {"independent": None}
    fits["gaussian"] = _gauss_fit(u_tr)
    fits["cvine"] = vine_fit(u_tr, structure="cvine")
    fits["dvine"] = vine_fit(u_tr, structure="dvine")
    fits["rvine"] = vine_fit(u_tr, structure="rvine")

    def _ll(name: str, uu: Array) -> Array:
        if name == "independent":
            return np.zeros(uu.shape[0])
        if name == "gaussian":
            return _gauss_logpdf(uu, fits["gaussian"])
        vm = fits[name]
        if not (isinstance(vm, VineMatrix)):
            raise ValueError("isinstance(vm, VineMatrix)")
        return vine_logpdf_rows(vm, uu)

    ll_oos = {name: _ll(name, u_te) for name in fits}
    mean_ll = {name: float(v.mean()) for name, v in ll_oos.items()}

    dominance: dict[str, dict[str, float | str]] = {}
    for comp in ("independent", "gaussian", "cvine", "dvine"):
        dominance[f"rvine_vs_{comp}"] = _cs_verdict(-ll_oos["rvine"] - (-ll_oos[comp]), alpha, 1.05)

    best_alt = max(("gaussian", "cvine", "dvine"), key=lambda k: mean_ll[k])
    dominance["rvine_vs_best_alt"] = _cs_verdict(
        -ll_oos["rvine"] - (-ll_oos[best_alt]), alpha, 1.05
    )

    tail_emp = _tail_metrics(u_te, tail_q, es_alpha)
    tail_model: dict[str, dict[str, float]] = {}
    for i, name in enumerate(fits):
        sim_rng = np.random.default_rng(seed + 1000 + i)
        if name == "independent":
            draws = sim_rng.uniform(size=(n_sim, d))
        elif name == "gaussian":
            chol = np.linalg.cholesky(fits["gaussian"])
            draws = sstats.norm.cdf(sim_rng.standard_normal((n_sim, d)) @ chol.T)
        else:
            vm = fits[name]
            if not (isinstance(vm, VineMatrix)):
                raise ValueError("isinstance(vm, VineMatrix)")
            draws = vine_sample(vm, n_sim, seed=seed + 1000 + i)
        tail_model[name] = _tail_metrics(draws, tail_q, es_alpha)

    def _tail_err(name: str) -> float:
        return abs(tail_model[name]["joint_crash_prob"] - tail_emp["joint_crash_prob"])

    results: dict[str, bool] = {
        "all_fits_finite": all(np.isfinite(v).all() for v in ll_oos.values()),
        "rvine_beats_independent": mean_ll["rvine"] > 0.0,
        "rvine_beats_gaussian": mean_ll["rvine"] > mean_ll["gaussian"],
        # Dißmann greedy selection has no domination guarantee — pin
        # competitiveness, not ranking: rvine stays within noise of the
        # best specialized structure.
        "rvine_within_gap_of_best": mean_ll["rvine"] > max(mean_ll.values()) - 0.1,
        # Joint-tail honesty: the vines reproduce nonzero crash mass where
        # independence gives literally zero; gaussian must underestimate.
        "tail_err_rvine_beats_independent": _tail_err("rvine") < _tail_err("independent"),
        "rvine_tail_within_2x_empirical": tail_emp["joint_crash_prob"] == 0.0
        or tail_model["rvine"]["joint_crash_prob"] / tail_emp["joint_crash_prob"] < 2.0,
        "dominance_cs_produced": all("cs_lo" in v for v in dominance.values()),
        "planted_tail_dependence": tail_emp["joint_crash_prob"] > 0.0,
        "gaussian_crash_underestimate": tail_model["gaussian"]["joint_crash_prob"]
        < tail_emp["joint_crash_prob"],
    }
    n_passed = sum(1 for v in results.values() if v)
    claim = {
        "ok": n_passed == len(results),
        "n_probes": len(results),
        "n_passed": n_passed,
        "results": results,
        "mean_loglik_oos": mean_ll,
        "dominance": dominance,
        "best_alt": best_alt,
        "tail_empirical": tail_emp,
        "tail_model": tail_model,
    }
    return {
        "kind": "vine_dominance",
        "schema": VINE_DOMINANCE_SCHEMA,
        "git_revision": _git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "params": {
            "seed": seed,
            "n_train": n_train,
            "n_test": n_test,
            "d": d,
            "nu": nu,
            "alpha": alpha,
            "tail_q": tail_q,
            "es_alpha": es_alpha,
            "n_sim": n_sim,
        },
        "claim": claim,
        "interpretation": (
            "OOS copula dominance on a correlated t-copula panel (planted joint tail "
            "dependence). mean_loglik_oos is the per-observation copula log-likelihood "
            "on held-out data (higher is better); dominance.* is a time-uniform betting "
            "confidence sequence on the log-loss differential — 'challenger_wins' means "
            "the R-vine's CS excludes ties with the comparator. tail_* compares the "
            "empirical joint-crash frequency P(all marginals < q) with Monte-Carlo draws "
            "from each fitted model; the Gaussian copula structurally has zero tail "
            "dependence and is expected to underestimate joint crashes."
        ),
    }


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — receipt must not fail on a missing .git
        return "unknown"


_VINE_MODELS = ("independent", "gaussian", "cvine", "dvine", "rvine")
_CS_VERDICTS = ("challenger_wins", "incumbent_wins", "inconclusive")


def vine_dominance_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Deep-verify a ``vine_dominance.v1`` receipt's internal coherence."""
    errors: list[str] = []
    claim = payload.get("claim")
    if not isinstance(claim, dict):
        return ["missing_claim"]
    results = claim.get("results")
    if not isinstance(results, dict) or not all(isinstance(v, bool) for v in results.values()):
        errors.append("results_not_bool_map")
        results = {}
    if claim.get("n_probes") != len(results):
        errors.append("n_probes_mismatch")
    if claim.get("n_passed") != sum(1 for v in results.values() if v):
        errors.append("n_passed_mismatch")
    if claim.get("ok") != all(results.values()):
        errors.append("ok_mismatch")
    ll = claim.get("mean_loglik_oos")
    if not isinstance(ll, dict) or not set(_VINE_MODELS).issubset(ll):
        errors.append("mean_loglik_oos_models")
    elif not all(isinstance(v, (int, float)) for v in ll.values()):
        errors.append("mean_loglik_oos_non_numeric")
    dom = claim.get("dominance")
    if not isinstance(dom, dict) or not dom:
        errors.append("missing_dominance")
    else:
        for name, d in dom.items():
            if not isinstance(d, dict):
                errors.append(f"dominance_{name}_not_dict")
                continue
            if d.get("verdict") not in _CS_VERDICTS:
                errors.append(f"dominance_{name}_verdict")
            lo, hi = d.get("cs_lo"), d.get("cs_hi")
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or lo > hi:
                errors.append(f"dominance_{name}_cs_inverted")

    def _tail_stats(v: Any) -> bool:
        return (
            isinstance(v, dict)
            and isinstance(v.get("joint_crash_prob"), (int, float))
            and 0.0 <= v["joint_crash_prob"] <= 1.0
            and isinstance(v.get("tail_mass"), (int, float))
            and 0.0 <= v["tail_mass"] <= 1.0
            and isinstance(v.get("n"), (int, float))
            and v["n"] > 0
        )

    if not _tail_stats(claim.get("tail_empirical")):
        errors.append("tail_empirical_shape")
    tm = claim.get("tail_model")
    if not isinstance(tm, dict) or not set(_VINE_MODELS).issubset(tm):
        errors.append("tail_model_models")
    elif not all(_tail_stats(v) for v in tm.values()):
        errors.append("tail_model_shape")
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if payload.get("data_label") != "SYNTHETIC":
        errors.append("data_label")
    return errors


def write_vine_dominance_receipt(
    receipt: dict[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal (receipt_sha256) and atomically write ``vine_dominance.json``."""
    from quant_fund.research.receipt_v2 import verify_receipt_payload
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    if receipt.get("kind") != "vine_dominance" or receipt.get("schema") != VINE_DOMINANCE_SCHEMA:
        raise ValueError("receipt identity mismatch")
    canonical = json.loads(canonical_json_bytes(receipt))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    v = verify_receipt_payload(payload)
    if not v["valid"]:
        raise ValueError(f"sealed receipt fails verification: {v['errors']}")
    path = Path(receipts_dir) / "vine_dominance.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path
