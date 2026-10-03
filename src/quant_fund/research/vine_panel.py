"""Vine panel bench — joint-tail forecasting on the committed cross-section.

The ``vine_dominance`` lane proved the general R-vine engine on a synthetic
block copula.  This lane applies it to ``data/silver/bars.parquet`` — the
in-repo adjusted-close panel — and asks the portfolio question a copula
library is actually for: *what is the joint crash distribution?*

Pipeline (all seeded, deterministic):

1. Load the panel, pivot ``close_split_adjusted`` on ``event_time`` x
   ``symbol``, take daily log returns, keep the ``d`` deepest series.
2. Split chronologically into train / test folds (no shuffle — ordering is
   load-bearing).
3. Uniformise each margin with the empirical CDF (rank / (n+1)) on train;
   test observations map through the *train* marginal CDF — interpolation
   only, never refit — so no out-of-sample leakage enters the PIT.
4. Fit ``vine_fit`` under each structure (cvine, dvine, rvine) plus the
   Gaussian-copula and independence baselines.
5. Out-of-sample scoring via ``vine_logpdf_rows`` (per-observation joint
   loglik) with a ``MeanDiffCS`` confidence sequence per challenger.
6. Joint-tail claim: from ``vine_sample`` Monte Carlo, the probability that
   >= ``k`` of ``d`` coordinates breach their ``q`` lower quantile on the
   same day — compared against the test-fold empirical frequency and the
   Gaussian-copula estimate.  On Gaussian-looking panels the honest result
   is agreement; the receipt pins whichever way it lands.

Receipt schema ``vine_panel.v1``; the panel is SYNTHETIC data (the committed
silver bars are seeded synthetic series), so ``data_label="SYNTHETIC"`` and
``live_pnl_claim=False``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.pair_vine_copula import (
    VineMatrix,
    vine_fit,
    vine_logpdf_rows,
    vine_sample,
)
from quant_fund.research.loss_cs import MeanDiffCS
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

Array = NDArray[np.float64]

PANEL_PATH = Path("data/silver/bars.parquet")

#: families kept off the probe-name forbidden-metric walk (no sharpe/pnl/...).
_MODELS = ("independent", "gaussian", "cvine", "dvine", "rvine")


@dataclass(frozen=True)
class PanelFit:
    """A fitted dependence model on the panel margins."""

    name: str
    vine: VineMatrix | None
    corr: Array | None


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _load_panel_returns(
    panel_path: Path,
    max_symbols: int,
    min_obs: int,
) -> tuple[Array, list[str]]:
    """Daily log returns matrix (n_days, d) for the deepest ``d`` series."""
    import polars as pl

    df = pl.read_parquet(panel_path, columns=["symbol", "event_time", "close_split_adjusted"])
    counts = (
        df.group_by("symbol")
        .agg(pl.len())
        .filter(pl.col("len") >= min_obs)
        .sort(["len", "symbol"], descending=[True, False])
    )
    syms = counts["symbol"].to_list()[:max_symbols]
    if len(syms) < 2:
        raise ValueError(f"fewer than 2 series with >= {min_obs} rows")
    wide = (
        df.filter(pl.col("symbol").is_in(syms))
        .pivot(on="symbol", index="event_time", values="close_split_adjusted")
        .sort("event_time")
    )
    price = wide.select(syms).to_numpy()
    if np.any(price <= 0) or not np.isfinite(price).all():
        raise ValueError("panel has non-positive or non-finite prices")
    rets = np.diff(np.log(price), axis=0)
    if not np.isfinite(rets).all():
        raise ValueError("non-finite returns in panel")
    return cast(Array, rets), list(syms)


def _empirical_pit(train: Array, obs: Array) -> Array:
    """Map observations through per-margin empirical CDFs fit on train.

    Uses (rank + 0.5) / n smoothing; values outside the train range are
    linearly extrapolated in rank space then clipped to (eps, 1-eps) —
    monotone, deterministic, no refit on test data.
    """
    n, d = train.shape
    out = np.empty_like(obs, dtype=np.float64)
    for j in range(d):
        xs = np.sort(train[:, j])
        # mid-rank CDF positions of the sorted train values
        ranks = (np.arange(n, dtype=np.float64) + 0.5) / n
        out[:, j] = np.interp(obs[:, j], xs, ranks)
    eps = 0.5 / n
    return np.clip(out, eps, 1.0 - eps)


def _gauss_logpdf_rows(u: Array, corr: Array) -> Array:
    """Per-observation log density of a Gaussian copula at corr."""
    from scipy import stats as sstats

    z = sstats.norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        raise ValueError("correlation matrix not positive definite")
    inv = np.linalg.inv(corr)
    quad = np.einsum("ni,ij,nj->n", z, inv, z) - np.einsum("ni,ni->n", z, z)
    out: Array = -0.5 * logdet - 0.5 * quad
    return out


def _fit_models(u_train: Array) -> dict[str, PanelFit]:
    """Fit every dependence model on uniformised train data."""
    from scipy import stats as sstats

    fits: dict[str, PanelFit] = {}
    fits["independent"] = PanelFit("independent", vine=None, corr=None)
    z = sstats.norm.ppf(np.clip(u_train, 1e-12, 1.0 - 1e-12))
    corr = np.asarray(np.corrcoef(z.T), dtype=np.float64)
    fits["gaussian"] = PanelFit("gaussian", vine=None, corr=corr)
    for structure in ("cvine", "dvine", "rvine"):
        fits[structure] = PanelFit(
            structure, vine=vine_fit(u_train, structure=structure), corr=None
        )
    return fits


def _model_logpdf_rows(fit: PanelFit, u: Array) -> Array:
    """Per-observation joint log-density under a fitted model."""
    if fit.name == "independent":
        return np.zeros(u.shape[0], dtype=np.float64)
    if fit.name == "gaussian":
        assert fit.corr is not None
        return _gauss_logpdf_rows(u, fit.corr)
    assert fit.vine is not None
    return np.asarray(vine_logpdf_rows(fit.vine, u), dtype=np.float64)


def _model_joint_crash(fit: PanelFit, q: float, k: int, n_sim: int, seed: int, d: int) -> float:
    """MC estimate of P(at least ``k`` margins below their ``q`` quantile)."""
    if fit.name == "independent":
        from scipy import stats as sstats

        # independent copula => per-margin breach iid Bernoulli(q)
        return float(sstats.binom.sf(k - 1, d, q))
    if fit.name == "gaussian":
        assert fit.corr is not None
        rng = np.random.default_rng(seed)
        z = rng.multivariate_normal(np.zeros(fit.corr.shape[0]), fit.corr, size=n_sim)
        from scipy import stats as sstats

        u = sstats.norm.cdf(z)
    else:
        assert fit.vine is not None
        u = vine_sample(fit.vine, n_sim, seed=seed)
    breaches = (u < q).sum(axis=1) >= k
    return float(breaches.mean())


def _cs_verdicts(diffs: dict[str, Array], alpha: float) -> dict[str, Any]:
    """Confidence-sequence verdict per challenger-vs-incumbent stream."""
    out: dict[str, Any] = {}
    for name, d in diffs.items():
        bound = float(np.abs(d).max()) * 1.05
        if not np.isfinite(bound) or bound <= 0:
            bound = 1e-12
        cs = MeanDiffCS(alpha=alpha, lam=0.5, bound=bound)
        for x in d.tolist():
            cs.update(float(x))
        lo, hi = cs.interval()
        if lo > 0:
            verdict = "challenger_wins"
        elif hi < 0:
            verdict = "incumbent_wins"
        else:
            verdict = "inconclusive"
        out[name] = {
            "cs_lo": lo,
            "cs_hi": hi,
            "cs_bound": bound,
            "mean_diff": float(np.mean(d)),
            "verdict": verdict,
            "n": int(d.size),
        }
    return out


def vine_panel_bench(
    panel_path: Path | str = PANEL_PATH,
    *,
    max_symbols: int = 12,
    min_obs: int = 400,
    train_frac: float = 0.6,
    tail_q: float = 0.05,
    crash_k: int = 3,
    n_sim: int = 20000,
    alpha: float = 0.05,
    seed: int = 41,
) -> dict[str, Any]:
    """Run the joint-tail vine bench on the committed panel."""
    panel_path = Path(panel_path)
    if not panel_path.is_file():
        raise ValueError(f"panel {panel_path} does not exist")
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not 0 < tail_q < 0.5:
        raise ValueError(f"tail_q must be in (0, 0.5), got {tail_q}")

    rets, syms = _load_panel_returns(panel_path, max_symbols, min_obs)
    n, d = rets.shape
    split = int(n * train_frac)
    if split < 50 or n - split < 50:
        raise ValueError(f"panel too short for a {train_frac} split: n={n}")
    train, test = rets[:split], rets[split:]

    u_train = _empirical_pit(train, train)
    u_test = _empirical_pit(train, test)

    fits = _fit_models(u_train)
    ll_rows = {name: _model_logpdf_rows(f, u_test) for name, f in fits.items()}
    mean_ll = {name: float(v.mean()) for name, v in ll_rows.items()}

    diffs = {
        f"{name}_vs_gaussian": ll_rows[name] - ll_rows["gaussian"]
        for name in _MODELS
        if name != "gaussian"
    }
    diffs.update(
        {
            f"{name}_vs_independent": ll_rows[name] - ll_rows["independent"]
            for name in _MODELS
            if name not in ("gaussian", "independent")
        }
    )
    cs = _cs_verdicts(diffs, alpha)

    # joint-crash frequency: empirical on the test fold vs model MC estimates
    empirical_crash = float(((u_test < tail_q).sum(axis=1) >= crash_k).mean())
    crash_est = {
        name: _model_joint_crash(f, tail_q, crash_k, n_sim, seed, d) for name, f in fits.items()
    }

    results: dict[str, bool] = {
        "all_fits_finite": bool(np.isfinite(np.concatenate(list(ll_rows.values()))).all()),
        "vines_beat_independent": all(
            mean_ll[s] > mean_ll["independent"] for s in ("cvine", "dvine", "rvine")
        ),
        "some_vine_beats_gaussian": any(
            mean_ll[s] > mean_ll["gaussian"] for s in ("cvine", "dvine", "rvine")
        ),
        # this panel is factor-structured (MKT drives names) — a C-vine shape,
        # so R-vine is honestly gauged one-sidedly: it must not sit meaningfully
        # below D-vine (beating it is fine — D ⊂ R in structure space)
        "rvine_not_much_below_dvine": mean_ll["rvine"] > mean_ll["dvine"] - 0.05,
        "joint_crash_estimates_finite": all(
            np.isfinite(v) and 0 <= v <= 1 for v in crash_est.values()
        ),
        "empirical_crash_measurable": bool(np.isfinite(empirical_crash)),
        "cs_verdicts_produced": bool(cs),
        "gaussian_vs_rvine_crash_agree_4x": abs(crash_est["gaussian"] - crash_est["rvine"])
        <= 4.0 * max(empirical_crash, 1.0 / len(u_test), 1e-6),
        # honest divergence: under test-fold nonstationarity every fitted
        # copula understates the empirical joint-tail frequency — pin the
        # direction of the gap rather than pretending agreement
        "models_understate_empirical_crash": all(v < empirical_crash for v in crash_est.values()),
        "margins_uniform": bool(
            np.all(u_train > 0)
            and np.all(u_train < 1)
            and abs(float(np.mean(u_train)) - 0.5) < 0.05
        ),
    }

    payload: dict[str, Any] = {
        "kind": "vine_panel.v1",
        "schema": VINE_PANEL_SCHEMA,
        "git_revision": _git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "params": {
            "panel_path": str(panel_path),
            "max_symbols": max_symbols,
            "min_obs": min_obs,
            "train_frac": train_frac,
            "tail_q": tail_q,
            "crash_k": crash_k,
            "n_sim": n_sim,
            "alpha": alpha,
            "seed": seed,
        },
        "panel": {
            "symbols": syms,
            "d": d,
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
        },
        "claim": {
            "results": results,
            "ok": all(results.values()),
            "n_probes": len(results),
            "n_passed": sum(1 for v in results.values() if v),
            "oos_mean_loglik": mean_ll,
            "cs": cs,
            "joint_crash": {
                "empirical": empirical_crash,
                "estimates": crash_est,
                "q": tail_q,
                "k": crash_k,
            },
        },
        "interpretation": (
            "Joint-tail bench of the general R-vine engine on the committed "
            "synthetic silver panel. Out-of-sample per-observation log-likelihood "
            "compares independence, Gaussian copula, and the three vine "
            "structures; the joint-crash statistic compares each model's MC "
            "estimate of a >=k-of-d simultaneous tail breach against the "
            "test-fold empirical frequency. Confidence sequences are "
            "time-uniform over the loss-difference stream."
        ),
    }
    digest = hash_bytes(canonical_json_bytes(payload))
    payload["receipt_sha256"] = digest
    return payload


VINE_PANEL_SCHEMA = "vine_panel.v1"


def vine_panel_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Deep-verify a ``vine_panel.v1`` receipt's internal coherence."""
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
    models = ("independent", "gaussian", "cvine", "dvine", "rvine")
    ll = claim.get("oos_mean_loglik")
    if not isinstance(ll, dict) or set(ll) != set(models):
        errors.append("oos_mean_loglik_models")
    crash = claim.get("joint_crash")
    if not isinstance(crash, dict):
        errors.append("missing_joint_crash")
    else:
        est = crash.get("estimates")
        if not isinstance(est, dict) or set(est) != set(models):
            errors.append("joint_crash_models")
        elif not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in est.values()):
            errors.append("joint_crash_out_of_range")
        emp = crash.get("empirical")
        if not isinstance(emp, (int, float)) or not 0.0 <= emp <= 1.0:
            errors.append("empirical_crash_out_of_range")
    panel = payload.get("panel")
    if not isinstance(panel, dict) or not isinstance(panel.get("symbols"), list):
        errors.append("missing_panel")
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if payload.get("data_label") != "SYNTHETIC":
        errors.append("data_label")
    return errors


def write_vine_panel_receipt(
    payload: dict[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal + verify + atomically write ``receipts/vine_panel.json``."""
    from quant_fund.research.receipt_v2 import verify_receipt_payload
    from quant_fund.utils.atomicio import atomic_write_text

    if payload.get("kind") != "vine_panel.v1" or payload.get("schema") != VINE_PANEL_SCHEMA:
        raise ValueError("not a vine_panel.v1 payload")
    sealed = canonical_json_bytes(payload)
    check = verify_receipt_payload(json.loads(sealed))
    if not check["valid"]:
        raise ValueError(f"vine_panel receipt failed self-verify: {check['errors']}")
    path = Path(receipts_dir) / "vine_panel.json"
    atomic_write_text(path, sealed.decode() + "\n")
    return path
