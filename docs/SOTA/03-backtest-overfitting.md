# 03 — Backtest overfitting & evaluation integrity

Lane: **BACKTEST OVERFITTING & EVALUATION INTEGRITY**. Status: research note +
adoption plan. This file changes nothing; it proposes changes.

Framing is fixed by the repo honesty contract (AGENTS.md): research results are
**proper scores** (pinball, CRPS, PIT, QLIKE, Brier, ECE, Kupiec). The
Bailey/Borwein/López de Prado family of statistics is implemented here as a
*deflator applied to a date-level score series* (the IC the runner already
ranks), not as a headline performance ratio. `FORBIDDEN_RESEARCH_METRIC_KEYS`
(`quant_fund.research.catalog.registry`) and `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`
stay the enforcement point; every new key below must avoid those tokens.
Receipts are immutable evidence: a diagnostic that cannot be recomputed from a
receipt hash is not a diagnostic.

---

## 1. Verdict

The statistical machinery is **already at or above published SOTA** for this
lane: CPCV with per-block purge and two-sided embargo, CSCV/PBO with the logit
degeneracy diagnostic, PSR/DSR/MinTRL with an effective-trial count from
average-linkage Mantegna clustering, White's Reality Check, Hansen's SPA,
Romano–Wolf StepM, Hansen–Lunde–Nason MCS, a pre-registered provenance sweep,
and a Gençay-style leaky-oracle red team. That is more than most papers in the
area reproduce.

The failure modes left in the repo are **wiring failures, not math failures**:

1. CPCV exists but the ranker that is actually *selected* is evaluated on a
   purged walk-forward path only — the \(\binom{N-1}{k-1}\) reconstructed
   paths are counted, never used.
2. The multiplicity denominator (`n_trials`) is per-run. Search intensity that
   accumulates across runs, days, and verifier lanes is not carried into the
   deflation that gets stamped on the receipt.
3. Two promotion rules exist and they do not talk to each other:
   `validation/gates.py` (research correctness) and `hedge_lab/promotion.py`
   (IC + DM + StepM + RC + SPA). Neither consults the other's evidence.

Section 3 gives the gap analysis with file evidence; section 4 gives the
adoption plan (module names, gate hooks, pseudocode, tests).

---

## 2. Techniques and citations

### 2.1 Purging and embargoing

Overlapping labels break the i.i.d. assumption of K-fold CV: a 5-bar forward
return at decision date \(t\) is a function of prices inside a test block that
starts before \(t+h\). Purge every training sample whose label window
\((t, t+h]\) reaches a test window \([t_s, t_e]\); embargo the next \(e\) bars
after each test block to remove autoregressive/serial dependence (López de
Prado 2018, ch. 7 & 12; Bailey et al. 2014 §on overlapping observations).
Purge **per contiguous test block** — a train block between two test blocks
survives when its label reaches neither.

### 2.2 Combinatorial purged cross-validation (CPCV)

Partition the timeline into \(N\) contiguous groups, form every combination of
\(k\) groups as test: \(\binom{N}{k}\) splits. Stitching the test forecasts
across splits yields \(\phi = \binom{N-1}{k-1} = \frac{k}{N}\binom{N}{k}\)
**reconstructed backtest paths**, each using every observation out-of-sample
exactly once. Path-level dispersion is the object of interest: it is what
distinguishes a strategy from a lucky split (López de Prado 2018, ch. 12;
López de Prado 2020, ch. 3).

### 2.3 CSCV and the probability of backtest overfitting (PBO)

Cut an aligned score panel into an even number \(S\) of contiguous slices
(\(S \le 16\) in the published illustration). Each of the \(\binom{S}{S/2}\)
symmetric partitions is in-sample; the complement is out-of-sample. Let
\(\omega_c\) be the *relative* OOS rank of the IS-best configuration in split
\(c\), \(\lambda_c = \ln(\omega_c / (1-\omega_c))\). Then

\[
\mathrm{PBO} = \Pr\!\left[\lambda_c < 0\right]
             = \Pr\!\left[\text{IS-best lands below the OOS median}\right].
\]

Under pure noise \(\mathrm{PBO} \to \tfrac12\); low PBO with high trial count
is *not* evidence of skill, only of non-degenerate selection (Bailey, Borwein,
López de Prado & Zhu 2014; 2017). Their companion result: the expected number
of independent trials needed to reach a given PBO — i.e. log-trial growth buys
deflation almost for free.

### 2.4 PSR, DSR, MinTRL, MinBTL

With \(\widehat{\mathrm{SR}}\) the per-period mean/std of the score series,
skewness \(\gamma_3\) and **raw** kurtosis \(\gamma_4\), Lo's (2002) asymptotic
standard error gives

\[
\mathrm{SE} = \sqrt{1 - \gamma_3\,\widehat{\mathrm{SR}} + \tfrac{\gamma_4-1}{4}\,\widehat{\mathrm{SR}}^2},
\qquad
\mathrm{PSR}[\mathrm{SR}^\ast] = \Phi\!\left(\frac{(\widehat{\mathrm{SR}}-\mathrm{SR}^\ast)\sqrt{T}}{\mathrm{SE}}\right).
\]

Deflation raises the hurdle to the expected maximum of \(N\) independent
centered trials, using the Euler–Mascheroni approximation with
\(Z\sim\mathcal N(0,1)\), \(\gamma_{\mathrm{EM}}\approx0.5772\):

\[
\mathrm{SR}^\ast = \sqrt{\mathbb V[\widehat{\mathrm{SR}}]}\,
\Big((1-\gamma_{\mathrm{EM}})\,\Phi^{-1}\!\big(1-\tfrac1N\big)
   + \gamma_{\mathrm{EM}}\,\Phi^{-1}\!\big(1-\tfrac1{Ne}\big)\Big),
\qquad \mathrm{DSR} = \mathrm{PSR}[\mathrm{SR}^\ast] \le \mathrm{PSR}.
\]

\(\mathrm{MinTRL}\) inverts PSR at confidence \(1-\alpha\); \(\mathrm{MinBTL}\)
bounds the observation count at which the expected maximum spurious score stops
dominating (Bailey & López de Prado 2012; 2014). **Unit safety is the whole
ballgame**: feeding an annualized ratio into a per-period \(T\) inflates the
z-statistic by \(\sqrt{252}\approx15.9\times\).

### 2.5 Data-snooping tests: RC, SPA, StepM, MCS

On a \(T\times K\) matrix of performance differentials \(f_{t,k}\) against a
benchmark, resample with the stationary bootstrap (Politis & Romano 1994) at
the Politis–White (2004) automatic block length:

- **Reality Check** — \(V = \max_k \sqrt T\,\bar f_k\), bootstrap recentred at
  the full-sample means; one-sided p-value against \(H_0:\max_k \mathbb E f_k \le 0\)
  (White 2000).
- **SPA** — studentized RC with three recentring variants
  (`lower`/`consistent`/`upper`), monotone in that order under shared draws;
  consistent power against irrelevant alternatives (Hansen 2005).
- **StepM** — step-down FWER-adjusted p-values over the *set* of superior
  models; controls the probability of any false rejection rather than the max
  statistic alone (Romano & Wolf 2005).
- **MCS** — iteratively eliminate the worst model by a standardized range or
  \(T_{\max}\) statistic until the surviving set is not rejected at \(\alpha\);
  the output is a *set*, not a winner (Hansen, Lunde & Nason 2011).

### 2.6 Trial-count deflation for published-factor research

Harvey, Liu & Zhu (2016) argue a new factor needs \(t>3.0\) once ~316 published
factors are counted; Harvey & Liu (2015) price the same adjustment as a
**haircut** on the reported statistic under four multiple-testing corrections
(Bonferroni, Holm, BHY-FDR, and a Bayesian cross-sectional shrinkage). The
transferable idea is the *ledger*: the denominator of every deflator is a
count of what was tried, and that count is a bookkeeping artifact, not a
statistical assumption.

### 2.7 Structural leakage defeats statistical deflation

Gençay (2026, arXiv:2608.27734) builds deliberately leaky oracles (look-ahead
into the label window) and shows they reach arbitrary score levels while
**passing DSR and PBO**: both statistics condition on a search process and a
sample; neither can see that the feature was constructed from the future. The
conclusion is that guardrails must be *structural* — a registry-validated
feature space, AST/point-in-time scanning, and PIT ingest gates — layered under
trial-count deflation, never instead of it.

---

## 3. Gap analysis vs this repo

### 3.1 Inventory (what already exists)

| Area | Module | Status |
| --- | --- | --- |
| Purge / embargo | `src/quant_fund/validation/purging.py`, `embargo.py` | Per-block, horizon-aware; `overlaps` predicate |
| CPCV splits + paths | `src/quant_fund/validation/cpcv.py` | `combinatorial_purged_cv`, `cpcv_n_splits`, `cpcv_n_paths`, `cpcv_path_assignments`, `stitch_group_paths`, `combinatorial_purged_indices` |
| Walk-forward | `src/quant_fund/validation/walk_forward.py` | Expanding/rolling per `config.validation.scheme` |
| PSR / DSR / PBO / MinTRL | `src/quant_fund/metrics/overfitting.py` | `probabilistic_sharpe`, `expected_max_sharpe`, `deflated_sharpe`, `probability_of_backtest_overfitting`, `min_track_record_length`, `effective_n_trials`, `cscv_performance`, `overfitting_diagnostics` |
| CSCV / DSR / PSR drivers | `src/quant_fund/reality/{cscv,dsr,psr}.py` | Unit-safe wrappers; `psr_from_returns` blocks the annualization bug |
| RC / SPA / StepM / MCS | `src/quant_fund/metrics/snooping.py` | Stationary bootstrap, Politis–White block, three SPA recentrings |
| Composite report + verdict | `src/quant_fund/reality/report.py`, `cli.py` | `insufficient_evidence` / `deflated` / `pass`, research-diagnostic only |
| FDR | `src/quant_fund/reality/fdr.py`, `validation/fdr.py`, `validation/multiple_testing.py` | BHY + MCS bridge |
| Pre-registered sweep | `src/quant_fund/research/reality_sweep.py`, `research/reality/preregistration.json` | Frozen grid → `ProvenanceDB.insert_trial` |
| Leakage red team | `src/quant_fund/validation/leakage_redteam.py` | Gençay leaky-oracle study, `structural_lookahead_audit`, `trial_count_deflation` |
| Optuna containment | `src/quant_fund/validation/optuna_guard.py` | Raises if holdout is passed to the tuner |
| Receipt schema | `src/quant_fund/research/receipt_schema.py` | Schema 2; `backtest_overfitting` block required, `dsr <= psr`, unit-interval probabilities |
| Promotion | `src/quant_fund/hedge_lab/promotion.py`, `validation/gates.py` | IC + DM + StepM + RC + SPA; research-correctness gate |
| Narrative | `docs/BACKTEST_OVERFITTING.md` | Field-by-field reading guide, published numeric examples |

### 3.2 Gaps

**G1 — CPCV is counted but not used for selection (highest value).**
`research/benches/ranking.py:oos_rank_scores` builds folds with `walk_forward`
only. `overfitting_diagnostics` (`metrics/overfitting.py:412`) imports
`cpcv_n_splits` / `cpcv_n_paths` / `combinatorial_purged_indices` and stamps the
*counts* on the receipt; the reconstructed paths themselves never feed a score.
Consequence: the receipt asserts "there were 5 paths" while the selected ranker
was chosen from a single chronological split — exactly the object CPCV exists
to replace. `research/benches/intervals.py:bench_cpcv_audit` checks date
separation and purge disjointness on synthetic dates, which is a correctness
bench, not an evaluation path.

**G2 — Trial multiplicity is per-run, not cumulative.**
`run_ranker_research` passes `n_trials` for the configs evaluated in that run
(`research/agent.py:1678`). `research/reality_sweep.py` records cells on the
provenance ledger and computes `deflated_sharpe` on the sweep itself, but the
two denominators are never merged. A researcher who runs the notebook on five
consecutive days with five different grids pays \(N=K\) five times instead of
\(N=5K\) once. Harvey–Liu–Zhu's argument and Bailey–López de Prado's
\(\mathrm{SR}^\ast(N)\) both scale with the *cumulative* count.

**G3 — Two promotion rules, no shared evidence.**
`hedge_lab/promotion.py:clears_cs_promotion` requires positive mean IC, DM
preferring the challenger on \(-\mathrm{IC}\), membership in `stepm_rejected`,
and RC *and* SPA p-values below \(\alpha\). `validation/gates.py:validate_candidate`
requires a verified immutable receipt, causal panels, and multi-fold stability.
Neither path consults the `backtest_overfitting` block it is standing next to.
The honesty contract is right that DSR/PBO must not become a headline gate —
but the *verdict* string from `reality/report.py` (`insufficient_evidence` /
`deflated` / `pass`) is already a research-integrity judgment and can be
recorded as a blocking precondition of the CS rule without introducing a ratio.

**G4 — MCS is implemented and used in benches, absent from promotion.**
`metrics/snooping.py:model_confidence_set` is called from
`validation/multiple_testing.py`, `research/agent.py:468`, and
`research/sota_evidence.py:322`, but the promotion rule stops at StepM
rejection. Hansen–Lunde–Nason's set is the natural output of a lane that is
honest about ties: it says "these survive," not "this one won."

**G5 — Structural lookahead audit runs on planted names, not the live registry.**
`structural_lookahead_audit` is exercised by `research/benches_w810.py` on a
hard-coded list (`future_return_5d`, `lead_1_volume`) and by unit tests. Nothing
scans the feature registry that research runs actually consume. Gençay's result
says this is the layer that matters most, because DSR/PBO cannot compensate for
it.

**G6 — Score-panel aggregation across instruments is unspecified.**
`snooping.py` operates on a \(T\times K\) differential matrix; the research
runner aligns configs onto a *date-level* grid. When a panel has multiple
instruments per date, the cross-sectional collapse (mean IC vs. equal-weight
portfolio IC vs. per-instrument columns) changes the bootstrap block structure
and therefore the p-values. The choice is currently implicit in the runner.

**G7 — Embargo default is horizon-derived, but not asserted at the fold level.**
`config/models.py:1115` defaults `embargo_bars` to `max(config.horizons.bars)`,
which is the right rule. `validation/cpcv.py` takes `embargo_bars` as a caller
argument; nothing asserts that a fold actually dropped \(\ge\) that many bars
after each test block at the timestamp level (as opposed to `bench_cpcv_audit`'s
set-disjointness check on dates).

**G8 — Repo hygiene: a stray nested tree `src/src/` (426 files).**
It contains a duplicate `quant_fund` (including `validation/leakage_redteam.py`,
`research/agent.py`, `metrics/snooping.py`). Any tool that globs `src/**`
(lint, mypy, coverage, grep-based audits) silently double-counts or resolves
the wrong copy. AGENTS.md already warns about the dropped `tests/tests` mirror;
this is the same class of defect on the package side. Not a math gap, but it
directly undermines the reproducibility claim of every audit above.

---

## 4. Adoption plan

Everything below is additive and diagnostic-first. No new headline metric, no
live-trading claim, no rewrite of an existing receipt (schema 1 stays
`legacy_uncomputed`; schema 2 gains fields only).

### 4.1 Module map

| New / changed | Path | Purpose |
| --- | --- | --- |
| new | `src/quant_fund/validation/cpcv_eval.py` | Turn CPCV paths into a *score panel* + path dispersion; no ratio keys |
| new | `src/quant_fund/research/trial_ledger.py` | Append-only cumulative trial registry; derives `n_trials_cumulative`, `n_trials_effective_cumulative` |
| new | `src/quant_fund/research/benches/cpcv_ranking.py` | Bench that selects a ranker on reconstructed paths (G1) |
| new | `src/quant_fund/validation/registry_audit.py` | AST/name scan of the *live* feature registry (G5) |
| change | `src/quant_fund/research/agent.py` | Feed cumulative trials into `overfitting_diagnostics`; stamp path dispersion |
| change | `src/quant_fund/hedge_lab/promotion.py` | Add `research_verdict` + MCS-survivor precondition to the CS rule (G3, G4) |
| change | `src/quant_fund/research/receipt_schema.py` | Schema 2 additive fields; forbid ratio tokens in the new keys |
| change | `src/quant_fund/validation/gates.py` | Assert fold-level embargo (G7); reject stray-tree imports (G8) |
| doc | `docs/BACKTEST_OVERFITTING.md` | Document the new fields and the cumulative-denominator rule |

### 4.2 Gate hooks

```
research run
  └─ validate_feature_registry()            # G5 fail-closed: unknown / lookahead-shaped name
  └─ cpcv_eval.score_panel(...)             # G1: paths, per-path score, dispersion
  └─ trial_ledger.append(run_id, configs)   # G2: cumulative denominator
  └─ overfitting_diagnostics(n_trials=cumulative, ...)
  └─ receipt_schema.validate(schema 2 + additive fields)   # dsr <= psr, probs in [0,1]
promotion
  └─ validation/gates.validate_candidate    # receipts verified, causal panels, multi-fold stability
  └─ reality/report.build_reality_report    # verdict: insufficient_evidence | deflated | pass
  └─ hedge_lab/promotion.clears_cs_promotion(+ research_verdict, + mcs_survivors)
```

The verdict is a **research-integrity precondition**, recorded on the receipt
and in the run manifest with `claim: "research_only"`. It does not size a book
and does not move `blend_weight`.

### 4.3 Pseudocode — CPCV as the evaluation object (G1)

```python
def cpcv_score_panel(
    times, y, predict_fn, *, n_groups=6, k=2,
    horizon_bars, embargo_bars, score=pinball_or_crps,
) -> CpcvPanel:
    folds = combinatorial_purged_cv(           # validation/cpcv.py, unchanged
        times, n_groups, k, horizon_bars, embargo_bars,
    )
    assert_fold_integrity(folds, times, horizon_bars, embargo_bars)   # G7

    oos = {}                                   # (group_id, time) -> prediction
    for fold in folds:
        train = fold.train_times               # already purged + embargoed
        model = predict_fn.fit(train)
        for t in fold.test_times:
            key = (fold.group_of(t), t)
            assert key not in oos, "test forecast reused across paths"
            oos[key] = model.predict(t)        # point-in-time features only

    paths = []
    for path_id in cpcv_path_assignments(n_groups, k):   # phi = C(N-1, k-1)
        rows = [(t, oos[(g, t)]) for g in path_id for t in group_times[g]]
        assert covers_every_group_once(rows, path_id)
        paths.append(per_period_score(rows, score))       # proper score, not a ratio

    return CpcvPanel(
        n_splits=len(folds),
        n_paths=len(paths),
        per_path=paths,                        # (phi, T_path) proper-score series
        dispersion=path_dispersion(paths),     # IQR / MAD across paths, score units
        claim="research_diagnostic_only",
    )
```

Selection then uses the *worst* path, not the mean:

```python
def select_on_paths(panel, configs):
    return {
        cfg: {
            "score_mean": mean_over_paths(panel[cfg]),
            "score_worst_path": min(panel[cfg].per_path_means),   # robust criterion
            "path_dispersion": panel[cfg].dispersion,
            "n_paths": panel.n_paths,
        }
        for cfg in configs
    }
```

Feed `panel[cfg].score_mean` (per-period, in score units) into CSCV/PBO exactly
as `overfitting_diagnostics` does today — the only change is that the panel now
comes from \(\phi\) reconstructed paths instead of one chronological split.

### 4.4 Pseudocode — deflated score as a diagnostic (G2)

```python
def deflated_score_diagnostic(score_series, *, trial_ledger, run_id):
    # 1. cumulative multiplicity, not per-run
    hist = trial_ledger.history(through=run_id)        # append-only, hash-chained
    n_trials = hist.total_configs_evaluated            # includes unaligned ones
    n_eff = effective_n_trials(hist.aligned_matrix)[0] # Mantegna clustering, rho >= 0.5

    # 2. moments in the sampling frequency of the series (never annualized)
    sr, skew, kurt = moments_from_returns(score_series)   # kurt is RAW

    # 3. hurdle = expected max of n_eff centered trials
    sr_star = expected_max_sharpe(n_eff, var_sr=hist.score_variance)
    dsr = deflated_sharpe(sr, sr_star, skew, kurt, n_obs=len(score_series))
    psr = probabilistic_sharpe(sr, 0.0, skew, kurt, len(score_series))
    assert dsr <= psr + 1e-12

    # 4. CSCV/PBO on the same aligned panel
    pbo = probability_of_backtest_overfitting(panel, n_slices=choose_cscv_slices(len(panel)))

    return {
        "metrics_status": "computed" if all_finite(dsr, psr, pbo) else "unavailable",
        "dsr": dsr, "psr": psr, "pbo": pbo,
        "dsr_counted_trials": deflated_sharpe(..., n_eff=n_trials),   # no clustering
        "min_trl": min_trl_from_returns(score_series, confidence=0.95),
        "n_trials": n_trials,                 # cumulative
        "n_trials_effective": n_eff,          # cumulative, clustered
        "n_trials_run": run_scoped_count,     # kept for per-run attribution
        "claim": "research_diagnostic_only",
    }
```

Ledger rules (the part that is easy to get wrong):

- Append-only and hash-chained, like `receipts/`; a run may not rewrite history.
- Every evaluated config counts, including ones that failed to align — dropping
  a config may not shrink the multiplicity.
- Clustering reduces `n_trials` → `n_trials_effective` at a threshold fixed in
  advance (`CORRELATED_TRIAL_MIN_RHO`), never fit to a result.
- Both numbers ship: `dsr` (clustered) and `dsr_counted_trials` (strict), so a
  reader can see how much of the deflation is the clustering.
- Verdict mapping stays as in `reality/report.py`: `insufficient_evidence` when
  `n_trials < 10` or `n_eff < 3`; `deflated` when DSR is non-finite or `< 0.95`,
  or PBO `> 0.5`, or SPA fails to reject.

### 4.5 Promotion hook (G3, G4)

```python
def clears_cs_promotion_v2(*, name, mean_ic, dm_preferred, dm_p,
                           stepm_rejected, rc_p, spa_p, mcs_survivors,
                           research_verdict, alpha=ALPHA):
    if not clears_cs_promotion(name=name, mean_ic=mean_ic,
                               dm_preferred=dm_preferred, dm_p=dm_p,
                               stepm_rejected=stepm_rejected,
                               rc_p=rc_p, spa_p=spa_p, alpha=alpha):
        return False
    if research_verdict != "pass":      # "deflated" / "insufficient_evidence" block
        return False
    return name in mcs_survivors        # HLN set, not a single winner
```

Rationale: StepM answers "is some model superior," MCS answers "which models
survive elimination," and the reality verdict answers "did the search intensity
deflate the result." They are complementary; the CS rule currently uses two of
the three.

### 4.6 Structural registry audit (G5)

```python
def validate_feature_registry(registry, *, extra_patterns=()) -> RegistryAudit:
    findings = structural_lookahead_audit(registry.names, extra_patterns=extra_patterns)
    for feature in registry:
        ast_hits = scan_expression_ast(feature.expr)      # shift(-k), lead(), future_*, .rolling over t+h
        pit_hits = not feature.point_in_time_verified     # datasource gate must attest
        findings.extend(ast_hits + pit_hits)
    assert findings.fail_closed, "registry admits a lookahead-shaped feature"
    return findings
```

Run it in the research entrypoint *before* any scoring, and record the audit
hash on the receipt. This is the layer Gençay shows DSR/PBO cannot substitute
for.

---

## 5. Test strategy

Property/unit (fast, deterministic, no network — the PR gate):

1. **Fold integrity (G7)** — for every fold: train/test time sets disjoint; for
   every test block, no train timestamp lies within `embargo_bars` *after* the
   block end; no train label window \((t, t+h]\) intersects any test window.
   Randomized timestamp grids, not just synthetic daily dates.
2. **Path accounting (G1)** — `len(paths) == C(N-1, k-1)`; each group appears
   exactly once per path; every test forecast used exactly once
   (`assert key not in oos`); `sum(len(p) for p in paths) == phi * n_groups * group_len`.
3. **No-leakage oracle control** — on a pure-noise label, PBO must concentrate
   near 0.5 and DSR must not exceed 0.95 at \(N\ge10\); on a Gençay-style leaky
   oracle, `validate_feature_registry` must fail closed *even though* DSR/PBO
   look fine. This is the single most important test in the lane: it encodes
   "statistics cannot substitute for structure."
4. **Unit safety** — `psr_from_returns(x)` vs `probabilistic_sharpe(annualized(x), T)`
   must differ by ≈\(\sqrt{252}\) in the z-statistic; regression guard for the
   historical `hedge_lab/scoreboard.py` bug.
5. **Monotonicity** — DSR decreasing in `n_trials`; `dsr <= psr` always;
   `dsr_counted_trials <= dsr`; MinTRL increasing in required confidence;
   PBO invariant to relabeling configs.
6. **Ledger append-only** — re-running with the same `run_id` is idempotent;
   inserting a config for an earlier run changes the hash chain and is rejected;
   unaligned configs still increment `n_trials`.
7. **Published numbers** — the two worked examples already in
   `docs/BACKTEST_OVERFITTING.md` (2012 frequency table: 2.73y daily / 2.83y
   weekly / 3.24y monthly; 4.99y at skew −0.72, kurt 5.78. 2014 example:
   DSR ≈ 0.9004 at 100 trials, ≈ 0.9505 at 46 trials / normal moments) become
   parametrized tests with a tolerance of 1e-3.
8. **Snooping controls** — SPA `p_lower <= p_consistent <= p_upper` under shared
   draws; bootstrap p-values never exactly 0; a single strictly dominant column
   yields MCS = {that column} and StepM rejects it; an all-noise \(T\times K\)
   panel yields no rejections at \(\alpha=0.05\) over ≥200 seeds (FWER check).

Contract/schema:

9. **Receipt schema** — the additive fields validate on schema 2; a schema-1
   receipt still verifies and migrates to `legacy_uncomputed` without inventing
   probabilities; any new key containing a forbidden token
   (`sharpe|sortino|calmar|pnl|nav`) fails `family_blob_forbidden_metrics_absent`.
10. **Honesty inheritance** — `tests/fx1/test_honesty_inheritance.py` must stay
    green: no new key may drift between
    `FORBIDDEN_RESEARCH_METRIC_KEYS` and `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`.
11. **Promotion rule** — a config that clears IC/DM/StepM/RC/SPA but carries
    `research_verdict="deflated"` must not promote; a config absent from
    `mcs_survivors` must not promote; SYNTHETIC never takes a champion alias and
    never sets `blend_weight > 0`.

Bench/integration (slow lane, `make test-full` / `make fx1-gate`):

12. `bench_cpcv_eval` — end-to-end CPCV panel on a bounded synthetic grid with
    a fixed seed; asserts path counts, dispersion finiteness, and
    `claim == "research_diagnostic_only"`.
13. `bench_registry_audit` — the live registry passes and a planted
    `future_return_5d` fails.
14. `verify-research` receipt round-trip — recompute DSR/PBO/PSR from the
    receipt hash and assert bit-stable equality within 1e-12.

---

## 6. Explicit non-goals

- No Sharpe/Sortino/Calmar/P&L/NAV headline anywhere, including in the
  diagnostic block. Score-based deflation only.
- No live-trading claim, no broker connectivity; see
  `docs/INSTITUTIONAL_READINESS.md` for the five minimum-evidence conditions.
- No rewrite of existing receipts. Immutable evidence stays immutable; new
  runs get the new fields, old ones keep `legacy_uncomputed`.
- DSR/PBO never become a *performance* gate. They gate the **integrity of the
  search process**, which is the only thing they can legitimately speak to.

---

## 7. References

1. Bailey, D. H., & López de Prado, M. (2012). The Sharpe ratio efficient
   frontier. *Journal of Risk*, 15(2), 107–143. (PSR; MinTRL; Lo's SE.)
2. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014).
   Pseudo-mathematics and financial charlatanism: The effects of backtest
   overfitting on out-of-sample performance. *Notices of the AMS*, 61(5),
   458–471.
3. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The
   probability of backtest overfitting. *Journal of Computational Finance*,
   20(4), 39–69. (CSCV; PBO; degenerate \(\lambda\) diagnostic.)
4. Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio:
   Correcting for selection bias, backtest overfitting, and non-normality.
   *Journal of Portfolio Management*, 40(5), 94–107. (DSR; \(\mathrm{SR}^\ast\);
   MinBTL.)
5. White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5),
   1097–1126.
6. Hansen, P. R. (2005). A test for superior predictive ability. *Journal of
   Business & Economic Statistics*, 23(4), 365–380.
7. Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalized
   data snooping. *Econometrica*, 73(4), 1237–1282.
8. Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
   *Econometrica*, 79(2), 453–497.
9. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*,
   89(428), 1303–1313.
10. Politis, D. N., & White, H. (2004). Automatic block-length selection for the
    dependent bootstrap. *Econometric Reviews*, 23(1), 53–70.
11. Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts
    Journal*, 58(4), 36–52.
12. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
    Ch. 7 (cross-validation), ch. 12 (backtesting through CV; CPCV).
13. López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge
    University Press. Ch. 3 (CPCV, path reconstruction, Mantegna distance).
14. Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of
    expected returns. *Review of Financial Studies*, 29(1), 5–68.
15. Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio
    Management*, 42(1), 13–28. (Haircut Sharpe ratios.)
16. Mantegna, R. N. (1999). Hierarchical structure in financial markets.
    *European Physical Journal B*, 11(1), 193–197.
17. Gençay, E. (2026). What survives honest evaluation? Leakage-safe,
    search-aware assessment of LLM-driven trading strategy discovery.
    arXiv:2608.27734.
18. In-repo: `docs/BACKTEST_OVERFITTING.md` (field semantics + worked examples),
    `src/quant_fund/research/research100.json` R072 (AFML/CPCV entrypoint),
    `verifier/README.md` (v8 run-manifest `claim: "research_only"`).
