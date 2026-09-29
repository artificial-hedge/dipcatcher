# 28 — Backtest-Overfitting Controls & Promotion Gating (audit + fix wave)

Date: 2026-09-29 (UTC+5:30 local)
Lane: `src/quant_fund/validation/**`, `src/quant_fund/conformal/**`,
`src/quant_fund/hedge_lab/**` and their test dirs.
Status of each work item: **CONFIRMED / ALREADY FIXED / NOT AS DESCRIBED**, with
`file:line` evidence.

All numbers below are **proper scores** (IC, pinball/CRPS-shaped losses,
coverage, PIT/logit statistics). Nothing in this lane or this document
headlines Sharpe, Sortino, Calmar, P&L or NAV. `DSR`/`PSR`/`MinTRL` appear only
as *overfitting-control probabilities and observation counts*, which is the
published role of those statistics and is already mirrored by
`fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`.

---

## 0. Executive summary

| # | Work item | Verdict | Action taken |
|---|---|---|---|
| 1 | CPCV purge & embargo, CSCV/PBO, DSR trial & moment adjustments | **ALREADY CORRECT** (math verified against the papers), but the embargo guarantee was *asserted nowhere at bar level* | Added `fold_embargo_report` / `assert_fold_embargo` / `cpcv_folds_verified` — an enforcement wrapper, not another annotation |
| 2 | Worst-path / deflated ranking vs raw in-sample best | **CONFIRMED gap in `validation/`**: no conservative selection criterion existed over reconstructed CPCV paths | Added opt-in `rank_configs(..., criterion="worst_path")`; default stays `"mean"` so no caller changes silently |
| 3 | Computed-but-unenforced gates | **CONFIRMED — real bug**, `hypothesis_family_split` | Wired into both `research_ok` and `promotion`; regression tests prove rejection |
| 4 | Conformal coverage calibration | **NOT AS DESCRIBED** — `src/quant_fund/conformal/**` does not exist | Reported; coverage measured read-only against the modules that do exist (`models/conformal.py`) |

Baseline gates before any edit: `ruff check` clean on the lane, `mypy
src/quant_fund` = **Success: no issues found in 671 source files**.

---

## 1. CPCV / CSCV overfitting controls

### 1.1 Purge & embargo — CORRECT

`combinatorial_purged_cv` (`src/quant_fund/validation/cpcv.py:33-96`) applies
purge and embargo **per contiguous test group**, not over the min/max span of a
non-contiguous combination. That detail is what makes it right: taking the span
of e.g. groups 0 and 3 would wipe the intervening train groups 1–2.

Verified behaviours (each locked by a test):

* Train label windows `(i, i+horizon]` reaching a holdout are dropped
  (`cpcv.py:214-219`, `purge_mask`).
* `embargo` sessions are dropped both **after** `hi` and **before** `lo` of each
  block (`cpcv.py:220-222`).
* Folds whose purged train set empties are **omitted**, never padded — an
  honest residual: fewer folds, never a silent claim of full combinatorial
  coverage (`cpcv.py:224-227`).
* Duplicate timestamps are de-duplicated before grouping (`cpcv.py:59-64`).

Split/path combinatorics match López de Prado, *AFML* (2018) ch. 12 exactly:
`cpcv_n_splits` = C(N,k), `cpcv_n_paths` = C(N−1,k−1) = C(N,k)·k/N
(`cpcv.py:106-120`). For N=6,k=2: 15 splits, 5 paths — asserted in
`test_stitched_cpcv_paths_feed_the_worst_path_criterion_end_to_end`.

`cpcv_path_assignments` (`cpcv.py:122-143`) balances incidence so **each
(split, group) pair with group in split appears on exactly one path** — stitched
paths never reuse a test forecast. It raises `RuntimeError` if the incidence is
unbalanced rather than returning a subtly-wrong assignment.

### 1.2 The real gap: nothing asserted the embargo at bar level

`bench_cpcv_audit`
(`src/quant_fund/research/benches/intervals.py:618-648`) checks
`train.isdisjoint(test)` and stamps `purge_embargo_valid`. **Set-disjointness is
strictly weaker than the embargo guarantee.** A fold can be perfectly disjoint
and still train on the bar *immediately after* a holdout block — which is
exactly the serial dependence the embargo exists to cut. The flag name promised
"purge/embargo valid"; the check only proved "no shared dates".

Fix (new, `src/quant_fund/validation/cpcv.py:230-374`):

* `fold_embargo_report(folds, times, horizon_bars=, embargo_bars=)` counts four
  distinct violation classes per **contiguous** test block:
  `train_in_test`, `label_reaches_test`, `embargo_after_violation`,
  `embargo_before_violation`. `ok` is True only when all four are zero **and**
  at least one block was inspected, so "nothing inspected" cannot masquerade as
  "nothing wrong".
* `assert_fold_embargo(...)` is the **enforcement** wrapper: raises
  `AssertionError` on any violation, returns the report so a caller can stamp
  the evidence on a receipt *after* the assertion passed.
* `cpcv_folds_verified(...)` returns folds plus the already-checked report, for
  callers who want the guarantee at construction time.

Fail-closed inputs: negative horizon/embargo, a fold timestamp absent from
`times`, and an empty test block all raise `ValueError`
(`cpcv.py:283-306`). `claim="validation_integrity_only"` and the key set carry
no ratio token — asserted in
`test_rank_configs_is_research_only_and_carries_no_ratio_keys` and
`test_fold_embargo_report_passes_for_real_cpcv_folds`.

New tests: 15 collected in `tests/unit/research/test_cpcv_extremes.py` for this
item alone, including
`test_fold_embargo_report_detects_a_hollowed_out_embargo`, which builds a
**disjoint** fold that lets the three embargoed bars back in and requires it to
be caught. That test fails if the enforcement ever regresses.

### 1.3 CSCV / PBO — CORRECT

`probability_of_backtest_overfitting`
(`src/quant_fund/metrics/overfitting.py:91`) implements Bailey, Borwein,
López de Prado & Zhu, *J. Computational Finance* 20(4), 2017: relative OOS rank
\(\omega_c = \mathrm{rank}/(N+1)\), logit \(\lambda_c = \ln(\omega_c/(1-\omega_c))\),
\(\mathrm{PBO} = \Pr(\lambda_c < 0)\).

Verified two ways:

* **Hand cases.** An overfit configuration (IS-best is OOS-worst on every split)
  gives `PBO = 1.0`; a clean configuration gives `PBO = 0.0`.
* **Equivalence.** The paper's plain-language definition — "the IS-best is below
  the OOS median" — is the *same predicate* as \(\lambda_c<0\) whenever ranks are
  untied, because \(\omega_c < \tfrac12 \iff \mathrm{rank} < (N+1)/2\). Checked
  over 300 random grids: **300/300 exact agreement**, 0 disagreements. The rank
  form is the one that stays well-defined under ties, which is why the code uses
  it.

`choose_cscv_slices` enforces \(S \ge 2\) and even, as the paper requires.

### 1.4 DSR trial count & moment adjustments — CORRECT

`deflated_sharpe` (`src/quant_fund/metrics/overfitting.py`) uses the Lo (2002)
non-normal asymptotic standard error
\(\mathrm{SE}=\sqrt{1-\hat\gamma_3 \mathrm{SR}+\tfrac{\hat\gamma_4-1}{4}\mathrm{SR}^2}\)
with **raw** (not excess) kurtosis, and the Bailey & López de Prado (*JPM* 40(5),
2014) hurdle \(\mathrm{SR}^*=\sqrt{V[\mathrm{SR}]}\left[(1-\gamma)Z^{-1}(1-\tfrac1N)+\gamma Z^{-1}(1-\tfrac1{Ne})\right]\)
with \(\gamma\) the Euler–Mascheroni constant.

Reproduced the paper's numerical example to 4 decimal places:

| Case | Computed | Published | \|Δ\| |
|---|---|---|---|
| \(\mathrm{SR}^*\), N=100 | 0.1131720 | ≈0.1132 | 3e-5 |
| DSR, N=100 | **0.900397** | 0.9004 | 3.2e-6 |
| DSR, N=46 | **0.950502** | 0.9505 | 1.7e-6 |
| DSR, Normal moments, N=88 | **0.950491** | ≈0.9505 | 9e-6 |

Hand-derivation of the N=100 case (independent of the code path):
\(\mathrm{SR}=2.5/\sqrt{250}=0.15811388\), \(\mathrm{SE}=1.23717082\),
\(\mathrm{SR}^*=0.11317200\), \(z=1.283816\), \(\Phi(z)=0.900397\). ✔

`min_track_record_length` reproduces the *J. Risk* 15(2), 2012 §5 frequency
table — **2.7310 / 2.8288 / 3.2398 years** against published 2.73 / 2.83 / 3.24
for daily (252), weekly (52), monthly (12) — and **4.9913 years** against the
paper's 4.99 for the HFR moment pair (skew −0.72, raw kurtosis 5.78). These are
locked in `test_min_trl_matches_bailey_lopezdeprado_2012_frequency_table`.

Note on convention: the annualization divisor must match the observation count
(1250 daily obs = 5 × 250 → `sqrt(250)`). Mixing `sqrt(252)` with a 1250-length
record reproduces the paper to only ~3e-3, not ~3e-6. The code is consistent;
the tests pin the consistent convention.

Random-grid invariants (0 violations over 400 draws each, now also property
tests): DSR ≤ PSR, DSR monotone non-increasing in trial count, both in [0,1].
`effective_n_trials` uses the correlation haircut with
`CORRELATED_TRIAL_MIN_RHO`, so correlated trials do not understate multiplicity.
`receipt_schema.py:141-176` additionally rejects a receipt where
`n_trials_effective > n_trials`, or where `pbo`/`dsr` are missing while
`n_trials` is present — the stamp cannot be partially filled in.

---

## 2. Worst-path ranking

**Finding: CONFIRMED gap in `validation/`.** Before this wave the lane had
`stitch_group_paths` producing \(C(N-1,k-1)\) reconstructed paths and *no
criterion for choosing a config over them*. The only worst-case selection
anywhere in the repo was `nested_worst_univariate_scores`
(`src/quant_fund/hedge_lab/mirror.py`), which takes the worst **train-fold IC**
inside one nested CV — a different, narrower question. So CPCV paths existed but
selection over them was left entirely to the caller, who would naturally reach
for the mean.

Added (`src/quant_fund/validation/cpcv.py:376-560`):

* `rank_configs(path_scores, criterion="mean"|"worst_path") -> ConfigRanking`.
  Accepts `(n_paths,)` or `(n_paths, n_periods)` (reduced per-path by mean).
  `detail` exposes `mean`, `worst_path`, `best_path`, `path_dispersion`,
  `n_paths`.
* **Default is `"mean"`.** Deliberate: silently flipping every existing caller to
  a conservative criterion would change results without review. Opt-in only.
* `ConfigRanking` is `claim="research_diagnostic_only"`, `research_only=True`.
  It ranks proper scores (IC, pinball, CRPS — whatever the caller stitched),
  **does not promote anything, and does not move `blend_weight`.**

Why worst-path is the right conservative criterion: on a `lucky` config that
scores 0.90 on one path and 0.01 on the other three, the mean (0.2325) beats a
`steady` config at 0.20 on every path — the in-sample-best anti-pattern, now
reproducible from a *legitimate* CPCV run. `criterion="worst_path"` selects
`steady`. `test_stitched_cpcv_paths_feed_the_worst_path_criterion_end_to_end`
drives this through the **real** `cpcv_path_assignments` + `stitch_group_paths`
path with N=6,k=2: `lucky` gets four 0.30 paths and one −0.50 path (mean 0.14 >
0.10, worst path −0.50 ≪ 0.10), so mean picks `lucky` and worst-path picks
`steady`.

Fail-closed: unknown criterion, empty mapping, empty/non-1-D/2-D or non-finite
scores, and **mismatched path counts** all raise `ValueError`. The last one
matters for honesty — silently dropping a config would understate the selection
multiplicity that PBO/DSR are supposed to correct for.

Property test
`test_worst_path_ranking_never_picks_a_worse_worst_path_than_the_mean_winner`
locks the defining invariant: the worst-path winner's minimum is never below the
mean winner's minimum, and both rankings cover every config. Worst-path ranking
adds 10 test functions (18 collected) to
`tests/unit/research/test_cpcv_extremes.py`.

---

## 3. Computed-but-unenforced gates

**Finding: CONFIRMED — a real fail-open bug.**

`src/quant_fund/validation/gates.py:270-285` computed
`gates["hypothesis_family_split"]` and appended
`"hypothesis_family_split_missing"` to `reasons`. Neither `research_ok`
(`gates.py:317-323`) nor `promotion_decision` (`gates.py:305`) consumed it.

Reproduced before the fix, with a receipt-valid, run-id-bound, worktree-bound,
**non-synthetic** candidate carrying hypotheses filed under non-pre-registered
families:

```
gates.hypothesis_family_split = False
ok                            = True     <- fail-open
promote                       = True     <- fail-open
gates.promotion               = True
reasons                       = ['hypothesis_family_split_missing']   <- computed, unread
```

Why this matters: a notebook that registers hypotheses but files none under
`calibration` / `discovery` / `bound` cannot separate "we predicted this" from
"we found this". That is the exact distinction FDR control across
`_apply_family_fdr(hyps, "calibration")` / `"discovery"`
(`src/quant_fund/research/agent.py:1589-1591`) depends on. A candidate could
therefore reach `promote=True` with an uninterpretable multiplicity story.

**Fix** (`gates.py:269-334`): a `family_split_failed` flag now forces
`promo["promote"] = False` **before** `gates["promotion"]` is read, and is a
conjunct of `research_ok`. After the fix, the identical candidate:

```
gates.hypothesis_family_split = False
ok                            = False    <- enforced
promote                       = False    <- enforced
gates.promotion               = False
reasons                       = ['hypothesis_family_split_missing']
```

**Blast radius checked, and it is nil for real runs.** `HypothesisResult.family`
defaults to `"discovery"` (`src/quant_fund/research/agent.py:128`) and
`_build_hypotheses` explicitly applies both calibration and discovery FDR, so
every agent-produced notebook satisfies the gate. The gate bites only on
stripped, hand-authored, or legacy-migrated notebooks — the correct fail-closed
direction. Every pre-existing test passes `"hypotheses": []`, and an **empty**
hypothesis list is deliberately *not* a family-split violation (that case is
caught by the separate notebook/evidence gates), so no existing expectation
moved.

Regression tests (7 new collected, `tests/unit/pipeline/test_validate_gates.py`,
now 45 total):
`test_hypotheses_without_family_split_are_rejected` is parametrized over four
distinct failure shapes (families stripped, a non-pre-registered family,
`family=None`, non-dict rows) and asserts `ok is False`, `promote is False`,
`gates["promotion"] is False` and `promotion["promote"] is False` — **with a
perfect mean IC and a fully bound receipt**. `test_family_split_present_still_validates`
parametrizes each of the three valid families and confirms the gate still
promotes. `test_family_split_gate_blocks_even_when_other_gates_all_pass`
asserts all six sibling gates are green while the verdict is still rejected, so
the new conjunct is provably the deciding one.

Full lane result: **45 passed** in `test_validate_gates.py`.

### 3.1 Other computed-only values audited in the lane

Swept for the same anti-pattern (`reasons.append` / dict-stamp without a
consumer):

* `overfitting_diagnostics`'s `purge_disjoint` (`metrics/overfitting.py:444,548`)
  is stamped and then **validated** by `receipt_schema.py:160-166`
  (`backtest_overfitting_{key}` errors) — enforced at receipt level, not
  fail-open.
* `mcs_included` / `mcs_p_values` (`research/agent.py:489-492`) are checked by
  `research/catalog/hypotheses.py:377-383`, which asserts
  `mcs_n_included == len(included)`.
* `reality` verdicts are consumed: `reality/cli.py:166-176` exits non-zero
  unless `verdict == "pass"`, and `research/reality_sweep.py:866-868` gates on
  the same report.

No further fail-open gate found in the lane.

---

## 4. Conformal coverage calibration

**Finding: NOT AS DESCRIBED.** `src/quant_fund/conformal/**` **does not exist**
(glob returned 0 files). The conformal subsystem lives outside this lane, in
`src/quant_fund/models/`:

* `models/conformal.py` — `SplitCQR`, `SplitOneSided`, `AdaptiveConformal`
  (Gibbs–Candès ACI, `alpha_{t+1} = clip(alpha_t + gamma*(alpha - err_t))`),
  `LocalizedCQR`, per-stratum ACI.
* `models/conformal_dist.py`, `models/localized_conformal.py`,
  `models/enbpi.py`, `ConformalRiskControl`.

Because those paths are forbidden to this lane, **nothing was edited**. I
measured coverage read-only to report whether calibration holds:

| Regime | Nominal | Empirical | Verdict |
|---|---|---|---|
| Exchangeable, iid N(0,1), `SplitCQR` α=0.10, n_cal=n_test=400, 60 seeds | 0.9000 | **0.8986** (sd 0.0244) | Calibrated; \|Δ\|=0.0014, well inside the ±0.02 finite-sample band |
| AR(1) ρ=0.8 + 3× noise break on the test half | 0.9000 | 1.0000 for `SplitCQR`; ACI drove `alpha_t` from 0.1000 → 0.0010 | See caveat |

The exchangeable result is the meaningful one: `conformal_quantile` uses the
correct finite-sample \(\lceil (n+1)(1-\alpha)\rceil / n\) form, so coverage is
marginally conservative rather than under-covering, which is the right direction.

**Caveat on the non-exchangeable row, stated honestly:** in that scratch
experiment I widened the *base* band on the test half by the same 3× factor as
the noise, so the intervals stayed valid and coverage pinned at 1.0 with ACI
correctly shrinking `alpha_t` toward its floor. That demonstrates the ACI
recursion moves in the right direction under over-coverage, but it is **not** a
demonstration of under-coverage recovery, and it is **not** market evidence —
it is a synthetic correctness observation. A proper adversarial test (base band
held at the training scale while test noise triples, forcing genuine
under-coverage) belongs in the `models/` lane, which is forbidden here. Flagged
as follow-up rather than claimed as verified.

Existing coverage-calibration enforcement in-repo is already proper-score based
and outside this lane: `kupiec_pof` on conformal hit sequences
(`agent.py:774-779` files a `conformal` family hypothesis only when `kupiec_p`
is finite), plus `bench_crc` (`benches/intervals.py:651+`) reporting CRC
expected-hit risk with Kupiec POF.

---

## 5. Files changed

| File | Change |
|---|---|
| `src/quant_fund/validation/gates.py` | **Fix:** enforce `hypothesis_family_split` in `research_ok` and `promotion` |
| `src/quant_fund/validation/cpcv.py` | **New:** `fold_embargo_report`, `assert_fold_embargo`, `cpcv_folds_verified`, `rank_configs`, `ConfigRanking`, `_contiguous_blocks` |
| `tests/unit/pipeline/test_validate_gates.py` | +7 tests locking the enforced gate |
| `tests/unit/research/test_cpcv_extremes.py` | +20 test functions (33 collected): bar-level embargo assertion — 15 collected, incl. the hollowed-out-embargo detector — and worst-path ranking — 18 collected |
| `tests/property/test_backtest_overfitting.py` | +4 property tests: embargo assertion holds on every random geometry, reintroducing one embargoed bar is always caught, worst-path ≥ mean-winner invariant, constant-config agreement |
| `docs/SOTA/28-overfitting-promotion-findings.md` | this report |

`pyproject.toml`, `Makefile`, `uv.lock`, `README.md`, `.github/**`, `configs/**`,
`receipts/**`, `verifier/**`, `scripts/**`, and every other forbidden path were
not touched. No new dependency; `numpy`/`scipy`/`hypothesis` were already
present.

---

## 6. Gates

| Gate | Result |
|---|---|
| `ruff check` (lane) | clean |
| `ruff format --check` (lane) | clean (14 files already formatted) |
| `mypy src/quant_fund` | **Success: no issues found in 671 source files** |
| `pytest` lane + all `validation.cpcv` consumers (19 files, incl. `test_validate_gates.py` 45, `test_cpcv_extremes.py` 50, `test_backtest_overfitting.py` unit+property) | **146 passed, 0 failed** |

Out-of-lane failures observed in a broad `tests/unit tests/property` sweep
(`test_estimators_cov.py` VPIN/Corwin-Schultz, `test_fast_replay_byte_identity`,
`test_lob_invariants`, `test_pit_asof_never_future`, `test_score_propriety`,
hedge-lab disk-budget, torch-backend) were reproduced **identically on a
stashed clean baseline without any wave-28 edit** — pre-existing /
environmental (the hedge-lab ones are the real 100 GiB disk-budget guard firing
against a 1.4 TiB payload tree). `hedge_lab` imports only
`validation.walk_forward`, never the changed modules.

---

## 7. Reproducing

```bash
make lint
make typecheck
.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_validate_gates.py \
    tests/unit/research/test_cpcv_extremes.py \
    tests/unit/research/test_backtest_overfitting.py \
    tests/property/test_backtest_overfitting.py -q -p no:cacheprovider
```

Published anchors used for verification: Bailey & López de Prado, *JPM* 40(5)
2014 (DSR numerical example); Bailey & López de Prado, *J. Risk* 15(2) 2012 §3,
§5 (PSR, MinTRL frequency table); Bailey, Borwein, López de Prado & Zhu, *J.
Computational Finance* 20(4) 2017 (PBO/CSCV); López de Prado, *AFML* 2018
ch. 8, 12 (purge/embargo, CPCV path count); Lo 2002 (non-normal SR standard
error); Gibbs & Candès 2021 (ACI).
