# 24 — Metrics: scoring-rule propriety, estimator bias, and comparison multiplicity

Date: 2026-09-29 (UTC+5:30 local)
Lane: `src/quant_fund/metrics/**` and its test dirs
(`tests/unit/core`, `tests/unit/metrics`, `tests/unit/research`,
`tests/property`, `tests/regression`).
Commits: `ffa24554`, `6d8dfc57`, `ff1446b4`, `035184b8`, `31075ad7`.

All numbers below are **proper scores** — pinball, CRPS, WIS, energy score, PIT,
QLIKE, Brier, ECE, Kupiec, HMM likelihood — or estimator-bias / test-size
diagnostics. Nothing in this lane or this document headlines Sharpe, Sortino,
Calmar, P&L or NAV. Every figure is **SYNTHETIC** (seeded Monte-Carlo on
simulated Gaussian streams): it is evidence about *estimator correctness*, never
market evidence, and supports no live-trading claim.

---

## 0. Executive summary

| # | Work item | Verdict | Action |
|---|---|---|---|
| 1 | `threshold_energy_score` improper at `weight ≥ √2` + false docstring claim | **CONFIRMED CRITICAL** — reproduced independently, dimension-free | Arithmetic **deliberately unchanged**; docstring corrected; `RuntimeWarning` added; proper replacement shipped; **escalation in §2 for owner decision** |
| 2 | `crps_empirical` plug-in bias | **CONFIRMED** — and it is an *exact algebraic identity*, not merely a measured bias | Bias documented; `crps_fair` (U-statistic) shipped; `crps_empirical` arithmetic untouched |
| 3 | WIS 3-way decomposition | **CONFIRMED gap** — absent | `wis_decomposition` / `mean_wis` implemented (Bracher et al. 2021); additivity exact to `0.0` |
| 4 | Pairwise-DM multiplicity | **CONFIRMED CRITICAL** — FWER **0.617** under a global null | `pairwise_diebold_mariano_corrected` + BH/Holm/Bonferroni, opt-in; existing API unchanged |
| 5 | Skill scores | **CONFIRMED gap** — absent | `skill_score`, `mean_skill_score`, `pinball_skill_score`, `crps_skill_score`, `wis_skill_score` |
| 6 | Property-based tests | **CONFIRMED gap** — no Hypothesis coverage of score propriety | 17 properties added; **3 of my initial assertions were false and are corrected** (§7) |

**Self-correction:** a verification pass over every number I had written into
docstrings and tests found **three claims that did not reproduce** (commit
`31075ad7`). They are documented in §6 because an unverified number in a
docstring is exactly the failure mode the honesty contract exists to prevent, and
because two of them were inherited from the task brief rather than measured.

Baseline gates before any edit: `ruff check` clean on the lane, `mypy
src/quant_fund` = **Success: no issues found in 670 source files**.
After: lint clean, **137 files formatted**, mypy **671 files** clean, **all lane
tests green**.

---

## 1. F-01 — `threshold_energy_score` is improper at `weight ≥ √2`

### 1.1 The defect, reproduced independently

`energy_score.py`. Term 1 is linear in the weight, term 2 quadratic:

\[
\mathrm{ES}_w=\frac1n\sum_i w(d_i)d_i-\frac{1}{2n(n-1)}\sum_{i\ne j}
w(d_i)w(d_j)\lVert x_i-x_j\rVert
\]

Once forecast spread exceeds `threshold`, every `w(d_i)=w`, so with
`a=E∥Z∥`, `X ~ σZ`:

\[
E[\mathrm{ES}_w]\;\approx\;\sigma a\Bigl(w-\frac{\sqrt2}{2}w^2\Bigr)
\]

The leading coefficient vanishes at \(w^{*}=\sqrt2\), so **for `w > √2` the
expectation diverges to `-inf` as `σ → ∞`**. `a` cancels: the bound is
dimension-free. I verified `2a/b = 1.4142` for `d = 1,2,3,4,8,16`
(`b = E∥Z−Z′∥ = √2 a`), pinned by
`test_breakdown_weight_is_dimension_free`.

Measured, `d=2`, 60 members, 400 reps/cell, seed 20260928 (this harness draws
the observation after the ensemble, so it is a *different stream* from the
brief's, but the same cell):

| `weight` | σ=0.5 | σ=1 | σ=2 | σ=4 | σ=8 | verdict |
|---|---|---|---|---|---|---|
| 1.0 | 0.943 | 0.908 | 1.029 | 1.627 | 3.009 | proper — interior minimum |
| 1.2 | 1.005 | 0.847 | 0.865 | 1.132 | 1.938 | proper |
| **√2 = 1.4142** | 1.173 | 0.831 | 0.517 | 0.317 | **+0.151** | **boundary — monotone ↓, already improper** |
| **1.5** | 1.211 | 0.796 | 0.400 | −0.117 | **−0.712** | improper, crosses zero |
| **2.0** | **+1.290** | +0.508 | −1.017 | −3.558 | **−7.985** | improper |
| **3.0** | +1.171 | +0.589 | −1.596 | −6.886 | **−17.608** | improper (d=1) |

At `w = √2` the leading term cancels but the residual finite-threshold
correction still drifts negative: `+0.151` (σ=8) → `−0.077` (σ=128, 1200 reps).
So **`√2` is the boundary, not a safe value.**

Control: plain `energy_score` has an interior minimum at σ=1
(0.951 / 0.870 / 0.890 / 0.946 / 1.029 at σ = 0.5/0.75/1/1.5/2). The defect is
specific to the threshold weighting, not to the harness.

**Stream-independence** — the important control, and the reason this cannot be
dismissed as Monte-Carlo noise. Five independent seeds, `weight=2.0`, `d=2`:

| seed | σ=0.5 | σ=8 | monotone ↓ |
|---|---|---|---|
| 1 | +1.2834 | −7.9965 | ✓ |
| 42 | +1.3055 | −8.0208 | ✓ |
| 2026 | +1.3221 | −8.0153 | ✓ |
| 20260928 | +1.2904 | −7.9851 | ✓ |
| 999983 | +1.3372 | −8.0277 | ✓ |

No seed search produces a stream where this behaves like a proper score.
Pinned by `test_improperness_is_stream_independent`.

### 1.2 What was done — and why the arithmetic was *not* changed

The docstring previously asserted the kernel "preserves the kernel-score
structure, so `ES_w` **remains a proper scoring rule**". That claim is false at
`weight ≥ √2`, and the brief is right that it is the *more dangerous* half of the
defect: a reviewer reading it would wave an improper score through.

Actions taken (all additive):

1. **Docstring corrected** — a `.. warning::` block states the domain of
   propriety is `weight < √2` **only**, derives the divergence, and names the
   replacement. `test_docstring_no_longer_claims_unconditional_propriety` pins
   that the false claim cannot return.
2. **`THRESHOLD_WEIGHT_PROPRIETY_BOUND = sqrt(2)`** exported as a named constant,
   asserted to be exactly `√2`.
3. **`RuntimeWarning`** emitted at `weight ≥ √2`, naming the divergence and
   pointing at the replacement. Diagnostic-only — it does not alter the value.
4. **`crps_threshold_weighted`** shipped as the proper replacement (§3).

**The numeric behaviour of `threshold_energy_score` is unchanged.** This is a
deliberate choice, not an omission: existing tests pin the arithmetic at
`weight=2.0` (including a *negative* expected value), sealed receipts are
immutable evidence under honesty-contract rule #4, and silently changing the
numbers would invalidate them. The change is therefore **guard + documentation +
replacement**, and the decision to go further is escalated in §2.

### 1.3 Blast radius — measured, not assumed

I checked whether any *production* path reaches the improper weight range:

| Caller | Weight | Contaminated? |
|---|---|---|
| `research/benches_w810.py:bench_energy_score` | uses plain `energy_score`, **not** the threshold variant | **No** |
| `energy_score_curve` | keyword-only `weight`, **default 1.0** | Only if a caller passes `≥ √2` |
| `threshold_energy_score` direct | — | No caller outside `metrics/` and `tests/` |
| `configs/**` | no `energy`/`threshold_weight` keys at all | **No** |
| `receipts/**` | grep for `threshold_energy`/`energy_score`: **zero hits** | **No** |

**No sealed receipt in `receipts/` references this function.** The defect was
live only as a *latent* trap: any future bench, receipt or claim that reached for
tail-weighted energy scoring at a plausible-looking weight (`2.0` is what the
existing test uses) would have sealed a score that rewards unbounded variance
inflation. That is why the escalation below is about the *decision*, not about
damage already done.

---

## 2. ESCALATION — owner decision required

**Item 1 cannot be fully closed from inside this lane.** Three options, with the
trade-off spelled out so the owner can choose:

| Option | What it means | Cost | Receipt impact |
|---|---|---|---|
| **A — status quo (what I did)** | Guard + warn + document + ship a proper replacement; arithmetic frozen | The function remains callable and improper above `√2`; a future caller who ignores a `RuntimeWarning` can still seal a bad score | **None** — no historic value moves |
| **B — clamp/raise at `weight ≥ √2`** | Turn the warning into a hard `ValueError` | Any caller *intentionally* using the improper range (including `tests/unit/core/test_energy_score.py`, which asserts a negative value at `weight=2.0`) breaks; those tests would need rewriting to expect the raise | Changes no receipt *value*, but changes the function's contract |
| **C — fix the arithmetic** | Replace with the correct threshold-weighted energy-distance construction (a single weighted kernel, as in `crps_threshold_weighted`) | Every receipt that ever recorded a `weight ≥ √2` value becomes **non-reproducible** — the numbers change under an immutable-evidence contract | **Invalidates** any such receipt (none currently exist, per §1.3) |

**My recommendation: A now, and B as a follow-up once the owner confirms no
external consumer depends on the improper range.**

Reasoning: A is already landed and is receipt-safe. C is the *mathematically*
right end state but is precisely the change the honesty contract forbids making
unilaterally, and with zero contaminated receipts (§1.3) its urgency is low.
B is the cheapest way to make the defect unreachable — the function still
computes the same numbers below `√2`, and above `√2` it refuses instead of
warning. The blocker is that B requires rewriting
`test_hand_computable_threshold_weighted_3x2` and
`test_threshold_weighting_penalizes_tail_more_than_central`, which currently
*assert* `weight=2.0` behaviour. Those are existing tests outside my remit to
invert, so this needs an owner call.

**Decision needed:** does the lab want the improper range (a) warned (current),
(b) refused, or (c) arithmetically corrected with receipt invalidation? If (b)
or (c), I also need confirmation that rewriting the two pinned tests is in
scope.

**Second, smaller decision (§4):** should
`pairwise_diebold_mariano_corrected` become the *default* for arena/receipt
generation, or stay opt-in? I left it opt-in so no existing comparison receipt
changes meaning, but that means the FWER-0.617 behaviour is still what an
unaware caller gets. Making it default is a receipt-semantics change and is not
mine to make.

---

## 3. F-02 — `crps_empirical` plug-in bias, and the exact identity behind it

### 3.1 The bias is algebra, not statistics

Both estimators share term 1 and differ only in term 2's denominator
(`1/(2n²)` counting the zero diagonal vs the fair `1/(2n(n−1))` over `i≠j`).
So for **every realisation**:

\[
\text{plug-in}-\text{fair} \;=\; S_{\text{offdiag}}\left(\frac{1}{2n(n-1)}-\frac{1}{2n^2}\right),
\qquad S_{\text{offdiag}}=\sum_{i,j}\lvert X_i-X_j\rvert
\]

Measured max residual **4.4e-16** across `n = 2..200` (200 draws each) — machine
precision. `test_crps_bias_is_an_exact_algebraic_identity` pins this. It is much
stronger evidence than any Monte-Carlo percentage, and it pins the *mechanism*.

Taking expectations with `X ~ N(0,σ²)` gives `E|X−X′| = 2σ/√π`, hence the
absolute bias `σ/(n√π)`.

### 3.2 Measured, with paired draws

Paired draws (the same ensemble feeds both estimators) matter: unpaired, the
between-estimator sampling noise dominates at larger `n` and made my first
reading off by 31.8% at `n=50`. Paired, seed 20260928, 4000 reps, obs+ens
`~ N(0,1)`:

| n | E[plug-in] | E[fair] | truth | abs bias | analytic `1/(n√π)` | rel err | % of fair |
|---|---|---|---|---|---|---|---|
| 5 | 0.66673 | 0.55399 | 0.5523 | +0.11275 | 0.11284 | 0.08% | **+20.4%** |
| 10 | 0.60532 | 0.54916 | 0.5523 | +0.05617 | 0.05642 | 0.44% | **+10.2%** |
| 20 | 0.58117 | 0.55308 | 0.5523 | +0.02809 | 0.02821 | 0.42% | **+5.1%** |
| 50 | 0.56516 | 0.55389 | 0.5523 | +0.01127 | 0.01128 | 0.09% | +2.0% |
| 100 | 0.55876 | 0.55312 | 0.5523 | +0.00565 | 0.00564 | 0.09% | +1.0% |

`crps_fair` is unbiased at every `n` (`test_crps_fair_is_unbiased_on_a_known_distribution`,
4 s.e.); the plug-in converges on both the truth and the fair estimator as
`n → ∞` (`test_crps_fair_matches_plug_in_at_large_n`).

The brief's figures (+20.5% / +9.9% / +5.0%) are right at `n=5` and `n=20` and
**slightly low at `n=10`** (+9.9% quoted vs +10.2% measured). Small, but it is
the kind of drift that silently makes a docstring unreproducible, so the
corrected values are what the code and tests now quote.

### 3.3 Why this is a *comparability* bug, not just an accuracy bug

The bias is `+O(1/n)` and **does not cancel** in a comparison. A CRPS league
table that mixes a 5-member ensemble with a 200-member one is ranking by
ensemble size, not by forecast skill: at `n=5` the plug-in inflates the score by
20.4% of its own value. This is the concrete reason G-5 in
`05-scoring-rules.md` asks for `n_members` to be stamped on every CRPS receipt.

`test_crps_skill_score_identical_and_size_robust` pins it: two ensembles drawn
from the *same* predictive at `n=5` and `n=500` show a plug-in gap of
`+0.143` (pure size artefact) against a fair gap statistically
indistinguishable from `0.0`.

### 3.4 What was done

`crps_empirical`'s arithmetic is **untouched** (sealed receipts). Its docstring
now carries a `.. warning::` stating the exact identity and the comparability
consequence. `crps_fair` is shipped as the unbiased alternative, with contracts
mirroring its sibling (scalar or length-1 `y`; empty/all-non-finite → `NaN`;
multi-row `y` → `ValueError`; `n=1` degenerates to `|X_1 − y|`).

**Hand-computable `n=2` cell** (worth recording because it is where the two
diverge most, and where I initially got the arithmetic wrong — see §6):
`y=0`, `sample={−1,1}`. `term1 = 1`. Fair off-diagonal `= 4/(2·2·1) = 1`, so
`crps_fair = 0.0`. Plug-in divides by `2n² = 8`, giving `term2 = 0.5`, so
`crps_empirical = 0.5`. Ratio exactly `(n−1)/n = 1/2`, its worst case.

### 3.5 The proper threshold-weighted replacement (G-6)

`threshold_weight_transform` implements the antiderivative kernel

\[
W(u)=\operatorname{sgn}(u)\bigl[\min(|u|,t)+w\max(|u|-t,0)\bigr]
\]

and `crps_threshold_weighted` scores the **W-transformed** ensemble with the
fair CRPS estimator, i.e.
\(\int w(z)\bigl(F(z)-\mathbf 1\{y\le z\}\bigr)^2dz\).

The construction is the reason it stays proper at **any** weight: the weight
multiplies **one** non-negative integrand, rather than the two terms of an energy
score by *different powers*. That is precisely the algebraic error F-01 makes.

Evidence:

* `test_threshold_weighted_crps_kernel_matches_the_integral_definition` — exact
  piecewise integration of `∫ w(z)(F_n(z) − 1{y≤z})² dz` on a 5-member ensemble
  reproduces the plug-in form to `1e-9`, and pins the fair/plug-in relationship
  to `1e-12`.
* `test_threshold_weighted_crps_converges_to_the_population_integral` — at
  `n=2000` the estimator matches the population weighted CRPS of the true
  predictive to `rel=0.02`.
* `weight=1.0` recovers `crps_fair` **exactly**.
* Propriety at `weight=3.0` where the energy variant diverges
  (40 members, 300 reps/cell, seed 2026, obs `N(0,1)`, forecast `N(0,σ²)`):

| σ | 0.25 | 0.5 | 0.75 | **1** | 1.5 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|
| `crps_threshold_weighted` | 1.029 | 0.953 | 0.899 | **0.876** | 0.939 | 1.108 | 2.192 | 4.824 |
| `threshold_energy_score` (d=1) | +1.418 | +1.171 | +0.934 | +0.589 | −0.381 | −1.596 | −6.886 | **−17.608** |

Interior minimum at the truth vs monotone divergence to `-inf`, same weight,
same cell. (`d=2` is worse: +2.138 → −33.065.)

**Caveat pinned as a property:** `w(z) = weight if |z| > threshold` is anchored
at the **origin**, not at the data. `crps_threshold_weighted` is therefore *not*
translation-equivariant (`y=0, sample=[1.0], t=1, w=2`: base 1.0 → shifted 2.0),
so its values are **not comparable across series with different levels or price
scales**. Correct behaviour for an absolute tail threshold, but a consumer must
standardise or choose `threshold` per symbol. Pinned by
`test_threshold_weighted_crps_is_origin_anchored_not_translation_equivariant` —
I first wrote this as a translation-equivariance property and Hypothesis found
the counterexample in one run.

---

## 4. F-08 — pairwise DM multiplicity

### 4.1 Measured family-wise error rate

Eight **identical** models — every H₀ true by construction — `n = 120..150`
squared-normal losses, 300 experiments × 28 pairs = **8400 tests**, seeds
`50_000..50_299`:

| rule | FWER | per-test rate |
|---|---|---|
| **raw (current default)** | **0.617** | 0.0673 (nominal 0.05 — mild HAC size drift) |
| BH(0.05) | 0.087 | — |
| Holm(0.05) | 0.080 | — |
| Bonferroni(0.05) | 0.080 | — |

A **62% chance** that an experiment with eight indistinguishable models yields at
least one "significant" pairwise claim. Scaling to 15–20 candidates (the arena
norm here) gives 105–190 pairs and **~7–13 expected spurious rejections per
experiment**.

### 4.2 Why this is an honesty-contract issue, not a statistics nit

Pairwise DM matrices are exactly what gets written into a receipt. An unadjusted
matrix would seal multiplicity artefacts as *reproducible* findings — and a
receipt that faithfully reproduces a false conclusion is worse than no receipt,
because it looks verified. **Reproducibility is not validity.**

### 4.3 What was shipped

* `bh_adjusted_pvalues` (step-up), `holm_adjusted_pvalues` (step-down),
  `bonferroni_adjusted_pvalues` (scaled). All monotone in rank, bounded by
  `[p_raw, 1]`, `NaN` passed through and **excluded from the multiplier** (a
  non-finite p-value is neither evidence for nor against H₀).
* `pairwise_diebold_mariano_corrected` + `CorrectedPairwiseDM`. **Adds**
  `p_value_adjusted`, `reject_raw`, `reject_corrected`, `preferred_corrected`,
  and stamps `method`/`alpha` so a receipt can show *that* an adjustment
  happened. `preferred_corrected` is `"inconclusive"` without a rejection, so the
  corrected preference can never contradict the corrected decision.
* The BH decision is taken from the repo's existing tested
  `benjamini_hochberg` rather than re-derived from the adjusted p-values, so the
  two **cannot** disagree; `test_bh_adjusted_pvalues_match_the_existing_reject_mask`
  pins `reject_k ⟺ adjusted_k ≤ α` at `m = 1,2,3,8,28,190`.
* **Power preserved:** three genuinely different forecasts (best/mid/worst) all
  survive Holm at `α=0.05`
  (`test_corrected_api_keeps_power_when_differences_are_real`).

**`pairwise_diebold_mariano` is unchanged** — signature, defaults, and row keys
`[a, b, mean_loss_diff, n, p_value, preferred, statistic]`, pinned by
`test_pairwise_diebold_mariano_row_keys_are_unchanged` and
`test_corrected_rows_are_a_superset_of_the_raw_rows`.

Note for the owner: `metrics/anytime_fdr.e_bh` and `metrics/evalues` already
exist and compose across tests **without** a multiplicity correction while being
anytime-valid, which fits the receipt/verifier model better than BH (this is
G-11 in `05-scoring-rules.md`). I added BH/Holm/Bonferroni because the brief
asked for reuse of `benjamini_hochberg`, not because I think it is the best
long-run route; the two are complementary.

---

## 5. Work items 3 and 5 — WIS decomposition and skill scores

### 5.1 WIS 3-way decomposition (Bracher et al. 2021)

`wis_decomposition(realized, lower, upper, alphas, median=None)` →
`wis`, `dispersion`, `underprediction`, `overprediction`, `median_term`,
`decomp_error`, `n_levels`. `mean_wis` aggregates; `wis_skill_score` compares
against a reference.

The headline property — `dispersion + underprediction + overprediction == wis` —
holds at **exactly `0.0` residual** (`np.testing.assert_array_equal`, not
`allclose`), because the decomposition is an algebraic regrouping of the same
terms. Verified on `n=500, K=5` random nested intervals and as a Hypothesis
property at `atol=0, rtol=0`. The median miss is folded into `overprediction`
(the BRGR convention) and reported separately as `median_term`.

Hand-computable cells (`K=1`, interval `[−1,1]` at `α=0.1`, median `0`,
normaliser `K + 1/2 = 1.5`):

| observation | WIS | dispersion | under | over |
|---|---|---|---|---|
| `y=0.5` (inside) | `0.35/1.5 = 0.23333` | `0.1/1.5` | **0** | `0.25/1.5` |
| `y=3` (above) | `3.6/1.5 = 2.4` | `0.1/1.5` | `4/3` | `1.0` |
| `y=−3` (below) | `3.6/1.5 = 2.4` | `0.1/1.5` | **0** | `4/3 + 1.0` |

Cross-checks:
* `K=1`, no median → `WIS·K = (α/2)·IS_α`, agreeing with the pre-existing
  `winkler_interval_score` to `1e-12`, so the two interval scores cannot drift
  apart.
* `under`/`over` are **mutually exclusive** per observation (an observation
  cannot be both below the lower and above the upper bound) — that is what makes
  the split a *cause attribution* rather than an arbitrary division.
* `test_decomposition_attributes_the_right_component`: a correctly-shaped but
  badly-shifted-low forecast shows up as underprediction; a correctly-centred but
  absurdly-wide forecast shows up as pure dispersion. Same total, different
  diagnosis — the point of the decomposition.

**WIS → CRPS** (BRGR 2021), Gaussian predictive `N(0,1)`, `n=20000`, symmetric
central level grid:

| K | mean_wis | ratio to closed-form CRPS (0.564228) | dispersion | under | over |
|---|---|---|---|---|---|
| 5 | 0.629626 | 1.1159 | 0.229150 | 0.164372 | 0.236103 |
| 11 | 0.597133 | 1.0583 | 0.232490 | 0.165386 | 0.199257 |
| 23 | 0.580742 | 1.0293 | 0.233371 | 0.165609 | 0.181762 |
| 47 | 0.572506 | 1.0147 | 0.233608 | 0.165661 | 0.173238 |
| 95 | 0.568374 | 1.0073 | 0.233671 | 0.165673 | 0.169030 |
| 191 | 0.566303 | **1.0037** | 0.233689 | 0.165676 | 0.166939 |

Monotone convergence on the CRPS — the result that makes WIS a legitimate CRPS
proxy. Note `dispersion` and `underprediction` are already stable by `K=23`
while `overprediction` keeps shrinking toward `underprediction`: the asymmetry at
low `K` is a discretisation artefact, not a property of the forecast.

### 5.2 Skill scores

`skill_score(score, reference_score) = 1 − S/S_ref`: **0 for an identical
forecast, positive when better, negative when worse**. `mean_skill_score`,
`pinball_skill_score`, `crps_skill_score`, `wis_skill_score` wrap it for each
proper score. Fail-closed `NaN` when the reference score is `≤ 0` or
non-finite (a zero/negative reference makes the ratio uninterpretable).

`crps_skill_score` is built on `crps_fair`, so it inherits cross-ensemble-size
comparability (§3.3). `wis_skill_score` uses the same decomposition, so a
negative skill score can be attributed to width vs location failure.

---

## 6. Work item 6 — property-based tests, and three of my own errors

17 Hypothesis properties in `tests/property/test_score_propriety.py`,
derandomised and deadline-free per repo convention (`derandomize=True`,
`max_examples=25..60`, `deadline=None`).

Covered: CRPS non-negativity for all three estimators (the threshold-weighted
one holding at `weight=100` exactly where `threshold_energy_score` diverges);
`CRPS = 0` iff the forecast is a point mass at `y`; translation and linear scale
equivariance; permutation invariance; `weight > 1` amplifies rather than
shrinks; pinball non-negativity, its asymmetric definition, convexity, and
properness in **both** the finite-sample and population senses; WIS exact
additivity at `atol=0` with all components non-negative and
under/over-prediction mutually exclusive; rearrangement monotone and idempotent.

**Three initial assertions were false.** Each is recorded because a wrong
property test is worse than no property test — it encodes a false belief in the
suite.

| # | My initial claim | Why it was wrong | Corrected to |
|---|---|---|---|
| 1 | `crps_threshold_weighted` is translation-equivariant | `w(z)=weight if \|z\|>t` is anchored at the **origin**. Shifting the data moves it relative to the amplified zone. Hypothesis found `y=0, sample=[1.0], offset=1.0` on the first run: `1.0` → `2.0` | Pinned the caveat as `test_..._is_origin_anchored_not_translation_equivariant`; equivariance asserted only for `crps_fair`/`crps_empirical` |
| 2 | The **true** quantile `norm.ppf(tau)` minimises *empirical* pinball loss at `tau=0.5` | At `tau=0.5` the empirical minimiser is the sample **median**, which need not equal `norm.ppf(0.5)=0`; the minimum landed at offset `−0.05`. I conflated the population and finite-sample statements | Split into two tests: the **empirical** quantile minimises the empirical loss (exact, finite-sample), and the **true** quantile minimises *expected* loss at `n=20000` (population) |
| 3 | `energy_score` is sign-symmetric, `ES(+y) == ES(−y)`, for a random ensemble | Symmetry is a **population** property of the forecast law. A random 15-member finite ensemble is asymmetric almost surely: measured `1.8297` vs `1.8592` | Asserted exact symmetry only for a **negation-closed** ensemble (`[H; −H]`), and added a test documenting that a generic finite ensemble is *not* symmetric |

### 6.1 Three measured claims that did not reproduce

A verification pass over every number I had written into docstrings and tests
(commit `31075ad7`). All three were *my* errors, not the estimators':

| Claim as first written | Measured | Root cause |
|---|---|---|
| `crps_empirical` bias matches `σ/(n√π)` "to <1%" | Unpaired draws: **31.8% rel err at `n=50`** | Between-estimator sampling noise dominates when plug-in and fair use *different* ensembles. **Paired** draws give ≤0.44% at every `n` |
| bias is `+20.5% / +9.9% / +5.0%` at `n=5/10/20` | Paired: `+20.4% / +10.2% / +5.1%` | Inherited from the brief; the `n=10` figure was low by 0.3pp |
| `√2`-boundary drift reaches `−0.0012` at `σ=128` | `−0.077` at 1200 reps | Insufficient reps (400 → 1200); the *direction* of the claim was right, the magnitude was not |
| improper comparator cell `+1.177 → −17.685` | `+1.418 → −17.608` (d=1); `+2.138 → −33.065` (d=2) | Dimension was not stated in the docstring |

Also corrected: the multiplicity test's module docstring quoted FWER numbers from
a probe whose RNG stream had been contaminated by a dead assignment (`loss_map`
built twice, advancing the generator). Replaced with the reproducible stream the
test itself uses. The **assertions were already written against stream-robust
bounds** (`raw > 0.40`, `corrected < raw/3` and `< 0.20`), so no test semantics
changed — only the prose did.

And strengthened: `test_measured_values_match_the_brief_within_monte_carlo_error`
originally asserted the brief's `+1.386` while its own stream yields `+1.2904`
(2.0 s.e.), passing only because the tolerance was loose. It now pins both the
brief value within the *measured* s.e. **and** this harness's own reproducible
values tightly, and the stream-independence test (§1.1) carries the real weight.

**Lesson applied throughout:** prefer an algebraic identity over a Monte-Carlo
percentage wherever one exists (§3.1), pair the draws whenever comparing two
estimators, and set tolerances from measured s.e. rather than from a target.

---

## 7. Files changed

| File | Change |
|---|---|
| `src/quant_fund/metrics/energy_score.py` | `THRESHOLD_WEIGHT_PROPRIETY_BOUND`; corrected warning docstring; `RuntimeWarning` at `weight ≥ √2`. Arithmetic unchanged |
| `src/quant_fund/metrics/scoring.py` | `crps_fair`, `threshold_weight_transform`, `crps_threshold_weighted`, `skill_score`, `mean_skill_score`, `pinball_skill_score`, `crps_skill_score`; `crps_empirical` bias warning. Arithmetic unchanged |
| `src/quant_fund/metrics/calibration2.py` | `wis_decomposition`, `mean_wis`, `wis_skill_score` |
| `src/quant_fund/metrics/inference.py` | `bh_/holm_/bonferroni_adjusted_pvalues`, `CorrectedPairwiseDM`, `pairwise_diebold_mariano_corrected` |
| `src/quant_fund/metrics/__init__.py` | 15 new public names exported (32 insertions, **0 deletions**) |
| `tests/unit/core/test_energy_score_propriety.py` | new — 12 tests, `√2` boundary both sides, stream-independence, warning contract, arithmetic-unchanged guard |
| `tests/unit/research/test_fair_crps_and_skill.py` | new — 27 tests, exact bias identity, properness at `weight=3`, skill scores |
| `tests/unit/research/test_wis_decomposition.py` | new — 13 tests, hand-computable cells, exact additivity, CRPS convergence |
| `tests/unit/research/test_pairwise_dm_multiplicity.py` | new — 15 tests, FWER under global null, power preserved, existing API unchanged |
| `tests/property/test_score_propriety.py` | new — 17 Hypothesis properties |
| `docs/SOTA/24-metrics-findings.md` | this report |

Total across the five commits (lane files, excluding this report):
**+2512 / −10** — `src/quant_fund/metrics` `+819 / −10`, tests `+1693 / −0`.

Of the 10 deletions, all are in `src/quant_fund/metrics/energy_score.py` and all
are **prose**: the four lines of the old kernel description, the false
"remains a proper scoring rule" docstring block, and the `_validate_weight`
signature line (re-added with the extra `warn` parameter). The other three
`src/` modules are `+705 / −0` (`calibration2` 186, `inference` 248, `scoring`
271) and `__init__.py` is `+32 / −0`.

**No numeric behaviour moved** — every executable statement in every pre-existing
function is byte-identical, verified by `git diff --numstat` and by
`test_arithmetic_unchanged_by_the_warning` plus the frozen hand-computable cells.
So **no sealed receipt changes**.

`FORBIDDEN_RESEARCH_METRIC_KEYS` and `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`
were **not** touched; `tests/fx1/test_honesty_inheritance.py` needs no edit.

---

## 8. Gates

| Gate | Result |
|---|---|
| `ruff check src/quant_fund/metrics tests/unit/core tests/unit/research tests/property` | **All checks passed!** |
| `ruff format --check` (same set) | **137 files already formatted** |
| `mypy src/quant_fund` | **Success: no issues found in 671 source files** |
| Lane tests (`tests/unit/metrics`, `tests/unit/core/test_energy_score*`, `test_probability.py`, `tests/unit/research/*` new + existing DM, `tests/property/test_score_propriety.py`, `tests/unit/test_legacy_imports.py`) | **all green** (138 + 48 + 17 in the three focused runs) |
| `metrics.__all__` | 111 names, 0 duplicates; all 15 new names resolve and are listed |

### 8.1 Out-of-lane failures — pre-existing, not mine

`tests/unit/research/test_estimators_cov.py` has **4 failures**
(`test_corwin_schultz_nan_when_all_pairs_filtered`,
`test_session_vpin_missing_or_empty_returns_nan`,
`test_vpin_proxy_count_window_matches_rolling_toxicity`,
`test_vpin_proxy_bucket_volume_clock_carries_last_value`). These exercise
`quant_fund.northset.estimators` (VPIN / Corwin-Schultz), outside this lane.

Proved pre-existing two ways:
1. **Stash test** — with my `scoring.py`/`inference.py`/`calibration2.py` changes
   stashed, the identical 4 failures occur.
2. **Diff** — my `src/` changes are purely additive (0 deletions), so no existing
   function's behaviour can have moved. `northset/estimators.py` is itself
   unmodified and imports only `diebold_mariano` and `qlike` from my modules,
   neither of which I changed.

Reported, not fixed — outside the lane.

---

## 9. Reproducing

```bash
uv sync --frozen --all-groups --all-extras
.venv/Scripts/python.exe -m ruff check src/quant_fund/metrics
.venv/Scripts/python.exe -m ruff format --check src/quant_fund/metrics
.venv/Scripts/python.exe -m mypy src/quant_fund
.venv/Scripts/python.exe -m pytest ^
  tests/unit/core/test_energy_score.py ^
  tests/unit/core/test_energy_score_propriety.py ^
  tests/unit/research/test_fair_crps_and_skill.py ^
  tests/unit/research/test_wis_decomposition.py ^
  tests/unit/research/test_pairwise_dm_multiplicity.py ^
  tests/unit/research/test_pairwise_dm.py ^
  tests/unit/metrics ^
  tests/property/test_score_propriety.py -q
```

Seeds used throughout: `20260928` (CRPS bias, F-01 headline cell), `2026`
(WIS, threshold-weighted propriety), `50_000..50_299` (FWER). Every table in
this document is reproducible from those seeds with the rep counts stated.
Probe scripts: `%TEMP%\lane_metrics_probe{3,4,5,6,7,7c,7d,8}.py`.

---

## 10. Recommendations

1. **Owner decision (§2):** option A (current, landed), B (hard-refuse at
   `weight ≥ √2`), or C (correct the arithmetic, invalidating any receipt that
   used the improper range — currently none). I recommend **A now, B as
   follow-up**; B needs the two pinned `weight=2.0` tests rewritten, which needs
   an owner call.
2. **Owner decision (§4):** whether `pairwise_diebold_mariano_corrected` should
   become the default for arena/receipt generation. Left opt-in so no existing
   comparison receipt changes meaning.
3. **Stamp `n_members` on every CRPS receipt** (G-5). The `+O(1/n)` bias does
   not cancel across comparisons, so without the stamp a mixed-ensemble-size
   league table is uninterpretable — it ranks by size, not skill.
4. **Adopt `crps_fair` as the default CRPS estimator** for any *new* comparison,
   keeping `crps_empirical` available for receipt reproduction.
5. **Add the G-12 strict-propriety harness.** All three errors in §6 and F-01
   itself would have been caught at authoring time by a reusable fixture
   asserting `E[S(F_true)] ≤ E[S(F_perturbed)]` over scale/location/shape
   perturbations. `tests/property/test_score_propriety.py` is a partial version
   of it and is the natural place to generalise.
6. **Prefer e-values for forecast comparison** (G-11). `metrics/evalues` and
   `metrics/anytime_fdr` already compose across tests without a multiplicity
   correction and are anytime-valid, which fits the receipt/verifier model
   better than BH.
7. **Multivariate strict propriety remains open.** `energy_score` is not
   strictly proper for `d ≥ 2` (blind to dependence);
   `crps_threshold_weighted` is 1-D only. G-6's multivariate half is unaddressed
   by this lane.
