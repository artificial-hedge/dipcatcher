# 05 — Proper scoring rules & calibration diagnostics depth (SOTA lane)

**Status:** research mapping + implementation audit, 2026-09-28. Sources: web
survey (2026-09-28) + in-tree inspection of `src/quant_fund/metrics/`
(`scoring.py`, `probability.py`, `energy_score.py`, `calibration2.py`,
`density_forecast.py`, `var_backtest.py`, `es_backtest.py`, `forecast_eval.py`,
`inference.py`, `vol_eval.py`), `src/quant_fund/models/qrf.py`,
`src/fx1/eval/calibration_eval.py`, `src/fx1/honesty.py`,
`src/quant_fund/research/benches_w810.py`. Every quantitative claim below was
reproduced numerically against the installed package; probe scripts and their
raw output are listed in §7.

Honesty contract: **proper scores only** — pinball, CRPS, PIT, QLIKE, Brier,
ECE, Kupiec, log score, energy score, variogram score, Fissler–Ziegel, HMM
likelihood. No Sharpe/Sortino/Calmar/P&L/NAV anywhere in this lane. SYNTHETIC
results below (all simulations are seeded synthetic draws) are *correctness*
evidence about estimator behaviour, never market evidence. No live-trading
claim is made or implied.

---

## 0. Verdict up front

The scoring layer is **broad and mostly well-built**, with three problems that
matter and a long tail of convention hazards.

**Materially wrong (one of these can invert a research conclusion):**

1. **`threshold_energy_score` is not a proper scoring rule for `weight ≥ √2`,
   despite a docstring that asserts it is.** Its expected value diverges to
   −∞ as forecast dispersion grows, so it *rewards* unbounded variance
   inflation. The breakdown weight is **exactly `√2`, dimension-free**.
   Measured: at `weight=2.0` the score goes 1.386 → −8.007 as σ goes
   0.5 → 8. The existing test suite pins the current arithmetic at `weight=2.0`
   (asserting a *negative* energy score) and even documents the quadratic
   blow-up as intended behaviour. (§3 F-01)
2. **`crps_empirical` uses the biased plug-in form** (`1/(2n²)` including the
   diagonal) while its sibling `energy_score` in the same package uses the fair
   U-statistic (`1/(2n(n−1))`). Bias is exactly `E|X−X′|/(2n)`: **+20.5% at
   n=5, +9.9% at n=10, +5.0% at n=20**. Two sibling estimators of the same
   object disagree, and the biased one is the CRPS. (§3 F-02)
3. **The forecast-comparison layer has no multiplicity control, and the
   small-sample-corrected test is unavailable for proper scores.**
   `pairwise_diebold_mariano` emits raw p-values (FWER = 0.62 with eight
   *identical* models) even though `benjamini_hochberg` lives in the same file;
   `hln_test` hardcodes squared-error loss so CRPS/QLIKE/pinball differentials
   cannot get the HLN correction at all; `fluctuation_test` has empirical size
   **0.306** against a nominal 0.10. (§3 F-08, F-09, F-11)

**Hazardous but arithmetically fine:** an unguarded `alpha` convention
collision across five exported functions (§3 F-03), `1.0 - cdf` p-value
precision loss in ~20 call sites (§3 F-06), and a `qlike` name collision with
two *opposite* edge-case policies (§3 F-07).

**Confirmed correct (positive controls — do not "fix" these):**
`energy_score` (fair, proper), `fissler_ziegel_loss` (proper; verified by
analytic derivation *and* a 25×25 perturbation grid), the fx1 oracle-binned
ECE (deterministic and genuinely noise-free, as its docstring claims),
`expected_calibration_error`'s bin-edge parenthesisation (AST-verified), and
`crps_from_quantiles`'s cell widths (uniform, not ragged).

**Two hypotheses investigated and rejected.** They are recorded here because a
future reader will plausibly re-suspect them: (a) an operator-precedence bug in
`expected_calibration_error`'s `sel = (p >= edges[i]) & (p < edges[i+1] if ... else ...)`
line — the parentheses *are* present and the AST parses as intended (§3 NF-1);
(b) improperness of `fissler_ziegel_loss` — an early test scaled VaR and ES
together, which moves the pair to a different quantile level and is not a
propriety test (§3 NF-2).

---

## 1. Per-metric best practice: formulas + citations

### 1.1 Framework — what "proper" means

A scoring rule `S(F, y)` is **proper** if the true distribution minimises
expected score, `E_{y~G}[S(G,y)] ≤ E_{y~G}[S(F,y)]` for all `F`, and
**strictly proper** if the inequality is strict for `F ≠ G`. Gneiting & Raftery
(2007) is the reference taxonomy: bin/quantile/distribution scores, Savage
representations, Bregman and kernel scores. Two consequences drive this lane:

- **Only strictly proper scores admit honest model comparison.** A merely
  proper score can be minimised by a *set* of distributions, so "lower score"
  does not imply "closer to truth". F-01 is a failure of even propriety.
- **Finite-sample estimators of a proper score need not be unbiased.** A
  plug-in ensemble estimator with `O(1/n)` bias still ranks correctly in the
  limit but distorts finite-`n` league tables and, critically, **the bias
  differs per model when ensemble sizes differ** — so it does not cancel in a
  comparison. That is F-02.

### 1.2 Pinball / quantile score

\[
\mathrm{QS}_\tau(q,y) = \max\{\tau (y-q),\; (\tau-1)(y-q)\}
= (\tau - \mathbf{1}\{y<q\})(y-q).
\]

Strictly proper for the τ-quantile (Koenker & Bassett 1978; Gneiting 2011
"Making and evaluating point forecasts"). Repo: `scoring.pinball_loss`,
`scoring.mean_pinball`, `calibration2.pinball_score` — **correct**, with a
clean `(0,1)` guard on τ and honest NaN on empty-after-mask.
`quantile_crossing_rate` + `rearrange_quantiles` implement Chernozhukov,
Fernández-Val & Galichon (2010) monotone rearrangement; correct and useful.

### 1.3 CRPS

**Definition (probabilistic-integral form):**

\[
\mathrm{CRPS}(F,y) = \int_{-\infty}^{\infty}\bigl(F(z)-\mathbf{1}\{y \le z\}\bigr)^2\,dz .
\]

**Quantile representation** (the one the repo uses; Gneiting & Raftery 2007 §4.2):

\[
\mathrm{CRPS}(F,y) = 2\int_0^1 \mathrm{QS}_\tau\bigl(F^{-1}(\tau),y\bigr)\,d\tau .
\]

**Closed forms.** Gaussian (Grünewald et al.; standard):

\[
\mathrm{CRPS}\bigl(N(\mu,\sigma^2),y\bigr)
= \sigma\Bigl[z\bigl(2\Phi(z)-1\bigr) + 2\varphi(z) - \tfrac{1}{\sqrt{\pi}}\Bigr],
\quad z = \tfrac{y-\mu}{\sigma}.
\]

Repo `crps_gaussian`: **correct**. Gaussian mixtures: Grimit, Gneiting,
Ruan & Milks (2006) give the closed form with the `Φ`/`φ` cross terms; repo
`crps_gaussian_mixture` implements it and is the right choice over sampling for
the repo's mixture-density heads. Student-t: Jordan, Krüger & Lerch (2019)
("Evaluating predictive count data distributions in retail sales forecasting",
*JRSS-A*) give the closed form via the incomplete beta; **the domain of
validity is ν > 1** (finite mean). Repo `crps_student_t` restricts to `ν > 2`,
documented as a deliberate stricter lab contract (see F-15).

**Ensemble estimators — the part that matters.** For draws `X₁..Xₙ`:

| form | formula | bias |
|---|---|---|
| plug-in | `\frac{1}{n}\sum_i\lvert X_i-y\rvert - \frac{1}{2n^2}\sum_{i,j}\lvert X_i-X_j\rvert` | `+\frac{E\lvert X-X'\rvert}{2n}` |
| **fair / unbiased U-statistic** | `\frac{1}{n}\sum_i\lvert X_i-y\rvert - \frac{1}{2n(n-1)}\sum_{i\ne j}\lvert X_i-X_j\rvert` | 0 |

The plug-in form is the "empirical CDF plugged into the CRPS integral"; it is
*positively* biased because the diagonal `i=j` terms contribute zero to the sum
but `n` counts to the denominator. Northrop (2022) and the `scoringRules`
literature (Jordan, Krüger & Lerch 2019, *JSS*) both recommend the fair form.
For `X ~ N(0,σ²)`, `E|X−X′| = 2σ/√π`, so bias `= σ/(n√π)` — verified
numerically to within 0.3% (§3 F-02).

**Weighted Interval Score (WIS)** — Bracher, Ray, Gneiting & Reich (2021),
"Evaluating epidemic forecasts in an interval format", *PLOS Comp. Biol.*
17(6):e1008618. For a predictive median `m` and `K` central intervals with
levels `α_k`, `w_k = α_k/2`, `w_0 = 1/2`:

\[
\mathrm{WIS}_{\alpha_{0:K}}(F,y) = \frac{1}{K+\tfrac12}\Bigl(
w_0\lvert y-m\rvert + \sum_{k=1}^{K} w_k\,\mathrm{IS}_{\alpha_k}(F,y)\Bigr),
\]
\[
\mathrm{IS}_\alpha(F,y) = (u-l) + \tfrac{2}{\alpha}(l-y)\mathbf{1}\{y<l\}
+ \tfrac{2}{\alpha}(y-u)\mathbf{1}\{y>u\}.
\]

WIS → CRPS as `K → ∞`, and decomposes **exactly** into
`dispersion + overprediction + underprediction`. This decomposition is the
single most useful missing diagnostic in the repo (§4 G-2): it turns "CRPS got
worse" into "the forecast became under-dispersed" vs "it shifted". Repo status:
`calibration2.winkler_interval_score` is the Winkler (1994) interval score
(i.e. `IS_α` unnormalised) — correct but standalone; there is **no WIS
aggregation and no dispersion/over/under decomposition**.

### 1.4 Energy score and kernel scores (multivariate)

\[
\mathrm{ES}(F,y) = E_F\lVert X-y\rVert - \tfrac12 E_F\lVert X-X'\rVert ,
\]

strictly proper for distributions with finite first moment when the dimension
is `d ≥ 1`… **except** that it is only proper (not strictly) for `d ≥ 2` with
respect to *dependence*: Székely & Rizzo (2013) show ES cannot distinguish all
joint distributions in `d ≥ 2` (it is blind to certain copula differences).
Gneiting & Raftery (2007) §4.2 establish the kernel-score framework: `S` is a
strictly proper kernel score iff the kernel is **characteristic**; the
energy/distance kernel is characteristic only up to the `d ≥ 2` limitation
above. Best practice: report ES *alongside* the **variogram score** (Scheuerer
& Hamill 2015, *Mon. Wea. Rev.* 143(7):2439–2454), which is sensitive to
dependence structure that ES misses:

\[
\mathrm{VS}_p(F,y) = \frac{1}{d(d-1)}\sum_{i=1}^{d}\sum_{j\ne i}
\Bigl( E_F\lvert X_i-X_j\rvert^{p} - \lvert y_i-y_j\rvert^{p}\Bigr)^2 .
\]

Note the `1/(d(d−1))` normalisation — it is what makes VS comparable across
dimensions (F-13). For genuinely strict multivariate propriety use the
**multivariate CRPS / kernel score with a characteristic kernel** (e.g. the
Gaussian RBF kernel, or the `d=1`-marginal sum); see Ziegel (2016) and Mühlemann
& Ziegel (2021) on strict propriety of ES under additional assumptions.

**Threshold weighting.** Gneiting & Ranjan (2013) and Stroud & Stein (2020)
*do* construct proper threshold-weighted scores — but only via a **kernel
reweighting that preserves negative definiteness**, i.e.
`\mathrm{ES}_w = \int w(x)\bigl(...\bigr)` derived from the weighted
energy-distance kernel, not by multiplying the two terms by different powers of
the weight. The repo's multiplicative `w(d_i)` on term 1 vs `w(d_i)w(d_j)` on
term 2 is not that construction (F-01).

### 1.5 QLIKE and volatility-loss robustness

\[
\mathrm{QLIKE}(\hat\sigma^2,\sigma^2) = \frac{\sigma^2}{\hat\sigma^2}
- \log\frac{\sigma^2}{\hat\sigma^2} - 1 .
\]

**Patton (2011)**, "Volatility forecast comparison using imperfect volatility
proxies", *JoE* 164(1):20–50: a loss is **robust** to unbiased proxy noise iff
it is equivalent (up to affine transform) to **MSE** or **QLIKE** — necessary
*sufficient* conditions. The robustness identity requires the *true* variance
`σ² > 0` in the log term. Repo status: `vol_eval.qlike` (elementwise, raises on
`x ≤ 0`) and `scoring.qlike` (mean, clips `y` to `1e-12`) implement the same
formula with **opposite** edge-case policies (F-07). Patton's other losses are
all present and correct in `vol_eval`: `mse`, `mse_log`, `hmse`
(Bollerslev–Ghysels / Patton 2011), `mae` (not robust — correctly *not* in the
robust set). `vol_loss_diff` gives a Newey-West-se'd loss differential, i.e. a
DM-ready primitive.

**Best practice for QLIKE at `y = 0`:** do not clip. Either (a) filter those
observations out and report the count, or (b) use the MSE-robust loss for that
subsample. Clipping to `1e-12` manufactures a ~18–23-unit penalty per zero and
destroys the very robustness property that motivates using QLIKE (F-07).

### 1.6 Log score and Brier

\[
\mathrm{LS}(f,y) = -\log f(y), \qquad
\mathrm{Brier}(p,y) = (p-y)^2 .
\]

Both strictly proper (Gneiting & Raftery 2007). Brier admits the **Murphy
(1973) decomposition** `Brier = Reliability − Resolution + Uncertainty`, which
`calibration2.murphy_decomposition` implements — correct and worth headlining
because it is the honest way to report "the Brier improved" (reliability vs
resolution). `log_loss`'s interior `eps = 1e-12` clip bounds a hard
falsification at **27.631 nats** rather than `+∞`; monotone and finite, so
defensible, but the bound is a *choice* that materially moves rankings at small
`n` (F-18). `calibration2.dawid_sebastiani` is the Dawid–Sebregondi (1999)
Gaussian score; `calibration2.hosmer_lemeshow` the Hosmer–Lemeshow (1980)
goodness-of-fit; both correct.

### 1.7 PIT and randomized PIT

For a continuous predictive CDF `F`, `u = F(y) ~ U(0,1)` under calibration.
Diagnostics, in the order best practice applies them:

- **Marginal uniformity:** Kolmogorov–Smirnov (`probability.pit_ks`) or the
  Diebold–Günther–Tay (1998) `χ²` histogram (`density_forecast.pit_histogram`).
- **Serial independence:** Berkowitz (2001) censored-normal AR(1) LR
  (`density_forecast.berkowitz_test`) and Ljung–Box on `Φ⁻¹(u)`
  (`density_forecast.pit_autocorrelation`). Both present and correctly
  specified.
- **Randomized PIT for discrete / binned outcomes:** Czado, Gneiting & Held
  (2009), "Predictive model assessment for count data", *Biometrics* 65(4):
  1254–1261. With `F⁻(y) = P(Y<y)` and `F(y) = P(Y≤y)`,
  \[
  u = F^-(y) + V\,\bigl(F(y) - F^-(y)\bigr),\quad V \sim U(0,1).
  \]
  This is exactly what a discretely-supported forecast (count data, binned
  returns, LLM-emitted probability buckets) needs; a non-randomized PIT has
  atoms and the KS/`χ²` tests are then invalid.

**Repo status — the capability exists but is siloed.** `models/qrf.py::QuantileRegressionForest.pit`
implements the randomized PIT correctly (`f_minus + rng.uniform() * mass`), and
`pit_oob_train` does the leave-one-out variant so no fitting tree contributes
to its own diagnostic — that is a genuinely careful piece of work. But it is
(a) not exported through `quant_fund.metrics`, (b) unavailable to any other
model class, and (c) uncited. `scoring.pit_values` is the metrics-layer PIT and
it is the *non*-randomized interpolation variant, which clips out-of-grid
observations to exactly `0.0`/`1.0` (F-05).

### 1.8 Expected Calibration Error — variants and pitfalls

\[
\mathrm{ECE} = \sum_{b=1}^{B}\frac{n_b}{N}\bigl\lvert \bar y_b - \bar p_b \bigr\rvert .
\]

ECE is a *binned approximation* to calibration error, and the literature is
blunt about its defects:

- **Guo, Pleiss, Sun & Weinberger (2017)** (ICML) — equal-width binning fails
  when confidence mass is concentrated; empty bins contribute nothing.
- **Naeini, Cooper & Hauskrecht (2015)** — **equal-mass / adaptive binning**
  (ACE, "Obtaining well calibrated probabilities using Bayesian binning").
- **Kumar, Sarawagi & Jain (2019)** (NeurIPS) — **mmCE**, a kernel-based
  *differentiable* and statistically consistent calibration measure; plus
  Trainable Calibration Measures.
- **Roelofs, Fridovich-Keil, Palatri & Recht (2022)**, "Mitigating bias in
  calibration error estimation" (AISTATS) — **the key result: ECE is upward
  biased, the bias scales as `≈ B/N`, and it is largest for *perfectly
  calibrated* models.** They give a debiased estimator.
- **Błasiok, Gopalan, Hu & Nakkiran (2023)**, "A unifying theory of
  distance to calibration" (STOC) — smooth calibration error (smCE) is the
  right functional; binned ECE is a rough surrogate.
- **Zhao, Roelofs, Karras, Vemuri & Frankle (2024)** — top-label vs marginal
  calibration distinction.

Measured bias in this repo (§3 F-16), true calibration error = 0 in every cell:

| `n` | B=5 | B=10 | B=20 | B=50 | B=100 |
|---|---|---|---|---|---|
| 250 | 0.0470 | 0.0665 | 0.0916 | 0.1434 | 0.1962 |
| 1000 | 0.0236 | 0.0326 | 0.0462 | 0.0727 | 0.1019 |
| 5000 | 0.0107 | 0.0148 | 0.0207 | 0.0325 | 0.0460 |

`B` is a free hyperparameter that moves the reading by ~4× at fixed `n` while
the model is unchanged. That is the central ECE pitfall and the repo ships only
equal-width binning with no bias correction.

### 1.9 Joint (VaR, ES) scores

**Fissler & Ziegel (2016)**, "Higher order elicitability and Osband's principle",
*Ann. Statist.* 44(4):1680–1707 — `(VaR_α, ES_α)` is *jointly* elicitable
(though ES alone is not; Gneiting 2011). **Nolde & Ziegel (2017)**, "Elicitability
and backtesting: perspectives for banking regulation", *Ann. Appl. Stat.*
11(4):1833–1874 — the FZ *0-homogeneous* member `G₁ ≡ 0, G₂ = log` is the
recommended choice for positive losses:

\[
S(v,e;L) = \frac{1}{1-\alpha}\mathbf{1}\{L>v\}\frac{L-v}{e} + \frac{v}{e} - 1 + \log e .
\]

Analytic verification that this minimises at `(VaR_α, ES_α)`, with `p = 1−α`:
`∂/∂v = (1/e)(1 − P(L>v)/p) = 0 ⟹ P(L>v) = p ⟹ v = q_α(L)`; substituting gives
`E[S] = ES/e + log e − 1`, so `∂/∂e = −ES/e² + 1/e = 0 ⟹ e = ES_α`. Confirmed
on a 25×25 perturbation grid (§3 NF-2). Repo `scoring.fissler_ziegel_loss`
matches, including the `1/(1−α)` factor whose absence would misplace the ES
minimiser — the docstring records that this was once wrong and was fixed.

### 1.10 Score comparison

- **Diebold & Mariano (1995)**, "Comparing predictive accuracy", *JBES* 13(3):
  253–263. `d_t = L(F₁,y_t) − L(F₂,y_t)`; `DM = \bar d / \sqrt{\hat\Omega/n}`
  with `\hat\Omega` an HAC long-run variance → `N(0,1)`.
- **Newey & West (1987)** / **Andrews & Monahan (1992)** — LRV estimation;
  Andrews–Monahan adds GLS prewhitening, which materially improves size when
  `d_t` is strongly autocorrelated. Repo uses Bartlett-kernel Newey–West
  (`inference.newey_west_variance`, `forecast_eval._lrv`) with a fixed
  `lags = ⌊1.5 n^{1/3}⌋` rule and **no prewhitening**.
- **Harvey, Leybourne & Newbold (1997)**, "Testing the equality of prediction
  mean squared errors", *IJF* 13(2):281–291. Small-sample correction
  \[
  \mathrm{HLN} = \mathrm{DM}\cdot\sqrt{\frac{n+1-2h+h(h-1)/n}{n}},
  \]
  referenced to `t_{n−1}`, not `N(0,1)`. Without it the DM test is badly
  oversized for `n ≲ 50`.
- **Clark & West (2007)**, "Approximately normal tests for equal predictive
  accuracy in nested models", *JoE* 138(1):291–311. Raw MSPE differences are
  biased *toward the smaller nested model* by parameter-estimation noise; the
  adjustment is `f_t = e_{1t}² − e_{2t}² + (f_{1t} − f_{2t})²`, one-sided,
  normal reference. **Inputs are raw errors, not squared errors.**
- **Giacomini & White (2006)**, "Tests of conditional predictive ability",
  *Econometrica* 74(6):1545–1578 — conditional (rather than unconditional)
  predictive ability; the right test when comparing *estimated* models on
  finite windows.
- **Giacomini & Rossi (2010)**, "Detecting and predicting forecast
  instabilities", *Econometrica* 78(2):677–714 — the fluctuation test. Critical
  values are **tabulated against the window fraction `m/T`** (and differ between
  the two-sided fluctuation and the one-sided alternative); a single constant
  is not the paper's result.
- **Diebold & Mariano (2002)** 20th-anniversary retrospective and Mariano
  (2021), "Comparing predictive accuracy: twenty years later" — both stress
  that DM/HLN require *non-nested* comparisons and that with many models you
  need **SPA / Reality-Check / White (2000)** data-snooping control or at
  minimum **Benjamini–Hochberg (1995)** FDR on the pairwise matrix.

---

## 2. Repo inventory

| Module | Public scoring surface | Status |
|---|---|---|
| `metrics/scoring.py` | `pinball_loss`, `mean_pinball`, `coverage`, `interval_width`, `quantile_crossing_rate`, `rearrange_quantiles`, `crps_from_quantiles`, `crps_gaussian`, `mean_crps_gaussian`, `crps_student_t`, `mean_crps_student_t`, `log_score_gaussian`, `mean_log_score_gaussian`, `crps_gaussian_mixture`, `gaussian_mixture_quantiles`, `crps_empirical`, `qlike`, `overlap_aware_qlike`, `name_level_qlike`, `one_step_density_summary`, `name_level_one_step_density_summary`, `pit_values`, `fissler_ziegel_loss`, `mean_fissler_ziegel` | core; F-02, F-05, F-07, F-14, F-15 |
| `metrics/probability.py` | `brier_score`, `log_loss`, `expected_calibration_error`, `kupiec_pof`, `pit_ks`, `christoffersen_independence`, `christoffersen_cc` | F-03, F-04, F-05, F-16, F-18 |
| `metrics/energy_score.py` | `energy_score`, `threshold_energy_score`, `energy_score_curve` | **F-01**; `energy_score` correct |
| `metrics/calibration2.py` | `murphy_decomposition`, `winkler_interval_score`, `dawid_sebastiani`, `hosmer_lemeshow`, `reliability_diagram`, `variogram_score`, `pinball_score`, `spread_skill` | F-13 |
| `metrics/density_forecast.py` | `_pit`, `pit_histogram`, `berkowitz_test`, `pit_autocorrelation` | F-05, F-06 |
| `metrics/var_backtest.py` | `kupiec_test`, `christoffersen_test`, `tuff_test`, `basel_zone` | correct incl. `x=0`; F-03 (guards), F-06 |
| `metrics/es_backtest.py` | `acerbi_szekely_test`, `mcneil_frey_test`, `du_escanciano_test` | F-06 |
| `metrics/forecast_eval.py` | `hln_test`, `clark_west_test`, `giacomini_white_test`, `encompassing_test`, `fluctuation_test`, `_lrv`, `_paired` | **F-09, F-10, F-11, F-12** |
| `metrics/inference.py` | `diebold_mariano`, `pairwise_diebold_mariano`, `newey_west_variance`, `mean_tstat`, `newey_west_se`, `benjamini_hochberg`, `stationary_bootstrap_indices` | **F-08** |
| `metrics/vol_eval.py` | `mincer_zarnowitz`, `mse`, `qlike`, `mse_log`, `hmse`, `mae`, `vol_loss_diff` | correct; F-07 (name clash), F-06 |
| `metrics/anytime_fdr.py` | `e_bh`, `stopped_e_bh`, `ELond` | correct; relevant to F-08 |
| `models/qrf.py` | `QuantileRegressionForest.pit`, `.pit_oob_train` | randomized PIT, siloed (F-19) |
| `fx1/eval/calibration_eval.py` | `run_calibration_eval`, `_equal_width_bins`, `_realized_outcome`, `build_calibration_bank` | F-17 |
| `research/benches_w810.py` | `bench_energy_score` (uses `energy_score`, **not** the threshold variant) | unaffected by F-01 |

Coverage of the mandated honesty set: pinball ✓, CRPS ✓, PIT ✓, QLIKE ✓,
Brier ✓, ECE ✓, Kupiec ✓. `FORBIDDEN_RESEARCH_METRIC_KEYS`
(`research/catalog/registry.py`) and `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`
remain in sync at `{sharpe, sortino, calmar, pnl, nav}`. **Nothing in this
lane's findings requires changing either set** — every proposed fix is a proper
score or an inference correction, which is honesty-compatible by construction.

---

## 3. Findings

Severity: **CRITICAL** = can invert a research conclusion; **HIGH** =
materially wrong numbers or wrong inference in realistic use; **MED** =
edge-case/robustness defect with plausible real triggers; **LOW** =
documentation, convention, or deliberate-contract items.

### F-01 — `threshold_energy_score` is improper for `weight >= √2`; the docstring asserts the opposite (CRITICAL)

`energy_score.py:137`. Implemented:

\[
\mathrm{ES}_w = \frac1n\sum_i w(d_i)d_i \;-\; \frac{1}{2n(n-1)}\sum_{i\ne j}
w(d_i)w(d_j)\lVert x_i-x_j\rVert ,\qquad
w(r) = \begin{cases}\text{weight} & r>\text{threshold}\\ 1 & \text{else}\end{cases}
\]

**The docstring claims:** "The product kernel `w(d_i) w(d_j)` on the second
term preserves the kernel-score structure, so `ES_w` remains a proper scoring
rule (not strictly proper when weight > 1)."

**That claim is false for `weight >= √2`, and the failure is unbounded.** Term 1
is linear in `w`; term 2 is quadratic. Once the forecast spread exceeds the
threshold, every `w(d_i)` equals `w`, so with `a = E∥Z∥` and
`X ~ σ·Z`:

\[
E[\mathrm{ES}_w] \;\approx\; w\,\sigma a \;-\; \tfrac{w^2}{2}\,\sigma\sqrt{2}\,a
\;=\; \sigma a\Bigl(w - \tfrac{\sqrt2}{2}w^2\Bigr),
\]

using `E∥X − X′∥ = σ√2 · E∥Z∥ = σ√2 a` (since `Z − Z′ ~ N(0, 2I_d)`). The
leading coefficient vanishes at

\[
w^{*} \;=\; \frac{2}{\sqrt 2} \;=\; \sqrt{2},
\]

so **for `w > √2`, `E[ES_w] → −∞` as `σ → ∞`**. Note `a` cancels: the threshold
is **exactly `√2` and dimension-free**. Verified numerically that
`2a/b = 1.414214` for `d = 1, 2, 3, 4, 8, 16` (with `b = E∥Z − Z′∥ = √2 a`),
and empirically the score crosses zero just above `√2` at every `d`:

| `d` | w=1.300 | w=1.380 | w=1.4142 | w=1.450 | w=1.550 |
|---|---|---|---|---|---|
| 1 | 0.9686 | 0.5977 | 0.4141 | 0.2195 | **−0.3675** |
| 2 | 1.2011 | 0.4949 | 0.1398 | **−0.2079** | −1.2942 |
| 4 | 1.7215 | 0.6829 | 0.1770 | **−0.4106** | −2.0653 |
| 8 | 2.5231 | 0.9623 | 0.2264 | **−0.5540** | −2.9900 |

(σ = 8, `threshold = 1.0`, 80 members, 400 reps.) At `w = √2` exactly the
leading term vanishes and the residual finite-threshold correction still drifts
negative (σ = 8 → +0.160, σ = 32 → +0.065, σ = 128 → **−0.0012**), so `√2` is
the *boundary*, not a safe value.

**A forecaster that inflates its variance without limit is rewarded without
limit.** This is not "not strictly proper" — it is *not proper at all*, and it
fails in the direction that most benefits a variance-inflating model, which is
exactly the failure mode sharpness-penalised forecast evaluation exists to
prevent.

Measured (60 ensemble members, `d=2`, `threshold=1.0`, 400 reps/cell, seed 20260928):

| `weight` | σ=0.5 | σ=1 | σ=2 | σ=4 | σ=8 | verdict |
|---|---|---|---|---|---|---|
| 1.0 | 0.943 | 0.908 | 1.029 | 1.627 | 3.009 | ok (interior minimum) |
| 1.2 | 1.005 | 0.847 | 0.865 | 1.132 | 1.938 | ok |
| **1.4142 (√2)** | 1.173 | 0.831 | 0.517 | 0.317 | **0.147** | **monotone ↓ — improper** |
| **1.5** | 1.211 | 0.796 | 0.400 | −0.117 | **−0.712** | **improper, crosses zero** |
| **2.0** | 1.386 | 0.549 | −0.989 | −3.551 | **−8.007** | **improper** |
| **3.0** | 1.279 | −0.986 | −6.186 | −15.469 | **−32.867** | **improper** |

Control: plain `energy_score` has an interior minimum at σ=1
(0.951 / 0.870 / 0.890 / 0.946 / 1.029 at σ = 0.5/0.75/1/1.5/2) — proper, as
documented. So the defect is specific to the threshold weighting.

**Aggravating factor: the tests pin the bug.**
`tests/unit/core/test_energy_score.py::test_hand_computable_threshold_weighted_3x2`
asserts the exact pathological value `expected = (2.0 − 2.0√2)/3 ≈ −0.276` at
`weight=2.0` — a **negative** energy score, which for a proper rule bounded
below by `E∥X−y∥ − ½E∥X−X′∥ ≥ 0` in the population is impossible. And
`test_energy_score_curve_flat_at_weight_one` asserts
`amplified.scores[0] < plain` with the comment "the weighted score is
`2*term1 − 4*term2 = 2*plain − 2*term2 < plain`" — i.e. the quadratic blow-up
is documented as intended behaviour. `test_threshold_weighting_penalizes_tail_more_than_central`
asserts `tw_tail − tw_central > plain_tail − plain_central`, which is the
amplification property and is *fine*; it just does not test propriety. **No
existing test checks that the score is bounded below or minimised at the true
distribution.**

**Fix.** Three options, in increasing order of work:
1. **Guard:** reject `weight >= √2` in `_validate_weight` with a message naming
   the propriety breakdown (`√2` is the boundary, not a safe value — see the
   drift at `w = √2` above), and correct the docstring to state that
   `weight < √2` is required for propriety. Cheapest, and honest. Note this
   *would* reject `weight = 2.0`, the value the existing test suite pins.
2. **Correct kernel:** implement the Gneiting & Ranjan (2013) / Stroud & Stein
   (2020) threshold-weighted score properly — derive the weighted score from
   the *weighted energy-distance kernel* so both terms receive the same
   weight structure and negative definiteness is preserved.
3. **Replace:** drop the threshold variant in favour of a **tail-weighted
   CRPS** (Gneiting & Ranjan 2013 use threshold weightings on CRPS, where the
   construction is clean because CRPS is a 1-D integral) plus the plain
   `energy_score` for the multivariate channel.

Recommended: (1) now, (2) if the tail-emphasis diagnostic is actually used in a
research claim, else (3). Add a propriety test: for a fixed observation
distribution, `E[ES_w(F_true)] ≤ E[ES_w(F_σ)]` over a grid of σ spanning
`[0.25, 8]`, asserted at the shipped `weight` default (`1.0`, which is safe)
and rejected (xfail with a named reason) for `weight >= √2`. Note the guard
would break `test_hand_computable_threshold_weighted_3x2`, which uses
`weight=2.0` — that test must be rewritten to a sub-threshold weight or moved
to an xfail that documents the improperness.

---

### F-02 — `crps_empirical` is the biased plug-in while its sibling `energy_score` is the fair U-statistic (HIGH)

`scoring.py:396`. The docstring states the formula explicitly with
`\frac{1}{2n^2}\sum_{i,j}` — the sum *includes* the diagonal. Those terms are
zero, so the denominator counts `n` extra zero contributions, shrinking term 2
by a factor `(n−1)/n`:

\[
\widehat{\mathrm{CRPS}}_{\text{plug-in}}
= \widehat{\mathrm{CRPS}}_{\text{fair}} + \frac{1}{2n}E\lvert X-X'\rvert
+ o(n^{-1}).
\]

Measured (Gaussian predictive, 4000 reps/cell, seed 20260928):

| n | E[plug-in] | E[fair] | measured bias | predicted `σ/(n√π)` | rel. err | **% of score** |
|---|---|---|---|---|---|---|
| 5 | 0.662235 | 0.549685 | 0.112550 | 0.112838 | 0.3% | **+20.5%** |
| 10 | 0.624437 | 0.568113 | 0.056323 | 0.056419 | 0.2% | **+9.9%** |
| 20 | 0.590676 | 0.562515 | 0.028161 | 0.028209 | 0.2% | **+5.0%** |
| 50 | 0.579293 | 0.568008 | 0.011285 | 0.011284 | 0.0% | +2.0% |
| 100 | 0.567351 | 0.561711 | 0.005641 | 0.005642 | 0.0% | +1.0% |

The analytic bias matches to 0.3%, confirming the mechanism exactly.

**Why this is HIGH and not cosmetic:** small ensembles are the normal case for
research ensembles (n = 5–20 members). At n=5 a model's CRPS is inflated 20%.
Because the bias is `∝ 1/n`, **it does not cancel when comparing two models
with different ensemble sizes** — a 5-member ensemble is penalised ~20% relative
to a 50-member one purely for its size. Any CRPS league table mixing ensemble
sizes is invalid. Separately, `metrics/energy_score.py:105` already implements
the *correct* fair form `1/(2n(n−1))` for the multivariate analogue, so the
package contradicts itself: the `d=1` CRPS is biased, the `d≥1` energy score is
not.

**Fix.** Change term 2 to the off-diagonal mean over `n(n−1)`, i.e.
`np.sum(pairwise) / (2 * n * (n - 1))` with `n ≥ 2`, and `NaN` (or the
degenerate `|X₁−y|`) for `n == 1`. This is a **value-changing** fix, so it must
land with:
- a frozen-value regression test updated deliberately (note
  `tests/regression/test_scoring_frozen.py` already pins `pinball_loss` and
  `qlike` — add `crps_empirical` there);
- the docstring math updated to `\frac{1}{2n(n-1)}\sum_{i\ne j}`;
- a note in `INFLIGHT` / the receipt ledger that prior receipts quoting
  `crps_empirical` at small `n` carry a known `+E|X−X′|/(2n)` offset.

Optionally expose both as `crps_empirical(..., estimator="fair"|"plugin")` with
`"fair"` the default, so historic numbers remain reproducible — that is the
receipt-friendly variant and is preferred here given the immutability rule.

---

### F-03 — `alpha` means three different things across exported functions, and two of them accept the wrong one silently (HIGH)

The package uses `alpha` for **all three** of: (a) VaR *coverage* level
(≈0.95/0.99), (b) VaR *tail/breach* level (≈0.05/0.01), and (c) expected *hit
rate* under the null. There is no type-level or runtime distinction.

| function | `alpha` means | guards wrong convention? |
|---|---|---|
| `scoring._require_alpha_coverage` | coverage, "e.g. 0.95" | **no** — only `(0,1)` |
| `scoring.fissler_ziegel_loss` | coverage | **no** |
| `probability.kupiec_pof` | expected hit rate ≈0.05 | **no** |
| `probability.christoffersen_cc` | expected hit rate ≈0.05 | **no** |
| `var_backtest.kupiec_test` | tail level, requires `α ∈ (0.5, 1)` | **yes** — raises |
| `var_backtest.christoffersen_test` / `tuff_test` / `basel_zone` | tail level, default 0.99 | **yes** |

Measured on `n=500`, `x=20` (true breach rate 0.040):

```
probability.kupiec_pof(hits, 0.05)      -> (0.04, LR=    1.1267, p=0.2885)   correct
probability.kupiec_pof(hits, 0.95)      -> (0.04, LR= 2710.0106, p=0.0)      SILENTLY WRONG
var_backtest.kupiec_test(hits, 0.95)    -> {'statistic': 1.1267, ...}         correct
var_backtest.kupiec_test(hits, 0.05)    -> ValueError: alpha should be a tail level in (0.5, 1)
scoring.fissler_ziegel_loss(..., 0.05)  -> mean 0.9945, finite, NO error      SILENTLY WRONG
```

A well-calibrated VaR reads `LR = 1.13, p = 0.29` (clean pass) under the right
convention and `LR = 2710, p = 0.0` (catastrophic rejection) under the inverted
one. **Both outputs are finite, plausible, and indistinguishable from the
return value alone.** `kupiec_test` proves the guard is trivial to add — it just
was not added to the two functions that *are* re-exported through
`quant_fund.metrics.__init__` (`kupiec_pof`, `christoffersen_cc`,
`expected_calibration_error`, `pit_ks` are all in `__all__`).

**Fix.** In `kupiec_pof` / `christoffersen_cc` reject `alpha ≥ 0.5` with
`"kupiec_pof expects the expected hit rate (e.g. 0.05), not a coverage level;
did you mean var_backtest.kupiec_test?"`. In `_require_alpha_coverage` reject
`alpha ≤ 0.5` for the coverage-convention functions. Longer term, rename to
`breach_rate` / `coverage_level` / `tail_level` so the type carries the
convention; a `NewType` per convention would make mypy catch it. Also worth
noting `probability.christoffersen_cc` and `var_backtest.christoffersen_test`
are two implementations of Christoffersen (1998) with the same collision.

---

### F-04 — `kupiec_pof` returns NaN at `x = 0` / `x = n` where the LR statistic is perfectly well defined (MED)

`probability.py:100-102`:

```
rate = x / n
if x == 0 or x == n:
    return rate, float("nan"), float("nan")
```

The likelihood-ratio statistic has exact finite limits at both boundary cells.
Setting the MLE `\hat π = x/n` and taking limits:

\[
x=0:\; \mathrm{LR} = -2n\log(1-\alpha), \qquad
x=n:\; \mathrm{LR} = -2n\log(\alpha).
\]

Measured at `n = 500, α = 0.05`:

| x | `kupiec_pof` | analytic LR |
|---|---|---|
| 0 | **NaN** | 51.293294 (p = 7.955e−13) |
| 1 | 42.755 | — |
| 2 | 36.993 | — |
| 499 | 2975.416 | — |
| 500 | **NaN** | 2995.732 |

**`x = 0` — zero violations in 500 trials — is the single most informative cell
in the whole test**: it says the VaR is grossly *too conservative*, which for
capital-allocation purposes is a real finding (over-reserving). Returning NaN
discards it. And `var_backtest.kupiec_test` **does** implement it:

```
var_backtest.kupiec_test(hits_all_zero, 0.95) ->
  {'statistic': 51.29329438755058, 'pvalue': 7.954747971439247e-13,
   'failures': 0.0, 'expected': 25.0, 'rate': 0.0}
```

exactly matching the analytic limit. So the repo contains a correct and an
incorrect implementation of Kupiec's (1995) POF test, and the incorrect one is
the one exported through `quant_fund.metrics.__init__`.

**Fix.** Add the two boundary branches to `kupiec_pof` mirroring
`var_backtest.kupiec_test`, or (better) delete `kupiec_pof` and re-export the
`var_backtest` version under both names so there is one implementation. Frozen
test on the `x=0` analytic value `−2n log(1−α)`.

---

### F-05 — `pit_values` emits exact 0.0/1.0 that crash `density_forecast._pit` and poison `pit_ks` (MED)

Three functions in the PIT pipeline have mutually incompatible boundary
contracts:

1. `scoring.pit_values:882` — `np.interp(y[i], q[i], t, left=0.0, right=1.0)`.
   Any observation outside the quantile grid gets **exactly** `0.0` or `1.0`.
2. `density_forecast._pit:27` — `if np.any(p <= 0) or np.any(p >= 1): raise
   ValueError("PIT values must lie in the open interval (0, 1)")`.
3. `probability.pit_ks:120` — `np.clip(u, 1e-9, 1.0 - 1e-9)`.

Measured with a forecast grid spanning `[−1, 1]` and observations
`[−5, −0.5, 0.5, 5]`:

```
pit_values -> [0.0, 0.275, 0.725, 1.0]        exact 0.0/1.0 produced: True
density_forecast._pit(padded to n>=50) -> ValueError: PIT values must lie in the open interval (0, 1)
pit_ks(500 exact-0.0 + 500 uniform)    -> KS = 0.5000, p = 1.065e-231
```

So (1)→(2) is a hard crash, and (1)→(3) silently converts a *narrow-forecast*
diagnostic into a *boundary-point-mass* diagnostic. The KS statistic then
measures the clipping artefact: a forecast that is merely too narrow
(observations outside the reported grid) becomes indistinguishable from one
with a genuine atom at the boundary — and the clipping makes the p-value
astronomically small, which reads as an emphatic calibration failure of the
wrong kind.

**This is not exotic.** It is the normal case whenever the quantile grid is the
standard 19-point `0.05…0.95` set (`crps_from_quantiles`'s default companion)
and the target is heavy-tailed FX returns: a 3σ+ move lands outside the grid.

**Fix.** Make the boundary contract explicit and consistent:
- `pit_values` should return NaN (or a documented sentinel) for
  `y < q_min` / `y > q_max` and separately report the out-of-grid *fraction*,
  which is itself the useful diagnostic (it is exactly the tail-mass defect
  F-14 describes).
- Add `pit_randomized` (§4 G-1) as the primary PIT for any discretely-supported
  or grid-truncated forecast — randomization removes the atoms by construction
  (Czado, Gneiting & Held 2009) and makes KS/`χ²` valid.
- `pit_ks` should reject (not clip) any input containing exact 0/1, with a
  message pointing at `pit_randomized`.
- A property test asserting
  `∀ y, q, τ: pit_values(y,q,τ) ∈ (0,1) ∪ {NaN}` would have caught the
  (1)→(2) crash at authoring time; note
  `tests/property/test_pit_asof_never_future.py` shows the property-test idiom
  already exists in-repo.

---

### F-06 — `1.0 - cdf(...)` p-values floor at exactly 0.0, destroying FDR rank ordering (MED)

`2.0 * (1.0 - stats.t.cdf(...))` and `1.0 - stats.chi2.cdf(...)` lose all
precision in the upper tail because `cdf → 1.0` in double precision. Measured:

| statistic | `1 − chi2.cdf` | `chi2.sf` | | `2(1 − t.cdf)` | `2·t.sf` |
|---|---|---|---|---|---|
| 10 | 2.540e−10 | 2.540e−10 | | 0.000e+00 | 2.467e−19 |
| 40 | — | — | | 0.000e+00 | — |
| 60 | 9.437e−15 | 9.486e−15 | | 0.000e+00 | — |
| 80 | **0.000e+00** | 3.744e−19 | | 0.000e+00 | — |
| 100 | **0.000e+00** | 1.524e−23 | | 0.000e+00 | — |

Concrete instance: `hln_test(e1, e2)` with a 6× sd ratio gives
`statistic = −9.9622, pvalue = 0` — the correct p is `3.180e−19`.

**Why it matters beyond aesthetics.** `p == 0.0` is (a) indistinguishable from
"rejected", (b) *non-orderable*: ten tests all with true p between `1e−19` and
`1e−80` become ten exact ties, so Benjamini–Hochberg cannot rank them and the
FDR step-up procedure degenerates, and (c) not a p-value — a p-value of exactly
0 is inadmissible under a continuous null. This directly blocks the F-08 fix:
adding BH to `pairwise_diebold_mariano` is only meaningful if the p-values are
orderable.

**~20 call sites**, all in the metrics/models layer: `vol_eval.py:54`,
`density_forecast.py:46,103,128`, `var_backtest.py:53,88,95,123`,
`panel.py:144,176,200`, `dependence.py:67,127,155`, `es_backtest.py:150`,
`models/dynamic_panel.py:156`, `models/event_study.py:86,114,137,162`, plus
`forecast_eval.py:63,95` (`hln_test`, `clark_west_test`).
`probability.kupiec_pof` and `pit_ks` already use `.sf` correctly — so the repo
knows the right idiom; it just is not applied uniformly.

**Fix.** Mechanical: `stats.X.cdf(a, df) → stats.X.sf(a, df)`,
`2*(1-cdf(|t|)) → 2*sf(|t|)`. Add a ruff-style repo grep guard or a lint test
that fails on the pattern `1\.0?\s*-\s*stats\.\w+\.cdf` under `src/`, so this
cannot regress. Low risk, high value, and it is a prerequisite for F-08.

---

### F-07 — Two `qlike` functions with opposite zero-handling policies; the clip breaks Patton (2011) robustness (MED)

| | `scoring.qlike` | `vol_eval.qlike` |
|---|---|---|
| returns | scalar mean | elementwise array |
| `y = 0` | **clips to `floor=1e-12`, returns finite** | **raises** `ValueError: qlike requires positive forecasts and proxies` |
| `y < 0` | masked out (documented, correct) | raises |
| min length | none | `≥ 10` |

Same metric name, opposite policies, both reachable. The `scoring.qlike`
docstring argues *against* clipping — "clipping a negative variance to a tiny
positive floor would fabricate a plausible finite QLIKE from malformed input" —
and then applies the mask `y >= 0.0`, which **admits `y == 0` and clips it**.
The stated principle is sound; the boundary is off by one case.

Manufactured penalty per zero observation (`floor = 1e-12`), verified against
the closed form `f/fc − log(f/fc) − 1`:

| `forecast_var` | `scoring.qlike` at `y=0` | `−log(ratio)` |
|---|---|---|
| 1e−4 | 17.4207 | 18.421 |
| 1e−3 | 19.7233 | 20.723 |
| 1e−2 | 22.0259 | 23.026 |

**Robustness impact** (Patton 2011 requires `σ² > 0` strictly for the log
term). Simulated with true variance constant at `1e−3` — a *perfect* forecast,
true QLIKE = 0:

| zero-proxy fraction | `scoring.qlike` | `vol_eval.qlike` |
|---|---|---|
| 0.00 | 0.0000 | 0.0 |
| 0.02 | **0.3945** | NaN (raises) |
| 0.10 | **1.9723** | NaN (raises) |

A perfect forecast reports QLIKE = 1.97 when 10% of proxies are zero. Under
Patton's conditions QLIKE should be *invariant* to unbiased proxy noise; the
floor clip destroys that invariance precisely at `y = 0`. And mixing one zero
into a 250-observation clean panel moves the reported mean from `0.000000` to
`0.078893`.

`y = 0` is realistic in this repo's domain: realized variance over a bar with no
trades, a flat bar, or a single-print window all give exactly zero.

**Fix.** In `scoring.qlike`, change the validity mask to `y > 0.0` and count the
rejected zeros, returning them as `n_zero_proxy` in a companion field (or in the
`overlap_aware_qlike` / `name_level_qlike` dicts, which already carry
`n_origins_*` bookkeeping — the honest place for it). Keep the `floor`
parameter but apply it to `yhat` only (where it guards a genuine division) and
document that `y = 0` is an invalid observation, not a small one. **Caution:**
`tests/regression/test_scoring_frozen.py::test_qlike_formula_frozen` pins
`qlike([e], [1.0]) == e − 2`; that case has `y > 0` and is unaffected.

---

### F-08 — `pairwise_diebold_mariano` performs no multiplicity control (HIGH)

`inference.py:592`. Row keys are exactly
`['a', 'b', 'mean_loss_diff', 'n', 'p_value', 'preferred', 'statistic']` — raw
p-values only. `benjamini_hochberg` is defined at `inference.py:342`, in the
**same module**, and is used elsewhere in the repo
(`reality/fdr.py:16`, `research/agent.py:31`, `northset/sweep_research.py:24`,
`models/conformal_rank.py:21`) — just not here.

Expected false positives per experiment at nominal α = 0.05:

| models | pairs | E[false positives] |
|---|---|---|
| 4 | 6 | 0.30 |
| 6 | 15 | 0.75 |
| 10 | 45 | 2.25 |
| 15 | 105 | 5.25 |
| 20 | 190 | 9.50 |

Measured with **eight identical models** (every H₀ true by construction,
`n = 120–150` squared-normal losses):

- raw rejection rate `0.0683` across 8400 tests (nominal 0.05 — mild HAC
  size distortion on top of multiplicity);
- **FWER = 0.622**: a 62% chance that an experiment with eight indistinguishable
  models produces at least one "significant" pairwise DM claim;
- applying BH(0.05) post-hoc to the same p-values: **0.088**.

A model arena with 15–20 candidates is the normal case in this repo
(`research/benches*`, arena sweeps), where the unadjusted expectation is 5–10
spurious "A beats B" claims per experiment. Given that pairwise DM matrices are
exactly what gets written into a receipt as evidence, this is a
honesty-contract-adjacent defect: the receipts would faithfully record
conclusions that are multiplicity artefacts.

**Fix.** Add `p_value_bh` (and optionally `p_value_bonferroni`) columns to every
row via the existing `benjamini_hochberg`, plus a top-level
`n_pairs` / `fdr_alpha` stamp so a receipt records that adjustment happened.
Default `include_bh=True`. Also expose `preferred` recomputed on the adjusted
p-value, since the current `preferred` field will otherwise contradict the
adjusted result. This is a ~5-line change with an existing, tested dependency.
Note `include_e_process` already offers a Choe–Ramdas-style e-process route
(`metrics/evalues.e_process_dm`) and `anytime_fdr.e_bh` exists — e-values
compose across tests *without* a multiplicity correction, so the e-process path
is the more principled long-term answer and should be preferred where the
caller can accept e-values instead of p-values. Document both.

---

### F-09 — `hln_test` hardcodes squared-error loss, so the small-sample correction is unavailable for proper scores (HIGH)

`forecast_eval.py:56-64`:

```
a, b = _paired(e1, e2)
d = a**2 - b**2
```

It takes **errors** and squares them internally. `inference.diebold_mariano`
takes **losses** (`d = a - b`) and is loss-agnostic. Two DM implementations,
incompatible input conventions:

| | `forecast_eval.hln_test` | `inference.diebold_mariano` |
|---|---|---|
| input | raw errors | loss series |
| loss | **squared error, hardcoded** | caller-supplied (any) |
| correction | HLN small-sample ✓ | none (plain DM) |
| reference dist | `t_{n−1}` ✓ | normal |
| HAC lags | `h = lag or 1` | `⌊1.5 n^{1/3}⌋` |

**Consequence:** CRPS, QLIKE, pinball, energy-score and log-score differentials
— i.e. *every score the honesty contract permits as a headline* — can only be
tested with the uncorrected `diebold_mariano`. The HLN correction, which exists
specifically because DM is badly oversized at `n ≲ 50` (and FX research samples
are routinely `n = 60–250`), is structurally unavailable for them. A repo whose
stated primary metrics are proper scores has wired its small-sample correction
to squared error only.

Cross-check that the two agree on squared-error inputs (so neither is
independently broken): `hln_test(e1,e2)` gives `stat = −1.5027, p = 0.1345`;
`diebold_mariano(e1², e2²)` gives `stat = −1.4731, p = 0.1423`; the HLN
adjustment factor `√((n+1−2h+h(h−1)/n)/n)` at `h=1, n=200` is `0.997497`, and
the observed statistic ratio is `1.020081` — consistent up to the differing HAC
lag choices (`h−1 = 0` vs `⌊1.5·200^{1/3}⌋ = 8`). Both are fine; they are just
not the same test.

**Fix.** Generalise to `hln_test(loss_a, loss_b, *, lag=None)` operating on the
loss differential directly (keeping an `errors_a/errors_b` convenience wrapper
that squares), and unify the HAC lag rule with `diebold_mariano`. Then a single
function serves `dm`/`hln` for any proper score. Add Andrews–Monahan (1992)
prewhitening as an option while touching `_lrv` — for strongly autocorrelated
overlapping-horizon loss differentials (the `overlap_aware_qlike` case) Bartlett
NW alone is known to understate the variance.

---

### F-10 — `clark_west_test` docstring and body disagree on raw vs squared errors (MED)

`forecast_eval.py:67-96`. The docstring says:
"`errors_alt`: squared-error residuals of the larger (alternative) model."
The body does `d = e1**2 - e2**2 + fd**2`. If a caller follows the docstring
and passes *squared* errors, the body squares them **again**.

Clark & West (2007) define `f_t = e_{1t}² − e_{2t}² + (f_{1t} − f_{2t})²` on
**raw** errors. So the body is right and the docstring is wrong — but the
consequence is a silently meaningless statistic:

```
raw errors (CW's definition):   {'statistic': 0.5516, 'pvalue': 0.2906, 'mspe_adj': 0.1044}
squared errors (as documented): {'statistic': 0.0293, 'pvalue': 0.4883, 'mspe_adj': 0.0361}
```

Same function, same data, ~19× different statistic, and the documented usage is
the wrong one. Also: `p = 1.0 - stats.norm.cdf(stat)` (F-06 precision), and the
`≥ 30` length floor is arbitrary relative to CW's own simulation guidance.

**Fix.** Rename parameters `errors_alt → raw_errors_alt` (or
`residuals_alt`), fix the docstring to say *raw* residuals, and add an assertion
or heuristic warning when the inputs look pre-squared (e.g. all non-negative
with mean ≈ `E[e²]`). Switch to `stats.norm.sf`. Frozen test with a hand-computed
nested example where the CW adjustment is known to flip the sign of the raw
MSPE difference — that is the entire point of the test and is currently untested.

---

### F-11 — `fluctuation_test`: hardcoded critical value with empirical size 0.31, and an unresolved docstring (HIGH)

`forecast_eval.py:160-196`:

```
# GR (2010) asymptotic critical values for two-sided test, m->inf:
# 10% ~ 3.18, 5% ~ 3.68 approx for the fluctuation statistic.
cv = {0.10: 3.18, 0.05: 3.68}.get(alpha, 3.18)
reject = float(sup > cv)
```

Giacomini & Rossi (2010) tabulate critical values **as a function of the window
fraction `m/T`**, and the fluctuation test's sup-statistic distribution is
non-standard (a functional of Brownian motion), not a constant. The `.get(alpha,
3.18)` fallback silently applies the 10% value for any unlisted α. The docstring
itself is unfinished and self-contradicting:

> "…scaled here by `sqrt(window)/...` we use the sup-of-|stat| with the standard
> 10% asymptotic bound `~ 3.18/sqrt(1-2*log(...))` — in practice we report the
> sup and let the caller compare."

The `sqrt(1-2*log(...))` expression is truncated mid-formula and no such scaling
appears in the code.

**Measured empirical size under H₀** (equal skill, `n=400`, nominal 10%, 500 reps):

| window | cv used | rejections | **empirical size** |
|---|---|---|---|
| 20 | 3.18 | 153 | **0.306** |
| 40 | 3.18 | 60 | 0.120 |
| 80 | 3.18 | 34 | 0.068 |

At `window=20` the test over-rejects by **3×**. The `m/T` dependence is exactly
what the hardcoded constant ignores. Any receipt recording "forecast skill is
unstable" from this function at a small window is over-claiming by a factor of
three.

**Fix.** Either (a) implement GR's tabulated `m/T`-dependent critical values
(their Table 1) with interpolation, or (b) calibrate by simulation at import
time / in a checked-in table keyed by `(window/n, alpha)`, or (c) demote the
function to report `sup` and the `stats` array only, with `reject` removed and
the docstring stating that no critical value is supplied. (c) is the most
honest minimal change and matches what the docstring already half-admits.
Whichever is chosen, add a **size test**: under H₀ over ≥2000 reps the rejection
rate must be within Monte-Carlo error of nominal at several `window/n`.

---

### F-12 — `_lrv` floor at `1e-20` manufactures arbitrary statistics on degenerate windows (MED)

`forecast_eval.py:46`: `return max(g, 1e-20)`. When a loss-difference segment has
genuinely zero variance, the statistic becomes `mean / sqrt(1e-20 / window)` —
finite but driven entirely by the floor constant:

| segment | `_lrv` | mean | statistic |
|---|---|---|---|
| all-zero | 1.000e−20 | 0.000e+00 | 0.000e+00 |
| **constant 0.5** | 1.000e−20 | 5.000e−01 | **+2.236e+10** |
| tiny noise 1e−12 | 1.000e−20 | 1.000e−12 | +4.472e−02 |

A `2.2e10` statistic. Since `fluctuation_test` takes `max(|stats_roll|)` over
all windows, **one flat window anywhere in the scan sets the sup and flips
`reject`**. Flat windows are realistic: a constant forecast vs a constant
benchmark over a holiday/weekend gap, a degenerate `d_t` after masking, or
duplicate bars.

**Fix.** Detect the degenerate case explicitly rather than flooring: if the
segment is constant (or `g < tol · mean²`), return NaN for that window's
statistic and exclude it from the sup, recording `n_degenerate_windows`. A
constant nonzero loss differential with zero variance is not "infinitely
significant" — it is an uninformative segment.

---

### F-13 — `variogram_score` omits the `1/(d(d−1))` normalisation, so it is not comparable across dimensions (MED)

`calibration2.py:168-187` computes `np.sum((exx - dy) ** 2)` over the **full
`d × d` grid** with no normalising constant. Scheuerer & Hamill (2015) Eq. (3)
define

\[
\mathrm{VS}_p = \frac{1}{d(d-1)}\sum_{i=1}^{d}\sum_{j\ne i}
\bigl(E_F\lvert X_i-X_j\rvert^p - \lvert y_i-y_j\rvert^p\bigr)^2 .
\]

Measured (200-member ensembles, standard normal, `p = 0.5`):

| d | shipped VS | VS/(d(d−1)) |
|---|---|---|
| 3 | 0.4407 | 0.0735 |
| 6 | 5.4862 | 0.1829 |
| 12 | 17.8239 | 0.1350 |
| 24 | 93.4592 | 0.1693 |
| 48 | 333.0406 | 0.1476 |

Raw VS grows like `d²`; the per-pair value is flat in `d`. So a 48-asset
portfolio scores ~750× a 3-asset one for *identical* per-pair calibration
quality. Any multi-asset or multi-horizon league table using this function is
silently comparing different scales — and the direction always favours
low-dimensional forecasts.

Two sub-points, one benign: the `i == j` terms contribute
`(|x_i−x_i|^p − |y_i−y_i|^p)² = 0`, so **including the diagonal is harmless
here** — the missing `1/(d(d−1))` is the actual defect (using `1/d²` instead
would be wrong by a constant factor `(d−1)/d`). Also, the docstring's first line
is garbled: `"``VS = sum_ij w_ij |y_i - y_j|^p * |y_i - y_j|^p ...``"` — an
unfinished formula with a stray weight `w_ij` that does not appear in the code.

**Fix.** Divide by `d(d−1)`; keep the raw sum available as
`variogram_score(..., normalize=False)` if any historic receipt needs it. Fix
the docstring formula. Add a scale test: VS must be `O(1)` in `d` for a fixed
per-pair calibration quality.

---

### F-14 — `crps_from_quantiles` truncates 10% of the integral and under-reports tail events (LOW–MED)

`scoring.py:100-120` is a left-endpoint Riemann sum
`2 Σ_k QS_{τ_k}(q_k, y) Δτ_k` with `Δτ_k = τ_k − τ_{k−1}` (`τ₀ ≡ 0`). Two
things I initially suspected and then **disproved**: the cells are *uniform*
(`Δτ = 0.05` throughout on the 19-point `0.05…0.95` grid, including the first —
verified `np.allclose(dt, dt[0]) == True`), and the sum is a legitimate
quadrature. The only real defect is **tail truncation**: with `τ ∈ [0.05, 0.95]`,
`0.05 + 0.05 = 10%` of the CRPS integral is never sampled.

Against the closed form for `N(0,1)` (closed-form CRPS at `y=0.7` = 0.421569):

| k | tails at | CRPS | rel. err |
|---|---|---|---|
| 19 | 0.050 | 0.419254 | −0.55% |
| 19 | 0.020 | 0.418656 | −0.71% |
| 39 | 0.050 | 0.421949 | +0.09% |
| 99 | 0.010 | 0.421466 | −0.02% |
| 999 | 0.001 | 0.421568 | −0.00% |

Modest in the bulk — but it grows with the extremity of the observation, which
is where FX risk work lives:

| y | grid CRPS (k=19, tails 0.05) | closed form | rel. err |
|---|---|---|---|
| 0.7 | 0.4193 | 0.4216 | −0.55% |
| 3.0 | 2.4038 | 2.4366 | −1.34% |
| 8.0 | 7.1538 | 7.4358 | −3.79% |
| 20.0 | 18.5538 | 19.4358 | **−4.54%** |

The estimator systematically **under-reports** extreme observations, because the
unsampled 10% of tail mass carries most of the CRPS for a far-out `y`. Two
models differing mainly in tail behaviour are therefore compared on a truncated
objective. Note the error is **not monotone in k** (−0.55% at k=19/0.05 vs
−0.71% at k=19/0.02 vs +0.09% at k=39/0.05): tail truncation and quadrature
error have opposite signs, so refining the grid alone does not monotonically
help.

**Fix.** Document the truncation in the docstring with the measured bias, and
add an explicit tail-extrapolation option: assume the forecast is Gaussian (or
GPD) beyond `τ_min`/`τ_max` and add the analytic tail contribution.
Alternatively require callers to state the grid's tail coverage in the receipt
so a reader can bound the error. Also expose `taus` validation that warns when
`τ_min > 0.02` on heavy-tailed data. **Do not** silently widen the grid —
that changes frozen values.

---

### F-15 — `crps_student_t` requires `ν > 2` although the closed form is valid for `ν > 1` (LOW — deliberate, documented)

`scoring.py:212` masks on `nu_arr > 2.0`. The Jordan–Krüger–Lerch closed form
the docstring cites is valid for `ν > 1` (finite mean). Measured:

| ν | repo | Monte-Carlo truth (2e5 draws, 3·s.e.) |
|---|---|---|
| 1.5 | **NaN** | 0.3678 ± 0.0025 |
| 2.0 | **NaN** | 0.3354 ± 0.0022 |
| 2.5 | 0.318346 | 0.3178 ± 0.0021 ✓ |
| 4.0 | 0.297128 | 0.2979 ± 0.0020 ✓ |

The formula is correct where it is evaluated (ν ≥ 2.5 matches MC within 2 s.e.).
The `ν > 2` guard is **explicitly documented as a deliberate lab contract**:
"stricter than the finite-mean `ν > 1` domain — fail-closed with finite
variance". That is a defensible design decision (it refuses to score a forecast
whose variance is undefined), and it fails *closed* to NaN rather than to a
wrong number — which is the honest failure mode.

**Not a bug. Recorded so it is not "fixed" by mistake.** The cost to note: FX
return tails are frequently estimated at `ν ∈ (1.5, 3)`, so the heaviest-tailed
regime — the one where CRPS matters most — returns NaN. If that blocks real
work, extend the guard to `ν > 1` *behind an explicit opt-in*
(`crps_student_t(..., allow_infinite_variance=True)`) and stamp the choice in
the receipt, rather than loosening the default.

---

### F-16 — `expected_calibration_error`: equal-width only, no bias correction, no adaptive binning (MED)

`probability.py:60-87`. Correct implementation of the equal-width binned ECE
(Guo et al. 2017) — AST-verified parenthesisation, textbook-matching values
(`E[ECE]` identical to a reference implementation to 5 decimals at
`n = 60…20000`). But it is the *only* calibration-error estimator in the repo,
and it carries the well-documented defects:

**Bias `≈ B/N`, worst for well-calibrated models** (Roelofs et al. 2022).
Measured `E[ECE]` for a *perfectly* calibrated forecaster (true CE = 0), so every
number is pure estimator bias:

| n | B=5 | B=10 | B=20 | B=50 | B=100 |
|---|---|---|---|---|---|
| 250 | 0.0470 | 0.0665 | 0.0916 | 0.1434 | 0.1962 |
| 1000 | 0.0236 | 0.0326 | 0.0462 | 0.0727 | 0.1019 |
| 5000 | 0.0107 | 0.0148 | 0.0207 | 0.0325 | 0.0460 |

Roughly linear in `B`, roughly `∝ 1/N`. A debiased estimator (subtracting the
analytic within-bin variance term `Σ p_i(1−p_i)/m²` under the square root)
removes **~58%** of this bias at every `n` tested (n=60: 0.1322 → 0.0555;
n=250: 0.0657 → 0.0274; n=1000: 0.0328 → 0.0140).

**Equal-mass / adaptive binning (Naeini et al. 2015) is absent — and it is not
a free win.** Tested on data with a real, localised miscalibration (90% of mass
at `p=0.5` well calibrated, 10% at `p=0.99` with true rate 0.60; true weighted
CE = 0.039):

| binning | B=5 | B=10 | B=20 | B=50 | B=100 | B=200 |
|---|---|---|---|---|---|---|
| equal-width (shipped) | 0.0404 | 0.0404 | 0.0404 | 0.0404 | 0.0404 | 0.0404 |
| equal-mass (ACE) | 0.0450 | 0.0474 | 0.0498 | 0.0634 | 0.0778 | 0.0906 |

Equal-width is remarkably stable here (the construction puts the overconfident
subpopulation in its own bin at every B); equal-mass *drifts upward* with B,
because it splits the dominant `p=0.5` mass into many bins whose empirical
frequencies are noisy. **So ACE trades bias-in-B for variance-in-B; it is not
uniformly better.** The honest conclusion is that no binned estimator is
trustworthy at a single `(n, B)`, which is why the modern recommendation is
**smCE** (Błasiok et al. 2023) or **mmCE** (Kumar et al. 2019) — bin-free,
`O(1/√N)`-consistent, and with a known rate.

**Fix (gap G-3).** Ship three things alongside the existing function, without
changing it: `expected_calibration_error(..., binning="equal-mass")`,
`debiased_ece`, and `smoothing_calibration_error` (smCE, with the Błasiok et al.
kernel). Report `n`, `B`, and the binning rule in every receipt that carries an
ECE — the number is meaningless without them. Add a test asserting the bias
table above is reproduced, so the estimator's known bias is *documented as a
measured artefact* rather than rediscovered.

---

### F-17 — the fx1 calibration gate passes its own documented negative control on 7/8 seeds (MED)

`fx1/eval/calibration_eval.py:356-433`. **First, what is right:** the module's
central design claim is *correct and verified*. Because `targets` are the
**closed-form true probabilities** (not Bernoulli realizations), the binned
statistic is deterministic. Monte Carlo confirms zero variance and zero bias at
every `n`:

| n | B | oracle-binned (fx1) | outcome-binned (standard) |
|---|---|---|---|
| 60 | 10 | **0.00000000** | 0.131786 |
| 250 | 10 | **0.00000000** | 0.066142 |
| 1000 | 10 | **0.00000000** | 0.032905 |

So the fx1 gate is **not** subject to the `B/N` noise floor of §1.8 / F-16, and
the module docstring's "keeps the ECE deterministic and free of single-draw
Monte Carlo noise" is accurate. The `0.05` threshold is reachable: an oracle
scores exactly `0.000000` and passes on all 8 seeds tested. Constant
forecasters fail closed (`f ≡ 1.0` → ECE 0.789, Z = NaN, `passed=False`;
`f ≡ 0.5` → ECE 0.289, Z = NaN, `passed=False`). The `observed_frequency`
field name is explained in both the `CalibrationBin` and module docstrings —
not a defect.

**What is wrong is gate power.** The module docstring names
`p → 0.5 + 0.9(p − 0.5)` as the "systematically shrunk oracle" control. Across
40 seeds at `n=60, B=10, ece_threshold=0.05`:

| forecaster | pass % | mean ECE | mean \|Z\| | Z-fail % | ECE-fail % |
|---|---|---|---|---|---|
| oracle (perfect) | 100.0% | 0.00000 | 0.669 | 0.0% | 0.0% |
| **0.90 shrink (the docstring's control)** | **87.5%** | 0.04207 | 1.372 | 12.5% | **0.0%** |
| 0.80 shrink | 0.0% | 0.08415 | 2.109 | 65.0% | 100% |
| 0.70 shrink | 0.0% | 0.12622 | 2.739 | 97.5% | 100% |
| 0.50 shrink | 0.0% | 0.21037 | 3.868 | 100% | 100% |
| **1.10 overconfident** | **77.5%** | 0.02853 | 1.072 | 22.5% | **0.0%** |
| 1.30 overconfident | 2.5% | 0.05401 | 3.145 | 70.0% | 85.0% |

**The documented negative control passes 87.5% of the time, and the ECE
criterion rejects it on 0% of seeds** — all rejections come from Spiegelhalter's
Z. The gate has essentially no power against a 10% shrinkage or a 10%
overconfidence, which is precisely the magnitude of calibration drift a
fine-tuning run would plausibly introduce. At `ece = 0.042` the control sits at
84% of the threshold.

**Why binning is the culprit:** because the shrinkage is monotone in `p`, all
within-bin errors share a sign, so the binned ECE collapses to the pointwise MAE
— verified: ECE is *identical* (0.0407) at B = 5, 10 and 20. Binning buys
nothing here, and the per-bin averaging is pure signal loss. Proper scores on the
same data separate the controls sharply and are also noise-free (targets are
oracle probabilities):

| forecaster | mean \|f − true\| | binned ECE | **Brier vs true p** | **log-loss vs true p** |
|---|---|---|---|---|
| oracle | 0.00000 | 0.00000 | **0.000000** | 0.269715 |
| 0.90 shrink | 0.04070 | 0.04070 | **0.001723** | 0.284465 |
| 0.80 shrink | 0.08140 | 0.08140 | 0.006892 | 0.310139 |
| 0.70 shrink | 0.12210 | 0.12210 | 0.015507 | 0.341754 |
| 0.50 shrink | 0.20350 | 0.20350 | 0.043074 | 0.418891 |
| 1.10 overconfident | 0.02917 | 0.02917 | 0.001015 | 0.438321 |
| 1.30 overconfident | 0.05987 | 0.05987 | 0.004966 | 1.029768 |

Brier-vs-truth separates the 0.90 shrink at `0.00172` vs the oracle's exact
`0.0` — a ratio of ∞, versus the ECE's `0.041` vs `0.000`. Log-loss separates
the *overconfident* controls (which Brier compresses) far better: 1.10 →
0.438 and 1.30 → 1.030 vs the oracle's 0.270.

**Fix.** Add Brier-vs-true-probability and log-loss-vs-true-probability as
**co-primary** gate criteria alongside ECE and Z. Both are already available
(`probability.brier_score`, `probability.log_loss`), both are proper scores
(honesty-compatible), and both are deterministic on this bank. A
`brier_threshold` derived from the seeded oracle's own value (≈0) plus a small
tolerance would give the gate real power against 10% drift. Keep ECE for
continuity but stop treating it as the discriminating criterion. Add a **gate
power test** asserting the documented negative control is *rejected* on ≥95% of
seeds — this is the test that should have existed and would have caught the
issue at authoring time.

---

### F-18 — `log_loss` caps a hard falsification at 27.631 nats (LOW — defensible, undocumented consequence)

`probability.py`, interior clip `eps = 1e-12`. Measured
`log_loss(p, y=1)`: `p = 0` → 27.631021; `1e−15` → 27.631021; `1e−12` →
27.631021; `1e−9` → 20.723266; `1e−6` → 13.815511; `1e−3` → 6.907755.
Monotone, finite, and the clip only bounds the blow-up — so this is **not a
bug**. The consequence worth documenting: `27.631 = −log(1e−12)` is an arbitrary
cap on how badly one maximally-overconfident wrong answer can hurt. At the fx1
bank size (`n = 60`) a single falsification moves the mean log-loss by
`27.631/60 = 0.460` nats — larger than the entire oracle-to-0.90-shrink gap
(0.270 → 0.284). So the `eps` choice materially affects rankings at small `n`,
and the log score is strictly proper only on the *interior* of the simplex.

**Fix.** Document `eps` and its consequence in the docstring; make `eps` a
required-visible parameter rather than a default where log-loss feeds a gate;
consider reporting the count of clipped entries alongside the score so a receipt
records that a boundary falsification occurred.

---

### F-19 — randomized PIT exists but is siloed in `models/qrf.py` (LOW — see gap G-1)

`QuantileRegressionForest.pit` and `.pit_oob_train` implement
`F⁻(y) + V·(F(y) − F⁻(y))` correctly, and the leave-one-out training variant is
a genuinely careful design (no fitting tree or response contributes to its own
diagnostic). But it is not exported through `quant_fund.metrics`, is usable only
by one model class, and cites neither Czado–Gneiting–Held (2009) nor the
randomization it performs. Meanwhile the metrics-layer `pit_values` is the
non-randomized variant with the F-05 boundary defect. See G-1.

---

### F-20 — housekeeping: a stray untracked `src/src/` duplicate tree (out of lane, noted)

`rg` over `src` matched **two** copies of `quant_fund/metrics/probability.py`:
the canonical `src/quant_fund/...` and `src/src/quant_fund/...`. Audit:

- 151 `.py` files under `src/src/` (all `quant_fund`, no `fx1`), plus ~hundreds
  of `__pycache__/*.pyc` for **three** interpreter versions (3.12/3.13/3.14);
- **118 of 151 have drifted** from canonical, 25 identical, 8 orphaned;
- the drift includes *reverted refactors*: `src/src/.../probability.py`
  re-inlines `_as_1d` / `_require_same_length` instead of importing them from
  `quant_fund.utils.series`, and is 4 lines longer at a different offset;
- `git status --porcelain src/src` → `?? src/src/`; **0 tracked files**;
- `pyproject.toml` ships `packages = ["src/fx1", "src/quant_fund"]` /
  `["fx1", "quant_fund"]`, so the stray tree is **not packaged**;
- `ruff check src` reports 2 findings (`C901`), **none** under `src/src`.

**Impact on this lane: none.** Not importable as `quant_fund` (only as
`src.src.quant_fund`), not packaged, not linted, not tested. All findings above
were verified against the canonical tree. It is nonetheless a live
footgun: an editor or grep-driven edit could land in the stale copy and appear
to succeed, and the stale `probability.py` is exactly the file this lane
audits. **Recommend deleting `src/src/` and adding it to `.gitignore`** —
reported here, not actioned, because the task scope forbids modifying any file
other than this document.

---

### Non-findings (positive controls — verified correct, do not change)

- **NF-1 — `expected_calibration_error` bin-edge parenthesisation is correct.**
  The line `sel = (p >= edges[i]) & (p < edges[i + 1] if i < int(n_bins) - 1 else p <= edges[i + 1])`
  *looks* like a precedence bug (Python's conditional expression binds looser
  than `&`). AST parse confirms the `IfExp` is inside the right-hand paren:
  `BinOp(Compare(p >= edges[i]), BitAnd, IfExp(test=i < n_bins-1,
  body=p < edges[i+1], orelse=p <= edges[i+1]))`. Element-wise comparison on a
  5-bin example matches the intended selection exactly in all bins, and
  `E[ECE]` matches an independent textbook reference to 5 decimals at
  `n = 60…20000`. **No defect.**
- **NF-2 — `fissler_ziegel_loss` is jointly proper.** Analytically,
  `∂E[S]/∂v = 0 ⟹ v = q_α(L)` and then `E[S] = ES/e + log e − 1 ⟹ e = ES_α`.
  Empirically, at `α = 0.95` and `α = 0.99` with `m = 6e5` Student-t(5) draws:
  every single-coordinate perturbation of VaR (×0.70…×1.20) and ES (×0.85…×1.30)
  increases the loss; max violation `0.000e+00`; a 25×25 grid over
  `VaR × [0.6,1.8]`, `ES × [0.6,1.8]` has its global minimum **exactly** at the
  truth cell `(×1.00, ×1.00)`. The docstring's note that an earlier variant
  without the `1/(1−α)` factor minimised at `e = (1−α)·ES` and was corrected is
  consistent with this. *(An earlier probe appeared to show improperness; it had
  used `mean(L[L ≤ VaR])` as the ES target instead of the tail mean
  `E[L | L ≥ VaR_α]`. The function is correct.)* Residual item: `_require_alpha_coverage`
  accepts `α = 0.05` for a coverage-convention function (F-03).
- **NF-3 — `energy_score` is the fair U-statistic and is proper.** Uses
  `Σ/(2n(n−1))`; measured `E[ES]` has an interior minimum at `σ = 1`
  (0.951 / 0.870 / 0.890 / 0.946 / 1.029 at `σ = 0.5/0.75/1/1.5/2`). Correct,
  and the docstring's caveat that the U-statistic can dip slightly below 0 for a
  calibrated finite ensemble is accurate and worth keeping.
- **NF-4 — the fx1 oracle-binned ECE is genuinely noise-free.** See F-17; the
  module docstring's claim is verified, not merely asserted.
- **NF-5 — `crps_from_quantiles` cell widths are uniform.** `Δτ = 0.05` for every
  cell including the first on the 19-point grid; there is no ragged first cell.
  Only tail truncation is at issue (F-14).
- **NF-6 — `pinball_loss`, `crps_gaussian`, `crps_gaussian_mixture`,
  `log_score_gaussian`, `vol_eval.{mse,mse_log,hmse,mae}`,
  `var_backtest.kupiec_test` (including its `x=0` branch), `murphy_decomposition`,
  `winkler_interval_score`, `dawid_sebastiani`, `hosmer_lemeshow`,
  `density_forecast.{berkowitz_test,pit_autocorrelation}`,
  `inference.{newey_west_variance,benjamini_hochberg}`** — inspected and found
  consistent with their cited formulas. `probability.kupiec_pof` and `pit_ks`
  already use `stats.*.sf`, i.e. the repo knows the correct p-value idiom.
- **NF-7 — `bench_energy_score`** (`research/benches_w810.py:105`) uses plain
  `energy_score`, **not** `threshold_energy_score`. The F-01 defect therefore
  does not contaminate any wave-8 research receipt.

---

## 4. Gap list

Capabilities absent from the repo entirely, ordered by value-per-unit-effort.

| ID | Gap | Reference | Value |
|---|---|---|---|
| **G-1** | **Randomized PIT in the metrics layer.** `pit_randomized(y, cdf_minus, cdf_at, rng)` exported from `quant_fund.metrics`, usable by any model with a discrete/binned/grid-truncated predictive — not only QRF. Generalise `models/qrf.py::pit` / `pit_oob_train` and cite Czado–Gneiting–Held (2009). | Czado, Gneiting & Held 2009, *Biometrics* 65(4):1254–1261 | **High** — fixes F-05 at the root; the only valid PIT for count/binned data |
| **G-2** | **WIS + its exact dispersion / overprediction / underprediction decomposition.** `wis(y, lower, upper, alphas, median)` returning the three components. The repo has `winkler_interval_score` (single interval) but no aggregation and no decomposition. | Bracher, Ray, Gneiting & Reich 2021, *PLOS Comp. Biol.* 17(6):e1008618 | **High** — turns "CRPS worsened" into a diagnosable cause; proper by construction |
| **G-3** | **Modern calibration-error estimators.** `smCE` (Błasiok et al. 2023), `mmCE` (Kumar et al. 2019), `debiased_ece` (Roelofs et al. 2022), and equal-mass binning (Naeini et al. 2015). Measured: debiasing removes ~58% of the `B/N` bias. | see §1.8 | **High** — F-16; ECE is currently the only option and its bias is unreported |
| **G-4** | **DM-family inference for proper scores.** Loss-agnostic HLN (F-09), Andrews–Monahan prewhitening, BH/e-value multiplicity control on pairwise matrices (F-08), Giacomini–White conditional predictive ability *with* correct CVs, GR fluctuation test with `m/T`-tabulated critical values (F-11). | Harvey–Leybourne–Newbold 1997; Andrews & Monahan 1992; Giacomini & White 2006; Giacomini & Rossi 2010 | **High** — every model-comparison receipt depends on this |
| **G-5** | **Fair/bias-corrected ensemble CRPS as the default** (F-02), plus an explicit `n_members` stamp in every CRPS receipt so bias is auditable. | Northrop 2022; Jordan, Krüger & Lerch 2019, *JSS* | **High** — small ensembles are the normal case |
| **G-6** | **Threshold-weighted CRPS** (1-D, where the construction is clean) as the honest replacement for the improper `threshold_energy_score`, plus a **strictly proper multivariate alternative** where ES is blind to dependence. | Gneiting & Ranjan 2013; Stroud & Stein 2020; Székely & Rizzo 2013; Ziegel 2016; Mühlemann & Ziegel 2021 | **Med–High** — F-01 replacement; ES is not strictly proper for `d ≥ 2` |
| **G-7** | **CRPS decomposition beyond WIS.** Gneiting, Heygster, Jonkman, Kley, Krüger, Schmitz & Siegert (2023) give an miscalibration/discrimination/uncertainty decomposition for CRPS — the CRPS analogue of Murphy's decomposition (which the repo *does* have for Brier). | Gneiting et al. 2023 | **Med** — symmetric with the existing `murphy_decomposition` |
| **G-8** | **Log score for general predictive densities.** The repo has `log_score_gaussian` only. A grid/kernel-density log score would let non-Gaussian heads be compared on the same axis. | Gneiting & Raftery 2007 §3.1 | **Med** |
| **G-9** | **PIT serial-dependence battery for the *randomized* PIT**, and a Diebold–Günther–Tay `χ²` that is valid post-randomization. `pit_histogram` exists but assumes a non-randomized PIT in `(0,1)`. | Diebold, Günther & Tay 1998; Berkowitz 2001 | **Med** — pairs with G-1 |
| **G-10** | **Calibration under distribution shift / conditional calibration.** All current diagnostics are marginal. `localized_conformal.py` and `regime_eval.py` exist in the repo; a conditional-calibration diagnostic (Vovk 2012; Feldman & Ringel 2023) would connect the conformal lane (SOTA 04) to this one. | Vovk 2012; Feldman & Ringel 2023 | **Med** |
| **G-11** | **e-value-based forecast comparison as the primary route.** `metrics/evalues.e_process_dm` and `metrics/anytime_fdr.e_bh` exist; e-values compose across tests without a multiplicity correction and are anytime-valid, which fits the receipt/verifier model better than BH. Currently opt-in behind `include_e_process=False`. | Vovk & Wang 2021; Wang & Ramdas 2022; Choe & Ramdas 2022 | **Med–High** |
| **G-12** | **Strict-propriety test harness.** A reusable fixture that, for any candidate score `S`, asserts `E[S(F_true)] ≤ E[S(F_perturbed)]` over a perturbation family (scale, location, shape) and a seeded observation distribution. Would have caught F-01, F-13 and NF-2 at authoring time. | — | **High** — cheap, general, prevents recurrence |

---

## 5. Adoption plan

Tiering by (risk of the change) × (value). Every item is honesty-contract-safe:
all proposed metrics are proper scores or inference corrections, so neither
`FORBIDDEN_RESEARCH_METRIC_KEYS` nor `FORBIDDEN_HEADLINE_TOKENS` changes, and
`tests/fx1/test_honesty_inheritance.py` needs no update.

### Tier 0 — pure additions, no value change, no receipt invalidation

| Change | Module | Tests |
|---|---|---|
| G-12 strict-propriety fixture (`perturb_scale`, `perturb_location`, seeded obs) | `tests/property/test_score_propriety.py` (new) | self-testing |
| G-1 `pit_randomized` exported from metrics | `metrics/scoring.py` or new `metrics/pit.py`; re-export in `metrics/__init__.py` | `tests/unit/core/test_pit_randomized.py` (new): uniformity under a discrete predictive; LOO variant; determinism under a fixed seed |
| G-2 `wis` + 3-component decomposition | `metrics/calibration2.py` | `tests/unit/research/test_calibration2.py`: hand-computable `K=1` case; `dispersion+over+under == wis` identity; `wis → crps_from_quantiles` as `K→∞` |
| G-3 `smCE`, `mmCE`, `debiased_ece`, equal-mass binning | `metrics/probability.py` (new functions; **do not touch** `expected_calibration_error`) | `tests/unit/core/test_probability.py`: bias table of §1.8 reproduced; `debiased_ece ≤ ece` on calibrated data; smCE `O(1/√N)` rate |
| F-17 gate power: add Brier-vs-truth and log-loss-vs-truth as **co-primary** criteria | `fx1/eval/calibration_eval.py` | `tests/fx1/test_calibration_eval.py`: **the documented 0.90-shrink control is rejected on ≥95% of seeds**; oracle still passes exactly |
| F-11(c) demote `fluctuation_test.reject` → report `sup`/`stats` only, fix the truncated docstring | `metrics/forecast_eval.py` | `tests/unit/research/test_forecast_eval.py`: size test under H₀ at 3 window fractions if `reject` is retained |

### Tier 1 — bug fixes that change values (require deliberate frozen-value updates + receipt notes)

| Change | Module | Tests | Receipt impact |
|---|---|---|---|
| F-01 guard `weight >= √2` + correct the false propriety docstring | `metrics/energy_score.py` | `tests/unit/core/test_energy_score.py`: **update** `test_hand_computable_threshold_weighted_3x2` (it asserts a negative score at `weight=2.0`); add a bounded-below/propriety test; xfail-with-reason for `weight >= √2` | any receipt using `threshold_energy_score` at `weight >= √2` must be re-run |
| F-02 fair U-statistic CRPS (prefer `estimator="fair"\|"plugin"`, default `"fair"`) | `metrics/scoring.py` | `tests/regression/test_scoring_frozen.py` (add `crps_empirical`); `tests/unit/research/test_scoring_edges.py`: bias `= σ/(n√π)` at n ∈ {5,10,20,50,100} | receipts quoting small-`n` CRPS carry a known `+E\|X−X′\|/(2n)` offset — stamp `n_members` |
| F-03 `alpha` guards on `kupiec_pof`, `christoffersen_cc`, `_require_alpha_coverage` | `metrics/probability.py`, `metrics/scoring.py` | `tests/unit/core/test_probability.py`: inversion raises with a message naming the right function | none (only rejects previously-silent misuse) |
| F-04 `x=0`/`x=n` limiting LR in `kupiec_pof` (or delete it and re-export `var_backtest.kupiec_test`) | `metrics/probability.py` | frozen value `−2n log(1−α)`; cross-module consistency test vs `var_backtest.kupiec_test` | none |
| F-06 `1.0 - cdf` → `.sf` at all ~20 sites + a lint guard against regression | `metrics/{vol_eval,density_forecast,var_backtest,panel,dependence,es_backtest,forecast_eval}.py`, `models/{dynamic_panel,event_study}.py` | grep-guard test; p-value monotonicity/precision test at statistic = 80 | p-values that were exactly `0.0` become finite — **strictly better for BH ordering** |
| F-07 `scoring.qlike` mask `y > 0` + `n_zero_proxy` counter | `metrics/scoring.py` | `test_scoring_frozen.py::test_qlike_formula_frozen` unaffected (`y = e > 0`); add a `y=0` test asserting exclusion, not a 20-nat penalty | receipts over data containing zero realized variance change |
| F-10 `clark_west_test` docstring/param rename to *raw* errors + `norm.sf` | `metrics/forecast_eval.py` | hand-computed nested example where CW flips the raw MSPE-difference sign | any receipt that followed the old docstring was already meaningless |
| F-12 `_lrv` degenerate-segment NaN + `n_degenerate_windows` | `metrics/forecast_eval.py` | constant-segment test: statistic is NaN, not `2.2e10` | `fluctuation_test` sups may change |
| F-13 `variogram_score` `/ (d(d−1))` (+`normalize=False` escape hatch) + fix garbled docstring | `metrics/calibration2.py` | scale test: VS is `O(1)` in `d` | multi-asset VS receipts change by `d(d−1)` |
| F-08 BH (and e-value) columns on `pairwise_diebold_mariano`, `preferred` recomputed on adjusted p | `metrics/inference.py` | FWER test: 8 identical models → adjusted-FWER ≤ ~0.10 (measured 0.088 post-hoc) | pairwise DM receipts gain columns; **unadjusted claims become unquotable** |

### Tier 2 — structural (design work, cross-module)

| Change | Modules | Notes |
|---|---|---|
| F-09 loss-agnostic `hln_test` unified with `diebold_mariano`; optional Andrews–Monahan prewhitening | `metrics/forecast_eval.py`, `metrics/inference.py` | one DM/HLN entry point serving CRPS, QLIKE, pinball, ES, log score; aligns the HAC lag rule |
| G-4 full comparison battery: HLN + CW + GW + GR with correct CV tables, all loss-agnostic, all multiplicity-controlled | `metrics/forecast_eval.py`, `metrics/inference.py` | the "compare two forecasts" story becomes receipt-grade |
| G-6 threshold-weighted **CRPS** (1-D, proper construction) + a strictly proper multivariate score for `d ≥ 2` | `metrics/scoring.py`, `metrics/energy_score.py` | replacement path for F-01 if tail emphasis is needed in a claim |
| G-11 e-value comparison as the *default* route (anytime-valid, composes without correction) | `metrics/evalues.py`, `metrics/anytime_fdr.py`, `metrics/inference.py` | fits the verifier/receipt model better than BH |
| G-7 CRPS miscalibration/discrimination/uncertainty decomposition | `metrics/scoring.py` | symmetric with the existing Brier `murphy_decomposition` |
| G-10 conditional calibration / calibration under shift | new `metrics/conditional_calibration.py` | links to SOTA 04 (conformal) and `validation/regime_eval.py` |
| F-05 unify the PIT boundary contract across `pit_values` / `_pit` / `pit_ks` | `metrics/scoring.py`, `metrics/density_forecast.py`, `metrics/probability.py` | out-of-grid fraction becomes a first-class reported diagnostic |
| F-20 delete `src/src/`, add to `.gitignore` | repo hygiene | **out of this lane's scope** — reported only, not actioned |

### Sequencing

1. **Tier 0 first** — it is additive, cannot invalidate a receipt, and G-12's
   propriety fixture is what makes the Tier 1 fixes *verifiable*.
2. **F-06 before F-08** — BH is meaningless on p-values that tie at exactly 0.0.
3. **F-01 and F-02 together**, as one commit, since both change score values and
   both need frozen-test updates; land with an `INFLIGHT` note naming the
   affected receipt classes.
4. **F-17 with Tier 0** — it is additive (new co-primary criteria) and it is the
   one fix that directly strengthens a *gate*, which is the highest-leverage
   place to spend effort.
5. Tier 2 last, behind design review.

Per `AGENTS.md`: fx-1 lane changes (F-17) land directly on `main`; the
cross-cutting `quant_fund.metrics` changes should go through a PR. Run
`make lint`, `make typecheck`, `make test`, `make fx1-gate` before committing.

---

## 6. Honesty-contract interactions

- **Nothing here proposes a forbidden metric.** Every addition (WIS, smCE,
  mmCE, debiased ECE, randomized PIT, threshold-weighted CRPS, HLN, BH,
  e-values) is a proper score or an inference correction.
  `FORBIDDEN_RESEARCH_METRIC_KEYS` (`research/catalog/registry.py`) and
  `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` stay in sync and unchanged;
  `tests/fx1/test_honesty_inheritance.py` needs no edit.
- **F-01 is an honesty issue, not just a math issue.** An improper score can be
  driven to −∞ by inflating variance. If it ever headlined a receipt, the
  receipt would faithfully record a conclusion that rewards the opposite of
  sharpness. The docstring's incorrect propriety claim is the more dangerous
  half: it would let a reviewer wave the score through.
- **F-08 is an honesty issue.** Receipts are immutable evidence, and a pairwise
  DM matrix with FWER = 0.62 under a global null would seal multiplicity
  artefacts as reproducible findings. Reproducibility is not the same as
  validity — the receipt hash would reproduce the false claim perfectly.
- **F-02 is an honesty issue at small `n`.** A 20% CRPS inflation that depends
  on ensemble size does not cancel across models, so a league table can be
  ordered by ensemble size rather than skill.
- **SYNTHETIC labelling.** Every simulation in §3 is a seeded synthetic draw
  used to measure *estimator* behaviour (bias, size, propriety). None is market
  evidence, and none should be quoted as such. Probe seeds are recorded in §8.
- **No live-trading claim** is made anywhere in this lane; there is no broker
  connectivity and none is proposed. See `docs/INSTITUTIONAL_READINESS.md`.
- **F-15's fail-closed NaN is the honest failure mode** and should be preserved
  even if the guard is later relaxed behind an opt-in: a refused score is
  better than an uninterpretable one.

---

## 7. Reproduction

All numbers in §3 come from standalone probes run against the installed package
(`.venv/Scripts/python.exe`, `sys.path` → `D:\dipcatcher\src`). Seeds
`20260928` (probe 12–17, 19) and `20260927` (`benches_w810._SEED`, unchanged).
Scripts and raw output:

| Probe | Covers | Script | Output |
|---|---|---|---|
| 10 | NF-1 AST parse of the ECE bin selection; F-20 first sighting | `%TEMP%\scoring_probe10.py` | `scoring_probe10.txt` |
| 11 | F-20 stray-tree scope/drift/packaging/lint audit | `%TEMP%\scoring_probe11.py` | `scoring_probe11.txt` |
| 12 | F-02 bias table, F-01 propriety sweep, F-07, F-05, F-04, F-14, F-13, F-12, F-06, F-15, F-03 | `%TEMP%\scoring_probe12.py` | `sp12.txt` |
| 13 | F-16 corrected (probe 12's block G drew `p` and `y` independently — see below), F-07 corrected | `%TEMP%\scoring_probe13.py` | `sp13.txt` |
| 14 | F-17: fx1 oracle-binned vs outcome-binned ECE; binning cancellation | `%TEMP%\scoring_probe14.py` | `sp14.txt` |
| 15 | F-17: gate power over 40 seeds, 7 forecasters | `%TEMP%\scoring_probe15.py` | `sp15.txt` |
| 16 | F-09, F-06, F-12, F-11 (size), F-08 (FWER/BH), F-10 | `%TEMP%\scoring_probe16.py` | `sp16.txt` |
| 17 | F-03 silent inversion, F-18, F-14 cell widths + tails, NF-2 (first, flawed) | `%TEMP%\scoring_probe17.py` | `sp17.txt` |
| 18 | NF-2 second attempt — **also flawed** (ES target was `mean(L[L≤VaR])`) | `%TEMP%\scoring_probe18.py` | `sp18.txt` |
| 19 | NF-2 corrected (tail-mean ES), analytic derivation, 25×25 grid; NF-3 | `%TEMP%\scoring_probe19.py` | `sp19.txt` |
| 20 | F-01 exact breakdown weight `√2`, dimension-freeness across `d=1…16`, boundary drift | `%TEMP%\scoring_probe20.py` | `sp20.txt` |

**Two probe bugs corrected during this audit**, recorded so the numbers above
can be trusted and so the mistakes are not repeated:

1. **Probe 12 block G** drew probabilities from `rng.beta(2,2,n)` and outcomes
   from an *independent* `rng.random(n) < q2` draw, so the "perfectly
   calibrated" case was not calibrated — it reported `E[ECE] ≈ 0.23` at `n=60`.
   Probe 13 shares one draw (`y = rng.random(n) < q`) and gives the correct
   `0.1318`, which matches an independent textbook ECE implementation to 5
   decimals. **The `0.132` figure in §1.8 / F-16 / F-17 is the correct one.**
2. **Probes 17 and 18 block 6** tested `fissler_ziegel_loss` propriety by
   scaling VaR and ES *together*, which moves the pair to a different quantile
   level and is not a propriety test; probe 18 then used `mean(L[L ≤ VaR])` as
   the ES target instead of the tail mean `E[L | L ≥ VaR_α]`. Probe 19 corrects
   both and the function is **proper** (NF-2).

Reproduction is a few minutes of CPU per probe; none requires network, market
data, or `MOONSHOT_API_KEY`.

---

## 8. References

**Foundations**
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477):359–378.
- Gneiting, T. (2011). Making and evaluating point forecasts. *JASA* 106(494):746–762.
- Savage, L. J. (1971). Elicitation of personal probabilities and expectations. *JASA* 66(336):783–801.
- Osband, K. H. (1989). Statistical investment analysis and the theory of scoring rules. *Journal of Accounting and Economics* 11(1):27–75.
- Ziegel, J. F. (2016). Coherence and elicitability. *Mathematical Finance* 26(4):901–918.

**CRPS / quantile scores**
- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica* 46(1):33–50.
- Matheson, J. E. & Winkler, R. L. (1976). Scoring rules for continuous probability distributions. *Management Science* 22(10):1087–1096.
- Grimit, E. P., Gneiting, T., Berrocal, V. J. & Milks, G. A. (2006). The continuous ranked probability score for circular data. *Mon. Wea. Rev.* 134(10):292–299.
- Jordan, A., Krüger, F. & Lerch, S. (2019). Evaluating predictive count data distributions in retail sales forecasting. *JRSS-A* 182(3):787–803.
- Jordan, A., Krüger, F. & Lerch, S. (2019). `scoringRules`: R package for proper scoring rules. *Journal of Statistical Software* 90(1).
- Northrop, P. J. (2022). `properscoring`: proper scoring rules via the CRPS, and how to use them. arXiv:2110.12636.
- Bracher, J., Ray, E. L., Gneiting, T. & Reich, N. G. (2021). Evaluating epidemic forecasts in an interval format. *PLOS Computational Biology* 17(6):e1008618.
- Gneiting, T., Heygster, J., Jonkman, J., Kley, T., Krüger, F., Schmitz, M. & Siegert, J. (2023). A decomposition of the CRPS. (miscalibration / discrimination / uncertainty)
- Chernozhukov, V., Fernández-Val, I. & Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica* 78(3):1093–1125.
- Winkler, R. L. (1994). Evaluating probabilities: asymmetric scoring rules. *Management Science* 40(11):1395–1405.

**Energy score / multivariate**
- Székely, G. J. (2003). *E-statistics: The Energy of Statistical Samples*. Technical Report BGSU No. 03-05.
- Székely, G. J. & Rizzo, M. L. (2013). Energy statistics: a class of statistics based on distances. *Journal of Statistical Planning and Inference* 143(8):1249–1272.
- Gneiting, T. & Ranjan, R. (2013). Threshold-weighted scoring rules. (talk/working-paper line of work; see also Stroud & Stein 2020)
- Stroud, J. S. & Stein, M. L. (2020). Proper scoring rules for spatial asymmetries. *JRSS-C* 69(5):1277–1301.
- Scheuerer, M. & Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of wind and precipitation. *Mon. Wea. Rev.* 143(7):2439–2454.
- Mühlemann, A. P. & Ziegel, J. F. (2021). Isotonic distributional regression. (strict propriety of the energy score; related discussion)
- Dawid, A. P. & Sebregondi, E. (1999). On the existence of proper scoring rules. *Bayesian Analysis* / technical note (Dawid–Sebastiani score).

**Volatility losses**
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 164(1):20–50.
- Bollerslev, T. & Ghysels, E. (1996). Periodic autoregressive conditional heteroscedasticity. *JBES* 14(2):139–151. (HMSE)
- Andersen, T. G., Bollerslev, T., Diebold, F. X. & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica* 71(2):579–625.

**PIT / density diagnostics**
- Rosenblatt, M. (1952). Remarks on a multivariate transformation. *Ann. Math. Statist.* 23(3):470–472.
- Diebold, F. X., Günther, T. A. & Tay, A. S. (1998). Evaluating density forecasts, with applications to financial risk management. *International Economic Review* 39(4):863–883.
- Berkowitz, J. (2001). Testing density forecasts, with applications to risk management. *JBES* 19(4):465–474.
- Czado, C., Gneiting, T. & Held, L. (2009). Predictive model assessment for count data. *Biometrics* 65(4):1254–1261. **(randomized PIT)**
- Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69(2):243–268.
- Hong, Y., Li, H. & Zhao, F. (2007). Forecasting conditional volatility and applications in risk management. *Review of Economics and Statistics* 89(4):650–655.

**Calibration error**
- Murphy, A. H. (1973). A new vector decomposition of the probability forecast error score. *J. Appl. Meteor.* 12(4):595–600.
- Hosmer, D. W. & Lemeshow, S. (1980). Goodness of fit for the multiple logistic regression model. *Communications in Statistics* A9(10):1043–1069.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*, PMLR 70:1321–1330.
- Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *AAAI 2015*.
- Kumar, A., Sarawagi, S. & Jain, P. (2019). Trainable calibration measures for neural networks from kernel mean embeddings. *NeurIPS 2019*. **(mmCE)**
- Roelofs, R., Fridovich-Keil, S., Palatri, C. & Recht, B. (2022). Mitigating bias in calibration error estimation. *AISTATS 2022*, PMLR 151.
- Błasiok, J., Gopalan, P., Hu, Y. & Nakkiran, P. (2023). A unifying theory of distance to calibration. *STOC 2023*.
- Zhao, J., Roelofs, R., Karras, S., Vemuri, A. & Frankle, J. (2024). Top-label calibration and multiclass-to-binary reductions.
- Spiegelhalter, D. J. (1986). Probabilistic prediction in patient management and clinical trials. *Statistics in Medicine* 5(5):421–433.

**Risk backtesting**
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives* 3(2):73–84.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review* 39(4):841–862.
- Acerbi, C. & Székely, B. (2014). Back-testing expected shortfall. *Risk* 27(11):76–81.
- McNeil, A. J. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. *Journal of Empirical Finance* 7(3–4):271–300.
- Du, Z. & Escanciano, J. C. (2017). Backtesting expected shortfall: accounting for tail risk. *Management Science* 63(4):940–958.
- Fissler, T. & Ziegel, J. F. (2016). Higher order elicitability and Osband's principle. *Annals of Statistics* 44(4):1680–1707.
- Nolde, N. & Ziegel, J. F. (2017). Elicitability and backtesting: perspectives for banking regulation. *Annals of Applied Statistics* 11(4):1833–1874.
- Basel Committee on Banking Supervision (1996). Amendment to the Basel Capital Accord (backtesting framework / traffic-light zones).

**Forecast comparison & inference**
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES* 13(3):253–263.
- Harvey, D., Leybourne, S. J. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13(2):281–291.
- Clark, T. E. & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics* 138(1):291–311.
- Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6):1545–1578.
- Giacomini, R. & Rossi, B. (2010). Detecting and predicting forecast instabilities. *Econometrica* 78(2):677–714.
- Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3):703–708.
- Andrews, D. W. K. & Monahan, J. C. (1992). An improved heteroskedasticity and autocorrelation-consistent covariance matrix estimator. *Econometrica* 60(4):953–966.
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5):1097–1126.
- Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. *Econometrica* 79(2):453–497.
- Diebold, F. X. & Mariano, R. S. (2002). Comparing predictive accuracy — twenty years later. *JBES* 20(1):134–144.
- Mariano, R. S. (2021). Comparing predictive accuracy: twenty years later. (retrospective / survey line)
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *JRSS-B* 57(1):289–300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics* 29(4):1165–1188.
- Vovk, V. & Wang, R. (2021). E-values: calibration, combination, and applications. *Annals of Statistics* 49(3):1215–1236.
- Wang, R. & Ramdas, A. (2022). False discovery rate control with e-values. *JRSS-B* 84(3):822–852.
- Choe, S. K. & Ramdas, A. (2022). Nonparametric e-variables for comparing predictive accuracy / e-processes for model comparison.
- Ramdas, A., Grünwald, P., Vovk, V. & Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science* 38(4):576–601.
- Vovk, V. (2012). Conditional validity of inductive conformal predictors. *ACML 2012*, PMLR 29:475–490.
- Feldman, S. & Ringel, R. (2023). Achieving high-precision conditional calibration via calibrated-probability estimation.
