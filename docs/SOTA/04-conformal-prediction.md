# 04 — Conformal prediction for time series (SOTA lane)

Lane: **CONFORMAL PREDICTION FOR TIME SERIES**. Status: research notes +
gap analysis + adoption plan, 2026-09-28. This document summarizes
state-of-the-art practice (2014–2026) for distribution-free uncertainty
quantification on non-exchangeable sequences; audits the dipcatcher harness
(`src/quant_fund/models/*conformal*`, `metrics/conformal.py`,
`pipeline/forecast/conformal.py`, `research/benches/intervals.py`,
`research/benches_w810.py`) against it; and proposes a concrete adoption
plan. It modifies no code — it is the design record for the next conformal
wave.

Honesty contract applies: every measurement discussed here is a **proper
score or coverage diagnostic** — coverage rate, interval width, pinball,
Winkler interval score, Kupiec POF, Christoffersen CC, PIT, CRPS, conformal
p-values, e-values. No Sharpe/Sortino/Calmar/P&L/NAV is headlined anywhere
(`FORBIDDEN_RESEARCH_METRIC_KEYS` in `quant_fund.research.catalog`); any
proposed metric key below is chosen to survive
`family_blob_forbidden_metrics_absent` (no `*_pnl`, `*_sharpe`, … tokens).
SYNTHETIC results are correctness tests, never market evidence. No
live-trading claims; conformal intervals feed position caps
(ADR-014), never a broker.

Related in-tree docs: `docs/SOTA_CANON_ROADMAP_2026_09.md` §2.2,
`docs/SOTA_GAP_ANALYSIS.md` (day-wave ledger), ADRs
`docs/decisions/008-conformal-wraps-baselines.md` (conformal wraps, never
rewrites, PIT/pinball/CRPS), `009-mondrian-normalized-cqr.md`,
`010-evalues-anytime-valid.md`, `011-jackknife-plus.md`,
`012-conformal-risk-control.md`, `013-weighted-conformal.md`,
`016-cv-plus.md`, `017-localized-conformal.md`, `018-conformal-topk.md`,
`019-online-crc.md`, `020-portfolio-conformal.md`,
`docs/INSTITUTIONAL_READINESS.md`.

---

## 1. Verdict up front

The harness is **already one of the deepest conformal-prediction stacks
outside the original authors' codebases**: split CQR, Mondrian CQR/ACI,
ACI, AgACI/FACI-EG, EnbPI, weighted (covariate-shift) CQR, localized
(kernel) CQR, Jackknife+, CV+, batch CRC, online CRC, conformalized
Student-t heads, conformal top-k ranking, conformal test martingales
(WATCH), and a portfolio conformal family — all with coverage/width/Kupiec
outputs wired into research receipts and H-table consistency gates (H7,
H8, H10, H11, H12, H15, H18, H19).

The remaining SOTA gaps are **structural, not numerical**:

1. **No sequential/residual-dynamics calibrator in the operational path.**
   `pipeline/forecast/conformal.py::conformal_sets_asof` — the wrapper that
   stamps `AssetForecast.interval_lo/hi/alpha/method` consumed by
   `portfolio/interval_risk.py` — implements only *static* split CQR and
   Mondrian CQR (`IntervalMethod = Literal["mondrian_cqr", "split_cqr"]`).
   ACI/AgACI/EnbPI/online-CRC exist as models and benches but are never
   exposed as decision-date interval producers. SPCI (Xu & Xie 2023) and
   Conformal PID (Angelopoulos, Candès & Tibshirani, NeurIPS 2023) are
   absent entirely.
2. **No multi-horizon / trajectory calibration.** Everything is calibrated
   per (label, horizon) independently at one-step resolution; there is no
   MSCP-style horizon-dependent α allocation, no CopulaCPTS joint-trajectory
   region, no Bonferroni-corrected multi-step envelope. `models/hstep.py`
   produces h-step *distributions* but they are never conformally calibrated.
3. **No modern conditional-coverage diagnostic.** `metrics/conformal.py`
   has tercile `conditional_coverage` + `worst_slice_coverage`; the 2025
   literature (ERT — Braun, Holzmüller, Jordan & Bach, arXiv:2512.11779;
   Winkler decompositions) supersedes binned coverage with classifier-based
   excess-risk metrics that have real statistical power at lab sample sizes.

Adoption priority (leverage per unit of new math, detailed in §4):
**(1)** wire the existing sequential calibrators (ACI/AgACI/EnbPI/online
CRC) into `conformal_sets_asof` behind an extended `IntervalMethod` literal
with honest scope labels; **(2)** add SPCI + Conformal PID as
`models/spci.py` / `models/conformal_pid.py` (both reuse the ACI plumbing);
**(3)** add horizon-aware calibration (per-h split + Bonferroni/α-split
envelope for h-step bands, CopulaCPTS later); **(4)** add ERT-style
conditional-coverage diagnostics to `metrics/conformal.py` and the bench
families. All four stay inside ADR-008: conformal wraps the baselines,
never rewrites their proper scores.

---

## 2. Technique summaries (literature SOTA, with citations)

### 2.1 Foundations: split conformal & exchangeability

- **Split conformal prediction.** Vovk, Gammerman & Shafer (2005),
  *Algorithmic Learning in a Random World*; Lei, G'Sell, Rinaldo, Tibshirani
  & Wasserman (2018, *JASA* 113:1660–1672). Calibration scores
  \(s_i\) on a held-out exchangeable sample; the
  \(\lceil (n+1)(1-\alpha)\rceil\)-th smallest score gives exact marginal
  coverage \(\ge 1-\alpha\), attainable iff \(\alpha \ge 1/(n+1)\). Repo:
  `metrics/conformal.py::conformal_quantile` implements this exact rule
  (with an honest clip-to-max fallback when the level is unattainable).
- **Impossibility of exact conditional coverage.** Foygel Barber, Candès,
  Ramdas & Tibshirani (2021, *Information and Inference* 10(4):921–940;
  arXiv:1903.04684). Any distribution-free method with exact conditional
  coverage must have infinite expected width. Consequence: every claim in
  this lane is *marginal* validity plus *approximate/diagnosed* conditional
  behavior — which is exactly how the repo's `coverage_guarantee_scope =
  "marginal_exchangeable"` honesty key (Wave 41) frames Jackknife+/CV+.
- **Tutorial.** Angelopoulos & Bates (2021), *A gentle introduction to
  conformal prediction* (arXiv:2107.07511) — the standard entry point;
  covers time-series and shift extensions.

### 2.2 CQR family (heteroscedastic, static)

- **Conformalized Quantile Regression (CQR).** Romano, Patterson & Candès
  (2019, NeurIPS 32; arXiv:1905.03222). Score
  \(s_i = \max\{\hat q_{\alpha/2}(x_i) - y_i,\; y_i - \hat q_{1-\alpha/2}(x_i)\}\);
  intervals \([\hat q_{\alpha/2} - \hat q,\ \hat q_{1-\alpha/2} + \hat q]\).
  Marginal finite-sample validity, locally adaptive widths driven by the
  base quantile regressor. Repo: `models/conformal.py::SplitCQR`
  (+ `SplitOneSided` for VaR-style upper bounds).
- **Mondrian / group-conditional CQR.** Vovk et al. (2005) Mondrian
  machines; per-stratum calibration gives *exact* group-conditional
  validity at the cost of per-stratum sample size. Repo: `MondrianCQR`
  with `min_count=12` fallback to the global \(\hat q\) (ADR-009 uses
  vol-normalized scores).
- **Weighted conformal under covariate shift.** Tibshirani, Barber, Candès
  & Ramdas (2019, NeurIPS 32; arXiv:1904.06019). Weighted conformal
  quantiles with \(w_i = dP_{test}(x_i)/dP_{cal}(x_i)\); coverage holds
  under *weighted exchangeability* when the density ratio is right. Repo:
  `models/weighted_conformal.py::WeightedSplitCQR` (histogram density
  ratio on 1-d vol, Laplace-smoothed, clipped to \([10^{-3},10^{3}]\);
  ADR-013).
- **Localized conformal prediction (LCP).** Lei & Wasserman (2014, *JRSS-B*
  76:59–81); Guan (2023, *Biometrika* 110(3):703–717) — kernel-weighted
  scores \(w_i(x) \propto K((x - x_i)/h)\) give approximately
  coverage-conditional-on-x bands. Repo: `models/localized_conformal.py::
  LocalizedCQR` (RBF on vol, median-distance bandwidth, Kish-ESS gate at
  `min_ess=12`, ADR-017).
- **Jackknife+ / CV+.** Barber, Candès, Ramdas & Tibshirani (2021, *Annals
  of Statistics* 49(1); arXiv:1905.02928) — coverage \(\ge 1-2\alpha\)
  marginal under exchangeability for any symmetric algorithm; the K-fold
  **CV+** variant (paper §4, coverage \(\ge 1-2\alpha\) classic /
  \(\ge 1-\alpha\) for the minmax aggregation) is the same reference. Repo:
  `models/jackknife_plus.py`, `models/cv_plus.py` (ADR-011/016; Bian &
  Barber 2023 training-conditional caveat documented in-module;
  `marginal_exchangeable` scope key enforced by
  `research/catalog/hypotheses.py`).
- **Beyond exchangeability (nexCP).** Barber, Candès, Ramdas & Tibshirani
  (2023, *Annals of Statistics* 51(2); arXiv:2202.13415). Weighted
  quantiles robust to drift + randomization for asymmetric algorithms;
  coverage loss bounded by a total-variation term. Repo: **PARTIAL** —
  weighted CQR exists, but the drift weights (exponential recency) and the
  randomization trick are not implemented; ACI's sliding `score_window` is
  the closest analogue.
- **Classification with valid & adaptive coverage (RAVEN).** Romano, Sesia
  & Candès (2020, NeurIPS 33; arXiv:2006.02544) — adaptive scores for
  approximate conditional coverage in classification; relevant if fx-1
  directional calls are ever wrapped as conformal sets (repo:
  `conformal_rank.py` covers the ranking side instead).

### 2.3 Sequential / online conformal for time series (the core lane)

- **EnbPI.** Xu & Xie (2021, ICML PMLR 139:11559–11569; 2023, *IEEE TPAMI*
  45(10); arXiv:2010.09107). Bootstrap-aggregate ensemble + leave-one-out
  residual tracking + sliding residual window; *approximate* marginal
  coverage with error \(O(n^{-1/3}\sqrt{\log n})\) under strong mixing and
  slowly-varying conditional distributions; no retraining per step. Repo:
  `models/enbpi.py` (circular block bootstrap, `enbpi_residual_bounds`
  signed-residual band, streaming `predict_online(X, y, batch_size)`;
  Wave-9 INFLIGHT notes the documented φ_t deviation and that the interval
  is the paper's width-minimising signed band, not split-absolute).
- **ACI — Adaptive Conformal Inference.** Gibbs & Candès (2021, NeurIPS
  34:1660–1672; arXiv:2106.00170). Online update
  \(\alpha_{t+1} = \mathrm{clip}(\alpha_t + \gamma(\alpha - \mathrm{err}_t))\)
  on the miscoverage indicator; guarantees long-run coverage frequency
  \(\to \alpha\) for *arbitrary* data sequences, with efficiency analysis
  for exchangeable and Markov cases. Repo:
  `models/conformal.py::AdaptiveConformal` (+ `MondrianACI` per-stratum
  variant; H7 gate on its Kupiec p).
- **AgACI / FACI.** Zaffran, Dieuleveut, Féron, Goude & Josse (2022, ICML
  PMLR 162; arXiv:2202.07282). Exponentiated-gradient aggregation over an
  ACI expert grid in γ (and over τ-levels in FACI), scored by pinball —
  parameter-free in γ with worst-case regret vs the best expert. Repo:
  `models/agaci.py::AggregatedACI` (pinball-loss EG + quantile
  rearrangement).
- **Conformal PID control.** Angelopoulos, Candès & Tibshirani (2023,
  NeurIPS 36; arXiv:2307.16895). Treats \(\alpha_t\) as the control output
  of a PID controller on the miscoverage error stream:
  \(\alpha_t = \alpha + K_p(\alpha - \mathrm{err}_t) + K_i \sum_{s\le t}(\alpha - \mathrm{err}_s) + K_d(\mathrm{err}_{t-1} - \mathrm{err}_t)\),
  clipped and monotone-quantile-inverted. Prospective (uses the current
  error immediately, unlike ACI's lagged update) — demonstrably better
  width under trends/seasonality; simplifies and strengthens the ACI
  asymptotic analysis. Repo: **ABSENT** — natural fit: pure-Python stateful
  controller reusing `conformal_quantile` + `expand_interval`.
- **SPCI — Sequential Predictive Conformal Inference.** Xu & Xie (2023,
  ICML PMLR 202; arXiv:2212.03463). Instead of adapting α, adapt the
  *score quantile itself*: regress the next nonconformity score on lagged
  scores (and covariates) with any conditional-quantile learner (e.g. QRF);
  \(\hat C_t = \hat m_t \pm \hat q_{1-\alpha}(\text{lag features})\).
  Asymptotic *conditional* coverage under consistency of the quantile
  regressor; large width gains when residuals are serially dependent (the
  common volatility-clustering regime). Repo: **ABSENT** — but the
  ingredients exist in-tree: `models/qrf.py`, `models/quantile_forest.py`,
  `metrics/conformal.py` scores, and `AdaptiveConformal`'s score history.
- **KOWCPI.** Lee, Xu & Xie (2025, ICLR; arXiv:2405.16828).
  Kernel-based optimally-weighted CP intervals for dependent data;
  reweighted-Nadaraya–Watson quantile regression on lagged scores with
  learned weights; conditional coverage under strong mixing. The
  SOTA successor to SPCI; heavier (kernel ridge inner loop) — Tier-3 here.
- **MSCP / AcMCP — multi-step online conformal.** Wang & Hyndman (2024/2026),
  *Online conformal inference for multi-step time series forecasting*
  (arXiv:2410.13115) introduce **MSCP** (online multi-step split conformal:
  per-horizon score pools updated online, which removes the near-horizon
  undercover / far-horizon overcover bias of a single shared quantile) and
  **AcMCP** (h-step optimal forecast errors carry serial correlation up to
  lag h−1 under non-stationary AR DGPs; ACI-style adaptation with
  horizon-aware score updates and a finite-sample coverage-error bound
  growing in h). Repo: **ABSENT** for calibration (see §3.2). Related:
  Xu, Jiang & Xie (2024), *Conformal prediction for multi-dimensional
  time-series* (ICML 2024 spotlight) — the multivariate-set sibling.
- **CopulaCPTS.** Sun & Yu (2024, ICLR; arXiv:2212.03281). Joint
  multi-target/multi-step prediction regions: calibrate marginals
  conformally, then estimate the dependence of the conformal score vector
  with a Gaussian copula; pick a level set achieving finite-sample joint
  coverage. Repo: **ABSENT**; `models/copula.py` + `empirical_copula.py`
  give the dependence primitives. Also see Diquigiovanni, Fontana &
  Vantini (2022, *Statistical Papers*) for functional-data conformal bands
  — the continuous-horizon analogue.
- **Trajectory-wise CP.** Stankeviciute, Alaa & van der Schaar (2021,
  NeurIPS 34): conformal bounds for whole predicted trajectories using
  per-horizon scores under a dependency assumption; the medical-ICU
  setting. Same structural gap as MSCP/CopulaCPTS for this repo.
- **Bias-corrected ACI (BC-ACI).** Lade, Krishna & Kumar (2026,
  arXiv:2604.13253). ACI only moves the threshold; a post-shift persistent
  *bias* forces symmetric widening. BC-ACI adds an online EWMA bias
  estimate that re-centers scores before quantile inversion, with a
  dead-zone to suppress noise corrections; reported 13–17% Winkler
  reductions under mean-shift regimes with graceful finite-sample coverage
  degradation. Cheap and directly composable with the repo's ACI class —
  Tier-2.
- **Conformal floors / baselines discipline.** Manokhin (2026):
  *Conformal Seasonal Pools* (arXiv:2605.03789) and *Report the floor*
  (arXiv:2606.09473). A training-free split-conformal band around the
  random-walk / seasonal-naive forecast beats most learned probabilistic
  baselines on coverage and is within 2% of learned conformal predictors on
  relative Winkler; SPCI/ACI/AgACI remain the leaders (9–33% narrower).
  Adoption rule for this repo: **every interval bench must carry the
  conformal-naive floor** so a fancy calibrator is only credited above it
  (mirrors the existing GARCH-benchmark floor discipline).
- **Online risk control (beyond coverage).** Feldman, Ringel, Bates &
  Romano (2023, *TMLR*; arXiv:2205.09095) — rolling/online control of
  *any* monotone bounded risk with ACI-style integration; Angelopoulos,
  Bates, Fisch, Lei & Schuster (2022/ICLR 2024; arXiv:2208.02814) batch
  CRC with \((n\hat L_n(\lambda) + B)/(n+1) \le \alpha\). Repo: EXISTS
  (`models/crc.py`, `models/online_crc.py`; ADR-012/019; H11 gate).

### 2.4 Monitoring: exchangeability tests & conformal martingales

- **Conformal martingales / online p-values.** Vovk, Gammerman & Shafer
  (2005), *ALitRW* ch. 7–8 (sequential conformal p-values are i.i.d.
  Uniform(0,1) under exchangeability, Thm 8.1); Shafer, Shen, Vereshchagin
  & Vovk (2011), *Test martingales, Bayes factors and p-values*,
  *Statistical Science* 26(1):84–101 (arXiv:0912.4269 — the Ville
  exaggeration bound); Vovk (2021), *Testing randomness online*,
  *Statistical Science* 36(4); Vovk, Petej, Nouretdinov, Ahlberg, Carlsson
  & Gammerman (2021), *Retrain or not retrain: conformal test martingales
  for change-point detection*, COPA, PMLR 152 (the **Simple Jumper**);
  Volkhonskiy, Nouretdinov, Gammerman, Vovk & Burnaev (2017), *Inductive
  conformal martingales for change-point detection* (arXiv:1706.03415).
  Repo: EXISTS (`metrics/conformal_martingale.py` implements
  power/mixture/simple-jumper betting with the same citation stack).
- **WATCH — weighted conformal test martingales.** Prinster, Han, Liu &
  Saria (2025, ICML; arXiv:2505.04608). Generalizes CTMs with weights to
  adapt to mild covariate shift, detect harmful shifts, and *diagnose*
  concept vs out-of-support covariate shift; time-uniform FPR ≤ α by Ville.
  Repo: EXISTS as `metrics/watch.py` / `models/watch.py` with the honest
  caveat (INFLIGHT wave 9-WATCH): the practical frozen-bag + estimated
  density ratios make post-adaptation alarms *diagnostic*, not guaranteed.

### 2.5 Conditional-coverage diagnostics (the evaluation SOTA)

- **Binned coverage & worst-slice.** The classical toolkit; repo:
  `metrics/conformal.py::conditional_coverage` + `worst_slice_coverage`
  on PIT-safe terciles. Known weaknesses: binning bias, no multiplicity
  control, low power at lab n.
- **Kupiec POF / Christoffersen CC on miss indicators.** Repo EXISTS
  (`metrics/probability.py::kupiec_pof`, `christoffersen_cc`; wired into
  H4/H7/H8/H11/H12 with skip-on-nonfinite honesty, Day Waves 11–17).
- **Winkler interval score.** Winkler (1972); Bracher et al. (2021),
  *Evaluating epidemic forecasts in an interval format* (PLoS Comp Bio) —
  the proper-score way to compare interval predictors (coverage and width
  jointly). Repo: `metrics/calibration2.py::winkler_interval_score`
  EXISTS but is *not* wired into the conformal bench families, which
  report coverage + mean/median width separately.
- **ERT — excess risk of the target coverage.** Braun, Holzmüller, Jordan
  & Bach (2026; arXiv:2512.11779). Cast conditional coverage estimation as
  a classification problem: conditional coverage is violated iff some
  classifier beats the trivial miscoverage predictor under a proper loss;
  the risk gap conservatively bounds L1/L2 miscoverage distance, separates
  over/under-coverage, and has far higher power than CovGap-style binned
  metrics. Ships as an open package. Repo: **ABSENT** — the strongest
  evaluation-side gap; a light in-tree version (logistic/gradient-boosted
  miscoverage classifier on PIT-safe features + cross-fitted risk gap)
  would sit naturally beside `worst_slice_coverage`.
- **Conformal e-values & FDR.** Vovk & Wang (2021), *E-values for
  calibration*, AISTATS 2021, PMLR 130; Bates et al. (2021) conformal
  p-values for FDR; Jin & Candès (2023) cfBH (conformalized Benjamini–
  Hochberg for outlier detection). Repo: EXISTS on the selection side
  (`models/conformal_rank.py`; H19), and the e-value stack
  (`metrics/evalues.py`, `anytime_fdr.py`) composes with conformal p-values
  per ADR-010.

### 2.6 What the 2025–2026 frontier added (beyond §2.3–2.5)

- **Ellipsoidal/joint regions for multivariate streams**: SPACE
  (arXiv:2608.17333), filtered conformal ellipsoids
  (arXiv:2606.17014) — Mahalanobis scores + split conformal, mixing-aware
  coverage bounds. Relevant only if the panel gets cross-asset joint sets
  (`portfolio_conformal` is the in-tree cousin).
- **Localized online CP**: OLCP / OLCP-Hedge (arXiv:2605.05497) — ACI +
  kernel localization with bandwidth experts; would unify
  `AdaptiveConformal` and `LocalizedCQR`.
- **Gate-localized CP on nonstationary multivariate streams**: ABF-T-GLCP
  (arXiv:2607.23165). Regime-gated residual selection — the repo's HMM
  regime labels (`metrics/watch.py` covariate adaptation; `regime_eval.py`)
  are the natural in-tree conditioning variable.
- **Foundation-model conformal calibration**: STOIC (arXiv:2606.31804) —
  in-context calibration of residuals; watch-list only (fx-1 lane).

---

## 3. Gap analysis: SOTA × in-tree

### 3.1 Inventory (what exists, where)

| Capability | Module | Notes |
|---|---|---|
| Split CQR / one-sided | `models/conformal.py::SplitCQR`, `SplitOneSided` | exact \((n{+}1)\) quantile; ADR-008 |
| Mondrian CQR + Mondrian ACI | `models/conformal.py` | vol terciles, `min_count=12`; H8 gate |
| ACI (Gibbs–Candès) | `models/conformal.py::AdaptiveConformal` | date-grouped `run`, sliding score window; H7 gate |
| AgACI / FACI-EG | `models/agaci.py::AggregatedACI` | pinball-scored EG over γ experts |
| EnbPI | `models/enbpi.py` | block-bootstrap LOO residuals, streaming `predict_online` |
| Weighted CQR (covariate shift) | `models/weighted_conformal.py` | histogram density ratio on vol; H12 gate; ADR-013 |
| Localized CQR | `models/localized_conformal.py` | RBF + ESS gate; ADR-017 |
| Jackknife+ / CV+ | `models/jackknife_plus.py`, `models/cv_plus.py` | `coverage_guarantee_scope` honesty key; H10/H15 |
| CRC / online CRC | `models/crc.py`, `models/online_crc.py` | monotone tail losses; H11; ADR-012/019 |
| Conformalized Student-t head | `models/conformal_dist.py` | causal 2/3-fit, 1/3-shift calibration |
| Conformal top-k / ranking | `models/conformal_rank.py` | cfBH-flavored; H19; ADR-018 |
| WATCH / conformal martingales | `models/watch.py`, `metrics/watch.py`, `metrics/conformal_martingale.py` | diagnostic post-adaptation caveat documented |
| Conformal primitives | `metrics/conformal.py` | `conformal_quantile`, `cqr_scores`, `expand_interval`, `set_metrics`, `conditional_coverage`, `worst_slice_coverage`, `assign_terciles` |
| Operational path | `pipeline/forecast/conformal.py::conformal_sets_asof` | scaled-(Student-)t base bands → Mondrian/split CQR on cal → asof-day sets → `ForecastIntervals` → `AssetForecast.interval_*` → `portfolio/interval_risk.py` caps (ADR-014) |
| Benches | `research/benches/intervals.py` (12 bench fns), `research/benches_w810.py::bench_ts_conformal` | families: conformal, evalues, jackknife_plus, crc, weighted_conformal, interval_risk, quantile_bandit, cv_plus, cpcv, localized_conformal, conformal_rank, online_crc, portfolio_conformal + ts_conformal (OPTIONAL) |

Proper-score plumbing the conformal lane already leans on:
`metrics/scoring.py` (pinball, CRPS quantile/Gaussian/Student-t/empirical,
`rearrange_quantiles`), `metrics/calibration2.py` (Winkler, pinball score
dicts), `metrics/probability.py` (Kupiec POF, Christoffersen ind/CC),
`metrics/var_backtest.py` (kupiec/christoffersen at VaR levels), PIT/KS via
`metrics/calibration_tests.py`.

### 3.2 Gap matrix (method × status)

| Method | Citation | Status | Gap |
|---|---|---|---|
| Split CQR | Romano et al. 2019 | EXISTS | — |
| Mondrian CQR/ACI | Vovk et al. 2005 | EXISTS | — |
| Weighted CP (covariate shift) | Tibshirani et al. 2019 | EXISTS | weights = histogram on 1-d vol only |
| Localized CP | Lei & Wasserman 2014; Guan 2023 | EXISTS | 1-d covariate only |
| Jackknife+ / CV+ | Barber et al. 2021 | EXISTS + scope honesty | — |
| ACI | Gibbs & Candès 2021 | EXISTS (model+bench) | **not in operational path** |
| AgACI/FACI | Zaffran et al. 2022 | EXISTS (model) | **not in bench families nor operational path** |
| EnbPI | Xu & Xie 2021/2023 | EXISTS (model + `bench_ts_conformal`) | streaming API unused by pipeline |
| Conformal PID | Angelopoulos et al. 2023 | **ABSENT** | new `models/conformal_pid.py` |
| SPCI | Xu & Xie 2023 | **ABSENT** | new `models/spci.py` on `models/qrf.py` |
| KOWCPI | Lee et al. 2025 | ABSENT | Tier-3 (heavy kernels) |
| BC-ACI (bias re-centering) | Lade et al. 2026 | **ABSENT** | small extension of `AdaptiveConformal` |
| MSCP (online multi-step split) | Wang & Hyndman 2024 (arXiv:2410.13115) | **ABSENT** | horizon loop in bench + wrapper |
| AcMCP (horizon-aware online) | Wang & Hyndman 2024 | **ABSENT** | extends ACI score bookkeeping |
| CopulaCPTS (joint trajectory) | Sun & Yu 2024 | **ABSENT** | needs copula over score vectors |
| Trajectory CP | Stankeviciute et al. 2021 | ABSENT | subsumed by MSCP/CopulaCPTS plan |
| Batch CRC / online RC | Angelopoulos et al. 2022; Feldman et al. 2023 | EXISTS | — |
| Conformal martingales / WATCH | Vovk et al. 2005; Prinster et al. 2025 | EXISTS | diagnostic caveat documented |
| cfBH / conformal FDR | Bates et al. 2021; Jin & Candès 2023 | EXISTS (`conformal_rank`) | — |
| nexCP (drift weights + randomization) | Barber et al. 2023 | PARTIAL | weighted CQR lacks drift/randomization modes |
| ERT conditional diagnostics | Braun et al. 2026 | **ABSENT** | new `metrics/` functions |
| Winkler in interval benches | Winkler 1972; Bracher et al. 2021 | PARTIAL (metric exists; not wired to conformal benches) | wire it |
| Conformal-naive floor baseline | Manokhin 2026 | **ABSENT** | bench hygiene rule |

### 3.3 Structural gaps (why the inventory understates the distance)

1. **The operational wrapper is static.** `conformal_sets_asof` refits
   scaled-(Student-)t base bands per asof and calibrates split/Mondrian CQR
   on the trailing cal window — a *batch* construction repeated per decision
   date. Under drift this is precisely the setting where ACI/PID/SPCI show
   width and coverage-frequency gains. Because `IntervalMethod` is a closed
   `Literal["mondrian_cqr", "split_cqr"]`, downstream consumers
   (`AssetForecast.interval_method`, `interval_risk` caps, receipts) cannot
   yet *name* a sequential method even though four exist in `models/`.
2. **One horizon at a time.** `_horizon_bars(label, config)` picks a single
   label/horizon; multi-step claims are made by `models/hstep.py`
   (√h-scaling / overlapping empirical sums) with **no conformal
   calibration and no joint coverage statement** over the horizon vector.
   SOTA (MSCP/AcMCP/CopulaCPTS) treats the h-vector as the prediction
   object.
3. **Sequential models are never scored end-to-end on the panel.**
   `bench_conformal` runs ACI on the panel split, but AgACI has no bench
   family (only `AggregatedACI` unit tests), EnbPI is benched only on
   SYNTHETIC AR(1) in `bench_ts_conformal`, and no bench reports Winkler or
   relative-width-vs-floor for any of them. The receipt record therefore
   cannot answer "which sequential calibrator, at what width cost, on the
   lab panel".
4. **Conditional coverage power.** Tercile coverage + worst-slice at
   lab n (~hundreds of cal rows per stratum) has weak power; ERT-style
   cross-fitted miscoverage classifiers are the 2026 standard and compose
   with existing PIT-safe features (vol, breadth, cs_dispersion, regime
   labels).

---

## 4. Adoption plan

Sequenced by leverage; every item is research/infrastructure only — no
live broker, no `live_pnl_claim`, all SYNTHETIC fixtures labeled.

### Tier 1 — wire what already exists (days, no new math)

**1a. Extend the operational wrapper with a sequential mode.**
Location: `src/quant_fund/pipeline/forecast/conformal.py` (the natural home
of "the conformal calibration wrapper"; ADR-008 keeps it a wrapper over
the existing scaled-t base bands).

- Extend the schema literal (one-line change, versioned):

```python
# schemas/forecast.py
IntervalMethod = Literal[
    "mondrian_cqr", "split_cqr",          # existing (batch)
    "aci", "agaci", "enbpi", "online_crc", # sequential (Tier 1a)
    "spci", "conformal_pid",               # Tier 2
    "mscp_hstep",                          # Tier 3
]
```

- API sketch (the wrapper keeps its current signature; the method is
  config-selected, and sequential state is carried in the existing
  `_CONFORMAL_CACHE`-style process-local store keyed by
  `(label, horizon, security-scope, method)`):

```python
def conformal_sets_asof(
    frame, asof, config, *,
    alpha: float = INTERVAL_ALPHA,
    method: str | None = None,   # None -> config-driven default (split/mondrian, unchanged)
    ...
) -> ForecastIntervals | None:
    """Batch path unchanged. When method in SEQUENTIAL_METHODS:
    1. build base quantile bands exactly as today (scaled-t wrappee),
    2. replay the cal+recent window through the sequential calibrator
       (AdaptiveConformal / AggregatedACI / EnbPI.predict_online /
       OnlineCRC) in date order,
    3. emit the asof-day sets from the calibrator's current alpha_t/qhat_t,
    4. stamp ForecastIntervals(method=..., cal_event_times=...) and persist
       the calibrator state so the next asof continues the path (no leak:
       state only ever consumes y strictly before the asof date)."""
```

- Honesty scope: sequential methods get a scope string in `ModelMeta.extra`
  — `"coverage_scope": "long_run_frequency"` for ACI/AgACI/PID (asymptotic
  coverage frequency, Gibbs–Candès), `"approximate_marginal_mixing"` for
  EnbPI — mirroring the `marginal_exchangeable` key discipline. Receipts
  must carry it; `research/verify.py` already fail-closes on missing scope
  keys for JP/CV+ and the same helper generalizes.

**1b. Bench-family completion.**
Location: `src/quant_fund/research/benches/intervals.py` (+ registration in
`research/agent.py` families dict and `catalog/constants.py`
`OPTIONAL_BENCHMARK_FAMILIES` — **do not touch** `REQUIRED_BENCHMARK_FAMILIES`
or `BENCHMARK_FAMILY_ORDER` per the docs-consistency contract).

- New optional family `agaci`: panel replay of `AggregatedACI` against
  `bench_conformal`'s ACI path; keys: `coverage`, `mean_width`,
  `median_width`, `kupiec_p`, `christoffersen_cc_p`, `winkler_mean`,
  `relative_winkler_vs_floor`, `alpha_t_final`, `dgp`, `claim="research_metric_only"`.
- Extend `bench_ts_conformal` (waves 8–10 battery): add the **conformal
  naive floor** (split conformal around random-walk residuals on the same
  SYNTHETIC stream) and report `enbpi_relative_width_vs_floor`; add an
  AgACI reading beside EnbPI on the vol-break stream (the shift regime is
  where AgACI's γ-adaptation should visibly beat fixed-γ ACI — asserted
  directionally on SYNTHETIC only).
- Wire `metrics/calibration2.py::winkler_interval_score` into every
  interval family blob (proper score; currently coverage+width only).
- All new keys pass `family_blob_forbidden_metrics_absent` (no forbidden
  tokens: `winkler`, `coverage`, `kupiec_p`, `christoffersen_cc_p`,
  `relative_winkler_vs_floor` are clean).

**1c. H-table consistency for new families.** Follow the established
soft-gate pattern in `research/catalog/hypotheses.py`: finite `kupiec_p`
in `families["agaci"]` ⇒ mint `H20_agaci_coverage` (calibration row,
skip-on-nonfinite like H7/H8/H11/H12). No invented p-values.

### Tier 2 — new models with strong published evidence (1–2 weeks each)

**2a. `models/conformal_pid.py` — Conformal PID controller**
(Angelopoulos, Candès & Tibshirani 2023, arXiv:2307.16895).

```python
class ConformalPID:
    """PID control on the miscoverage error stream.

    alpha_t = clip(alpha + Kp*(alpha - err_t) + Ki*sum_s(alpha - err_s)
                   + Kd*(err_{t-1} - err_t), eps, 1-eps)
    Sets via conformal_quantile(scores_window, alpha_t) + expand_interval.
    Prospective: err_t includes the current step (unlike ACI's lag).
    """
    def __init__(self, alpha=0.10, kp=0.05, ki=0.0, kd=0.0,
                 score_window: int | None = None): ...
    def initialize(self, y, lower, upper, scale=None) -> None: ...
    def run(self, y, lower, upper, dates, scale=None) -> ACIPath: ...
```

Reuse `ACIPath` as the return type (fields already carry alpha_t/qhat_t);
`ki=kd=0, kp=γ` must reproduce `AdaptiveConformal` up to the
prospective/lagged error timing — that identity is the property test.

**2b. `models/spci.py` — Sequential Predictive Conformal Inference**
(Xu & Xie 2023, arXiv:2212.03463), built on `models/qrf.py`
(Meinshausen QRF already in-tree, ADR-007 keeps the no-deep-NN rule):

```python
class SPCI:
    """Conformal intervals whose score quantile is itself forecast.

    Features: lagged absolute/signed residuals (p lags), optional
    PIT-safe covariates (vol_20). Learner: QRF conditional quantile at
    1-alpha (paper default; consistent => asymptotic conditional coverage).
    predict_interval(x_t) = m_t +/- qhat_t; update(r_t) appends the
    observed residual. Warmup falls back to EnbPI-style empirical
    residual quantile until >= min_train scores exist.
    """
    def __init__(self, alpha=0.10, n_lags=4, qrf_kwargs=None,
                 min_train=64): ...
    def fit_warmup(self, X, y) -> SPCI: ...
    def predict_online(self, X, y, batch_size=1) -> SPCIResult: ...
```

`SPCIResult` mirrors `EnbPIResult` (lower/upper/covered/qhat_t/width) so
`bench_ts_conformal` can consume both identically. SYNTHETIC assertion:
on an AR(1)+GARCH(1,1) error stream, SPCI mean width ≤ EnbPI mean width at
equal empirical coverage (the paper's headline regime; direction asserted,
magnitude never claimed as market evidence).

**2c. BC-ACI dead-zone bias re-centering** (arXiv:2604.13253) as an opt-in
flag on `AdaptiveConformal` (`bias_correction="ewm_deadzone"`): score
re-centering before quantile inversion + documented finite-sample caveat
(coverage guarantee degrades with bias-estimation error — stamp
`"coverage_scope": "long_run_frequency_with_bias_caveat"`).

**2d. ERT-lite conditional diagnostics** — `metrics/conformal.py`:

```python
def excess_risk_target_coverage(
    y, lower, upper, features, *, learner="logistic", folds=5, seed=0
) -> dict[str, float]:
    """Cross-fitted miscoverage-classifier risk gap (Braun et al. 2026).

    Returns: ert_l1 (conservative bound on L1 miscoverage distance),
    over_coverage_share / under_coverage_share decomposition,
    n, learner, folds. Pure diagnostic: a *proper-score* companion to
    worst_slice_coverage; never a promotion gate by itself.
    """
```

Cross-fitted sklearn logistic (in-tree dependency; gradient-boosted
variant optional). Wire into `bench_conformal` / `bench_weighted_conformal`
blobs as `ert_l1`, `ert_over_share`, `ert_under_share`; keys are
forbidden-token-clean. Add a soft H-row only if a finite `ert_l1` appears
in a family blob (same pattern as H7).

### Tier 3 — multi-horizon & trajectory calibration (gated on Tier 1–2)

**3a. `mscp_hstep` — horizon-split calibration.** Location: extend
`pipeline/forecast/conformal.py` + `models/hstep.py`. For each horizon
h ∈ {1, …, H} calibrate a split-CQR score on the h-step label (the
overlapping-sum machinery already exists in `hstep.py::_overlap_sums`);
allocate α across horizons (uniform split α/H — MSCP's conservative
default — or Bonferroni) so the *joint* trajectory coverage is
\(\ge 1-\alpha\) up to the union bound. Stamp
`interval_method="mscp_hstep"` and record per-h `qhat` in
`ForecastIntervals` (needs a `dict[str, float]` → per-horizon extension of
the frozen dataclass; keep `lower/upper` per-horizon-keyed as today since
`AssetForecast.interval_lo/hi` are already horizon-keyed dicts).
Honesty: the union-bound guarantee is *conservative marginal*, not joint
exact — scope key `"coverage_scope": "joint_union_bound_marginal"`.

**3b. AcMCP horizon-aware online update** (arXiv:2410.13115) — once 3a
lands: ACI whose error signal is the *per-horizon* miscoverage indicator
with horizon-lagged score updates; finite-sample coverage-error bound grows
with h — stamp it.

**3c. CopulaCPTS joint regions** (arXiv:2212.03281) — calibrate marginals
(3a), estimate the score-vector copula from calibration (`models/copula.py`
Gaussian copula primitives exist), choose the level set for finite-sample
joint coverage. Larger surface (needs a `TrajectoryRegion` schema);
gate on 3a/3b receipts.

**3d. nexCP completion** (arXiv:2202.13415) — add drift weights
(exponential recency w_i ∝ ω^{n−i}) and the randomization device to
`WeightedSplitCQR` as opt-in modes; TV-bound diagnostic in metadata.

### 4. Benchmark protocol (frozen before scoring)

1. **Streams.** (a) Lab panel via the existing `_gaussian_interval_split`
   path (real-data slice, walk-forward dates, no y-leakage into cuts);
   (b) SYNTHETIC AR(1) stationary, AR(1)+vol-break, AR(1)+GARCH(1,1),
   mean-shift (BC-ACI regime) — all seeded, labeled `dgp` per fixture.
2. **Baselines, always run:** conformal-naive floor (random-walk +
   split conformal), split CQR, Mondrian CQR, ACI (γ grid {0.01, 0.05,
   0.2}), AgACI, EnbPI, online CRC; challengers (PID, SPCI, BC-ACI,
   mscp_hstep) are credited only on top of the floor.
3. **Metrics (proper only):** empirical coverage, coverage-frequency in
   rolling 60-day windows (ACI's actual guarantee), mean/median width,
   Winkler mean, relative Winkler vs floor, Kupiec POF p, Christoffersen
   CC p, ERT-L1 + over/under decomposition, worst-slice tercile coverage,
   PIT of the base bands (unchanged by conformal — ADR-008).
4. **Receipts.** Every bench run goes through the existing
   `research/verify.py` path; family blobs carry `claim=
   "research_metric_only"`, `dgp`, seed, and the `coverage_scope` key;
   `verify-research` must stay green (`errors=[]`).
5. **No promotion semantics.** Interval metrics never gate capital
   decisions beyond the existing ADR-014 caps; `INSTITUTIONAL_READINESS.md`
   minimum-evidence conditions unchanged.

### 5. Test strategy

Layered exactly like waves 8–10 (all gates: `make lint`, `make typecheck`,
`make test` — network/slow excluded; new files under `tests/unit/models/`
and `tests/unit/core/`):

- **Unit correctness (closed-form).** `conformal_quantile` exactness on
  hand-computed score sets (already covered); PID: `kp=γ, ki=kd=0` ⇒
  alpha_t path matches ACI modulo prospective-error timing (assert both
  paths explicitly on a fixed miss sequence); SPCI: with a degenerate
  1-lag score model and constant residuals, qhat_t equals the empirical
  quantile; ERT-lite: on a stream with *perfectly* exchangeable misses,
  `ert_l1 ≈ 0` within tolerance; on a planted vol-conditional miscoverage
  shift, `ert_l1 > 0` and `worst_slice_coverage` also degrades (direction
  agreement, not magnitude).
- **Property tests (Hypothesis, `ci` profile derandomized).** For any
  finite inputs: intervals never inverted (`expand_interval` contract);
  coverage ∈ [0,1]; widths finite when scores finite; NaN/empty ⇒ honest
  NaN (never silent 0.0 — the Wave 6–17 house rule); `alpha ∉ (0,1)` ⇒
  ValueError at every new constructor.
- **Extremes fixtures** (repo convention `*_extremes.py`): single score,
  all-equal scores, α below 1/(n+1) (unattainable-level clip path),
  weights all-zero, ESS collapse in localized mode, PID integral windup
  (clip bounds respected), SPCI warmup below `min_train`.
- **SYNTHETIC science assertions** (seeded, labeled): stationary AR(1) ⇒
  EnbPI/SPCI/PID coverage within MC tolerance of nominal; vol-break ⇒
  fixed split CQR coverage error *exceeds* ACI/PID coverage error
  (asserted directionally, as `bench_ts_conformal` already does); BC-ACI
  Winkler ≤ ACI Winkler under mean shift, equal within tolerance under no
  shift.
- **Pipeline tests** (`tests/unit/pipeline/`): `conformal_sets_asof(method=
  "aci")` cache-key includes method (no cross-method cache poisoning —
  the existing row-hash fingerprint discipline); sequential replay is
  causal (shuffling future dates cannot change asof-day sets);
  `ForecastIntervals.method` ∈ extended literal; `AssetForecast`
  validators unchanged (interval_alpha finite in (0,1), lo ≤ hi).
- **Receipt/honesty tests** (`tests/unit/research/`): new family blobs are
  `family_blob_forbidden_metrics_absent`; `coverage_scope` presence
  enforced for sequential families (extend the existing
  `jp_cv_blob_requires_marginal_coverage_scope` pattern); H20 mint
  consistency mirrors `h7_hypothesis_consistency_errors`; determinism
  (same seed ⇒ identical blob, per `test_benches_w810.py`).
- **Gate wiring.** Nothing new in CI beyond the default lanes; the fx1
  lane is untouched (no `tests/fx1` changes).

### 6. Sequencing & effort

| Step | Content | Est. | Gate |
|---|---|---|---|
| 1a | `IntervalMethod` extension + sequential mode in `conformal_sets_asof` | 2–3 d | pipeline tests + honesty keys |
| 1b | AgACI family, floor baselines, Winkler wiring in benches | 2 d | bench determinism + forbidden-key hygiene |
| 1c | H20 soft gate | 0.5 d | `test_hypothesis_semantics.py` pattern |
| 2a | ConformalPID | 2–3 d | ACI-identity property test |
| 2b | SPCI on QRF | 3–5 d | SYNTHETIC width-vs-EnbPI direction |
| 2c | BC-ACI flag | 1–2 d | mean-shift fixture |
| 2d | ERT-lite | 2–3 d | exchangeable-zero + planted-shift tests |
| 3a | mscp_hstep | 1 wk | joint union-bound honesty scope |
| 3b/3c/3d | AcMCP / CopulaCPTS / nexCP | 1–2 wk each | gated on 3a receipts |

Each step lands with its ADR (039+ numbering: "sequential interval
methods in the operational wrapper", "SPCI", "PID", "multi-horizon
calibration") and an INFLIGHT entry per the wave convention.

---

## 7. References

Foundations & static CP
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*. Springer.
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). Distribution-free predictive inference for regression. *JASA* 113(523):1660–1672.
- Angelopoulos & Bates (2021). A gentle introduction to conformal prediction. arXiv:2107.07511.
- Romano, Patterson & Candès (2019). Conformalized quantile regression. NeurIPS 32. arXiv:1905.03222.
- Romano, Sesia & Candès (2020). Classification with valid and adaptive coverage. NeurIPS 33. arXiv:2006.02544.
- Foygel Barber, Candès, Ramdas & Tibshirani (2021). The limits of distribution-free conditional predictive inference. *Information and Inference* 10(4):921–940. arXiv:1903.04684.
- Barber, Candès, Ramdas & Tibshirani (2021). Predictive inference with the jackknife+. *Annals of Statistics* 49(1). arXiv:1905.02928.
- Barber, Candès, Ramdas & Tibshirani (2023). Conformal prediction beyond exchangeability. *Annals of Statistics* 51(2). arXiv:2202.13415.
- Tibshirani, Barber, Candès & Ramdas (2019). Conformal prediction under covariate shift. NeurIPS 32. arXiv:1904.06019.
- Lei & Wasserman (2014). Distribution-free conditional predictive inference using kernel methods. *JRSS-B* 76(1):59–81.
- Guan (2023). Localized conformal prediction: a generalized inference framework. *Biometrika* 110(3):703–717.
- Jin & Candès (2023). Conformalized multiple testing with FDR control. (cfBH line; see also Bates, Fisch, Lei, Schuster 2021, conformal p-values.)

Sequential / time-series CP
- Xu & Xie (2021). Conformal prediction interval for dynamic time-series. ICML, PMLR 139:11559–11569.
- Xu & Xie (2023). Conformal prediction for time series. *IEEE TPAMI* 45(10). arXiv:2010.09107. (EnbPI)
- Gibbs & Candès (2021). Adaptive conformal inference under distribution shift. NeurIPS 34:1660–1672. arXiv:2106.00170.
- Zaffran, Dieuleveut, Féron, Goude & Josse (2022). Adaptive conformal predictions for time series (AgACI/FACI). ICML, PMLR 162. arXiv:2202.07282.
- Angelopoulos, Candès & Tibshirani (2023). Conformal PID control for time series prediction. NeurIPS 36. arXiv:2307.16895.
- Xu & Xie (2023). Sequential predictive conformal inference for time series (SPCI). ICML, PMLR 202. arXiv:2212.03463.
- Lee, Xu & Xie (2025). Kernel-based optimally weighted conformal time-series prediction (KOWCPI). ICLR 2025. arXiv:2405.16828.
- Wang & Hyndman (2024). Online conformal inference for multi-step time series forecasting (AcMCP). arXiv:2410.13115.
- Xu & Xie (2024). Multi-step conformal prediction beyond online correction (MSCP). NeurIPS-line preprint.
- Sun & Yu (2024). Copula conformal prediction for multi-step time series forecasting (CopulaCPTS). ICLR 2024. arXiv:2212.03281.
- Stankeviciute, Alaa & van der Schaar (2021). Conformal time-series prediction. NeurIPS 34.
- Diquigiovanni, Fontana & Vantini (2022). Conformal prediction bands for multivariate functional data. *Statistical Papers* 63:2047–2077. arXiv:2106.01792.
- Lade, Krishna & Kumar (2026). Bias-corrected adaptive conformal inference for multi-horizon time series forecasting (BC-ACI). arXiv:2604.13253.
- Manokhin (2026). Training-free probabilistic time-series forecasting with conformal seasonal pools. arXiv:2605.03789; Report the floor: a training-free conformal interval is a mandatory baseline. arXiv:2606.09473.
- Angelopoulos, Bates, Fisch, Lei & Schuster (2022/ICLR 2024). Conformal risk control. arXiv:2208.02814.
- Feldman, Ringel, Bates & Romano (2023). Achieving risk control in online learning settings. *TMLR*. arXiv:2205.09095.

Monitoring & diagnostics
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*. Springer.
- Shafer, Gammerman & Vovk (2011). Test martingales, Bayes factors and p-values. *Statistical Science* 26(1):84–101. arXiv:0912.4269.
- Vovk, Wang & Ramdas (2021). E-values: calibration, combination, and applications. *Annals of Statistics* 49(3):1736–1754. arXiv:1912.06116.
- Vovk, Petej, Nouretdinov, Ahlberg, Carlsson & Gammerman (2021). Retrain or not retrain: conformal test martingales for change-point detection (the Simple Jumper line). COPA, PMLR 152. arXiv:2102.10439.
- Ho (2005). A martingale framework for concept change detection in time-varying data streams. ICML, 321–327; Ho & Wechsler (2010), *IEEE TPAMI* 32(12):2113–2127.
- Prinster, Han, Liu & Saria (2025). WATCH: adaptive monitoring for AI deployments via weighted-conformal martingales. ICML 2025. arXiv:2505.04608.
- Braun, Holzmüller, Jordan & Bach (2026). Conditional coverage diagnostics for conformal prediction (ERT). arXiv:2512.11779.
- Winkler (1972). A decision-theoretic approach to interval estimation. *JASA* 67:187–191.
- Bracher, Ray, Gneiting & Reich (2021). Evaluating epidemic forecasts in an interval format. *PLoS Computational Biology* 17(2):e1008618.
- Christoffersen (1998). Evaluating interval forecasts. *International Economic Review* 39(4):841–862; Kupiec (1995), *JFACS* 19.

Frontier watch-list (2025–2026, not adopted yet)
- SPACE (arXiv:2608.17333); filtered conformal ellipsoids (arXiv:2606.17014); OLCP/OLCP-Hedge (arXiv:2605.05497); ABF-T-GLCP (arXiv:2607.23165); STOIC (arXiv:2606.31804).
