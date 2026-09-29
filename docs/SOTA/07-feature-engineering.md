# 07 — Financial feature engineering & labeling (SOTA lane)

> Lane: **FINANCIAL FEATURE ENGINEERING & LABELING** (López de Prado *AFML*
> + 2020–2026 successors). Status doc — research summaries + in-tree
> inventory + gap list + adoption plan. **No source file is modified here;**
> every snippet below is a proposal to land in a separate PR.
>
> Honesty contract applies to this document: research results are **proper
> scores** (pinball, CRPS, QLIKE, Brier, log score, ECE, PIT, Kupiec, HMM
> likelihood). Nothing below is market evidence, a live-P&L claim, or a
> Sharpe/Sortino/Calmar headline (`research_only=true`,
> `live_pnl_claim=false`). Where a cited paper evaluates with return-based
> metrics, that is noted and **not** reproduced as a headline here.
>
> Related in-tree docs: `docs/SOTA_GAP_ANALYSIS.md` (gold-panel and ranker
> history), `docs/SOTA_CANON_ROADMAP_2026_09.md` (canon waves 1–9),
> `INFLIGHT` (canon-wave landing notes for `labels/barriers`,
> `features/bars`, `models/fracdiff`, `models/realized`),
> `docs/DETERMINISTIC_SIMULATION.md` (seed/replay discipline),
> `docs/EXPLAINABILITY.md`, `MATH_SPEC.md`, `RESEARCH_REFERENCES.md`.

---

## 1. Technique summaries (with citations)

### 1.1 Information-driven bars (AFML ch. 2)

**Why not time bars.** Calendar-time sampling makes returns heteroskedastic,
non-normal, and serially correlated in ways that violate the iid assumptions
of most ML estimators; time is a poor proxy for information arrival because
news intensity varies enormously within and across days (Clark 1973;
Ané & Geman 2000; Easley, López de Prado & O'Hara 2012, "The Volume Clock",
*J. Portfolio Management* 39(1):19–29 —
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2093089>). The fix is to
sample on a **clock driven by activity**: a bar closes when a cumulative
activity statistic crosses a threshold, so every bar carries approximately
the same information content.

**Bar families** (AFML §2.3–2.5):

| Bar | Trigger statistic | Notes |
|---|---|---|
| Tick | count of ticks ≥ `ticks_per_bar` | simplest activity clock |
| Volume | Σ size ≥ θ_V | volume clock; foundation of VPIN |
| Dollar | Σ (price × size) ≥ θ_$ | **recommended default** — invariant to price-level drift and splits (AFML 2.3.3) |
| Tick imbalance (TIB) | \|Σ b_t\| ≥ E[T]·E[\|θ\|], b_t = tick-rule sign | samples on *one-sided* flow (AFML 2.5.2) |
| Volume/dollar imbalance (VIB/DIB) | same with size- or dollar-weighted signs | |
| Tick run (TRB) | max same-sign run ≥ E[T]·E[max buy fraction] | captures sustained one-sided pressure (AFML 2.5.3) |
| Volume/dollar run (VRB/DRB) | run statistic on signed volume/dollar | |

- **Trade signs without trade direction:** the tick rule (Lee & Ready 1991,
  *J. Finance* 46(2):733–746) — sign of the price change, zeros carried
  forward; at aggregated frequencies, **Bulk Volume Classification**
  (Easley, López de Prado & O'Hara 2016, *J. Financial Markets* 30:19–37)
  splits a bar's volume into buy/sell parts using the bar return and a
  normal CDF — no per-trade data needed.
- **Threshold calibration:** E[T] (expected ticks/volume/dollars per bar) is
  an EWMA of realized bar lengths, updated at each bar close
  (AFML 2.3.2.2/2.4.1.2). Imbalance-bar thresholds use the EWMA of the
  absolute imbalance statistic itself, so the bar rate self-calibrates to
  the prevailing regime.
- **Evidence that it matters:** VPIN — volume-synchronized probability of
  informed trading, built on volume bars — rose persistently before the 2010
  flash crash (Easley, López de Prado & O'Hara 2011, "The Microstructure of
  the 'Flash Crash'", *J. Portfolio Management* 37(2):118–128,
  <https://ssrn.com/abstract=1695041>; and 2012, "Flow Toxicity and
  Liquidity in a High-frequency World", *RFS* 25(5):1457–1493,
  <https://ssrn.com/abstract=1695596>); bar returns are empirically closer
  to Gaussian and less autocorrelated than time-bar returns (AFML 2.4; the
  repo asserts exactly this in
  `tests/unit/data/test_bars.py::test_bar_returns_more_gaussian_ish`).
- **Precondition:** information-driven bars need a tick/quote/trade or at
  least a fine intraday corpus. On daily OHLCV there is nothing to resample;
  the lane where this bites is the lab's 4h/1m crypto corpora
  (`data/adapters/hf_ohlcv_1m.py` already scans 1-minute vendor parquet).

### 1.2 Event-based labeling (AFML ch. 3 & 5)

**Fixed-horizon labels** (the repo's current `build_labels` path) attach
\(y_t = r_{t\to t+h}\) at every bar. Faults (AFML 3.1): (i) labels overlap,
so samples are not independent — CV without purging is invalid; (ii) the
threshold is volatility-blind — the same ±x% means different things in calm
vs crisis regimes; (iii) one horizon for all events ignores that the
information half-life differs across regimes and assets.

- **Symmetric CUSUM filter** (Page 1954; Lam & Yam 1997, *J. Futures
  Markets* 17(8):921–944; AFML 2.5.2.1): emit an event when the cumulative
  deviation of a series from its running reference crosses ±h, resetting the
  reference at each event. Events mark *structural* moves, not noise around
  a rolling mean. Applied to returns, log-price residuals, or a model's own
  forecast errors.
- **Triple-barrier method** (AFML ch. 3): from event \(t_0\) with side
  \(s\in\{-1,+1\}\), barriers are
  \[
  \text{upper} = P_{t_0}(1 + pt\cdot\sigma_{t_0}),\quad
  \text{lower} = P_{t_0}(1 - sl\cdot\sigma_{t_0}),\quad
  \text{vertical} = t_0 + t_1,
  \]
  where \(\sigma_{t_0}\) is a point-in-time volatility estimate (EWMA of
  returns or ATR). The label is the **first** barrier touched; the vertical
  barrier bounds holding time. Volatility-scaled barriers equalize the
  risk across regimes — the single most consequential design choice in the
  labeling stack (AFML 3.2–3.3; mizarlabs' `TripleBarrierMethodLabeling`
  makes `volatility_adjusted_horizontal_barriers` the default).
- **Meta-labeling** (AFML 3.3–3.7): decouple *side* from *size*. A primary
  model (rule-based, fundamental, or technical) emits the side; a secondary
  ML model estimates \(P(\text{side correct}\mid \text{features})\), trained
  on the meta-label \(F = \mathbf{1}\{s\cdot y > 0\}\). Benefits: filters
  false positives, sizes by conviction, keeps the primary model
  interpretable, and raises precision without needing the secondary model to
  predict direction. Consolidated theory + framework: Joubert 2022,
  "Meta-Labeling: Theory and Framework", *J. Financial Data Science* 4(3):
  31–44 (<https://www.pm-research.com/content/iijjfds/4/3/31>);
  architecture: Meyer, Joubert & Alfeus 2022, *JFDS* 4(4):10–24;
  **calibration and position sizing**: Meyer, Barziy & Joubert 2023, *JFDS*
  5(2):23–40 — probability calibration (Platt/isotonic) significantly
  improves *fixed* sizing rules; data-learned sizers gain less. Bet sizing
  from meta-probability is AFML ch. 10.
- **Trend-scanning labels** (AFML 5.4.1): for each origin, regress log-price
  on time over candidate windows \(w\in\{2..W\}\); the label is the sign of
  the slope t-stat of the best (most significant) fit, optionally zeroed
  below a `min_t` threshold. Produces a continuous trend-strength signal and
  doubles as a horizon selector (the winning \(w\) is the locally dominant
  trend scale).
- **The Label Horizon Paradox** (Song, Liu & Chen 2026, arXiv:2602.03395,
  ICML 2026 poster 65178 — <https://arxiv.org/abs/2602.03395>): minimizing
  training error on the canonical target horizon \(t+\Delta\) does **not**
  maximize generalization on \(t+\Delta\); the optimal supervision signal
  often lives at an *intermediate* horizon \(t+\delta\), governed by a
  signal-realization vs noise-accumulation trade-off. Their fix is a
  **bi-level optimization**: an inner loop trains model weights on a weighted
  mixture of candidate-horizon labels, an outer loop updates the mixture
  weights against validation error on the true target. Directly relevant
  here: the repo already persists `future_return_{1,5,20}` — a proxy-label
  mixture study is infrastructure-cheap.
- **Adaptive Event-Driven Labeling (AEDL)** (2025, *Applied Sciences*
  15(24):13204 — <https://www.mdpi.com/2076-3417/15-24/13204>): multi-scale
  barrier horizons + causal-inference filtering (Granger causality /
  transfer entropy between features and candidate labels) + MAML
  meta-learning of per-asset labeling parameters. Caution: the paper
  headlines return-based metrics that this repo's honesty contract classifies
  as improper headline scores; adopt the **multi-scale horizon scan and
  causal filter** ideas and re-verify with proper scores (pinball/CRPS on
  the downstream forecaster, Brier/ECE on meta-labels), never the paper's
  headline numbers.

### 1.3 Sample uniqueness & sequential bootstrap (AFML ch. 4)

Overlapping event labels mean concurrent samples share information: a bar
\(t\) covered by \(c_t\) active labels contributes its return to \(c_t\)
labels. AFML ch. 4 quantifies and corrects this:

- **Concurrency** \(c_t\) = number of labels whose \([t_{i,0}, t_{i,1}]\)
  span covers \(t\) (AFML 4.1).
- **Average uniqueness**
  \(u_i = \operatorname{mean}_{t\in[t_{i,0},t_{i,1}]} 1/c_t\) (AFML 4.2) —
  the natural sample weight: a label that overlaps nothing has \(u_i=1\);
  a label buried in a crowded window has small \(u_i\).
- **Sample-weight variants** (AFML 4.5, 4.10): *return attribution*
  (\(|r_i|\) split across the concurrency it overlaps), *time decay*
  (linear decay floored at a `decay` fraction, newest = 1), and binary
  uniqueness thresholding.
- **Sequential bootstrap** (AFML 4.3): draw samples with probability
  proportional to *current* uniqueness, then remove the drawn label's
  contribution from the concurrency matrix so overlapping candidates lose
  draw weight. Produces a less-redundant training set than iid resampling
  and — critically — does not pretend the data are iid. Used for bagging
  and for choosing which events to train on when labels are dense.
- **Why it matters for this repo:** purged/embargoed CV (`validation/`)
  fixes the *validation* side of overlap; uniqueness weights fix the
  *estimation* side. Without them, an estimator trained on overlapping
  labels over-weights clustered events (e.g. every barrier in a volatility
  storm) and the effective sample size is far below the row count.

### 1.4 Fractionally differentiated features (AFML ch. 5)

- **The trade-off.** ML models need stationary features, but integer
  differencing (\(d=1\)) erases all memory. Fractional differentiation
  \(d\in(0,1)\) makes a series stationary while retaining long-memory
  structure (Hosking 1981, "Fractional differencing", *Biometrika* 68(1):
  165–176; Granger & Joyeux 1980, *J. Time Series Analysis* 1(1):15–29).
  Weights follow the binomial recursion
  \[
  w_0 = 1,\qquad w_k = -w_{k-1}\,\frac{d-k+1}{k},
  \]
  and \(\Delta^d x_t = \sum_{k\ge 0} w_k x_{t-k}\).
- **Unit-root preservation.** Granger & Ding (1996, "Varieties of long
  memory models", *J. Econometrics* 73(1):277–291) showed that even \(d=1\)
  retains *correlation* memory (the ACF is preserved in the limit) though
  variance diverges — the empirical motivation for scanning \(d\) rather
  than defaulting to 0 or 1. AFML 5.3 makes the same point with
  variance-vs-correlation plots.
- **FFD (fixed-width window fracdiff**, AFML 5.4.2): truncate weights where
  \(|w_k| < \tau\) (e.g. \(10^{-5}\)). Memory becomes bounded, the transform
  is a finite causal convolution (cheap and PIT-obvious), and — the key
  result — FFD retains the *maximum* level of memory among fixed-width
  filters for the given \(\tau\). The expanding-window version is exact but
  has weights that grow with \(t\) (slower, and its warm-up prefix differs).
- **Minimum-d search** (AFML 5.5): the smallest \(d\) such that the FFD
  series passes a unit-root test (ADF; KPSS as a complement) at level
  \(\alpha\). Empirically log-prices need \(d\approx 0.3\text{–}0.6\) —
  far below 1, i.e. most of the memory survives. **The d-search is an
  estimation step**: fitting it on the full sample and emitting the result
  as a feature is data snooping (LH003-class leakage); it must be fold-local
  or expanding-window.
- **Long-memory estimation** (for choosing/scanning \(d\) independently of
  ADF): GPH (Geweke & Porter-Hudak 1983), local Whittle (Künsch 1987;
  Robinson 1995), Whittle ARFIMA, Lo (1991) modified R/S — all already
  in-tree (`models/long_memory.py`).
- **Variance-equation cousins:** FIGARCH(1,d,1) (Baillie, Bollerslev &
  Mikkelsen 1996, *J. Econometrics* 74(1):3–25) puts the fractional
  operator in the volatility recursion; APARCH (Ding, Granger & Engle 1993,
  *J. Empirical Finance* 1(1):83–99) adds a power + asymmetric-leverage
  term. Both are in `models/garch_ext.py` (QMLE, news-impact curve).
- **Tempered fracdiff / ARTFIMA** (Giraitis, Kokoszka & Leipus 2000,
  *Stat. Sinica*; Sabzikar, McLeod & Meerschaert 2019, "Parameter
  estimation for ARTFIMA time series", *J. Statistical Planning and
  Inference* 200:129–145; `artfima` R package,
  <https://www.stt.msu.edu/users/mcubed/ARTFIMAfit.pdf>): multiply weights
  by \(e^{-\lambda k}\). Covariance becomes summable → stationary for
  *any* \(d\), with **semi-long memory**: power-law decay out to a horizon
  set by \(\lambda\), exponential beyond. Whittle-estimable. This is the
  principled extension when min-d FFD leaves residual non-stationarity or
  when the memory horizon itself is regime-dependent.

### 1.5 Volatility features: realized measures & the HAR family

- **Realized volatility** (Andersen, Bollerslev, Diebold & Labys 2001,
  *JASA* 96(453):42–55; Andersen, Bollerslev, Diebold & Labys 2003,
  *Econometrica* 71(2):579–625): \(RV_t=\sum_i r_{t,i}^2\) over intraday
  returns; \(\log RV\) is near-Gaussian, strongly persistent, and the
  canonical realized-measure target. **Optimal sampling** under
  microstructure noise: Aït-Sahalia, Mykland & Zhang (2005, *J. Finance*
  60(3):1279–1325); Liu, Patton & Sheppard (2015, "Does anything beat
  5-minute RV?", *J. Econometrics* 187(1):293–311) — 5-min RV is the robust
  default across asset classes.
- **Noise-robust estimators:** TSRV (Zhang, Mykland & Aït-Sahalia 2005,
  *JASA* 100(472):1394–1411), realized kernels (Barndorff-Nielsen, Hansen,
  Lunde & Shephard 2008, *Econometrica* 76(6):1481–1536), pre-averaging
  (Jacod, Li, Mykland, Podolskij & Vetter 2009, *Stochastic Processes and
  their Applications* 119(7):2249–2276).
- **Range-based estimators** (daily OHLC, ~2.5–5× more efficient than
  squared close-to-close): Parkinson (1980, "The Extreme Value Method for
  Estimating the Variance of the Rate of Return", *J. Business*
  53(1):61–65); Garman–Klass (1980, "On the Estimation of Security Price
  Volatilities from Historical Data", *J. Business* 53(1):67–78);
  Rogers–Satchell (1991, *J. Applied Econometrics* 6(4):315–329);
  Yang–Zhang (2000, *J. Business & Economic Statistics* 18(3):295–303,
  drift- and jump-robust with overnight gaps); Corwin–Schultz (2012,
  *J. Finance* 67(4):1373–1403)
  spread from high-low ratios. Hansen & Lunde (2005, *J. Applied
  Econometrics* 20(7):873–889) compared 330 models: **nothing beat
  GARCH(1,1) consistently on daily equity data** — a standing warning
  against fancy vol features that are never benchmarked properly.
- **Jumps and signed variation:** bipower variation and the BNS jump test
  (Barndorff-Nielsen & Shephard 2004, "Power and Bipower Variation with
  Stochastic Volatility and Jumps", *J. Financial Econometrics* 2(1):1–37;
  2006, "Econometrics of Testing for Jumps in Financial Economics Using
  Bipower Variation", *J. Financial Econometrics* 4(1):1–30);
  Lee–Mykland threshold jump tests (2008, "Jumps in Financial Markets: A
  New Nonparametric Test and Jump Dynamics", *Review of Financial Studies*
  21(6):2535–2563; 2012, "Jumps in Equilibrium Prices and Market
  Microstructure Noise", *J. Econometrics* 168(2):396–406);
  tripower quarticity; realized semivariance (Barndorff-Nielsen, Kinnebrock
  & Shephard 2010, *Quantitative Finance* 10(9):977–989); **good/bad
  volatility** (Patton & Sheppard 2015, "Good volatility, bad volatility:
  What should forecasters care about?", *J. Econometrics* 187(2):683–699) —
  downside semivariance has stronger predictive content for future vol than
  total RV; realized quarticity gives \(\operatorname{Var}(RV)\).
- **HAR** (Corsi 2009, "A simple approximate long-memory model of realized
  volatility", *J. Financial Econometrics* 17(2):174–196): regress next-day
  RV (or \(\sqrt{RV}\)) on daily/weekly/monthly averages
  \[
  RV_{t+1} = \beta_0 + \beta_d RV_t + \beta_w \overline{RV}_{t-4..t}
  + \beta_m \overline{RV}_{t-21..t} + \varepsilon_{t+1},
  \]
  a parsimonious cascade that mimics the Heterogeneous Market Hypothesis
  (Müller et al. 1997) and remains one of the strongest baselines
  (Patton 2011, "Volatility forecast comparison using imperfect volatility
  proxies", *J. Econometrics* 164(1):20–30 — also the source of the
  **QLIKE** and MSE-log loss functions this repo must use for vol scoring).
  Extensions: **HARQ** (Bollerslev, Patton & Quaedvlieg 2016, "Forecasting
  accuracy: Multivariate Heterogeneous Autoregressive", *J. Econometrics*
  194(1):1–18) adds realized-quarticity interactions; **HAR-RS / HAR-SJ**
  (Patton & Sheppard 2015) replace total RV with signed semivariances;
  log-HAR and GARCH-MIDAS (Engle, Ghysels & Sohn 2013, *Review of Economics
  and Statistics* 95(3):776–789) for mixed-frequency conditioning;
  **Realized GARCH** (Hansen, Huang & Shek 2012, *J. Financial
  Econometrics* 10(3):401–433) jointly models the measurement and variance
  equations — the family the pipeline already fits on daily Parkinson
  (`models/realized_garch.py`).
- **Rough volatility & signatures** (frontier): volatility paths are rough,
  Hurst ≈ 0.1 (Gatheral, Jaisson & Rosenbaum 2018, "Volatility is rough",
  *Quantitative Finance* 18(6):933–949); roughness-boosted HAR/log-vol
  models improve short-horizon forecasts (Bayer, Stemmer & Gatheral,
  arXiv:2410.11056); **path-signature features** of \((\log S, \log\sigma)\)
  summarize the joint path in a basis that ML models can consume
  (Lemercier, Wang & Zhang 2021, "Characteristic signatures of the rough
  Bergomi model", *Quantitative Finance* 21(11):1867–1877). Signature
  transforms are a candidate *feature* family for the 4h/1m corpus —
  deterministic, causal (truncated at \(t\)), and already adjacent to the
  repo's SSA/EMD decomposition tooling.

### 1.6 Feature selection & clustered importance (2020)

- **Substitution effects break naive importance.** When two features share
  predictive information, MDI dilutes across them and MDA (permutation)
  shuffles one while its substitute carries the signal — both understate
  importance and misrank (López de Prado 2020, "Clustered Feature
  Importance (Presentation Slides)", SSRN 3517595 —
  <https://ssrn.com/abstract=3517595>; and *Machine Learning for Asset
  Managers*, Cambridge Elements 2020, §6.5.2, SSRN 3558728).
- **CFI recipe:** (1) compute a **codependence** matrix — correlation-based
  \(d_{ij}=\sqrt{\tfrac12(1-\rho_{ij})}\) or information-theoretic
  (mutual information, distance correlation — the latter catches *nonlinear*
  redundancy); (2) cluster with the **ONC** algorithm (optimal number of
  clusters via eigengap + k-means silhouette, MLAM ch. 4) or hierarchical
  clustering on the distance; (3) **clustered MDI** = sum of impurity
  decrease within each cluster; **clustered MDA** = shuffle entire clusters
  simultaneously and score the drop; **SFI** = out-of-sample score of each
  feature *in isolation* (cross-section, no joint effects); (4)
  intra-cluster, pick a representative (highest SFI / medoid).
- Reference implementation semantics: mlfinlab
  `feature_importance/importance.py` + `feature_clusters`
  (<https://github.com/hudson-and-thames/mlfinlab>,
  <https://random-docs.readthedocs.io/en/latest/implementations/feature_clusters.html>).
- **mRMR** (Ding & Peng 2005, *IEEE TPAMI* 27(8):1226–1238) — already in
  `metrics/feature_select.py` — selects for max relevance / min redundancy;
  CFI is the complementary *importance* layer (cluster-level attribution),
  and both consume the same codependence primitives the repo already has
  (`metrics/dependence.py`: distance correlation, HSIC, MMD, Chatterjee ξ,
  KSG MI; `metrics/mic.py`: MIC; `models/spectral_clustering.py`;
  `metrics/cluster_validity.py`: silhouette / CH / DB).

### 1.7 The 2020–2026 frontier beyond AFML

1. **Meta-labeling matured into a three-part architecture** (side model /
   meta model / sizer) with calibration as a first-class concern
   (Meyer, Barziy & Joubert 2023, *JFDS* 5(2):23–40): fixed sizers gain
   materially from calibrated probabilities (Platt/isotonic), learned sizers
   less so; "sigmoid optimal position sizing" is the novel sizer. Practical
   consequence: a meta-label gate without a calibration + Brier/ECE harness
   is half-built.
2. **Label-horizon selection became a learnable object** (Song et al. 2026,
   §1.2): bi-level optimization over a mixture of horizon labels, trained
   end-to-end. The static version — train on an intermediate proxy horizon,
   evaluate on the target — is already testable with the repo's
   `future_return_{1d,5d,20d}` panel.
3. **Adaptive/event-driven labeling** (AEDL 2025, §1.2): multi-scale
   barriers + causal filtering (Granger/transfer entropy) + per-asset
   meta-learned parameters. The causal-filter component is the durable
   idea: screen candidate label constructions against features for
   *predictive causality*, not just correlation.
4. **Agentic / LLM factor mining** (hypothesis generators, not oracles):
   AlphaAgent (arXiv:2502.16789) regularizes LLM alpha mining with AST-based
   novelty, complexity control, and hypothesis-alignment scoring;
   AlphaQuant (SSRN 5124841) couples an LLM generator with an evolutionary
   illumination loop over hyper-tuned models; an LLM-powered MCTS mines
   formulaic alphas with backtest feedback and frequent-subtree avoidance
   (arXiv:2505.11122); R&D-Agent(Q) (arXiv:2405.01708) jointly evolves
   factors and models; a risk-aware multi-agent strategy-finding framework
   appeared at EMNLP 2025 Findings
   (<https://aclanthology.org/2025.findings-emnlp.1005.pdf>). **Contract
   stance:** these systems propose candidate features; nothing enters this
   repo's gold panel without purged-CV proper-score evidence and a receipt.
   Alpha decay is the failure mode they themselves regularize against.
5. **Conformal + distributional supervision** (adjacent lane, already
   in-tree): ACI online level adjustment (Gibbs & Candès 2021, NeurIPS,
   arXiv:2111.07460) wraps any quantile forecaster; the repo's
   `scripts/_aci_col.py` challenger applies it to a GARCH-t base. Feature
   engineering for *distributional* targets (pinball/CRPS-trained LightGBM
   with microstructure columns — `scripts/_lgbmqv_col.py`) is where new
   features should be scored: at quantile levels, not just means.

---

## 2. In-tree inventory

### 2.1 What exists (don't re-buy it)

| Capability | Module | State | Tests |
|---|---|---|---|
| Tick / volume / dollar bars | `features/bars.py` (`tick_bars`, `volume_bars`, `dollar_bars`) + numba `reset_bounds` | **Implemented, isolated** — no `src/` importer | `tests/unit/data/test_bars.py` (8) |
| Tick-imbalance / tick-run / dollar-imbalance bars | `features/bars.py` + numba `imbalance_bounds`, `run_bounds`; Lee–Ready-style `_tick_signs`; EWMA expectations | Implemented, isolated; **no volume/dollar run bars (VRB/DRB), no BVC** | same file |
| CUSUM event filter | `labels/barriers.py::cusum_filter` | Implemented, isolated | `tests/unit/quant_models/test_barriers.py` (13) |
| Triple-barrier labeling (vol-scaled, `min_ret`, touch/t_touch/ret outputs) | `labels/barriers.py::triple_barrier` | Implemented, isolated | same |
| Meta-labels (side-agreement) | `labels/barriers.py::meta_labels` | Implemented, isolated | same |
| Trend-scanning labels (best-t over sub-windows, `min_t`) | `labels/barriers.py::trend_scanning_labels` | Implemented, isolated | same |
| Concurrency / average uniqueness / sequential bootstrap (seeded) / time-decay / return-attribution weights | `labels/barriers.py` (5 functions) | Implemented, **no consumer** | same |
| Meta-label *gate* (expanding logistic, reduce-only sizing multiplier) | `lightspeed/metalabel.py` (`meta_label_gate`, `metalabel_multiplier`) | Wired into the lightspeed lane; **no calibration/ECE harness around it** | lightspeed tests |
| Fracdiff: weights, FFD, `frac_diff`, min-d ADF scan, expanding exact | `models/fracdiff.py` | Implemented; **not a feature column anywhere** | `tests/unit/models/test_fracdiff.py` (9) |
| Long-memory d estimators (GPH, local Whittle, Whittle ARFIMA, Lo R/S) | `models/long_memory.py` | Implemented | `tests/unit/models/test_long_memory.py` |
| FIGARCH(1,d,1) QMLE + APARCH + news-impact curve | `models/garch_ext.py` | Implemented | unit tests |
| HAR-RV / HARQ fit + forecast, Newey–West SEs | `models/har.py` | Implemented; used by `research/vol_bench.py` and `scripts/_har_col.py`; **not a feature column** | `tests/unit/models/test_har.py` (4) |
| Realized-measure canon: RV, bipower, tripower quarticity, BNS & Lee–Mykland jumps, TSRV, realized kernel, pre-averaged RV, semivariance, RQ, RV confidence band | `models/realized.py` | Implemented; **research-only** — the pipeline's vol path uses `models/realized_garch.py` (Hansen–Huang–Shek log-linear on daily Parkinson) instead | `tests/unit/models/test_realized.py` |
| Daily OHLC variance estimators (Parkinson/GK/RS/Yang–Zhang, Corwin–Schultz spread, Kyle λ, Roll spread, OFI, session RV, realized semivariance, Abdi–Ranaldo) | `northset/estimators.py` | Implemented; session RV not a gold feature | northset tests |
| Base feature engine (`features.v4`): returns 1/5/20, overnight & open–close (Lou–Polk–Skouras), momentum 5–252 + 12-1 + skip (Jegadeesh–Titman), reversal, z-vs-MA20, vol 20/60/EWMA(λ=0.94)/Parkinson/Garman–Klass/of-vol/downside, MAX/MIN (Bali–Cakici–Whitelaw), realized skew/kurt (Amaya et al.), 52w-high proximity (George–Hwang), Amihud/ADV/turnover/volume-vol, beta/idio-vol/residual momentum (Blitz–Huij–Martens). **Naming caution:** `vol_realized` here is the 20-bar RMS of *daily log returns* (`features/engine.py:107`) — not intraday realized variance and not the `models/realized_garch.py` Parkinson measure; the three must never be conflated in a receipt | `features/engine.py` (452 ln, 45-entry `feature_catalog` with family/lookback/PIT flags) | **Wired** into `pipeline/dataset.py` gold path | `tests/unit/pipeline/test_features_labels.py` etc. |
| Cross-sectional winsorize + robust-z (median/MAD, SD fallback) + percentile rank per `event_time`; sector-relative | `features/cross_sectional.py` | Wired; `PUBLIC_FEATURES` = 40 core + 3 long-lookback neutral-filled (Gu–Kelly–Xiu convention) in `models/ranking.py` | `tests/property/test_cross_sectional_rankic.py` |
| ~50-indicator TA canon; cycle/spectral filters (Goertzel, FFT dominant cycle, Hilbert IF, Ehlers SuperSmoother/roofing/bandpass); liquidity estimators (Amivest, Lesmond zero-freq, Lot, FHT, Pástor–Stambaugh, Hasbrouck λ, Glosten–Harris) | `features/indicators.py`, `features/cycles.py`, `features/liquidity.py` | Implemented (canon wave); not on the public card | canon-wave tests |
| Fixed-horizon labels: `future_return/log/excess/idio_{1d,5d,20d}`, `future_realized_vol/var`, `future_max_drawdown`, `future_tail_event`, sector-relative, **persisted `label_end_time_{h}`** for sparse-safe purging | `labels/engine.py`; `labels/forward.py` (LH001-exempt) | **Wired** | pipeline tests |
| Purged + embargoed CV, CPCV (paths, stitching, indices), walk-forward with `assert_no_label_overlap`, split builder consuming `label_end_time` | `validation/purging.py`, `validation/cpcv.py`, `validation/walk_forward.py`, `pipeline/train/splits.py` | **Wired** | unit + property |
| Feature selection: mRMR, univariate screen, forward-orthogonalized | `metrics/feature_select.py` | Implemented | unit tests |
| Attribution: deterministic permutation importance, SHAP | `research/explainability/attribution.py` | Implemented; **no clustered variant** | `tests/property/test_explainability_properties.py` |
| Codependence primitives for CFI: distance correlation, HSIC, MMD, Chatterjee ξ, KSG MI, MIC; spectral clustering; silhouette/CH/DB | `metrics/dependence.py`, `metrics/mic.py`, `models/spectral_clustering.py`, `metrics/cluster_validity.py` | Implemented | canon-wave tests |
| Vol challenger lanes scored on CRPS/pinball: `dip_har` (Parkinson-based √RV HAR, clipped coefficients, FHS quantiles), `dip_egarch`/`dip_egarchl`, `dip_mid` (GARCH-MIDAS), `dip_volm` (volume surprise), `dip_kde`, `dip_seas` (calendar slots), `dip_xbeta` (cross-asset), `dip_evt` (POT/GPD tails), `dip_aci` (Gibbs–Candès), `dip_lgbm_qv` (park_vol/rng_ratio/vol_surp/amihud feature table), `dip_stack`/`dip_stack2`, `dip_gmm_k` | `scripts/_*_col.py` (122 scripts total) | Wired into the shard/arena evaluation; **feature tables are vendored per-script, not shared** | arena receipts |
| Synthetic vol benchmark (GARCH-vol / rough-vol / break-vol shards; roll/EWMA/HAR/realized-GARCH forecasters; QLIKE-style scoring; receipt writer) | `research/vol_bench.py` | Wired (research lane) | unit tests |
| Leakage hunter LH001–LH014 (backward shift, centered windows, fit-before-split, as-of on `event_time`, frozen universe, forward-diff naming, unit-inflated PSR, forbidden headline strings, direct parquet, backfill, layering, comments, helper call sites) + allowlists | `leakage/rules.py`, `leakage/ast_scan.py` | **Wired**, clean-src gate | `tests/unit/test_leakage_clean_src.py` |
| Determinism: `derive_seed(base, index)`; simtest `DeterministicRuntime` (PCG64 stream, SHA-256 state hashes, `ReplayDivergence`); Hypothesis `derandomize=True`, CI profile `max_examples=100`; numba kernels under a bit-identical-to-Python contract (`@njit(cache=True)`, pure fallback) | `compute/parallel.py`, `simtest/`, `features/_kernels.py` | Wired | `tests/property/*` (27 files) |

Stray-tree note: `src/src/quant_fund/` (151 files) and `tests/tests/`
(457 files) are leftover mirrors; per `AGENTS.md` they stay out of default
collection. Verified: the canon-wave modules central to this lane
(`features/bars.py`, `labels/barriers.py`, `models/har.py`,
`models/fracdiff.py`, `models/realized.py`) exist **only** in the primary
tree — inventory claims below refer to `src/quant_fund/`.

### 2.2 Wired vs isolated — the core finding

The AFML canon is **implemented and unit-tested but electrically
disconnected** from the training path:

- `labels/barriers.py` has exactly one importer: its own test file. The gold
  label panel is `labels/engine.py` fixed-horizon only. No event set
  \([t_{i,0}, t_{i,1}]\), no side, no meta-label column, no uniqueness
  weight reaches `pipeline/dataset.py` or `pipeline/train/`.
- `features/bars.py` has no `src/` importer. The dataset path reads
  `silver/bars.parquet` (time bars) and calls `build_features` directly.
  The bar constructors also return tick-index dicts without `event_time` —
  they cannot be persisted as a panel as-is (see G2).
- `models/fracdiff.py` is imported by nothing outside `models/garch_ext.py`
  (which re-implements its own `_fracdiff_weights` for FIGARCH) and tests.
  No `fdiff_*` column exists in `feature_catalog` or `PUBLIC_FEATURES`.
- `models/har.py` and `models/realized.py` are research-lane only. The
  *pipeline's* volatility stack is `models/realized_garch.py` (daily
  Parkinson measure). HAR components (daily/weekly/monthly RV),
  semivariance ratios, and jump indicators are absent from the feature
  engine even at daily frequency, and `northset/estimators.py::
  session_realized_variance` is not consumed by the gold card.
- `sequential_bootstrap` / `average_uniqueness` / `time_decay_weights` /
  `return_attribution_weights` have no consumers. Training is unweighted;
  walk-forward correctness rests on purging/embargo alone.
- The challenger scripts each **vendor their own feature tables** (e.g.
  `_lgbmqv_col.py` re-derives park_vol/rng_ratio/vol_surp/amihud inline;
  `_stack2_col.py` copies other lanes' math to stay self-contained). That
  is deliberate for shard-binding determinism, but it means microstructure
  feature definitions are duplicated ~15× with no shared, versioned,
  catalog-registered source.

### 2.3 Infrastructure this lane can lean on

- **PIT choke points:** `pit/` vault + `asof` on `known_at`
  (property-tested: `tests/property/test_pit_asof_never_future.py`),
  LH004 gating, digest-keyed gold panel cache (`pipeline/dataset.py`
  `_PANEL_CACHE`, SHA-256 of parquet bytes — new columns auto-invalidate).
- **Versioning:** `FeatureMetadata` (name/version/lookback/frequency/
  source columns/PIT flag/family) + `FEATURE_SET_VERSION` ("features.v4").
  Any new feature family bumps the version and lands catalog entries —
  the receipt chain then distinguishes gold vintages.
- **Scoring:** `metrics/scoring.py` (CRPS/pinball), QLIKE-style losses in
  the vol lanes, `metrics/calibration2` (wave 8), VaR backtests
  (Kupiec/Christoffersen/TUFF/Basel, wave 3), DM/SPA/MCS machinery in
  `reality/` + `research/benches/`.
- **Seeds:** `derive_seed(base_seed, index)`; simtest PCG64 discipline;
  Hypothesis derandomized profiles.

---

## 3. Gap list (ranked by leverage)

| # | Gap | Evidence | Impact |
|---|---|---|---|
| **G1** | **Event-label stack unwired**: triple-barrier/meta-label/trend-scanning/CUSUM exist but no gold columns, no \([t_0,t_1]\) persistence, no side pipeline | `labels/barriers.py` imported only by its tests; `build_labels` is fixed-horizon | The entire AFML labeling canon (and its sample-weight machinery) is unusable by `pipeline/train`; meta-labeling research can't start |
| **G2** | **Information-driven bars unwired + incomplete output contract**: no `event_time` on produced bars, no threshold-calibration helper (EWMA E[T] bootstrap), no VRB/DRB, no BVC; no tick corpus in the lake (1m adapter exists but bars path expects ticks) | `features/bars.py` returns index-keyed dicts; zero `src/` importers | 4h/1m crypto lanes cannot test the volume/dollar-clock hypothesis; bar-scheme metadata absent from feature versioning |
| **G3** | **Fractionally differentiated features absent from the card**: no `fdiff_*` columns, no Polars-native FFD, min-d not fold-local anywhere | `feature_catalog()` has no memory family; `frac_diff` unused | Stationarity-vs-memory features — the canonical fix for non-stationary price levels — never reach the rankers |
| **G4** | **HAR/RV estimators not first-class features**: no rv daily/weekly/monthly components, no semivariance (good/bad vol) columns, no jump-indicator columns, session RV unused on the card; `realized.py` canon divorced from the pipeline vol path | `models/har.py` importers = `vol_bench` + tests; `PUBLIC_FEATURES` vol family is 20/60-day rolling only | Vol forecasting features stop at Parkinson/GK/EWMA; the strongest published baselines (HAR, HARQ, HAR-SJ) are research-only |
| **G5** | **Clustered feature importance missing**: no ONC, no clustered MDI/MDA/SFI; attribution is per-feature permutation/SHAP; selection is mRMR/univariate/forward-orthogonalized | `metrics/feature_select.py`, `research/explainability/attribution.py` | With 43 public CS features (many near-duplicates by construction), substitution effects will misrank importance; cluster-level attribution is assembly work on primitives the repo already has |
| **G6** | **Sample-uniqueness weights unused in training**: no weight columns in gold; `sequential_bootstrap` unconsumed; overlapping fixed-horizon labels handled only by purge/embargo | §2.2 | Estimators over-weight clustered events; effective sample size overstated; bagging cannot be uniqueness-aware |
| **G7** | **Meta-label gate has no calibration/scoring harness**: `lightspeed/metalabel.py` emits a probability multiplier; no Brier/ECE/log-score evaluation, no Platt/isotonic step, no link to bet-sizing theory (AFML ch. 10; Meyer et al. 2023) | module + tests | The gate's probability is uncalibrated by construction; sizing claims can't be honestly evaluated |
| **G8** | **Label-horizon selection unstudied**: three horizons persisted, but no proxy-label mixture, no horizon-per-regime scan, no bi-level study (Song et al. 2026) | `labels/engine.py`; `config.horizons` static | Potential free lunch: training on 5d labels while targeting 1d/20d is testable today with existing columns |
| **G9** | **Challenger feature tables vendored, not shared**: ~15 scripts re-derive microstructure features inline; no versioned `microstructure_features` module feeding both arena and gold | `scripts/_lgbmqv_col.py`, `_stack2_col.py`, `_volm_col.py` docstrings | Definition drift between arena and pipeline; new features must be re-implemented per lane |
| **G10** | **VPIN / bulk volume classification absent**: no flow-toxicity metric on volume bars (tick-rule signs exist; BVC does not) | `features/bars.py`, `northset/estimators.py` (OFI only) | The toxicity feature with the strongest published microstructure evidence is unavailable even for synthetic-LOB correctness tests |

---

## 4. Adoption plan

Design constraints honored throughout: proper scores only; SYNTHETIC
results labeled as correctness tests; every new column is
`FeatureMetadata`-registered and version-bumped; determinism via
`derive_seed` + derandomized Hypothesis; PIT via LH001–LH014 and the
`known_at` vault; no live-trading claims.

### 4.1 Module design

**Phase 1 — Event labels + uniqueness weights (G1, G6). New
`labels/events.py`** (inside the LH001-allowlisted `labels/` package):

```python
# proposal — labels/events.py
@dataclass(frozen=True)
class EventLabelSpec:
    cusum_h: float | None          # None = every bar is an event
    pt: float; sl: float           # barrier multipliers on sigma_t
    horizon: int                   # vertical barrier, bars
    vol_estimator: str = "ewma"    # "ewma" | "atr" | "parkinson"
    vol_lambda: float = 0.94

def build_event_labels(bars: pl.DataFrame, spec: EventLabelSpec,
                       side: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per security: CUSUM (or all-bar) events -> triple_barrier on
    close_total_return with fold-visible sigma_t -> columns
    event_t0, event_t1 (= t_touch), event_label, event_ret, event_touch,
    event_side (from `side` join when meta-labeling), event_meta_label.
    Then concurrency/average_uniqueness over [t0, t1] ->
    event_uniqueness, event_weight_decay, event_weight_return.
    sigma_t uses only bars <= t (LH001: trailing EWMA/ATR)."""
```

Gold integration: `pipeline/dataset.py` gains an optional
`event_labels` parquet beside `labels.parquet`, joined on
`(security_id, event_t0)`; `label_end_time` semantics carry over
(`event_t1` is the exact endpoint the purge/embargo splitter already
consumes via `_aligned_label_end_times`). Weights flow into
`design_matrix` as a `sample_weight` column — **training metadata only,
never a feature** (uniqueness encodes \(t_1\), i.e. the future).

**Phase 2 — Fracdiff feature columns (G3). Extend `features/engine.py`**
with `add_memory_preserving_features(df, d_map)`:

- Polars-native FFD convolution per security (weights from
  `models/fracdiff.py::frac_weights_ffd`, cached on `(d.hex(), threshold)`);
  NaN warm-up prefix of `len(w)-1` bars, never zero-padded (matches
  `frac_diff` semantics).
- Columns: `fdiff_log_price` (primary), optionally `fdiff_mom_20`.
  Family `"memory"`, lookback = FFD window, `point_in_time_safe=True`.
- **`d_map` is fold-local**: an expanding-window min-d scan
  (`min_d_stationary` on data up to the fold boundary, grid 0–1 step 0.05,
  α=0.05) produces per-security d *inside* `pipeline/train/splits.py`
  folds; the chosen d and its ADF p-value are persisted in the receipt.
  A single global d fitted on the full sample is forbidden (LH003-class).
- `FEATURE_SET_VERSION` → `features.v5`.

**Phase 3 — HAR/RV feature columns (G4). New `features/vol_features.py`:**

- Daily card (from OHLCV alone, PIT-trailing):
  `rv_d` (Parkinson proxy — the pipeline's existing realized measure),
  `rv_w` = mean(rv_d, 5), `rv_m` = mean(rv_d, 22) — the HAR cascade inputs;
  `har_fore` = fold-local `har_rv_fit` 1-step prediction (√RV form,
  coefficients clipped ≥ 0 as in `scripts/_har_col.py`);
  `semivar_ratio` = downside/total from `models/realized.py::semivariance`
  applied to trailing returns (Patton–Sheppard good/bad vol);
  `jump_flag_20` = fraction of trailing bars flagged by a trailing
  threshold test (Lee–Mykland on the 20-bar window);
  `rv_q` interaction column for HARQ when quarticity is finite.
- 4h/1m card (when the intraday corpus is in the lake):
  `session_rv` from `northset/estimators.py::session_realized_variance`,
  `session_semivar_up/down`, pre-averaged RV and TSRV as noise-robust
  alternates, all aggregated to the decision bar with sessions defined
  explicitly (24/7 crypto — no implicit exchange calendar).
- Every HAR *fit* is expanding-window per fold; the forecast column at
  row \(t\) uses coefficients estimated on rows \(< t\) only.

**Phase 4 — Clustered feature importance (G5). New
`metrics/clustered_importance.py`:**

```python
# proposal — metrics/clustered_importance.py
def codependence_distance(x, method="dcor") -> Array      # sqrt(1-dCor)/MI options
def onc_cluster(x, method="dcor") -> dict[str, Array]     # eigengap + silhouette (ONC)
def clustered_mdi(model, x, y, clusters) -> AttributionResult
def clustered_mda(predict_fn, x, y, clusters, seed) -> AttributionResult
def sfi(x, y, column, fit_score_fn, seed) -> AttributionResult
def cluster_representatives(x, clusters, sfi_scores) -> dict[str, int]
```

Reuses `metrics/dependence.py::distance_correlation`,
`metrics/cluster_validity.py::silhouette_score`,
`models/spectral_clustering.py`, and the deterministic
`research/explainability/attribution.py` permutation machinery (shuffle
whole clusters instead of single columns; `seed=derive_seed(base, k)`).
MDA/SFI score with **proper losses only** (pinball/CRPS/QLIKE/Brier —
injected via `fit_score_fn`); no return-based headline may enter an
`AttributionResult`.

**Phase 5 — Bars panel + calibration (G2, G9, G10). New
`features/bars_panel.py`:**

- `calibrate_bar_threshold(ticks, scheme, alpha_ewma)` → EWMA E[T]
  bootstrap from the first N bars (AFML 2.3.2.2), deterministic.
- `build_bar_panel(tick_frame, scheme, threshold, tz)` → Polars frame with
  `event_time` = bar-close timestamp, OHLCVD, `bar_scheme`, `bar_threshold`
  columns; schemes: `tick|volume|dollar|tib|trb|dib|vrb|drb` (adds the
  missing volume/dollar run bars via a `run_bounds` variant on signed
  volume/dollar).
- `bulk_volume_classify(ohlcv)` → BVC buy/sell volume split
  (Easley et al. 2016) — unlocks VPIN and dollar-imbalance bars on
  *aggregated* data (the 1m corpus), not just ticks.
- `vpin(volume_bars, n_buckets)` → flow-toxicity column, validated first
  on `microstructure/synthetic_lob.py` output (SYNTHETIC correctness test).
- A shared `microstructure/features.py` extracted from the vendored
  challenger tables (`park_vol`, `rng_ratio`, `vol_surp`, `amihud_20`) so
  arena scripts and the gold card cite one definition; scripts keep their
  vendored copies until re-spliced (shard-binding hashes make silent
  drift detectable).

**Phase 6 — Meta-label calibration + horizon study (G7, G8), research
lanes with receipts:**

- `lightspeed/metalabel.py` gains an optional in-fold calibrator
  (isotonic or Platt on the expanding window) and emits
  `meta_prob_calibrated`; a scoring harness reports **Brier, log score,
  ECE** (via `metrics/calibration2`) per fold, plus Kupiec/Christoffersen
  on the gate's implied coverage. Sizing stays reduce-only; no live claim.
- `labels/horizon_study.py`: static proxy-label experiment — train the
  ranker on `future_return_5d` (or a fixed mixture over 1d/5d/20d), score
  on each target horizon under CPCV; report pinball/CRPS deltas with DM
  tests. The bi-level learnable mixture (Song et al. 2026) is a follow-on
  only if the static study shows a horizon gap worth optimizing.

### 4.2 Determinism & seed requirements

1. **Features are RNG-free.** Every column in Phases 2/3/5 is a pure
   function of (data bytes, config, feature-set version). The only seeded
   consumers are `sequential_bootstrap`, clustered MDA shuffles, and any
   bagging — all take `seed: int` sourced from
   `compute/parallel.py::derive_seed(base_seed, index)` and recorded in the
   run receipt.
2. **Numba bit-identity.** New kernels (VRB/DRB run bounds, FFD
   convolution if JITed) follow the `features/_kernels.py` contract:
   `@njit(cache=True)`, same float-op order as the Python reference, and a
   bit-identical equivalence test (pattern:
   `tests/unit/features/test_hot_path_equivalence.py`).
3. **Weight caches** key on exact float hex (`d.hex()`, `threshold.hex()`)
   — no tolerance-based cache hits.
4. **Fold-local estimators** (min-d, HAR coefficients, calibrators, ONC
   clusters) persist their fitted parameters + seed into the receipt so a
   rerun reproduces the columns byte-for-byte; the digest-keyed gold cache
   (`_PANEL_CACHE`) invalidates automatically when any of it changes.
5. **Property tests run derandomized** (`derandomize=True`, CI profile
   `max_examples=100`) — an invariant that fails only on some seeds is a
   bug, not a flake.
6. **Bar panels are reproducible from (tick bytes sha256, scheme,
   threshold, alpha_ewma)** — the same binding discipline the challenger
   column scripts already use (`bars_sha256` + config fields).

### 4.3 PIT / leakage contract (LH mapping)

| Rule | Application to this lane |
|---|---|
| LH001 (backward shift) | Event labels live under `labels/` (allowlisted path). `sigma_t` for barriers, FFD weights, HAR lags, session RV aggregates are trailing-only in `features/`. |
| LH002 (centered windows) | All rolling stats trailing; trend-scanning regressions end at \(t\). |
| LH003 (fit-before-split) | min-d scan, HAR/HARQ fits, meta-label calibrators, ONC clustering, CFI-MDA baselines are fold-local or expanding; never full-sample. |
| LH004 (as-of on event_time) | Bar panels and event joins keyed on `known_at` where restatements exist; tick corpora are append-only. |
| LH005 (frozen universe) | Event/bar construction runs per security over the PIT membership frame, same as `build_features`. |
| LH006 (forward-diff naming) | Barrier/meta columns are `event_*`/`fwd_*`-prefixed targets; never `delta_*`. |
| LH008/LH013 (forbidden headlines) | Attribution/scoring APIs accept proper-loss callables only; doc-level discipline mirrored in code review. |
| LH010 (backfill) | FFD warm-up prefix stays NaN; no `bfill` on event frames. |
| New (proposed LH015) | **Uniqueness/weight columns may not appear in any feature list** — they encode \(t_1\); gate in `design_matrix` with an explicit deny-list (`event_uniqueness`, `event_weight_*`, `event_t1`). |

### 4.4 Property-based tests per feature (Hypothesis invariants)

Strategies: `st.lists(st.floats(...))` for prices/returns with
`min_value>0` where needed; stateful machines where order matters; all
derandomized. Reference oracles are pure-Python loops (the numba contract).

**Bars (`features/bars.py` + `bars_panel.py`)**

| Invariant | Sketch |
|---|---|
| Volume/dollar conservation | Σ bar volumes ≤ Σ tick sizes; equality iff the final partial bar is included; dollar bars conserve Σ p·s likewise |
| OHLC consistency | per bar: low ≤ {open, close} ≤ high; open = first tick price; close = last; high/low = reduceat extremes |
| Monotone thresholds | larger θ ⇒ fewer-or-equal bars, and bar boundaries are a subsequence (prefix-stability) |
| Determinism / kernel identity | numba bounds == Python reference bounds, bitwise, on random tick streams |
| Tick-sign rules | `_tick_signs` output ∈ {−1,0,+1}; zeros only before the first nonzero move; forward-fill never creates a sign from nothing |
| Imbalance/run firing | a bar closes only when \|Σ signed flow\| (or max run) crossed the EWMA threshold at that tick — recompute threshold path independently |
| Gaussianity (SYNTHETIC) | on a synthetic mixed-frequency tick stream, dollar-bar returns have lower autocorrelation(1) and higher normality (JB p) than time-bar returns — labeled correctness test, not market evidence |
| BVC | buy+sell volume = bar volume; sign flips with return sign; monotone in return |

**Event labels (`labels/barriers.py`, `labels/events.py`)**

| Invariant | Sketch |
|---|---|
| Label domain | `label ∈ {−1,0,+1}`; `touch ∈ {−1,0,+1}`; `label = touch` whenever `touch ≠ 0` |
| Touch-time bounds | `t0 < t_touch ≤ min(t0+horizon, n−1)`; `ret` equals the path return at `t_touch` (recomputed independently) |
| First-touch | if both barriers are crossed within the path, the earlier index wins; equal-index ties resolve to the upper barrier (document + test) |
| Vol-scaling homogeneity | scaling `vol` by c>0 scales barrier distances by c; labels invariant under a joint (price, σ) rescale that preserves crossings |
| Monotone barriers | larger pt ⇒ fewer +1 labels (weakly), larger sl ⇒ fewer −1 labels |
| Meta-labels | `meta = 1[s·y>0]`; side=0 ⇒ meta=0; flipping both side and label leaves meta unchanged |
| Trend scanning | NaN prefix length = window−1; `|t_best| < min_t ⇒ 0`; sign(t) = sign(best slope); on a strictly log-linear ramp, label = +1 for every valid origin |
| CUSUM | events strictly increasing; between events the cumulative deviation never crossed ±h (recompute); h↑ ⇒ events⊆ events(h↓) |
| Concurrency/uniqueness | `c_t ≥ 1` on every event span; `u_i ∈ (0,1]`; disjoint spans ⇒ all `u_i = 1`; nested spans ⇒ inner uniqueness < outer |
| Sequential bootstrap | same seed ⇒ identical draws; draw count exact; drawn multiset's mean uniqueness ≥ iid-bootstrap mean uniqueness (AFML 4.3 claim, tested statistically with a fixed seed pair); removing a drawn window's contribution never makes any `c_t` < 1 (the `maximum(counts,1)` floor) |
| Weights | time-decay: newest = 1, oldest = decay, monotone in age; return-attribution: weights ≥ 0, scale linearly in \|r\|, decrease when concurrency increases |
| Endpoints | `event_t1` equals `t_touch`; splitter purge using `event_t1` leaves zero train/test label overlap (`assert_no_label_overlap` over generated event sets) |

**Fracdiff (`models/fracdiff.py` + engine columns)**

| Invariant | Sketch |
|---|---|
| Weight recursion | `w_k = −w_{k−1}(d−k+1)/k` for all k; `w_0 = 1`; integer d terminates at `w_{d+1} = 0` |
| d=0 / d=1 identities | `frac_diff(x, 0) == x` (after warm-up); `frac_diff(x, 1) == np.diff(x, prepend=x[0])` within FFD truncation bound |
| FFD ≈ expanding | on finite series, `frac_diff` (FFD) and `expanding_frac_diff` agree within the documented tail bound `τ·Σ|x|` after the warm-up prefix |
| Warm-up honesty | first `len(w)−1` outputs are NaN, never 0/ffill |
| Constant series | FFD of a constant decays toward 0 as memory rolls off (the AFML 5.4.2 demo) |
| min-d | on SYNTHETIC ARFIMA(d*) with known d*: returned d̂ ≥ d*−grid step and ADF(FFD(x, d̂)) rejects at α; d̂ = NaN when no grid point rejects (fail-closed, never d=0 fallback) |
| Memory preservation | corr(FFD(x,d), x) at lag 1 decreases in d and stays > corr(diff(x), x) for d<1 (the memory-vs-stationarity trade-off, tested on a persistent synthetic series) |
| Column PIT | `fdiff_log_price` at row t is unchanged when rows > t are appended/restated (prefix-stability under the FFD window) |

**Volatility features (`models/har.py`, `models/realized.py`, `features/vol_features.py`)**

| Invariant | Sketch |
|---|---|
| RV family | RV ≥ 0; bipower ≤ RV when a jump is injected (SYNTHETIC); up+down semivariance = RV; quarticity ≥ 0; TSRV ≤ RV under injected microstructure noise; pre-averaged RV → IV on clean SYNTHETIC diffusion within tolerance |
| Jump tests | BNS/Lee–Mykland flag injected jumps on synthetic diffusions with ≥ nominal coverage; no-jump synthetic keeps false-positive rate ≤ α (Monte-Carlo with fixed seeds) |
| HAR fit | on synthetic RV generated from known HAR coefficients, OLS recovers them within Newey–West CIs; NW SEs > 0 and ≥ iid SEs under positive autocorrelation; `harq_fit` reduces to `har_rv_fit` when quarticity is constant |
| HAR forecast | forecasts ≥ 0 (√RV form with clipped coefficients); 1-step forecast is a deterministic function of the trailing (1,5,22) components |
| Feature columns | `rv_w` at t = mean of `rv_d` over the trailing 5 rows (recompute); `har_fore` at t uses only coefficients fit on rows < t (prefix-stability: appending future rows does not change past `har_fore`) |
| QLIKE sanity (SYNTHETIC) | on synthetic HAR-generated RV paths, the true-coefficient forecaster beats a shuffled-coefficient forecaster on QLIKE — correctness test of the loss wiring, labeled SYNTHETIC |
| Session RV | aggregation over a synthetic 24/7 session matches the sum of intraday squared returns; session boundaries explicit, never inferred from gaps |

**CFI (`metrics/clustered_importance.py`)**

| Invariant | Sketch |
|---|---|
| Distance metric | dCor-distance ∈ [0,1], symmetric, zero iff independent (on planted independent SYNTHETIC columns), captures a planted nonlinear duplicate that Pearson misses |
| ONC | every feature in exactly one cluster; singleton clusters when all columns independent; perfect partition recovery on planted block-correlated SYNTHETIC data |
| Clustered MDI | sums to the same total as plain MDI (partition invariance); a planted signal's cluster ranks above noise clusters |
| Clustered MDA | shuffling a whole informative cluster degrades the proper score more than shuffling one member; deterministic under fixed seed |
| SFI | SFI of a feature in isolation ≤ its joint contribution when complements exist (planted XOR-style interaction); SFI ordering stable across seeds within tolerance |
| Substitution recovery (the CFI claim) | on planted duplicate pairs, plain MDA splits importance across the pair while clustered MDA concentrates it on the cluster — tested with fixed synthetic construction |

**Meta-label calibration (G7)**

| Invariant | Sketch |
|---|---|
| Calibrator monotonicity | isotonic/Platt output monotone in raw score; calibrated probabilities ∈ [0,1] |
| ECE/Brier wiring | on SYNTHETIC perfectly-calibrated draws, ECE ≈ 0 within MC tolerance (fixed seed); Brier ≤ 0.25 always; Brier decomposition = calibration + refinement (identity test) |
| Gate causality | `meta_label_gate` at row t trained only on rows < t (expanding window; prefix-stability test) |

### 4.5 Verification & proper-score benchmarks

- **Vol features:** QLIKE + MSE-on-log-RV (Patton 2011) against
  `future_realized_var_{h}` (already persisted) under CPCV; DM tests via
  the existing bench machinery; the synthetic `research/vol_bench.py`
  shards (garch/rough/break) gate correctness before any real-corpus run.
- **Distributional columns:** pinball at the arena's `LGBM_TAUS` grid and
  CRPS via `crps_from_quantiles` — the same convention the challenger
  scripts use, so arena and gold evidence stay comparable.
- **Meta-labels:** Brier, log score, ECE per fold; Kupiec/Christoffersen
  on implied coverage; **never** a return-based headline.
- **Label-scheme comparison (G8):** CPCV-path pinball/CRPS of the same
  downstream model trained under {fixed-horizon, triple-barrier,
  trend-scanning, proxy-mixture} labels; schemes are evidence-ranked only
  through these scores, with receipts (`verify-research`).
- **CFI:** planted-signal SYNTHETIC suites first (correctness), then
  attribution stability across CPCV folds on the real card (evidence).
- Every benchmark run writes a receipt binding: data sha256, feature-set
  version, fold-local estimator parameters, seeds, and score definitions.

### 4.6 Suggested landing order

1. **Phase 1** (events + uniqueness weights + LH015 deny-list) — unblocks
   G1/G6 and every labeling study; all math already exists and is tested.
2. **Phase 4** (CFI) — pure assembly on existing primitives; immediately
   useful for pruning the 43-column public card before adding families.
3. **Phase 2** (fracdiff columns, fold-local min-d) — small surface,
   version bump to `features.v5`.
4. **Phase 3** (HAR/RV columns, daily first) — reuses `models/har.py` +
   `realized.py`; 4h session-RV variant follows the intraday corpus.
5. **Phase 6** (meta-label calibration + horizon study) — research lane,
   receipted; static horizon study before any bi-level machinery.
6. **Phase 5** (bars panel + BVC/VPIN + shared microstructure features) —
   largest data dependency (tick/1m corpus in the lake); lands last but the
   `bars_panel.py` contract (event_time, scheme metadata, threshold
   calibration) should be agreed first so G9's shared module can target it.

Each phase: unit tests + property tests (§4.4) green, `make lint`,
`make typecheck`, `make test`, leakage clean-src gate green, catalog +
version bump, receipt for any benchmark claim.

### 4.7 Explicitly deferred / rejected

- **AEDL's MAML meta-learning** — per-asset learned labeling parameters
  multiply the estimation surface without a determinism story that fits the
  receipt model; adopt only its multi-scale horizon scan (Phase 6) and
  causal-filter idea (transfer entropy already exists in `metrics/`).
- **Agentic LLM factor mining as an automatic feature source** — rejected
  as a pipeline input (alpha-decay regularization is the papers' own
  admission of fragility); acceptable only as a hypothesis generator whose
  candidates must pass CFI + purged-CV proper scores + receipts like any
  human-proposed feature.
- **ARTFIMA/tempered fracdiff columns** — deferred until min-d FFD shows
  residual non-stationarity on real cards; the Whittle-estimation machinery
  would be a new dependency surface for an unproven marginal gain here.
- **Signature-transform features** — promising for the rough-vol regime
  (Lemercier et al. 2021) but premature before the intraday bar panel
  (Phase 5) exists; revisit as a `features/signatures.py` study.
- **Volume/dollar imbalance bars on daily data** — meaningless without
  intraday flow; daily "imbalance" proxies (overnight vs intraday split)
  are already on the card.

---

## 5. References

**Foundations (AFML canon).**
López de Prado (2018), *Advances in Financial Machine Learning*, Wiley —
ch. 2 (bars), ch. 3 (labeling, meta-labeling), ch. 4 (sample weights,
sequential bootstrap), ch. 5 (fractional differentiation, trend scanning),
ch. 7 (purging/embargo), ch. 8 & 12 (CPCV), ch. 10 (bet sizing).
López de Prado (2020), *Machine Learning for Asset Managers*, Cambridge
Elements (SSRN 3558728) — ch. 4 (ONC), §6.5.2 (CFI).
López de Prado (2020), "Clustered Feature Importance", SSRN 3517595.

**Bars & microstructure.**
Easley, López de Prado & O'Hara (2012), "The Volume Clock", *JPM* 39(1).
Easley, López de Prado & O'Hara (2012), "Flow Toxicity and Liquidity in a
High-frequency World", *RFS* 25(5):1457–1493.
Easley, López de Prado & O'Hara (2011), "The Microstructure of the 'Flash
Crash'", *JPM* 37(2):118–128. — Easley, López de Prado & O'Hara (2016),
"Bulk Volume Classification from Aggregated and Individual Trades",
*J. Financial Markets* 30. — Lee & Ready (1991), *J. Finance* 46(2).
Clark (1973), *J. Business* 46; Ané & Geman (2000), *Order Flow, Trading
Activity and Financial Risk*, Wiley. Lam & Yam (1997), *J. Futures
Markets* 17(8). Page (1954), *Biometrika* 41.

**Meta-labeling.**
Joubert (2022), "Meta-Labeling: Theory and Framework", *JFDS* 4(3):31–44.
Meyer, Joubert & Alfeus (2022), "Meta-Labeling Architecture", *JFDS*
4(4):10–24. Meyer, Barziy & Joubert (2023), "Meta-Labeling: Calibration and
Position Sizing", *JFDS* 5(2):23–40.

**Fractional differentiation & long memory.**
Hosking (1981), *Biometrika* 68(1):165–176. Granger & Joyeux (1980),
*J. Time Series Analysis* 1(1). Granger & Ding (1996), *J. Econometrics*
73(1):277–291. Geweke & Porter-Hudak (1983), *J. Time Series Analysis*
4(1). Künsch (1987), *Ann. Statistics* 15(3); Robinson (1995),
*J. Econometrics* 68(1). Lo (1991), *J. Finance* 46(4). Baillie,
Bollerslev & Mikkelsen (1996), *J. Econometrics* 74(1):3–25. Ding, Granger
& Engle (1993), *J. Empirical Finance* 1(1). Giraitis, Kokoszka & Leipus
(2000), *Statistica Sinica* 10. Sabzikar, McLeod & Meerschaert (2019),
*J. Statistical Planning and Inference* 200:129–145 (ARTFIMA parameter
estimation + `artfima` R package, MSU).

**Realized volatility & HAR.**
Andersen, Bollerslev, Diebold & Labys (2001), *JASA* 96(453); (2003),
*Econometrica* 71(2). Corsi (2009), *J. Financial Econometrics* 17(2):
174–196. Bollerslev, Patton & Quaedvlieg (2016), *J. Econometrics*
194(1):1–18 (HARQ). Patton & Sheppard (2015), *J. Econometrics*
187(2):683–699. Patton (2011), *J. Econometrics* 164(1):20–30 (QLIKE).
Barndorff-Nielsen & Shephard (2004), *J. Financial Econometrics* 2(1);
(2006), *J. Financial Econometrics* 4(1); Barndorff-Nielsen, Kinnebrock &
Shephard (2010), *Quantitative Finance* 10(9). Lee & Mykland (2008),
*RFS* 21(6); (2012), *J. Econometrics*
168(2):396–406. Zhang, Mykland & Aït-Sahalia
(2005), *JASA* 100(472). Barndorff-Nielsen, Hansen, Lunde & Shephard
(2008), *Econometrica* 76(6). Jacod, Li, Mykland, Podolskij & Vetter
(2009), *SP&TA* 119(7). Aït-Sahalia, Mykland & Zhang (2005), *J. Finance*
60(3). Liu, Patton & Sheppard (2015), *J. Econometrics* 187(1). Parkinson
(1980), *J. Business* 53(1):61–65; Garman & Klass (1980), *J. Business*
53(1):67–78; Rogers & Satchell (1991); Yang & Zhang
(2000); Corwin & Schultz (2012); Hansen & Lunde (2005). Engle, Ghysels &
Sohn (2013), *REStat* 95(3) (GARCH-MIDAS). Hansen, Huang & Shek (2012),
*J. Financial Econometrics* 10(3) (Realized GARCH). Müller et al. (1997),
Heterogeneous Market Hypothesis. Gatheral, Jaisson & Rosenbaum (2018),
*Quantitative Finance* 18(6). Bayer, Stemmer & Gatheral (2024),
arXiv:2410.11056. Lemercier, Wang & Zhang (2021), *Quantitative Finance*
21(11).

**Feature selection.**
Ding & Peng (2005), *IEEE TPAMI* 27(8) (mRMR). Székely, Rizzo & Bakirov
(2007), distance correlation. Chatterjee (2021), *JASA* 116(536) (ξ).
Reshef et al. (2011), *Science* 334 (MIC).

**2020–2026 successors.**
Song, Liu & Chen (2026), "The Label Horizon Paradox: Rethinking Supervision
Targets in Financial Forecasting", arXiv:2602.03395 / ICML 2026.
"Adaptive Event-Driven Labeling: Multi-Scale Causal Framework with
Meta-Learning for Financial Time Series" (2025), *Applied Sciences*
15(24):13204. AlphaAgent (2025), arXiv:2502.16789. AlphaQuant (2025),
SSRN 5124841. LLM-MCTS alpha mining (2025), arXiv:2505.11122.
R&D-Agent(Q) (2024), arXiv:2405.01708. "Automate Strategy Finding with LLM
in Quant Investment" (2025), EMNLP Findings. Gibbs & Candès (2021),
NeurIPS, arXiv:2111.07460 (ACI). Gu, Kelly & Xiu (2020), *RFS* 33(1)
(characteristic neutral-fill convention). Bailey, Borwein, López de Prado
& Zhu (2017), "The Probability of Backtest Overfitting", *J. Computational
Finance* 20(4) (CSCV context). Gneiting & Raftery (2007), *JASA* 102(477)
(proper scores).
