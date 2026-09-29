# 14 — Volatility & distributional model benchmarks (SOTA lane)

**Status:** research mapping, 2026-09-28. Sources: web survey (2026-09-28) +
in-tree inspection of `scripts/*_col.py`, `scripts/sota_eval_kronos.py`,
`scripts/benchmark_garch.py`, `src/quant_fund/models/`,
`src/quant_fund/research/vol_bench.py`, `src/quant_fund/metrics/`.
Honesty contract: **proper scores only** — QLIKE, MSE(−log/HMSE variants),
pinball, CRPS, PIT, log-score, HMM likelihood. No Sharpe/Sortino/Calmar/P&L.
SYNTHETIC results (e.g. `vol_bench` shards) are correctness evidence, never
market evidence; receipts seal every claim.

---

## 1. Model-family summaries (literature SOTA)

### 1.1 GARCH family (close-to-close conditional variance)

- **GARCH(1,1)** — Bollerslev (1986). Still the benchmark to beat: Hansen &
  Lunde (2005, *JAE* 20(7):873–889) compared 330 ARCH-type models on DM/$ and
  IBM with SPA/Reality-Check; nothing beat GARCH(1,1) on FX, and only
  leverage models beat it on equity returns. Repo: `dip_garch_t` (arena),
  `models/volatility.py::GARCHVol`, `research/garch_benchmark.py`.
- **EGARCH** — Nelson (1991). Log-link, asymmetric leverage via the gamma
  term. Repo: `dip_egarch` (o=0 fallback spec — the leverage channel never
  engaged), `dip_egarch_l` (true o=1; negative gamma on 86% of daily origins,
  leverage confirmed), `models/egarch.py`.
- **GJR-GARCH** — Glosten, Jagannathan & Runkle (1993). Quadratic leverage;
  the `dip_fhs` vol path and `garch_benchmark.py` candidate `gjr_t`. Repo:
  `models/egarch.py` (GJR QMLE), `arch` fallback.
- **APARCH / FIGARCH** — Ding, Granger & Engle (1993); Baillie, Bollerslev &
  Mikkelsen (1996). Power-asymmetry and fractional persistence. Repo:
  `models/garch_ext.py` (QMLE both), exposed in `GARCHVol` (Wave 106).
- **GAS / score-driven** — Creal, Koopman & Lucas (2013, *JAE* 28(5):777–795);
  Harvey & Chakravarty (2008) **Beta-t-(E)GARCH** is the same idea for vol.
  The scaled score of the observation density drives the parameter, so with a
  Student-t density large returns *damp* the variance update (outlier
  robustness that GARCH lacks). Blasques, Koopman & Lucas (2015,
  *Biometrika*) show score-driven updates are information-theoretically
  optimal in a KL sense; Koopman, Lucas & Scharth (2016, *REStat*) give the
  parameters-vs-observations-driven comparison. VaR evidence: Ardia et al.
  (arXiv:1611.06010, GAS R package) find GAS-t beats GARCH-t at 1%/5% alpha
  by average QLIKE. Repo: `models/gas.py` (Gaussian + Student-t, `inv_sqrt`
  and CKL `unit` scalings) — **symmetric only; no beta-t-EGARCH leverage**.
- **MSGARCH (Markov-switching GARCH)** — Hamilton (1989); Hamilton & Susmel
  (1994, RC-SVGARCH); Gray (1996) / Klaassen (2002) path-collapse
  approximations; Haas, Mittnik & Paolella (2004) regime-independent
  likelihood; Marcucci (2005, *JFE*): MRS-GARCH beats all standard GARCH at
  short horizons (not long); Ardia, Bluteau, Boudt & Catania (2018, *IJF*
  34(4):733–747): large-scale study — MSGARCH gives better VaR/ES/left-tail
  forecasts than single-regime, and **accounting for parameter uncertainty
  (Bayesian/MCMC) improves tails independently of switching**. MSGARCH R
  package: Ardia et al. (2019, *JSS* 91(4)). Repo: **absent** —
  `models/regime_switch.py` is a Markov-switching *mean* regression;
  `dip_regime` is an HMM-mixed *distribution* head, not switching GARCH
  parameters.
- **GARCH-MIDAS** — Engle, Ghysels & Sohn (2013): unit-mean short-run GARCH
  times a beta-weighted MIDAS long-run component on macro/RV blocks. Repo:
  `models/garch_midas.py` (full EG-S QMLE); arena `dip_mid` is only a
  stylized coarse-scale *tilt* of EWMA — it ranked 17th (honest negative).
- **Component / long-memory alternatives** — Engle & Lee (1999) component
  GARCH; ARFIMA-RV. Kilic (FEDS 2025-61, rev. 2026-09): **ARFIMA leads at
  monthly horizons**, MS-HAR at short ones, on S&P 500 + 40 equities. Repo:
  `models/long_memory.py` (GPH etc., diagnostics), no ARFIMA-RV forecaster.

### 1.2 Realized-volatility (RV) models & estimators

- **HAR-RV** — Corsi (2009, *J. Fin. Econometrics* 7(3):174–196), cascade
  over daily/weekly/monthly RV inspired by Müller et al. (1997); the de-facto
  RV baseline. Log-HAR (regression on log RV) is preferred given
  log-normality (Corsi 2009). Repo: `models/har.py` (OLS + Newey-West),
  `HARVol`, arena `dip_har` (sqrt-RV form on **Parkinson range variance**,
  r² fallback), `vol_bench` registry entry `har` (DM reference).
- **HARQ** — Bollerslev, Patton & Quaedvlieg (2016, *JoE* 192(1):64–78):
  daily coefficient modulated by realized quarticity → time-varying
  measurement-error attenuation; significant QLIKE gains over HAR. Repo:
  `models/har.py::harq_fit/harq_forecast` — **implemented but not wired into
  any benchmark registry or arena column**.
- **HEAVY** — Shephard & Sheppard (2010, *JAE* 25(2):197–231): joint
  dynamics for squared returns and a realized measure, two MEM-style
  equations; direct-projection multi-step gains. Fat-tailed extension:
  Opschoor, Janus, Lucas & van Dijk (2018, *JBES* 36(4):643–657); GAS-HEAVY
  (score-driven, rescaled-t returns / rescaled-F RM). Repo: **absent**.
- **Realized GARCH** — Hansen, Huang & Shek (2012, *JAE* 27(6):877–906):
  GARCH variance + measurement equation linking realized measures; Hansen &
  Huang (2016, *JBES*) exponential-GARCH version. Repo:
  `models/realized_garch.py` (log-linear HHK on **daily Parkinson**, disclosed
  as not HF-RV), `vol_bench` registry `realized_garch`.
- **RV estimators (proxy quality matters — Patton 2011 robustness is w.r.t.
  a conditionally unbiased proxy)**:
  - Realized variance / kernel / TSRV / pre-averaging: Andersen, Bollerslev,
    Diebold & Labys (2003); Barndorff-Nielsen, Hansen, Lunde & Shephard
    (2008); Zhang, Mykland & Aït-Sahalia (2005); Jacod et al. (2009).
    Repo: `models/realized.py` (all four).
  - **Bipower/tripower** — Barndorff-Nielsen & Shephard (2004, *JFE* 2(1):
    1–37; 2006 *Econometrica*): jump-robust IV; RV−BV identifies jump
    variation. Podolskij & Vetter (2009) quarticity variants. Repo:
    `models/realized.py` (BV, tripower quarticity, BNS jump test,
    Lee–Mykland 2008, semivariance).
  - **MinRV / MedRV ("medianshift")** — Andersen, Dobrev & Schaumburg (2012,
    *JoE* 169(1):40–63, NBER w15533): nearest-neighbour truncation,
    `(π/(6−4√3+π))·Σ min²(|r|)` / `(π²/(π²+2−2π))·Σ med³(|r_{j−1}|,|r_j|,
    |r_{j+1}|)^{2/3}`; jump-robust **with a feasible asymptotic theory under
    the jump alternative** (BV lacks this) and robust to zero returns that
    bias multipower estimators. Recent evidence (Computational Economics
    2024, 10.1007/s10614-024-10694-2): **MedRV is the best target for deep
    RV models**, 5-min RV for GARCH-type. Repo: **absent** from
    `models/realized.py`.
  - **Range estimators from daily OHLC** (no intraday tape in this repo —
    standing data gap): Parkinson (1980, ~5× efficiency), Garman–Klass (1980,
    ~7.4×, zero-drift/no-gap assumption), Rogers–Satchell (1991, drift-robust),
    **Yang–Zhang (2000, *J. Business* 73(3))** — overnight + open-to-close +
    RS components, ~14× efficiency, handles gaps and drift; Alizadeh, Brandt
    & Garcia (2002, *JF*) log-SV on ranges; Corwin & Schultz (2012)
    high-low spread estimator. Repo: `northset/estimators.py` implements all
    as *diagnostics* with QLIKE-vs-close-to-close and DM comparisons;
    `dip_har`/`dip_lgbm_qv`/`dip_volm`/`realized_garch` consume Parkinson;
    **YZ/GK/RS are not used as forecast-model RV targets**.

### 1.3 Stochastic volatility (SV)

- **SV / log-SV** — Taylor (1986); Harvey & Shephard (1996) log-linear
  QMLE; Kim, Nelson & Siegel (1998); Yu (2005) MCMC. Repo:
  `models/stoch_vol.py` (Harvey–Shephard Kalman QMLE).
- **SV with leverage & mixture-of-normals SV** — Jacquier, Johannes &
  Polson (2007, *JASA*) MCMC SV with fat tails/leverage; Contessi &
  Ferrari; the arXiv:1605.00230 horserace (weekly returns) finds **SV models
  generally outperform GARCH/GAS/MS-GARCH on density scores**, SV-leverage
  strongest in turmoil. Repo: **no SV-leverage, no mixture-SV**;
  `stoch_vol.py` never enters the arena as a density challenger.
- **Rough vol** — Gatheral, Jaisson & Rosenbaum (2018, *Math. Finance*):
  log-vol ~ fBM with H≈0.1; RFSV ≈ exp(fOU). Survey: Bank of Japan IMES
  24-E-06 (2024). Mixed forecasting record at daily horizons — gains are
  mostly intraday/short-lag; non-Markovian, costly. Repo:
  `models/rough_vol.py` (Hurst diagnostic, fOU simulation, variance curve)
  — **simulation/diagnostics only, no RFSV forecaster**; `vol_bench` has a
  synthetic rough shard generator.

### 1.4 Regime-switching / HMM for volatility

- Hamilton (1989) regime framework; Hamilton & Susmel (1994) RC-SWARCH;
  Gray (1996)/Klaassen (2002) collapsing; Haas et al. (2004); Caporale &
  Zekokh (2019) MS-GARCH mixture comparison. Evidence: MS wins at short
  horizons and for tails (Marcucci 2005; Ardia et al. 2018); MS-HAR wins at
  1-day (Kilic FEDS 2025-61). HMM filtered likelihood is a **proper score**
  (log-score) — allowed by the honesty contract.
- Repo: `models/regime.py` (`GaussianHMMRegime`, hmmlearn, AIC/BIC),
  `models/regime_dist.py` (`dip_regime`: 2-state Gaussian HMM → per-state
  empirical CDFs mixed with one-step filtered probabilities; MCS-included in
  the daily mega-arena), `models/regime_switch.py` (MS mean regression).
  **Missing:** switching *GARCH parameters* (MSGARCH), parameter uncertainty
  (Bayesian averaging over regimes à la Ardia et al. 2018).

### 1.5 Density forecasting (KDE / mixtures / EVT / skew)

- **Scoring canon** — Gneiting & Raftery (2007, *JASA* 102:359–378): CRPS is
  the standard proper score for real-valued predictive distributions; log
  score is local. Weighted/threshold-weighted proper scores for tail
  emphasis: Gneiting & Ranjan (2011); Holzmann & Klar (2017). Repo:
  `metrics/scoring.py` (CRPS Gaussian/Student-t/mixture/empirical/
  from-quantiles, pinball), `metrics/calibration2.py` (PIT/pinball).
- **KDE** — bandwidth selection is the whole game; vol-adaptive bandwidths
  beat fixed rules-of-thumb in returns (repo's `dip_kde`: Silverman ×
  clipped EWMA vol ratio; arena verdict: Gaussian tails too thin vs ECDF —
  honest negative). Mixture-of-normals SV and finite mixtures are the
  parametric alternative (Paolella 2011 SSRN 1956462; Opschoor et al. 2018).
  Repo: `models/mixture.py` (Gaussian/Student-t EM, BIC/ICL), `dip_gmm_k`
  (BIC-selected K∈{1,2,3}, closed-form mixture CRPS).
- **EVT tails** — Pickands (1975)/Balkema & de Haan (1974) POT-GPD;
  McNeil & Frey (2000) GPD on GARCH residuals. Repo: `dip_evt` (two-sided
  POT-GPD MoM fits over empirical body, EWMA vol rescale) — **daily MCS #2**,
  between `dip_fhs` and `dip_egarch_l`.
- **Skew/leverage densities** — Hansen (1994) skewed Student-t (repo
  `dip_skt`, `models/skew_t.py`); Fernandez & Steel (1998); skew-t FHS
  (`models/fhs.py`). `dip_fhs` (GJR + FHS residuals, Barone-Adesi et al.
  1999) is the **reigning arena #1 at both frequencies**.
- **Conformal / calibration** — Romano, Patterson & Candès (2019) CQR;
  Gibbs & Candès (2021) ACI; Zaffran et al. (2022) AgACI/FACI. Repo:
  `dip_conf_t`, `dip_aci`, `models/agaci.py`, `models/conformal_dist.py`,
  `enbpi`. RV-interval conformal (Cañete et al. ICML-WS 2023 market-implied
  conformal vol intervals; PCP-RV pooled conformal calibration) — **not yet
  applied to the vol lane**.
- **Combinations** — linear/quantile averaging, exponentiated-gradient
  online stacks (Raftery et al. 2005; Bassetti et al. 2018 beta-mixture
  density pooling). Repo: `dip_stack`, `dip_stack2` (EG weights on pinball,
  Vincentized).

### 1.6 ML volatility models

- **LightGBM/XGBoost quantile** — Ke et al. (2017); QLIKE-objective boosting
  beats GARCH-family on 4/7 assets (DQuant study, 2025); LGBM probabilistic
  BTC-RV with 69 predictors (arXiv:2511.20105). Repo: `dip_lgbm_q`
  (8 causal features), `dip_lgbm_qv` (+5 microstructure: park_vol,
  rng_ratio, vol_surp, amihud, gap), `models/lgbm_q2.py`, `TreeVol`.
- **QRF / NGBoost / DL** — Meinshausen (2006, *JMLR*); Duan et al. (2020,
  ICML); QRF beats LGBM-quantile in lower tails on crypto panels (Solana
  study, 2025). Transformer/DL RV evidence (MDPI JRFM 18(12):685, 2025;
  Modern Finance 2025): transformers and **GARCH×NN hybrids (TDNN/GRU-GJR)**
  lead in volatile periods; Kilic (FEDS 2025-61): ML gives **no systematic
  advantage** over the broader econometric set — horizon-dependent, regime
  models win short. Repo: `models/qrf.py`, `models/quantile_forest.py`,
  `models/ngboost_lite.py`, `models/nbeats.py`/`dlinear.py` (general TS, not
  vol-headed). Kronos/Chronos/TimesFM foundation models are the arena's
  published targets — **all four excluded from MCS at both frequencies**.

### 1.7 Which score? (QLIKE vs MSE — the lane's evaluation canon)

- **Patton (2011, *JoE* 160(1):246–256)**: with a noisy but conditionally
  unbiased proxy, most common losses (MAE, MAPE, MSE-SD, MSE-log, MSE-PROP,
  HMSE…) **re-rank forecasts spuriously**; only MSE and QLIKE are robust, and
  QLIKE has greater test power (confirmed by Patton's simulations and the
  multivariate extension, Laurent, Rombouts & Violante 2013). QLIKE penalizes
  *under*-prediction harder — the right asymmetry for risk.
  **Verdict: QLIKE primary for variance forecasts; MSE as the robust
  companion; never headline MAE/MAPE/R² alone.**
- Repo already complies: `metrics/vol_eval.py` implements the Patton-robust
  QLIKE `x/f − ln(x/f) − 1`, plus mse/mse_log/hmse/mae and
  Mincer–Zarnowitz; `vol_bench.py` scores **QLIKE + MSE with DM t-stats
  (`dm_qlike_*`)**; `garch_benchmark.py` selects on validation QLIKE;
  `northset/estimators.py` ranks range estimators by QLIKE.
- **Return densities:** CRPS + pinball (proper, `metrics/scoring.py`), PIT
  calibration; **HMM log-likelihood** permitted as a proper score.
- **Inference:** Diebold & Mariano (1995) with Harvey, Leybourne & Newbold
  (1997) small-sample correction; multiplicity via White (2000) RC, Hansen
  (2005) SPA, Romano & Wolf (2005) StepM, Hansen, Lunde & Nason (2011) MCS —
  all in `metrics/inference.py` + `metrics/snooping.py`, already wired into
  the arena (`aligned_inference_losses`, stationary block bootstrap,
  block-length sensitivity).

---

## 2. Coverage matrix

Legend: ✅ in tree & benchmarked · 🟡 in tree, not benchmarked/wired ·
❌ absent.

| Capability | Canonical reference | Repo status | Location |
|---|---|---|---|
| GARCH(1,1)-t | Bollerslev 1986 | ✅ arena leader tier | `dip_garch_t`, `garch_benchmark.py` |
| EGARCH (sym. + true leverage) | Nelson 1991 | ✅ | `dip_egarch`, `dip_egarch_l`, `models/egarch.py` |
| GJR-GARCH | Glosten et al. 1993 | ✅ (inside `dip_fhs`, frozen bench) | `_challenger_col`, `garch_benchmark.py` |
| APARCH / FIGARCH | Ding et al. 1993; BBM 1996 | 🟡 model only, never arena-scored | `models/garch_ext.py` |
| GAS / score-driven (symmetric) | Creal et al. 2013 | 🟡 model only, never arena-scored | `models/gas.py` |
| **Beta-t-(E)GARCH / GAS leverage** | Harvey & Chakravarty 2008; Harvey 2013 | ❌ | — |
| **MSGARCH (switching GARCH params)** | Haas et al. 2004; Klaassen 2002; Ardia et al. 2018 | ❌ (MS *mean* only; HMM *distribution* only) | `regime_switch.py`, `dip_regime` ≠ |
| GARCH-MIDAS | Engle, Ghysels & Sohn 2013 | 🟡 full model unbenched; arena `dip_mid` is a tilt proxy (17th, honest negative) | `models/garch_midas.py` |
| HAR-RV (Parkinson proxy) | Corsi 2009 | ✅ arena (`dip_har`, FHS quantiles) + `vol_bench` registry | `_har_col.py`, `models/har.py` |
| **HARQ** | Bollerslev, Patton & Quaedvlieg 2016 | 🟡 implemented, **zero benchmark rows** | `models/har.py::harq_*` |
| Log-HAR | Corsi 2009 §4 | ❌ (sqrt-RV form only) | `_har_col.py` |
| **HEAVY / fat-tailed HEAVY** | Shephard & Sheppard 2010; Opschoor et al. 2018 | ❌ | — |
| Realized GARCH | Hansen, Huang & Shek 2012 | ✅ (daily Parkinson, disclosed proxy limit) | `models/realized_garch.py`, `vol_bench` |
| RV: bipower/tripower, BNS & Lee–Mykland jump tests, TSRV, realized kernel, pre-averaged | BNS 2004/2006/2008; ZMA 2005; Jacod 2009 | ✅ | `models/realized.py` |
| **MinRV / MedRV (medianshift)** | Andersen, Dobrev & Schaumburg 2012 | ❌ | — |
| Range estimators (Parkinson/GK/RS/YZ/Corwin–Schultz) | Parkinson 1980; GK 1980; RS 1991; YZ 2000 | 🟡 diagnostics + DM/QLIKE comparisons only; YZ/GK never a model's RV target | `northset/estimators.py` |
| SV (Kalman QMLE log-linear) | Harvey & Shephard 1996 | 🟡 model only, never arena-scored | `models/stoch_vol.py` |
| **SV-leverage / mixture SV** | Jacquier, Johannes & Polson 2007 | ❌ | — |
| Rough vol (Hurst, fOU/RFSV) | Gatheral, Jaisson & Rosenbaum 2018 | 🟡 simulation + diagnostics; no forecaster | `models/rough_vol.py` |
| HMM regime distribution (2-state) | Hamilton 1989 lineage | ✅ arena, MCS-included | `models/regime_dist.py`, `dip_regime` |
| HMM likelihood as score | — | ✅ allowed (proper log-score); `aic_bic` present | `models/regime.py` |
| KDE density (vol-adaptive bandwidth) | Silverman; Gneiting–Raftery scoring | ✅ arena (honest negative: tails too thin) | `_kde_col.py` |
| GMM density (BIC-selected) | McLachlan & Peel | ✅ arena | `_gmm_col.py`, `models/mixture.py` |
| EVT POT-GPD tails | Pickands 1975; Balkema–de Haan 1974 | ✅ arena daily #2 | `_evt_col.py` |
| Skew-t (Hansen 1994), skew-t FHS | Hansen 1994; Barone-Adesi et al. 1999 | ✅ arena #1 (`dip_fhs`), `dip_skt` | `models/fhs.py`, `skew_t.py` |
| CAViaR / QAR | Engle & Manganelli 2004; Koenker & Xiao 2006 | ✅ | `models/caviar.py`, `qar.py`, `dip_qar` |
| Conformal: CQR, ACI, AgACI, CV+ | Romano et al. 2019; Gibbs & Candès 2021; Zaffran et al. 2022 | ✅ | `dip_conf_t`, `_aci_col.py`, `models/agaci.py` |
| **Conformal RV intervals** | Cañete et al. 2023; PCP-RV | ❌ (ACI wraps return quantiles only) | — |
| LGBM quantile (return features / +microstructure) | Ke et al. 2017 | ✅ arena | `_lgbmqv_col.py`, `models/lgbm_q2.py` |
| QRF / NGBoost | Meinshausen 2006; Duan et al. 2020 | 🟡 models exist; no vol-lane column | `models/qrf.py`, `ngboost_lite.py` |
| Online stacks (EG on pinball) | Raftery et al. 2005 | ✅ arena | `dip_stack`, `dip_stack2` |
| Volume/vol conditioning | MDH/GARCH-X literature | ✅ arena (honest negative vs leaders) | `_volm_col.py` |
| QLIKE-robust evaluation | **Patton 2011** | ✅ first-class | `metrics/vol_eval.py`, `vol_bench.py` |
| DM / SPA / StepM / MCS | DM 1995; HLN 1997; White 2000; Hansen 2005; HLN 2011 | ✅ | `metrics/inference.py`, `metrics/snooping.py` |
| Synthetic vol shards (GJR, rough-fOU, break) | — | ✅ (labeled SYNTHETIC) | `research/vol_bench.py` |
| Intraday tape → true HF-RV | — | ❌ standing data gap (disclosed) | `SOTA_CANON_ROADMAP_2026_09.md` §2.6 |

---

## 3. Adoption plan

Priority = (literature effect size) × (distance from existing in-tree code) ×
(honesty-contract fit). All new columns must follow the established arena
contract: deterministic, causal, shard-bound via `bars_sha256` + config
fields, honest NaN on failure with counters, spliced through
`splice_challenger_column.py`, receipt-sealed.

### Tier 1 — wire what already exists (days, no new math)

1. **HARQ column + `vol_bench` entry.** `models/har.py::harq_fit` is
   implemented and cited but has zero scored rows. Build `_harq_col.py` as a
   strict delta from `_har_col.py` (RQ term from `realized_quarticity`,
   same Parkinson proxy), add `harq` to `VOL_MODEL_REGISTRY`. Expected gain
   per BPQ (2016): measurable QLIKE improvement over HAR, largest when
   measurement error varies (crypto 4h bars qualify).
2. **APARCH + FIGARCH + symmetric GAS arena columns.** `garch_ext.py` and
   `gas.py` are tested models the arena has never scored; GAS-t is the
   outlier-robust GARCH alternative with published QLIKE wins (Ardia et al.
   arXiv:1611.06010). Cheap: reuse the `_arch_fit`-style harness and the
   Student-t CRPS path.
3. **SV (Kalman QMLE) density column.** `stoch_vol.py` → h=1 predictive via
   lognormal state + Gaussian/t innovations; scored on CRPS + pinball +
   **HMM/log-likelihood** (all permitted). Closes the "SV vs GARCH"
   comparison the literature says matters (arXiv:1605.00230: SV wins on
   density at weekly horizons).

### Tier 2 — new models with strong published evidence (1–2 weeks each)

4. **MSGARCH (Haas et al. 2004 parameterization).** Two-regime GARCH(1,1),
   regime-independent likelihood (avoids Gray/Klaassen collapsing bias),
   Gaussian and t innovations, deterministic seeded optimizer; predictive
   density = regime-probability-weighted mixture (filtered probabilities —
   causal). Evidence: Marcucci (2005) short-horizon wins; Ardia et al. (2018)
   tail-risk wins; Kilic (FEDS 2025-61) MS-HAR best at h=1. Optional v2:
   parameter-uncertainty averaging (posterior draws) per Ardia's key
   finding. This is the largest true gap: the honesty contract explicitly
   allows HMM likelihood as a score, and the repo already ships
   `hmmlearn` — but nothing switches GARCH parameters.
5. **Beta-t-EGARCH (Harvey & Chakravarty 2008; Harvey 2013).** Extend
   `models/gas.py` with the exp-link, t-score recursion
   `b_{t+1} = ω + φb_t + κ·s_t` where the t-score is already implemented,
   plus the leverage extension (asymmetric score term, Harvey & Sucarrat
   2014). Gives the outlier-robust + leverage combination EGARCH can't do
   without exploding (see `dip_egarch_l`'s SIG_CAP war stories).
6. **MinRV / MedRV in `models/realized.py`** + a **HAR-MedRV** column.
   Two ~20-line estimators (Andersen, Dobrev & Schaumburg 2012 formulas);
   they fix BV's zero-return bias and finite-sample jump distortion. On the
   4h crypto grid with stale-print bars this is the most likely proxy
   upgrade to change rankings (Computational Economics 2024: MedRV target
   changes which model class wins). Also upgrade `dip_har`'s proxy choice as
   a disclosed variant column, never a silent swap.
7. **HEAVY (Shephard & Sheppard 2010)** on the Parkinson proxy (disclosed
   range-as-RM limitation, same posture as `realized_garch.py`): joint
   (squared return, realized measure) MEM pair, direct-projection multi-step
   forecasts; natural `vol_bench` registry entry with h∈{1,5}.

### Tier 3 — exploratory, gated on Tier 1–2 results

8. **Yang–Zhang RV target variant.** `northset/estimators.py::yang_zhang_variance`
   exists; a `dip_har_yz` column swaps the Parkinson proxy for YZ (~14×
   close-to-close efficiency, gap- and drift-robust — crypto trades 24/7 so
   "overnight" gaps are funding-window jumps). Cheap delta, real efficiency
   argument.
9. **Conformal RV intervals (Cañete et al. 2023; PCP-RV pooling).** Wrap the
   HAR/HEAVY point forecasts with `agaci.py`-style online calibration to get
   coverage-guaranteed vol intervals; scored with interval QLIKE / pinball,
   reported with empirical coverage. Cross-asset pooled margin (PCP-RV)
   matches the existing cross-asset arena panel.
10. **Log-HAR** (Corsi 2009's preferred functional form) as a `_har_col.py`
    variant, and **RFSV forecaster** from `rough_vol.py` pieces — only if the
    Hurst diagnostic on the actual bars shows H<0.5 stability (otherwise
    documented as a negative result; daily-horizon gains are contested).

### 4. Benchmark protocol (frozen before scoring)

Extends `research/vol_bench.py` + the `EVAL_REPORT_SOTA.md` arena contract;
every run receipt-sealed (`verify-research`), `live_pnl_claim: false`.

**Cells.** (a) **Return-density arena** (existing): daily + 4h Binance
majors, walk-forward origins, causal trailing-window fits, target = next-bar
close-to-close return. (b) **Variance lane** (`vol_bench` protocol, extended
to real bars alongside the SYNTHETIC shards): target = h-step cumulative
realized variance proxy, h ∈ {1, 5}, nonoverlapping origins, refit only on
pre-origin data; proxy = Parkinson by default with **MedRV/YZ sensitivity
columns** (proxy choice is a disclosed protocol field, never retuned
post-hoc).

**Scores (proper only).**
- Variance lane: **QLIKE primary** (Patton 2011 robust + higher power),
  MSE companion, Mincer–Zarnowitz efficiency regression as diagnostic.
- Density lane: **CRPS primary** (closed-form where available;
  512-equiprobable `crps_empirical` for quantile-grid/ensemble models — the
  arena's existing disclosed conventions), pinball at τ∈{0.05, 0.5, 0.95}
  plus the LGBM_TAUS grid, PIT uniformity, and **HMM log-likelihood** for
  regime models (allowed: proper log-score).
- Coverage diagnostics: Kupiec/Christoffersen on the 5%/95% pinball targets
  (`metrics/var_backtest.py`).

**Inference.** Per-aligned-timestamp equal-weight cross-asset mean losses on
the balanced chronological panel (existing `aligned_inference_losses`):
pairwise **Diebold–Mariano with HLN small-sample t** and Newey–West/
overlap-aware HAC lags; **Hansen SPA** (lower/consistent/upper) per
benchmark; **MCS @ α=0.10** via stationary block bootstrap (n_boot=2000);
paired-bootstrap effect intervals; block-length sensitivity sweep. Any new
column enters through `splice_challenger_column.py` (bars_sha256 +
protocol-field bound) so row alignment is verified by construction; failure
modes emit honest NaN with counters in meta — never fabricated rows.

**Multiplicity discipline.** New challengers are *additions* to an existing
32-model arena; per-family selection happens on validation windows only
(`garch_benchmark.py` convention: select on validation QLIKE, score test
once), and every headline claim names the full tested set so SPA/MCS account
for the search. SYNTHETIC vol-shard results are always labeled correctness
tests.

**Acceptance bar (pre-registered).** A new baseline is adopted into the
standing arena iff it (i) lands in the MCS @0.10 on ≥1 real cell, or
(ii) DM-significantly improves QLIKE vs `har` in the variance lane at h=1
and h=5, or (iii) is retained as a *documented honest negative* with its
failure counters (the `dip_kde`/`dip_mid`/`dip_volm` precedent). Receipts in
`receipts/`, verifier acceptance under `verifier/vN/`.

---

## 5. References

- Alizadeh, Brandt & Garcia (2002). Lognormality of prices and volatility
  dynamics: theory and applications to range estimation. *JF* 57(6).
- Andersen, Bollerslev, Diebold & Labys (2003). Modeling and forecasting
  realized volatility. *Econometrica* 71(2).
- Andersen, Dobrev & Schaumburg (2012). Jump-robust volatility estimation
  using nearest neighbor truncation. *JoE* 169(1):40–63. (NBER w15533)
- Ardia, Bluteau, Boudt & Catania (2018). Forecasting risk with
  Markov-switching GARCH models: a large-scale performance study. *IJF*
  34(4):733–747. — Ardia et al. (2019). MSGARCH R package. *JSS* 91(4).
- Ardia, Bluteau, Boudt & Catania (2016). Value-at-Risk prediction in R with
  the GAS package. arXiv:1611.06010.
- Baillie, Bollerslev & Mikkelsen (1996). Fractionally integrated GARCH.
  *JoE* 74(1).
- Balkema & de Haan (1974); Pickands (1975). POT/GPD tail theory.
- Barndorff-Nielsen & Shephard (2004). Power and bipower variation with
  stochastic volatility and jumps. *JFE* 2(1):1–37. — BNS, Hansen, Lunde &
  Shephard (2008) realized kernels.
- Barone-Adesi, Giannopoulos & Vosper (1999). VaR without correlations for
  portfolios of derivative securities. *JF* (FHS).
- Bollerslev (1986). GARCH. *JoE* 31(3). — Bollerslev, Patton & Quaedvlieg
  (2016). Exploiting the errors: HARQ. *JoE* 192(1):64–78.
- Cañete et al. (2023). Market implied conformal volatility intervals.
  *PMLR* v204. — PCP-RV: Forecasting realized volatility intervals with
  pooled conformal calibration (2025/2026).
- Caporale & Zekokh (2019). Modelling volatility with Markov-switching
  GARCH: MS vs mixture-of-GARCH. *JoE* / Liverpool WP.
- Corsi (2009). A simple approximate long-memory model of realized
  volatility. *J. Fin. Econometrics* 7(3):174–196.
- Creal, Koopman & Lucas (2013). Generalized autoregressive score models
  with applications. *JAE* 28(5):777–795. — Blasques, Koopman & Lucas (2015)
  *Biometrika* 102(2); Koopman, Lucas & Scharth (2016) *REStat* 98(1).
- Diebold & Mariano (1995). Comparing predictive accuracy. *JBES* 13(3). —
  Harvey, Leybourne & Newbold (1997) *JBES* 15(2).
- Ding, Granger & Engle (1993). APARCH. *Financial Markets & Portfolio Mgmt*.
- Duan, Anand, Ding, Basu, Ng & Schuler (2020). NGBoost. *ICML* PMLR 119.
- Engle (1982). ARCH. *Econometrica* 50(4). — Engle & Manganelli (2004).
  CAViaR. *JBES* 22(4). — Engle, Ghysels & Sohn (2013). GARCH-MIDAS. *JoE*.
- Gatheral, Jaisson & Rosenbaum (2018). Volatility is rough. *Math. Finance*
  28(1). — BOJ IMES 24-E-06 (2024) survey.
- Gibbs & Candès (2021). Adaptive conformal inference under distribution
  shift. *NeurIPS* 34. — Zaffran et al. (2022). AgACI. *ICML*.
- Glosten, Jagannathan & Runkle (1993). *JF* 48(5).
- Gneiting & Raftery (2007). Strictly proper scoring rules. *JASA*
  102(477):359–378. — Gneiting & Ranjan (2011); Holzmann & Klar (2017).
- Gray (1996). *JFE* 42; Klaassen (2002). *Empirical Economics* 27;
  Haas, Mittnik & Paolella (2004). *JFE* 2.
- Hamilton (1989). *Econometrica* 57(2). — Hamilton & Susmel (1994). *JoE*.
- Hansen (1994). Autoregressive conditional density estimation. *JAE* —
  Hansen (2005). SPA test. *JBES* — Hansen, Lunde & Nason (2011). MCS.
  *Econometrica* 79(2). — Hansen, Huang & Shek (2012). Realized GARCH. *JAE*
  27(6):877–906. — Hansen & Huang (2016). *JBES* 34(2). — Hansen & Lunde
  (2005). *JAE* 20(7):873–889.
- Harvey & Shephard (1996). Estimation of an asset pricing model with
  stochastic volatility. — Kim, Nelson & Siegel (1998). — Yu (2005).
- Harvey & Chakravarty (2008). Beta-t-(E)GARCH. — Harvey (2013). *Dynamic
  Models for Volatility and Heavy Tails* (CUP). — Harvey & Sucarrat (2014).
  *JoE* 178.
- Jacod, Li, Mykland, Podolskij & Vetter (2009). Pre-averaging. — Zhang,
  Mykland & Aït-Sahalia (2005). TSRV. *JASA*.
- Ke et al. (2017). LightGBM. *NeurIPS*.
- Kilic (2025, rev. 2026). Linear and nonlinear econometric models versus
  ML: realized-volatility prediction. FEDS 2025-61.
- Laurent, Rombouts & Violante (2013). Loss functions for multivariate
  volatility models. *JoE* 173(1).
- Lee & Mykland (2008). Jumps in financial markets. *RFS*.
- Marcucci (2005). Forecasting stock market volatility with Markov-switching
  GARCH. *JFE* 13(4).
- McNeil & Frey (2000). Estimation of tail-related risk measures. *JEFIMC*.
- Meinshausen (2006). Quantile regression forests. *JMLR* 7:983–999.
- Müller, Dacorogna et al. (1997). Fractionally integrated intraday return
  dynamics. — Nelson (1991). EGARCH. *Econometrica* 59(2).
- Opschoor, Janus, Lucas & van Dijk (2018). New HEAVY models for fat-tailed
  realized covariances and returns. *JBES* 36(4):643–657.
- Parkinson (1980). *J. Business* 53. — Garman & Klass (1980). *J. Business*
  53. — Rogers & Satchell (1991). *JAE*. — Yang & Zhang (2000). *J. Business*
  73(3). — Corwin & Schultz (2012). *JF*.
- Patton (2011). Volatility forecast comparison using imperfect volatility
  proxies. *JoE* 160(1):246–256.
- Podolskij & Vetter (2009). Bipower/tripower quarticity.
- Romano, Patterson & Candès (2019). Conformalized quantile regression.
  *NeurIPS*. — Romano & Wolf (2005). StepM. *Econometrica*.
- Shephard & Sheppard (2010). HEAVY. *JAE* 25(2):197–231.
- Taylor (1986). *Modelling Financial Time Series*.
- White (2000). A reality check for data snooping. *Econometrica* 68(5).

**In-repo:** `docs/EVAL_REPORT_SOTA.md` (arena results, mega-arena
leaderboards), `docs/GARCH_BENCHMARK.md` (frozen v1 protocol + real-data
extension), `docs/SOTA_CANON_ROADMAP_2026_09.md` (inference-stack status,
standing data gaps), `src/quant_fund/research/vol_bench.py` (QLIKE+MSE+DM
walk-forward, SYNTHETIC-labeled), `src/quant_fund/metrics/vol_eval.py`
(Patton losses), `src/quant_fund/metrics/snooping.py` (RC/SPA/StepM/MCS).
