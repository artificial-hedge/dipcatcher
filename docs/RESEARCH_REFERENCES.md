# Research references

Methodological anchors. Implementations may differ; deviations are in [MATH_SPEC.md](MATH_SPEC.md).

## Cross-sectional asset pricing / ML

- Fama, MacBeth (1973). Risk, Return, and Equilibrium: Empirical Tests. *Journal of Political Economy*. Date-level CS slopes averaged; catalog `fm`. Ridge per-date slopes: `fm_ridge`.
- Jegadeesh (1990). Evidence of Predictable Behavior of Security Returns. *Journal of Finance*. Short-term reversal; `classic` sign on `cs_z_reversal_1`.
- Jegadeesh, Titman (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*. Skip-week 20-session momentum; `cs_z_mom_skip_5_20`.
- Blitz, Huij, Martens (2011). Residual Momentum. *Journal of Empirical Finance*. CAPM residual 20-session momentum; `cs_z_idio_mom_20`.
- Bali, Cakici, Whitelaw (2011). Maxing Out: Stocks as Lotteries. *Journal of Financial Economics*. MAX; `classic` short `cs_z_max_ret_20`.
- Ang, Hodrick, Xing, Zhang (2006). The Cross-Section of Volatility and Expected Returns. *Journal of Finance*. Idiosyncratic vol; `classic` short `cs_z_idio_vol_60`.
- Amihud (2002). Illiquidity and Stock Returns. *Journal of Financial Markets*. `classic` long `cs_z_amihud`.
- George, Hwang (2004). The 52-Week High and Momentum Investing. *Journal of Finance*. `classic` long `cs_z_high_52w_prox`.
- Gu, Kelly, Xiu (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*. NBER w25398. PCR `pcr`, PLS `pls`, Huber GBRT `gbrt`. Neural nets not implemented (ADR-007). Linear autoencoder ≡ IPCA.
- Kelly, Pruitt (2015). The Three-Pass Regression Filter: A New Approach to Forecasting Using Many Predictors. *Journal of Econometrics* 186(2). Automatic-proxy 3PRF Tables 1–2; catalog `tprf`. PLS is the no-intercept special case.
- Kelly, Malamud, Pedersen (2023). Principal Portfolios. *Journal of Finance* 78(1). Open access DOI [10.1111/jofi.13199](https://doi.org/10.1111/jofi.13199). NBER w27388. \(\Pi=E[R_{t+1}S_t']\); catalog `pp`.
- Kelly, Malamud, Zhou (2024). The Virtue of Complexity in Return Prediction. *Journal of Finance* 79(1). Open access DOI [10.1111/jofi.13298](https://doi.org/10.1111/jofi.13298). NBER w30217. RFF (Rahimi–Recht) + ridge \(\hat\beta(z)=(zI+T^{-1}S'S)^{-1}T^{-1}S'R\); catalog `rff`. Adaptations in MATH_SPEC.
- Kozak, Nagel, Santosh (2020). Shrinking the Cross-Section. *Journal of Financial Economics*. NBER w24070. SDF ridge \(b=(\Sigma+zI)^{-1}\mu\) (`sdf_ridge`) and HJ-distance elastic net eq. 28 (`sdf_en`).
- Kelly, Pruitt, Su (2019). Characteristics Are Covariances: A Unified Model of Risk and Return. *Journal of Financial Economics*. NBER w24540. IPCA ALS, \(\Gamma'\Gamma=I_K\); catalog `ipca`. Joint unrestricted \(F_{\mathrm{aug},t}=(1,f_t)'\): `ipca_alpha`.
- Rapach, Strauss, Zhou (2010). Out-of-Sample Equity Premium Prediction: Combination Forecasts and Links to the Real Economy. *Review of Financial Studies*. Equal-weight univariate OLS; catalog `combo`. Non-negative train date-IC weights: `combo_ic`.
- Zou (2006). The Adaptive Lasso and Its Oracle Properties. *Journal of the American Statistical Association*. Catalog `alasso`.
- Lettau, Pelger (2020). Factors That Fit the Time Series and Cross-Section of Stock Returns. *Review of Financial Studies* 33(5). NBER w24858. RP-PCA \(S=(1/T)X'X+\gamma\bar X\bar X'\); catalog `rp_pca`.
- Giglio, Xiu (2021). Asset Pricing with Omitted Factors. *Journal of Political Economy*. NBER w23527. Three-pass PCA / CS / TS estimator; catalog `gx3pass`.
- Freyberger, Neuhierl, Weber (2020). Dissecting Characteristics Nonparametrically. *Review of Financial Studies* 33(5). NBER w23227. Adaptive group LASSO on quadratic splines of rank-transformed characteristics; catalog `fnw`.
- Feng, Giglio, Xiu (2020). Taming the Factor Zoo: A Test of New Factors. *Journal of Finance*. Post-double-selection LASSO then OLS; catalog `ds_lasso`.
- Learning-to-rank (LambdaRank, XE-NDCG) as used in LightGBM.
- Factor-neutral portfolio construction (standard industry practice; Grinold–Kahn).

## Distributional forecasting

- Koenker, Bassett (1978). Regression Quantiles.
- Pedersen, Thomas Q. Predictable Return Distributions.
- Gneiting, Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*. CRPS, pinball.

## Volatility

- Bollerslev (1986). GARCH.
- RiskMetrics EWMA.
- Parkinson (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return.
- Hansen, Huang, Shek (2012). Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility. *Journal of Applied Econometrics*.
- Corsi (2009). HAR-RV.
- Patton (2011). Volatility forecast comparison / QLIKE.

## Covariance

- Ledoit, Wolf (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. Default `optimizer.covariance=ledoit_wolf` is trailing 2004 linear shrinkage plus the GARCH/RGARCH overlay (Wave 139) and stays Ledoit–Wolf when \(T\le N\), not silent sample and not one-step \(H_{t+1}\). Named `optimizer.covariance=sample` is trailing unbiased (`ddof=1`) sample covariance plus the overlay (Wave 132). Factor stays unwired.
- Ledoit, Wolf (2020). Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices. *Annals of Statistics*. Named `optimizer.covariance=ledoit_wolf_nonlinear` is trailing analytical 2020 spectral shrinkage plus the overlay (Wave 140), not 2004 linear shrinkage and not numerical QuEST (2017). Generic `ledoit_wolf_2017` / `quest` stay unknown.
- Chen, Wiesel, Eldar, Hero (2010). Shrinkage Algorithms for MMSE Covariance Estimation. *IEEE Transactions on Signal Processing*. Named `optimizer.covariance=oas` is trailing listwise OAS plus the GARCH/RGARCH overlay (Wave 131), not Ledoit–Wolf and not one-step \(H_{t+1}\).
- J.P. Morgan (1996). RiskMetrics Technical Document. Named `optimizer.covariance=ewma` is one-step \(H_{t+1}\) on the trailing complete window (Wave 130), not listwise-deleted in-sample last \(H_t\).
- Engle (2002). Dynamic Conditional Correlation.
- Bollerslev (1990). Modelling the Coherence in Short-run Nominal Exchange Rates: A Multivariate Generalized ARCH Model. *Review of Economics and Statistics*. Catalog `ccc` is two-stage Gaussian GARCH + constant \(R\) (Wave 133). Named `optimizer.covariance=ccc` is one-step \(H_{t+1}\) without overlay (Wave 134). Not Engle DCC.
- Engle, Sheppard (2001). Theoretical and empirical properties of DCC.
- Cappiello, Engle, Sheppard (2006). Asymmetric Dynamics in the Correlations of Global Equity and Bond Returns. Scalar `adcc` is a catalog estimator (Wave 128) and a named optimizer path (Wave 129). Diagonal `agdcc` is a catalog estimator (Wave 135) and a named optimizer path (Wave 136). Unrestricted full-matrix `agdcc_full` is a catalog estimator (Wave 137) and a named optimizer path (Wave 138). Student-t DCC is a catalog estimator (Wave 126) and a named optimizer path (Wave 127).
- Bauwens, Laurent, Rombouts (2006). Multivariate GARCH models: a survey. Covariance Student-t DCC likelihood (scale \(((ν-2)/ν)R\)).
- Higham (1988). Computing a nearest symmetric positive semidefinite matrix.

## Regime

- Hamilton (1989). A new approach to the economic analysis of nonstationary time series.
- Rabiner (1989). A tutorial on hidden Markov models.
- Jurafsky, Martin. *Speech and Language Processing* 3ed, Appendix A
  (Hidden Markov Models). https://web.stanford.edu/~jurafsky/slp3/A.pdf
  Discrete first-order \(\lambda=(A,B,\pi)\); Eisner ice-cream HMM.
  Implemented as `quant_fund.hmm` (Wave 155). Does not replace
  ADR-006 `GaussianHMMRegime`.

## Tail risk

- Acerbi, Tasche (2002). On the coherence of expected shortfall.
- Rockafellar, Uryasev (2000). Optimization of conditional value-at-risk.
- Optional EVT / peaks-over-threshold (Embrechts et al.) — not in v1 core.

## Execution

- Almgren, Chriss (2000). Optimal execution of portfolio transactions.
- Implementation shortfall (Perold). Square-root impact as a baseline, not a law of nature.

## Backtest robustness

- Bailey, López de Prado. The Deflated Sharpe Ratio; Probabilistic Sharpe Ratio.
- Bailey, Borwein, López de Prado, Zhu. Probability of Backtest Overfitting / CSCV.
- López de Prado. *Advances in Financial Machine Learning* — purged/embargoed CV, CPCV.
- Diebold, Mariano (1995). Comparing predictive accuracy.

## Foundation-model comparison targets

- Shi et al. (2025). Kronos: A Foundation Model for the Language of Financial
  Markets. *NeurIPS 2025* (arXiv:2508.02739); repo `shiyu-coder/Kronos` (weights under HF org `NeoQuasar`). Used as a
  published SOTA comparison target for causal one-step-ahead return-distribution
  forecasting; see `scripts/sota_eval_kronos.py` and `.dsh-24x7/evidence-sota-eval-*.json`.
- Ansari et al. (2024/25). Chronos / Chronos-Bolt / Chronos-2: Amazon's
  time-series foundation models (`amazon-science/chronos-forecasting`, weights
  under HF org `autogluon`). Chronos-2 evaluated via native quantile output
  (quantile-integral CRPS) under the same causal protocol.
- Das et al. (2024). TimesFM: Google's decoder-only time-series foundation
  model (`google/timesfm`, weights `google/timesfm-2.5-200m-pytorch`), evaluated
  zero-shot under the same protocol.

## Data-snooping / multiple testing

- White, H. (2000). A Reality Check for Data Snooping. *Econometrica* 68(5).
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics* 23(4).
- Romano, J. P., Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica* 73(4).
- Hansen, P. R., Lunde, A., Nason, J. M. (2011). The Model Confidence Set. *Econometrica* 79(2).
- Politis, D. N., Romano, J. P. (1994). The Stationary Bootstrap. *JASA* 89(428).
- Politis, D. N., White, H. (2004). Automatic Block-Length Selection for the Dependent Bootstrap. *Econometric Reviews* 23(1).

## Conformal prediction

- Romano, Patterson, Candès (2019). Conformalized Quantile Regression. *NeurIPS*.
- Gibbs, Candès (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS*.
- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World* — Mondrian conformal predictors.
- Angelopoulos, Bates. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.

## Portfolio

- Markowitz mean-variance.
- Grinold, Kahn. *Active Portfolio Management* — IC, ICIR, residual alpha.

---

# Research references (Day Wave 1 update — 2026-09-16)

Concrete citations for SOTA methods used or targeted by this repo. Map each to `src/quant_fund/` (or tests).
**Honesty:** SYNTHETIC benches are research-only; never treat as live P&L / vendor fills.

## Conformal prediction (finance / time series, 2023–2026)

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511 (updated guide) | Split CP, coverage semantics, nonconformity | `models/conformal*.py`, CRC paths, fixtures in `tests/unit/models/test_*conformal*` |
| Gibbs & Candès, Adaptive Conformal Inference (ACI) / DtACI line | Non-exchangeable sequential coverage | Online CRC / adaptive λ paths (`online_crc`, CRC extremes) |
| Angelopoulos et al., Conformal Risk Control (CRC) | Control E[loss] not just miscoverage | `crc` modules + `test_crc.py` |
| Kaya et al., *Conformal Prediction for Reliable Stock Selections*, PMLR 2025 | CP sets for stock selection | Ranking / conformal rank benches |
| *Conformal Predictive Portfolio Selection (CPPS)*, arXiv:2410.16333 | Portfolio-level conformal intervals | `portfolio_conformal` + extremes tests |
| Cocouvi (2025), *CP for Financial Time Series Under Asset Pricing Models* | ICP / EnbPI / Mondrian / time-weighted vs GARCH | Weighted / localized conformal |
| QDtACI VaR conformal layer, *Risk Management* (2026) | Model-agnostic intervals around VaR | Risk backtest + conformal calibration |
| Adaptive conformal for VaR/ES on crypto, *J. Risk Financial Manag.* 17(6) 2024 | ACI vs GARCH for market risk | Risk battery (Kupiec/Christoffersen/ES) |
| Barber, Candès, Ramdas & Tibshirani, *Predictive Inference with the Jackknife+*, Ann. Statist. 49(1) (2021) | Marginal exchangeable coverage ≥1−2α (Jackknife+/CV+); agg-specific CV+ floors | `models/jackknife_plus.py`, `models/cv_plus.py`, `bench_jackknife_plus` / `bench_cv_plus`; H10/H15 bound floors |
| Bian & Barber, *Training-conditional coverage for distribution-free predictive inference*, Electron. J. Statist. 17 (2023) 2044–2066 (arXiv:2205.03647); follow-ons on stability | Guarantees are typically **marginal**; training-conditional coverage can fail for Jackknife+ without stability — do **not** read H10/H15 `coverage_floor` as training-conditional | Day Wave 41 `coverage_guarantee_scope=marginal_exchangeable` on JP/CV+ meta + benches |

## CPCV / PBO (López de Prado / Bailey)

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| López de Prado, *Advances in Financial Machine Learning* (2018), ch. 7 & 12 | Combinatorial purged CV, purge + embargo; ``φ = C(N-1, k-1)`` backtest paths | `validation/cpcv.py` (`cpcv_path_assignments`, `combinatorial_purged_indices`); `docs/BACKTEST_OVERFITTING.md` |
| Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*, J. Computational Finance 20(4) (2017); PDF: davidhbailey.com/dhbpapers/backtest-prob.pdf | PBO via CSCV; noise rate 1/2 | `cscv_performance` + `probability_of_backtest_overfitting`; notebook `backtest_overfitting.pbo` |
| Bailey & López de Prado, *The Deflated Sharpe Ratio*, J. Portfolio Management 40(5) (2014) | DSR numerical example: SR0≈0.1132, DSR≈0.9004 (N=100); 0.9505 at N=46 and at Normal N=88 | `deflated_sharpe` / `expected_max_sharpe`; `tests/unit/research/test_backtest_overfitting.py` |
| Bailey & López de Prado, *The Sharpe Ratio Efficient Frontier*, J. Risk 15(2) (2012) | PSR and MinTRL; 2.73/2.83/3.24-year normal table; 4.99-year HFR moment pair | `probabilistic_sharpe`, `min_track_record_length` |
| Mantegna, *Hierarchical structure in financial markets*, Eur. Phys. J. B (1999) | Distance `sqrt((1-ρ)/2)` used to cluster correlated trials at a fixed `ρ=0.5` | `effective_n_trials`; receipt `n_trials_effective` |

## Proper scoring rules (Gneiting)

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102(477) (2007) | CRPS, logarithmic score, energy score, propriety | `crps_from_quantiles` (Riemann); `crps_gaussian` / `mean_crps_gaussian` (closed-form); `crps_student_t` / `mean_crps_student_t` (Day Wave 45); `crps_empirical` (ensemble); `log_score_gaussian` / GARCH one-step `log_score_one_step` (Day Wave 105); bench keys `crps_gaussian_closed` / `dm_crps_*` (Day Wave 8) + `crps_scaled_student_t_closed` (Day Wave 45); MATH_SPEC |
| Jordan, Krüger & Lerch, *Evaluating Probabilistic Forecasts with scoringRules*, J. Stat. Softw. 90(12) (2019) | Closed-form CRPS for parametric families incl. location-scale Student-t | `crps_student_t` / `mean_crps_student_t`; bench `crps_scaled_student_t_closed` (Day Wave 45); MATH_SPEC |
| Gneiting et al., *Probabilistic Forecasts, Calibration and Sharpness* | PIT, sharpness subject to calibration | PIT diagnostics |
| Pinball / quantile loss (Koenker; Gneiting quantile scores) | Proper for τ-quantile | Pinball / mean_pinball |
| Annual Review: *Proper Scoring Rules for Estimation and Forecast Evaluation* (2024/25) | Modern loss-orientation convention | Distribution extremes tests |

## Diebold–Mariano & sequential comparison

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Diebold & Mariano, *Comparing Predictive Accuracy*, JBES (1995/2002) | Loss-differential test for **forecasts** | DM pairwise helpers |
| Diebold, *Comparing Predictive Accuracy, Twenty Years Later* (NBER WP 18391) | DM is for forecasts not models | Research agent / scorecard honesty |
| Choe & Ramdas, *Comparing Sequential Forecasters*, Operations Research 72(4) (2023) | Anytime-valid CS / e-processes for score diffs | `e_process_loss_diff` / `e_process_dm` (Day Wave 4); optional pairwise DM flag; `bench_volatility` `e_dm_*` (Day Wave 9); `bench_distribution` `e_dm_crps_*` (Day Wave 18) + scaled `e_dm_crps_scaled_*` (Day Wave 19) |

## E-values / testing by betting

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| Grünwald, de Heide, Koolen; Ramdas et al. e-value / Ville inequality line | Anytime-valid evidence; capital process | `evalues` + Ville fixtures |
| Choe & Ramdas (2023) above | E-processes for forecaster comparison | `metrics.evalues` loss-diff capital + Ville; `pairwise_diebold_mariano(..., include_e_process=True)`; vol bench `e_dm_*`; distribution bench `e_dm_crps_*` (Day Wave 18) + `e_dm_crps_scaled_*` (Day Wave 19) |

## Portfolio conformal / risk / execution

| Citation | Why it matters | Repo map |
|----------|----------------|----------|
| CPPS arXiv:2410.16333 | Conformal portfolio selection | `portfolio_conformal` |
| Almgren & Chriss, *Optimal Execution of Portfolio Transactions*, J. Risk 3 (2001) | Mean–variance execution frontier, IS | `almgren_chriss.py` + extremes (Wave 22) |
| Kyle, *Continuous Auctions and Insider Trading*, Econometrica 53 (1985) | λ from signed volume vs mid change | Northset `kyle_lambda` |
| Roll, *A Simple Implicit Measure of the Effective Bid-Ask Spread*, JF 39 (1984) | \(2\sqrt{-\gamma_1}\) from mid-change autocov | Northset `roll_spread` |
| Parkinson, *The Extreme Value Method for Estimating the Variance of the Rate of Return*, JB 53 (1980) | Range variance vs close-to-close | Northset `parkinson_vs_close_to_close` |
| Garman & Klass, *On the Estimation of Security Price Volatilities from Historical Data*, JB 53 (1980) | OHLC variance | Northset `garman_klass_vs_close_to_close` |
| Rogers & Satchell, *Estimating Variance from High, Low and Closing Prices*, Annals of Applied Probability (1991) | Drift-robust OHLC variance | Northset `rogers_satchell_vs_close_to_close` |
| Yang & Zhang, *Drift-Independent Volatility Estimation…*, JB 73 (2000) | Overnight + RS combination | Northset `yang_zhang_variance` |
| Corwin & Schultz, *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices*, JF 67 (2012) | High-low spread | Northset `corwin_schultz_spread` |
| Amihud, *Illiquidity and stock returns*, JFM 5 (2002) | \|r\| / dollar volume | Northset `amihud_illiquidity` |
| Cont, Kukanov & Stoikov, *The Price Impact of Order Book Events*, J. Financial Econometrics (2014) | OFI | Northset `order_flow_imbalance` |
| Abdi & Ranaldo, *A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices*, RFS 30 (2017) | Close-high-low spread | Northset `abdi_ranaldo_spread` |
| Barndorff-Nielsen & Shephard, *Econometrics of Testing for Jumps in Financial Economics…*, JFE 4 (2006) | Bipower vs RV jump share | Northset `session_bipower_jump` |
| Easley, López de Prado & O’Hara, *Flow Toxicity and Liquidity…*, RFS 25 (2012) | VPIN (bulk-volume proxy) | Northset `vpin_proxy` / `session_vpin` |
| Osler, *Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive Success of Technical Analysis*, JF 58 (2003) | Stop-cluster / sweep dynamics | Northset `liquidity_sweep_frame` |
| Acerbi & Tasche, Expected Shortfall coherence / ES definition | ES as coherent risk measure | ES metrics + Acerbi–Székely style backtests |
| Kupiec POF; Christoffersen (1998) independence/CC | VaR hit-rate & clustering tests | `kupiec_pof` / `christoffersen_*` + `var_backtest_hooks`; `bench_tail` ind/CC (Day Wave 16) |
| Acerbi & Székely (2014), Backtesting Expected Shortfall | Unconditional Z1 / conditional Z2 ES tests | `acerbi_szekely_z1`/`z2` + `var_backtest_hooks` (Day Wave 2); `bench_tail` primary + unscaled/scaled (Day Wave 15) |
| Fissler & Ziegel (2016), Higher order elicitability and Osband's principle | Joint (VaR, ES) elicitability | `fissler_ziegel_loss` / `mean_fissler_ziegel` (Day Wave 3); `bench_tail.fissler_ziegel_mean` (Day Wave 15) |
| Nolde & Ziegel (2017), Elicitability and backtesting | FZ0 0-homogeneous joint score; comparative backtests | FZ0 formula in MATH_SPEC + `var_backtest_hooks.fissler_ziegel_mean` + `bench_tail` (Day Wave 15) |

## Option pricing / vol / allocation (quant-models)

- Black, Scholes (1973); Merton (1973). BSM with dividend yield. `quant_models.black_scholes`, `greeks`.
- Cox, Ross, Rubinstein (1979). Binomial tree, American early exercise. `quant_models.binomial`.
- Heston (1993); Albrecher, Mayer, Schoutens, Tistaert (2007) little-trap CF. `quant_models.heston`.
- Gatheral, Jacquier (2014). Arbitrage-free SVI. `quant_models.svi`.
- Nelson, Siegel (1987); Svensson (1994). NSS zeros. `quant_models.nss`.
- López de Prado (2016). Building Diversified Portfolios that Outperform Out of Sample. *Journal of Portfolio Management*. HRP. `quant_models.hrp`.
- Raffinot (2018). Hierarchical Clustering-Based Asset Allocation. HCAA. `quant_models.hrp.hcaa_weights`.
- Maillard, Roncalli, Teiletche (2010). The Properties of Equally Weighted Risk Contribution Portfolios. ERC. `quant_models.risk_parity`.
- Moskowitz, Ooi, Pedersen (2012). Time Series Momentum. *Journal of Financial Economics*. `quant_models.tsmom`.
- Baltussen, Da, Lammers, Martens (2021). Hedging Demand and Market Intraday Momentum. *Journal of Financial Economics*. GEX last-hour follow/fade. `quant_models.gex.last_hour_decide` (no broker).
- SqueezeMetrics / dealer-gamma sign convention (calls +, puts −; an assumption). `quant_models.gex`.
- Krauss, Do, Huck (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*. Linear window in `quant_models.krauss`; trees already `gbrt`. DNN behind ADR-007.
- Gu, Kelly, Xiu (2020). OOS \(R^2\) vs zero forecast. `quant_models.gkx.r2_oos`.
- Blume (1975). Betas and Their Regression Tendencies. *Journal of Finance*. `quant_models.beta.blume_beta`.
- Buehler, Gonon, Teichmann, Wood (2019). Deep hedging. Baseline is discrete BS delta in `quant_models.monte_carlo`; nets behind ADR-007.
- Source notebooks: https://github.com/davidalmeida90/quant-models and README sibling repos. Dipcatcher port: ADR-033.

## Lightspeed / time-series momentum books

- Jegadeesh, Titman (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*. 63-day total-return score in `lightspeed.momentum`.
- Moskowitz, Ooi, Pedersen (2012). Time Series Momentum. *Journal of Financial Economics*. TQQQ book is a 20/180 EMA rotation, not the 12-month TSMOM sign; TSMOM itself is `quant_models.tsmom`.
- López de Prado (2018). *Advances in Financial Machine Learning*. Meta-labeling + purged CV. `lightspeed.metalabel` is reduce-only.
- Frozen specs: `tqqq-long-full-v1`, `stock-momentum-v1`, `nautica-momentum-v1` from https://github.com/cosmic-hydra/lightspeed. Dipcatcher port: ADR-034. Yahoo expected-metrics there are not Dipcatcher live claims.

## Risk-controlled size (paper book)

- Moreira, Muir (2017). Volatility-Managed Portfolios. *Journal of Finance*. `risk.gates.vol_target`.
- Barroso, Santa-Clara (2015). Momentum has its moments. *Journal of Financial Economics*. Time-varying vol scale on a momentum book; implemented as the same vol-target gate, not a new CS ranker.
- Thorp (2006). The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market. Fractional Kelly `κ μ/σ²`; negative μ → 0 (ADR-032). `risk.gates.kelly_leverage`; `BookRiskOverlay`.
- MacLean, Thorp, Ziemba (2011). *The Kelly Capital Growth Criterion*. Fractional Kelly cap.
- Angelopoulos, Bates, Malik, Jordan (2022). Conformal Risk Control. CRC size on trailing losses. `risk.gates.crc_leverage`. ADR-012 / ADR-036.
- Romano, Wolf (2005). Stepwise Multiple Testing as Formalized Data Snooping. Expanding-window allow/deny size. `risk.gates.stepm_leverage`.
- Asness, Moskowitz, Pedersen (2013). Value and Momentum Everywhere. *Journal of Finance*. Public-OHLCV proxy `vme` (12–1 + George–Hwang 52w; no B/M).
- Moskowitz, Ooi, Pedersen (2012). Time Series Momentum. CS analog `tsmom` on `cs_z_mom_12_1` (the TS book remains `quant_models.tsmom`).
- Krauss, Do, Huck (2017). Linear logistic CS challenger `krauss` on the same purged walk-forward as ridge; DNN behind ADR-007.

## K-line foundation models

- Shi, Fu, Chen, Zhao, Xu, Zhang, Li (2025). Kronos: A Foundation Model for the Language of Financial Markets. *arXiv:2508.02739*. Hierarchical discrete K-line tokens + autoregressive decoder. Dipcatcher engine: **robinhood+** (ADR-023). MIT. https://github.com/shiyu-coder/Kronos
- Zhao, Goyal, Mentzer, Van Gool (2024). Binary Spherical Quantization. *arXiv:2406.07548*. Tokenizer stage used by Kronos / robinhood+.

## Repo honesty anchors

- [MATH_SPEC.md](MATH_SPEC.md), [VALIDATION.md](VALIDATION.md), [DATA_SOURCE_LABELS.md](DATA_SOURCE_LABELS.md)
- `FORBIDDEN_RESEARCH_METRIC_KEYS` / `family_blob_forbidden_metrics_absent` (no Sharpe/Sortino/Calmar/pnl/nav in **research family** headlines; paper `analytics_export` may nest equity/stress pnl/nav under `live_pnl_claim=false`)
- Paper/validate: `would_promote_live=false` on SYNTHETIC; `live_pnl_claim=false`


## Research canon wave (metrics/models/portfolio/execution)

Serial dependence and unit roots — `quant_fund.metrics.serial`:

- Lo, MacKinlay (1988). Stock Market Prices Do Not Follow Random Walks.
  *Review of Financial Studies* 1. Variance-ratio test (VR(q), z1/z2).
- Chow, Denning (1993). A Simple Multiple Variance Ratio Test. *Journal of
  Econometrics* 58. `chow_denning_test`.
- Wright (2000). Alternative Variance-Ratio Tests Using Ranks and Signs.
  *JBES* 18. `wright_variance_ratio`.
- Ljung, Box (1978); Box, Pierce (1970); Engle (1982) ARCH-LM; Jarque, Bera
  (1980); Bartels (1982) rank test; Wald–Wolfowitz runs test.
- Dickey, Fuller (1979); Kwiatkowski et al. (1992) KPSS; Phillips, Perron
  (1988); Zivot, Andrews (1992) — `unit_root_battery` via `arch.unitroot`.

Extreme value theory — `quant_fund.metrics.extremes`:

- Hill (1975); Pickands (1975); Dekkers, Einmahl, de Haan (1989) tail-index
  estimators.
- Balkema, de Haan (1974) / Pickands (1975) GPD peaks-over-threshold;
  Jenkinson (1955) GEV block maxima; return levels.
- Leadbetter et al. (1983) extremogram; Davis, Mikosch (2009) empirical
  extremogram.

Entropy and fractal diagnostics — `quant_fund.metrics.entropy`,
`quant_fund.metrics.fractal`:

- Shannon (1948); Pincus (1991) ApEn; Richman, Moorman (2000) SampEn;
  Bandt, Pompe (2002) permutation entropy; Kaspar, Schuster (1987)
  Lempel–Ziv; Schreiber (2000) transfer entropy.
- Hurst (1951) R/S; Peng et al. (1994) DFA; Katz (1988); Higuchi (1988)
  fractal dimensions; Sevcik (1998).

Multiple-testing corrections — `quant_fund.validation.fdr`:

- Benjamini, Hochberg (1995); Benjamini, Yekutieli (2001); Holm (1979);
  Hochberg (1988); Bonferroni; Sidak (1967); Storey (2002) pi0.

HAC inference — `quant_fund.metrics.hac`:

- Newey, West (1987); Andrews (1991) kernels and automatic bandwidth;
  Andrews, Monahan (1992) AR(1) prewhitening; Kiefer, Vogelsang (2005).

Fractional differentiation and AFML labeling — `quant_fund.models.fracdiff`,
`quant_fund.labels.barriers`, `quant_fund.features.bars`:

- Lopez de Prado (2018). *Advances in Financial Machine Learning*, ch. 3–5:
  fracdiff weights/FFD/min-d ADF scan; cUSUM events, triple barrier,
  meta-labels, trend scanning, uniqueness/decay weights, sequential
  bootstrap; tick/volume/dollar/imbalance/run bars.
- Hosking (1981). Fractional differencing. *Biometrika* 68.
- Easley, Lopez de Prado, O'Hara (2012) — information-driven bars.

State space and point processes — `quant_fund.models.state_space`,
`quant_fund.models.point_process`, `quant_fund.models.changepoint`:

- Kalman (1960); local level/trend; Elliott, van der Hoek, Malcolm (2005)
  pairs-trading hedge ratio via Kalman.
- Ornstein–Uhlenbeck MLE (exact AR(1) transition); half-life.
- Hawkes (1971). Spectra of some self-exciting point processes — exp-kernel
  MLE, branching ratio, compensator residuals (Ogata 1988), thinning
  simulation (Lewis, Shedler 1979).
- Adams, MacKay (2007) BOCPD; Page (1954) CUSUM; Page–Hinkley; Scott,
  Knott (1974) binary segmentation; Jackson et al. (2005) optimal
  partitioning; Killick, Fearnhead, Eckley (2012) PELT cost function.

Random matrix theory — `quant_fund.models.rmt`:

- Marchenko, Pastur (1967); Laloux et al. (1999); Plerou et al. (2002);
  eigenvalue clipping + detoning (Lopez de Prado 2018, ch. 2);
  Kritzman, Li, Page, Rigobon (2011) absorption ratio; Meucci (2009)
  effective rank; inverse participation ratio.

Portfolio allocators — `quant_fund.portfolio.allocators`:

- Lopez de Prado (2016) HRP; Maillard, Roncalli, Teiletche (2010) ERC;
  Choueifaty, Coignard (2008) maximum diversification; Black, Litterman
  (1992); Kelly (1956)/Thorp (2006); Rockafellar, Uryasev (2000) CVaR LP;
  Hallerbach (2012) volatility targeting.

Execution and impact — `quant_fund.execution.impact`:

- Almgren et al. (2005) permanent/sqrt impact; Bouchaud et al. (2009)
  propagator; Gatheral (2010) no-dynamic-arbitrage; Perold (1988)
  implementation shortfall; Kissell, Glantz (2003) POV/benchmark slippage.

Forecast combinations and bandits — `quant_fund.models.ensemble`,
`quant_fund.models.bandits`:

- Bates, Granger (1969); Granger, Ramanathan (1984); Stock, Watson (2004)
  trimmed/median combinations.
- Auer, Cesa-Bianchi, Fischer (2002) UCB1; Auer et al. (2002) EXP3;
  Thompson (1933); Kaufmann, Cappe, Garivier (2012) Bayes-UCB; Sutton,
  Barto (2018) epsilon-greedy.

Copulas — `quant_fund.models.copula`:

- Sklar (1959); Nelsen (2006); Genest, Favre (2007) IFM; Embrechts,
  McNeil, Straumann (2002); Demarta, McNeil (2005) t copula; Marshall,
  Olkin (1988) Archimedean simulation via frailty.

Parametric risk and drawdown — `quant_fund.metrics.risk_parametric`,
`quant_fund.metrics.drawdown`:

- Cornish, Fisher (1937); Zangari (1996) modified VaR; McNeil, Frey (2000)
  t ES; Checkalov, Uryasev, Zabarankin (2004) CDaR; Keating, Shadwick
  (2002) Omega; Kaplan, Knowles (2004) Kappa; Martin, McCann (1989) ulcer
  index; Zephyr pain/tail ratios; Magdon-Ismail, Atiya et al. (2004)
  expected max drawdown.

Technical-analysis canon — `quant_fund.features.indicators`:

- Wilder (1978) RSI/ATR/DMI/ADX; Appel MACD; Bollinger bands; Keltner;
  Kaufman KAMA; Ehlers Fisher transform; Elder force/Elder-ray; Pring KST;
  Granville OBV; Chaikin ADL/CMF/oscillator/volatility; Arms TRIN; Lane
  stochastics; Williams %R/UO; Lambert CCI; Donchian; Hosoda Ichimoku;
  Chande Aroon/CMO; Botes–Siepman Vortex; Mulloy DEMA/TEMA; Hull HMA;
  MFI/EOM/VWAP/stochRSI/TRIX/zlema.

Cycle and spectral filters — `quant_fund.features.cycles`:

- Goertzel (1958); FFT dominant cycle; Marple (1999) analytic signal /
  Hilbert instantaneous frequency; Ehlers (2013) SuperSmoother, roofing
  filter, bandpass.

Regression and IV diagnostics - quant_fund.metrics.regression:

- Gauss-Markov OLS; White (1980) HC0-HC3; Cribari-Neto (2004) HC4; Newey,
  West (1987) HAC; Durbin-Watson (1950); Breusch-Godfrey (1978);
  Breusch-Pagan (1979); White (1980) het test; Ramsey (1969) RESET;
  Goldfeld-Quandt (1965); Brown-Durbin-Evans (1975) CUSUM/CUSUMSQ; Cook
  (1977) distance; Belsley-Kuh-Welsch (1980) VIF; Theil-Sen; Huber (1964)
  IRLS; Koenker-Bassett (1978) quantile regression; Sargan (1958)/Hansen
  (1982) J test; 2SLS.

Cross-sectional factor models - quant_fund.models.factor_models:

- Fama, MacBeth (1973) two-pass; Shanken (1992) EIV; Fama, French (1993)
  2x3 sorts; Jegadeesh, Titman (1993) momentum; Carhart (1997); Bai, Ng
  (2002) IC criteria; Stock, Watson (2002) diffusion indices; Jensen
  (1968) alpha.

Trend/cycle decomposition - quant_fund.models.filters:

- Hodrick, Prescott (1997); Ravn, Uhlig (2002) lambda rule; Baxter, King
  (1999); Christiano, Fitzgerald (2003); Hamilton (2018); Beveridge,
  Nelson (1981); Corbae, Ouliaris (2006) DFT bandpass.

Realized measures - quant_fund.models.realized:

- ABDL (2001) RV; Barndorff-Nielsen, Hansen, Lunde, Shephard (2008)
  realized kernels; Zhang, Mykland, Ait-Sahalia (2005) TSRV; Jacod et al.
  (2009) pre-averaging; Barndorff-Nielsen, Shephard (2004/2006) bipower;
  Lee, Mykland (2008) jump test; Huang, Tauchen (2005) relative jump;
  Podolskij, Vetter (2009) tripower quarticity; BNKS (2010) semivariance.

VAR, cointegration, connectedness - quant_fund.models.var_coint:

- Sims (1980) VAR; Lutkepohl (2005); Pesaran, Shin (1998) generalized
  IRF/FEVD; Diebold, Yilmaz (2012) spillover index; Engle, Granger (1987);
  Johansen (1991/1995) trace/max-eig; MacKinnon-Haug-Michelis (1999)
  critical values; Granger (1969) causality F-tests; VECM reduced-rank.

Microstructure liquidity - quant_fund.features.liquidity:

- Amivest ratio; Lesmond, Ogden, Trzcinka (1999) LOT; Lesmond (2005) FHT;
  Pastor, Stambaugh (2003) gamma; Hasbrouck (1991) lambda; Glosten,
  Harris (1988) spread components; Holden (2009) effective tick.

Distribution and normality battery - quant_fund.metrics.distribution:

- Shapiro, Wilk (1965); Anderson, Darling (1954); Cramer-von Mises;
  Lilliefors (1967, MC p-values); Dagostino-Pearson (1973) K2; Pearson
  (1900) chi2; Mardia (1970) multivariate; Brys-Hubert-Struyf (2004)
  medcouple; Rousseeuw, Croux (1993) Qn scale.

Signal decomposition - quant_fund.models.decomposition:

- Broomhead, King (1986); Vautard-Ghil (1992); Golyandina-Nekrutkin-
  Zhigljavsky (2001) SSA + R-forecasting; Huang et al. (1998/2003) EMD,
  Hilbert-Huang instantaneous frequency; robust median-filter trend.

Calibration extensions - quant_fund.metrics.calibration2:

- Murphy (1973) Brier decomposition; Winkler (1972) interval score;
  Dawid, Sebastiani (1999); Hosmer, Lemeshow (1980); Murphy, Winkler
  (1977) reliability diagrams; Scheuerer, Hamill (2015) variogram score;
  pinball/check loss; Candille, Talagrand (2005) spread-skill.

Mixture models - quant_fund.models.mixture:

- Dempster, Laird, Rubin (1977) EM; McLachlan, Peel (2000); Peel,
  McLachlan (2000) robust t-mixtures (ECM with latent-scale weights);
  Schwarz (1978) BIC; Biernacki-Celeux-Govaert (2000) ICL-style entropy.

Pairs selection - quant_fund.models.pairs:

- Gatev, Goetzmann, Rouwenhorst (2006) distance approach; Vidyamurthy
  (2004) cointegration approach; Leung, Li (2016) OU optimal entry/exit;
  Chan (2013) half-life filters; Elliott-van der Hoek-Malcolm (2005).


## Wave 3 — Dynamic & nonlinear-inference canon

### Regime switching
- Hamilton, J.D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica* 57(2) — Markov-switching regression via EM + Hamilton filter.
- Kim, C.J. (1994). "Dynamic linear models with Markov-switching." *J. Econometrics* 60 — Kim smoother.
- Klaassen (2002). Improving GARCH volatility forecasts with regime-switching GARCH.

### Extended GARCH
- Ding, Granger & Engle (1993). "A long memory property of stock market returns and a new model." *J. Empirical Finance* 1 — APARCH.
- Baillie, Bollerslev & Mikkelsen (1996). "Fractionally integrated generalized autoregressive conditional heteroskedasticity." *J. Econometrics* 74 — FIGARCH.
- Engle & Ng (1993). "Measuring and testing the impact of news on volatility." *J. Finance* 48 — news impact curve.

### Forecast evaluation
- Clark & West (2007). "Approximately normal tests for equal predictive accuracy in nested models." *J. Econometrics* 138.
- Harvey, Leybourne & Newbold (1997). "Testing the equality of prediction mean squared errors." *IJF* 13 — HLN small-sample DM.
- Harvey, Leybourne & Newbold (1998). "Tests for forecast encompassing." *JBES* 16 — ENC-T.
- Giacomini & White (2006). "Tests of conditional predictive ability." *Econometrica* 74.
- Giacomini & Rossi (2010). "Forecast comparisons in unstable environments." *J. Applied Econometrics* 25 — fluctuation test.
- McCracken (2007). "Asymptotics for out of sample tests of Granger causality." *J. Econometrics* 140.

### Bootstrap & subsampling
- Wu (1986); Mammen (1993): wild bootstrap for heteroskedastic errors.
- Buhlmann (1997). "Sieve bootstrap for time series." *Bernoulli* 3.
- Politis & Romano (1994). "Large sample confidence regions based on subsamples." *Ann. Statist.* 22.
- MacKinnon (2009). Bootstrap hypothesis testing. In *Handbook of Computational Econometrics*.

### Panel econometrics
- Levin, Lin & Chu (2002). "Unit root tests in panel data." *J. Econometrics* 108 — LLC.
- Im, Pesaran & Shin (2003). "Testing for unit roots in heterogeneous panels." *J. Econometrics* 115 — IPS.
- Maddala & Wu (1999). "A comparative study of unit root tests with panel data." *Oxford Bull.* 61 — Fisher-ADF.
- Hadri (2000). "Testing for stationarity in heterogeneous panel data." *Econometrics J.* 3.
- Pesaran (2004/2015). "General diagnostic tests for cross section dependence in panels." *Cambridge WP* / *J. Applied Econometrics* — CD test.
- Anderson & Hsiao (1981). "Estimation of dynamic models with error components." *JASA* 76.
- Arellano & Bond (1991). "Some tests of specification for panel data." *Rev. Econ. Studies* 58 — diff-GMM + Sargan/AR(2).

### Nonlinear dependence
- Szekely, Rizzo & Bakirov (2007). "Measuring and testing dependence by correlation of distances." *Ann. Statist.* 35 — dCor.
- Gretton et al. (2005). "Measuring statistical dependence with Hilbert–Schmidt norms." *ALT* — HSIC.
- Gretton et al. (2012). "A kernel two-sample test." *JMLR* 13 — MMD.
- Chatterjee (2021). "A new coefficient of correlation." *JASA* 116 — xi_n.
- Kraskov, Stogbauer & Grassberger (2004). "Estimating mutual information." *Phys. Rev. E* 69 — KSG kNN MI.

### Kernel methods
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Scholkopf, Smola & Muller (1998). "Nonlinear component analysis as a kernel eigenvalue problem." *Neural Computation* 10 — kernel PCA.
- Shawe-Taylor & Cristianini (2004). *Kernel Methods for Pattern Analysis* — kernel ridge.
- Williams & Seeger (2001). "Using the Nystrom method to speed up kernel machines." *NeurIPS*.

### Event studies
- MacKinlay (1997). "Event studies in economics and finance." *J. Econ. Literature* 35.
- Brown & Warner (1985). "Using daily stock returns: the case of event studies." *J. Financial Econ.* 14.
- Patell (1976). "Corporate forecasts of earnings per share and stock price behavior." *J. Accounting Research* 14.
- Corrado (1989). "A nonparametric test for abnormal security-price performance." *J. Financial Econ.* 23.
- Boehmer, Musumeci & Poulsen (1991). "Event-study methodology under conditions of event-induced variance." *J. Financial Econ.* 30 — BMP.

### VaR backtesting
- Kupiec (1995). "Techniques for verifying the accuracy of risk measurement models." *J. Derivatives* 3.
- Christoffersen (1998). "Evaluating interval forecasts." *IER* 39.
- Haas (2001). "New methods in backtesting." In *Research in Finance*.
- Basel Committee on Banking Supervision (1996). *Supervisory framework for backtesting* — traffic-light zones.
- Berkowitz, Christoffersen & Pelletier (2011). "Evaluating value-at-risk models with desk-level data." *Management Science* 57.

### Long memory
- Geweke & Porter-Hudak (1983). "The estimation and application of long memory time series models." *JTSA* 4 — GPH.
- Kunsch (1987); Robinson (1995). "Gaussian semiparametric estimation of long range dependence." *Ann. Statist.* 23 — local Whittle.
- Fox & Taqqu (1986). "Large-sample properties of parameter estimates for strongly dependent stationary Gaussian time series." *Ann. Statist.* 14 — Whittle.
- Lo (1991). "Long-term memory in stock market prices." *Econometrica* 59 — modified R/S.

### Survival analysis
- Kaplan & Meier (1958). "Nonparametric estimation from incomplete observations." *JASA* 53.
- Nelson (1972); Aalen (1978): cumulative hazard estimation.
- Mantel (1966). "Evaluation of survival data and two new rank order statistics." *Cancer Chemo. Reports* 50 — log-rank.
- Cox (1972). "Regression models and life-tables." *JRSS-B* 34 — proportional hazards.
- Greenwood (1926): variance of the survival estimator.

### SOTA canon wave 8 — anytime-valid inference, calibration, multivariate scores (2026-09-27)
- Székely (2003), InterStat; Gneiting & Raftery (2007), *JASA* 102 — energy score strict propriety. `metrics/energy_score.py`.
- Gneiting & Ranjan (2013), *Electron. J. Statist.* 7 — threshold/kernel weighting of scoring rules.
- Gneiting, Balabdaoui & Raftery (2007), *JRSS-B* 69 — calibration principle for distributional forecasts.
- Thorarinsdottir & Gneiting (2010), *JRSS-A* 173 — variance scaling. `models/posthoc_calibration.py`.
- Déqué (2007), *Global Planet. Change* 57 — quantile mapping.
- Chernozhukov, Fernández-Val & Galichon (2010), *Econometrica* 78 — rearrangement for non-crossing quantiles.
- Zadrozny & Elkan (2002), *KDD*; Meinshausen (2006), *JMLR* 7 — quantile regression / distribution recovery.
- Wang & Ramdas (2022), *JRSS-B* 84 — e-BH: FDR control on e-values under arbitrary dependence. `metrics/anytime_fdr.py`.
- Wang, Dandapanthula & Ramdas (2025), *Statist. Probab. Lett.* — stopped e-BH under optional stopping (arXiv:2502.08539).
- Xu & Ramdas (2024), AISTATS — online FDR with e-values (e-LOND) (arXiv:2311.06412).
- Shin, Ramdas & Rinaldo (2023), *Ann. Statist.* 51 — e-detectors: anytime-valid sequential change detection (arXiv:2203.03532). `metrics/e_detectors.py`.
- Howard, Ramdas, McAuliffe & Sekhon (2021), *Ann. Statist.* 49 — time-uniform concentration; Ville (1939).
- Hoeffding (1963), *JASA* 58 — bounded e-values. Shiryaev (1963) — geometric-prior mixture.
- Zaffran et al. (2022), ICML — AgACI (arXiv:2202.07282); Zaffran et al. (2022), NeurIPS — FACI aggregation under distribution shift. `models/agaci.py`.
- Gibbs & Candès (2021), NeurIPS 34 — ACI; Koenker & Bassett (1978) — pinball; Cesa-Bianchi & Lugosi (2006) — EG updates; Gaillard, Stoltz & Van Erven (2014), COLT — ML-OGD.
- Zhang, Wei, Ren & Zou (2025), ICML — e-GAI: e-value generalized alpha-investing; e-LORD + adaptive e-SAFFRON online FDR under arbitrary dependence (arXiv:2506.01452); SAFFRON base: Ramdas, Zrnic, Wainwright & Jordan (2018), *ICML* — `metrics/anytime_fdr.py` `ELord`/`ESaffron`.

### SOTA canon wave 9 — distributional forecasts and change monitoring (2026-09-27)
- Prinster, Han, Liu & Saria (2025). "WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales." *ICML*, PMLR 267, arXiv:2505.04608. `metrics/watch.py`.
- Tibshirani, Foygel Barber, Candès & Ramdas (2019), NeurIPS — conformal prediction under covariate shift (density-ratio weights).
- Xu & Xie (2021), ICML, PMLR 139; Xu & Xie (2023), *IEEE TPAMI* 45(10) — EnbPI: ensemble bootstrap conformal intervals for time series, signed-residual width-minimising band (arXiv:2010.09107); Politis & Romano (1992) — circular block bootstrap. `models/enbpi.py`.
- Meinshausen (2006), *JMLR* 7 — quantile regression forests (leaf-weight conditional CDF). `models/qrf.py`.
- Duan et al. (2020), ICML — NGBoost: natural gradient boosting for probabilistic prediction (arXiv:1910.03225); Amari (1998) — natural gradient. `models/ngboost_lite.py`.
- Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World* — conformal test martingales; Vovk et al. (2021), COPA — Simple Jumper / retrain-on-alarm (arXiv:2012.14246); Fedorova et al. (2012), ICML — plug-in martingales; Ville (1939) — 1/α anytime alarm. `metrics/conformal_martingale.py`.
### SOTA canon wave 10 — sequential exchangeability monitoring, leaky-oracle red team (2026-09-27)
- Prinster, Han & Saria (2025), arXiv:2505.04608 — weighted-conformal test martingales (WCTM) for adaptive monitoring. `models/watch.py`.
- Vovk, Gammerman & Shafer (2005), Springer — conformal p-values; Tibshirani et al. (2019), arXiv:1904.06001 — weighted conformal; Vovk & Wang (2022), arXiv:2202.13095 — conformal testing; Volkhonskiy et al. (2017), arXiv:1706.02244 — power martingales; Shafer (2021), JRSS-A — testing by betting; Grünwald, de Heide & Koolen (2019), arXiv:1906.07801 — safe testing; Ville (1939).
- Gençay (2026), arXiv:2608.27734 — leakage-safe, search-aware evaluation: leaky oracles survive DSR/PBO; structural look-ahead exclusion + search-trial-count deflation as fixes. `validation/leakage_redteam.py`.
- Bailey & López de Prado (2012), J. Investment Management — PSR; Bailey & López de Prado (2014), J. Portfolio Management 40(5) — DSR; López de Prado & Bailey (2014), J. Portfolio Management 40(4), arXiv:1405.3421 — PBO/CSCV; Bonferroni (1935; 1936) — log-count correction.

### SOTA canon wave 9 — distributional ML baselines, TS conformal, regime-conditional eval (2026-09-27)
- Duan, Avati, Ding, Thai, Basu, Ng & Schuler (2020), ICML, PMLR 119 — NGBoost natural-gradient boosting. `models/ngboost_lite.py`.
- Gneiting, Raftery, Westveld & Goldman (2005), *MWR* 133 — closed-form Gaussian CRPS; Jordan, Krüger & Lerch (2019), *JSS* 90 — Student-t CRPS; Lange, Little & Taylor (1989), *JASA* 84 — t Fisher information; Amari (1998) — natural gradient.
- Meinshausen (2006), *JMLR* 7:983–999 — quantile regression forests. `models/quantile_forest.py`.
- Xu & Xie (2021), ICML; (2023), *IEEE TPAMI* 45 — EnbPI ensemble batch prediction intervals (arXiv:2010.09107). `models/enbpi.py`.
- Gibbs & Candès (2021), NeurIPS 34 — ACI miscoverage recursion (transplanted onto the width scale, documented deviation); Angelopoulos et al. (2023), arXiv:2310.16828 — conformal PID.
- Politis & Romano (1994), *JASA* 89; Politis & White (2004), *Economet. Reviews* 23 — stationary/adaptive block bootstrap.
- Howard, Ramdas, McAuliffe & Sekhon (2021), *Ann. Statist.* 49 — time-uniform inference; Ramdas, Ruf, Larsson & Koolen (2022); Waudby-Smith & Ramdas (2024) — e-processes.
- Nystrup, Madsen & Lindström (2018), *J. Forecasting* 37 — regime-dependent density forecasting (regime-stratified OOS practice). `validation/regime_eval.py`.

### SOTA canon wave 11 — rough paths, optimal transport, DRO, conformal control (2026-09-28)
- Chen (1958), *Trans. AMS* — integration of paths via signature; Lyons (1998), *Rev. Mat. Iberoam.* — rough differential equations; Chevyrev & Kormilitzin (2016), arXiv:1603.03788 — signature primer; Bonnier et al. (2019), arXiv:1905.08494 — lead-lag/Lévy area; Kidger & Foster (2020), arXiv:2005.08328 — signature kernel; Witt (1937); Hall (1950); Reutenauer (1993); Magnus (1954) — free Lie algebra / Lyndon basis / series log. `models/path_signatures.py`.
- Villani (2003), *Topics in Optimal Transportation*; Cuturi (2013), NeurIPS, arXiv:1306.0895 — Sinkhorn; Schmitzer (2019), arXiv:1610.06519 — log-domain stabilization; Genevay, Peyré & Cuturi (2018), arXiv:1706.00292 — Sinkhorn divergence; Gelbrich (1990); Bhatia, Jain & Lim (2019), arXiv:1801.09287 — Bures–Wasserstein; Ramdas, Garcia Trillos & Cuturi (2017), arXiv:1509.02237. `metrics/wasserstein.py`.
- Mohajerin Esfahani & Kuhn (2018), *Math. Programming* — data-driven DRO duality (Thm 4.2); Blanchet & Murthy (2019), arXiv:1604.03064 — OT model risk; Gao & Kleywegt (2023), *Math. Programming* — W-ball DRSP. `portfolio/wasserstein_dro.py`.
- Angelopoulos, Candès & Tibshirani (2023), NeurIPS, arXiv:2307.16895 — conformal PID control (spec's arXiv:2310.16828 was a mis-citation — TD-MPC2; corrected in module); Gibbs & Candès (2021) — ACI baseline. `models/conformal_pid.py`.

### SOTA canon wave 12 — beyond-exchangeability conformal, CS, OT GoF, score canon, stacking, deep hedging (2026-09-29)
- Barber, Candès, Ramdas & Tibshirani (2023), *Ann. Statist.* 51(2), doi:10.1214/23-AOS2276, arXiv:2202.13415 — conformal prediction beyond exchangeability ("NexCP": normalized covariate/recency weights, Theorem 2/3 coverage bounds under TV perturbations). **Roadmap citation corrected**: the abbreviation traces to Barber & Tibshirani (2025), arXiv:2504.02292; no Liu/Wang/Xie NexCP paper exists. `models/nexcp.py`.
- Xu & Xie (2021/2023), arXiv:2010.09107 — EnbPI primitives reused per-horizon; Bonferroni (1935) — joint multi-horizon coverage bound. `models/enbpi_multihorizon.py`.
- Howard, Ramdas, McAuliffe & Sekhon (2021), *Ann. Statist.* 49, arXiv:1810.08240 — sub-Gaussian mixture / poly-stitching / poly-hedge-ε confidence sequences, empirical-Bernstein CS (Thm 4); Jamieson, Malloy, Nowak & Bubeck (2014), COLT — hedge-ε boundary; Waudby-Smith & Ramdas (2024), *JRSS-A*, arXiv:2010.09686 — WSR betting confidence sequences (hedged capital, truncated bets). `metrics/confidence_sequences.py`.
- Rabin, Flamary, Cuturi, Villani (2010), ECCV; Bonneel, Rabin, Peyré & Golland (2015), *CVPR* — sliced Wasserstein barycenters (rank-matching fixed point; documented limit-cycle non-convergence in d≥2 with finite directions); Kolouri, Pope, Martin & Rohde (2019), NeurIPS, arXiv:1804.01947 — generalized/max-sliced; Gao & Xiu (2021) — SW properties; Phipson & Smyth (2010), *Stat. Appl. Genet. Mol. Biol.* — permutation p-values; Székely & Rizzo (2013) — energy distance head-to-head. `metrics/sliced_wasserstein.py`.
- Bröcker (2012), *QJRMS* 138:1611–1617, doi:10.1002/qj.1891 — ensemble CRPS potential/quality/reliability decomposition; Kolassa (2016), ***IJF* 32(3):788–803** (lane brief said JRSS-A — corrected) — discrete CRPS decomposition; Zamo & Naveau (2018), *J. Climate* 31 — fair (unbiased) ensemble CRPS; Epstein (1969), *JAM* 9 — ranked probability score; Brier (1950), *MWR* 78; Good (1952) — spherical score; Siegert (2013), *Wea. Forecasting* 28 — exact Brier REL−RES+UNC identity. `metrics/score_decomposition.py`.
- Yao, Vehtari, Simpson & Gelman (2018), *Bayesian Analysis* 13(3), arXiv:1704.02030 — stacking of predictive distributions (LOO log-score maximization over the simplex); Clyde, Ghosh & Littman (2011), *JASA* 106 — Bayesian bootstrap pseudo-BMA+. `models/stacking.py`.
- Vovk, Gammerman & Shafer (2005), Springer; Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018), *JASA* 113; Tibshirani, Foygel Barber, Candès & Ramdas (2019), NeurIPS, arXiv:1802.09184 — density-ratio weighting (regime-posterior mixture likelihood ratio); Barber, Candès, Ramdas & Tibshirani (2023), *Ann. Statist.* 51, arXiv:2102.13415 — limits of distribution-free conditional coverage (weighting is heuristic absent exact posteriors); Hamilton (1989), *Econometrica* 57 — regime switching. `models/regime_conformal_var.py`.
- Buehler, Gonon, Teichmann & Wood (2019), *Quantitative Finance* 17(8), arXiv:1802.03042 — deep hedging under friction; Davis & Norman (1990), *Math. OR* 15 — proportional transaction costs; Rockafellar & Uryasev (2000), *OR* 50 — CVaR minimization; Föllmer & Schied (2016), de Gruyter — monetary risk measures, translation invariance. `models/deep_hedging.py` (torch-gated, `nn` extra).
- Zheng, Chiang, Sheng, Zhuang et al. (2023), NeurIPS D&B, arXiv:2306.05685 — MT-Bench judge protocol; Islam, Kannappan, Kiela, Qian, Scherrer, Vidgen (2023), arXiv:2311.11944 — FinanceBench evidence-QA format; Liu, Zhang et al. (2024), *ACL Findings*, arXiv:2403.07718 — FinToolBench tool-trace format. Adapter ports + sealed SYNTHETIC banks only — NOT real benchmark scores. `fx1/eval/ext_bench.py`, `fx1/eval/ext_bench_banks.py`.
- DeMillo, Lipton & Sayward (1978), *IEEE Computer* — mutation analysis hypothesis; Jia & Harman (2011), *IEEE TSE* 37 — AST mutation operator canon; off-by-one/boundary operators per Acree (1979). `scripts/mutation_test.py` (stdlib-only harness; sentinel-preflight overlay via sitecustomize meta-path finder).

### SOTA canon wave 13 — Tier A: hardest conformal/UQ theory (2026-09-29; all sources fetched and verified before implementation)
- Zhai, Cheng & Wu (2026), arXiv:2609.33868 — conformal coverage of time series: FDM-based non-asymptotic coverage-error bounds (Wu 2005a functional dependence measure), Bahadur representation + first realized-coverage CLT for *split* conformal under temporal dependence (Halkiewicz 2026 preceded for rolling-origin), nonoverlapping batch-means block SE (Flegal–Jones; ℓ=⌊n^{2/3}⌋) as the proof-backed estimator, fixed-B scaled-t reference √(B/(B−1))·t_{B−1} (Jones et al. 2006, their Eq. 24), long-memory paired-block subsampling with two-scale memory estimator β̂ (their App. C.1, Lemma 10; moving-block bootstrap/HAC fail per their Table 2). Companion: Zhai, Cheng & Wu (2026), arXiv:2609.33866 — interval-length accuracy for conformalized quantile/median regression (out of scope for the coverage layer; noted in module). `metrics/conformal_coverage_inference.py`.
- Bhattacharyya & Ramdas (2026), arXiv:2609.27179 — minimax-optimal conformal change detection: restart-weighted aggregations (sum/max) of betting-class mixture martingales over candidate changepoints (their Eq. 14, Def. 2.6); conformal e-process (‖w‖₁≤1, Thm 2.7(1)) and e-detector (‖w‖_∞≤1, Thm 2.7(2)); *restricted first-order* minimax over late changepoints (Thms 3.9/3.10; universal fixed-T consistency impossible, Prop. 3.1); polynomial k^{−(1+η)} / near-harmonic weights (Cor. 3.6, Rem. 3.7); mixtures alone insufficient (Ex. 2.4/2.5, CMM delay scale √(T log b), Thm 5.3); Vovk (2021) CTM/CUSUM/SR comparators. `metrics/conformal_e_detectors.py`.
- Cheng, Liang & Barber (2026), arXiv:2609.26951 — rolling conformal prediction for sequential model training: calibrate-then-roll p-values (their §2.2), universal factor-two marginal floor under exchangeability, tightened finite-n floor 1−(2α−1/(n+1))₊ (Rem. 1), tightness via alternating-fold construction (Prop. 1/App. A.4), i.i.d. training-conditional validity incl. time-uniform (Thm 3, Eq. 7), stability sharpening to 1−α (Thm 4, score-comparison stability Eq. 9). Primitive is the score function s_i(z; Z_{<i}), not a point forecaster. `models/rolling_conformal.py`.
- Khosravi & Huo (2026), arXiv:2609.32211 — rank confidence sequences: pairwise grid-averaged betting e-processes with predictable offset for superpopulation and finite-benchmark (without-replacement) sampling (Eq. 2–3), exact closed testing over **weak orders** (Fubini(M) cells; Algorithm 1; practical cap M≈8), polynomial-time transitivity-pooled shortcut (Algorithm 2; exact certification coNP-hard, Prop. A.2), rank intervals Eq. (6), tiers Thms 4.2–4.4. `metrics/rank_confidence_sequences.py`.
- Gao, Zhang, Xie, Jing, Wang & Liu (2026), arXiv:2609.32248 — BB-EDGE: block-factorized empirical-Bernstein e-processes with weight-proportional stakes (Prop. 1, ψ_E(λ)=−log(1−λ)−λ), **direct** e-Holm on e-processes (Eq. 8; Hartog & Lei 2025, arXiv:2501.09015, Thm 4.2) — distinct from static e-BH; anytime FWER under arbitrary within-block/cross-pair dependence (Thm 1), pilot-targeted Top-k certification (App. A), simultaneous rank intervals (App. B, Cor. 1). `metrics/rank_confidence_sequences.py`.
- Yang, Huang, Hou, Imbens & Jordan (2026), arXiv:2609.25388 — PICPIs: self-consistency P(Y=1 | p(X)∈I) ∈ I (Def. 2.1; regression form E[Y|·] via Rmk 3.6 rescaling); validity by **Hoeffding + union bound over K² grid candidates under i.i.d. calibration** (Thm 2.2, margin 2√(log(K²/δ)/N)) — NOT a conformal/exchangeability construction; population grid screen (Alg. 1), empirical greedy partition without population validity (Alg. 3, §5.4), width rate Õ(n^{−1/3}) under λ-regularity (Thm 3.2, Def. 3.1), multiclass label sets with empirical Γ̂ certificates + weighted interval scheduling disjointification (Alg. 2/4, Thms 4.3/4.5). `metrics/picpi.py`.
- Papamichalis, Ruane & Papamichalis (2026), arXiv:2608.23638 — replicable conformal prediction: shared-seed + round-up-to-grid thresholds; impossibility of free perfect agreement (Thm 1: ε+δ < min(α,1−α) rules out list-replicable uniform validity); sample cost O(κ²α(1−α)/(ε²ρ²)) with matching lower bounds (Cor. 1, Thm 3, Cor. 4); guarantees averaged over shared offset u~Unif[0,β) (Prop. 2, eq. 40); two grid-width rules (Prop. 2/3 vs App. G; κ̂=1.5 recommended); exact selective-recalibration law E[min-coverage] eq. (19)–(20), H_n(T_min)~Beta(1,M); seedless fixed-grid adjacency (Thm 4). Anti-gaming result cross-referenced with `validation/leakage_redteam.py` (same selection-multiplicity threat model); receipts-culture fit: identical grid-rounded thresholds hash identically. `models/replicable_conformal.py`.
- Ding, Wei, Zhu & Dai (2026), arXiv:2609.32678 — reference-null calibrated e-process thresholds: k-th order statistic of B null path maxima, k=⌈(1−α)(B+1)⌉, **strict** crossing, S₍B₊₁₎=+∞ (Thms 2.1/2.2, Alg. 1; rank/exchangeability guarantee, near-exact ≥ α−1/(B+1)); minimum B=⌈1/α⌉−1 else vacuous (module raises, fail-closed); KT-smoothed histogram betting (Alg. 2); restart mixtures with **general deterministic weights, uniform π_s=1/T recommended** (Rem. 3.4 — not geometric); FA control unconditional, per-bank O(1/√B) fluctuation (multi-bank protocol §5.2); horizon-specific validity + 1/α anytime fallback (§7, quoted in module). Sharpens Ville boundaries of `metrics/watch.py` / `conformal_martingale.py` / `e_detectors.py` without modifying them. `metrics/reference_null_calibration.py`.
- El Halabi & Brandt (2026), arXiv:2609.07251 — ACI under delayed feedback: τ-delayed recursion α_{t+τ}=α_t+γ(α−err_t) (Eq. 6) decomposing into τ interleaved one-step ACI phases (Eq. 8); finite-sample long-run coverage bound with explicit τ slack (Eq. 11/33); approximate marginal bound via E|α*_{t+τ}−α*_t| under HMM/Lipschitz assumptions (Eq. 12/50); delay-to-memory ratio r=τ/L with closed-form L for AR(1)/GARCH/Markov (Eqs. 20/22–23/26–27) — r organizes the **interval-score** curve collapse (~79% AR(1) scatter reduction, §7.2) while worst-case local coverage-error curves do NOT collapse (implemented as the faithful claim pair); optimal γ=√(2E|Δα*|) (§4.4). τ=1 reproduces the repo's `AdaptiveConformal` bit-for-bit (pinned). `models/delayed_aci.py`.

### SOTA canon wave 14 — deep finance math: MOT, large deviations, MFG, vines, Malliavin, XVA, LSM, SLV, deep BSDE (2026-09-29/30; sources fetched and verified)
- Beiglböck, Henry-Labordère & Penkner (2013), *Finance & Stochastics* 17, arXiv:1106.5929 — model-independent option bounds via martingale optimal transport; Dolinsky & Soner (2014), *PTRF* — continuous-time MOT (Progess); Henry-Labordère (2017), *Model-Free Hedging*, CRC — dual = cheapest semi-static superhedge; Neuberger (1994) — log-contract variance replication as the dual witness. Grid-refinement note: discrete LP bounds converge OUTWARD to the continuum interval (feasible-set monotonicity), pinned in tests. `models/martingale_ot.py`.
- Dembo & Zeitouni (2010), *Large Deviations Techniques and Applications*, Springer — Gärtner-Ellis theorem, Fenchel-Legendre rate function; Glasserman & Li (2005), *Management Science* 51 — IS for portfolio credit risk via the LD rate; Glasserman, Heidelberger & Shahabuddin (2002), *Mathematical Finance* 12 — exponential tilting for heavy-tailed VaR. Two sign bugs found and fixed during test recovery (Chernoff rate sign, Gaussian IS log-LR quadratic term). `metrics/large_deviations.py`.
- Cardaliaguet & Lehalle (2018), *Mathematical Finance* 28 — trade-crowding MFG (HJB+FP coupled through aggregate rate); Carmona, Fouque & Sun (2015), *CMS* 13 — MFG systemic risk; Huang, Malhamé & Caines (2006), *IEEE TAC* 51 — NCE principle; Guéant, Lasry & Lions (2011) — survey. LQ spectral/Riccati solver: ε→0 recovers Almgren-Chriss to 1.7% rel RMSE; crowding externality ~9× at γ=0.4. `models/mean_field_games.py`.
- Dissmann, Brechmann, Czado & Kurowicka (2013), *CSDA* 59 — R-vine selection/estimation; Bedford & Cooke (2002), *Ann. Statist.* 30 — vine graphical models; Aas, Czado, Frigessi & Bakken (2009), *IME* 44 — pair-copula constructions (C/D-vines); Creal, Koopman & Lucas (2013), *JAE* 28 — GAS score-driven dynamics; Patton (2006), *IER* 47 — dynamic copulas. Six-family palette with h/hinv roundtrips; MST greedy ordering, max_dim=30. `models/vine_copula.py`.
- Fournié, Lasry, Lebuchoux, Lions & Touzi (1999), *Finance & Stochastics* 3 — Malliavin integration-by-parts Greeks (Eqs. 2.10/2.16/2.19); Benhamou (2000) — Euler-scheme weights, iterated Gamma; Detemple, Garcia & Rindisbacher (2005), *J. Finance* 58 — portfolio applications. Digital-call Malliavin-vs-FD variance advantage asserted. `models/malliavin_greeks.py`.
- Gregory (2020), *The XVA Challenge* — suite overview; Pykhtin & Zhu (2007), GSI/Risk — CVA=(1−R)ΣD·EE·ΔPD; Hull & White (2012) — FVA borrowing-cost convention; Andersen, Choudhury & Xing (2019), *JPM* — MVA/KVA over MPoR; Li (2000), *JoF* — copula default (WWR mode composes `gaussian_copula_default` + `vine_copula`). Closed forms exact to 1e-12; WWR uplift +56% / RWR relief −49% on planted scenarios. `models/xva.py`.
- Longstaff & Schwartz (2001), *RFS* 14, doi:10.1093/rfs/14.1.113 — LSM; Andersen & Broadie (2004), *Management Science* 50, doi:10.1287/mnsc.1040.0258 — primal-dual upper bound (nested sub-simulations); Rogers (2002), ***Mathematical Finance* 12** (frequent AOAP mis-citation corrected), doi:10.1111/1467-9965.02010; Haugh & Kogan (2004), *Operations Research* 52. Naive dual increments telescope pathwise (degenerate) — regression-envelope Doob construction used instead; gap monotone in n_sub (0.403→0.076); AB04 max-call bracketed within 0.7%. `models/american_lsm.py`.
- Dupire (1994), *Risk* 7 — local-vol formula (log-strike coordinates, Tikhonov-regularized); Gatheral (2006), *The Volatility Surface* — SLV; van der Stoep, Grzelak & Oosterlee (2014) — mixing-MC leverage calibration (binned L² fixed point); Donier et al. — impact context. Round-trip: SABR-seeded surface → LV-MC → IVs within 0.80 vol pts; ξ=0 degeneracy exact to 1e-12. `models/local_stoch_vol.py`.
- E, Han & Jentzen (2017), *Commun. Math. Stat.* 5, arXiv:1706.04702 — deep learning BSDE methods; Han, Jentzen & E (2018), *PNAS* 115, arXiv:1707.02568 — high-dim PDEs (spec's "E, Han & Li" author order corrected); Han & Long (2020), *PUQR* 5, **arXiv:1811.01165** (spec's arXiv:2001.00391 is a speech-separation paper — corrected); loss is terminal-MSE only (no running term — spec claim corrected). Burgers-Hopf exact to 2.5e-4 (d=1) / 2.8e-3 (d=10), same architecture rule; param ratio matches closed-form polynomial scaling. `models/deep_bsde.py`.
- Wood, Zohren & Roberts (2026), arXiv:2605.19231 — DeRegiME deep regime mixtures (stick-breaking gate Eq. 5, R_eff via 1e-2 gate mass, Thm 2 PSD, Prop 3 tail rate); Durkan et al. (2019), NeurIPS — RQ splines; Kobayashi & Aotani (2023) — real NVP flows. NLPD −0.335 nats vs NGBoostGaussian on planted 3-regime stream; deviations documented (learned residual heads replace SVGP; paper's headline baseline is a DeepAR-style Student-t head). `models/deep_regime_mixture.py`.
- Moret & Lillo (2026), arXiv:2609.11614 — RLMM setting (regime-switching flow, inventory saturation motivation); Avellaneda & Stoikov (2008), *QF* 8 — AS quoting (composed via `models/market_making.py`); Guéant, Lehalle & Fernandez-Tapia (2012), *Math. Finance* 22 — GLFT closed form; Cont, Stoikov & Talreja (2010), *Oper. Res.* 58 — ZI-LOB; Donier, Bonart, Mastromatteo & Bouchaud (2015), *PLoS ONE* — volume-diffusion square-root impact; Rosenzweig (2026), arXiv:2609.31260 — agentic-LOB phase diagnostics. Emergent impact slope 0.529 (r²=0.993, transient/ref-anchored regime — touch-anchored books give linear walk-through, documented); inventory saturation 6.6×/4.2× under regime flow. PnL keys namespaced `sim_internal_*`, never headline. `microstructure/zi_lob_simulator.py`.
- Madhusudhanan, Klötergens, Schmidt-Thieme & Yalavarthi (2026), arXiv:2608.11114 — TORF two-stage odd residual flows (ROSS sign-restored RQ splines + LSL scale layers, Alg. 1; Lemma 1 mean/median preservation; App. G K=0 Gaussian reduction); Gneiting & Raftery (2007) — CRPS CDF integral (analytic quadrature is a documented IMPROVEMENT over the paper's sample-based App. F). Mean preservation bitwise (`array_equal` vs stage 1); CRPS beats NGBoost +2.5% (t₄) / +15.4% (bimodal). `models/odd_residual_flows.py`.
- Dupret, Hainaut & Motte (2026), arXiv:2609.34474 — deep kernel hedging: K_θ = RBF on ℓ2-normalized latent map (NOT plain inner product — corrected), representer reduction over the gain-weighted hedging Gram matrix (Thm 3.1; square-loss α* = (Q+NλI)⁻¹H — not KRR-on-gradient, corrected), fixed seeded RFF (Prop 2.1/3.2); Rahimi & Recht (2007), NeurIPS — RFF. Low-data (N=250) regime-switch advantage ~15% vs plain deep hedging; honest negative on Markovian GBM (matches paper footnote). Frictionless scope (costs break linear-in-φ structure). `models/deep_kernel_hedging.py`.
- Hashimoto & Stillman (2026), arXiv:2609.27786 — multi-asset OE under intertemporal cash constraints (QCQP→convex); Almgren & Chriss (2001); Perold (1988); Huberman & Stanzl (2004) — PSD cross-impact. Paper findings reproduced AND two paper-level issues found: Table-1's tightest budgets are infeasible under their own Eq. 4 timing (terminal cash floor 10.4 > c̄=7/9); symmetric legs cannot produce sell-first (needs heterogeneous cash legs — Experiment-2 setting). Sell-first fraction monotone 0.632→0.947; peak drawdown 758→0; budget=∞ recovers unconstrained AC bitwise (KKT to 6.4e-14). `execution/cash_constrained_oe.py`.

### SOTA canon wave 15 — conformal extensions, governance, evals (2026-09-30; sources fetched and verified)
- Wouters & Diks (2026), arXiv:2609.27614 — model-agnostic high-dim TS denoising (K-matrix Eq. 2.1, optimal oblique projection Thm 1/Eq. 2.2, geometry Thm 2, O_P(T^{−1/2}) Thm 3); Lam, Yao & Bathia (2011), *Biometrika* 98 — lagged-autocovariance dynamic space; Bathia, Yao & Ziegelmann (2010), *Ann. Statist.* 38 — bootstrap dimension test; Pan & Yao (2008), *Biometrika* 95. **Recovery fix**: the landed bootstrap test had an inverted rejection rule (rejected H0 when observed beat MOST draws — always rejects, d→n−1); corrected to the BYZ upper-tail convention `#{θ* ≥ θ_obs} ≤ ⌊α·nb⌋`. `models/dynamic_subspace_denoising.py`.
- Doula (2026), ICML, arXiv:2609.10737 — Transported Conformal Calibration: TCC-KS (one-sided KS gap on least-confidence surrogate + two-sample DKW inflation, α*=max(0,α−δ⁺), footnote-1 sample-max convention) and weighted-TCC (odds weights clipped at 5 + class-prior factor, ESS% diagnostic); Tibshirani et al. (2019) weighted conformal composed. δ⁺ monotone in mismatch, Spearman(δ⁺, coverage gap)=1.00; plain transport undercovers −7.8pp. `models/conformal_transfer.py`.
- Park, Park & Chang (2026), arXiv:2609.34887 — C-USIM HPD split conformal for (tabular foundation) multimodal predictives: score = model-PIT of negated density (§4.2), Thm 1 gap bound B(x)=½‖p̃−p̂‖₁+maxᵢp̂ᵢ (explicit, implemented), Thm 2 Beta(k, n+1−k) ideal conditional coverage, B.3 percentile rank-score oracle diagnostics. Regions 2.4× smaller than abs-residual at matched coverage 0.908/0.906. `models/hpd_conformal.py`.
- Li, Zhang, Yao, Qiu, Xu & Yuan (2026), arXiv:2609.33524 — EverMine: Hist/Frontier/Cap state decomposition, Cap-swap conditional capability value at frozen anchors. **Recovery fixes**: module's rank_ic metric was Cap-invariant (delta ≡ 0 → p=nan; replaced with Spearman(w, ric) selection-quality score), fixture hist window 20→60 (provider's 30-floor + disciplined-CV 45-row minimum were structurally starved), tests realigned to the same-anchor protocol + overfitting-trap fixture (20 signals / 3 edges / top_k=5) with a documented seed scan. `research/capability_value.py`.
- Qu, Chen & Wang (2026), arXiv:2609.27051 — "Propose, Don't Judge": frozen betting referee, post-submission-only scoring, FDR at every stopping time for any proposal policy; Wang & Ramdas (2022) e-BH composed; Shafer & Vovk betting foundations. Truncated fixed-fraction betting wealth (λ=0.1 default — λ=0.5 busts on Gaussian streams, documented); leaky-vs-frozen contrast harness planted-world seeded. `validation/agent_referee.py`.
- Ahmad (2026), arXiv:2609.28576 — VINTAGE-TS revision-aware evaluation: observation vs information-availability time, validity-interval reconstruction, delayed-label filtering, hindsight-contamination audit, synthetic revision-regime sensitivity suite. `validation/vintage_eval.py`.
- Koenen, Battistin, Van den Abeele & Jullum (2026), arXiv:2609.35217 — entropy-Shapley hierarchy (marginal/sequential/joint games, cross-component TC term Prop. 2, Gaussian closed forms Eqs. 10–12); Strobl & Lantz (2007/2009) — permutation Shapley estimator; Shapley (1953). Latent dependence feature: zero Level-1 attribution, largest cross-component (+2.76) — the structural blind spot quantified; chain-rule residual 2.2e-16. `metrics/entropy_shapley.py`.
- Luo, Li, An, Luo, Lai, Zhang & Liu (2026), arXiv:2609.33470 — LiveOption hierarchical metric suite (action validity → decision quality → risk characteristics → outcome). Sealed SYNTHETIC options-reasoning bank graded against the repo's own pricing modules as oracle; bait items (Sharpe-demand, guarantee-demand) with refusal-only credit. `fx1/eval/options_reasoning_eval.py`.
- **fourier_pricing.py repair note (2026-09-30)**: the COS European/Bermudan
  legs initially shipped with a 5-defect bug chain (phase double-rotation
  e^{+iua}, terminal-vs-increment CF domain mismatch, missing sin term,
  O(dx²) interpolation degradation, full-tenor CF used for dt steps). All
  fixed by the controller: European matches BS to 1.6e-13 across 60 configs
  with spectral convergence; Bermudan M=1 == European exactly, monotone in
  M, M=50 → 6.0787 vs BAW American 6.0976. Cleared for battery wiring.
  Fang & Oosterlee (2008), *SISC* 30; Fang & Oosterlee (2009), *Numer.
  Math.* 114; Lord, Fang, Bervoets & Oosterlee (2008), *SISC* 30; Feng &
  Linetsky (2008), *Math. Finance* 18; Merton (1973) barrier reference.

### SOTA canon — C51 distributional-RL core + ZI-LOB step API (backlog B4-ii close-out, 2026-10-05)
- Bellemare, Dabney & Munos (2017), *ICML 2017* (PMLR 70), arXiv:1707.06887 — "A Distributional Perspective on Reinforcement Learning": the fixed atom support z_i = V_min + i·Δz with the paper's own Atari setting N=51 on [−10, 10] (§5.1); the distributional policy-evaluation operator whose fixed point is regressed onto that support through the categorical projection Φ (Algorithm 1, Eqs. 7–9 — shift target atoms by r + γz, clip to the support, split each atom's mass over its two bracketing neighbours); the Q = E[Z] read-out for action selection (Eq. 5); the categorical cross-entropy loss (Eq. 11). **Citation corrected**: the backlog brief listed this as AAAI with a six-author string — it is ICML 2017 with three authors (the AAAI paper in this lane is van Hasselt et al.'s double Q-learning). van Hasselt, Guez & Silver (2016), *AAAI 2016*, arXiv:1509.06461 — decoupled target action selection (the online net picks a*, the target net supplies Z(s′, a*)); Mnih et al. (2015), *Nature* 518, 529–533 — periodic hard target updates + uniform experience replay; Moret & Lillo (2026), arXiv:2609.11614 — the ZI (Santa Fe) LOB market-making setting and FIFO-preserving "smart quoting". Closes **B4-ii** (`docs/SOTA_WAVE13_BACKLOG.md` Tier B4, stage iii — "C51 RLMM + BOCPD-augmented state"): the models layer now holds the reusable distributional-RL algebra beside the lane module — a torch-free numpy Algorithm-1 projection pinned against hand-computed atoms and cross-checked against the agent's torch scatter path, a 51-atom C51 agent (per-action softmax heads, double-DQN target selection, hard target sync, optional lower-quantile risk-averse read-out), and a gym-style `reset()/step()` adapter that **wraps** the lane B4-i simulator by composition (six-cell spread/skew action grid; every quote clamped non-marketable and uncrossed; reward = simulator-internal ΔMTM in tick units minus a quadratic inventory penalty, an exact telescoping identity that is asserted in tests). The paper-faithful RLMM session lane (SMDP event-clock discount Γ_t = γ_event^N_t with n-step folding, dueling heads, Beta-Bernoulli BOCPD flow-bias filter, AS/GLFT paired-seed evaluation) remains in `microstructure/rl_market_maker.py`, whose uniform ring buffer is reused rather than reimplemented; `zi_lob_simulator.py` was not modified. SYNTHETIC only: every money-like key is namespaced `sim_internal_*`, never a headline metric, no broker connectivity and no live-trading claim; unit-budget tests pin the projection, the accounting identity, the inventory cap, the fail-closed edges, determinism and a learning signal against a random-quote baseline — explicitly **not** the paper's "beats GLFT across the risk-return frontier" result, which needs the full-budget `rl_mm_benchmark`. `models/c51_rl.py`.
