# SOTA canon roadmap — September 2026 wave

Research-grounded gap analysis and implementation plan. Sources: 7-agent
swarm (4 repo recon lanes + 3 web-research lanes, 2026-09-27). This doc
complements — does not replace — `SOTA_GAP_ANALYSIS.md` (day-wave ledger)
and `REPO_IMPROVEMENT_PLAN.md` (economic-evidence sequence).

## 1. Verdict up front

The harness is already at or near SOTA across most of the validated-inference
stack: White RC / SPA / StepM / MCS, DSR/PSR/PBO, purged/embargoed CPCV,
BH/BY/Storey FDR, full conformal suite (CQR, CV+/Jackknife+, weighted,
localized, CRC, online CRC/ACI, conformal top-k), e-processes for
DM/loss-differential tests, PIT/pinball/CRPS/QLIKE first-class scoring,
Kupiec/Christoffersen/Acerbi–Székely backtests, ~330 research modules,
~4,900 unit tests. The remaining SOTA gaps are specific and named, not
generic. They are listed below with citations, ranked by leverage.

## 2. Gap matrix (researched method × in-tree status)

### 2.1 Anytime-valid / sequential inference — PARTIAL

| Method | Citation | Status |
|---|---|---|
| E-processes (single hypotheses, betting) | Shafer–Vovk; Ramdas et al. 2023 (arXiv:2210.01948) | EXISTS `metrics/evalues.py` |
| **e-BH** (FDR on e-values) | Wang & Ramdas, JRSS-B 2022 | EXISTS `metrics/anytime_fdr.py` |
| **Stopped e-BH** (anytime-valid FDR under optional stopping) | Wang & Ramdas 2025 (arXiv:2502.08539) | EXISTS `metrics/anytime_fdr.py` |
| **Online FDR with e-values** (e-LOND / e-LORD / e-SAFFRON) | Xu & Ramdas 2024; Zhang et al. 2025 (e-GAI) | EXISTS `ELond`, `ELord`, `ESaffron` in `metrics/anytime_fdr.py` |
| **E-detectors** (anytime-valid changepoint alarms) | Shin, Ramdas, Rinaldo 2023 (arXiv:2203.03532) | EXISTS `metrics/e_detectors.py` |
| Conformal test martingales (WATCH) | Prinster et al. 2025 (arXiv:2505.04608) | EXISTS `metrics/watch.py` (practical Gaussian plugin; post-adaptation alarms diagnostic) |

Leverage: the `TrialLedger` runs RC/SPA/StepM/MCS per research day but has
no FDR control valid under **optional stopping** — exactly what a sequential
research grind does. E-detectors give alpha-decay/regime alarms with
time-uniform error control; natural fit for `monitoring/` and the paper loop.

### 2.2 Distributional forecasting / calibration — NEAR-SOTA, specific holes

| Method | Citation | Status |
|---|---|---|
| CRPS-from-quantiles / energy-consistent scoring | Gneiting & Raftery 2007 | EXISTS `metrics/scoring.py` |
| **Energy score** (multivariate proper score) | Székely 2003; Gneiting & Raftery 2007 | EXISTS `metrics/energy_score.py` |
| Post-hoc distributional calibration (variance scaling, quantile mapping, isotonic-on-quantiles) | Gneiting et al. 2007; calibration literature | EXISTS `models/posthoc_calibration.py` |
| **AgACI / FACI** (multi-expert ACI) | Zaffran et al., ICML 2022 (arXiv:2202.07282) | EXISTS `models/agaci.py` |
| **NGBoost-lite / QRF** (tree-based distributional boosting) | Duan et al., ICML 2020; Meinshausen 2006 | EXISTS `models/ngboost_lite.py`, `models/qrf.py` |
| EnbPI (residual-bootstrap TS conformal) | Xu & Xie, TPAMI 2023 | EXISTS `models/enbpi.py` |

### 2.3 Search-aware evaluation (LLM-era) — STRATEGIC GAP

Gençay 2026 (arXiv:2608.27734): a deliberately leaky oracle with Sharpe 35
**survives Deflated Sharpe and PBO completely**. Fixes: structural
look-ahead exclusion (registry-validated feature space) + search-trial-count
deflation. fx-1 is exactly the LLM-search setting this targets. Plan: add a
synthetic leaky-feature strategy to the promotion battery + trial-count
deflation threshold next to `validate_candidate()`. (Own wave; touches
`hedge_lab/promotion.py` and `research/agent.py`.)

### 2.4 Engineering lane (from research-practices sweep)

Prioritized: (1) dataframe validation contracts at bronze→silver→gold
boundaries (pandera+polars — new dep, needs decision); (2) benchmark/perf
regression gate in CI; (3) GitHub Artifact Attestations on wheels (minutes);
(4) chaos/data-fault injection tests; (5) Hypothesis CI profile pinning
(`derandomize`/registered profile — cheap); (6) mutation testing on
`research/verify.py` + `catalog.py`; (7) harness-side model cards;
(8) verifier re-check CI step. See the engineering section of the swarm
report for citations (Pandera/GE landscape, pybroker asv pattern, SLSA).

### 2.5 fx-1 capability lane

fx-1 today is a rigorous integrity/compliance harness around a not-yet-trained
K3 LoRA; eval proves the model refuses to lie, not that it can reason.
The capability battery now exists and is wired
(`fx1.eval.capability.run_capability_eval`, `fx1 capability-eval`):
seeded SYNTHETIC banks for tool-use/agentic multi-step calls, retrieval with
`[doc_id]` citations, time-series reasoning (process id, pinball, coverage),
and calibrated uncertainty (ECE + Spiegelhalter Z, gate at 0.02). Remaining
gap vs 2024–2026 SOTA: external benchmark integration (FinTrace,
FinToolBench, FinAgentBench, FinanceBench, MTBench) and real-model scores —
the sealed banks are correctness gates, not market evidence.

### 2.6 Standing data gaps (unchanged, disclosed)

No PIT factor panel (factor covariance unwired); no intraday tape (all RV
is daily-bar fallback, northset sessions are synthetic reconstructions);
no true as-of vintages on the public tape; static sector map. These are
data-procurement items, not code items.

## 3. Wave plan

- **Wave 8 (this wave, 2026-09-27): anytime-valid inference + distributional
  calibration + multivariate scores.**
  - `metrics/anytime_fdr.py` — e-BH, stopped e-BH, online e-FDR (e-LOND style).
  - `metrics/e_detectors.py` — Shin–Ramdas–Rinaldo e-detectors (Gaussian,
    bounded, Bernoulli) with anytime-valid alarm control.
  - `metrics/energy_score.py` — multivariate energy score (+ optional
    threshold weighting).
  - `models/posthoc_calibration.py` — distributional recalibrators
    (Gaussian/t variance scaling, quantile mapping, isotonic-on-quantiles).
  - `models/agaci.py` — FACI aggregated adaptive conformal inference.
- **Wave 9 (2026-09-27): time-series conformal + tree/boosting distributional
  baselines + conformal change monitoring.**
  - `models/enbpi.py` — Xu & Xie EnbPI (block-bootstrap ensemble, OOB/LOO
    residuals, sliding residual window, batch online updates).
  - `models/qrf.py` — Meinshausen quantile regression forest (leaf-weight
    conditional CDF; all-tree or OOB weights; quantiles, CDF, PIT).
  - `models/ngboost_lite.py` — NGBoost-lite natural-gradient Gaussian boosting
    (log score / CRPS, line search, validation early stopping).
  - `metrics/conformal_martingale.py` — smoothed/weighted conformal p-values,
    power/mixture/Simple-Jumper test martingales, Ville alarm, WATCH-style
    reset monitor.
  - `metrics/watch.py` — covariate-aware WATCH extension. Randomized online ties;
    the practical frozen-bag Gaussian plugin is a diagnostic, not a proven
    anytime-valid alarm after adaptation.
- **Wave 10 candidates:** leaky-oracle red-team protocol; regime-conditional
  evaluation gate in `validation/gates.py`; QRF/NGBoost vs. existing quantile
  baselines on the research catalog (proper-score comparison only).
- **Engineering waves:** hypothesis CI profile pinning; chaos/fault-injection
  property tests; artifact attestations; mutation testing on the verify
  layer. Done: benchmark regression gate (`.github/workflows/perf-baseline.yml`
  — CI-runner baselines as artifacts, calibration-normalized compare on PRs).
- **fx-1 waves:** capability-side eval (tool-use, retrieval, TS-reasoning,
  calibration measurement) — done: `eval/capability.py` aggregate +
  `fx1 capability-eval` CLI, honesty gates + calibration gate hard.
  Next: external-benchmark ports (FinTrace/FinToolBench/MTBench).

## 4. Acceptance for this wave

Each module: citation-style docstrings; pure numpy/scipy/pandas (no new
deps); fail-closed edges (short/empty inputs raise, never silently return);
unit tests with simulated size/power assertions (FDR ≤ α under global null;
coverage under drift; CRPS non-worsening on calibration); ruff + ruff-format
+ mypy clean on touched files; no commits (repo convention: owner reviews).

## 5. Status log

- **Waves 8–10: LANDED + COMMITTED** (lane commits through b508108c), wired
  into the research battery as six OPTIONAL scorecard families
  (`benches_w810.py`); `verify-research` green at 32 families, errors [].
  The lane additionally extended `anytime_fdr` with e-LORD/e-SAFFRON
  (Zhang et al. 2025), wired `perf_baseline` into a CI perf gate
  (`.github/workflows/perf-baseline.yml`), and wired the fx-1 capability
  evals into the suite/CLI (`fx1/eval/capability.py`).
- **Wave 11: LANDED 2026-09-28** — rough-path signatures
  (`models/path_signatures.py`), Wasserstein/OT metrics
  (`metrics/wasserstein.py`), Wasserstein-DRO portfolios
  (`portfolio/wasserstein_dro.py`), conformal PID control
  (`models/conformal_pid.py`); 84 module tests + wired as four OPTIONAL
  scorecard families (`benches_w11.py`, 10 bench tests). Citations in
  RESEARCH_REFERENCES.md.
- **Wave 12: LANDED 2026-09-29** — nine parallel lanes; all four wave-11
  follow-ups consumed: NexCP beyond-exchangeability conformal + Theorem 2/3
  coverage bounds (`models/nexcp.py`; citation corrected to Barber, Candès,
  Ramdas & Tibshirani 2023, arXiv:2202.13415 — no Liu/Wang/Xie NexCP exists),
  multi-horizon EnbPI with Bonferroni joint bounds
  (`models/enbpi_multihorizon.py`, H=1 reproduces EnbPI bit-for-bit),
  regime-weighted conformal VaR (`models/regime_conformal_var.py`, exact
  reduction to unweighted split conformal under uniform posteriors), and
  torch-gated deep hedging (`models/deep_hedging.py`, Buehler et al. 2019).
  Plus: anytime-valid confidence sequences (`metrics/confidence_sequences.py`
  — Howard et al. 2021 mixture/stitching/hedge-ε + Waudby-Smith & Ramdas WSR
  + empirical-Bernstein), sliced/max-sliced Wasserstein with permutation GoF
  test (`metrics/sliced_wasserstein.py`), proper-score decomposition canon
  (`metrics/score_decomposition.py` — Bröcker 2012, Kolassa 2016 [IJF, not
  JRSS-A — corrected], exact Brier/RPS/spherical), Bayesian stacking +
  pseudo-BMA(+) (`models/stacking.py`, Yao et al. 2018), fx-1
  external-benchmark adapter ports with sealed SYNTHETIC banks
  (`fx1/eval/ext_bench*.py` — MT-Bench/FinanceBench/FinToolBench formats;
  capability.py/CLI wiring is an owner call), and a stdlib AST
  mutation-testing harness (`scripts/mutation_test.py`; first e2e run on
  `research/verify.py` scored 95% — 19/20 killed — and its sole survivor,
  the untested `p_value=0.0` boundary in `_p_value_valid`, was closed with a
  boundary test). ~372 lane tests green; ruff/format/mypy clean throughout;
  no new deps (deep hedging uses the existing torch `nn` extra).
- **Wave 13+ backlog: RESEARCHED 2026-09-29** — see
  `SOTA_WAVE13_BACKLOG.md` (live arXiv sweep; tiers: hardest conformal/UQ
  theory [coverage-inference CLT under temporal dependence arXiv:2609.33868,
  minimax-optimal conformal change detection arXiv:2609.27179, rolling CP
  arXiv:2609.26951, rank confidence sequences arXiv:2609.32211], heaviest
  torch lanes [deep kernel hedging arXiv:2609.34474, diffusion/flow
  forecasters, DeRegiME, ZI-LOB distributional RL], and agent-governance
  strategy [anytime-valid referee arXiv:2609.27051, EverMine-style
  capability-value accounting arXiv:2609.33524, vintage-aware evaluation
  arXiv:2609.28576]).
- **Wave 13: LANDED + INTEGRATED 2026-09-30** — Tier A complete: all eight
  theory modules (conformal coverage inference under temporal dependence,
  minimax-optimal conformal e-detectors [163× delay gap vs Vovk CTM at
  matched PFA], rolling CP [factor-two floor attained by the Prop.-1
  construction], rank confidence sequences + BB-EDGE, PICPIs, replicable
  conformal [exact-theory reproductions to 1e-4], reference-null e-threshold
  calibration [11.7% RMDD gain], delayed-feedback ACI [τ=1 bit-for-bit vs
  repo ACI]). 382 lane tests; wired as eight OPTIONAL families
  (`benches_w13.py`, 18 bench tests, battery 11.0s contended / 2.9s
  unloaded); `verify-research` → **52 families, errors []**. ~20 paper-vs-
  brief corrections recorded in RESEARCH_REFERENCES.md (fetch-verify-first
  protocol).
- **Wave 14: LANDED 2026-09-30** — deep finance math: martingale OT
  (model-free bounds, LP duality), large deviations (Gärtner-Ellis + IS for
  VaR; 2 sign bugs caught in recovery), mean field games (Cardaliaguet-
  Lehalle Riccati solver, ε→0 AC recovery 1.7%), vine + GAS copulas (48
  tests), Fourier COS/CONV/Hilbert (COS European leg has a documented open
  bug — excluded from battery), Malliavin Greeks (digital FD-efficiency
  result), XVA suite (closed forms exact to 1e-12, WWR +56%), LSM +
  Andersen-Broadie dual (gap 0.403→0.076 monotone; Rogers 2002 venue
  mis-citation corrected), Dupire/SLV (round-trip ≤0.80 vol pts), deep BSDE
  (Burgers d=10 to 2.8e-3; 4 citation corrections incl. a wrong arXiv id),
  DeRegiME, ZI-LOB simulator (emergent √-impact 0.529, r²=0.993; inventory
  saturation 6.6× — RL layer deferred), TORF odd residual flows (bitwise
  mean preservation), deep kernel hedging (15% low-data gain; honest GBM
  negative), cash-constrained OE (2 paper-level findings: infeasible Table-1
  budgets, symmetric-legs sell-first impossibility). Integration in flight
  (`benches_w14.py`, 14 families → 66 expected).
- **Wave 15: LANDED 2026-09-30** — conformal extensions + governance:
  dynamic-subspace denoising (bootstrap-rule inversion caught in recovery),
  conformal calibration transfer (TCC-KS + weighted-TCC), C-USIM HPD
  conformal (2.4× smaller multimodal regions), EverMine capability-value
  accounting (module Cap-invariance bug + fixture starvation caught in
  recovery; overfitting-trap sensitivity), frozen anytime-valid referee
  (post-submission-only e-values + e-BH; leaky-vs-frozen contrast), vintage-
  aware evaluation, entropy-Shapley uncertainty attribution (chain-rule
  residual 2.2e-16), fx-1 options-reasoning bank (LiveOption hierarchy).
  Integration queued (`benches_w15.py`, 7 families → 73 expected).
- **Process note (waves 14–15)**: two provider-quota saturation events
  killed agents mid-flight; recovery protocol proven — source lands before
  tests, controller writes tests directly, runs gates, and fixes
  module-level bugs found (5 real bugs caught this way: LD signs ×2,
  subspace bootstrap rule, capability_value Cap-invariance + hist
  starvation, fourier COS European [still open]). Ratchet discipline:
  untracked-lane `except Exception` handlers narrowed same-day (vine ×6,
  wasserstein_dro ×1) — count back at exactly 72.
- **Wave 16: LANDED on main 2026-09-30 (parallel session)** — six modules
  wired directly on main as OPTIONAL families (scorecard 74→80): vol
  loss-vs-model decomposition, generalized hierarchical CP, multi-source
  randomly localized conformal, ExTRA-WCP/-WCP-T tilt conformal, forecast
  selection (Soleimani 2609.26303), C51 RL market maker on the ZI-LOB, plus
  DiffPTS forecaster, e-PS, FinAutoRubric (`rubric_eval.py`/`rubric_banks.py`),
  conformal OCE, greek-neutral option portfolios.
- **Wave 17: INTEGRATED 2026-09-30 (this PR)** — eight module lanes + seven
  OPTIONAL families (`benches_w17.py`, scorecard 80→87): passive-impact
  execution with OFI price response (2607.28323; literal `scipy.expm`
  bidiagonal generator after Volterra conditioning documented exponential),
  stochastic tracking / regularized Obizhaeva–Wang (Nutz & Voss
  2608.29468), multilevel market making (2608.18195, torch-gated),
  scenario-bandit robust fine-tuning of C51 (Moret & Lillo 2609.11614),
  generic ExTRA tilt machinery (2609.30886), DiffPTS full-ELBO under LSNM
  (2609.32363, torch-gated), SGA multi-step UQ (2609.28582). One path
  collision vs wave-16 resolved in favor of this wave's
  `metrics/forecast_selection.py` (already on main; both test suites
  green on it). ~442 tests incl. bench wiring; ruff/format/mypy clean.
- **Wave 18: INTEGRATED 2026-09-30 (PR #446, stacked on #442)** — six module
  lanes + six OPTIONAL families (`benches_w18.py`, scorecard 87→93):
  agentic-LOB phase-transition diagnostics (2609.31260, on the w14
  ZI-LOB), FASE self-evolving eval protocol (2609.32689), G-SLiCE path-space
  flow matching (2605.28507, torch-gated), KiT OHLCV pipeline (2609.34507,
  torch lane gated), latent neural SDE forecaster (arXiv:2001.01328 —
  lane-spec ids corrected), StocBench fixed-budget sampler eval
  (2608.22309). 325 lane tests; ruff/format/mypy clean; research suite
  green. Remaining backlog: mutation campaign on verify.py + catalog,
  fx1 capability.py/CLI wiring (owner calls), Tier-B deep-kernel-hedging
  follow-ups.
- **Wave 19: INTEGRATED 2026-09-30 (PR #452, stacked on #446)** — all six
  lanes landed + six OPTIONAL families (`benches_w19.py`, scorecard
  93→99): generalized-Langevin latent-liquidity impact (2609.37872),
  event-time order-flow memory and subordinated observables
  (2609.13715), Fukasawa first-order implied-variance (2609.13961),
  AD-Seq-Vol IVS diffusion + no-arb penalties (2609.13402, torch-gated),
  RCCP retrieval-corrected conformal (2608.10553), DCP distribution-aware
  conformal (2605.26569). 285 lane tests + 16 bench tests;
  ruff/format/mypy clean; research suite green incl. executed-flag
  invariant. Remaining backlog: mutation campaign on verify.py +
  catalog, fx1 capability.py/CLI wiring (owner calls), Tier-B
  follow-ups.

- **Wave 20: INTEGRATED 2026-09-30** — five lanes landed (varswap
  optimal stopping; hidden-Markov equilibrium pricing with the §5.5
  calibration reproduced to printed digits; Gaussian normalized
  coordinates + CDF-deformation arb checks, Sun 2609.14212; liquidity-tail
  LOB equilibrium, Cetin-Lin-Livieri 2607.01198; adversarial-RL market
  making, Yang & Xu 2609.22785 — torch-gated), five OPTIONAL families
  wired (scorecard 99→105). A locally built `metrics/sga_uq` lane was
  dropped: wave-17 `sga_multistep_uq` already canonizes arXiv:2609.28582
  as a strict superset — the dup is recorded so later waves skip it.
  Canon: `devin/w20-canon`. PR #477 stacked on #452.

- **Wave 21: INTEGRATED 2026-09-30** — three lanes landed and wired
  (scorecard 105→108): BOCPD change points (Adams & MacKay 0710.3742),
  rough-vol pricing — fractional Riccati rHeston CF + rBergomi/Volterra
  simulators (El Euch & Rosenbaum 1609.02108; Abi Jaber-Larsson-Pulido
  1708.08796, corrected from the spec's 1708.07719), and signature
  features — Goursat-PDE signature kernel + lead-lag MMD
  (Chevyrev-Oberhauser 1810.10971; Salvi et al 2006.14794, corrected
  from the spec's 2006.14742). Canon: `devin/w21-canon`. PR #479
  stacked on #477. Duplicates recorded for later waves:
  `metrics/multipower_variation` (models/realized already canonizes
  BPV/TPQ/BNS/Lee-Mykland) and Hawkes GOF (hawkes_residuals owns the
  compensator+KS battery). Wave-22 lane signature_martingale_test
  running as a child — integrates on `devin/w22-canon`.
