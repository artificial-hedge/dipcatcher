# SOTA wave 13+ backlog — hardest/heaviest frontier work

Research sweep 2026-09-29 (arXiv API, live queries across stat.ML conformal/UQ,
e-processes/safe-testing, TS foundation models, diffusion/flow forecasting,
q-fin.TR execution/RL). Companion to `SOTA_CANON_ROADMAP_2026_09.md` (waves
8–11 landed; wave 12 in flight: NexCP/multihorizon conformal, deep hedging,
stacking/pseudo-BMA, confidence sequences, sliced Wasserstein, score
decomposition, regime-weighted conformal VaR, fx-1 ext-bench ports, mutation
testing). Ranked by difficulty × leverage × repo-fit. All items must obey the
honesty contract: proper scores only, SYNTHETIC-labeled correctness tests,
no live-trading claims, receipts for anything promotable.

## Tier A — hardest theory (statistical core; numpy/scipy; heaviest proofs to port)

### A1. Conformal coverage inference under temporal dependence — THE flagship
- Zhai, Cheng & Wu 2026, arXiv:2609.33868 ("Conformal Coverage of Time
  Series: Validity and Inference") + companion arXiv:2609.33866 ("Valid and
  Efficient Split Conformal Regression for Time Series").
- What: first CLT for *realized coverage* of split conformal under temporal
  dependence via the functional dependence measure (no mixing assumptions);
  Bahadur representation; consistent block-based SE estimator + asymptotic
  test; long-memory Gaussian linear processes get a non-Gaussian limiting law
  with block-sampling inference and estimated normalization; simultaneous
  non-asymptotic coverage + interval-length accuracy (first for TS), with a
  sharper length-error rate for long memory and a matching lower bound.
- Why hard: porting a Bahadur representation + FDM-based variance estimation
  into runnable, tested numerics; long-memory normalization is genuinely new
  ground. 70+ page theory papers.
- Repo fit: perfect — the repo has ~15 conformal modules but coverage is only
  ever *measured*, never *inferenced*. Adds `coverage_test()` /
  `coverage_se_block()` next to every existing conformal path; connects to
  `metrics/inference.py` stationary bootstrap.
- Scope: `metrics/conformal_coverage_inference.py` — FDM estimator, block SE,
  realized-coverage z-test, long-memory variant. Wave-13 flagship.

### A2. Minimax-optimal conformal change detection
- Bhattacharyya & Ramdas 2026, arXiv:2609.27179 (76 pp).
- What: proves standard conformal test martingales / e-detectors (Vovk 2021 —
  already in `metrics/conformal_martingale.py` and `metrics/e_detectors.py`)
  are *suboptimal*: Ω(T) PFA delay and Ω(√ARL) under changepoints. Proposes
  provably minimax-optimal conformal e-processes with Θ(log T) / Θ(log ARL)
  delay.
- Why hard: new theory that upgrades code already in tree; requires careful
  side-by-side delay/PFA Monte-Carlo to demonstrate the logarithmic vs
  polynomial gap on seeded streams.
- Repo fit: direct upgrade path for two existing modules; changepoint battery
  (`models/changepoint.py`) gives ground-truth changepoints for delay tests.
- Scope: `metrics/conformal_e_detectors.py` (optimal constructions) +
  comparison bench vs existing martingales (delay ratio, PFA at matched ARL).

### A3. Rolling conformal prediction (sequential model training)
- Cheng, Liang & Barber 2026, arXiv:2609.26951 (42 pp).
- What: distribution-free CP where the model at time n may depend on all
  previous data (one-pass training / continual fine-tuning — exactly the
  fx-1 LoRA setting). Calibrate-then-roll; universal factor-two marginal
  guarantee under exchangeability with *no* stability assumptions;
  high-probability training-conditional validity for i.i.d.; sharpening to
  1−α under stability.
- Why hard: the guarantee is subtle (models change every step); honest
  empirical demonstration requires a sequential-training simulator (online
  ridge/SGD refit loop) with exchangeable data.
- Repo fit: bridges conformal suite ↔ fx-1 training lane; `compare_runs`
  could gate on rolling-CP coverage of a continually fine-tuned model.
- Scope: `models/rolling_conformal.py` + online-refit SYNTHETIC harness.

### A4. Anytime-valid rank inference — leaderboards & ranker selection
- Khosravi & Huo 2026, arXiv:2609.32211 ("Rank Confidence Sequences") +
  Gao et al. 2026, arXiv:2609.32248 ("BB-EDGE": benchmark-weighted,
  block-factorized empirical-Bernstein e-processes + e-Holm, anytime FWER,
  top-k certification, simultaneous rank intervals).
- What: rank sets valid simultaneously for all models at all times under
  arbitrary within-item dependence — betting e-processes per ordered pair +
  closed testing over orderings.
- Why hard: closed testing over orderings is combinatorial; needs efficient
  implementation (the papers give the structure; making it scale to the
  repo's 6-ranker × N-date tournaments is real work).
- Repo fit: strategic — `research/agent.py` ranks rankers daily and
  `TrialLedger` does RC/SPA/StepM/MCS; rank confidence sequences make
  *anytime-valid* ranker selection possible for the grind loop, and fx-1
  `compare_runs` gets optional-stopping-safe model ordering. Composes with
  the wave-12 confidence-sequences lane.
- Scope: `metrics/rank_confidence_sequences.py` (+ BB-EDGE-style block
  factorization for multi-benchmark model cards).

### A5. PICPIs — prediction-interval-conditional prediction intervals
- Yang, Huang, Hou, Imbens & Jordan 2026, arXiv:2609.25388 (45 pp).
- What: self-consistency E[Y | p(X) ∈ I] ∈ I; data-adaptive strata without
  altering predictions; n^(−1/3) widths; inference procedures for
  probabilistic + multiclass settings.
- Why hard: heavy asymptotics; the conditioning event is itself random.
- Repo fit: strongest available answer to "marginal coverage is not enough"
  next to `localized_conformal.py`; pairs naturally with A1.
- Scope: `metrics/picpi.py`.

### A6. Replicable conformal prediction
- Papamichalis, Ruane & Papamichalis 2026, arXiv:2608.23638.
- What: shared seed + coarse threshold grid ⇒ independent calibrations
  produce the *identical* prediction set with quantified set-size cost;
  matching lower bounds; blocks gaming via recalibration selection.
- Why hard (moderate): theory is clean; the work is the sample-cost frontier
  experiments + anti-gaming demos.
- Repo fit: receipts/immutability culture — a replicable calibration is an
  auditable calibration; anti-gaming result speaks directly to the
  leaky-oracle red team (`validation/leakage_redteam.py`).
- Scope: `models/replicable_conformal.py` + red-team bench extension
  (selection-over-recalibrations gaming demo).

### A7. Sharper e-process thresholds via reference-null calibration
- Ding, Wei, Zhu & Dai 2026, arXiv:2609.32678 (45 pp).
- What: independent null samples calibrate sharper rejection boundaries than
  Ville's 1/α while preserving type-I control; histogram betting with
  Krichevsky–Trofimov smoothing + restart mixtures for shift detection;
  quantified power/delay gains.
- Repo fit: modular upgrade to `metrics/watch.py`, `conformal_martingale.py`,
  `e_detectors.py` alarm thresholds; pairs with A2.
- Also adjacent: Ramdas 2026 arXiv:2609.35714 (Bell-Cover randomization,
  competitive optimality of betting — theory note, low implementation value);
  Ramdas 2026 arXiv:2609.05752 (complete characterization of sequential
  testability — use as docstring-level grounding for e-detector fail-closed
  behavior); Lin et al. 2026 arXiv:2609.26651 (e-PS: e-value posterior
  sampling for *adaptive data collection* under e-BH — fits TrialLedger's
  budgeted-trial future).
- Scope: `metrics/reference_null_calibration.py`.

### A8. ACI under delayed feedback (bridge to wave-12 multihorizon lane)
- El Halabi & Brandt 2026, arXiv:2609.07251.
- What: τ-delayed ACI decomposes into τ interleaved ACI sequences;
  finite-sample long-run coverage bounds with explicit τ dependence;
  delay-to-memory ratio r = τ/L organizes performance under AR(1).
- Repo fit: the wave-12 multihorizon EnbPI lane creates exactly the
  delayed-feedback setting; this gives the *adaptive* layer on top.
- Scope: extend `models/conformal_pid.py`-adjacent module
  `models/delayed_aci.py` (do NOT merge into conformal_pid; compose).

## Tier B — heaviest ML (torch-gated; deepest engineering)

### B1. Deep kernel hedging — signature features meet RKHS hedging
- Dupret, Hainaut & Motte 2026, arXiv:2609.34474.
- What: hedging functional in an RKHS whose kernel is a learned neural
  embedding; *time-augmented truncated signature features* for path
  dependence; generalized representer theorem reduces training to finite
  dimensions; random-Fourier-feature scaling with convergence guarantees.
- Why hard: fuses three wave-11/12 assets (path signatures, deep hedging,
  kernel methods) + a representer-theorem-driven solver; RFF + deep kernel
  joint training is fiddly.
- Repo fit: the single most synergistic heavy item — directly composes
  `models/path_signatures.py` (wave 11) + `models/deep_hedging.py` (wave 12)
  + `models/kernel.py`.
- Scope: `models/deep_kernel_hedging.py` (torch-gated) + SYNTHETIC
  low-data-regime advantage demo vs both baselines.

### B2. Diffusion / flow-matching probabilistic forecaster (proper-score gated)
- DiffPTS (NeurIPS 2026, arXiv:2609.32363): full-ELBO training under
  location-scale noise model; unifies CSDI/TimeGrad-style paradigms.
- TORF (arXiv:2608.11114): two-stage odd residual normalizing flows —
  mean-preserving, sampling-free CRPS; decouples point accuracy from density.
- G-SLiCE (arXiv:2605.28507): flow matching on path space via controlled
  differential equations; universality theorem; irregular grids.
- KiT (arXiv:2609.34507): K-line diffusion-transformer foundation model —
  conditional *OHLCV path generation* via flow matching (relevant to the
  repo's `candle_order_book` family).
- Why hard: heaviest training-engineering lane in the backlog; ELBO
  correctness, sampler choice (DDPM/DDIM/DPM-2/few-step distillation), and
  honest CRPS/PIT evaluation against the repo's existing NGBoost/QRF/EnbPI
  baselines. StocBench (arXiv:2608.22309) is the sampler-budget study to
  imitate; AutoCast reliability study (arXiv:2606.12997) shows CRPS-trained
  ensembles often beat latent diffusion on coverage — build the comparison,
  don't assume diffusion wins.
- Repo fit: `models/diffusion_index.py`, `nbeats.py`, `dlinear.py`,
  `kronos.py` prove the torch lane exists; proper-score gates (CRPS, PIT,
  energy score) are first-class.
- Scope: staged — (i) TORF-style residual flow head (lightest, sampling-free),
  (ii) DiffPTS-style ELBO diffusion on SYNTHETIC regime-switching data,
  (iii) optional KiT-style OHLCV path generation against the sealed candle
  snapshot. Each stage gated on beating `ngboost_lite`/`qrf` CRPS on the same
  SYNTHETIC streams, labeled correctness evidence only.

### B3. DeRegiME — deep regime mixture-of-experts under distribution shift
- Wood, Zohren & Roberts 2026, arXiv:2605.19231.
- What: sparse variational GP with nonstationary regime-mixing kernel +
  Student-t likelihood; single GP posterior whose gate softly assigns
  forecast locations to learned residual-uncertainty regimes; stick-breaking
  pruning; kernel-validity and propriety proofs; 20.3% NLPD gain over
  DeepAR-style dynamic Student-t heads across ten benchmarks.
- Why hard: sparse VI GP with a custom nonstationary kernel is the heaviest
  single-model implementation in the backlog; the proofs (kernel validity,
  predictive propriety) must be reflected as tested invariants.
- Repo fit: composes `models/regime*.py` (HMM machinery), wave-12
  regime-weighted conformal VaR, and the regime_eval gate; residual-regime
  structure is exactly what `validation/regime_eval.py` stratifies.
- Scope: `models/deep_regime_mixture.py` (torch-gated).

### B4. Distributional-RL market making in a zero-intelligence LOB
- Moret & Lillo 2026, arXiv:2609.11614: Rainbow-style C51 RLMM calibrated
  in a ZI-LOB; beats GLFT across the risk-return frontier under stationarity;
  Bayesian online change-point filter on directional flow + queue-adjusted
  exposure imbalance restore profitability under regime-switching flow;
  scenario-bandit reweighting for stress.
- Also: Cheridito & Weiss 2026, arXiv:2608.18195 (multi-level MM with
  logistic-normal allocations + deep-set encoder + potential-based shaping);
  Rosenzweig 2026, arXiv:2609.31260 (agentic LOB phase transitions &
  non-square-root impact — the heaviest simulator variant).
- Why hard: requires building a ZI-LOB simulator first (the repo has
  `execution/impact.py` + `models/market_making.py` but no event-level LOB);
  distributional RL (C51) training loops; regime-switching flow calibration.
  All SYNTHETIC — and the honesty contract demands it stay that way (no
  live-trading claims; PnL here is simulator-internal diagnostics, never a
  headline).
- Repo fit: `models/deep_rl.py`, `bandits.py`, `changepoint.py` (BOCPD
  already in tree — the paper's flow filter can reuse it verbatim).
- Scope: staged — (i) `execution/lob_simulator.py` (ZI-LOB, seeded,
    calibrated), (ii) GLFT/Avellaneda-Stoikov baselines on it, (iii) C51
    RLMM + BOCPD-augmented state. Heaviest multi-week lane; gate each stage.

### B5. Multi-asset optimal execution under cash constraints + tracking rates
- Hashimoto & Stillman 2026, arXiv:2609.27786: Almgren-Chriss extended with
  intertemporal expected-cash constraints ⇒ QCQP with convex representation;
  sell-first schedule shifts; peak-cash-drawdown reduction in an agent-based
  simulator.
- Nutz & Voss 2026, arXiv:2608.29468: stochastic tracking with sharp
  O(√ε) regularization rates for generalized Obizhaeva-Wang execution.
- Barzykin et al. 2026, arXiv:2607.28323: passive-impact optimal execution
  (exponential fill-probability decay + OFI linear response; FX-calibrated).
- Repo fit: `execution/impact.py` has Perold IS decomposition, POV, VWAP
  slippage — but no constrained multi-asset optimizer and no passive-impact
  model. cvxpy already in tree (wave-11 DRO uses it).
- Scope: `execution/cash_constrained_oe.py` (QCQP via cvxpy) +
  `execution/passive_impact.py`. Moderate-hard; cleaner than B4.

## Tier C — agent/harness SOTA (strategic; the repo's identity)

### C1. Anytime-valid referee for LLM factor mining ("governed self-evolution")
- Qu, Chen & Wang 2026, arXiv:2609.27051 ("Propose, Don't Judge").
- What: an LLM agent proposes factors; a *frozen* betting-based statistical
  referee the agent cannot touch judges candidates only on post-submission
  outcomes ⇒ false-discovery guarantee at every stopping time for any
  proposal policy. Frozen referee admits 5–11× fewer sub-threshold factors
  than leaky referees; certificate costs ~500 trading days of waiting.
- Why hard (harness-heavy): implementing the referee protocol end-to-end —
  submission ledger, post-submission-only scoring, e-process accumulation,
  leaky-referee red-team contrasts — and wiring it as the gate in front of
  `research/agent.py` family promotion.
- Repo fit: THE strategic item. It is this repo's thesis (agent proposes,
  harness disposes) published independently; composes evalues + anytime_fdr +
  leakage_redteam + TrialLedger + receipts.
- Scope: `validation/agent_referee.py` + `research` wiring + SYNTHETIC
  planted-truth world with scripted/bandit/LLM proposer stubs.

### C2. Controlled self-evolution evaluation (EverMine-style)
- Li et al. 2026, arXiv:2609.33524.
- What: decompose research state into History / Frontier (current factor
  portfolio) / Capabilities; measure *conditional value of accumulated
  capability* by Cap-swaps at frozen (Hist, Frontier) states; replay-based
  analysis of experience-driven decisions. Headline finding: capability
  evolution shows no consistent end-to-end gain — a result this repo's
  shadow_journal/forward_shadow machinery is positioned to verify
  independently.
- Repo fit: `research/shadow_journal.py`, `forward_shadow.py`,
  `sota_protocol.py` already track state; adding the Hist/Frontier/Cap
  decomposition + Cap-swap protocol makes self-evolution claims falsifiable.
- Scope: `research/capability_value.py` + replay harness.

### C3. Revision-aware (vintage) forecasting evaluation — VINTAGE-TS
- Ahmad 2026, arXiv:2609.28576.
- What: distinguishes observation time from information-availability time;
  targets = first-published value and value available N days later; joint
  predictive distribution over both; ALFRED-style rolling evaluation;
  pretraining-overlap audit; delayed-label filtering; validity-interval
  reconstruction. Notably honest paper (synthetic demo executed, real
  experiments explicitly *not* claimed) — matches house style.
- Repo fit: hits the standing data gap "no true as-of vintages on the public
  tape" (roadmap §2.6) from the *evaluation* side: even without procuring
  vintage data, the harness can enforce vintage-correct targets and audit
  hindsight contamination. Extends `data/` contracts + `metrics/forecast_eval`.
- Scope: `validation/vintage_eval.py` (validity intervals, delayed-label
  filter, contamination audit) + SYNTHETIC revision-regime suite.

### C4. fx-1 external-benchmark deepening (builds on wave-12 lane 8)
- FinAutoRubric (Lee et al. 2026, arXiv:2609.35744): expert-guided automatic
  rubric generation for financial research agents; writer/reviewer agent loop
  with human escalation; Task Bank of reusable criteria; 100-query benchmark
  across 78 tasks / 8 asset classes. Port the *rubric machinery* (generation
  + review + validation protocol), not the proprietary bank.
- LiveOption (Luo et al. 2026, arXiv:2609.33470): hierarchical metric suite
  for LLM agents in structured option trading — action validity → decision
  quality → risk characteristics → outcome; standardized interaction
  protocol; finding that current agents fail nonlinear-payoff tasks. Port the
  *metric hierarchy* as a SYNTHETIC options-reasoning bank (repo has
  `models/options.py`, `iv_approx.py`, `sabr.py` for ground truth).
- FASE (Wang et al. 2026, arXiv:2609.32689): feedback-aware self-evolving
  forecasting agent with episodic memory + online policy learning over
  delayed feedback — the eval-side twin of C1/C2 for the forecast lane.
- Scope: `fx1/eval/rubric_eval.py`, `fx1/eval/options_reasoning_eval.py`.

### C5. Uncertainty attribution for multivariate predictive outputs
- Koenen, Battistin, Van den Abeele & Jullum 2026, arXiv:2609.35217:
  hierarchy of three entropy-Shapley games (marginal → joint) with a
  cross-component attribution term via conditional total correlation;
  chain-rule decomposition; closed-form + sample estimators; demonstrated on
  distributional regression through zero-shot TSFMs.
- Also: SGA (Hu et al. 2026, arXiv:2609.28582): DAG-based multi-step TSFM
  uncertainty; graph complexity bounds; scaling law (larger TSFM ⇒ lower
  multi-step uncertainty).
- Repo fit: `research/explainability/` exists; multivariate forecast outputs
  (wave-12 multihorizon EnbPI, B2 diffusion paths) need attribution to be
  auditable. Entropy machinery already in `metrics/entropy.py`.
- Scope: `metrics/entropy_shapley.py`.

## Tier D — supporting/adjacent (cheaper, high polish value)

- **C-USIM** (Park et al. 2026, arXiv:2609.34887): highest-predictive-density
  split CP for multimodal tabular foundation-model outputs; conditional-
  marginal coverage-gap bounds; percentile rank-score plots. Relevant once
  any TSFM/foundation-model adapter lands. `models/hpd_conformal.py`.
- **ExTRA conformal under exponential-tilt joint shift** (Choi 2026,
  arXiv:2609.30886): tilt-reweighting calibration + optional predictive
  tilting; documented failure modes (when tilting *hurts*) make it a good
  red-team addition. Extends `weighted_conformal.py`-adjacent module.
- **Conformal calibration transfer** (Doula 2026, ICML 2026,
  arXiv:2609.10737): transport labeled source calibration to target space
  via unlabeled paired data + mismatch correction; finite-sample guarantees
  adapting to observable mismatch. Bridge-domain evals (e.g. synthetic →
  sealed snapshot).
- **Generalized hierarchical CP** (arXiv:2608.15500; Lee et al. HCP 2026):
  group-structured prediction with a few in-group observations. Fits
  cross-sectional panels (per-asset groups).
- **Multi-source localized CP (MS-RLCP)** (arXiv:2609.14531; Hore & Barber
  RLCP 2025): heterogeneous-source conformal with envelope-distribution
  coverage bounds — relevant to multi-vendor data futures.
- **Model-agnostic high-dim TS denoising** (Wouters & Diks 2026,
  arXiv:2609.27614): optimal linear projection onto the dynamic subspace
  under observation noise; lagged-covariance + bootstrap dimension selection;
  parametric-rate consistency. Cheap, composes with `models/rmt.py`
  eigenclip/detone.
- **Forecast selection under a common target** (Soleimani 2026,
  arXiv:2609.26303): alignment/dilution decomposition of standardized
  cross-sectional forecasts; cautious selection rule. Directly relevant to
  `research/ranker_probability.py` + LLM-forecast admission.
- **Volatility loss-vs-model decomposition** (Tokajuk & Chudziak 2026,
  arXiv:2609.27024): validation-based forecast-level alignment separating
  loss-choice from model-choice effects — a clean upgrade to `vol_bench.py`
  comparisons.

## Sequencing recommendation

1. **Wave 13 (theory wave):** A1 + A2 + A6 (+ A7 if capacity) — pure
   numpy/scipy, same acceptance contract as waves 8–12, upgrades the
   repo's densest existing asset (conformal + e-processes).
2. **Wave 14 (agent-governance wave):** C1 + C2 (+ A4) — the strategic
   differentiator; frozen anytime-valid referee over the research grind,
   capability-value accounting, anytime-valid ranker selection.
3. **Wave 15 (deep-ML wave, torch-gated):** B1 first (synergy with waves
   11–12), then B2 stage (i) TORF, then B3. B4 only as a multi-week
   dedicated lane with staged gates.
4. **Continuous:** C3/C4/D items slot into any wave as single-module lanes;
   A3/A5/A8 attach to whichever conformal wave is running.

Difficulty ranking (hardest first): A1 > B4 > B3 > A2 > B2 > A5 > C1 > A4 >
B1 > A3 > C2 > B5 > A6 > C3 > A7 > rest.

## Standing caveats

- Every citation above was retrieved live from the arXiv API on 2026-09-29;
  verify version/author lists at implementation time (wave-11 lesson: spec
  citations were wrong twice and agents corrected them).
- None of these papers' *empirical market claims* transfer; the repo
  implements the algorithms as SYNTHETIC-gated correctness artifacts.
- Tier B lanes must stay torch-gated (`nn` extra) with clean skips, matching
  the deep-hedging lane pattern.
