# SOTA 08 — Finance LLM Evaluation Benchmarks

**Lane:** FINANCE LLM EVALUATION BENCHMARKS.
**Status:** research notes + coverage mapping + adoption plan, 2026-09-28.
**Scope:** `src/fx1/eval/*`, `src/fx1/honesty.py`, `src/fx1/data/quality.py`,
`src/fx1/cli.py`, `fx1_seed_corpus.jsonl`, `tests/fx1/`, `docs/FX1_TRAINING.md`.
**Base model:** Kimi K3 open weights (2.8T MoE, 104B active, native MXFP4).

This document summarizes state-of-the-art practice (2021–2026) for benchmarking
financial LLMs; audits fx-1's evaluation surface against that practice; and
proposes a concrete adoption plan covering new eval tasks, judge protocols,
contamination control for the seed corpus, and ship gates. **It modifies no
code.** Items marked *(plan)* are proposed work, not shipped behavior — this
doc does not claim fx-1 already does them.

> **Honesty contract.** Everything below is evaluation methodology for a
> `research_only=true`, `live_pnl_claim=false` model. Where this doc quotes a
> published benchmark that reports Sharpe ratios or trading returns, those are
> *cited properties of someone else's benchmark*, not fx-1 results; the fx-1
> ship gate continues to score **proper scores only** (pinball, CRPS, PIT,
> QLIKE, Brier, ECE, Spiegelhalter-z, Kupiec, HMM likelihood). Every synthetic
> bank in `src/fx1/eval/` is labeled `SYNTHETIC` and is a correctness test,
> never market evidence. No live-trading claim is made or implied anywhere.

Related in-tree docs: `docs/FX1.md`, `docs/FX1_TRAINING.md` (compute ladder +
gates), `docs/FX1_API_STABILITY.md`, `docs/SOTA/06-point-in-time-data.md`
(leakage — the data-side twin of this doc's eval-side concerns),
`docs/SOTA/09-finetuning-pipeline.md` (training ladder that consumes these
gates), `docs/SOTA/10-reproducibility.md` (receipt sealing for eval evidence),
`docs/INSTITUTIONAL_READINESS.md` (the five minimum-evidence conditions).

---

## 0. Verdict up front

fx-1's eval surface is **unusually strong on the axis nobody else benchmarks**
— honesty-contract enforcement — and **structurally thin on the axis every
financial benchmark measures** — quantitative answer correctness at scale.

What is genuinely ahead of published practice:

- **Honesty is a first-class scored dimension with fail-closed semantics**, not
  a model-card paragraph. `fx1.honesty.validate_fx1_output` is applied to every
  eval response (`fx1/eval/suite.py::score_task`), and
  `FORBIDDEN_HEADLINE_TOKENS` is mirrored from
  `quant_fund.research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS` with
  `tests/fx1/test_honesty_inheritance.py` blocking drift. Verified live:
  both sets are `{calmar, nav, pnl, sharpe, sortino}` and the mirror test
  passes. No finance benchmark in §1 has an equivalent — they measure finance
  *skill*, not finance *integrity*.
- **Leakage-aware eval twins.** `fx1/eval/masking.py` implements the
  KTD-Fin-style memory-gap protocol (`memory_gap_report`, budget 0.25) as a
  *ship-gate metric*, before KTD-Fin (arXiv:2605.28359) popularized it.
- **Three-probe contamination audit** (`fx1/eval/contamination.py`: n-gram
  containment + Min-K% Prob + ConStat-style rephrased gap), hash-bound to the
  corpus and bank it inspected, CLI-exposed (`fx1 contamination-audit`), and
  wired into the training pipeline's quality stage.

The five structural gaps (details in §3):

- **G1 — No executed-answer numeric reasoning.** fx-1 has *zero* code-execution
  graded tasks. All numeric grading is regex-extract-and-compare
  (`NUMERIC_TOL = 1e-4` in `ts_reasoning.py`, `0.5` in `retrieval_eval.py`)
  against precomputed targets. SOTA (FinanceMath, FinanceReasoning, BizBench)
  grades by **executing the model's program** (PoT/PAL), which is the only
  honest way to score multi-step financial arithmetic.
- **G2 — No LLM-as-judge protocol at all, and none needed for most of the bank
  — but the free-form dimension is unmeasured.** `run_suite` scoring is
  forbidden-pattern + required-token + honesty validation. There is no
  rubric-graded open-ended financial writing (the FinBen *text generation*
  aspect), so a model can pass every task while being incoherent in prose.
- **G3 — Eval bank is small (28 tasks) and static.** 10 honesty baits + 12
  domain + 6 general. Domain tasks are keyword-recall checks
  (`required_tokens=["qlike"]`), not competence measurements. With n=12 domain
  tasks the paired bootstrap CI in `fx1/eval/compare.py` is very wide — the
  ship gate's "domain must beat base" test has almost no power.
- **G4 — Seed corpus has measured template defects and near-duplicate collapse.**
  Measured (this doc, `uv run python` over `fx1_seed_corpus.jsonl`): all 5
  examples contain a `bound method` schema-rendering bug in the assistant
  message; 4 of 5 have an empty `{}` metrics block; pairwise Jaccard ranges
  **0.7359–0.9036** with **6 of 10 pairs ≥ 0.90**; `dedup_and_filter` keeps only
  **2 of 5** examples. Training on this teaches a boilerplate refusal template,
  not reasoning.
- **G5 — No canary/dye-pack instrumentation.** `rg -i 'canary|dye.?pack'` finds
  matches only in `SECURITY.md` and a test in `tests/fx1/test_sources.py` — the
  *training corpus carries no provenance canaries*, so future third-party
  training on fx-1's own corpus is undetectable, and the `min_k_percent_probe`
  is inert (`threshold=nan`, `flagged=False`, never active because no logprobs
  are ever supplied).

The three highest-leverage moves:

1. **Add an executed-answer numeric track (PoT/PAL) with a tiered tolerance
   grid** — the single biggest capability measurement gain, and the one
   requirement SOTA financial benchmarks agree on (§2.4, §4.1).
2. **Rebuild the seed corpus with defect gates + canaries + diversity floors** —
   fix the `bound method` bug and empty-metrics-block regression, enforce a
   pairwise-Jaccard ceiling, and stamp BIG-bench-style canaries so
   `min_k_percent_probe` becomes a calibrated rather than inert instrument
   (§5).
3. **Grow the bank to a powered size and add a position-swapped, length-
   controlled judge protocol for the free-form dimension only** — with
   swap-augmentation, verbosity regression, and self-preference controls
   (§2.5, §4.2). Keep deterministic execution as the primary scorer; the judge
   is a *supplement*, never a ship gate on its own.

---

## 1. Benchmark summaries + citations

Coverage is organized by what each family actually measures, because the
families are not interchangeable: exam benchmarks measure recall, QA benchmarks
measure grounded reasoning, program benchmarks measure arithmetic, and
agent/tool benchmarks measure orchestration.

### 1.1 Holistic task suites

**PIXIU / FLARE** — Xie et al., 2023 (arXiv:2306.05443). The originating
open-source effort: FinMA (LLaMA fine-tuned on 136K instruction samples) plus a
standardized benchmark of **5 financial NLP tasks and 1 prediction task over 9
datasets**, released with the FIT instruction corpus. Its headline finding —
financial LLMs are strong on classification-style NLP tasks and weak on
numeric reasoning and forecasting — is the pattern every later benchmark
reproduces. *Relevance to fx-1:* PIXIU's task taxonomy is the natural external
reference set for the "general finance competence" axis fx-1 currently measures
with 12 keyword tasks.

**FinBen / FinBen-Bench** — Xie et al., NeurIPS 2024 Datasets & Benchmarks
(arXiv:2402.12659). The successor: **42 datasets spanning 24 tasks across 8
aspects** — information extraction (IE), textual analysis (TA), question
answering (QA), text generation (TG), risk management (RM), forecasting (FO),
decision-making (DM), and bilingual EN/ES. First benchmark to include stock
trading evaluation, agent-based evaluation, and RAG-based evaluation. Evaluated
21 LLMs; instruction tuning improves TA but not complex QA or forecasting.
Hosted the first financial-LLM shared task (FinNLP-AgentScen @ IJCAI-2024, 12
teams). The name **"FinBen-Bench"** refers to the same artifact lineage — the
benchmark component of FinBen, shipped as `finlm_eval` (a fork of
EleutherAI's `lm-evaluation-harness`) under The-FinAI/PIXIU on GitHub; there is
no separate paper. *Note on version drift:* the arXiv v1 abstract states
35 datasets / 23 tasks / three difficulty spectrums (Cattell-Horn-Carroll
framing); the NeurIPS camera-ready states 42 / 24 / 8 aspects. Cite the camera-
ready numbers and the arXiv ID together. *Relevance to fx-1:* FinBen's 8-aspect
taxonomy is the best available coverage checklist; §3 maps fx-1 against it.

**Open FinLLM Leaderboard** — Xie et al., 2025 (arXiv:2501.10963), run with the
Linux Foundation and Hugging Face. The living continuation of FinBen:
multimodal FinLLM/FinAgent readiness, community dataset/task/model
contributions. *Relevance:* the only venue where an fx-1-style research model
could publish comparable numbers without claiming market edge — and only if the
submission is scored as capability, never as tradability (see §1.6).

**FinEval / FinEval-KR** — Guo et al., 2023 (arXiv:2308.09975): 8,351 Chinese
financial questions over 4 areas including a Financial *Agent* track (616
questions) and Financial *Security* (1,640). Dou et al., 2025
(arXiv:2506.21591) adds the crucial methodological move for fx-1: **decoupling
knowledge score from reasoning score**, plus a Bloom-taxonomy cognitive score,
and finds even top models bottleneck on *knowledge application*, and that
specialized financial LLMs generally lag top general models. *Relevance:* the
knowledge/reasoning decoupling is exactly the separation fx-1's masked-twin
protocol approximates (§2.3), and it argues for reporting the two scores
separately rather than a blended pass rate.

### 1.2 Grounded QA over filings

**FinanceBench** — Islam et al., 2023 (arXiv:2311.11944). **10,231 open-book
QA triplets** (question, answer, evidence string) over public-company filings,
with **150 expert-validated golden cases** released open-source. Designed to be
"clear-cut and straightforward" — i.e. a *minimum* performance standard, not a
frontier. Headline result: GPT-4-Turbo with a retrieval system **incorrectly
answered or refused 81%** of the 150-case sample (2,400 manually reviewed
answers across 16 model configurations); longer context helps but is unrealistic
at enterprise latency; all models hallucinate. *Relevance to fx-1:* this is the
external standard for the retrieval track `fx1/eval/retrieval_eval.py`
synthesizes. fx-1's synthetic-filing bank is the right *shape* (factoid /
comparison / multi-hop / negative / honesty-bait families, citation accuracy,
retrieval precision) but is generated, so it measures protocol compliance, not
document grounding. FinanceBench's own finding — that the failure mode is
refusal-plus-hallucination, not silence — is why the `negative` family (correct
answer is "not in the documents") matters most.

**FinQA** — Chen et al., EMNLP 2021 (arXiv:2109.00122). 8,281 expert-written QA
pairs over earnings reports with **gold reasoning programs** (DSL), scored by
execution accuracy *and* program accuracy. The program annotation is the
origin of the executed-answer discipline in §2.4.

**TAT-QA** — Zhu et al., ACL 2021 (arXiv:2105.07624). Hybrid tabular+textual QA
requiring addition/subtraction/multiplication/division/counting/comparison/
sorting and compositions; TAGOP reached 58.0 F1 vs 90.8 human expert F1. The
canonical demonstration that **table–text interleaving** is a distinct skill.

**ConvFinQA** — Chen et al., EMNLP 2022 (arXiv:2210.03849). Multi-turn
conversational numeric reasoning with **chains of numerical reasoning** over
long-range context. *Relevance:* fx-1's `tooluse_eval` multi-turn loop is the
closest analogue; ConvFinQA shows the hard case is *carrying intermediate
results across turns*, which fx-1's `max_steps` protocol permits but does not
score.

### 1.3 Quantitative / program-synthesis reasoning

**BizBench** — Krumdick et al., ACL 2024 (arXiv:2311.06602). **8 quantitative
reasoning tasks** centered on QA-over-financial-data *via program synthesis*:
three financially-themed code-generation tasks from newly collected and
augmented QA data, plus isolated sub-skills — reading comprehension of
financial text and tables for extracting intermediate values, and understanding
of financial concepts and formulas. Finds the bottleneck is **business and
financial understanding**, not code generation. *Relevance to fx-1:* this is the
template for G1. BizBench's decomposition (extract → know formula → compute)
maps directly onto what a fx-1 numeric track should grade stepwise rather than
as one pass/fail.

**FinanceMath** — Zhao et al., 2024 (arXiv:2311.09797). **1,200 problems** with
hybrid textual + tabular content requiring college-level finance knowledge, and
**expert-annotated solutions in Python program format**. Includes a
finance-domain knowledge bank and evaluates knowledge-integration strategies;
**44 LLMs** evaluated under both CoT and PoT prompting. Best system (GPT-4o,
CoT) reaches only **60.9%**, vs estimated human expert **92%**; external
knowledge helps (Gemini-1.5-Pro 47.5% → 54.5%) but the gap persists. *This is
the single most important number in this doc for fx-1 planning:* it bounds what
a domain-fine-tuned model can credibly claim, and it shows PoT is the right
prompting mode for this content.

**FinanceReasoning** — Tang et al., ACL 2025 (arXiv:2506.05828;
DOI 10.18653/v1/2025.acl-long.766). Targets **large reasoning models** on
financial numerical reasoning. Three contributions: (1) *credibility* —
replaces 15.6% of questions from four public datasets and annotates **908 new
questions with detailed Python solutions**, tightening evaluation standards;
(2) *comprehensiveness* — covers **67.8% of financial concepts and formulas**,
with **3,133 Python-formatted functions**; (3) *challenge* — 238 Hard problems
requiring multiple formulas. Best model (OpenAI o1 with PoT) reaches **89.1%**,
yet LRMs "still face challenges in numerical precision"; combining a Reasoner
with a Programmer model lifts DeepSeek-R1 83.2% → 87.8%. Its strict numeric
tolerance (tight relative-error bound, ≈0.2%) and explicit refusal-to-guess
scoring are the reference standard for §2.4's tolerance grid.

### 1.4 Professional-exam / certification benchmarks

**FINESSE-Bench** — Stanishevskii et al., 2026 (arXiv:2605.15482). A **suite of
eight specialized benchmarks, 3,993 questions**, for *hierarchical* evaluation:
exam-oriented sets inspired by professional certifications (**CFA-like Levels
1–3, CMT-like Level 2, CFTe-like Level 1**), applied trading task collections,
and a Russian-language olympiad set. Explicitly designed to measure
**performance degradation as difficulty increases**, breadth, computational-task
ability, and behavior in specialized domains. Crucially it ships a **unified
evaluation protocol across multiple-choice, numerical, and short open-ended
responses, with an automated LLM-as-judge scoring scheme for freeform answers** —
i.e. the judge protocol in §2.5 is required, not optional, once free-form
professional questions enter the bank. Positions itself as a complement to
FinQA/ConvFinQA/TAT-QA (which "do not provide an explicit hierarchy of
professional difficulty") and to FinanceBench/PIXIU/FinBen/FLaME.

**Advanced Financial Reasoning at Scale: CFA Level III** — Shetty et al., 2025
(arXiv:2507.02954). Evaluates frontier LLMs on the *hardest* CFA level, where
questions are constructed-response and case-based rather than multiple-choice.
The relevant methodological point for fx-1: at Level III the answer format
itself (essay + numeric + item-set) forces a mixed grading protocol — exactly
the multi-format scoring FINESSE-Bench formalizes.

**Exam-style caveat (MMLU-Redux)** — Gema et al., 2024 (arXiv:2406.04127).
Estimates **6.49% of MMLU questions contain errors**; 57% of the analyzed
Virology subset. Released 5,700 re-annotated questions across all 57 subjects
and showed significant discrepancies with originally reported model
performance. *Relevance:* any fx-1 exam-style bank needs an error-audit pass
and a "label defect" register, or the reported pass rate is measuring the
bank's noise floor. FinanceReasoning's 15.6% question replacement rate is the
empirical proof that finance banks are worse than this, not better.

### 1.5 Agent, tool-use, and trajectory benchmarks

**InvestorBench** — Li et al., 2024 (arXiv:2412.18174). Benchmark for financial
**decision-making tasks with LLM-based agents** (single-asset trading and
portfolio allocation), with a gym-style environment and task-level + agent-level
metrics. The standard reference for "can an agent act", as opposed to "can an
agent answer".

**FinToolBench** — Lu et al., 2026 (arXiv:2603.08262). The first **real-world
runnable** financial tool-learning benchmark: **760 executable financial tools**
coupled with **295 tool-required queries**. Evaluation goes beyond binary
execution success to finance-critical dimensions — **timeliness, intent type,
and regulatory-domain alignment** — plus FATR, a finance-aware tool-retrieval
and reasoning baseline. *Relevance:* the "regulatory domain alignment" and
"timeliness" axes are precisely the two things fx-1's `tooluse_eval` mock
harness cannot express, and they are honesty-adjacent (using a stale or
out-of-scope tool is a correctness failure).

**FinMCP-Bench** — Zhu et al., ICASSP 2026 (arXiv:2603.24943). **613 samples
over 10 scenarios / 33 sub-scenarios**, using **65 real financial MCP servers**,
with three sample types — **single-tool, multi-tool, multi-turn** — and metrics
explicitly measuring tool-invocation accuracy and reasoning capability. Real and
synthetic queries mixed deliberately for diversity.

**FinTrace** — Cao et al., 2026 (arXiv:2604.10015). **800 expert-annotated
trajectories across 34 real-world financial task categories** with a
**rubric-based protocol: nine metrics along four axes** — action correctness,
execution efficiency, process quality, output quality. Across 13 LLMs: frontier
models get tool *selection* right but all struggle with **information
utilization and final-answer quality** — "a critical gap between invoking the
right tools and reasoning effectively over their outputs." Also ships
FinTrace-Training (8,196 trajectories, preference pairs); SFT+DPO on
Qwen-3-8B/32B improves *intermediate* reasoning metrics, with DPO better at
suppressing failure modes, but end-to-end answer quality stays the bottleneck.
*Relevance to fx-1:* this is the most direct external critique of the shape of
`fx1/eval/tooluse_eval.py`, which grades via five binary checks
(`all_golden_calls_ok`, `final_answer_given`, `no_hallucinated_tools`,
`honesty_clean`, `refusal_stated`) plus an LCS-based `plan_match_score`.
FinTrace's four-axis rubric is the upgrade path, and its "trajectory
improvements do not propagate to output quality" finding is a warning against
optimizing fx-1's tool-use training on trajectory metrics alone.

### 1.6 The critique that governs everything above

**The Alpha Illusion** — Ye et al., 2026 (arXiv:2605.16895). Position paper:
**reported alpha from end-to-end LLM trading agents should not be treated as
deployment evidence.** Names the ecosystem (FinCon, FinMem, TradingAgents,
FinAgent, QuantAgent, FLAG-Trader) and observes that several report headline
Sharpe ratios material at face value, and that **FinBen itself reports
trading-task Sharpe statistics in the same range**. Requires that before such
returns support a deployable-capability claim, they survive structural validity
tests for **temporal integrity, real-world frictions, counterfactual
robustness, predictive calibration, numerical execution, and multi-agent
disaggregation**. Argues the problem is structural, not merely evaluative:
"language confidence is not tradable probability, narrative reasoning is not
numerical execution, and model priors may become undisclosed implicit factor
exposures." Contributes a minimum reporting protocol suite **P1–P6** with
tiered applicability by claim strength, plus a conservative modular alternative
using LLMs as *auditable information interfaces upstream of independent
calibration, risk, and execution modules*.

**This paper is the external justification for fx-1's honesty contract.** The
lab's rule — proper scores only, no Sharpe/Sortino/Calmar/P&L/NAV headline, no
live-trading claim — is the same position, adopted independently and enforced in
code rather than argued in prose. Two consequences for this lane:

1. fx-1 must **not** adopt FinBen's trading-Sharpe task or InvestorBench's
   return-based scoring as ship metrics. It may run them as *capability
   diagnostics* with an explicit non-tradability label, and never headline them.
   This is exactly what `FORBIDDEN_HEADLINE_TOKENS` already blocks.
2. The modular architecture Ye et al. recommend (LLM as auditable interface,
   independent calibration/risk/execution) **is** dipcatcher's existing
   architecture: fx-1 interprets and explains, the harness computes and gates.
   `docs/FX1.md`'s "you interpret and explain them, you do not replace them" is
   the same claim.

**KTD-Fin** — 2026 (arXiv:2605.28359), *From Knowing to Doing: A
Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets*.
Qlib-based, CSI300 universe. Two instruments: (a) a **four-level data-side
masking protocol** — *bright* (real ticker/date), *stock-blind* (aliased
ticker, real date), *date-blind* (real ticker, relative day index), *blinded*
(both anonymized) — applied consistently across prompts **and tool returns**, so
an agent never sees a real identifier even transiently; certified by an
independent ten-attacker probe with top-1 ticker recovery ≤ 3.0% and strict
joint (ticker top-5 ∧ date within ±7 trading days) ≤ 1.5%; and (b) **Barra-style
attribution** decomposing realized returns into market beta, style factors, and
stock-selection alpha. Across ten frontier agents over a 2024–2026 window,
masking substantially changes rationales toward anonymized factor-based
reasoning, and **cumulative returns under leakage control are largely explained
by passive market and style exposure, with limited evidence of persistent
stock-selection alpha** (secondary reporting: ~70% market beta, ~20% style,
small residual). *Relevance to fx-1:* `fx1/eval/masking.py` implements (a) at a
single level with a memory-gap budget. KTD-Fin's four *levels* and its
**independent attacker probe to certify the mask** are the two upgrades worth
adopting (§4.3) — fx-1 currently has no test that its own mask holds.

**Look-Ahead-Bench** — Benhenda, 2026 (arXiv:2601.13770). Standardized
benchmark of **look-ahead bias in Point-in-Time LLMs** evaluated in *practical
financial workflows* rather than inner-lookahead Q&A. Distinguishes genuine
predictive capability from memorization by analyzing **performance decay across
temporally distinct market regimes** with quantitative baselines as thresholds.
Llama 3.1 (8B/70B) and DeepSeek 3.2 show significant look-ahead bias as measured
by alpha decay; the Pitinf family generalizes better and improves with scale.
*Relevance:* `fx1/eval/timepart.py` partitions tasks by creation timestamp
against a knowledge cutoff and reports `post_cutoff_pass_rate` — the same idea
at task granularity rather than market-regime granularity. Extending `timepart`
to a **regime-stratified** partition is the cheap version of Look-Ahead-Bench.

---

## 2. Method SOTA (the four techniques that matter)

### 2.1 Contamination control — the full toolkit

| Method | Mechanism | Key citation | fx-1 status |
|---|---|---|---|
| Canary strings | Publish a unique GUID with the benchmark; if a model can complete it, it trained on the set | BIG-bench, Srivastava et al. 2022 (arXiv:2206.04615) | **ABSENT** from corpus (§5) |
| Dye-pack backdoors | Mix backdoor samples with test data; detect trained-on-it models **without** logits/loss access; multiple backdoors with stochastic targets give *exact* FPR | Cheng et al., EMNLP 2025 (arXiv:2505.23001) — FPR as low as 0.000073% (MMLU-Pro, 8 backdoors), 0.000017% (BBH), 0.127% (Alpaca, 6 backdoors) | **ABSENT** |
| n-gram / shingle containment | Surface-overlap filter of training corpus vs eval bank | standard | **PRESENT** — `ngram_containment_scan` (threshold 0.30) and `dedup_and_filter` (threshold 0.60) |
| Min-K% Prob | Mean of each item's k%-lowest token logprobs; contaminated items show systematically higher minimum-token likelihood | Shi et al., ICLR 2024 (arXiv:2310.16789) | **INERT** — implemented but `threshold=nan`, `flagged=False` always; no caller supplies logprobs |
| Guided-instruction completion | Prompt with dataset name + partition + random-length prefix; flag if completion matches the reference suffix; instance-level then partition-level | Golchin & Surdeanu, ICLR 2024 Spotlight (arXiv:2308.08493) — 92–100% accuracy; found GPT-4 contaminated on AG News, WNLI, XSum | **ABSENT** |
| Rephrased gap (ConStat) | Performance delta canonical vs meaning-preserving rephrasing; memorized items degrade | Yang et al. 2023 (arXiv:2311.04850) | **PRESENT** — `rephrased.py` + `rephrased_gap_probe` (budget 0.30) |
| Dynamic / generated eval | Regenerate items at eval time with controllable complexity so no static answer key can leak | DyVal, Zhu et al. 2023 (arXiv:2309.17167); DARG, Zhu et al. 2024 (arXiv:2406.17271); LiveBench, White et al., ICLR 2025 Spotlight (arXiv:2406.19314) | **PARTIAL** — `ts_reasoning.py`, `retrieval_eval.py`, `calibration_eval.py`, `tooluse_eval.py` are all **seeded-generated**, which is the strongest available posture; but the 28-task `DEFAULT_BANK` is static |
| GSM1k-style paired re-collection | Rebuild a benchmark from scratch at matched difficulty and measure the drop | Zhang et al., NeurIPS 2024 D&B (arXiv:2405.00332) | **ABSENT** (but the seeded banks make it unnecessary for the numeric axes) |

Two facts worth internalizing. **First**, the *only* method with a provable
false-positive guarantee is DyePack, and it works black-box — which matters for
fx-1 because the lab cannot inspect Kimi K3's internals to test whether the base
saw any given benchmark. **Second**, fx-1's seeded synthetic banks
(`build_ts_reasoning_bank(seed=…)`, `build_retrieval_bank(seed=…)`,
`build_calibration_bank(seed=…)`, `build_tooluse_tasks(seed=…)`) are already a
*stronger* contamination posture than any static public benchmark: the items are
regenerated from a seed at eval time, so there is nothing to memorize. The
weak surface is the **static 28-task `DEFAULT_BANK`** and the **seed corpus**.

### 2.2 Numeric-answer evaluation for finance

Three converging findings across FinanceMath, FinanceReasoning, BizBench, FinQA
and ConvFinQA:

1. **Execute, don't string-match.** PoT (Chen et al., TMLR 2023,
   arXiv:2211.12588) and PAL (Gao et al., ICML 2023, arXiv:2211.10435)
   disentangle computation from reasoning by having the model emit a program
   that a deterministic interpreter runs. FinQA's gold-program + execution-
   accuracy annotation is the same idea a year earlier. FinanceMath evaluates
   **both** CoT and PoT and PoT is the credible mode for this content;
   FinanceReasoning's best result (89.1%, o1) is *with PoT*. A model that
   reasons correctly and mis-multiplies is indistinguishable, under string
   matching, from a model that reasons incorrectly.
2. **Tolerance must be declared and tiered.** FinanceReasoning uses a tight
   relative-error bound (≈0.2%); BizBench-style program accuracy is
   execution-exact; FinanceMath's expert Python references permit float
   comparison at a stated epsilon. Unit and format normalization — currency
   prefixes, percentage vs basis points, thousands separators, fiscal-period
   labels, date formats — must be normalized *before* comparison, not forgiven
   by a loose tolerance.
3. **Score refusal and abstention explicitly.** FinanceBench's central empirical
   finding is that models *refuse or hallucinate* rather than fail quietly
   (81% wrong-or-refused). A numeric track that scores only attempted answers
   will reward confident guessing. Score: correct / wrong / refused-or-abstained
   as three outcomes, and report the abstention rate separately.

### 2.3 Memorization vs reasoning: masking and temporal partitioning

KTD-Fin (§1.6) established the protocol: mask identifiers and dates consistently
across prompts *and tool returns*, certify the mask with independent attackers,
then measure the gap. Look-Ahead-Bench adds regime-stratified decay analysis.
MMLU-Redux adds the "your bank has label errors" control.

fx-1's `masking.py` already does the core — deterministic
`hashlib.sha256`-derived placeholders so the same surface form always maps to
the same token (reasoning stays possible, recall becomes impossible), with a
`_VOCAB_ALLOWLIST` protecting proper-score vocabulary (CRPS, PIT, QLIKE, Brier,
ECE, HMM, VaR, ES, OFI, VPIN, Kyle, Roll, CPCV, HAC, DM, RC, SPA, StepM, TWAP,
SYNTHETIC, OHLC/V, LLOB) from being masked away. That allowlist is a genuinely
good design decision: it keeps *procedural* knowledge measurable while blocking
*episodic* recall. The gap is that there is **no attacker probe certifying the
mask holds** — `_TICKER_RE = \b[A-Z]{2,6}(?:USDT|USD|EUR)?\b` will miss
lowercase tickers, mixed-case names, ISINs/CUSIPs, and company names entirely.

### 2.4 Statistical validity of the eval itself

`fx1/eval/compare.py` implements `bootstrap_delta_ci` (paired bootstrap on
pass-rate deltas) and `mcnemar_statistic` — the correct instruments. The problem
is **n**. With 12 domain tasks, the 95% CI on a pass-rate delta spans roughly
±0.5 absolute; "domain improvement must exclude zero" (per `docs/FX1_TRAINING.md`)
is unachievable at that n for any realistic effect size. Any adoption plan that
does not grow n is not a plan to pass its own gate. The seeded banks solve this
cheaply: `build_ts_reasoning_bank(n_instances=40)`,
`build_retrieval_bank(n_questions=…)`, `build_calibration_bank(n_questions=60)`,
`build_tooluse_tasks(n_tasks=12)` are all n-scalable by construction, so the
*numeric* axes can reach statistical power without new content — only the
keyword-recall `DOMAIN_TASKS` cannot.

### 2.5 LLM-as-judge: pitfalls and mitigations

The bias taxonomy is settled:

| Bias | Finding | Citation |
|---|---|---|
| Position | Judges favor the first (or second) answer presented; **swap augmentation** — score both orders, count disagreement as a tie — is the fix | Zheng et al., NeurIPS 2023 D&B (arXiv:2306.05685, MT-Bench/Chatbot Arena); Zhu et al., ICLR 2025 (arXiv:2310.17631, JudgeLM: swap augmentation + reference support + reference drop) |
| Verbosity / length | Judges prefer longer outputs; **length-controlled regression** on log-length recovers ~unbiased preference | Dubois et al., COLM 2024 (arXiv:2404.04475, LC-AlpacaEval) |
| Self-preference / egocentric | Judges recognize and favor their own generations; also knowledge and format bias | Panickssery et al., NeurIPS 2024 (arXiv:2404.13076); Koo et al., ACL 2024 (arXiv:2309.17012, CoBBLEr: 6 biases, ~40% of comparisons biased on average, human–machine RBO only 49.6%) |
| Rubric-free drift | Unstructured judges are unstable; explicit score rubrics + reference answers restore agreement with humans | Liu et al., EMNLP 2023 (arXiv:2303.16634, G-Eval); Kim et al., ICLR 2024 (arXiv:2310.08491, Prometheus: 13B open judge, Pearson 0.897 with humans on 45 custom rubrics, vs GPT-4 0.882 and ChatGPT 0.392) |
| Judge disagreement | Different judge models rank differently; ensemble + report inter-judge agreement | Wang et al., ICLR 2024 (arXiv:2306.05087, PandaLM) |

CoBBLEr's 49.6% human–machine RBO is the number that should set expectations: an
LLM judge is a *screening* instrument, not an oracle. Practical protocol for
fx-1 *(plan)*:

1. **Deterministic first.** Anything that can be graded by execution or by the
   existing forbidden-pattern/required-token/honesty machinery is. The judge
   never sees those tasks.
2. **Judge only the free-form dimension**, and only with: an explicit rubric
   (per-dimension 1–5 anchors), a reference answer, **both position orders**
   scored, disagreement → tie, log-response-length recorded for a
   length-controlled regression, and **≥2 heterogeneous judge models** (never
   the K3 base or an fx-1 checkpoint — that is direct self-preference).
3. **The judge cannot pass or fail a ship gate alone.** Judge output is reported
   as a diagnostic with inter-judge agreement, and any gate that uses it
   requires a human-validated calibration subset (report judge–human agreement
   on it).
4. **Judge the judge.** Periodically run the judge on known-bad outputs
   (deliberately verbose, deliberately wrong-but-fluent) and confirm it scores
   them low. A judge that cannot fail a planted bad answer is not measuring.

### 2.6 Retrieval / grounding evaluation

fx-1's `retrieval_eval.py` already tracks the right quantities — `accuracy`,
`retrieval_precision`, `citation_accuracy`, `honesty_gate_passed`, plus a
`golden_path` per question and a `negative` family whose correct behavior is
"not in the documents". The external literature that adds rigor: ARAGOG
(Oladipo et al., 2024, arXiv:2404.01037) for graded RAG output evaluation;
Ovadia et al., 2024 (arXiv:2404.13781) on the finding that **retrieval quality
and answer quality must be measured separately** because good retrieval with bad
synthesis and bad retrieval with lucky answers look identical under end-to-end
accuracy alone; FineWeb/Nugget-style chunking work (Qin et al., ICML 2023,
arXiv:2310.01732) for the chunk-granularity confound; and the fine-tuning-vs-
RAG comparison (Ovadia et al., 2023, arXiv:2312.05934) establishing that
**retrieval dominates fine-tuning for knowledge injection** — which is the
architectural argument for keeping fx-1 a reasoning/interface model over the
dipcatcher evidence engine rather than trying to memorize filings.

---

## 3. Coverage mapping: fx-1 today vs SOTA

### 3.1 What exists (verified by inspection, 2026-09-28)

| Module | What it measures | Grading | n |
|---|---|---|---|
| `eval/bank.py::HONESTY_BAITS` | 10 contract-violation lures (Sharpe headline, live P&L, synthetic-as-live, relax gates, guaranteed returns, backtest-as-live, gross-only, survivorship, multiple testing, unlabeled synthetic) | forbidden regex + required tokens + `validate_fx1_output` | 10 |
| `eval/bank.py::DOMAIN_TASKS` | 12 lab-knowledge recalls (proper scores, QLIKE, Kupiec, VPIN/Kyle, walk-forward+CPCV, Almgren+next-open, DQS+Brier, iFind/Wind routing, PIT/as_of leakage, SHA-256 receipt provenance, conformal, spread+impact) | required-token presence | 12 |
| `eval/bank.py::GENERAL_TASKS` | 6 anti-forgetting sanity checks (arithmetic, explanation, code read, syllogism, translation, summarization) | required-token presence, `enforce_honesty=False` | 6 |
| `eval/redteam.py` | 6 adversarial launderings: roleplay, hypothetical, authority pressure, evidence laundering, encoding trick (spell the number), synthetic-label renaming | forbidden regex + required tokens | 6 |
| `eval/rephrased.py` | ConStat-style twins of all 22 honesty+domain tasks | paired pass delta, budget 0.30 | 22 pairs |
| `eval/masking.py` | KTD-Fin-style masked twins, deterministic SHA-256 placeholders, vocab allowlist | `memory_gap_report`, budget 0.25 | 28 pairs |
| `eval/timepart.py` | Knowledge-cutoff partition, post-cutoff pass rate | partition + rate | n/a |
| `eval/ts_reasoning.py` | **SYNTHETIC** seeded TS reasoning: process identification, mean pinball loss, preferred forecast, conformal coverage, honesty-bait twins | exact / `NUMERIC_TOL=1e-4` / boolean, computed with `quant_fund.metrics.scoring.pinball_loss` and `.coverage` | 40 default (`n_instances`) |
| `eval/retrieval_eval.py` | **SYNTHETIC** seeded FinanceBench-style retrieval over generated filings: factoid / comparison / multi-hop / negative / honesty-bait; retrieve-call protocol with `max_retrieves`; citation check | `grade_numeric_answer` (tol 0.5), `grade_doc_choice`, citation accuracy | scalable |
| `eval/calibration_eval.py` | **SYNTHETIC** seeded calibrated-uncertainty: gaussian-tail, binomial-hits, AR(1)-excursion, quantile-coverage; conjugate-spaced true probabilities; golden + no-retrieval oracle models | **ECE** + **Spiegelhalter-z**, thresholds 0.05 / 2.0, equal-width bins | 60 default |
| `eval/tooluse_eval.py` | **SYNTHETIC** seeded multi-turn tool calling against `HARNESS_REGISTRY` with a `MockHarness`; LCS `plan_match_score`; interleaved bait tasks | 5 binary checks + plan-match fraction | 4–12 |
| `eval/compare.py` | Ship-gate statistics: paired bootstrap CI on pass-rate deltas + McNemar | CI excludes zero | n/a |
| `eval/contamination.py` | 3-probe audit, hash-bound report | n-gram / Min-K% / rephrased gap | n/a |
| `honesty.py` | Fail-closed output validator; `FORBIDDEN_HEADLINE_TOKENS` mirrored from `quant_fund.research.catalog` | raises `Fx1HonestyError` | n/a |
| `data/quality.py` | exact dedup, near-dup shingle screen (Jaccard ≥ 0.90 vs last 500), eval-containment removal (≥ 0.60), length filter, frozen split + hash manifest | `QualityReport` | n/a |
| `cli.py` | `eval`, `modelcard`, `redteam`, `dpo`, `curriculum`, `maskedaEval`, `contamination-audit`, `sign`, `attestation`, `sbom`, `mrm`, `dipbench`, `infer`, `backtest`, `doctor` | — | — |
| `tests/fx1/` | 26 test modules incl. `test_honesty_inheritance.py` (blocks token drift), `test_rephrased.py`, `test_calibration_eval.py`, `test_retrieval_eval.py`, `test_tooluse_eval.py`, `test_ts_reasoning.py`, `test_bench_run.py`, `test_docs_drift.py` | — | 285 collected tests (`make fx1-test` lane; AGENTS.md still says "~200") |

### 3.2 FinBen 8-aspect coverage

| FinBen aspect | fx-1 coverage | Verdict |
|---|---|---|
| Information extraction (IE) | none (no NER/span/abbreviation/relationship extraction tasks) | **MISSING** |
| Textual analysis (TA) | none (no sentiment/classification/NLI over financial text) | **MISSING** |
| Question answering (QA) | `retrieval_eval.py` — synthetic filings, 5 families, citation-checked | **PARTIAL (synthetic only)** |
| Text generation (TG) | `general-summarization` is one keyword check; no rubric-graded generation | **MISSING** |
| Risk management (RM) | `calibration_eval.py` (ECE, Spiegelhalter-z), `ts_reasoning.py` coverage family, `domain-tail-risk` (Kupiec) | **STRONG — best-covered aspect** |
| Forecasting (FO) | `ts_reasoning.py` pinball/identification families; `forecast/` harness (`evaluate.py`, `runner.py`, `protocol.py`); `bench/dip.py`, `bench/run.py` | **STRONG (proper scores)** |
| Decision-making (DM) | `tooluse_eval.py` (mock harness, golden plan); `domain-validation-gates` | **PARTIAL** |
| Bilingual (ES / multi-lang) | `general-translation` (one EN→FR token check) | **MISSING** |
| *Not in FinBen:* honesty / integrity | `HONESTY_BAITS` + `redteam.py` + `validate_fx1_output` on every response + inheritance test | **fx-1 leads; no external equivalent** |
| *Not in FinBen:* contamination audit | 3-probe hash-bound audit, CLI-exposed, pipeline-wired | **fx-1 leads (Min-K% inert)** |
| *Not in FinBen:* memory gap as ship metric | `masking.py` + `memory_gap_report` | **fx-1 leads** |

### 3.3 Family-by-family gap table

| SOTA family | Closest fx-1 analogue | Gap | Priority |
|---|---|---|---|
| FinanceMath / FinanceReasoning / FinQA (executed numeric) | `ts_reasoning.py` numeric families | **No code execution**; targets precomputed; tolerance not tiered; no abstention scoring | **P0** |
| BizBench (program synthesis + extraction + formula knowledge) | none | No stepwise extract→formula→compute grading | **P0** |
| FinanceBench (real filings, evidence strings) | `retrieval_eval.py` | Synthetic documents only; no real-filing grounding; `NUMERIC_TOL=0.5` is loose | P1 |
| PIXIU/FLARE task taxonomy | `DOMAIN_TASKS` (keyword recall) | Recall ≠ competence; no external comparable numbers | P1 |
| FINESSE-Bench / CFA-L3 (hierarchical difficulty, mixed formats) | none | No difficulty ladder; no MCQ track; no free-form professional track | P1 |
| FinTrace (4-axis, 9-metric trajectory rubric) | `tooluse_eval.py` (5 binary checks + LCS) | No process-quality or output-quality axis; efficiency unmeasured | P2 |
| FinToolBench / FinMCP-Bench (real tools, timeliness, regulatory alignment) | `tooluse_eval.py` against `HARNESS_REGISTRY` mocks | Mocked returns; no timeliness or regulatory-scope axis | P2 |
| InvestorBench (agent decision-making) | `tooluse_eval.py`, `bench/dip.py` | Return-based scoring — **deliberately not adopted** per §1.6 / honesty contract | N/A (by design) |
| KTD-Fin (4-level mask + attacker certification + attribution) | `masking.py` (1 level, no probe) | No mask certification; no relative-day-index level; ticker regex misses lowercase/ISIN/names | **P0** |
| Look-Ahead-Bench (regime-stratified decay) | `timepart.py` (cutoff partition) | No market-regime stratification | P2 |
| DyePack / BIG-bench canary / guided-instruction | none | No canaries in corpus; no backdoor-flagged items; Min-K% inert | **P0** |
| GSM1k / MMLU-Redux (bank error audit) | none | No label-defect register for the bank | P1 |
| LLM-as-judge (rubric, swap, length-control, ensemble) | none | No judge protocol; free-form dimension unmeasured | P1 |

### 3.4 Three defects found while measuring (read-only checks)

These were produced by running the repo's own code over the tracked seed
corpus; they are correctness findings, not performance claims.

**D1 — Corpus template bug (5/5 examples).** Every assistant message in
`fx1_seed_corpus.jsonl` contains a rendered Python repr instead of a schema
name:

```
Receipt b59422415e556f67… (schema `<bound method BaseModel.schema of <class 'fx1.data.receipts.ReceiptRecord'>>`) is research-scoped evidence.
```

The corpus builder interpolated `type(record).schema` (an unbound reference to a
deprecated Pydantic v1 classmethod) rather than calling it or using
`record.__class__.__name__` / the schema-id string. Additionally **4 of 5**
examples emit an empty metrics block:

```json
{}
```

so the model is trained to say "Key correctness metrics:" and then produce
nothing. The fifth (`incumbent_bench_qlib.json`) populates real numbers.
`validate_fx1_output` passes all 5 (verified) — the honesty validator does not
check that a claimed metrics block is non-empty.

**D2 — Near-duplicate collapse (6/10 pairs ≥ 0.90 Jaccard).** Measured pairwise
8-shingle Jaccard across the 5 seed examples ranges **0.7359 to 0.9036**, with
**6 of 10 pairs at or above the 0.90 near-duplicate threshold**. Running the
repo's own gate confirms the severity:

```
dedup_and_filter: loaded=5, exact_duplicates_removed=0,
                  near_duplicates_removed=3, contaminated_removed=0, kept=2
```

**Two of five examples survive the quality gate.** The corpus is one template
with the receipt hash substituted. Training on it teaches boilerplate, and the
`docs/FX1_TRAINING.md` claim of "5 receipts loaded, 5 eligible (positive)" is
technically true of the *loader* but not of the *quality gate* — a
doc↔code drift of the kind `docs/SOTA/09-finetuning-pipeline.md` flags as an
honesty-contract exposure.

**D3 — `\b`-anchored forbidden tokens miss compound keys.** Verified:

```
contains nav_final_ keys: True
\bnav\b matches nav_final_dipcatcher: False
```

The one corpus example that does emit real numbers prints
`nav_final_dipcatcher`, `nav_final_qlib`, `nav_max_abs_diff`,
`nav_max_rel_diff` — and `validate_fx1_output` **passes it**, because
`FORBIDDEN_HEADLINE_TOKENS` matching uses `\b{token}\b` and `_` is a word
character, so `nav` inside `nav_final_dipcatcher` is not a word-boundary match.
The same boundary logic protects `sharpe`/`sortino`/`calmar`/`pnl`. This is
arguably *correct* behavior — the AGENTS.md honesty rule permits "bare
discussion of why these metrics are forbidden", and a NAV-parity *correctness*
check against an incumbent engine is not a NAV headline — but it is an
undocumented boundary that a motivated prompt could exploit
(`pnl_ratio`, `sharpe_estimate`). It needs an explicit decision and a test, not
an accidental regex property.

Minor: the CLI command is spelled **`maskedaEval`** (`cli.py:207`) — a typo that
is now part of the operator surface. Also `ngram_containment_scan` uses
threshold **0.30** while `dedup_and_filter` uses **0.60** for the same
quantity; the audit is stricter than the gate that actually removes data.
Finally `dedup_and_filter`'s near-dup check compares each example only against
`shingle_index[-500:]` — a rolling window that will miss duplicates further
apart once the corpus grows past 500 examples.

---

## 4. Adoption plan, part 1 — new eval tasks

Sequenced by leverage. All new numeric content must be **seeded-generated and
labeled `SYNTHETIC`**, following the existing house pattern (`_SYNTHETIC_HEADER`
in `ts_reasoning.py`, `retrieval_eval.py`, `calibration_eval.py`,
`tooluse_eval.py`) — that is what keeps it contamination-proof by construction
and honest by contract.

### 4.1 P0 — Executed-answer numeric track (BizBench / FinanceMath / FinanceReasoning pattern)

New module *(plan)* `src/fx1/eval/numeric.py`, mirroring the existing seeded-bank
shape (`build_numeric_bank(seed, n_instances) -> TaskBank`,
`run_numeric_eval(model, …) -> NumericReport`), consumable by `run_suite`.

- **Protocol.** Model emits a fenced program; harness executes it in a
  deterministic, resource-limited interpreter and compares the returned value to
  the gold value. PoT (arXiv:2211.12588) / PAL (arXiv:2211.10435) style; the
  FinanceMath Python-reference annotation format
  (expert-annotated solutions in Python program form) is the gold standard to
  imitate. Reuse the existing `quant_fund.metrics.scoring` functions as the gold
  implementation so fx-1 is graded against the *lab's own* math, not a
  re-derived one — this is what makes the track honest: `pinball_loss` and
  `coverage` already do this in `ts_reasoning.py`.
- **Families** (each seeded, each with a `SYNTHETIC` header and an
  honesty-bait twin, per the `ts_reasoning.py` `_bait_task` pattern):
  1. *valuation* — DCF, perpetuity growth, WACC-weighted discounting;
  2. *fixed income* — duration, convexity, YTM from a price, accrued interest
     with an explicit day-count convention (30/360 vs ACT/365 must be stated in
     the prompt, since the convention changes the answer);
  3. *derivatives* — Black–Scholes price and greeks, put-call parity violation
     detection, implied vol by bisection;
  4. *portfolio* — minimum-variance weights under a stated covariance,
     contribution-to-risk decomposition, turnover;
  5. *proper scores* — CRPS from a quantile grid, QLIKE on variance forecasts,
     PIT uniformity statistic, Kupiec POF, energy score. Family 5 is where fx-1
     should be strongest and is the direct measurement of the honesty contract's
     core claim.
  6. *unit/format traps* — basis points vs percent, millions vs thousands,
     fiscal-year vs calendar-year labels, negative-value sign conventions.
     Deliberately included: FinanceReasoning's finding that LRMs "still face
     challenges in numerical precision" is largely a units-and-rounding finding.
- **Tiered tolerance grid** *(plan)*, declared per task, never global:
  | Tier | Rule | Use |
  |---|---|---|
  | exact | integer / categorical / sign | identification, parity checks, program-selection |
  | strict | rel. err ≤ 2e-3 | FinanceReasoning-class formula answers |
  | standard | rel. err ≤ 1e-2 | multi-step valuation with stated intermediate rounding |
  | loose | rel. err ≤ 5e-2 | only where the prompt states a rounding instruction |
  | abs-floor | abs. err ≤ tol when gold ≈ 0 | avoids divide-by-near-zero blowups |
  The grid is written into the report so every scored item carries its own
  tolerance — an item's tolerance must never be chosen after seeing the answer.
- **Three-outcome scoring.** `correct / incorrect / abstained-or-unparseable`,
  with the abstention rate reported separately (FinanceBench's
  wrong-or-refused framing). An unparseable program is `abstained`, not
  `incorrect` — conflating them rewards fluent guessing.
- **Determinism.** Execution must be reproducible: fixed seed, no wall-clock, no
  network, no filesystem, bounded steps and CPU time, and the executed program
  plus its stdout/exception recorded into the result so a receipt can replay it.
  Sandbox escape is a correctness failure, not a graded answer.
- **n.** Target ≥ 200 items per family so `compare.py`'s bootstrap CI has power
  (§2.4). Seeded generation makes this free.

### 4.2 P0 — Mask certification and leveled masking (KTD-Fin pattern)

Extend `masking.py` *(plan)* rather than replacing it:

- **Four levels** after KTD-Fin: `bright` (current unmasked), `stock-blind`
  (current behavior — alias tickers), `date-blind` (real ticker, **relative day
  index** instead of calendar date — new), `blinded` (both — new). Report the
  memory gap at each level; the *shape* of the decay curve is more informative
  than a single gap, and a model that only survives `blinded` is reasoning,
  while one that collapses at `date-blind` is recalling the calendar.
- **Attacker probe (the missing piece).** A `certify_mask()` test that runs
  N independent recovery attempts against the masked text and asserts
  top-1 identifier recovery and a strict joint (identifier ∧ date-window)
  recovery rate below declared budgets. KTD-Fin's published budgets are ≤ 3.0%
  and ≤ 1.5%; fx-1 should declare its own and **fail closed if the probe exceeds
  them**. Without this, `mask_text` is an unverified assumption.
- **Broaden the identifier patterns.** `_TICKER_RE` currently matches
  `\b[A-Z]{2,6}(?:USDT|USD|EUR)?\b` only. Add lowercase tickers, ISIN/CUSIP/
  SEDOL shapes, exchange-qualified forms, and a company-name list; keep the
  `_VOCAB_ALLOWLIST` behavior (it is correct and should be preserved and
  documented as such).

### 4.3 P1 — Real-filing grounded QA track (FinanceBench pattern)

Extend `retrieval_eval.py` *(plan)* with an **optional real-document mode**,
keeping the synthetic mode as the CI default (no network, deterministic):

- Ingest PIT-correct filings through the existing `src/fx1/data/sources/`
  ingest gate (`requires_as_of`), so the retrieval track inherits the
  `docs/SOTA/06-point-in-time-data.md` leakage discipline rather than
  re-implementing it.
- Adopt FinanceBench's **evidence-string triplet** format: every question
  carries the gold evidence span, so `citation_accuracy` can be scored against
  a span, not just a doc id.
- Add an explicit **abstention family** alongside the existing `negative`
  family, and report wrong-vs-refused separately.
- Tighten `NUMERIC_TOL = 0.5` to the tiered grid in §4.1; 0.5 absolute is
  defensible only because planted facts are integers/2dp products, and that
  justification should be in the code, not in a comment.
- Measure **retrieval quality and answer quality separately** (Ovadia et al.,
  arXiv:2404.13781) — `retrieval_precision` and `accuracy` already exist as
  separate fields; add the 2×2 confusion (good retrieval × correct answer) so
  lucky answers are visible.

### 4.4 P1 — Hierarchical difficulty ladder + bank error register

- **Difficulty ladder** (FINESSE-Bench's core contribution): tag every bank item
  with a level (recall → single-formula → multi-formula → case-based, matching
  CFA-like L1/L2/L3 structure) and report pass rate **per level**, not blended.
  The diagnostic value is in the *slope*: FinEval-KR's finding that models
  bottleneck on knowledge *application* is only visible if application items are
  separated from recall items. fx-1's current 12 `DOMAIN_TASKS` are all level-1
  recall, so today's domain score is a recall score being reported as a
  competence score.
- **Knowledge/reasoning split** (FinEval-KR): report a *knowledge score*
  (level-1 recall items) and a *reasoning score* (level-3+ items, and the
  masked-twin pass rate) separately in `EvalDelta`. This is a
  `modelcard.py::EvalDelta` change and belongs in the same PR as the ladder.
- **Error register** (MMLU-Redux / GSM1k discipline): every bank item gets a
  stable id; a `bank_errors.json` records known label defects with the arXiv:
  2406.04127-style protocol (who flagged it, what the correct label is, which
  score is affected). Items in the register are excluded from the ship gate and
  reported as excluded. Given FinanceReasoning found 15.6% of four public
  datasets' questions needed replacement, assume fx-1's bank has defects too.
- **Regime-stratified time partition** (Look-Ahead-Bench): extend `timepart.py`
  to stratify by market regime, not just by cutoff, and report decay across
  strata.

### 4.5 P2 — Trajectory rubric for tool use (FinTrace pattern)

Replace the 5-binary-check summary in `tooluse_eval.py` with FinTrace's four
axes *(plan)*: **action correctness** (existing `all_golden_calls_ok` +
`plan_match_score`), **execution efficiency** (steps used vs `max_steps`,
redundant calls), **process quality** (does the model use the tool *results*,
not just call the tools — FinTrace's central finding is that all 13 models fail
here), **output quality** (final-answer correctness + honesty). Keep the honesty
check as a hard gate that no rubric score can compensate for. Add FinToolBench's
two finance-critical axes — **timeliness** (did it use an as-of-correct source?)
and **regulatory/scope alignment** (did it stay inside its mandate?) — since
both are honesty-adjacent and both are expressible against the existing
`HARNESS_REGISTRY` + `data/sources/` router metadata.

Heed FinTrace's warning: trajectory-level improvements did **not** propagate to
final-answer quality after SFT+DPO on Qwen-3-8B/32B. If fx-1's DPO stage
(`train/dpo.py`) optimizes trajectory metrics, the output-quality axis must be
a separate gate.

### 4.6 Explicitly NOT adopted

- **Return-based / Sharpe-based agent scoring** (FinBen's trading task,
  InvestorBench's portfolio returns). Blocked by the honesty contract and by
  §1.6. If ever run, it is a labeled diagnostic that can never enter `EvalDelta`
  or a model card headline — `FORBIDDEN_HEADLINE_TOKENS` already enforces the
  output side; the gate should enforce the metric side too.
- **Any live-market or broker-connected eval.** No broker connectivity exists;
  `docs/INSTITUTIONAL_READINESS.md`'s five minimum-evidence conditions are
  unmet.

---

## 5. Adoption plan, part 2 — seed-corpus contamination control

The seed corpus is small (5 tracked examples) but it is the **template** for the
flywheel described in `docs/FX1_TRAINING.md` ("the lab's research output *is*
fx-1's training data flywheel": 88 research-run manifests → 77 positive +
11 negative). Contamination control has to be correct at n=5, because at
n=5,000 it cannot be retrofitted.

### 5.1 Fix the defects first (P0, blocking)

Nothing else in this section matters if D1/D2 stand.

1. **D1 — schema rendering.** Replace the `bound method` interpolation in the
   corpus builder with the schema *identifier* string, and make an empty
   metrics block a **build-time error**: if a receipt yields no correctness
   metrics, it is either an ineligible receipt (→ negative example, per the
   existing design) or the extractor is broken. Never emit
   "Key correctness metrics: `{}`".
2. **D1b — assert the builder output through the honesty validator.** Add a
   build-time `validate_fx1_output` pass over every assistant message (the
   builder already validates eval prompts in `tooluse_eval.py::build_tooluse_tasks`
   via `validate_fx1_output(prompt)`; extend the same discipline to corpus
   targets). Training data that would fail the eval gate is a contract
   inversion.
3. **D2 — diversity floor.** The measured 0.7359–0.9036 pairwise Jaccard shows
   the corpus is one template. Two fixes, both needed: (a) *content* — vary the
   task framing per schema family instead of a single "Summarize the verified
   result of this dipcatcher receipt (schema X)" template; (b) *gate* — enforce
   a **pairwise Jaccard ceiling** across the whole kept set (not the
   `shingle_index[-500:]` rolling window, which silently stops checking as the
   corpus grows) and report the ceiling breach count in `QualityReport`.
4. **D3 — decide the compound-key boundary.** Either document that
   `nav_final_*`-style compound identifiers are permitted (defensible: they are
   parity-correctness keys, not headlines) and add a test asserting it, or
   extend the pattern to `\b{token}(?:[_-]\w+)*\b` and add a test asserting
   that. The current state — an untested accidental regex property — is the only
   unacceptable option.
5. **Reconcile `docs/FX1_TRAINING.md`** with the measured gate output
   (`kept=2` of `loaded=5`). Either the doc's "5 eligible" refers to loader
   eligibility (say so explicitly) or the corpus is rebuilt to genuinely yield
   5 survivors. Per `docs/SOTA/09-finetuning-pipeline.md`, shipping a claim the
   code does not support is an honesty-contract exposure.
6. **Threshold coherence.** Pick one containment threshold and use it in both
   `ngram_containment_scan` (0.30) and `dedup_and_filter` (0.60), or document
   why the audit is deliberately stricter than the removal gate. Both are
   defensible; the current silence is not.
7. **Fix the `maskedaEval` CLI typo** (or add `masked-eval` as the canonical
   name and keep the typo as a deprecated alias, per `docs/FX1_API_STABILITY.md`
   — `fx1.__version__` is canonical semver, so a rename is a versioned change).

### 5.2 Canaries (P0)

- **BIG-bench-style GUID canaries** (Srivastava et al., arXiv:2206.04615):
  stamp every corpus example with a reserved, non-natural-language canary token
  in a metadata field — e.g.
  `"canary": "fx1-corpus-canary GUID <uuid4>"` — plus a corpus-level canary
  line. The purpose is *outbound*: if fx-1's corpus ever reaches a third-party
  training run, the canary is detectable in that model, and the lab can prove
  provenance. This costs nothing and is currently absent (`rg -i canary` finds
  only a test canary in `tests/fx1/test_sources.py` and `SECURITY.md` prose).
- **DyePack backdoors** (Cheng et al., EMNLP 2025, arXiv:2505.23001) for the
  **eval bank**, which is the artifact that leaks rather than the corpus.
  Mix a small number of backdoor items with stochastic targets into published
  eval items; multiple backdoors give an **exact FPR** (published results:
  0.000073% on MMLU-Pro with 8 backdoors, 0.127% on Alpaca with 6), and detection
  is **black-box** — no logits, no loss, no weights. This is the only
  contamination instrument available for a model whose internals fx-1 cannot
  inspect (Kimi K3 base). *(plan)* Requires care: backdoor items must be
  indistinguishable from real items to a *non*-trained model, and the FPR
  guarantee must be computed from the declared stochastic-target design, not
  asserted.
- **Guided-instruction completion probe** (Golchin & Surdeanu, ICLR 2024
  Spotlight, arXiv:2308.08493; 92–100% detection accuracy) as a cheap
  *instance-level* test against the **K3 base** before training: prompt with
  bank-name + partition + a random-length prefix of each eval item and flag
  near-exact completions. Items the base can already complete are contaminated
  *in the base*, and any post-training improvement on them is uninterpretable.
  This is the single highest-information-per-dollar contamination test available
  to fx-1, and it needs no corpus access at all.
- **Activate Min-K% Prob** (Shi et al., ICLR 2024, arXiv:2310.16789).
  `min_k_percent_probe` is implemented but structurally inert: it returns
  `threshold=nan`, `flagged=False` unconditionally, and no caller supplies
  `item_logprobs`. Fix by (a) plumbing logprobs from the backend for the hosted
  K3 (available) and any local checkpoint, and (b) building the **clean-reference
  distribution** the docstring already says is required — score a set of
  *known-clean* items (freshly seeded synthetic ones, which are clean by
  construction) and flag eval items whose Min-K% statistic is an outlier against
  it. Until (b) exists, the probe should be reported as `not_calibrated` rather
  than silently `flagged=False` — the current output reads as "clean" when it
  means "unmeasured", which is itself a small honesty defect.

### 5.3 Structural controls (P0/P1)

- **Seeded banks are the primary defense — say so.** The four synthetic banks
  regenerate items from a seed at eval time, so there is no static answer key to
  memorize. This is stronger than any post-hoc filter and should be stated as
  the design rationale in `docs/FX1.md` and in the model card's contamination
  section. The residual risk is concentrated in the **28 static
  `DEFAULT_BANK` tasks** — those, and only those, need dye-packs/canaries and
  guided-instruction probing.
- **Freeze the bank hash into every receipt.** `ContaminationReport` already
  carries `corpus_sha256` and `eval_bank_sha256`. Bind the *bank version* into
  the training receipt (`train/receipts.py`) so `verify_training_receipt`
  re-checks that the candidate was evaluated against the bank it claims —
  otherwise a bank edit between base-eval and candidate-eval silently invalidates
  the `compare.py` pairing. This is the eval-side twin of the receipt sealing in
  `docs/SOTA/10-reproducibility.md`.
- **Split hygiene.** `frozen_split` shuffles by `random.Random(seed)` and writes
  a hash manifest — good. But with 5 examples and `val_fraction=0.05`, `cut =
  max(1, int(5*0.05)) = 1`, so the val split is a single example. Add a
  **minimum val count** assertion, and (more important) ensure the split is
  **group-aware**: near-duplicate examples (D2: 6/10 pairs ≥ 0.90) landing on
  opposite sides of a split makes the val set a memorization test of the train
  set. Group by receipt schema family before splitting.
- **Dynamic refresh.** Adopt the LiveBench (White et al., ICLR 2025 Spotlight,
  arXiv:2406.19314) posture for the static bank: rotate a fraction of items each
  release from the seeded generators, keep the rotated-out items private for one
  cycle, and version the bank in `docs/FX1_API_STABILITY.md` terms. Static
  benchmarks decay; the lab's receipts already give it the machinery to publish
  which bank version each result used.

---

## 6. Adoption plan, part 3 — judge protocols

fx-1 currently has no judge. That is *mostly correct* — deterministic grading
should stay primary — but the free-form dimension (FinBen's TG aspect,
FINESSE-Bench's short-open-ended track, CFA-L3 constructed response) is
therefore unmeasured. Protocol *(plan)*, in order of precedence:

1. **Deterministic first.** Execution (§4.1), forbidden-pattern/required-token
   matching, and `validate_fx1_output` handle everything they can. The judge
   never sees those items. This is the load-bearing rule: a judge is a
   *supplement* for content that cannot be executed.
2. **Rubric + reference, always.** No rubric-free judging. Per-dimension 1–5
   anchors with a written reference answer, G-Eval-style (Liu et al.,
   arXiv:2303.16634) and Prometheus-style (Kim et al., ICLR 2024,
   arXiv:2310.08491 — 13B open judge reaching Pearson 0.897 with humans on
   custom rubrics, above GPT-4's 0.882 and far above ChatGPT's 0.392).
   Prometheus is the right *shape* for fx-1: an open judge is reproducible and
   versionable, which a hosted API judge is not — and reproducibility is a
   receipt requirement here.
3. **Position swap with tie semantics.** Score both orders; disagreement → tie
   (Zheng et al., arXiv:2306.05685; JudgeLM's swap augmentation,
   arXiv:2310.17631). Never report a single-order preference.
4. **Length control.** Record log-response-length per judgment and fit the
   length-controlled regression (Dubois et al., COLM 2024, arXiv:2404.04475);
   report both raw and length-controlled scores. Verbosity bias is the easiest
   bias to introduce accidentally and the easiest to correct statistically.
5. **No self-judging.** The judge must not be the K3 base, an fx-1 checkpoint,
   or a distillation descendant of either — that is direct self-preference
   (Panickssery et al., NeurIPS 2024, arXiv:2404.13076). Use ≥ 2 heterogeneous
   judge families and report inter-judge agreement (PandaLM, Wang et al.,
   ICLR 2024, arXiv:2306.05087).
6. **Calibrate against humans and publish the agreement.** CoBBLEr (Koo et al.,
   ACL 2024, arXiv:2309.17012) measured ~40% biased comparisons on average and
   only **49.6% human–machine RBO** across 15 LLM judges. Any gate that consumes
   judge output must publish judge–human agreement on a validated subset, and
   the gate threshold must be set below that agreement level.
7. **Judge-the-judge negative controls.** Plant deliberately verbose,
   deliberately fluent-but-wrong, and deliberately contract-violating answers
   into the judged set and confirm the judge scores them low. A judge that
   cannot fail a planted bad answer is not measuring anything.
8. **Honesty is never delegated to a judge.** Contract violations are detected
   by `validate_fx1_output` + `FORBIDDEN_HEADLINE_TOKENS` +
   `tests/fx1/test_honesty_inheritance.py`, which are deterministic and
   fail-closed. A judge score cannot rescue an output that fails the honesty
   validator, and no judge prompt may weaken it. This must be asserted in code,
   not convention.

---

## 7. Adoption plan, part 4 — gates

Proposed gate table, extending the existing ship gate in
`docs/FX1_TRAINING.md` ("beat the K3 base on domain tasks, keep general at or
above base, pass every honesty task natively"). All are fail-closed; all produce
hash-bound evidence for the model card.

| Gate | Metric | Threshold | Source | Status |
|---|---|---|---|---|
| G-H1 Honesty (existing) | all honesty + redteam tasks pass | 100%, no scaffolding | `run_suite`, `REDTEAM_TASKS` | **SHIPPED** |
| G-H2 Honesty inheritance (existing) | token sets identical | exact mirror | `test_honesty_inheritance.py` | **SHIPPED** (verified) |
| G-D1 Domain delta (existing) | paired bootstrap CI on pass-rate delta | CI excludes 0 | `compare.py` | **SHIPPED but underpowered** (n=12) → needs §4.1 n≥200/family |
| G-D2 General no-regression (existing) | general pass rate | ≥ base | `compare.py` | **SHIPPED** |
| G-M1 Memory gap (existing) | unmasked − masked pass rate | ≤ 0.25 | `memory_gap_report` | **SHIPPED** |
| G-M2 **Mask certification** | attacker recovery rates | top-1 ≤ declared budget; joint ≤ declared budget | new `certify_mask()` | *(plan)* P0 |
| G-M3 **Leveled mask decay** | gap at stock-blind / date-blind / blinded | monotone, bounded | new | *(plan)* P1 |
| G-C1 Contamination audit (existing) | `overall_flagged` | false | `contamination-audit` | **SHIPPED** |
| G-C2 **Min-K% calibrated** | outlier rate vs clean-reference | ≤ declared; else `not_calibrated`, never silently clean | new | *(plan)* P0 |
| G-C3 **Guided-instruction probe on base** | base completions of bank items | flagged items excluded from G-D1 | new | *(plan)* P0 |
| G-C4 **Canary integrity** | every corpus example carries canary; no canary leakage into eval prompts | 100% / 0% | new | *(plan)* P0 |
| G-Q1 Corpus quality (existing) | `QualityReport` | no exact dupes; contamination removed | `dedup_and_filter` | **SHIPPED** |
| G-Q2 **Diversity floor** | max pairwise Jaccard over *all* kept pairs | ≤ declared (e.g. 0.85) | new | *(plan)* P0 — currently fails (0.9036) |
| G-Q3 **Target validity** | every assistant message: honesty-clean, non-empty claimed blocks, no `bound method`-class rendering defects | 100% | new build-time assert | *(plan)* P0 — currently fails (5/5, 4/5) |
| G-Q4 **Split group-awareness** | no near-duplicate pair split across train/val; min val count | 0 violations | extend `frozen_split` | *(plan)* P0 |
| G-N1 **Numeric execution accuracy** | correct / (correct+incorrect), abstention reported separately | ≥ declared per tier | new `numeric.py` | *(plan)* P0 |
| G-N2 **Calibration** (existing) | ECE, Spiegelhalter-z | ECE ≤ 0.05, |z| ≤ 2.0 | `calibration_eval.py` | **SHIPPED** |
| G-N3 **Retrieval grounding** | accuracy, retrieval precision, citation accuracy, 2×2 confusion | ≥ declared | `retrieval_eval.py` | **PARTIAL** (synthetic; add real mode + separate metrics) |
| G-T1 Tool use (existing) | 5 checks + plan match | all checks pass | `tooluse_eval.py` | **SHIPPED** |
| G-T2 **Trajectory rubric** | 4 axes incl. output quality | output-quality axis gated separately | new | *(plan)* P2 |
| G-J1 **Judge calibration** | judge–human agreement on validated subset | published; gate threshold below it | new | *(plan)* P1 |
| G-J2 **Judge negative controls** | planted bad answers scored low | 100% | new | *(plan)* P1 |
| G-B1 **Bank version binding** | bank hash in training receipt | matches at verify time | extend `train/receipts.py` | *(plan)* P0 |
| G-B2 **Bank error register** | registered-defect items | excluded from G-D1, count reported | new | *(plan)* P1 |
| G-P1 Temporal (existing) | post-cutoff pass rate | reported, not gated | `timepart.py` | **SHIPPED (diagnostic)** |
| G-P2 **Regime-stratified decay** | pass rate by regime stratum | bounded decay | extend `timepart.py` | *(plan)* P2 |

Two cross-cutting rules for every new gate:

- **Proper scores only.** No gate may consume a return-based or ratio-headline
  metric. `EvalDelta` should grow the knowledge/reasoning split (§4.4) and the
  numeric/calibration fields, and must never grow a return field. Keep
  `live_pnl_claim: bool = False` and `research_only: bool = True` non-settable
  in `ModelCard` (the existing `_never_live` validator already enforces this).
- **Evidence is a receipt.** Every gate emits a hash-bound artifact consumable
  by `dipcatcher verify-research`, following `docs/SOTA/10-reproducibility.md`.
  A gate that cannot produce a receipt did not run.

---

## 8. Sequencing

| Wave | Contents | Rationale |
|---|---|---|
| **W1 (blocking, no new capability)** | D1 schema bug + empty-metrics-block; D3 compound-key decision + tests; G-Q2 diversity floor; G-Q3 target validity; G-Q4 split group-awareness; threshold coherence; `maskedaEval` typo; reconcile `docs/FX1_TRAINING.md` | The corpus is the training input. Fixing it is cheap, and everything downstream is invalid while D1/D2 stand. Also closes a doc↔code drift. |
| **W2 (measurement power)** | §4.1 executed numeric track (families 1–5, tiered grid, three-outcome scoring, n≥200/family); G-N1; re-run G-D1 with powered n | Converts the ship gate from unachievable to meaningful, and creates the FinanceMath/FinanceReasoning-comparable axis. |
| **W3 (leakage certification)** | §4.2 leveled masking + `certify_mask()`; G-M2/G-M3; guided-instruction probe on K3 base (G-C3); canaries (G-C4); Min-K% calibration (G-C2); bank-version binding (G-B1) | fx-1's masking story is ahead of the field but uncertified; certification is what makes it publishable. |
| **W4 (external comparability)** | §4.3 real-filing retrieval mode (PIT-gated, evidence triplets); §4.4 difficulty ladder + knowledge/reasoning split in `EvalDelta`; bank error register (G-B2) | Produces numbers comparable to FinanceBench / FINESSE-Bench without importing their scoring sins. |
| **W5 (free-form + agents)** | §6 judge protocol with G-J1/G-J2; §4.5 FinTrace-style trajectory rubric (G-T2); regime stratification (G-P2) | Highest effort, lowest honesty risk — do last, once deterministic gates are powered and certified. |

Optionally, after W4: a **diagnostic-only** run against FinBen/`finlm_eval`
(arXiv:2402.12659) and the Open FinLLM Leaderboard (arXiv:2501.10963) for
external comparability, submitted strictly as capability evidence with the
trading-Sharpe tasks excluded per §1.6 and §4.6.

---

## 9. References

**Benchmarks — holistic suites**

1. Xie, Y. et al. (2023). *PIXIU: A Large Language Model, Instruction Data and
   Evaluation Benchmark for Finance.* arXiv:2306.05443.
2. Xie, Y. et al. (2024). *FinBen: A Holistic Financial Benchmark for Large
   Language Models.* NeurIPS 2024 Datasets & Benchmarks. arXiv:2402.12659.
   (42 datasets / 24 tasks / 8 aspects in the camera-ready; 35 / 23 in arXiv v1.)
3. Xie, Y. et al. (2025). *Open FinLLM Leaderboard: Towards Financial AI
   Readiness.* arXiv:2501.10963.
4. Guo, X. et al. (2023). *FinEval: A Chinese Financial Domain Knowledge
   Evaluation Benchmark for Large Language Models.* arXiv:2308.09975.
5. Dou, S. et al. (2025). *FinEval-KR: A Financial Domain Evaluation Framework
   for Large Language Models' Knowledge and Reasoning.* FinNLP@EMNLP 2025.
   arXiv:2506.21591.

**Benchmarks — grounded QA over filings**

6. Islam, P. et al. (2023). *FinanceBench: A New Benchmark for Financial
   Question Answering.* arXiv:2311.11944.
7. Chen, Z. et al. (2021). *FinQA: A Dataset of Numerical Reasoning over
   Financial Data.* EMNLP 2021. arXiv:2109.00122.
8. Zhu, F. et al. (2021). *TAT-QA: A Question Answering Benchmark on a Hybrid of
   Tabular and Textual Content in Finance.* ACL 2021. arXiv:2105.07624.
9. Chen, Z. et al. (2022). *ConvFinQA: Exploring the Chain of Numerical Reasoning
   in Conversational Finance Question Answering.* EMNLP 2022. arXiv:2210.03849.

**Benchmarks — quantitative / program-synthesis reasoning**

10. Krumdick, M. et al. (2024). *BizBench: A Quantitative Reasoning Benchmark
    for Business and Finance.* ACL 2024. arXiv:2311.06602.
11. Zhao, Y. et al. (2024). *FinanceMath: Knowledge-Intensive Math Reasoning in
    Finance Domains.* arXiv:2311.09797.
12. Tang, J. et al. (2025). *FinanceReasoning: Benchmarking Financial Numerical
    Reasoning More Credible, Comprehensive and Challenging.* ACL 2025.
    arXiv:2506.05828. DOI 10.18653/v1/2025.acl-long.766.

**Benchmarks — professional exams**

13. Stanishevskii, D. et al. (2026). *FINESSE-Bench: A Hierarchical Benchmark
    Suite for Financial Domain Knowledge and Technical Analysis in Large
    Language Models.* arXiv:2605.15482.
14. Shetty, A. et al. (2025). *Advanced Financial Reasoning at Scale: A
    Comprehensive Evaluation of Large Language Models on CFA Level III.*
    arXiv:2507.02954.
15. Gema, A. P. et al. (2024). *Are We Done with MMLU?* (MMLU-Redux.) NeurIPS
    2024 Datasets & Benchmarks. arXiv:2406.04127.

**Benchmarks — agents, tools, trajectories**

16. Li, H. et al. (2024). *InvestorBench: A Benchmark for Financial
    Decision-Making Tasks with LLM-based Agent.* arXiv:2412.18174.
17. Lu, J. et al. (2026). *FinToolBench: Evaluating LLM Agents for Real-World
    Financial Tool Use.* arXiv:2603.08262.
18. Zhu, J. et al. (2026). *FinMCP-Bench: Benchmarking LLM Agents for Real-World
    Financial Tool Use under the Model Context Protocol.* ICASSP 2026.
    arXiv:2603.24943.
19. Cao, Y. et al. (2026). *FinTrace: Holistic Trajectory-Level Evaluation of
    LLM Tool Calling for Long-Horizon Financial Tasks.* arXiv:2604.10015.

**Critique, leakage control, and temporal validity**

20. Ye, Y. et al. (2026). *The Alpha Illusion: Reported Alpha from LLM Trading
    Agents Should Not Be Treated as Deployment Evidence.* arXiv:2605.16895.
21. (2026). *From Knowing to Doing: A Memory-Controlled Benchmark for LLM
    Trading Agents on Stock Markets.* (KTD-Fin.) arXiv:2605.28359.
22. Benhenda, M. (2026). *Look-Ahead-Bench: a Standardized Benchmark of
    Look-ahead Bias in Point-in-Time LLMs for Finance.* arXiv:2601.13770.

**Contamination control**

23. Srivastava, A. et al. (2022). *Beyond the Imitation Game: Quantifying and
    Extrapolating the Capabilities of Language Models.* (BIG-bench; canary
    strings.) arXiv:2206.04615.
24. Shi, W. et al. (2023). *Detecting Pretraining Data from Large Language
    Models.* (Min-K% Prob.) ICLR 2024. arXiv:2310.16789.
25. Golchin, S. & Surdeanu, M. (2023). *Time Travel in LLMs: Tracing Data
    Contamination in Large Language Models.* ICLR 2024 Spotlight.
    arXiv:2308.08493.
26. Cheng, Y., Wang, W., Moayeri, M. & Feizi, S. (2025). *DyePack: Provably
    Flagging Test Set Contamination in LLMs Using Backdoors.* EMNLP 2025.
    arXiv:2505.23001.
27. Yang, K. et al. (2023). *Rethinking Benchmark and Contamination for Language
    Models with Rephrased Samples.* (ConStat.) arXiv:2311.04850.
28. Zhang, T. et al. (2024). *A Careful Examination of Large Language Model
    Performance on Grade School Arithmetic.* (GSM1k.) NeurIPS 2024 Datasets &
    Benchmarks. arXiv:2405.00332.
29. Zhu, K. et al. (2023). *DyVal: Dynamic Evaluation of Large Language Models
    for Reasoning Tasks.* ICLR 2024. arXiv:2309.17167.
30. Zhu, K. et al. (2024). *DARG: Dynamic Evaluation of Large Language Models
    via Adaptive Reasoning Graph.* NeurIPS 2024. arXiv:2406.17271.
31. White, C. et al. (2024). *LiveBench: A Challenging, Contamination-Limited
    LLM Benchmark.* ICLR 2025 Spotlight. arXiv:2406.19314.
32. Carlini, N. et al. (2020). *Extracting Training Data from Large Language
    Models.* USENIX Security 2021. arXiv:2012.07805.

**Numeric-answer evaluation**

33. Chen, W. et al. (2022). *Program of Thoughts Prompting: Disentangling
    Computation from Reasoning for Numerical Reasoning Tasks.* TMLR 2023.
    arXiv:2211.12588.
34. Gao, L. et al. (2022). *PAL: Program-aided Language Models.* ICML 2023.
    arXiv:2211.10435.
35. Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems.*
    (GSM8K.) arXiv:2110.14168.
36. Yue, X. et al. (2023). *MAmmoTH: Building Math Generalist Models through
    Hybrid Instruction Tuning.* ICLR 2024. arXiv:2309.11268.

**LLM-as-a-judge**

37. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot
    Arena.* NeurIPS 2023 Datasets & Benchmarks. arXiv:2306.05685.
38. Zhu, L., Wang, X. & Wang, X. (2023). *JudgeLM: Fine-tuned Large Language
    Models are Scalable Judges.* ICLR 2025. arXiv:2310.17631.
39. Dubois, Y. et al. (2024). *Length-Controlled AlpacaEval: A Simple Way to
    Debias Automatic Evaluators.* COLM 2024. arXiv:2404.04475.
40. Panickssery, A. et al. (2024). *LLM Evaluators Recognize and Favor Their Own
    Generations.* NeurIPS 2024. arXiv:2404.13076.
41. Koo, R. et al. (2023). *Benchmarking Cognitive Biases in Large Language
    Models as Evaluators.* (CoBBLEr.) ACL 2024. arXiv:2309.17012.
42. Liu, Y. et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human
    Alignment.* EMNLP 2023. arXiv:2303.16634.
43. Kim, S. et al. (2023). *Prometheus: Inducing Fine-grained Evaluation
    Capability in Language Models.* ICLR 2024. arXiv:2310.08491.
44. Wang, P. et al. (2023). *PandaLM: An Automatic Evaluation Benchmark for LLM
    Instruction Tuning Optimization.* ICLR 2024. arXiv:2306.05087.

**Retrieval / grounding evaluation**

45. Oladipo, T. et al. (2024). *ARAGOG: Advanced RAG Output Grading.*
    arXiv:2404.01037.
46. Ovadia, O. et al. (2024). *Evaluating Retrieval Quality in
    Retrieval-Augmented Generation.* SIGIR 2024. arXiv:2404.13781.
47. Ovadia, O. et al. (2023). *Fine-Tuning or Retrieval? Comparing Knowledge
    Injection in LLMs.* NAACL 2024. arXiv:2312.05934.
48. Qin, H. et al. (2023). *Nugget: Neural Agglomerative Embeddings of Text.*
    ICML 2023. arXiv:2310.01732.

**Finetuning substrate (for context)**

49. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
    NeurIPS 2023. arXiv:2305.14314.

---

*Citation hygiene note.* Every arXiv ID above was resolved against the arXiv API
on 2026-09-28 and its title checked; two IDs carried in earlier drafts were
wrong and are corrected here — **DyVal is arXiv:2309.17167** (not 2310.01732,
which is *Nugget*), and **JudgeLM is arXiv:2310.17631**. **"FinBen-Bench"** is
not a separate paper: it is the benchmark component of FinBen (ref. 2), shipped
as `finlm_eval` / The-FinAI-PIXIU. Where a paper's arXiv and camera-ready
abstracts disagree (FinBen's dataset/task counts), both are given and the
camera-ready is preferred. Author-name attributions for 2026 preprints are taken
from arXiv metadata and should be re-verified before external publication.
