# SOTA 20 — Research Workflow Engineering

Lane: **RESEARCH WORKFLOW ENGINEERING**. This note summarizes
state-of-the-art practice (2015–2026) for running a hypothesis-led research
program — pre-registration, experiment tracking, forking-path control,
literature grounding, result ledgers, systematic review, and a-priori power
analysis; maps the dipcatcher study lifecycle end-to-end against that
practice; names the friction points; and proposes an adoption plan
(machine-readable study-spec schema, ledger automation, power gate). It
modifies no code — it is the design record for the next workflow generation.

The honesty contract this lane serves: research results are **proper
scores** with receipts; every claim is reproducible from a receipt hash;
pending vs decided evidence is separated (ADR 038). This lane is about the
*process* that produces those receipts — making the study lifecycle
machine-checkable from idea to verdict, not just at the verdict.

---

## Part 1 — SOTA practice summaries

### 1.1 Study specs & pre-registration

**Why pre-register at all.** The replication crisis literature converged on a
single mechanism: *researcher degrees of freedom* — undisclosed flexibility in
data collection, exclusion rules, transformations, and stopping (Simmons,
Nelson & Simonsohn 2011, *JEP:G*). The quant analogue is the **garden of
forking paths** (Gelman & Loken 2013): a backtest that "looks preregistered"
because the config is a YAML file is not preregistered unless the *analysis
pipeline* was frozen before outcomes were observed. Bailey, Borwein, López de
Prado & Zhu (2017, *JPM*) formalize the same idea for finance:
**Pseudo-Mathematics and Financial Charlatanism** shows the deflated Sharpe
ratio collapsing as the number of *undocumented* configuration trials grows —
multiplicity lives in the workflow, not just in the statistics.

**The formats worth stealing:**

- **AsPredicted (2015)**: nine short fields — hypothesis, method, sample,
  exclusion rules, sample-size rule, stopping rule — signed by all authors
  *before* data collection. Its genius is the **sample-size rule** field: it
  forces an a-priori power statement ("we will collect N, or until M days of
  data, because…") instead of a post-hoc rationalization.
- **OSF Preregistration / Registered Reports** (Nosek et al. 2018, *Science*;
  Center for Open Science): structured JSON templates (hypotheses, secondary
  hypotheses, data-collection plan, analysis plan with exact code, deviations
  log). Two properties matter operationally: (1) registrations are
  **timestamped and immutable** — deviations are *appended*, never edited;
  (2) Registered Reports give **Stage 1 in-principle acceptance**: the
  *question and method* are judged before results exist, which is exactly the
  stage-gate discipline a research lab needs.
- **Clinical-trial registries** (ClinicalTrials.gov; ICMJE 2005 policy):
  prospective public registration as a *publication precondition*, with a
  machine-readable record (results sections posted separately from the
  design). The enforcement lesson: registration without a gate is decoration.
- **Pre-analysis plans in economics** (Casey, Glennerster & Miguel 2012,
  *QJE*): freeze the estimating equations and the decision rule ("primary
  outcome = X, powered at δ, family-wise α across K outcomes").

**Adaptation for backtests.** A quant study spec must freeze, at minimum:
data sources + hashes, universe, split/embargo dates, cost model, hypothesis
family, *proper-score* headline metric (per the repo honesty contract — not
Sharpe), the multiple-testing budget (how many trials will be run and
counted), and the kill/promotion thresholds. Everything else is a deviation
that gets logged. The repo already implements this idea in three places with
three schemas (see Part 2/3).

### 1.2 Experiment tracking: MLflow/W&B patterns and their limits

**MLflow's durable contribution** is the run → artifacts → model-version
chain: every run gets an immutable `run_id`; logged params/metrics are
append-only per run (corrections are new runs); the Model Registry layers
*mutable* labels over *immutable* versions. Since MLflow 2.9+/3.x, the
correct discipline is **aliases over stages**: stages (Staging/Production)
are deprecated in favor of named aliases (`champion`, `candidate`, `shadow`)
because aliases are first-class, auditable pointers with no implicit linear
promotion semantics (Zaharia et al., *Databricks* docs; MLflow 3.0 release
notes). W&B contributes the **run-comparison discipline**: saved searches
that pin a cohort, report tables that diff locked runs, and the "runs are
immutable, sweeps are declared YAML" separation.

**Where tracking stops being sufficient for research evidence.** Tracking
systems are *observability*: server-side mutable metadata, no content
hashing, no sealing. The 2023–2025 wave of ML-reproducibility work is
explicit that observability ≠ evidence: Pineau et al. (2021, *JMLR*) ML
Reproducibility Checklist asks for *code + data + environment hashes*; the
ACM Artifact Review and Badging v1.1 (2022) defines three badges
(available / evaluated / reproduced) that map onto receipt verification
tiers, not dashboards. **The pattern to adopt: tracking for exploration,
sealed receipts for claims, and a hard gate between them** (no promotion
without a verified receipt). The repo's `quant_fund.registry.mlflow_store`
already implements exactly this gate (`set_alias("champion")` requires
`promotion_is_approved()`); the gap is that *runs that produce no receipt*
are invisible to the ledger (Part 3, F5).

### 1.3 Hypothesis-led loops, forking paths, multiverse, stage gates

**The hypothesis-led loop.** Standard form in both academia and industry
labs: *idea → falsifiable hypothesis with an effect size → pre-registered
method → single analysis pass → decision rule executed mechanically →
deviation log*. Two industrial codifications are worth citing: the
**Google/Meta experiment-platform literature** (Tang et al. 2010, *KDD*,
overlapping experiments; Kohavi, Tang & Xu 2020, *Trustworthy Online
Controlled Experiments*) which enforces one-metric-that-matters + guardrails
+ pre-registered sample sizes; and **hypothesis-tree / issue-tree analysis**
in evidence-based management, where every branch of the tree is a testable
sub-hypothesis with a recorded verdict.

**Multiverse analysis** (Steegen et al. 2016, *Perspectives on Psychological
Science*; Silberzahn et al. 2018, *Psych. Science*, the many-analysts World
Cup study): instead of defending one analysis, run the *entire defensible
analysis space* and report the distribution of outcomes. In quant terms this
is Bailey & López de Prado's **Combinatorially Symmetric Cross-Validation**
and the CSCV **Probability of Backtest Overfitting** (Bailey et al. 2015,
*JPM*): PBO measures the chance that the in-sample-best configuration is
median-or-worse out of sample. The repo implements PBO, DSR, PSR, MinTRL,
SPA, e-values, and BH/stopped-e-BH (Part 2) — the statistical layer is
SOTA; the *workflow* layer around when those diagnostics get computed is the
subject of Part 3.

**Stage-gate kill criteria.** Pharma R&D (Paul et al. 2010, *Nature Reviews
Drug Discovery*) attributes its productivity problem partly to weak early
kill discipline; the fix is stage gates with *quantitative* go/no-go
criteria fixed before each stage (e.g., minimum effect size at minimum power,
multiplicity-adjusted). For backtests the analogue: a study that cannot
declare its MDE (minimum detectable effect) up front cannot be killed
honestly — it will be re-run until it passes, and every re-run inflates the
trial count that DSR/PBO must deflate. **The gate must fire before data, and
the trial counter must count every exploratory run, not just the promoted
ones** — which is precisely why ledgers need automation (Part 4).

### 1.4 Literature-grounded reference management

**Persistent identifiers + living indexes.** The scholarly-infrastructure
consensus (FORCE11; NISO; Crossref): cite by **DOI**, pin the version
(doi.org `10.x/zenodo.y` records give version DOIs; arXiv IDs pin
`vN`), and keep a *living annotated bibliography* in-repo (BibTeX/CSL-JSON +
a markdown index that maps claims → sources). **CITATION.cff** (Druskat et
al. 2021, *JORS*) standardizes "how to cite this software"; its sibling
pattern — a `CITATION.cff` per study that lists the papers the hypothesis
draws on — is cheap and makes literature grounding machine-readable.

**Systematic-review practice applied to quant ideas.** PRISMA 2020 (Page et
al. 2021, *BMJ*) and GRADE (Guyatt et al. 2011, *JCE*) are the reference
standards for evidence synthesis. Their transferable machinery for a research
lab:

- **PRISMA flow**: a counted funnel — identified → screened → eligible →
  included — so the *rejection reasons* are as auditable as the inclusions.
  A research-idea backlog with the same funnel (ideas logged, triaged,
  specced, run, decided) prevents silent idea attrition and gives the trial
  counter its denominator.
- **GRADE certainty levels** (high/moderate/low/very low) with explicit
  *downgrade reasons* (risk of bias, inconsistency, indirectness,
  imprecision, publication bias). A receipt verdict is a natural place for a
  GRADE-style certainty field: SYNTHETIC = *very low* by construction;
  real-data + pre-registered + multiplicity-adjusted = *high*; everything
  else names its downgrade reason.
- **PICO framing** (Population/Intervention/Comparator/Outcome) maps cleanly
  onto backtests: universe / strategy family / benchmark / proper score.

The repo maintains `docs/RESEARCH_REFERENCES.md` (568 lines, waves
1–10 canon + DOI/arXiv anchors) as the grounding index — the practice gap is
*linkage*, not existence: hypotheses in notebooks and receipts do not carry
a `references: [doi...]` field, so the claim → literature edge is human
memory (Part 3, F4).

### 1.5 Result ledgers: pending vs decided lifecycle

**The pattern.** Append-only logs with an explicit adjudication boundary are
standard wherever results must be tamper-evident: Certificate Transparency
(RFC 6962) for certificates, Trillian/Rekor for supply chain, clinical-trial
results reporting (design registered → results posted later, never
retro-edited), and — closest to this repo — **pre-commit-style verdict
ledgers** in evaluation harnesses where a *pending* row becomes *decided*
only through a verifier, and decided rows are immutable. The ADR 038 design
(pending `trials.jsonl` + `preregistration.json` at the ledger root; decided
studies archived byte-for-byte to `research/reality/studies/<study_id>/`) is
textbook: **the pending slot is a mutex, the decided archive is history**.

Two SOTA refinements worth adopting:

- **Verdicts are events, not states.** An adjudicated study should record
  *who/what* decided (gate name, receipt hash, git revision, timestamp) so
  the archive is self-describing without external context.
- **Denominator integrity.** The ledger's trial count is the multiplicity
  denominator for DSR/PBO/e-BH. Industry practice (López de Prado 2018,
  *Advances in Financial Machine Learning*, ch. 8–9) is that *every*
  configuration ever scored against the data counts — including abandoned
  ones. Automation is the only reliable way to capture that (Part 4).

### 1.6 Effect size & power analysis for backtests

**Long-run variance.** Returns are serially dependent, so naive i.i.d.
standard errors understate uncertainty. The standard estimators: Newey-West
(1987, *Econometrica*) and Andrews (1991) HAC with Bartlett/kernel weights;
Politikos, White & Sullivan (2013, *JMCB*) HAC for Sharpe ratios specifically;
stationary-bootstrap (Politikos & White 2004) for confidence intervals. The
repo's `forward_evidence.long_run_variance` (Bartlett, `lag`-windowed,
fail-closed on degenerate variance) implements the right primitive.

**From variance to sample size.** The a-priori power calculation for a paired
net-return difference with effect δ and long-run sd σ\*:

\[
N \;\ge\; \left\lceil \frac{(z_{1-\alpha} + z_{1-\beta})^2\,(\sigma^*)^2}{\delta^2} \right\rceil
\]

(one-sided) — exactly the formula in `forward_evidence.evidence_plan` and
`prospective_sota.validate_protocol`. The SOTA additions:

- **MinTRL** (Bailey & López de Prado 2014): the *minimum track-record
  length* for a claimed Sharpe to be distinguishable from zero at a given
  confidence, corrected for skew/kurtosis. The repo computes `min_trl` in the
  overfitting block — it belongs in the *spec* (as the planned N), not only
  in the verdict.
- **Deflated targets.** Harvey, Liu & Zhu (2016, *JF*, "…and the Cross-Section
  of Expected Returns") argue t > 3.0 is the new minimum for factor discovery
  given cumulative multiplicity; Bailey & López de Prado's DSR deflates by the
  *expected maximum* Sharpe over the trial count. A power gate that plans at
  nominal α without stating the multiplicity budget is planning for the wrong
  test.
- **MDE reporting.** Cohen (1988/1992) small/medium/large conventions are the
  default vocabulary; in finance the useful move is expressing MDE in bps of
  net return per session and comparing against transaction-cost uncertainty
  (Chordia, Subrahmanyam & Tong 2014: capacity/liquidity constraints mean
  gross-paper MDEs systematically overstate achievable net effects — the repo
  already freezes net-of-cost books).
- **Anytime-valid power.** When a study peeks (rolling settlement, daily
  grind), fixed-N power is invalid; the e-value literature (Ramdas, Grünwald,
  Vovk & Shafer 2023, *Statistical Science*) and the repo's stopped-e-BH
  suite are the correct frame, and the power statement must be expressed in
  e-value terms (e.g., expected e-value growth rate) rather than fixed-N β.

**Synthesis.** SOTA workflow = spec freezes hypothesis + MDE + N +
multiplicity budget; the run is instrumented so every scored configuration
lands in the ledger; the verdict applies multiplicity-corrected inference;
promotion requires a sealed receipt *and* a GRADE-style certainty label;
literature edges (DOIs) ride on the spec and the receipt. The repo has the
statistical layer (Part 2) and the receipt layer (SOTA 10) at or above SOTA;
the workflow layer between them is where the friction is.

---

## Part 2 — The current study lifecycle, end-to-end

### 2.1 Stage map

```text
IDEA ──▶ SPEC ──▶ RUN ──▶ DIAGNOSE ──▶ RECEIPT ──▶ VERDICT ──▶ PROMOTE/ARCHIVE
  │        │        │         │           │           │             │
catalog/  frozen   sweep/    PBO/DSR/    seal +      gate +      alias /
RESEARCH_ protocol agent    PSR/e-BH    verify      adjudicate  studies/<id>/
REFS.md   (3       runner   (reality/   (receipt_   (reality-   (ADR 038)
research100 schemas)        validation)  v2/verify)  gate)
.json
```

### 2.2 Stage-by-stage (verified by inspection)

| Stage | Artifacts | Code anchors | Frozen? |
|---|---|---|---|
| **Idea** | `docs/RESEARCH_REFERENCES.md` (waves 1–10 canon, DOI anchors); `research100.json` + `docs/RESEARCH100_CATALOG.md` (100-idea catalog); `INFLIGHT` / `day_grind_progress.md` (work tracking) | `quant_fund.research.research100*` | Human-curated; no machine schema |
| **Spec** | `docs/SOTA_PROTOCOL.yaml` (scoring contract); `research/reality/preregistration.json` (reality sweep); `prospective_sota` protocol JSON (power fields); `diffbacktest/spec.py` (boxes + honesty labels); `forward_shadow` freeze | `sota_protocol.SotaProtocol` (Pydantic, `extra="forbid"`, `protocol_sha256`); `prospective_sota.validate_protocol` (rejects `minimum_paired_origins` below the power bound); `forward_evidence.evidence_plan` | Yes, per-schema; three incompatible schemas |
| **Run** | `research/runs/<id>.json/.md`, `latest.json`; `research/reality/trials.jsonl`; `data/*` derived books; MLflow runs (JSONL fallback) | `research.agent.run_research` (23+ families, hypotheses: calibration/discovery/bound); `research.reality_sweep.run_sweep` (frozen costs/grids, ProvenanceDB insert, fails closed if audit ledger exists); `reality_sweep` equity train/val/holdout | Runner-side freeze (`_PREREG` check); no spec digest bound into the run record |
| **Diagnose** | `backtest_overfitting` block in research receipts (schema 2) | `receipt_schema.OVERFITTING_METRIC_KEYS` = pbo/dsr/psr/min_trl; `metrics.overfitting`, `reality.psr`, `validation.fdr`, `metrics.{evalues,anytime_fdr,conformal_martingale}` | Computed at run time; `legacy_uncomputed` migration path exists |
| **Receipt** | `receipts/*.json` (10 committed); notebook JSON/MD pointers | `research.receipt_v2` (`receipt.v2` envelope: dataset/params digests, `code_sha256`, environment fingerprint, `live_pnl_claim:false`, forbidden-key scan, `receipt_sha256` seal); `verify.verify_research_artifact` (H1–H99 hypothesis records, provenance requirements) | Sealed; 7/10 legacy receipts unsealed (SOTA 10 §2.2) |
| **Verdict** | `research/reality/report.json` + `docs/REALITY_*.md`; `verifier/v1..v8` acceptance history; `verifier/runs/` | `reality.report` (`RealityReport`: n_trials, n_effective_trials, psr, min_trl, dsr, pbo, spa_pvalue, bh_fdr_rejects, verdict ∈ pass/deflated/insufficient_evidence); `make reality-gate` | Fail-closed; verdicts published even when negative |
| **Promote/Archive** | registry aliases (`champion`/`candidate`/`shadow`/`retired`); `research/reality/studies/<study_id>/` (1 decided: `reality-us-liquid-daily-2026-09-27`, verdict `deflated`) | `registry.mlflow_store` (`set_alias("champion")` gated by `promotion_is_approved()`); ADR 038 archive move | Alias gate automated; archive move **manual** (`git mv` per ADR 038) |

### 2.3 What the lifecycle does well (at or above SOTA)

1. **Multiplicity machinery is first-class.** PBO/DSR/PSR/MinTRL/SPA/BH/
   stopped-e-BH are computed and carried in contracts
   (`proofcore.contracts.{TrialLedgerRow,RealityReport}`), not bolted on.
   Most shops have none of this.
2. **Honesty is structural.** `FORBIDDEN_RESEARCH_METRIC_KEYS` ↔
   `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` mirror with a drift-blocking test;
   `live_pnl_claim` is `Literal[False]`; SYNTHETIC results cannot hold the
   champion alias; `champion_alias_on_synthetic` defaults to `False` inside
   the frozen protocol itself.
3. **Specs fail closed where they exist.** `SotaProtocol` forbids extra keys
   and forbids `inherit`; `prospective_sota` refuses a protocol whose
   declared `minimum_paired_origins` is below the power bound implied by its
   own α/power/σ\*/δ fields — an a-priori power gate, already implemented.
4. **Verdicts are published either way.** The first reality study is archived
   with verdict `deflated` and the negative result stays published
   (ADR 038) — GRADE-grade honesty about evidence.
5. **The pending/decided boundary is explicit** and documented as an ADR —
   the lifecycle semantics that most labs never write down.

---

## Part 3 — Friction points

**F1 — Three pre-registration schemas, no shared spine (high).**
`SotaProtocol` (YAML/Pydantic), `prospective_sota` protocol (hand-validated
JSON with the power fields), and `research/reality/preregistration.json`
(checked by `reality_sweep._PREREG`) each freeze *some* of the required
elements from §1.1, with different key names, different hashing
(`protocol_sha256` vs `_sha`/`commitment` vs raw file digest), and different
"what counts as frozen" semantics. A study cannot cite one spec digest that
downstream gates all honor; each lane re-implements validation. New lanes
copy whichever schema was nearest, propagating the drift.

**F2 — The power gate exists but only on the prospective path (high).**
`prospective_sota` enforces the §1.6 sample-size bound and
`forward_evidence.evidence_plan` computes required sessions — but a standard
`research.agent` run or a reality sweep carries **no a-priori N/MDE
requirement**: `min_trl` appears only in the *verdict* block. A study can be
run, diagnosed, receipted, and killed without ever having declared what
effect size it was powered to detect — which is the field AsPredicted and
GRADE both treat as the load-bearing one. (Kill criteria without MDE become
"re-run until pass"; the trial counter deflates them, but the compute is
already spent.)

**F3 — Manual adjudication/archive choreography (medium-high).** ADR 038's
decided-study move (`git mv` to `research/reality/studies/<study_id>/`,
pending root left clean, nothing edited) is correct but human-executed; the
README documents it as prose. There is no `reality-archive` command that
(a) verifies the receipt + audit chain of the study being archived,
(b) atomically moves the directory, (c) appends a *verdict event*
(who/what/when/receipt-hash/revision) into the archived study, and
(d) asserts the pending root is clean afterwards. Every manual step is a
chance to contaminate the pending mutex or archive without its
self-describing verdict record.

**F4 — Hypotheses don't carry literature edges (medium).**
`docs/RESEARCH_REFERENCES.md` is a strong index, but nothing links a
notebook/receipt hypothesis to the DOIs it grounds on; the two reference
files (root `RESEARCH_REFERENCES.md`, 383 lines vs `docs/`, 568 lines) have
diverged in length, and `CITATION.cff` (if present) describes the software,
not the studies. Claim → source is currently human memory, so a verifier
cannot check "is this hypothesis grounded in the cited literature" and the
idea funnel (§1.5 PRISMA) has no machine-readable denominator.

**F5 — Exploratory runs escape the ledger (medium-high).**
`trials.jsonl` captures the reality sweep's scored configurations, and
`agent.py` writes `research/runs/<id>.json` — but there is no single append
point that *every* scored configuration crosses, and no counter of
exploratory runs that never became studies. DSR/PBO denominators are only as
honest as the ledger's coverage; today coverage depends on which runner was
used. MLflow runs (when enabled) are a second, unsealed view of the same
history with no reconciliation against `trials.jsonl`.

**F6 — Run records don't bind the spec digest (medium).** `run_research`
writes `runs/<id>.json` with config hash and receipt self-sealing, but the
frozen-protocol digest (`protocol_sha256`, prospective `commitment`) is not a
required field of the run record, so "this run executed under that frozen
spec" is asserted by convention (`reality_sweep._PREREG` checks presence, not
binding across lanes). Verification at receipt time can't prove the run
respected the spec — only that a spec existed.

**F7 — Unsealed receipts sit in the evidence directory (high; owned by SOTA
10).** 7/10 committed `receipts/*.json` fail `verify-receipt`
(`receipt_sha256_missing_or_invalid`) and no CI gate re-verifies the
directory. A workflow verdict is only as trustworthy as its receipt; this is
tracked as W1 in `docs/SOTA/10-reproducibility.md` and the receipts-gate job
proposed there is a prerequisite for the ledger automation below.

**F8 — No GRADE-style certainty label on verdicts (low-medium).**
`RealityReport.verdict` ∈ {pass, deflated, insufficient_evidence} is a
decision, not a certainty grade. SYNTHETIC vs real-data, pre-registered vs
exploratory, multiplicity-adjusted vs nominal — the inputs to certainty are
all present in the artifacts but never composed into one field a reader can
sort by.

---

## Part 4 — Adoption plan

Three deliverables, in dependency order, each additive and each gated.

### 4.1 `study.v1` — one machine-readable study spec (fixes F1, F4, F6)

A Pydantic model (`quant_fund.research.study_spec`) that is the *union* of
what the three existing schemas freeze, versioned like receipts were
(`study.v1`, extra keys forbidden, canonical-JSON `study_sha256` using
`utils.hashing.canonical_json_bytes`):

```jsonc
{
  "schema": "study.v1",
  "study_id": "reality-us-liquid-daily-2026-09-27",   // slug, stable forever
  "status": "proposed|frozen|running|decided",
  "hypothesis": {
    "statement": "...",                               // falsifiable, one sentence
    "family": "tsmom|calibration|discovery|bound|...",
    "primary_metric": "pinball|crps|qlike|ic|brier|ece|kupiec|hmm_ll",
    "guardrails": ["coverage_90"],
    "references": [                                   // F4: DOIs, pinned versions
      {"doi": "10.21314/JPM.2015.131", "why": "PBO/CSCV"},
      {"arxiv": "1504.04967v3", "why": "DSR"}
    ]
  },
  "data": {
    "sources": [{"id": "...", "sha256": "..."}],      // dataset identity
    "universe": [...], "split_dates": {...}, "embargo_bars": 5,
    "data_label": "REAL|SYNTHETIC"                    // honesty contract
  },
  "costs": {"one_way_bps": ..., "impact_model": "...", "frozen": true},
  "power": {                                          // F2: required, not optional
    "mde_bps": 3.0, "alpha": 0.05, "power": 0.8,
    "long_run_sd_source": "calibration_window_sha256",
    "required_n": 214,                                // computed, not asserted
    "min_trl_periods": 180,
    "multiplicity_budget": {"planned_trials": 40, "correction": "dsr+ebh"}
  },
  "analysis": {
    "runner": "reality_sweep|agent|prospective_sota|forward_shadow",
    "runner_config_sha256": "...",
    "protocol_ref": {"path": "docs/SOTA_PROTOCOL.yaml", "protocol_sha256": "..."}
  },
  "decision_rule": {                                  // stage gate, pre-committed
    "promote_if": "verdict == pass AND dsr > 0 AND pbo < 0.5",
    "kill_if": "n_obs < required_n OR verdict == deflated",
    "deviation_log": []                               // appended, never edited
  },
  "claim": "research_only", "live_pnl_claim": false,
  "study_sha256": "..."                               // seal over the above
}
```

Migration: adapters read each existing schema into `study.v1` (the
prospective validator's power block maps 1:1; `SotaProtocol` fields map into
`data`/`analysis`; reality `preregistration.json` maps into `data`/`costs`).
Runners gain one obligation: **write the `study_sha256` into every run record
and every receipt** (F6). `verify-research` gains a required check: run ↔
spec digest match, spec `status == frozen` before first observation,
`power.required_n` recomputed from spec fields (same formula as
`forward_evidence.evidence_plan`) and equal to the declared value. The two
reference files consolidate into `docs/RESEARCH_REFERENCES.md` as the single
index, with `references[].doi` validated against it (F4) — a DOI in a spec
that isn't in the index fails closed, which keeps the index living.

### 4.2 Ledger automation: `reality-archive` + one append point (fixes F3, F5, F8)

**One append point.** All scored configurations — `agent.py` runs,
`reality_sweep` trials, prospective settlements, forward-shadow looks —
append to a single per-study ledger via one function
(`research.ledger.record_trial(study_id, trial_row)`), which writes the
`TrialLedgerRow` contract *plus* `study_sha256` and a monotonic
`trial_seq`. `trials.jsonl` stays the reality-gate's input format (no
breaking change); the shared writer guarantees the DSR/PBO denominator is
the union of lanes (F5). MLflow, when enabled, is demoted explicitly to
observability: the ledger is the source of truth, and a reconciliation
check (`ledger ↔ mlflow runs by run_id`) flags divergence instead of
silently trusting either.

**`dipcatcher reality-archive <study_id>`** implementing ADR 038
mechanically:

1. Verify the study's `receipt.json` (`verify-receipt`) and audit chain
   (hash-chain + inclusion proof) — refuse to archive unverifiable evidence
   (depends on F7/SOTA-10 Phase 1 receipts gate).
2. Append a **verdict event** into the study directory: gate name,
   `RealityReport` digest, git revision + worktree hash, timestamp, actor.
3. Compose and write the **certainty label** (F8):
   `high` = REAL + frozen spec + multiplicity-adjusted verdict; `moderate` =
   REAL + frozen spec, nominal inference; `low` = REAL, exploratory;
   `very_low` = SYNTHETIC (by construction, matching the honesty contract).
4. `git mv` the pending study to `research/reality/studies/<study_id>/`
   byte-for-byte; assert the pending root is clean; fail closed on any
   existing destination.

The pending-slot mutex (`run_sweep` refusing a second study) stays exactly
as-is — automation wraps it, it does not reinterpret it.

### 4.3 Power gate for new studies (fixes F2)

Generalize `prospective_sota.validate_protocol`'s bound into a shared
precondition on `study.v1`:

- **Gate:** a study cannot transition `proposed → frozen` without
  `power.required_n` computed from declared `mde_bps`, α, power, and a
  σ\* calibration window whose digest is in the spec — reusing
  `forward_evidence.long_run_variance` (Bartlett HAC; stationary bootstrap
  as a sensitivity column, per Politikos–White). The *declared* N must be
  ≥ the *computed* N; mismatch fails closed, exactly as the prospective lane
  already does.
- **Anytime-valid variant:** studies that peek (forward shadow, daily grind)
  must express power as an e-value growth statement (expected log-e-value per
  session under the alternative) and cite the stopped-e-BH budget from
  `metrics.anytime_fdr` — fixed-N β is not accepted for peeking designs.
- **Multiplicity honesty:** `planned_trials` is a required field; the
  verdict's DSR uses `max(planned_trials, observed ledger count)` — planning
  fewer trials than the ledger shows is itself a flagged deviation.
- **Where it fires:** `run_research` and `run_sweep` refuse to start without
  a frozen spec that passed the power gate; the refusal writes a
  `verdict: "blocked"` receipt (an ungated run is evidence of a process
  failure, per SOTA 10 §3.6's same principle).

### 4.4 Roadmap

| Phase | Work | Gate added | Depends on |
|---|---|---|---|
| 1 (days) | Consolidate reference index; add `references[].doi` to existing protocol schemas (optional field, warn-only) | lint: DOI-in-index | — |
| 2 (1–2 wks) | `study.v1` model + canonical seal + adapters for the three existing schemas; run records bind `study_sha256` | `verify-research` spec-binding check | Phase 1 |
| 3 (1–2 wks) | Power gate: shared `required_n` validation; `proposed→frozen` transition blocked without it; blocked-run receipts | study-start precondition | Phase 2 |
| 4 (1 wk) | Ledger append point across runners + MLflow reconciliation check | trial-count audit test | Phase 2 |
| 5 (1 wk) | `reality-archive` CLI with verdict event + certainty label; ADR 038 prose updated to point at the command | archive self-check (clean pending root) | SOTA 10 Phase 1 (receipts gate) |
| 6 (opportunistic) | GRADE certainty in `docs/REALITY_*.md` headers; PRISMA-style funnel report over `research100.json` + ledger (`dipcatcher research-funnel`) | — | Phases 2–5 |

Every phase is additive: existing receipts, ledgers, and the reality gate
keep working unchanged; `study.v1` is a superset seal over schemas that
already fail closed.

---

## Part 5 — References

**Pre-registration & forking paths.**
1. Simmons, Nelson, Simonsohn. *False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis.* Psychological Science 22(11), 2011.
2. Gelman, Loken. *The Garden of Forking Paths.* arXiv/UNC preprint, 2013.
3. Nosek et al. *The preregistration revolution.* PNAS 115(11), 2018; Center for Open Science — OSF Preregistration & AsPredicted templates (2015–).
4. Casey, Glennerster, Miguel. *Reshaping Institutions: Evidence on Aid Impacts Using a Preanalysis Plan.* QJE 127(4), 2012.
5. ICMJE. *Clinical trial registration policy.* 2005; ClinicalTrials.gov record structure (design vs results sections).
6. Bailey, Borwein, López de Prado, Zhu. *Pseudo-Mathematics and Financial Charlatanism.* Notices of the AMS 61(5), 2014; and *The Probability of Backtest Overfitting.* Journal of Computational Finance 20(4), 2017.

**Experiment tracking & registries.**
7. Zaharia et al. / Databricks. *MLflow Model Registry* docs — aliases replacing deprecated stages (MLflow 2.9→3.x); *MLflow Tracking* run-immutability semantics.
8. Weights & Biases docs — *Sweeps* (declared YAML), *Reports* (locked run cohorts), immutable run history.
9. Pineau et al. *Improving Reproducibility in Machine Learning Research.* JMLR 22(164), 2021 (ML Reproducibility Checklist).
10. ACM. *Artifact Review and Badging v1.1.* 2022 (available / evaluated / reproduced badges).
11. Tang, Agarwal, et al. *Overlapping Experiment Infrastructure.* KDD 2010; Kohavi, Tang, Xu. *Trustworthy Online Controlled Experiments.* Cambridge UP, 2020.

**Multiverse, multiple testing, power.**
12. Steegen et al. *Increasing Transparency Through a Multiverse Analysis.* Perspectives on Psychological Science 11(5), 2016; Silberzahn et al. *Many Analysts, One Data Set.* Psychological Science 29(3), 2018.
13. Newey, West. *A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3), 1987; Andrews. *Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation.* Econometrica 59(3), 1991.
14. Politikos, White. *Improving Inference on Sharpe Ratios with HAC standard errors.* (2004); Politikos, White, Sullivan. *Robust Inference for Sharpe Ratios.* JMCB 45(S1), 2013.
15. Bailey, López de Prado. *The Minimum Track Record Length* / *The Deflated Sharpe Ratio.* Journal of Portfolio Management 40(5)/41(1), 2014/2015.
16. Harvey, Liu, Zhu. *…and the Cross-Section of Expected Returns.* Review of Financial Studies 29(1), 2016 (t > 3.0 hurdle).
17. López de Prado. *Advances in Financial Machine Learning.* Wiley, 2018 (ch. 7–9: cross-validation, backtest overfitting, trial count).
18. Cohen. *Statistical Power Analysis for the Behavioral Sciences.* 2nd ed., 1988; *A power primer.* Psychological Bulletin 112(1), 1992.
19. Ramdas, Grünwald, Vovk, Shafer. *Game-Theoretic Statistics and Safe Anytime-Valid Inference.* Statistical Science 38(4), 2023; Wang, Ramdas. *False discovery rate control with e-values.* 2022 (e-BH; stopped e-BH variants).
20. White. *A Reality Check for Data Snooping.* Econometrica 68(5), 2000; Hansen. *A Test for Superior Predictive Ability.* JBES 23(4), 2005; Hansen, Lunde, Nason. *The Model Confidence Set.* Econometrica 79(2), 2011.
21. Chordia, Subrahmanyam, Tong. *Liquidity and market efficiency.* JFE 2014 (capacity/net-effect caveats for MDEs).

**Literature management & evidence synthesis.**
22. Page et al. *The PRISMA 2020 statement.* BMJ 372:n71, 2021.
23. Guyatt et al. *GRADE guidelines 1–9.* Journal of Clinical Epidemiology, 2011 (certainty levels, downgrade reasons).
24. Druskat et al. *Citation File Format.* Journal of Open Source Software 6(64), 2021; FORCE11/NISO persistent-identifier principles; Crossref DOI versioning (Zenodo version DOIs; arXiv `vN` pinning).

**Ledgers & provenance.**
25. Laurie, Langley, Kasper. *RFC 6962 — Certificate Transparency.* 2013 (append-only logs; the pattern behind `quant_fund.audit`).
26. Repo-internal: `docs/decisions/038-reality-ledger-lifecycle.md`; `research/reality/README.md`; `docs/SOTA/10-reproducibility.md` (receipt sealing — F7/W1); `docs/FORWARD_SHADOW_POWER.md`; `docs/REALITY_PREREGISTRATION.md`.

**Repo-internal anchors.** `src/quant_fund/research/{agent,reality_sweep,sota_protocol,prospective_sota,forward_evidence,receipt_v2,receipt_schema,verify}.py`;
`src/quant_fund/reality/{psr,report}.py`; `src/quant_fund/proofcore/contracts.py`;
`src/quant_fund/metrics/{overfitting,inference,evalues,anytime_fdr}.py`;
`src/quant_fund/validation/fdr.py`; `src/quant_fund/registry/mlflow_store.py`;
`src/quant_fund/diffbacktest/spec.py`; `docs/RESEARCH_REFERENCES.md`;
`src/quant_fund/research/research100.json`.
