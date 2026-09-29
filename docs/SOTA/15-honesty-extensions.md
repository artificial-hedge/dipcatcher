# 15 — Honesty / integrity system extensions (SOTA lane)

Research-grounded survey, enforcement audit, and adoption plan for the lab's
honesty system. Lane rule: **strengthen, never weaken.** Nothing in this
document proposes relaxing `FORBIDDEN_RESEARCH_METRIC_KEYS` /
`FORBIDDEN_HEADLINE_TOKENS`, the fail-closed gates, the SYNTHETIC labeling
contract, the no-live-claim rule, or receipt immutability. The mirroring
contract (catalog ↔ `fx1.honesty`, blocked from drift by
`tests/fx1/test_honesty_inheritance.py`) stays exactly as it is; every
adoption item below only adds checks or shares existing stronger checkers.

This is a research-infrastructure note. It contains no performance claim, no
promotion, and no live-trading authorization.

---

## 1. Technique summaries and citations

### 1.1 Research-integrity automation (forbidden-metric linting, claim–evidence linking)

- **Structural key scanning** — enforce reporting rules on artifact *schemas*,
  not conventions: walk every mapping key of a research blob and fail closed
  on forbidden tokens. This is the design the repo already uses
  (`family_blob_forbidden_metrics_absent`, ADR-0008). The general principle
  matches the machine-checkable-checklist movement in ML venues: rules that a
  linter can decide beat rules that a reviewer must remember.
- **Automated statistical-report linting** — *statcheck* (Nuijten et al. 2016,
  *Behav. Res. Methods*; Polanin & Nuijten 2020, *Res. Synth. Methods*,
  doi:10.1002/jrsm.1408) extracts NHST results from text, recomputes p-values
  from the reported statistic and df, and flags inconsistencies (with "gross
  inconsistency" when the significance decision flips). Accuracy 96–99.9% on
  what it detects; it detects ~60% of tests. Lesson for this repo: **internal
  consistency re-derivation** (reported value vs recomputed value) is the
  highest-value automated integrity check, and partial coverage is still worth
  deploying.
- **Claim–evidence linking for agent-written research** — three 2024–2026
  systems converge on the same shape as our receipts:
  - *Chain-of-Evidence* (ScientistOne, arXiv:2605.26340): every claim carries
    an inline evidence pointer at write time; a Claim Verifier re-checks each
    claim against its declared source; claims that cannot be backed are
    restated conservatively, not deleted. Companion **CoE Audit** checks:
    score verification (re-run the code, compare the number), specification
    violation (metric gaming), reference verification (citations resolve
    against live APIs), method–code alignment. Their numerical **Claim
    Provenance Rate** = fraction of claims with valid evidence pointers.
  - *ReAgent* ("Beyond the Text", arXiv:2609.22111): static + dynamic audit of
    agent papers against their repositories; verdicts Matched / Mismatched /
    Blocked / Not Reproduced; catches numbers that reproduce but whose
    methodology deviates from the claim.
  - *ClaimVer* (arXiv:2403.09724): claim-level decomposition, verification
    against a knowledge graph, and a graded **KG Attribution Score**
    (−1 contradictory … +2 attributable) instead of a binary verdict;
    *PaperTrail* (arXiv:2602.21045) shows the same decomposition as a
    human-facing provenance UI.
  - Takeaway: graded attribution (supported / partially supported /
    unsupported / contradicted) plus a numeric provenance rate is the current
    SOTA presentation; our receipts are stronger than these systems' evidence
    objects (hash-sealed, immutable) but our *claims* (day-grind bullets,
    docs prose, fx-1 outputs) are not individually pointer-bound to them.

### 1.2 Statistical disclosure control (SDC)

- Classic SDC (Hundepool et al., *Handbook on Statistical Disclosure Control*,
  sdctools.github.io/HandbookSDC; Willenborg & de Waal 1996/2001) covers cell
  suppression with complementary suppression, record swapping, PRAM,
  micro-aggregation, and the risk–utility (R–U) confidentiality map (Duncan &
  Stokes; NISS). Magnitude tabular protection (τ-ARGUS modular/hypercube)
  suppresses cells dominated by one contributor.
- Differential privacy (Dwork & Roth 2014; Karr–Reiter statistical-data-
  privacy review, *Annu. Rev. Stat. Appl.* 2021,
  doi:10.1146/annurev-statistics-033121-112921) gives an adversary-independent
  guarantee at a quantifiable utility cost; relevant when an aggregate could
  be differenced against a published neighbor to recover a suppressed cell.
- Applicability here (honest scoping): the lab publishes no human-subject
  microdata, so formal DP is out of scope. The relevant SDC transfer is
  (a) *suppression discipline on published diagnostics* — the evidence page
  already redacts account-level values and forbidden-keyed fields
  (`build_evidence_report.redact_account_levels` / `redact_key`); (b) *small-
  cell rules* — event studies below an adequate-sample floor must not carry a
  rejection (already enforced: `sample_adequate=False` ⇒ `reject_fdr=True`
  fails verification); (c) *multi-release composition* — several published
  receipts over overlapping panels can jointly disclose what each suppresses
  (the differencing attack SDC exists to prevent). No repo mechanism currently
  tracks cumulative disclosure across published receipts.

### 1.3 Calibration of confidence in claims (evidence hierarchies)

- **GRADE** (Guyatt et al. 2011; Balshem et al. 2011; Cochrane Handbook ch.14;
  Core GRADE series, *BMJ* 2025 doi:10.1136/bmj-2024-083864): four certainty
  levels (high / moderate / low / very low) per *body* of evidence, starting
  from design (RCT high, observational low) and moving down for risk of bias,
  inconsistency, indirectness, imprecision, publication bias — one level for
  serious, two for very serious concerns; up only for large effects /
  dose–response / bias that would understate the effect. The key transferable
  idea: **certainty is a graded label with named downgrade reasons, attached
  to the evidence body, not the prose.**
- **Source-level evidence tags** — the ECE fact-checking system
  (arXiv:2607.18240) labels each supporting source L1 (direct execution /
  code-verified) … L5 (model-internal knowledge, no external evidence), and
  abstains rather than issuing confident verdicts on weak chains; abstentions
  concentrate at the lowest tier, which is exactly the behavior we want from
  fx-1 on unevidenced numeric claims.
- **Beyond p < 0.05 reporting** (Wasserstein, Schirm & Lazar 2019, *The
  American Statistician* 73(sup1), doi:10.1080/00031305.2019.1583913;
  Amrhein, Trafimow & Greenland 2019): report effect sizes with compatibility
  intervals, avoid dichotomous "significant" language, optionally convert
  p to **s-values** s = −log₂(p) (Shannon surprisal) to communicate evidence
  strength honestly. Directly applicable to hypothesis-table rendering.
- Repo fit: `data_label` (SYNTHETIC / public-file / public-sources / vendor)
  and `claim: research_only` are binary-ish stamps. The SOTA shape is a small
  ordered **evidence tier enum** spanning fixture → synthetic panel →
  retrospective public → PIT retrospective → prospective sealed journal →
  forward-shadow record, stamped on every receipt and echoed by fx-1 whenever
  it cites a number.

### 1.4 Pre-registration practices

- **Preregistration + registered reports**: freeze hypothesis, sample,
  analysis plan, and decision rule before data contact (Nosek et al. 2018,
  *Science* 361:26–28; Chambers & Tzavella 2022, *Nature Rev. Psych.*
  doi:10.1038/s44159-022-00029-z — registered reports make the acceptance
  decision before results exist).
- **Deviation reporting** (Willroth & Atherton 2024, *Adv. Methods Pract.
  Psychol. Sci.*, doi:10.1177/25152459231213802): preregistrations are not
  edited after results are seen; changes are reported in a deviations table
  (change, rationale, when in the pipeline, impact). The `prereg` Python
  package (PyPI) operationalizes the distinction mechanically: a frozen plan
  hash, `prereg check` fails on any post-freeze edit, and a log entry is an
  **amendment** iff nothing had been run / no results seen, otherwise a
  **deviation**.
- **Registration-to-paper comparison**: RegCheck (arXiv:2601.13330;
  github.com/priyankanagabhushana/regcheck) uses an LLM to compare published
  manuscripts against registrations dimension-by-dimension (sample size,
  exclusions, hypotheses, models…) and emits quoted evidence for each
  deviation judgment; findings: deviations are common and usually
  *undisclosed*, and registrations are often underspecified.
- Repo fit: `research/reality_sweep.py` already implements a strong version
  for one lane — frozen `preregistration.json` (SHA-256 stamped into the
  receipt), cell-count equality against the pre-registered grid, and
  `assert_cost_lock` refusing any cost/risk-gate drift. `PROSPECTIVE_SOTA.md`
  freezes a single primary test, splits alpha Bonferroni across exactly two
  contrasts, and permanently blocks a run on the first missed origin. What is
  missing is the *generalization*: ordinary `dipcatcher research` runs are
  not pre-registered, and there is no deviations channel (amendment vs
  deviation vs results-seen) for the labs that do freeze specs.

### 1.5 Multiple-comparisons discipline for research sweeps

- **FWER family**: Bonferroni; Holm (1979); Hochberg (1988); closed testing.
- **FDR family**: Benjamini–Hochberg (1995, *JRSS-B* 57:289–300);
  Benjamini–Yekutieli (2001, *Ann. Stat.* 29:1165–1188) for arbitrary
  dependence; Storey q-values (2002/2003).
- **Anytime-valid FDR** (the sequential-grind setting): e-values and e-BH
  (Wang & Ramdas 2022, *JRSS-B*, "False discovery rate control with
  e-values"); stopped e-BH under optional stopping (Wang, Dandapanthula &
  Ramdas 2025, arXiv:2502.08539); e-LOND online FDR (Xu & Ramdas 2024); safe
  testing (Grünwald, de Heide & Koolen 2024, *JRSS-B*; Vovk & Wang 2021,
  *Ann. Stat.*). All already implemented in `metrics/anytime_fdr.py` and
  `metrics/evalues.py` — they are the correct tool for a 24×7 grind that
  peeks and stops adaptively.
- **Data-snooping / selection-aware inference**: White Reality Check (2000);
  Hansen SPA (2005); Romano–Wolf StepM (2005); Hansen–Lunde–Nason MCS (2011)
  — all in `metrics/snooping.py` and wired into the ranking battery.
- **Finance-specific multiplicity**: Harvey, Liu & Zhu (2016, *J. Finance*
  71:5–68, "…and the Cross-Section of Expected Returns"): with hundreds of
  published factors, a new discovery needs t > 3.0, not 2.0; Harvey, Liu
  (2015+, "Backtesting", *J. Portfolio Mgmt.*): **haircut Sharpe ratios**
  non-linearly in the number of trials; Harvey, Sancetta & Zhao (2026, NBER
  w34898): thresholds that do not assume a known total trial count, plus
  **local FDR** (probability the null is true given the observed statistic)
  as a claim-calibration quantity a p-value cannot supply.
- **Bailey/López de Prado deflation**: PSR (2012), DSR (2014), PBO/CSCV
  (Bailey et al. 2017) — implemented and stamped on schema-2 receipts
  (`backtest_overfitting`), with trial counts and effective (clustered)
  trial counts.
- Repo fit: the strongest stack is already here. The *gap* is enforcement
  direction — the verifiers check that rejection flags are well-typed and
  that unavailable p-values never reject, but they never **re-derive** BH
  rejections from the stored p-values, and parameter-grid sweeps outside the
  reality lane have no mandated FDR/trial-count pass.

### 1.6 Garden of forking paths (Gelman & Loken)

- Gelman & Loken (2013/2014, "The garden of forking paths", Columbia stat
  dept; Gelman, Hill & Vehtari 2020, *Regression and Other Stories* ch. 6):
  even with zero "p-hacking" intent, a one-to-many mapping from data to
  analysis decisions (exclusions, transformations, subgroups, horizons)
  inflates Type-I error because the analysis path is chosen with knowledge of
  the data. The defense is not a post-hoc correction but **declaring the
  decision tree in advance** or **reporting the whole multiverse**.
- **Multiverse analysis** (Steegen, Tuerlinckx, Gelman & Vanpaemel 2016,
  *Perspect. Psychol. Sci.* 11:702–712, doi:10.1177/1745691616658637): run
  every defensible data-processing choice and report the distribution of
  results, not one.
- **Specification curve analysis** (Simonsohn, Simmons & Nelson 2020, *Nature
  Human Behaviour* 4:1208–1214, doi:10.1038/s41562-020-0912-z): three steps —
  enumerate theoretically justified, statistically valid, non-redundant
  specifications; display them all; run **joint inference across the curve**
  (permutation test of the median specification against the null).
  Tooling: `specr` (R), `multiverse` (R DSL).
- Simmons, Nelson & Simonsohn (2011, *Psychol. Sci.* 22:1359–1366, "False-
  positive psychology"): researcher degrees of freedom turn 5% tests into
  60%+ false-positive rates — the empirical motivation for both prereg and
  multiverse reporting.
- Repo fit: sweeps already **count every cell as a trial**
  (`disclosure: every_cell_counted_as_a_trial`; `n_trials` includes
  unalignable configs, and dropping them cannot shrink multiplicity —
  `docs/BACKTEST_OVERFITTING.md`), which is the correct multiverse-side
  behavior. `PRIMARY_EXECUTABLE_TEST` in `northset/sweep_research.py` is a
  predeclared primary with secondaries counted — exactly the Gelman–Loken
  prescription. What is missing is a specification-curve-style *rendering*
  requirement (the fold-stability and parameter-sensitivity blobs exist but
  no gate demands sign-stability reporting for headline hypotheses).

### 1.7 p-hacking detection heuristics

- **Caliper test** (Gerber & Malhotra 2008, *Sociol. Methods Res.*): excess
  mass just above vs just below a significance threshold (e.g. p ∈
  [0.039, 0.05) vs [0.05, 0.0626) — the 5% caliper around z = 1.96) signals
  threshold-chasing. Simulation evidence (Hilger, König & Löwel 2019,
  *Res. Synth. Methods*, PMC5712469): caliper tests are unbiased in false
  positive rate but low-powered unless K ≈ 1,000 studies; test-of-excess-
  significance (Ioannidis & Trikalinos 2007) often performs better under
  p-hacking; Egger's FAT best under file-drawer bias.
- **p-curve** (Simonsohn, Nelson & Simmons 2014, *J. Exp. Psychol. Gen.*):
  right-skew of significant p-values evidences real effects; left-skew
  evidences intensity p-hacking. **z-curve** (Bartoš & Schimmack 2022,
  *Meta-Psychology*) estimates expected replication rate and the discovery
  rate from significant results only.
- **Power of detection** (Elliott, Kudrin, Roth, Wu 2022/2024,
  arXiv:2205.07950, *Review of Economic Studies* — "(When) Can We Detect
  p-Hacking?"): combined bounds+monotonicity tests (**CS2B**) dominate;
  classical binomial caliper tests are often underpowered; density-
  discontinuity tests (Cattaneo et al. `rddensity`) control size better than
  the naïve symmetric-count bootstrap but are underpowered; caliper tests do
  not control size in general (Kudrin 2024).
- Honest scoping for this repo: these are *meta-analytic* tools for
  literature-scale corpora (hundreds of independent studies). Applied to one
  lab's own trial ledger they are weakly powered — but a **self-audit
  diagnostic** (caliper ratio and z-curve discovery rate over the accumulated
  `TrialLedger`/hypothesis rows, warning-level, never a gate) is cheap, uses
  data we already store, and catches threshold-chasing behavior in the grind
  loop before it becomes a headline.

### 1.8 Reporting standards (ML reproducibility)

- **ML Reproducibility Checklist v2.0** (Joelle Pineau, 2020,
  cs.mcgill.ca/~jpineau/ReproducibilityChecklist-v2.0.pdf; Pineau et al.
  2021, *JAIR* 70, "Improving Reproducibility in Machine Learning Research";
  Pineau, Vincent-Lamarre, Sinha, Larivière et al. 2020, arXiv:2003.12206):
  for results — hyperparameter *ranges* considered and selection method,
  **exact number of training/evaluation runs**, definition of the reported
  measure, **central tendency AND variation** (error bars), average runtime
  or energy, computing-infrastructure description; for datasets — statistics,
  splits, exclusions and preprocessing; for claims — clear statement plus
  assumptions.
- **Venue checklists**: NeurIPS paper checklist (~15 yes/no/NA items with
  justifications covering reproducibility, code/data access, seeds, error
  bars, compute); JAIR's four mechanisms (checklist + optional artifacts
  evaluation + reproducibility summary + replication studies).
- Repo fit: `receipt.v2` already exceeds most of the checklist on
  *environment* (python/numpy/polars/scipy versions, BLAS/LAPACK build,
  threadpools, `fingerprint_sha256`, `code_sha256`, `dataset_hash`,
  `params_hash`) and `docs/MODEL_CARDS.md` requires seeds, hyperparameters,
  trial counts, split dates, cost assumptions. Two Pineau items are not
  receipt-level requirements anywhere: **exact number of runs per reported
  statistic** and **variation accompanying every central tendency** (error
  bars/CI keys are per-lane, not schema-enforced).

---

## 2. Audit of current enforcement (what slips through)

Surfaces audited: `quant_fund.research.catalog.registry` (key scan),
`quant_fund.research.verify` (notebook verifier), `receipt_v2` (envelope),
`fx1.honesty` (text gate), `quant_fund.leakage.patterns` + `ast_scan`
(LH008/LH013), `reality/fdr.py` + `validation/fdr.py` (multiplicity),
`reality_sweep.py` (prereg), `audit/ledger.py` + `audit-trace` (claim→
receipt link), `fx1/reward.py` (reward shaping), `build_evidence_report.py`
(published diagnostics). All probes below were executed against the current
checkout (SYNTHETIC/local only; no market data touched).

### 2.1 What holds up well (do not touch)

1. **Key-level fail-closed scanning works**, including nesting through lists
   and dash-variants: `family_blob_forbidden_metrics_absent([{'sortino':1}])`
   → False; `{'sortino-ratio':1}` → False. Scorecards re-derive all four
   flags and detect forged ones (`scorecard_*_flag_forged`).
2. **Honesty inheritance is real**: the two forbidden sets are compared for
   set-equality in CI (`fx1.yml` runs `-k honesty` as a separate gate step),
   every lab key is parametrized through the fx-1 validator, and fx-1-only
   extra tokens also fail.
3. **Receipt sealing is layered**: canonical-JSON SHA-256 seal, environment
   fingerprint, code-map digest, payload/envelope label agreement
   (`payload_data_label_mismatch`, `payload_live_pnl_claim_not_false`),
   `live_pnl_claim` typed `Literal[False]`, immutable-twin equality
   (`immutable_receipt_mismatch`), artifact-root containment, and a
   hash-chained, Merkle-checkpointed, Ed25519-signed audit ledger with
   `audit-trace --metric` inclusion proofs.
4. **Overfitting stamp is verified, not trusted**: schema-2 receipts require
   `pbo/dsr/psr/min_trl/n_trials/n_trials_effective`, unit-interval checks,
   `dsr ≤ psr`, `n_trials_effective ≤ n_trials`, forbidden-key absence, and
   `research_diagnostic_only` claim.
5. **Reality lane preregistration is genuinely load-bearing**: frozen spec
   hash in the receipt, exact grid cell-count equality, and
   `assert_cost_lock` raising on any cost/risk-gate drift (commission,
   spread, impact, borrow, financing, participation, gross/net/name caps,
   fill rule, `frictionless=false`).
6. **Small-sample rejections are blocked**: `sample_adequate=False` with
   `reject_fdr=True` fails verification; non-finite p-values can never mint a
   rejection; families are never pooled for BH (`calibration` / `discovery` /
   `bound` split).

### 2.2 Gaps — text gate (fx1.honesty)

Probed with `validate_fx1_output` on the current checkout:

| Probe | Result | Note |
|---|---|---|
| `"Sharpe ratio came in at 2.35"` | **PASS** (slips) | fx-1 regex needs the number *immediately* after the token; `leakage.patterns.find_forbidden_headline` catches the same line (proximity window). Two matchers, one mirror-tested token set — the *matcher* is not mirrored. |
| Unicode homoglyph `"Shаrpe … 2.1"` (Cyrillic а) | **PASS** (slips) | `leakage.patterns._normalize` strips Cf chars + NFKC; `fx1.honesty` does neither. |
| Spanish `"el ratio de Sharpe fue 2.1"` | **PASS** (slips) | Non-English headline phrasing defeats the English-only claim patterns. |
| `"we made a profit of $4,200 on real capital"` | **PASS** (slips) | Live-money paraphrase without the exact `_FORBIDDEN_CLAIM_PATTERNS` strings. |
| `"SYNTHETIC …"` label far from a real-data numeric claim in the same text | **PASS** (slips) | Label check is whole-text, not claim-local: one `SYNTHETIC` token launders any number of claims about *real* data elsewhere in the message. |
| Lowercase `"synthetic results show 0.99 accuracy"` | BLOCK | Correct (label must be uppercase). |

None of these weaken the contract — they are *strictly stronger* matcher
semantics already implemented in `quant_fund.leakage.patterns` (LH008 core,
built as the "audit F3 fix" for precisely this immediate-number weakness).
The fx-1 lane simply never received the upgrade; the inheritance test only
pins the token **set**.

### 2.3 Gaps — structural key scan

- Values are not scanned (by design — mentioning is legal), so a *value-side*
  smuggle (`{"note": "Sharpe 2.1 achieved"}`) is clean for the key scan.
  Partially compensated by LH008/LH013 in Python sources and
  `build_evidence_report.contains_forbidden_token` on the evidence page —
  but neither covers research-notebook string *values* or published docs.
- Key tokenizer splits on `_`/`-` only: camelCase (`SortinoRatio`,
  `pnlAttribution`), spaced keys (`"Sharpe Ratio"`), and non-str keys
  (tuple `('sharpe','mean')` — `str(k)` tokenizes with punctuation) all
  **slip** (probed: all return True = clean). Renaming to a semantic synonym
  (`risk_adjusted_return_ratio`) slips — documented residual risk in ADR-0008;
  an alias table (as in `leakage.patterns._TOKEN_ALIASES`) would close the
  cheap part.

### 2.4 Gaps — verifier does not re-derive statistical claims

- **`reject_raw` / `reject_fdr` are trusted, not recomputed.** Probed:
  a hypothesis record with `p_value=0.9, reject_raw=True, reject_fdr=True`
  passes `_hypothesis_record_errors` with zero errors. The only cross-check
  is *unavailable p ⇒ no rejection*. A receipt that claims rejections its
  own p-values do not support is currently verifiable-valid. This is the
  statcheck lesson inverted: we have the inputs to recompute, and don't.
- **BH consistency across the hypothesis set is never checked**: given the
  stored p-values and families, the verifier could recompute the BH mask
  (`metrics/inference.benjamini_hochberg`) and fail closed on any mismatch.
  It does not.
- **No FDR/trial-count requirement for generic research sweeps**: the
  `backtest_overfitting` block counts trials for DSR/PBO, but a parameter
  sweep that produces *hypothesis rejections* has no mandate that the
  rejections be BH/e-BH adjusted, nor that the trial count feeding DSR be the
  sweep's full cell count (the reality lane does this correctly; the
  northset sweep lane does; the generic `run_research` battery relies on the
  agent applying `_apply_family_fdr`, which the verifier does not confirm).

### 2.5 Gaps — claim→receipt linkage

- `fx1/reward.py` awards `cites_receipt` (+2.0, 33% of a clean maximum) for
  **any 8–64-char hex string**. Probed: `"Receipt deadbeef12345678 confirms
  it"` scores identically to a real 64-hex receipt hash. The reward teaches
  citation-shaped strings, not citations.
- `audit-trace --metric` proves a *published number* links to a ledger entry
  and receipt digest — but it is opt-in per metric, and day-grind/docs claims
  are not machine-linked: nothing scans `docs/*.md` or `day_grind_progress.md`
  for numeric claims lacking a receipt pointer.
- **Published prose is an unlinted honesty channel**: LH008 scans `.py` only
  (`ast_scan` walks `*.py`).   Probing published docs (the mkdocs-excluded set
  aside) with `find_forbidden_headline` yields **159 lines across 48 files**,
  e.g. `docs/AUDIT_P62_STATS.md:30` (a headline-ratio token adjacent to a
  numeric value inside an audit finding; value redacted here). Most are legitimate
  meta-discussion (audit findings, "never live Sharpe" boilerplate) — which
  is exactly the ADR-0008 *mention-vs-claim* problem, currently resolved by
  hand. The mkdocs `exclude_docs` list is a manual, incomplete version of
  this gate (7 files excluded from the *site*; the repo docs remain
  unscanned).

### 2.6 Gaps — preregistration and deviations

- Only the reality lane and the prospective journal freeze specs. Ordinary
  research runs (`configs/research.yaml` sweeps) start with no frozen
  hypothesis/analysis plan, so the fork-in-the-road decisions (horizon
  choice, exclusion rules, control definition) are made with data visible.
- There is **no deviations channel**: no amendment-vs-deviation log keyed to
  "had results been seen" (the `prereg` package's rule), no requirement that
  a receipt disclose plan changes. `phase1_verify` checks holdout-disclosure
  agreement with the protocol — the closest existing analogue.

### 2.7 Gaps — evidence-tiering and disclosure

- Evidence provenance is multi-flag rather than ordered: `data_label`,
  `point_in_time`, `execution_claim`, `synthetic`, `dgp=fixture` exclusions,
  `sweep_evidence_scope`. There is no single tier label that a reader (or
  fx-1) can quote, and no GRADE-style **downgrade reasons** list per receipt.
- No cumulative-disclosure tracking across published receipts (§1.2c).
- Pineau items "exact number of runs" and "variation with every central
  tendency" are not schema-level requirements (§1.8).

---

## 3. Adoption plan

Ordered by leverage; every item is additive and fail-closed. Items touching
`fx1.honesty` semantics keep the token-set mirror and may only *tighten*
`validate_fx1_output` (the inheritance test and `tests/fx1/test_honesty.py`
must be extended, never relaxed, in the same change).

### A. FDR gates for research sweeps (closes §2.4)

1. **Verifier re-derivation (highest leverage, cheap).** In
   `research/verify._hypothesis_record_errors` (or a sibling pass):
   - recompute `reject_raw = (p < alpha)` for every record with a usable
     p-value; fail closed on mismatch (`hypothesis_reject_raw_inconsistent`).
   - recompute the BH mask per family (`calibration`, `discovery`; `bound`
     excluded, exactly as `reality/fdr.bh_fdr`) from the stored p-values and
     fail closed on any `reject_fdr` mismatch
     (`hypothesis_reject_fdr_inconsistent`). Ties broken deterministically.
   - keep unavailable-p semantics unchanged (NaN/null ⇒ both flags false).
2. **Sweep trial-count binding.** Require the schema-2 `backtest_overfitting`
   block's `n_trials` to be ≥ the number of hypothesis-minting sweep cells in
   the notebook (the northset lane's `n_counted_trials` pattern), so DSR
   deflation cannot use a smaller universe than the rejection table.
3. **Anytime-valid option for the grind.** Expose e-BH (`metrics/anytime_fdr`,
   already implemented and Monte-Carlo-asserted) as the adjustment for
   sequentially-stopped ledgers: `reality/fdr.py` gains an `e_bh` path keyed
   off stored e-values where lanes mint them; p-value BH remains the default.
   Cite Wang & Ramdas 2022 / arXiv:2502.08539 in docstrings.
4. **Self-audit p-hacking diagnostic (warning, never a gate).** A
   `dipcatcher reality phack-audit` diagnostic over the accumulated
   `TrialLedger`: 5%-caliper ratio, s-value distribution, z-curve-style
   discovery-rate estimate. Explicitly labeled low-powered at lab scale
   (Hilger et al. 2019; Elliott et al. 2022) — it watches the *grind*, not
   the literature.
   Acceptance: unit tests asserting forged rejections fail verification;
   BH recomputation matches `metrics/inference.benjamini_hochberg` on seeded
   fixtures; caliper diagnostic size-checked on global-null SYNTHETIC ledgers.

### B. Claim→receipt linkage checks (closes §2.5)

1. **Reward-side resolution.** `fx1/reward.score_response` takes an optional
   `receipt_resolver` (injectable; default = scan `receipts/` +
   `data/metadata/research/runs`): `cites_receipt` is awarded only when the
   cited hash resolves to a receipt that passes `verify_receipt_file` /
   `verify_research_artifact`. Unresolvable citations keep
   `honesty_clean` untouched but earn nothing. This is a strengthening of the
   reward contract, not the honesty contract.
2. **Claim Provenance Rate (CPR) for published prose.** A docs scanner
   (reusing `leakage.patterns` normalization) over `docs/**` +
   `day_grind_progress.md` that extracts numeric performance-shaped claims
   and reports which carry a resolvable receipt pointer within proximity.
   Output: `docs/evidence/claim_provenance.json` + a CI *warning* first
   (mirroring the leakage-scan WARN→error migration precedent, DESIGN.md
   §13). Meta-discussion allowlist by literal hash, exactly the LH008
   pattern (`LH008_LITERAL_ALLOWLIST`).
3. **Forbidden-headline lint for the published docs channel** with the
   same literal-hash allowlist mechanism; seed the allowlist from today's
   159 legitimate lines so day one is green, then fail closed on new ones.
   This replaces the manual mkdocs `exclude_docs` triage with a scanned,
   reviewable list.
4. **Upgrade the fx-1 text matcher by sharing the stronger one.** Make
   `fx1.honesty` delegate headline detection to the NFKC/casefold,
   alias-expanded, proximity-window, spelled-out-number matcher
   (`quant_fund.leakage.patterns.find_forbidden_headline` /
   `find_spelled_out_headline`) while keeping `FORBIDDEN_HEADLINE_TOKENS` as
   the mirrored set. Layering caution: `fx1` must not import `quant_fund`
   beyond the pinned edge list (ADR-0002) — so either vendor the matcher into
   `fx1.honesty` with a drift test asserting identical behavior on a shared
   fixture corpus, or extend the pin deliberately in an ADR. Add adversarial
   fixtures from §2.2 (homoglyph, sentence-gap, Spanish, paraphrased
   live-money, label-laundering) as blocking tests.
5. **Claim-local SYNTHETIC labels.** Tighten the synthetic rule from
   whole-text to per-claim-window (a `SYNTHETIC` label within N chars of the
   numeric presenting claim), so one label cannot launder unrelated
   real-data numbers in the same message.
   Acceptance: every §2.2 probe becomes a blocking test; reward probing shows
   fake hashes score 0 on `cites_receipt`; docs scanner green at HEAD with
   the seeded allowlist.

### C. Pre-registration of study specs (closes §2.6)

1. **Generalize the reality-lane freeze.** A `study_spec.v1` JSON schema
   (hypotheses + primary test + grid + costs/risk-gate + splits + decision
   rule + evidence-tier target), SHA-256-stamped into any receipt produced
   under it; the runner refuses configs that differ from the freeze
   (`assert_cost_lock` pattern, generalized beyond costs).
2. **Deviations log, mechanically classified.** Post-freeze changes append to
   an immutable deviations file inside the receipt directory: each entry
   carries `results_seen: bool` (per `prereg`'s amendment/deviation rule) and
   renders into a Willroth–Atherton-style deviations table in the receipt
   markdown. A receipt with `results_seen=true` deviations on the *primary*
   test is downgraded (§D), never deleted — blocked tournaments stay
   reviewable failure records, same philosophy.
3. **Prospective lanes keep the strongest form.** `PROSPECTIVE_SOTA.md`'s
   single-predeclared-primary + Bonferroni-split + permanent-interrupt
   protocol is the template; new prospective journals copy it verbatim.
4. Optional, later: a RegCheck-style LLM comparison of a study's prose
   write-up against its frozen spec, run as an *advisory* lane with quoted
   evidence — never as a sole gate (LLM judges are not fail-closed).
   Acceptance: seeded fixtures — drifted grid fails; drifted cost fails;
   undisclosed post-results spec edit fails; amendment-before-results passes
   and is labeled as such.

### D. Evidence-hierarchy labels (closes §2.7)

1. **Ordered `evidence_tier` enum on receipt.v2** (schema bump, additive
   field; schema-2 receipts without it stay valid):
   - `E0_fixture` — dgp=fixture toys; excluded from panel gates (existing
     rule) — correctness only.
   - `E1_synthetic` — seeded SYNTHETIC panels: engine-correctness evidence.
   - `E2_retrospective_public` — public/file adapters, session-close marks
     (current Yahoo/Stooq lanes; no as-of vintage).
   - `E3_retrospective_pit` — point-in-time-stamped collections with
     release/ingest timestamps (`public sources` lane, vendor skeleton).
   - `E4_prospective_sealed` — frozen-protocol journal with sealed labels
     (`PROSPECTIVE_SOTA` pattern).
   - `E5_forward_shadow` — deployment-shaped paper/shadow record.
   Tier is derived from existing stamps (`data_label`, `point_in_time`,
   `execution_claim`, scope enums) by a pure function, and the derivation is
   verified, not trusted — a receipt claiming `E3` without PIT stamps fails.
2. **GRADE-style downgrade reasons list** per receipt: `uncounted_trials`,
   `post_results_deviation_primary`, `unblinded_holdout_contact`,
   `single_seed`, `no_variation_reported`, `missing_run_count`,
   `cumulative_disclosure_flag`. Each reason present drops the tier one
   level; the report renders reasons verbatim (mirrors GRADE's "named
   downgrade domains").
3. **fx-1 quoting rule (strengthening):** whenever fx-1 output cites a
   numeric research result, the eval/reward channels reward stating the tier
   (`_EVIDENCE_CLASS_RE` generalized to the tier vocabulary), and the corpus
   teaches tier-echoing from receipt rows. Abstain-over-claim on unbacked
   numbers (ECE selective-verification behavior, arXiv:2607.18240).
4. **Cumulative-disclosure watch (SDC transfer):** the evidence page and
   `docs/evidence` index record which receipts share datasets
   (`dataset_content_sha256`); a new publication over the same panel emits a
   warning listing jointly-disclosable field deltas. Redaction functions
   stay as they are; this only *tracks* composition across releases.
5. **Pineau receipt items:** add optional-but-validated `n_runs`,
   `variation` (CI/std key presence when a mean is reported), and
   `runtime_seconds` to lane receipts; the model-card template already
   demands seeds/trial counts — align its wording with the checklist v2.0.
   Acceptance: tier derivation pure-function tests; forged-tier receipts
   fail verification; downgrade reasons render into markdown; no existing
   valid receipt changes verdict (schema compatibility test over
   `receipts/`).

### Sequencing

| # | Item | Effort | Depends on |
|---|---|---|---|
| A1 | Verifier re-derivation of reject flags + BH | S | — |
| B4/B5 | Shared hardened matcher + claim-local SYNTHETIC | M | ADR-0002 pin decision |
| B1 | Reward receipt resolution | S | — |
| D1/D2 | evidence_tier + downgrade reasons | M | — |
| A2 | Sweep trial-count binding | S | A1 |
| C1/C2 | study_spec.v1 freeze + deviations log | M | D1 (tier downgrade hook) |
| B2/B3 | Docs CPR scanner + forbidden-headline doc lint | M | B4 (matcher) |
| A3/A4 | e-BH ledger path + p-hack self-audit | S/M | A1 |
| D3–D5 | fx-1 tier vocabulary, disclosure watch, Pineau keys | M | D1 |

Gate discipline for every item: ruff + mypy + focused tests green; SYNTHETIC
fixtures only; no commits unless the lane convention says so; no change to
`FORBIDDEN_RESEARCH_METRIC_KEYS`, `FORBIDDEN_HEADLINE_TOKENS` membership
semantics, `live_pnl_claim` typing, or any fail-closed default without a new
ADR *and* an extended (never weakened) inheritance test.

---

## 4. References

1. Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B* 57(1):289–300.
2. Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Ann. Stat.* 29(4):1165–1188.
3. Wang, L. & Ramdas, A. (2022). False discovery rate control with e-values: e-BH. *JRSS-B* 84(4).
4. Wang, Dandapanthula & Ramdas (2025). Stopped e-BH under optional stopping. arXiv:2502.08539.
5. Grünwald, P., de Heide, R. & Koolen, W. (2024). Safe testing. *JRSS-B* (discussion paper).
6. Vovk, V. & Wang, R. (2021). E-values: calibration, combination, and applications. *Ann. Stat.* 49(3).
7. Xu, L. & Ramdas, A. (2024). Online multiple testing with e-values (e-LOND / e-SAFFRON).
8. Harvey, C.R., Liu, Y. & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies* 29(1):5–68.
9. Harvey, C.R., Sancetta, A. & Zhao, Y. (2026). What threshold should be applied to tests of factor models? NBER w34898 (local FDR recommendation).
10. Harvey, C.R. & Liu, Y. (2015+). Backtesting. *Journal of Portfolio Management* (haircut Sharpe ratios).
11. Bailey, D.H. & López de Prado, M. (2012/2014). The Sharpe ratio efficient frontier / the deflated Sharpe ratio. *JPM*.
12. Bailey, Borwein, López de Prado & Zhu (2017). The probability of backtest overfitting (CSCV). *J. Comput. Finance*.
13. White, H. (2000). A reality check for data snooping. *Econometrica* 68(5).
14. Hansen, P.R. (2005). A test for superior predictive ability. *JBES*.
15. Romano, J. & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*.
16. Hansen, Lunde & Nason (2011). The model confidence set. *Econometrica* 79(2).
17. Gelman, A. & Loken, E. (2013/2014). The garden of forking paths. Columbia University working paper / *Psych. Bull.* commentary.
18. Steegen, Tuerlinckx, Gelman & Vanpaemel (2016). Increasing transparency through a multiverse analysis. *Perspect. Psychol. Sci.* 11(5):702–712. doi:10.1177/1745691616658637.
19. Simonsohn, Simmons & Nelson (2020). Specification curve analysis. *Nature Human Behaviour* 4:1208–1214. doi:10.1038/s41562-020-0912-z.
20. Simmons, Nelson & Simonsohn (2011). False-positive psychology. *Psychol. Sci.* 22(11):1359–1366.
21. Gerber, A. & Malhotra, N. (2008). Do statistical reporting standards affect what is published? (caliper test). *Quarterly Journal of Political Science* / *Sociol. Methods Res.*
22. Hilger, König & Löwel (2019). Examining publication bias — simulation-based evaluation of statistical tests. *Res. Synth. Methods* (PMC5712469).
23. Elliott, Kudrin, Roth & Wu (2022). (When) can we detect p-hacking? arXiv:2205.07950 (CS2B; power of caliper/p-curve tests).
24. Simonsohn, Nelson & Simmons (2014). p-curve. *J. Exp. Psychol. Gen.* 143(2).
25. Bartoš, V. & Schimmack, U. (2022). z-curve 2.0. *Meta-Psychology* 6.
26. Nuijten, Polanin et al. (2020). statcheck: automatically detect statistical reporting inconsistencies. *Res. Synth. Methods*. doi:10.1002/jrsm.1408; statcheck.io.
27. Nosek, Ebersole, DeHaven & Mellor (2018). The preregistration revolution. *Science* 361:26–28.
28. Chambers, C. & Tzavella, L. (2022). The past, present and future of registered reports. *Nature Reviews Psychology* 1:29–42. doi:10.1038/s44159-022-00029-z.
29. Willroth, E.C. & Atherton, O.E. (2024). Best laid plans: a guide to reporting preregistration deviations. *AMPPS*. doi:10.1177/25152459231213802.
30. Nagabhushana, P. et al. (2026). RegCheck: structured comparisons between study registrations and papers. arXiv:2601.13330.
31. `prereg` (PyPI) — freeze-a-plan hash + amendment/deviation log keyed to results-seen state.
32. Guyatt, Oxman, Schünemann et al. (2011). GRADE guidelines 1–2. *J. Clin. Epidemiol.*; Balshem et al. (2011); Cochrane Handbook ch. 14; Core GRADE 4 (2025), *BMJ*. doi:10.1136/bmj-2024-083864.
33. Wasserstein, Schirm & Lazar (2019). Moving to a world beyond "p < 0.05". *The American Statistician* 73(sup1):1–19. doi:10.1080/00031305.2019.1583913.
34. Amrhein, Trafimow & Greenland (2019). Remove, rather than ignore, the null hypothesis significance testing frame. *The American Statistician* (s-values).
35. Yang, C. et al. (2026). Calibrated selective fact-checking via evidence chain evaluation (source levels L1–L5, abstention). arXiv:2607.18240.
36. ScientistOne (2026). Towards human-level autonomous research via chain-of-evidence. arXiv:2605.26340 (CoE Audit: score verification, spec violation, reference verification, method–code alignment; Claim Provenance Rate).
37. ReAgent (2026). Beyond the text: verifying that agent-written papers are backed by their artifacts. arXiv:2609.22111.
38. Dammu et al. (2024). ClaimVer: explainable claim-level verification and evidence attribution through knowledge graphs (KG Attribution Score). arXiv:2403.09724.
39. PaperTrail (2026). A claim-evidence interface for grounding provenance in LLM-based scholarly Q&A. arXiv:2602.21045.
40. Pineau, J. (2020). Machine Learning Reproducibility Checklist v2.0. cs.mcgill.ca/~jpineau/ReproducibilityChecklist-v2.0.pdf.
41. Pineau, Vincent-Lamarre, Sinha, Larivière et al. (2021). Improving reproducibility in machine learning research. *JAIR* 70; JAIR's four mechanisms (same source, doi via jair.org/index.php/jair/article/download/16905).
42. Hundepool, Anker, Espinosa et al. *Handbook on Statistical Disclosure Control* (sdctools.github.io/HandbookSDC); Willenborg & de Waal (2001). *Elements of Statistical Disclosure Control*.
43. Karr, A.F. & Reiter, J.P. (2021). Statistical data privacy: a song of privacy and utility. *Annu. Rev. Stat. Appl.* 8. doi:10.1146/annurev-statistics-033121-112921.
44. Sweeney, L. (2002). k-anonymity. *IJUFKS* 10(5–6):557–570; Dwork & Roth (2014). The algorithmic foundations of differential privacy.
45. Ioannidis, J. & Trikalinos, T. (2007). An exploratory test for an excess of significant findings (TES). *Clin. Trials*.

In-repo anchors: `src/quant_fund/research/catalog/registry.py`,
`src/quant_fund/research/verify.py`, `src/quant_fund/research/receipt_v2.py`,
`src/quant_fund/leakage/patterns.py`, `src/quant_fund/reality/fdr.py`,
`src/quant_fund/validation/fdr.py`, `src/quant_fund/metrics/anytime_fdr.py`,
`src/quant_fund/research/reality_sweep.py`, `src/quant_fund/audit/ledger.py`,
`src/fx1/honesty.py`, `src/fx1/reward.py`, `scripts/build_evidence_report.py`,
`docs/adr/0008-forbidden-metric-key-scan.md`,
`docs/BACKTEST_OVERFITTING.md`, `docs/RECEIPT_V2.md`,
`docs/RECEIPT_VERIFICATION.md`, `docs/REALITY_PREREGISTRATION.md`,
`docs/PROSPECTIVE_SOTA.md`, `docs/DATA_SOURCE_LABELS.md`,
`docs/INSTITUTIONAL_READINESS.md`, `tests/fx1/test_honesty_inheritance.py`.
