# Honesty Rating Engine — artificial-hedge/dipcatcher

**Engine version:** 1.0 · **Subject:** github.com/artificial-hedge/dipcatcher @ `0e05f653` (main tip, 2026-09-29)
**Core scale:** 0–1000 (Major factors 800 + Minor factors 200) · **Overdrive extension:** +150 (max attainable 1150)

---

## Correction notice — 2026-10-08

**The report below is a snapshot of `0e05f653` (2026-09-29). It is not current
and must not be quoted as a current assessment.** Re-measured on 2026-10-08 at
local `HEAD 5c2805685`. No score below has been changed; the deductions are
annotated in place and the arithmetic is deliberately left as-is. Re-scoring is
a separate, dated act.

**The H1–H5 anti-inflation rules in §0 are unchanged and are the durable asset
of this document.** They are reproduced verbatim. Only the snapshot is stale.

| # | Deduction | Snapshot claim | Position at 2026-10-08 | Status |
|---|---|---|---|---|
| **m6** | −10 (5/15) | "**No LICENSE file.**" | **VOID.** `LICENSE` exists: `ARTIFICIAL HEDGE PROPRIETARY LICENSE`, `Version 1.0, last updated 2026-09-30`. Verified `head -3 LICENSE`. The snapshot is dated one day before the license was written. | **Void — deduct.** The corrected position should be re-scored at the next re-baseline. |
| **M2 / H5** | −14 | "release.yml is tag-gated and **zero tags exist**" | **Pending resolution.** No *release* tag exists: `git tag -l 'v*' | wc -l` → `0`, so `release.yml` has never fired and the deduction is *not* yet wrong. (`git tag | wc -l` → `4`, but all four are `attic/receipt-provenance/*` provenance anchors dated 2026-09-27/28, not releases — "zero tags" is imprecise wording, "zero releases" is exact.) A first `v*` tag is expected once CI returns green, at which point H5's 60% cap lifts. | **Accurate; pending** |
| **m3** | 24/40, H5-capped | "**0 tags, 0 releases**" | **Pending resolution**, same condition as M2/H5 above. The 60% cap stands until a release actually executes. | **Accurate; pending** |
| **M6 / H5** | −10 | "signing/SBOM/SLSA never exercised — no tags, no releases" | **Pending resolution**, same condition as M2/H5 and m3. `git tag -l 'v*' | wc -l` → `0`: the release machinery remains unexercised. | **Accurate; pending** |
| **M7** | −18 | "**fx-1 is a plan, not a model** — no weights, `complete()` raises `NotImplementedError`, training manifest gate exists but no training run has occurred" | **Substantively still correct, with one correction of detail.** `complete()` still raises `NotImplementedError`; no training run has occurred; no fine-tuned fx-1 weights are published. The clause "**no weights**" is imprecise: `artifacts/fx1_tiny_lm/weights.safetensors` (127 KB, ~31k params) is tracked in git. It is a correctness fixture whose own card says *"not an fx-1 release candidate"*, `research_only: true`, with synthetic `eval_delta` placeholders. It does not rebut the deduction; the deduction's *reasoning* should be re-verified against the current tree before being carried forward. | **Correct in substance; re-verify before carry-forward** |
| **M3** | −8 | "drift already acknowledged by the repo itself (open PR #367: docs drift checker … needed, not yet merged)" | **Still accurate.** Re-checked 2026-10-08: no docs-drift checker is wired into any workflow. `grep -rn "drift" .github/workflows/` matches only unrelated honesty/epoch notes, and `scripts/drift_real_drill.py` is a *market* drift detector (changepoint on real tapes), not a docs checker. Only `atlas.yml` (generated diagrams) and `docs.yml` (`mkdocs --strict`) validate documentation, and neither checks prose claims against code. | **Accurate** |
| **M1 / H2** | −10 | "no hosted coverage publication (floor is enforced, trend is not visible)" | Unverified this pass — not re-measured on 2026-10-08. | **Not re-verified** |

**One further caveat the snapshot predates, which cuts *against* the repo.**
The scorecard below awards credit for machinery that is defined but has never
returned a verdict: `gh run list` shows **0 successful runs in 400 consecutive
attempts** as of 2026-10-08. H5 caps unexercised machinery at 60%; a pipeline
that has never once been observed green is a broader instance of that than the
release workflow the snapshot already penalizes. This does not change any
number here, but any re-scoring should weigh it. Findings and remediation:
[`docs/ULTRA_PROD_READINESS.md`](docs/ULTRA_PROD_READINESS.md).

**Rule for the next re-baseline:** re-score at a green `PROD_BASELINE` SHA with
its own date, keep H1–H5 fixed, and publish deductions *before* fixes. Never
remove a deduction without the evidence that voided it.

---

## 0. What makes this an *honesty* engine

Most repo scoreboards reward claimed quality. This one rewards **provable** quality and penalizes inflation. Five anti-inflation rules are applied before any points are awarded:

| Rule | Effect |
|---|---|
| **H1 — Evidence or zero.** A capability claimed in docs/README but not backed by committed, inspectable artifacts (code + tests + CI gate) earns 0 for that line item. | Kills marketing points |
| **H2 — Self-reported metrics cap at 50%.** Any metric not enforced by a CI gate (e.g. a coverage number only humans check) can earn at most half its points. | Kills unenforced dashboards |
| **H3 — SYNTHETIC ≠ live.** Evidence labeled synthetic/research counts toward *correctness machinery*, never toward *market performance*. Mirrors the repo's own `DATA_LABEL` contract. | Kills backtest-to-alpha laundering |
| **H4 — Red-main penalty.** Recurring "fix broken main gates" commits in history subtract from the CI/CD factor — gates that break weekly are weaker than gates that hold. | Kills badge theater |
| **H5 — Unexercised machinery cap at 60%.** Pipelines never run for real (e.g. a release workflow with zero tags) earn at most 60% of their points. | Kills on-paper infrastructure |

Scoring anchors per factor: **100%** = industry-grade, externally verifiable · **75%** = strong, minor gaps · **50%** = real but partial · **25%** = exists, weak · **0%** = absent or claim-only.

---

## 1. Major factors (800 pts)

### M1. Correctness & testing depth — 160 max → **125**

Evidence found:
- Dedicated test lanes: `tests/{unit, property, regression, end_to_end, perf, formal, fx1, native, leakage_fixtures}` — far beyond a single `pytest` folder.
- Hypothesis property suites with a **seed-stable CI profile** (falsifying examples re-run identically), a differential fuzzer asserting event-engine ≡ fast_replay NAV within 1e-9, chaos fault-injection over source adapters (17 fault classes), KATs for money paths, and a mutmut mutation lane (PR #372, open).
- Coverage floor `fail_under = 80` in `pyproject.toml`, enforced in CI (with the hidden-`.coverage` upload fix).
- Leakage scanner test-fixtured both ways: 16 adversarial positives + 5 documented negatives pinning the *residual ceiling* — the repo tests its own watchdog's blind spots. Rare.

Deductions: −15 for H4 (multiple "repair main-broken gates" / "unblock main lint/test gates" commits — the gates bite, but main has been red repeatedly); −10 for no hosted coverage publication (floor is enforced, trend is not visible); −10 for mutation/property lanes partially in open PRs rather than merged (H1 applies to unmerged work).

### M2. CI/CD & automation — 120 max → **92**

Evidence found:
- 17 workflows: `ci.yml` (28 KB), OS×Python matrix (ubuntu/macos/windows × 3.12–3.14, count-sharded), `codeql`, `dependency-review`, `scorecard`, `secret-scan` (full-history gitleaks), `docs` (mkdocs `--strict`), `perf-baseline` with calibration-normalized timings, `property-nightly`, `adversarial-properties`, `atlas` (diagram freshness gate), `simtest`, `proofcore`, `fx1`, `replay_viz`, `web_explorer`, `release`.
- All actions SHA-pinned with least-privilege permissions — supply-chain hygiene most repos never reach.

Deductions: −14 H4 (red-main recurrence); −14 H5 (release.yml is tag-gated and **zero tags exist** — the pipeline has never shipped anything).

> **[Annotation 2026-10-08]** Still accurate in substance — `git tag -l 'v*' | wc -l` → `0`, so the release pipeline has never shipped. (`git tag | wc -l` → `4`, all `attic/receipt-provenance/*` provenance anchors from 2026-09-27/28, not releases.) Marked *pending*: H5's cap should lift once a first `v*` tag executes a real release build. Separately, `gh run list` reports 0 successful runs in 400 attempts, so "red-main recurrence" understates the position. See the [correction notice](#correction-notice--2026-10-08).

### M3. Documentation & knowledge — 100 max → **82**

Evidence found:
- ~90 doc files: ARCHITECTURE + generated **atlas with a CI freshness gate**, ADRs, 58 KB operations runbook, 104 KB MATH_SPEC, audit ledgers (P61 money / P62 stats / P63 data / P64 dist / P65 micro / P66 infra), pre-registration doc, published deflated trial, evidence index **rendered only from committed receipts** (`make evidence`), RESEARCH100 catalog mapping 100 papers to executable components.

Deductions: −10 for bloat/rot surface (DATA_CONTRACTS.md 279 KB, SOTA_GAP_ANALYSIS.md 151 KB committed monoliths); −8 for drift already acknowledged by the repo itself (open PR #367: "docs drift checker for code refs + honesty sweep of stale paths" — needed, not yet merged).

> **[Annotation 2026-10-08]** Re-verified: still no docs-drift checker in CI. `grep -rn "drift" .github/workflows/` matches only unrelated honesty/epoch comments; `scripts/drift_real_drill.py` is a *market* changepoint detector, not a docs checker. Only `atlas.yml` (generated diagrams) and `docs.yml` (`mkdocs --strict`) validate docs, and neither checks prose claims against the tree. Deduction stands.

### M4. Architecture & code quality — 120 max → **95**

Evidence found:
- Enforced layering gates (SCC import rules, quant_fund↔fx1 edge pinned to version re-export only), public-API snapshot tests, `py.typed`, **mypy --strict allowlist of 393 modules that cannot shrink**, mccabe ratchet (196 → 74 ceiling), except-Exception ratchet (count can only fall), Ruff format+lint, lazy-import discipline with a 1s CLI cold-start budget.
- Two-package monorepo (quant_fund lab + fx1 model program) with a hard live-order refusal at config load.

Deductions: −15 (72–75 broad `except Exception` handlers remain — ratcheted, but that is a lot of swallow sites for a money-adjacent codebase); −10 (mccabe 74 ceiling is 5–7× conventional ceilings; ratchet is a floor-hold, not yet a cleanup).

### M5. Reproducibility & evidence integrity — 120 max → **108**

The crown jewel, and the reason this engine exists:
- Sealed receipts (`receipt_sha256` self-hash, `inputs_sha256`, `script_sha256`, `bar_files_sha256`), `verify-research` fails closed on any mismatch; proof bundles with HMAC, hash-chained trial ledgers, cross-thread provenance recorder, decision clock derived from data (not wall clock).
- **Published negative results**: `selected_band: null` / `development_eligible: false` receipts committed; a blocked tournament stays a reviewable sealed failure.
- Pre-registered reality trial on real Yahoo daily data — verdict published as **"deflated"**. Almost no quant repo publishes its own deflation.
- qlib parity receipt (nav_max_rel_diff ≈ 1.03e-07) with a verbatim disclaimer explicitly disclaiming superiority claims.
- SYNTHETIC labeling enforced in CI smoke; gitleaks with narrowly scoped allowlists.

Deductions: −12 (seal drift has occurred — multiple "re-seal" commits show hashes had to be rebound after refactors; the machinery self-corrects, but sealing is still reactive).

### M6. Security & supply chain — 90 max → **74**

Evidence found: CodeQL + Scorecard + dependency review + full-history gitleaks, SHA-pinned actions, CycloneDX SBOM + Sigstore signing + SLSA provenance defined for releases, SECURITY.md private disclosure, API key/loopback-only default, config-path allowlist, `.env.example`, secrets-scan allowlists scoped to sha256 evidence pins.

Deductions: −10 H5 (signing/SBOM/SLSA never exercised — no tags, no releases); −6 (no external security review; 900 KB uv.lock dependency surface un-audited by third party).

> **[Annotation 2026-10-08]** The H5 half is still accurate — `git tag -l 'v*' | wc -l` → `0`, so no release has run and SBOM/Sigstore/SLSA remain unexercised; mark *pending*. The −6 external-security-review half is unaffected by that and remains open.

### M7. Domain substance — 90 max → **62**

Evidence found: genuine quant machinery — PIT vault with watermark policy, leakage scanner LH001–LH014 (AST-based, evasion-hardened), CPCV/DSR/PBO/SPA reality filters, e-processes/anytime-valid inference (e-LORD, e-SAFFRON), conformal + calibration batteries, proper scoring rules (pinball/CRPS/PIT/QLIKE/Brier/ECE/Kupiec), exchange calendars with verified unscheduled closures, Almgren–Chriss execution, agent-based LOB simulator, Rust kernels.

Deductions (H3 applies hard): −18 (**fx-1 is a plan, not a model** — no weights, `complete()` raises NotImplementedError, training manifest gate exists but no training run has occurred; the "7T fine-tune" ambition has zero committed evidence); −10 (nearly all evidence is SYNTHETIC; the one real-data trial is deflated — honest, but alpha is unproven by design).

> **[Annotation 2026-10-08]** Re-verified: `complete()` still raises `NotImplementedError`, still no training run, still no fine-tuned fx-1 weights published. The deduction stands. One detail is imprecise — "no weights" is not literally true. `artifacts/fx1_tiny_lm/weights.safetensors` (127 KB, ~31k params) is tracked in git, but its own model card says *"not an fx-1 release candidate"*, `license_tier: internal_research`, `research_only: true`, and its `eval_delta` values are synthetic ship-gate placeholders rather than measured evals. It is a correctness fixture. Correct the wording, not the score.

**Major subtotal: 638 / 800**

---

## 2. Minor factors (200 pts)

### m1. Repo hygiene & git discipline — 40 max → **18**

- **>100 open PRs** (#302–#401 effectively all open) — a mountain of finished-looking work stranded unmerged; several agent branches (`codex/*`, `cursor/*`, `copilot/*`, `devin/*`) stale.
- Committed bloat/tool litter: `.test_durations` (777 KB), `day_grind_progress.md` (139 KB), `INFLIGHT` (21 KB), `.cursor/`, `.serena/`, `.freebuff/`, `.box-soft-verify/`, `.dsh-24x7/` in the tree.
- Plus side: rich conventional-commit messages with co-author trailers and root-cause narratives — genuinely excellent commit craft. That's what saves this from single digits.

### m2. Community & collaboration — 30 max → **17**

CONTRIBUTING.md, CODE_OF_CONDUCT (Contributor Covenant 2.1), CITATION.cff, APPLY.md, AGENTS.md all present. But **zero issues ever opened** — no issue tracker culture, no discussions, one human + a bot fleet. A repo nobody can file a bug against is not collaborable yet.

### m3. Release & packaging maturity — 40 max → **24**

pyproject + uv.lock (single-numpy pin discipline), console scripts (`dipcatcher`, `fx1`), Dockerfile + compose, `.python-version`, versioned `0.4.0`. But: **0 tags, 0 releases**, PyPI trusted publishing deliberately off, CHANGELOG seeded-but-manual. H5 capped this factor: 24 = 60% of what the packaging would otherwise earn.

> **[Annotation 2026-10-08]** Still accurate in substance — `git tag -l 'v*' | wc -l` → `0`, so "0 releases" is exactly right ("0 tags" is imprecise: `git tag | wc -l` → `4`, all `attic/receipt-provenance/*` provenance anchors, not releases). Marked *pending*: the H5 cap should lift once `release.yml` actually executes for a real tag. Note also that `fx1.__version__` = `0.4.0` is the **harness API** version, not a model release — see `docs/FX1_API_STABILITY.md`.

### m4. Performance engineering — 30 max → **26**

Rust native kernels with enforced speedup floors, numba kernel + interpreted fallback proven byte-identical, calibration-normalized perf gate (sort-reference, median-of-7), 1s CLI startup budget, count-balanced test sharding. Docked only for perf gates being young and threshold-tuned twice already.

### m5. UX / DX — 30 max → **22**

`make sync && uv run dipcatcher research` quickstart, `doctor` preflight with nonzero exit, honest `DATA_LABEL=SYNTHETIC` printing, web receipt explorer + canvas replay viewer with a 2 MB fixture cap. Docked for steep domain ramp and no PyPI install path (`pip install fx-1` doesn't exist yet).

### m6. Licensing & legal — 15 max → **5**

**No LICENSE file.** For a repo this engineered, the single cheapest legal artifact is missing. CITATION.cff and third_party/ awareness (HF license doc) keep it off zero.

> **[Annotation 2026-10-08 — THIS DEDUCTION IS VOID.]** `LICENSE` exists: `ARTIFICIAL HEDGE PROPRIETARY LICENSE`, `Version 1.0, last updated 2026-09-30`. This snapshot is scoped to `0e05f653` (2026-09-29) — the license postdates it by one day. The original text is preserved above as the engine recorded it; it is not to be carried forward. This factor should be **re-scored** at the next re-baseline. See the [correction notice](#correction-notice--2026-10-08).

### m7. Project management & tracking — 15 max → **9**

INFLIGHT log + day_grind_progress.md + ADRs + audit ledgers form an unusual but real paper trail. No milestones, no roadmap board, no issue triage — PM is a flat file, not a system.

**Minor subtotal: 121 / 200**

---

## 3. Verdict

```
╔══════════════════════════════════════════════════╗
║  MAJOR   638 / 800                               ║
║  MINOR   121 / 200                               ║
║  ────────────────────────                        ║
║  CORE SCORE          759 / 1000                  ║
║  OVERDRIVE EARNED      +3 / +150                 ║
║  EFFECTIVE             762 / 1150                ║
╚══════════════════════════════════════════════════╝
```

**Reading:** 759/1000 places dipcatcher in the top few percent of solo-maintained quant repos on *engineering integrity* — the evidence machinery (M5) is genuinely rare anywhere, including professional shops. The score is held back not by what the repo claims, but by what it hasn't done: ship a release, merge its own queue, license itself, and exist as a model.

---

## 4. Overdrive tier — how to push **past** 1000 (+150 ceiling)

The core 1000 measures "is this repo excellent." Overdrive measures what no rubric can anticipate: external proof. Points here stack on top of 1000, giving a theoretical max of **1150**.

| # | Overdrive factor | Pts | Status today |
|---|---|---|---|
| O1 | **External reproduction** — a stranger clones, runs `make test` + `verify-research`, and publishes the result | +30 | 0 — never attempted |
| O2 | **Prospective real-data track record** — 3–6 months of sealed forward-shadow receipts on live data, pre-registered, win *or* lose | +30 | **+3** partial (one deflated pre-registered trial) |
| O3 | **Published artifact** — paper/DOI/venue citing the methodology | +20 | 0 |
| O4 | **Adoption** — external contributors, stars, forks, derivative use | +25 | 0 |
| O5 | **fx-1 exists** — a trained checkpoint with sealed evals beating the K3 base on the capability battery | +30 | 0 — code + plan only |
| O6 | **Independent security review** — third-party pentest/audit of the API + supply chain | +15 | 0 |

**Fastest path over 1000** (ordered by points-per-effort):

1. **Merge or close the ~100 open PRs** and cut tag `v0.4.0` → exercises release.yml, SBOM, Sigstore, SLSA for real. Recovers ~24 (M2) + ~12 (m3) + ~14 (m1), and clears the H5 caps. *Gets to ~800.*
2. **Add a LICENSE** (one file). +8–10 (m6). *~810.*
   > **[Annotation 2026-10-08]** **Done — this step is obsolete.** `LICENSE` exists (proprietary v1.0, 2026-09-30), so m6's "No LICENSE file" deduction is void. Preserved above as the snapshot recorded it; do not carry forward.
3. **Open the issue tracker**, file known gaps as issues, invite one external contributor. +8 (m2), starts O4. *~818.*
4. **Publish hosted coverage + run the mutation lane on main.** +10 (M1). *~828.*
5. **Keep the forward-shadow running on real data for 90 days, sealed, pre-registered.** Up to +27 more of O2. *~855 + O2 progress.*
6. **Train even a small fx-1 student** and seal the base-model comparison. Partial O5 (+10–15 for an honest small model beats 0 for a theoretical 7T one). *~870+.*
7. **External reproduction + audit** (O1, O6) last — they need 1–6 done first. Full house: **759 → ~1000 core, +150 overdrive = 1150.**

---

## 5. Honesty ledger (what this engine refused to count)

- "7T-parameter Kimi-K3 fine-tune" — no weights, no training run → **0 pts under O5/M7**, not partial credit.
- Release/SBOM/signing infrastructure — never executed → capped at 60% (H5).
- All unmerged PR work (#302–#401) — not on main → excluded from M-scores (H1), though it signals velocity.
- Synthetic backtest performance — counts toward correctness machinery only, never toward alpha (H3).
- The deflated reality trial — counted as a **positive** under M5 (honesty) and a **zero** under alpha evidence. Both simultaneously, which is the entire point.

*Engine rule for re-runs: re-score after every merged PR batch or tag. Factors M2, m1, m3 move fastest.*
