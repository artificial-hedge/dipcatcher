# Audit — fx1 internals, wave 3 (re-verification depth pass)

Scope: every module under `src/fx1/` (66 files) re-read line-by-line after
waves 1–2 (`AUDIT_FX1.md`, `AUDIT_FX1_EVALHARNESS.md`) marked them audited.
Target surface: eval-harness internals, corpus builders, reward/MRM edge
cases, hash staleness, tokenizer/token determinism, and subprocess/network
calls in train lanes.

Verdict per module family below; defects found this round are listed with
their fixes. Regression coverage: `tests/fx1/test_audit_fixes_wave3.py`
(32 tests).

## Defects found and fixed

### Eval surface / contamination (`eval/`)

1. **`eval/__init__.py` — `eval_prompt_surface()` raised `NameError`**
   (order-dependent). The function referenced `DEFAULT_BANK`,
   `REDTEAM_TASKS`, `rephrased_twins`, `masked_twins` as bare globals, but
   the package's PEP 562 `__getattr__` is never consulted for bare-global
   lookups inside a function — the call only worked if some earlier
   attribute access had already populated `globals()`. A quality gate that
   crashes on a cold import is not fail-safe. Fixed with explicit local
   submodule imports inside the function.
2. **`eval/__init__.py` — prompt surface screened only `role == "user"`**
   messages. A corpus that copied an eval item's *seeded assistant
   completion* or *system prompt* would not overlap-screen against it.
   Surface now collects every message of every role across bank, redteam,
   rephrased, and masked twins.
3. **`eval/contamination.py` — vacuous pass on empty inputs.**
   `run_contamination_audit` with an empty corpus or empty prompt list
   produced a clean bill of health over nothing. Now raises `ValueError`.
4. **`eval/suite.py` — malformed tasks admitted.** `required_tokens`
   accepted blank strings (a task could "require" nothing while reporting a
   check), `forbidden_patterns` were never compiled at validation time (a
   broken regex only surfaced — or silently didn't — at scoring), and
   `run_suite` accepted duplicate task names, which made downstream
   name-keyed comparisons collide. All three now fail at construction /
   suite start.

### Corpus builders (`data/`)

5. **`data/quality.py` — near-dup detection was windowed.** Dedup compared
   each candidate's shingles only against the last 500 kept examples that
   shared a shingle (`shingle_index[-500:]`): a near-duplicate separated by
   >500 inserts evaded detection entirely. Replaced with a per-shingle
   inverted index so every kept example sharing any shingle is compared.
6. **`data/quality.py` — `_hash_lines` was boundary-ambiguous.**
   `"".join(lines)` hashes `["ab","c"]` and `["a","bc"]` identically. Now
   hashes the exact file bytes (`lines + trailing newline`), matching what
   `split_manifest` claims to pin.
7. **`data/quality.py` — vacuous frozen splits certified.** `frozen_split`
   could hash and write a manifest for a degenerate train/val pair where
   one side was empty — a receipt "proving" a split that never happened.
   Now raises `ValueError`; split writes are atomic (`atomic_write_text`).
8. **`data/quality.py` — silent drops uncounted.** Empty and over-length
   examples were dropped without counters, hiding data loss from the
   quality report. Added `empty_removed` and `over_length_removed` to
   `QualityReport`.
9. **`data/receipts.py` — truthy non-bool `research_only` passed.**
   `_eligibility` treated `"yes"` / `1` as research-only, letting a live
   claim be reclassified. Now `research_only` must be the literal `True`
   (or the `claim` field equal to `"research_only"`).
10. **`data/ledgers.py` — truthy non-bool `live_pnl_claim` evaded.**
    `_claims_live` only caught `True`/`"true"`; `1`, `"yes"`, `"recorded"`
    produced positive SFT examples asserting live P&L. New
    `_is_live_claim_value` covers bool, numeric, string, and structured
    values.
11. **`data/traces.py` — `admit` trusted a self-declared flag.** A
    trajectory with `verify_ok = True` and *no* artifact receipts was
    admitted as positive evidence. Admission now requires `verify_ok` AND a
    non-empty `artifact_receipts` list; otherwise the trace is a labeled
    negative.
12. **`data/ledger.py` — hash fields were free-form.** `LedgerEntry`
    accepted short/garbage digests into the Merkle chain.
    `transform_sha256` and `example_sha256` are now pinned to 64 chars.
13. **`data/notebooks.py` — silent content truncation.**
    `notebook_examples` embedded only `content[:1500]` while its contract
    was per-`chunk_chars` chunking — corpus rows silently dropped code
    beyond 1500 chars. Full content is now embedded.
14. **`data/sources/base.py` — `as_of` accepted any string.** The PIT
    leakage gate keys on `as_of`; `"banana"` satisfied "present" while
    pinning a garbage observation date. Now strict `YYYY-MM-DD` regex +
    `date.fromisoformat` realness (rejects `2026-02-30`).

### Honesty gate (`honesty.py`)

15. **Reporting-verb / function-word evasion.** `_contains_forbidden_headline`
    only bridged token→number through `of|=|:|is|was|at|to` plus
    punctuation — so "Sharpe of the strategy is 2.1", "calmar ratio for the
    whole period was 0.8", "nav reached 1.9" all passed. Bridge and
    connector vocabularies widened (reporting verbs, qualifiers, domain
    nouns) and the repetition cap raised to 6; verified non-headline
    discussion texts still pass. Token list itself unchanged (no
    `FORBIDDEN_RESEARCH_METRIC_KEYS` sync needed).

### Score contracts (`hypotheses.py`, `bench/`)

16. **`hypotheses.py` — substring evasion of the score contract.**
    `validate_trace_scores` compared whole tokens, so `crps_realizedpnl`
    or `brier_drawdownsharpe` passed while smuggling a forbidden headline.
    Substring check added (allowed-score tokens exempted so `sharpness_*`
    still passes), and the validator is now wired into `ResearchTrace` as a
    pydantic field validator — previously it ran only at admission.
17. **`bench/dip.py` — recovery horizon off-by-one.** Recovery scanned
    `range(i, i+bars)` — the trigger bar consumed one slot, so a 3-bar
    horizon only gave 2 post-trigger bars. Now `range(i+1, i+1+bars)`;
    unobservable horizons still yield `None`, never imputed.
18. **`bench/dip.py` — silent forecast drops.** `evaluate_forecasts`
    skipped forecasts referencing events outside the frozen universe,
    inflating scores by omission. Now raises listing the unmatched
    `(asset, trough_date)` pairs.
19. **`bench/dip.py` — normalized-key evasion of the bench honesty gate.**
    `assert_bench_output_honest` did a raw substring check on the key, so
    `p&l`, `p_n_l`-style separator injection, or `cumulativepnl` evaded.
    Keys are normalized to alnum-lowercase before the forbidden-token scan.

### Train lane / ship gate (`train/`)

20. **`train/pipeline.py` — `eval_base.json` taken on faith.**
    `run_eval_candidate` never re-verified the base summary against the
    hash the training receipt pinned at EVAL_BASE time: a tampered or stale
    `eval_base.json` silently re-baselined the ship gate. Now re-hashes the
    artifact and compares to `receipt.eval_base_sha256`, requires the base
    and candidate summaries to pin the same `eval_bank_sha256`, pairs
    results by task *name* with per-kind set equality (was positional), and
    fails closed on non-dict summaries, results without task names, and an
    empty domain task set.
21. **`train/run.py` — eval gate only checked the top flag.** A summary
    with `honesty_gate_passed: true` but nonzero per-result or
    summary-level `honesty_violations` passed the gate. The gate now
    re-derives violations from every recorded field. Manifest write is
    atomic.
22. **`train/receipts.py` — non-atomic receipt write** → `atomic_write_text`.
23. **`mrm.py` — non-dict contamination report crashed instead of
    flagging.** A report artifact containing valid JSON that isn't an
    object (`[]`) raised `AttributeError`; now flagged as contaminated.
    Dossier write is atomic.
24. **`bench/run.py` — non-atomic receipt write** → `atomic_write_text`.

### Harness / serve

25. **`harness.py` — `--config` path not contained.** `Harness.run` passed
    a caller-supplied `config` (kwarg or `--config`/`--config=` inside
    `extra_args`) straight to the subprocess — paths outside `configs/`
    were reachable. A containment check against `cwd/configs` now rejects
    escapes for both forms.
26. **`serve/backends.py` — null completion became the string `"None"`.**
    `str(payload[...]["content"])` coerced a missing/null content field
    into plausible-looking text. Non-str content now raises
    `RuntimeError`.

## Verified clean (re-checked this wave)

- **Subprocess/network surface**: `harness._subprocess_runner` is argv-only
  (no shell), registry-constrained, injectable, timeout-bounded.
  `train/receipts._git_revision` is argv-only and fails closed
  (`"unknown"`, `dirty=True`) when git is unavailable — a receipt that
  can't prove a clean tree marks itself dirty. `serve/backends` is the
  only network call (pinned Moonshot URL, 120s timeout) and lives outside
  the train lanes. `data/sources` adapters run bundled plugin CLIs with a
  timeout.
- **Tokenizer/token determinism**: `data/quality` tokenization is a fixed
  lowercase word-split used identically by dedup, near-dup, and overlap
  screening (documented in-module); `train/curriculum` stages are
  deterministic enumerations.
- **MRM staleness**: `compile_dossier` reads and cites artifact hashes at
  dossier-build time (hashes of the artifacts as compiled, not a stored
  expectation); a mutated artifact after compile invalidates the citation
  — that check is external (recompile), noted as a contract property.
- **Eval bank pinning**: `eval_bank_sha256` is baked into suite summaries
  and receipts, and now cross-checked base↔candidate (item 20).
