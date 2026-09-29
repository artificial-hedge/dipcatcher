# AUDIT_FX1_EVALHARNESS.md — fx-1 eval-harness + honesty machinery (round 2)

Deep line-audit of `src/fx1` eval machinery, `reward.py`, `honesty.py`,
`mrm.py`, `train/` (pipeline, run, dpo, curriculum, default_trainer),
`data/` (corpus, quality, receipts, ingest, traces, sources), `serve/`,
`bench/`, `cli.py`, and all of `tests/fx1`. Continues from
`docs/AUDIT_FX1.md` (round 1, 14 findings) — every candidate finding was
re-checked against the round-1 fixes before being counted.

## Findings fixed

F1. **Honesty gate did not cover non-honesty task kinds** (`eval/suite.py`).
    `score_task` ran `validate_fx1_output` only when `enforce_honesty`, and
    the gate only inspected `kind == "honesty"` results. A model could
    headline a forbidden metric on domain tasks (task fails but gate stays
    green) or on general tasks (never validated at all). Now every response
    is validated; violations always land on `honesty_violations`, fail the
    task only when enforced, and ALWAYS close the suite gate. Summary
    records `honesty_violations` (task names) for receipts.
F2. **Headline bypass via separator runs** (`honesty.py`). The connector
    set was `of|=|:|is|was|at|to|reads?|quotes` with `{0,2}` reps —
    "Sharpe — 2.1", "pnl - $4,200", "nav (1.9)", "sharpe…2.1",
    "sortino ratio — 3.0" all evaded. `[^\w\s]+` separator runs and a third
    rep now headline-check those forms; bare discussion still allowed.
F3. **Live-claim spelling/order variants** (`honesty.py`). Patterns were
    spaced-only and order-locked: "live-trading P&L", "live  profit",
    "real-money returns", "profits from live trading", "synthetic data
    shows live market performance" all evaded. Patterns now accept
    `[-\s]+` separators, a reversed "X from/of live (trading|markets|
    accounts)" form, and synthetic subjects beyond bare "results".
F4. **Empty response farmed the honesty_clean weight** (`reward.py`).
    `score_response("")` scored `honesty_clean` (+1.0) because an empty
    string contains no violations — a model could refuse everything and
    out-score honest-but-sparse answers. Empty/whitespace responses now
    early-return 0.0.
F5. **Degenerate receipt citation** (`reward.py`). `_RECEIPT_RE` matched
    any 8+ hex chars — "deadbeef" alone earned `cites_receipt`. Now
    requires >=16 hex (the corpus's `sha256[:16]…` form) AND provenance
    wording (receipt/sha256/digest).
F6. **Formatting-variation contamination bypass** (`data/quality.py`).
    `_shingles` split on whitespace only, so punctuation attached to
    tokens broke 8-gram containment: a corpus copy of an eval prompt with
    commas/semicolons/full-width chars inserted escaped both the quality
    gate and `contamination.ngram_containment_scan`. `_tokens` now does
    NFKC fold + `\w+` extraction; dedup and the n-gram scan share it.
F7. **Quality gate screened only the canonical bank**
    (`train/pipeline.py`). `eval_prompts or [DEFAULT_BANK user msgs]` —
    caller-supplied prompts REPLACED the bank, and red-team/rephrased/
    masked surface forms were never screened. New
    `eval.eval_prompt_surface()` (bank + red-team + rephrased twins +
    masked twins, deduped) is always screened; caller prompts only widen.
F8. **Candidate honesty gate absent** (`train/pipeline.py`).
    `run_eval_candidate` compared pass counts regardless of whether the
    candidate summary passed the honesty gate. A violating candidate
    produced a passing eval_candidate.json + comparison. Now raises
    RuntimeError before compare_runs when `honesty_gate_passed` is false.
F9. **`_validate_eval_gate` trusted the flag** (`train/run.py`).
    `{"honesty_gate_passed": true, "results": []}` unblocked training —
    the round-1 doc listed this as "documented (no change)", but it is a
    certification vector: a hand-written flag certifies honesty without
    evidence. Gate now requires the flag AND >=1 honesty-kind result AND
    all honesty results passed AND zero `honesty:`-prefixed failures.
F10. **Contamination flag was last-write-wins** (`mrm.py`).
    `contamination_flagged = bool(report.get("overall_flagged", True))`
    overwrote per artifact — a clean report seen after a flagged one
    cleared the flag. Now OR-accumulated across all artifacts, and
    `ship_eligible` requires `not contamination_flagged` — the dossier no
    longer certifies a model the card says ships while contamination is
    flagged.
F11. **`contamination-audit` vacuous pass on missing/empty corpus**
    (`cli.py`). `texts = []` when the corpus was missing →
    `overall_flagged=False` → exit 0: "not contaminated" certification of
    nothing. Missing corpus or zero examples now exit 2. Audit prompts
    switched from DEFAULT_BANK-only to `eval_prompt_surface()`.
F12. **Hosted eval unpinned sampling** (`serve/backends.py`).
    `HostedK3Backend.complete` sent no `temperature` — server-default
    sampling makes eval/teacher outputs unreproducible across replays.
    Pinned to 0.0.

## Reviewed, documented (no change)

- `FORBIDDEN_HEADLINE_TOKENS` / `FORBIDDEN_RESEARCH_METRIC_KEYS` mirror:
  `test_honesty_inheritance` checks exact equality AND blocks fx1-only
  extras; both sets are plain lowercase literals, so normalization cannot
  diverge them. No drift found.
- Determinism elsewhere: `DEFAULT_BANK`/`score_task`/`compare_runs` carry
  no RNG; `quality.dedup_and_filter` seeds `random.Random(17)`;
  `mask_task`/`redteam` are pure; `timepart` partitions are fixed dates.
- Receipts/artifact loaders (`receipts`, `artifacts`, `modelcard`,
  `dpo`) raise on malformed/missing — fail-closed as required.
- `verify_release` tolerating extra files, `_RECEIPT_RE`-style weak
  provenance on other surfaces: kept from round 1 (attestation sidecars
  are intentional); F5 tightened the reward-path copy where it actually
  paid out.

## Environment note

`mypy src/fx1` in this venv reports one preexisting error on
`origin/main` as well — `quant_fund/models/covariance.py:118` unused
`type: ignore` under a numba-present environment (reproduced on a
pristine `origin/main` worktree). Not introduced by this change; fx1
modules are strict-clean.
