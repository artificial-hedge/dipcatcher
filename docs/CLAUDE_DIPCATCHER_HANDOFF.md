# Claude handoff — production dipcatcher

**2026-10-12 · Second slice delivered: durable operation, budgets, family gate,
endpoint guidance, console hardening.** The three work packages below
("next quarter" / "final quarter") were the plan at 2026-10-08; the table
immediately below records what the second slice closed and what remains.

| Area | Delivered 2026-10-12 | Files |
|---|---|---|
| Durable sessions | Append-only session journal (schema v1, 0600) with turns, model switches, job transitions, approvals, redaction; `/sessions`, `/resume`, `dipcatcher chat --resume`, `fxi sessions|resume`; interrupted-job reconciliation at startup (nothing re-run). | `src/fx1/interactive/session.py`, `concierge.py`, `tests/fx1/test_session.py`, `tests/fx1/test_concierge_production.py` |
| Cancellation + deadlines | `Harness.run(cancel_event=, timeout_s=)`; default runner drains output on threads, escalates SIGTERM → SIGKILL on cancel/deadline, caps output, fails closed without the executable; `/superpower` forwards the cancel event and remaining wall budget; cancelled/timed-out steps are reported as such. | `src/fx1/harness.py`, `superpower.py`, `tests/fx1/test_harness.py` |
| Budgets | `RunBudget` (env `FX1_BUDGET_*`) + thread-safe `BudgetLedger`; `/budget`; budget-exceeded steps and partial-run verdicts; deterministic summary when a model call is not affordable. | `src/fx1/interactive/budget.py`, `tests/fx1/test_budget.py` |
| Family eligibility | `dipcatcher research-families` (read-only JSON) + `fx1.interactive.families` (aliases, `family:` tokens, retired/unknown refusal, unavailable-catalog fail-closed); planner excludes `research` for retired-only goals and the model cannot reinstate it; `/families`. | `src/quant_fund/cli/families_cmds.py`, `src/fx1/interactive/families.py`, `tests/fx1/test_families.py`, `tests/unit/cli/test_research_families_cmd.py` |
| Endpoint behaviour | Failure classes with next actions (auth/not-found/rate-limited/server/timeout/unreachable/malformed/unconfigured/bad-url), bounded retries for transient classes only, `HostedK3Backend.identity_probe` + `/keys test` (requested vs served identity, latency). | `src/fx1/interactive/endpoint.py`, `src/fx1/serve/backends.py`, `tests/fx1/test_endpoint.py` |
| Console | `LineReader` (POSIX select / Windows msvcrt / blocking), `InterruptPolicy` (Ctrl-C once cancels, twice exits), secret-filtered `History`; `/use` continuity with an attribution note; `/doctor`; untrusted framing of web content; `--version`. | `src/fx1/interactive/lineio.py`, `concierge.py`, `src/dipcatcher_cli/app.py`, `tests/fx1/test_lineio.py` |
| Evidence classes | Every capability run names its evidence class (verified-research-artifact, synthetic-correctness, harness-output-unverified, heuristic-web, memory, model-interpretation, deterministic-summary); `result: complete|partial`. | `superpower.py`, `tests/fx1/test_superpower.py` |

**Still open after this slice** (see the product contract gates): the
`research` bench executes its configured scorecard (no per-family execution
constraint inside `quant_fund.research.agent`); typed dataset/config arguments
and data-availability checks; real authenticated smoke runs for both models
(P2); the frozen 100-journey suite and five-user trial (P1/P3); automatic safe
resume of idempotent steps; multiline editing/paste/resize; the 24-hour soak
(P6); signed release artifacts and the platform matrix (P7). The repository's
McCabe ratchet (tracked as deliberately red in `day_grind_progress.md`) and
`mypy --strict src/fx1` (45 errors in `serve/*_audit.py` and `train/tiny_lm.py`
at the base revision) have pre-existing failures outside the dipcatcher files;
this slice adds none and removes two McCabe baseline pins.

---

**2026-10-08 · First implementation slice delivered; remaining work below.**

The user requested 25% of the actual work. This slice delivers the usable chat
entry point and core response/planning guards. The product is not certified
production-ready; the remaining three work packages are family orchestration,
resilience/UX, and production validation. The split is a work allocation, not a
measured readiness percentage.

## Implemented

| Area | Delivered behavior | Files |
|---|---|---|
| Primary CLI | Bare `dipcatcher` opens chat when stdin/stdout are terminals. `dipcatcher chat --model fx1-lite` selects the starting model. Bare redirected invocation prints help without consuming stdin. Explicit lab commands keep their output and exit codes. | `src/dipcatcher_cli/`, `pyproject.toml`, `tests/unit/cli/test_dipcatcher_launcher.py` |
| Compatibility | `quant` keeps the independent lab entry point; `fxi` stays compatible. Registered harness subprocesses pass through the new launcher without recursion. Neither model nor research library imports the launcher. | Same launcher; existing harness/dependency boundaries retained |
| In-chat setup | `/keys set [model]` prompts for a masked key and endpoint URL. Invalid/cancelled input preserves the previous profile. Startup shows the configured host or `unconfigured`, with setup guidance. | `src/fx1/interactive/concierge.py`, `tests/fx1/test_concierge.py` |
| Ordinary chat honesty | Model answers and tool preambles pass the existing honesty validator before output/history/tool execution. Refused prose is withheld; completed tool-call/results stay in context so a follow-up knows what already ran. | Same concierge files |
| Planner bounds | Model refinement can select another registered capability, but honors `max_harness`, removes duplicate IDs, and retains retrieval-first/synthesis-last ordering. Malformed replies fall back. Execution also deduplicates steps; consequential approvals remain enforced. | `src/fx1/interactive/superpower.py`, `tests/fx1/test_superpower.py` |
| Basic terminal behavior | Idle POSIX polls keep one prompt; `NO_COLOR` is respected. Windows uses blocking input rather than unsupported `select` on console stdin, with a notice that progress appears between submitted lines. | Same concierge files |
| Product definition | Chat-first requirements, research-family execution contract, and P1–P7 release criteria. CLI guides describe the implemented working-tree behavior. | [Product contract](DIPCATCHER_PRODUCT_CONTRACT.md), [CLI guide](DIP_CONCIERGE.md), [FXI guide](FXI.md) |

These changes are in the working tree, not committed or published. Preserve
other agents' concurrent changes. `uv lock` was run after changing project
metadata; the lockfile content did not change.

## Try this slice

After reinstalling the package from this checkout:

```sh
uv sync --frozen --all-groups --all-extras
dipcatcher
```

Inside chat, `/keys set fx1` accepts the service-supplied completion URL and key.
Use `/keys set fx1-lite` for the second model, `/use fx1-lite` to switch,
`/superpower help` to inspect command capabilities, and `/exit` to leave.
A direct starting-model selection is `dipcatcher chat --model fx1-lite`.
No production endpoint hostname or model availability is invented here.

## Verification and limits

- **Passed:** final combined run: **169 tests in 85.82 seconds**, covering the
  launcher, dependency boundaries, interactive console, concierge, planner,
  honesty validator, and honesty inheritance. Individual focused runs included
  13 launcher tests, 45 concierge tests, and 37 planner tests.
- **Passed:** touched-file Ruff/format checks; strict launcher types with
  `--follow-imports=skip`; focused concierge/planner mypy checks.
- **Passed:** actual `uv sync --frozen --all-groups --all-extras` in an isolated
  environment, wheel build, and installation of that wheel.
- **Passed from outside the checkout:** installed module location, redirected
  help, explicit chat pipe refusal, lab help/error exit behavior, `quant`/`fxi`
  compatibility, and registered `Harness.run('research', extra_args=['--help'])`
  through the installed `dipcatcher` executable.
- **Passed in a real macOS PTY using that wheel:** idle prompt remains singular;
  setup for both models uses masked fixture keys and stores mode-0600 profiles;
  model switching, `/help`, `/superpower help`, `NO_COLOR`, and clean exit work.
  This was offline orchestration testing, not a hosted-model evaluation.
- **Not green:** `UV_NO_SYNC=1 make lint` reported undefined
  `_assert_valuation_not_stale` and `_decision_inputs` in the concurrently edited
  `src/quant_fund/backtest/engine.py`. This slice did not edit that module.
- **Not green:** the existing direct-module startup suite had 3 timing failures:
  `quant_fund.cli.main --help` 4.107 s (budget 1.5), `fx1.cli --help` 1.799 s
  (budget 1.0), `fx1.cli doctor` 3.743 s (budget 1.0). Other checks in that run
  passed. Investigate/rebaseline the execution environment with evidence;
  do not relax the budgets to hide failures.
- A broad import-graph mypy run was interrupted; the scoped launcher check above
  passed. Full lab/fx1 release gates, live endpoints, Linux/Windows terminals,
  usability trials, soak tests, and signed epoch integration are still owed.

Validation logs/wheel/PTY transcript for this session are under
`/var/folders/m5/xfql4h014511zfhn74l9rfcw0000gn/T/dipcatcher-cli-validation-2xgpmdt1/`.
These temporary files are diagnostic artifacts, not immutable research receipts.

## Claude — next quarter: qualified research orchestration

Start here. `/superpower` still selects coarse registered commands, not research
families or typed dataset/config arguments. The bounded planner fix does not
solve this. Expose qualified family metadata through a registered read-only
harness interface; avoid importing research internals into `fx1`. Make selection
constrain the benches actually executed, not just the final report.

Implement stable family/version IDs, scientific relevance, required input and
point-in-time checks, typed runner arguments, dependencies, qualification
references, resource limits, and receipt verification. Unknown, retired,
unqualified, or unavailable families must fail closed; a missing qualification
audit is not permission. Preserve verification of historical retired-family
receipts. Use [Benchmark family lifecycle](BENCHMARK_FAMILY_LIFECYCLE.md).

Deliver one vertical case: supplied dataset → relevant qualified family → real
runner → independently verified immutable receipt → grounded answer. Extend to
the frozen relevance suite in the product contract. The current honesty check
is lexical; it does not verify a claim or the receipt it names.

## Claude — next quarter: durable operation and complete chat UX

Add session/job persistence, safe restart/resume, active subprocess cancellation,
retry/idempotency semantics, and hard budgets through planning and synthesis.
Keep completed actions in context on all error paths; ordinary backend errors
still use the older rollback behavior and need the same side-effect audit.
The current Windows fallback is a compatibility floor, not a responsive terminal
implementation. Finish multiline editing, paste, history, resize, progress
rendering, and platform accessibility. Model switching still clears chat context;
define and test the intended continuity behavior.

Run both models against their real service contract, check identity and routing,
normalize endpoint semantics, audit secret isolation and Windows credential
storage, and preserve scoped authorization. Reconcile remaining README/root
instructions and historical design text against the implemented behavior.

## Claude — final quarter: production evidence and release

Close P1–P7 in [the product contract](DIPCATCHER_PRODUCT_CONTRACT.md), including
99/100 frozen chat journeys and all critical paths. Fix the reported repository
gate failures with their owning lanes. Recheck dated findings in
[ULTRA_PROD_READINESS.md](ULTRA_PROD_READINESS.md); its prose is not fresh CI
proof. Keep harness readiness, each model's release, and live-trading eligibility
as separate decisions. No benchmark count or fixture establishes state of the art.

Useful integration tests:

```sh
PYTHONPATH=src uv run --no-sync pytest tests/unit/cli/test_dipcatcher_launcher.py tests/unit/test_fx1_dependency_edge.py tests/fx1/test_interactive.py tests/fx1/test_concierge.py tests/fx1/test_superpower.py tests/fx1/test_honesty.py tests/fx1/test_honesty_inheritance.py -n 0
```

Before committing, follow all applicable `AGENTS.md` gates, including full lab
and fx1 checks; passing the focused command does not replace them. Cross-cutting
launcher work follows the PR convention. Preserve immutable receipts and the
`quant_fund`/`fx1` dependency guards.

Source/docs are covered by epoch integrity. Coordinate the append-only
`make stamp-epochs` → `make sign-pins` → `make anchor-pins` → `make checkpoint`
sequence from [Operations](OPERATIONS_RUNBOOK.md) with the integration owner.
This slice did not update signed evidence or rewrite research receipts.
