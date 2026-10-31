# Claude handoff — production dipcatcher

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
