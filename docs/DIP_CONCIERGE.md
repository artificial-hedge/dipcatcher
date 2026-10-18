# dipcatcher CLI — chat-first orchestration for fx1 and fx1-lite

dipcatcher is Artificial Hedge's state-of-the-art harness for orchestrating
its **fx1** and **fx1-lite** models ([artificialhedge.co](https://artificialhedge.co))
from the command line. The CLI is designed like Codex CLI: you open it, you
talk to it, and it does the work. Plain text is a conversation with the
active model; everything else is a slash command. `/superpower` is how the
conversation reaches dipcatcher's research families — it picks the
capabilities that fit your goal, runs them, and explains what it used and
why.

> **Status (2026-10-12).** The packaged `dipcatcher` launcher opens chat when
> stdin and stdout are terminals (`dipcatcher chat --model fx1-lite` chooses
> the starting model, `--resume <id>` replays a prior session, `--version`
> prints the distribution version); redirected bare invocation prints help,
> and explicit lab commands retain their behavior. Sessions are durable
> (`$FX1_CONFIG_DIR/sessions/*.jsonl`): every turn, model switch, job
> transition and approval is journaled, and jobs left open by a dead process
> are marked *interrupted* at the next start. `/superpower` runs under hard
> budgets (wall clock, harness steps, model calls, retries, web sources),
> propagates cancellation and deadlines into owned lab subprocesses
> (SIGTERM → SIGKILL), labels every result with its evidence class, and
> checks research-family eligibility against the lab's read-only catalog —
> retired or unknown families fail closed. Model turns retry only transient
> failures and otherwise explain the failure class with a next action;
> `/keys test` records requested vs served model identity. Ctrl-C once
> cancels the running job, twice exits. What remains open is listed under
> [Known limitations](#known-limitations-honest-gaps) and in
> [the product contract](DIPCATCHER_PRODUCT_CONTRACT.md): the `research`
> bench still executes its configured scorecard (family selection narrows
> eligibility and reporting, not the benches run), live hosted-model
> verification needs a working endpoint, and the frozen journey suite /
> moderated trial have not been run. Nothing here is proof of live model
> availability.

## Design: Codex-style, chat-first

About 99% of the interface is the conversation. The rules:

| Principle | What it means here |
|---|---|
| One command, one conversation | Bare `dipcatcher` opens the chat. There is no subcommand tree to learn first. |
| Plain text is a turn | Ask in your own words; the active model answers and calls tools (memory, web, harness) as needed. |
| Slash commands for control | `/superpower`, `/research`, `/use`, `/status`, `/cancel` — discoverable with `/help`, never required to get started. |
| Approval before consequence | Consequential lab work (`train`, `optimize`, `paper`) asks first; a refusal is recorded, not retried. |
| Long work runs in the background | Jobs stream progress while you keep typing; `/cancel` and `/redirect` take effect at the next step. |
| Honest by default | Ordinary chat and `/superpower` synthesis pass the existing honesty validator before display. This lexical check does not establish semantic correctness or verify a cited receipt. |
| Subcommands are the other 1% | `dipcatcher research --config …` and the rest stay for scripts, CI, and receipts — the non-interactive path. |

New operator-facing features belong in the conversation (a slash command or
a model tool in `src/fx1/interactive`), not as new top-level subcommands.

**Boundary rule.** The chat lives on the fx1 side (`src/fx1/interactive`),
and `quant_fund` never imports `fx1` (enforced by
`tests/unit/test_fx1_dependency_edge.py` and `configs/arch_boundaries.toml`).
The outer launcher is `src/dipcatcher_cli`, outside both libraries. The chat
reaches the lab exclusively through the registered
harness commands in `fx1.harness.HARNESS_REGISTRY`, executed as
subprocesses — the model can never touch an unregistered lab surface.

## Quickstart

```sh
uv sync --frozen --all-groups --all-extras
dipcatcher              # opens chat in an interactive terminal
# Inside chat: /keys set fx1 prompts for a masked API key and endpoint URL.
# Use the completion URL supplied by your service; no endpoint is assumed live.
# Inside chat: /help, /superpower help, /use fx1-lite, /exit
# Or select the starting model explicitly:
dipcatcher chat --model fx1-lite
```

Without an endpoint the console still works: slash commands run, web research
runs, flash context runs — only model turns (chat replies, plan refinement,
synthesis) report honestly that no endpoint is configured and fall back to a
deterministic summary.

## The session

Plain text is a chat turn with the active model (`fx1` or `fx1-lite`, switch
with `/use`). The model gets four tools, executed locally with the lab's
fail-closed posture:

| Tool | What it does | Gate |
|---|---|---|
| `flash_retrieve` | ranks persistent research memory against the query | none (local read) |
| `web_lookup` | one bounded search + up to two page reads | fetch policy |
| `flash_save` | persists a finding with task/uncertainty | none (local write) |
| `harness_run` | runs one registered `dipcatcher` command | approval when consequential |

Slash commands:

```text
/superpower <goal> [--sync]     plan + coordinate research capabilities
/superpower help                list the capability catalog (+ family catalog once loaded)
/families [goal]                research-family catalog; classify the families a goal names
/research <goal> [--budget-s N] [--max-sources N] [--queries a|b] [--sync]
/status                         background job state, elapsed time, budget ledger
/cancel                         stop the background job; owned subprocesses are signalled
/redirect <new goal>            re-target the running research
/budget [dim value]             show or tighten run budgets (wall|harness|calls|retries|sources|tokens)
/flash add|list|show|remove|correct|search|refresh ...
/sessions [n]                   recent sessions from the journal; /resume <id> replays one
/keys list|set|test|remove      model endpoints (presence-only); test = identity probe
/doctor                         readiness: endpoint, store, journal, budget, family catalog
/harness list|describe|run      registered lab commands
/use <model>  /model  /help  /exit
```

Background jobs run on their own thread; the console polls stdin between
lines, so progress renders while you keep typing, and `/cancel` /
`/redirect` take effect at the job's next step boundary — a lab subprocess
owned by the job is sent SIGTERM (its whole process group) and SIGKILL five
seconds later if it ignores that. On piped stdin the loop degrades to
blocking reads (jobs still stream progress).

On Windows the console polls `msvcrt` instead of `select`, so background
progress renders while you type; a build without `msvcrt` falls back to
blocking reads. Input history lives in `$FX1_CONFIG_DIR/history` (mode 0600;
`/keys` lines and anything that looks like a credential are never stored).

**Ctrl-C** follows the Codex convention: the first press cancels the running
job (or just clears the line when idle) and prints a hint; a second press
within two seconds exits. Ctrl-D and `/exit` leave immediately; the session
journal is closed and any still-running job is cancelled and joined (a job
that does not stop within five seconds is reported, not hidden).

### Sessions and recovery

Every session appends to `$FX1_CONFIG_DIR/sessions/<id>.jsonl` (schema
version 1, directory 0700, file 0600): a header (model, endpoint host, code
revision, fx1 version) and then turns, model switches, job start/step/finish
records, approvals, endpoint changes, cancel requests, and `session_end`.
Secrets are redacted before they reach the file. On startup the console
reconciles earlier files: a job with a start but no finish and no session end
is marked `interrupted` — nothing is re-run — and the affected session ids are
printed. `/sessions` lists recent sessions; `/resume <id>` prints a read-only
transcript (also `dipcatcher chat --resume <id>`, `fxi sessions`,
`fxi resume <id>`). A journal line is a mutable job record, not an immutable
research receipt.

### Budgets

Each `/superpower` run gets a `BudgetLedger` from the session's `RunBudget`
(defaults: 1800 s wall, 4 harness steps, 8 model calls, 2 retries per call,
8 web sources; prompt tokens uncapped because provider prices are unknown —
an unknown price never implies a spend cap). Environment defaults come from
`FX1_BUDGET_WALL_S`, `FX1_BUDGET_HARNESS_STEPS`, `FX1_BUDGET_MODEL_CALLS`,
`FX1_BUDGET_RETRIES`, `FX1_BUDGET_WEB_SOURCES`, `FX1_BUDGET_PROMPT_TOKENS`;
`/budget harness 2` tightens one dimension for the session. A step that would
exceed the budget is recorded `budget-exceeded` and the run says *partial*;
the remaining wall budget is passed to each harness subprocess as its
deadline (it can only tighten the registry timeout).

### Model switching

`/use fx1-lite` keeps the conversation. A system note is appended so the new
model attributes earlier assistant turns to the previous model, the switch is
journaled, and the prompt shows the new model and endpoint host. Switching to
an unconfigured model is allowed — slash commands keep working and model
turns say what is missing — so endpoint setup can happen in-chat afterwards.

### Endpoint failures

Model turns go through a bounded retry (`RetryPolicy`: two retries with
capped exponential backoff) that only retries transient classes — rate limit
(408/429), server (5xx), timeout, unreachable. Authentication (401/403), a
wrong route (404), a rejected request (400/422 — the model id may be unknown
to the service), a malformed payload, or a bad URL are reported once with a
failure class and a next action, never retried, and never echo the URL or
key. `/keys test [model]` sends one minimal authenticated completion and
renders requested vs served model id, latency, and the failure class on
error; a served identity that differs from the requested id is flagged
(provider routing), recorded, and not treated as success of anything else.

## /superpower — capability orchestration

`/superpower <goal>` answers "which of dipcatcher's research capabilities
should run for this goal, and why?" The catalog is every registered harness
command — `research` (the scientific benches that make up the research
families; see [RESEARCH_CENTRE.md](RESEARCH_CENTRE.md)), plus `validate`,
`forecast`, `backtest`, `northset`, `kyle-ofi`, and the other lab surfaces —
together with web research, flash retrieve/save, and synthesis.
`/superpower help` prints the live list. It runs in four phases
(`fx1.interactive.superpower`):

1. **Plan.** The deterministic planner scores every capability against the
   goal (trigger-term catalog, data not code) and selects the relevant
   subset — never a blind run-everything. With a model endpoint available,
   the hosted model may refine the selection, but its reply is validated
   against the registry: unknown ids are dropped and any malformed or failed
   model pass falls back to the deterministic plan. Both paths apply
   `max_harness=4`; refinement deduplicates IDs and preserves retrieval first
   and synthesis last. The model may select another registered command that
   the keyword pass missed, within that same cap, but it is never offered —
   and cannot reinstate — a capability the family gate blocked.
   **Family gate.** When the plan would run `research`, the lab's read-only
   family catalog is loaded through the registered `research-families`
   command (cached per session). Families named in the goal — by name,
   alias (`tail risk` → `tail`, `CRPS` → `distribution`, …) or explicit
   `family:<name>` token — are classified *required*, *optional*, *retired*
   or *unknown*. A goal that names only retired or unknown families cannot
   select `research`: the plan prints `excluded: research … no suitable
   family`. An unavailable catalog never blocks, but the plan says that no
   family-level claim will be made. Because `dipcatcher research` still
   executes its configured scorecard, the plan also says that selection
   narrows eligibility and reporting, not the benches executed.
2. **Recall.** Flash context is retrieved first (cheap, local) so prior
   findings feed everything downstream.
3. **Execute.** Web research (budgeted, cancellable) runs before harness
   commands; its content enters the synthesis context under an explicit
   `UNTRUSTED WEB CONTENT` marker. Each harness step is admitted in order:
   not cancelled, wall budget left, step budget left, approval granted when
   consequential (`train`, `optimize`, `paper` — recorded `approval-denied`
   when refused). The subprocess receives the cancel event and the remaining
   wall budget as its deadline; a cancelled step is recorded `cancelled`, a
   deadline hit `failed` ("deadline reached"). Harness output keeps its
   `DATA_LABEL=SYNTHETIC` honesty markers into the synthesis context.
4. **Synthesize.** The final answer is composed from capability results only
   (sources and receipt paths cited), then passes
   `fx1.honesty.validate_fx1_output`: forbidden headline metrics
   (Sharpe/Sortino/Calmar/P&L/NAV), live-performance claims, or unlabeled
   synthetic evidence cause the answer to be **withheld** with the gate's
   reason — violating text is never emitted. Without an endpoint (or when
   the budget forbids another model call) a deterministic summary is
   emitted and labelled as such. The run itself is saved to flash context
   with its goal and capability list.

The plan is printed with a per-capability rationale, the family verdicts and
the budget before anything runs. Every result carries an explain block
(`capabilities used and why`) in which each capability names its **evidence
class** — `verified-receipt` (a verifier such as `verify-research`/`doctor`/
`validate` exited 0), `synthetic-correctness` (output labelled SYNTHETIC),
`harness-output-unverified`, `heuristic-web`, `memory`,
`model-interpretation`, or `deterministic-summary` — followed by a
`result: complete|partial — N/M capabilities ran` verdict. A run with any
failed, skipped, cancelled or budget-exceeded step says *partial*; the
synthesis prompt is told the same, and unverified prose never inherits
verified status.

## Deep web research

`fx1.webresearch` is a stdlib-only research engine — no API keys, no new
dependencies:

- **Search** (`fx1.webresearch.search`): keyless DuckDuckGo (lite endpoint,
  html fallback). Failures report an error and yield zero hits; hits are
  never fabricated. Queries derive from the user's goal text only — private
  flash context never enters a search query.
- **Fetch** (`fx1.webresearch.fetch`): http/https only; credentials in URLs
  refused; private/loopback/link-local/reserved addresses refused **including
  on redirects** (no SSRF lane into your network); size caps with explicit
  truncation; redirect caps; robots.txt honored (conservative subset,
  per-origin cache); HTML reduced to title/text/links with script, style, and
  site-chrome filtering.
- **Iterate** (`fx1.webresearch.engine`): seed queries include a
  deliberately adversarial angle ("criticism OR contradictions") so
  conflicting evidence is searched for, not avoided. The loop searches,
  ranks hits by goal overlap with domain diversity, fetches, extracts
  goal-relevant sentences as claims, and follows ranked in-page references
  within the depth budget.
- **Verify**: claims are clustered across sources by token overlap; clusters
  report `corroborated` (≥2 sources), `single-source`, or `contested`
  (negation or antonym polarity detected across different sources, listed as
  explicit CONFLICT pairs).

Every report is labeled **HEURISTIC WEB VERIFICATION** — cross-source checks
of public pages, never a sealed dipcatcher receipt.

Budgets are hard and checked mid-loop: wall-clock (`--budget-s`), source
count (`--max-sources`), searches, per-source bytes, reference depth, and an
inter-search pause. Cancellation is cooperative and checked between network
steps; partial results are kept and the report is stamped `(cancelled)`.

## Flash context — persistent research memory

`fx1.flash` stores findings as append-only JSONL under
`$FX1_CONFIG_DIR/flash/entries.jsonl` (default `~/.fx1/flash/`), shared by
every `fxi` process — memory survives sessions.

Each entry (`fx1.flash.store.FlashEntry`) carries: text, task, tags, sources
(URL/title/accessed-at), created/updated timestamps, uncertainty
(`low|medium|high`), conflicting evidence, verified and private flags, an
optional staleness window, refresh state, and usage counts.

- **Retrieve** (`fx1.flash.retrieve`): deterministic hashed-vector cosine
  over text/tags/task, tempered by recency (30-day half-life) and staleness;
  every hit explains itself (`term-match`, `tag-match`, `stale`, `private`,
  ...). Entries below the relevance threshold are left behind instead of
  filling the model's context.
- **Correct/remove**: corrections append a revision and tombstone the old
  line — history is never rewritten; removal tombstones. Torn trailing lines
  are reported via `read_errors()`, never silently dropped.
- **Refresh** (`fx1.flash.refresh`): re-fetches recorded sources and stamps
  reachability honestly; unreachable sources are named in the refresh note so
  retrieval can demote stale findings.
- **Private**: entries flagged `--private` render with a do-not-share marker
  in prompt context; the approval gate covers private-data sharing.

```text
/flash add "Fed held rates in Sept 2026; dot plot shows one cut" \
    --task rates --tags macro,fed --source https://... --uncertainty medium
/flash search fed rate path
/flash correct <id> --uncertainty high --conflict "two members dissented"
/flash refresh <id>
/flash remove <id>
```

## Approvals and safety model

- Approvals default to **deny** whenever no interactive operator can answer
  (piped stdin, scripts, tests); the console prompts `[y/N]` and anything but
  an explicit yes is a refusal. Every decision is recorded
  (`fx1.interactive.approvals.ApprovalGate`).
- Consequential actions: harness `MODEL_TRAINING` commands (`train`,
  `optimize`, `paper`) in chat tools and in `/superpower` plans.
- Private-data sharing: private flash entries are marked in prompt context;
  web-search queries are built from goal text only.
- Credentials: presence-only reporting (`...wxyz` fingerprints), keys travel
  as environment values, never argv; store file mode 0600.

## One-shot mode (scripting)

```sh
dipcatcher superpower "assess conformal coverage regressions" --sync
dipcatcher superpower "train the ranking family" --sync --approve-consequential
dipcatcher sessions --limit 5
dipcatcher resume <session-id>
dipcatcher keys test fx1-lite
dipcatcher flash search "ecb rates"
# fxi remains compatible for the same one-shots:
fxi superpower "assess conformal coverage regressions" --sync
fxi research "ECB rate path expectations" --budget-s 120 --max-sources 5 --sync
fxi flash add "finding" --task t --tags a,b --uncertainty low
fxi flash refresh <id>
dipcatcher --version
dipcatcher research-families --text   # the family catalog the planner consults
```

`--sync` runs inline (no background thread), which is what CI and pipes want.
Non-interactive runs never wait for an approval: consequential commands are
denied unless `--approve-consequential` states the policy for that run.

## Tests and verification

Offline suites (network seams injected; run with `make fx1-test` or
`uv run pytest tests/fx1/test_flash.py tests/fx1/test_webresearch.py
tests/fx1/test_superpower.py tests/fx1/test_concierge.py
tests/fx1/test_concierge_production.py tests/fx1/test_interactive.py
tests/fx1/test_session.py tests/fx1/test_budget.py tests/fx1/test_endpoint.py
tests/fx1/test_families.py tests/fx1/test_lineio.py tests/fx1/test_harness.py`):

- `tests/fx1/test_session.py` — journal header/schema, redaction, torn
  lines, listing, prefix lookup, interrupted-job reconciliation
  (idempotent), transcript truncation, thread safety, disk-failure
  tolerance.
- `tests/fx1/test_budget.py` / `tests/fx1/test_harness.py` — budget
  validation, env parsing, ledger boundaries and thread safety; real
  subprocess cancellation (SIGTERM → SIGKILL on a SIGTERM-ignoring child),
  deadlines, output caps, legacy runner compatibility, missing-executable
  fail-closed.
- `tests/fx1/test_endpoint.py` — failure-class table, secret-free summaries,
  retry/backoff/cancel semantics, identity probe rendering and wire shape.
- `tests/fx1/test_families.py` — catalog parsing (fail-closed), alias and
  `family:` resolution, retired/unknown refusal, unavailable-catalog
  behaviour, prompt block.
- `tests/fx1/test_lineio.py` — Ctrl-C policy, history filters and modes,
  POSIX/Windows/blocking readers.
- `tests/fx1/test_concierge_production.py` — the integrated behaviour:
  journaled turns/jobs/approvals, `/use` continuity, `/budget`, `/status`,
  `/doctor`, `/keys test`, retries with guidance, untrusted web framing,
  family-aware planning (refusal, blocked refinement), evidence classes,
  console startup reconciliation, Ctrl-C cancel/exit, history without
  secrets.

- `tests/fx1/test_flash.py` — CRUD, tombstones, torn-line reporting, threaded
  appends, relevance ranking, staleness, privacy flags, context blocks,
  refresh, cross-process persistence.
- `tests/fx1/test_webresearch.py` — URL/SSRF policy, redirect re-checks,
  size caps, robots subset + caching, DDG parsing against the live HTML
  shape, claim extraction filters, contradiction detection, budgets
  (time/search/source), cancel with partial results, redirect, fragment
  dedupe, IRI encoding.
- `tests/fx1/test_superpower.py` — registry/harness parity, consequential
  set, planner selection + caps, model-refinement validation and fallbacks,
  coordination order, approval denial, honesty-gate withholding, offline
  summary, cancel semantics.
- `tests/fx1/test_concierge.py` — slash dispatch, chat tool loop (flash
  retrieve/save, harness approval + registry refusal, backend fault
  rollback), background research lifecycle (status/cancel/redirect/error),
  superpower sync/background, flash commands, console driver over scripted
  stdin.

Live verification (performed against the real internet during development):
`fxi research --sync` fetched real sources (DDG lite), extracted real claims
with URLs, and deduped fragments; `fxi superpower --sync` planned, ran live
web research, saved to flash, and produced the offline synthesis; a separate
`fxi flash list/search` process retrieved the saved entry — cross-session
memory confirmed; a full piped REPL session (help → flash add/search/list →
plain chat → exit) ran end-to-end, with the no-endpoint chat turn answering
honestly instead of fabricating a model reply. Live *model* turns were not
verifiable with the stored credentials — see Known limitations.

## Known limitations (honest gaps)

- **Search provider**: keyless DDG HTML parsing breaks if DDG changes markup;
  the failure mode is an explicit error plus zero hits, never fabricated
  results. No paid-SERP integration yet.
- **No JS rendering**: SPA-only pages yield little text.
- **Heuristic claims**: sentence-level extraction and token-overlap grouping
  are lexical, not semantic; mirrors/syndication of one article count as
  multiple corroborating sources (fragment URLs are deduped, distinct hosts
  are not).
- **Family execution constraint**: the catalog gate decides which families
  may be *selected and claimed*; `dipcatcher research` still executes its
  configured scorecard. Constraining the benches actually executed needs a
  family filter inside `quant_fund.research.agent.run_research` — not done.
  `research-families` lists every retired name. A name in that set
  classifies as retired (new selection blocked; historical receipts still
  verify). A name in none of required, live optional, or retired classifies
  as unknown. Both refuse. A payload that omits `retired` still classifies
  only its 20-name sample, and unknown stays fail-closed.
- **Cost budgets**: wall clock, harness steps, model calls, retries and web
  sources are hard ceilings; a prompt-token cap applies only when
  `FX1_BUDGET_PROMPT_TOKENS` is set, and provider prices are unknown, so no
  spend cap in currency is claimed.
- **Background jobs** live in-process; they do not survive a console restart.
  The journal records them, and the next start marks them *interrupted*
  (flash entries and reports written before the restart persist); nothing is
  resumed automatically.
- **Retrieval** is hashed bag-of-words — no embeddings, so paraphrased queries
  with no shared vocabulary miss.
- **Contested detection** catches explicit negation/antonymy only; subtle
  numeric disagreement is shown as separate claims, not a conflict.
- **Hosted model turns**: chat replies, model plan refinement, and synthesis
  need a working endpoint. At the time of writing the operator's stored
  endpoint (`inference.artificialhedge.co`) rejected its stored key
  (`invalid_api_key`) and the stored base URL lacks the
  `/v1/chat/completions` path, so live model turns were **not** verifiable —
  they are covered offline by scripted-backend tests on the repo's own
  `HostedK3Backend` wire client. Fix with `fxi keys set fx1` (full chat URL +
  valid key); every non-model capability (web research, flash, planning,
  coordination, approvals) was verified live end-to-end.
