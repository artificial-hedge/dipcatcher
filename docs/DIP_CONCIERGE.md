# dipcatcher CLI — chat with fx1 and fx1-lite

`dipcatcher` is the chat-first command line for Artificial Hedge's **fx1**
and **fx1-lite** models ([artificialhedge.co](https://artificialhedge.co)).
Plain text is a turn with the active model. Slash commands run research,
sessions, and endpoint setup. Explicit lab commands stay available for
scripts.

This page describes the checkout that was exercised here. It is not a
release announcement. No live model was authenticated: model calls in this
environment used the loopback stub in `tests/fx1/stub_model_server.py`
(HTTP on `127.0.0.1`). The only built-in endpoint constant is Moonshot's
chat-completions URL, `https://api.moonshot.ai/v1/chat/completions`
(`fx1.serve.backends.MOONSHOT_API_URL`). An operator endpoint is whatever
URL you store or export. Nothing here is a latency, price, context-window,
or model-release claim.

> **Status (2026-10-12).** On a terminal, bare `dipcatcher` opens chat.
> With stdin or stdout redirected, bare `dipcatcher` prints a short preface
> and the lab help, and does not read a prompt. `dipcatcher chat` refuses
> a non-terminal and exits 2. `fxi` opens the same concierge and, with
> stdin closed, prints the banner and exits on EOF. `quant` is the lab CLI
> without that preface. `/superpower` loads the capability catalog and the
> read-only family surface; a family that cannot be bound with `--only` is
> refused instead of widening to the full scorecard. Selective family
> records are partial (`verified=false` on the evidence item) and are not
> sealed receipts. Sessions persist under `$FX1_SESSION_DIR`, else
> `$FX1_CONFIG_DIR/sessions`, else `~/.fx1/sessions`. A job that was still
> `running` is marked `interrupted` on resume and is not started again.
> Live authenticated smoke for both models, a five-user trial, a 24-hour
> soak, and the macOS/Windows matrix are open. See
> [the product contract](DIPCATCHER_PRODUCT_CONTRACT.md).

## Honesty contract

Research results are proper scores (pinball, CRPS, PIT, QLIKE, Brier, ECE,
Kupiec, HMM likelihood). The honesty gate withholds an answer that
headlines Sharpe, Sortino, Calmar, P&L, or NAV, that makes a
live-performance claim, or that presents synthetic evidence with a number
and no `SYNTHETIC` label. SYNTHETIC results are correctness tests, always
labeled, and are not market evidence. There is no broker connectivity and
no live-trading claim. Receipts are immutable evidence; a verification
claim needs a receipt the harness checked. The lexical gate does not prove
that a cited receipt is the right one for the sentence.

A loopback reply of `The strategy Sharpe 2.1 beat the book.` came back as
exec status `withheld`, exit code 3, with this answer:

```text
answer withheld by the fx-1 honesty gate (fx-1 output headlines forbidden research metric token 'sharpe'; research results are proper scores, not Sharpe/P&L headlines.). Ask a follow-up to continue.
```

## Install and first run

From a checkout, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen --all-groups --all-extras
```

The project entry points are `dipcatcher` (`dipcatcher_cli.app:main`),
`quant` (`quant_fund.cli.main:app`), `fxi` (`fx1.interactive.app:main`),
and `fx1`. In this environment the console scripts were not on `PATH`; the
commands below were run as `python -m dipcatcher_cli` and
`python -m fx1.interactive.app` with `PYTHONPATH=src`. Set
`FX1_CONFIG_DIR` when you do not want state under `~/.fx1`.

```sh
dipcatcher                  # terminal: chat. redirected: preface + lab help
dipcatcher chat --model fx1-lite
dipcatcher version
dipcatcher exec "ping"      # one turn; exit 5 until an endpoint exists
dipcatcher models list
dipcatcher models check     # offline; does not open a socket
```

`dipcatcher version` prints `dipcatcher` and the package version.
`fxi --version` prints `fxi` and the same version.

Without an endpoint the console still accepts slash commands. A model turn
says no endpoint is configured and does not invent a reply. Observed exec
JSON (`status` `no_endpoint`, exit 5):

```text
no model endpoint configured — set one with `/keys set fx1`, or export MOONSHOT_API_KEY / FX1_BASE_URL. Slash commands still work.
```

## Entry points

| Command | What happened here |
|---|---|
| `dipcatcher` | Terminal: chat. Not a terminal: preface, then lab `--help`. Does not read stdin. |
| `dipcatcher chat [--model fx1\|fx1-lite] [--resume]` | Chat. Requires a terminal on stdin and stdout. `--resume` reloads the newest session and does not restart an interrupted job. |
| `dipcatcher exec ...` | One turn, no prompts. Plain mode prints the answer on stdout and the footer on stderr. `--json` prints one `dipcatcher.exec.v1` document. |
| `dipcatcher models list [--json]` | Presence table for both models. Not a liveness claim. |
| `dipcatcher models check [model] [--live] [--json] [--out PATH] [--timeout-s N]` | Offline configuration check, or one call per model with `--live`. `--out` is atomic mode `0600`. |
| `dipcatcher sessions list\|show\|export\|rm\|prune` | Saved chats. JSON schema `dipcatcher.sessions.v1`. |
| `dipcatcher resume [session] [--model fx1\|fx1-lite]` | Opens chat on a session. Requires a terminal. `last` or a unique prefix. |
| `dipcatcher version` | Package version. Extra arguments exit 2: `usage: dipcatcher version`. |
| `dipcatcher <lab-command>` | Passed to the lab CLI. Examples: `capabilities`, `families`, `doctor`, `research`. |
| `quant` | Lab CLI only. No chat preface. |
| `fxi` | Same concierge. `fxi shell` is the older one-line shell. `fxi setup` is the endpoint wizard. |

The preface printed by redirected `dipcatcher` is:

```text
Start a conversation: dipcatcher (interactive terminal)
Choose a model: dipcatcher chat --model fx1-lite
Script one turn: dipcatcher exec "..."  (exit codes: dipcatcher exec --help)
Check models: dipcatcher models list | dipcatcher models check --live
```

## Configuring both models

Each model has its own key and completion URL. Precedence, checked
independently, is:

1. Stored profile in `$FX1_CONFIG_DIR/credentials.json` (default `~/.fx1/credentials.json`).
2. Per-model environment: `DIPCATCHER_FX1_API_KEY` and `DIPCATCHER_FX1_BASE_URL`, or `DIPCATCHER_FX1_LITE_API_KEY` and `DIPCATCHER_FX1_LITE_BASE_URL`.
3. Shared environment: `MOONSHOT_API_KEY` with `FX1_BASE_URL`, otherwise the Moonshot constant above.

A per-model key with no per-model URL uses `FX1_BASE_URL` when that is
set, otherwise the Moonshot constant. A per-model URL without a per-model
key does not replace the shared URL. A stored profile does not fall
through when its URL is invalid (`invalid-url`). Resolve does not write
`os.environ`.

Listings are presence-only. A stored key's fingerprint is the last four
characters (`...wxyz`). An env-backed row in `dipcatcher models list`
shows fingerprint `-` and source `env:model` or `env:shared`. The chat
banner prints the resolved fingerprint (`endpoint source env:model
fingerprint ...aaaa` against the stub). An unset row still shows host
`api.moonshot.ai` with status `unset`. That host is the constant above,
not a configured endpoint.

Inside chat:

```text
/keys list
/keys set fx1
/keys set fx1-lite
/keys remove fx1
```

`/keys set` prompts `› fx1 API key:` with echo off, then `› fx1 endpoint
URL [<default>]:`. Enter confirms the stored completion route. Observed
after storing `http://127.0.0.1:9`:

```text
POST http://127.0.0.1:9/v1/chat/completions stored for 'fx1' in <FX1_CONFIG_DIR>/credentials.json (mode 0600)
```

The file mode observed on Linux was `0600`. The config directory was
`0700`. A symlink at the store or its directory is refused. On Windows
the store is restricted with `icacls` for the current user; that path was
not executed here. A failed ACL is reported by store health as
`Windows ACL was not applied to the credentials store`.

`fxi keys set` is the same store from outside chat. `--value` exists and
lands in shell history; the prompt does not.

`dipcatcher exec` refuses `/keys set` because it would prompt
(`exec refuses /keys set because it would prompt for a secret`).

### Endpoint text

Pasted URLs are normalized before they are stored. A missing path or `/`
becomes prefix `/v1` and the route `/chat/completions` is added. A path
that already ends in `/chat/completions` is kept. Scheme and host are
lower-cased. Observed:

| Input | Result |
|---|---|
| `https://api.moonshot.ai` | `https://api.moonshot.ai/v1/chat/completions` (route added) |
| `https://api.moonshot.ai/v1` | same URL (route added) |
| `https://API.Moonshot.AI/v1/chat/completions` | same URL (route already present) |
| `http://127.0.0.1:9` | `http://127.0.0.1:9/v1/chat/completions` |
| `http://127.0.0.1:9/custom/chat/completions` | prefix `/custom`, route kept |
| `http://example.test` | `must use HTTPS unless the host is a literal loopback address` |
| `http://localhost:9` | same rejection (`localhost` is a name, not a literal loopback address) |
| `https://user:pass@example.test/v1` | `must not embed userinfo credentials` |
| `https://example.test/v1?x=1` | `must not carry params, query, or fragment` |

Bearer credentials may be sent over HTTPS, or over HTTP to a literal
loopback address. Plain HTTP to any other host is rejected before a
request is built. Percent-encoding, backslashes, dot-segments, and
non-ASCII hostnames are rejected. Error text does not echo the URL you
typed.

## The chat

On a terminal the prompt is the active model and a `›`, for example
`fx1 › `. The banner with `NO_COLOR=1` was plain text:

```text
dipcatcher · fx1 @ unconfigured · unset
fx1 @ unconfigured
No endpoint configured. Run /keys set fx1 to add your API key and URL.
/help, /superpower, /exit
```

A successful stub turn then printed the answer and a footer. The footer
keeps the requested model id. When the served id differs, a warning
follows:

```text
hello from stub
· fx1 · 0.00s · 8 tokens
```

```text
· fx1 · 0.00s · 6 tokens · served someone-else
warning: served model someone-else differs from requested model fx1
```

`dipcatcher exec` puts that footer on stderr in plain mode, and in the
JSON `warnings` array. The answer is the only stdout line.

The honesty gate runs before the answer is shown. A withheld turn stays
in the session so you can ask a follow-up. Completed tool calls from that
turn are kept; a blocked answer does not undo them.

While the prompt is visible, transcript lines are written above it and
the typed line is redrawn. On an interactive terminal a background job
uses one status line. `NO_COLOR` (any value) turns color off, including
when `FORCE_COLOR` is also set, and disables animated orbs. The static
banner still printed in the scripted console. Bracketed paste
(`\x1b[?2004h`) was still enabled on the TTY engine. A slash error from
the non-engine console is printed by typer and can still contain a red
SGR sequence.

Ctrl-C at an idle prompt prints `press Ctrl-C again to exit, or /exit`. A
second Ctrl-C within the window exits. That pair was observed on a
pseudo-terminal. The console's code path requests cancel instead of
exiting when a background job is running; that branch was not exercised
with a live job here. Ctrl-D before any line exits (closed stdin on `fxi`
printed the banner and exited). Ctrl-D after a
multiline block has started cancels that block (`InputCancelled`) and
stays in the session. A trailing odd number of backslashes continues the
line (`hello \` then `world` became `hello \nworld`). A line that is
exactly three backticks, or three double-quotes, opens a fence closed by the same line.

History is `$FX1_CONFIG_DIR/history`, a JSON list, mode `0600`. After
`/model` on the TTY it contained `["/model"]`. Secrets from `/keys set`
are not stored in history.

On Windows the console uses blocking `input()` because `select` is not
used on console handles. Queued background progress appears after you
submit a line. The notice string is `On Windows, background progress
updates after you submit a line.` Windows was not executed here. Treat
that console as a compatibility floor.

`/use fx1-lite` keeps the conversation. Observed:
`active model 'fx1-lite'; conversation kept`.

## Slash commands

`/help` prints this list. One line each:

| Command | Meaning |
|---|---|
| plain text | One turn with the active model (flash, web, harness tools). |
| `/help` | Print the command list. |
| `/use <fx1\|fx1-lite>` | Switch model and keep the conversation. |
| `/model` | Print `active model: <id>`. |
| `/models` | Both models: status, host, source, fingerprint. |
| `/superpower <goal> [--sync]` | Plan and run capabilities. `--sync` runs inline. |
| `/superpower help` | List capabilities the planner may select. |
| `/research <goal> [--budget-s N] [--max-sources N] [--queries a\|b] [--sync]` | Background web research. A live search was not run here. |
| `/status` | Job state. `no background job` when idle. |
| `/cancel` | Cancel the background job. `no running background job` when idle. |
| `/redirect <goal>` | Retarget background research. Empty goal: `usage: /redirect <new goal>`. |
| `/evidence [E1]` | Ledger items, or one item. Idle: `no receipt paths on the last job`. |
| `/budget` | Run ceilings and the chat usage line. Not a dollar cap. |
| `/budget max-steps N` | Also `max-seconds`, `max-families`, `max-harness`, `max-retries`. Observed `budget max_steps=3`. |
| `/session` | Id, model, job state, turn count. |
| `/sessions [N]` | Newest sessions. |
| `/resume [id]` | Prefix, `last`, or id. A job is not restarted. |
| `/export [md\|json]` | Redacted transcript. Observed a mode-`0600` markdown file under `exports/`. |
| `/new` | Fresh session, same model. Refuses while a job is running. |
| `/flash add\|list\|show\|remove\|correct\|search\|refresh` | Local research memory. Empty store: `(no flash entries)`. |
| `/harness list\|describe\|run` | The 32 registered lab commands. |
| `/keys list\|set\|remove` | Endpoints, presence-only. |
| `/approvals` | Recorded decisions. Idle: `no approvals recorded`. |
| `/approvals revoke [key]` | Drop one grant, or all of them. |
| `/diagnostics [--bundle PATH]` | Local report. The bundle is a mode-`0600` zip and is not uploaded. |
| `/exit` | Leave. |

An unknown slash command prints `unknown command /<name>; try /help`.

`/budget` with no arguments printed:

```text
budget max_steps=12 max_seconds=3600 max_families=4 max_harness=4 max_retries=1; no price catalog is configured; this budget is time/step limits only, not a dollar cap
wall_s 0/180 (not started); steps 0/6; model_calls 0/4; retries 0 (max 1 per step); tokens 0/24000 (tokens only; unit price is unknown so no spend cap is implied); 0 calls reported no usage
```

The first line is the run ceiling. The second is the chat-turn usage
ceiling. Token counts are not a price.

A turn that crosses a usage ceiling says `turn stopped by usage ceiling.`
and then that usage line. Status is `budget`.

## Approvals

Consequential harness commands (`train`, `optimize`, `paper`) and
private-data sharing ask before they run. On the console the prompt is:

```text
approval required — <question> [y]es once / [a]lways this session / [N]o:
```

`y` or `yes` is once. `a` or `always` is this session, and only for the
same action key and the same bounds. Anything else, EOF, or Ctrl-C
denies. A denial is recorded and not retried.

Non-interactive callers (exec, pipes, scripts) do not prompt. The default
is deny. `dipcatcher exec --approve harness:research` allows that key for
one decision. The kind `consequential` is never allowed by kind alone.
An empty `--approve` exits 2: `empty approval key`.

`/approvals` lists what was decided. `/approvals revoke` clears session
grants.

## /superpower

`/superpower help` prints the planner catalog (harness commands plus
`flash_retrieve`, `flash_save`, `web_research`, and `synthesize`).
`train`, `optimize`, and `paper` are marked `approval-required`.

A qualified run does more than that keyword list:

1. It loads `dipcatcher capabilities list`. In this tree that command
   printed 190 rows: 11 `qualified` / `exec yes`, 179 `unqualified` /
   `exec no`. Unqualified rows are shown and are not run. The live list
   is the authority; the counts will change when the catalog changes.
2. Only capabilities that are executable and registered in
   `fx1.harness.HARNESS_REGISTRY` become harness steps. Arguments are
   typed. A step missing a required argument is excluded. Observed:
   `excluded score-forecasts: invalid_args: missing_required:--data`.
3. `dipcatcher families` is the read-only family surface. For the goal
   `qlike volatility forecast` it selected family `volatility` with
   runner `research --only volatility` and score `qlike`. It does not
   execute benches.
4. When a research step cannot bind `--only`, the full scorecard is
   refused. Observed: `excluded research: catalog research contract has
   no --only argument; refusing the full scorecard`. The same goal
   produced a typed `research-only` step with `--only: ['volatility']`
   and `evidence_class` `SYNTHETIC`. Families that are not eligible are
   not replaced by "run everything".
5. Each attempt gets a directory `$FX1_CONFIG_DIR/runs/<run_id>/`, mode
   `0700` on POSIX. The id looks like `r-20261012T073017Z-934caa7f`.
   Datasets are not invented for the plan.
6. The evidence ledger is harness-authored. `/evidence` prints it.
   Classes observed by recording items and by verifying one committed
   receipt locally:

| Class | What was recorded |
|---|---|
| Verified receipt | `verdict=VERIFIED`, detail `sealed receipt verified`. The local file `receipts/coherence_2dd641ab766a536a.json` verified as schema `receipt.v2` with data label `SYNTHETIC`. That label means a correctness fixture, not market evidence. |
| Partial family record | `add_artifact` stored kind `RECEIPT`, verdict `UNVERIFIABLE`, detail `partial family record — not a sealed research receipt; verified=false`. `research --only` output that is not a sealed receipt is this class. `verified=false` is not upgraded because a flag says otherwise. |
| Heuristic web | Kind `WEB`, verdict `UNVERIFIABLE`, detail `HEURISTIC WEB VERIFICATION — not a sealed receipt`. |
| Flash | Kind `FLASH`, verdict `UNVERIFIABLE`, even when the entry's verified flag is set: `flash verified_flag set; not a sealed receipt`. |
| Missing receipt | Verdict `MISSING`, detail `receipt file not found`. |
| Model interpretation | Header `Model interpretation (not independently verified):`, or the cited-evidence header when a verified receipt is actually on the ledger. A withheld interpretation uses `Model interpretation withheld by the honesty gate:`. |

Web, flash, and model prose are never `VERIFIED`. A verified receipt mixed
with anything else is a partial answer so the unverified items do not
inherit the receipt.

Background is the default. `--sync` runs inline, which is what a pipe
wants. `/cancel` prints `[<label>] cancel requested` and
`cancellation pending` until the job ledger confirms `cancelled`.
`/redirect` applies to background web research.

This documentation pass planned goals and did not execute benches, so no
new receipt was produced by `/superpower`.

## Sessions and resume

`dipcatcher sessions list` prints `id`, `model`, `status`, `turns`,
`updated`. `show` adds job state and a turn table. Message bodies are
omitted unless `show --messages`. `export --out` writes markdown or JSON,
mode `0600`, and refuses to overwrite. Exit codes: 0 ok, 1 runtime, 2
usage, unknown or ambiguous session, or refusal.

`dipcatcher chat --resume` and `dipcatcher resume` reload a session.
Observed after a record whose job state was `running`: the stored job
became `state=interrupted`, `resume=manual`, and the notice was
`resumed session <id> on model fx1; the previous job was not restarted`.
Completed steps were left in place. Nothing was launched.

`/resume` inside chat uses the same rule. `/new` starts a new id and
keeps the model.

The job ledger directory is `$FX1_CONFIG_DIR/jobs`. Reconcile on an empty
ledger returned no interrupted, orphan, or corrupt ids. Startup text for
a non-empty ledger names each interrupted job and says it was not
restarted. A process that cannot be shown to be ours is reported, not
killed, unless a killer was injected.

## exec

```sh
dipcatcher exec --help
dipcatcher exec --json --model fx1-lite "ping"
dipcatcher exec --timeout-s 300 --session new "ping"
dipcatcher exec --approve harness:research "run the registered research command"
dipcatcher exec --output-last-message answer.txt "ping"
```

`--model` is `fx1` (default) or `fx1-lite`. `--timeout-s` defaults to 300
and covers the whole turn, including a background job. `--session` is
`none` (default, ephemeral), `new`, `last`, or a 32-hex session id.
`--output-last-message` writes the answer atomically, mode `0600`.
Prompt `-` reads stdin when stdin is not a terminal. A terminal with no
prompt exits 2: `exec needs a prompt; stdin is a terminal so it will not
block`.

| Exit | Status | Meaning |
|---|---|---|
| 0 | `ok` | Answer printed. |
| 1 | `error` | Runtime, model, or tool error. |
| 2 | | Usage or configuration. Unknown model: `unknown model 'gpt'; choose from ['fx1', 'fx1-lite']`. |
| 3 | `withheld` | Honesty gate. |
| 4 | `timeout` or `budget` | Deadline or usage ceiling. |
| 5 | `no_endpoint` | No key for the selected model. |
| 130 | `cancelled` or SIGINT | Interrupted. |
| 143 | SIGTERM | Terminated. |

A stub `pong` with usage exited 0. Stdout was `pong`. Stderr was
`· fx1 · 0.00s · 4 tokens`. JSON for fx1-lite included `served_model`,
`usage`, and `latency_s`. Those timings are the stub on loopback, not a
hosted-service measurement.

## Telemetry and /diagnostics

Telemetry is a local JSONL file under `$FX1_CONFIG_DIR/logs/<session-id>.jsonl`.
Nothing is uploaded. Prompt and answer text are not fields: a log written
during a stub turn that replied `pong` did not contain `pong` or the API
key. Events observed included `console.start` and `concierge.turn` with
`model`, `phase`, `outcome`, and token counts.

`DIPCATCHER_LOG` selects `off`, `error`, `info`, or `debug`. The default
is `info`. An unknown value falls back to `info` and is not echoed.

`/diagnostics` prints a presence-only report: version, paths, env var
names, a harness-registry digest, and the honesty token list. In this
tree it reported `harness: commands: 32`. The credentials permission slot
is redacted because the name matches the secret filter (`credentials:
[REDACTED]`). Sessions, jobs, logs, and history show `ok` or `absent`.
`--bundle PATH` writes a zip, mode `0600`, and refuses to overwrite. The
zip is not uploaded.

## Security posture

These are design properties covered by tests. They are not a guarantee
that a future change still holds, and they were not a penetration test.

| Property | Where it is specified | Tests |
|---|---|---|
| Tool and page text is wrapped in an untrusted-content envelope (`<<<UNTRUSTED_DATA` … `<<<END_UNTRUSTED_DATA`) so the body cannot look like the end of the envelope. | `fx1.interactive.policy` | `tests/fx1/test_policy.py`, `tests/fx1/test_policy_live.py`, `tests/fx1/test_concierge_security.py` |
| Owned lab children get a scrubbed environment. `*_API_KEY`, `MOONSHOT_*`, and `FX1_BASE_URL` are not copied. `DIPCATCHER_CHILD=1` and `PYTHONUNBUFFERED=1` are set. Argv summaries passed to telemetry are redacted. | `fx1.interactive.procs.scrub_env` | `tests/fx1/test_harness.py` |
| Approvals are deny, once, or session. A session grant does not widen bounds. Non-interactive policy denies `consequential` unless the key is listed. | `fx1.interactive.approvals` | `tests/fx1/test_approvals_scoped.py` |
| Diagnostics and session exports redact secrets. | `fx1.interactive.diagnostics`, `fx1.interactive.redact` | `tests/fx1/test_diagnostics.py`, `tests/fx1/test_concierge_diagnostics.py` |

`scrub_env` was also run in this environment: `MOONSHOT_API_KEY` and
`DIPCATCHER_FX1_API_KEY` were absent from the child map. A parent
`OMP_NUM_THREADS` was kept; the thread pins default to `1` only when the
parent did not set them.

The classic `fxi shell` `use` command still injects `MOONSHOT_API_KEY` and
`FX1_BASE_URL` into the shell process, by that module's own contract.
Owned concierge harness runs use the scrubbed map instead. A shell child
was not launched here.

Web research is a separate engine (`fx1.webresearch`): http(s) only,
credentials in URLs refused, private and loopback targets refused
including after redirects, with budgets and a cancel flag. Reports are
heuristic web verification, not receipts. Those rules are what
`tests/fx1/test_webresearch.py` locks. A live search was not repeated
for this page. Keyless HTML search fails closed (an error and zero hits)
when the provider markup changes. There is no JavaScript renderer.

## Troubleshooting

The host and status numbers below are the shapes that were returned. The
loopback stub produced the HTTP examples. The next step is the sentence
the error already contains.

| Kind | Text | Next step in that text |
|---|---|---|
| no endpoint | no model endpoint configured — set one with `/keys set fx1`, or export `MOONSHOT_API_KEY` / `FX1_BASE_URL`. Slash commands still work. Exit 5. | `/keys set <model>` or the shared env vars. |
| `auth` | `fx1 @ <host>: HTTP 401 (invalid_api_key) — The API key was rejected; run /keys set fx1 with a key for this model.` Exec prefixes `error:` and exits 1. | `/keys set <model>` with a key for that model. |
| `unknown_model` | `HTTP 404 (model_not_found) — The server does not recognize model fx1; confirm the model name and endpoint. Next: /models` | `/models`, then the model id and the endpoint. |
| `route_not_found` | `HTTP 404 (not_found) — The completion route looks wrong; expected a URL ending in /v1/chat/completions.` | Store a URL that normalizes to that route. |
| `rate_limited` | `HTTP 429 (rate_limit_exceeded) — The server is rate-limiting requests; wait before trying again.` | Wait, then retry. |
| `quota` | `HTTP 402 (insufficient_quota) — The account has no remaining quota or billing is blocked; retrying will not succeed.` | Do not retry the same key. Fix the account outside the CLI. |
| `timeout` | `timed out (TimeoutError) — The model server did not answer before the timeout; retry.` | Retry. Exec's overall deadline is `--timeout-s` (exit 4, status `timeout`). |
| `network` | `connection failed (ConnectionRefusedError) — Could not reach <host>; check the host and the network path.` | Check the host and the path. |
| `tls` | `tls verification failed (SSLError) — TLS verification failed for <host>; check the certificate, the system clock, and any intercepting proxy.` | Certificate, clock, or intercepting proxy. |
| `malformed` | `malformed completion (not json) — The server response was not a valid chat completion and was discarded.` | The body was discarded. |
| identity mismatch | `dipcatcher models check --live` status `failed`, `error_kind` `identity_mismatch`, hint `served model 'not-fx1' did not match requested 'fx1'`. Exit 1. The record note says `this record is not a release claim`. | Point the model at an endpoint that serves the requested id. |
| budget | `turn stopped by usage ceiling.` plus the usage line that says unit price is unknown. Exec exit 4, status `budget`. | The ceiling is steps, calls, tokens, or wall time. It is not a dollar cap. `/budget` shows and sets the run fields. |
| honesty withheld | The Sharpe sentence in the honesty section. Exit 3, status `withheld`. | Ask a follow-up. Do not treat the withheld text as a result. |
| chat without a terminal | `chat requires an interactive terminal on stdin and stdout; use explicit lab commands or fxi one-shot commands for scripts` | `dipcatcher exec`, or a real terminal. |
| bad endpoint paste | The normalization errors in the table above. The URL is not stored. | Paste an https URL, or literal-loopback http, with no userinfo and no query. |

`dipcatcher models check` without `--live` prints `offline configuration
check; nothing was contacted` and exits 5 when either model is unset.
With `--live` against the stub it contacted the socket once and exited 0
when the served id matched.

## Lab commands the model may run

`fx1.harness.HARNESS_REGISTRY` has 32 commands in four roles
(verification 6, data engine 4, evaluation 19, model training 3).
`/harness list` prints them. `/diagnostics` reported `commands: 32`.
Anything not in that registry is not a concierge harness command. The
lab CLI that `dipcatcher <name>` forwards to is larger; `capabilities
list` shows those extra names as unqualified and not executable.

`dipcatcher capabilities list` and `dipcatcher families` are read-only.
`families --help` states that the command does not execute benches.

## Tests that lock this surface

Offline, with network seams injected:

- `tests/fx1/test_concierge.py` — slash dispatch, tool loop, background jobs.
- `tests/fx1/test_superpower.py` — planner, approvals, honesty withholding.
- `tests/fx1/test_flash.py` — flash memory.
- `tests/fx1/test_webresearch.py` — fetch policy and web budgets.
- `tests/fx1/test_approvals_scoped.py` — once / session / deny.
- `tests/fx1/test_docs_drift.py` — `fx1` commands, `make` targets, and `fx1` imports cited from `README.md` and `docs/FX1*.md` exist.

This page's command transcript is from a local run, not from that suite.

## Known limitations

- **Search provider.** Keyless HTML search breaks if the provider changes
  markup. The failure mode is an explicit error and zero hits. No paid
  search API is wired. A live search was not run for this revision of the
  page.
- **No JavaScript rendering.** Pages that only paint in a browser yield
  little text.
- **Heuristic claims.** Sentence extraction and token overlap are lexical.
  Mirrors of one article can count as several sources. Contested detection
  catches explicit negation, not a subtle numeric disagreement.
- **Retrieval** is hashed bag-of-words. A paraphrase with no shared words
  misses.
- **Ceilings are not prices.** Step, time, and token ceilings are enforced.
  There is no price catalog, so they are not a dollar cap. Unmetered calls
  (no `usage` object) stay unmetered rather than getting a made-up total.
- **Background jobs have a durable ledger** under `$FX1_CONFIG_DIR/jobs`
  and a per-run directory under `$FX1_CONFIG_DIR/runs/<run_id>/`. A
  restart marks a running job interrupted and does not start it again.
  The session record and flash entries from before the restart remain.
  The worker thread itself does not survive the process.
- **Windows console** is a compatibility floor: blocking input, progress
  between submitted lines, credential ACL via `icacls`. It was not run
  here.
- **Hosted model turns were not verified live.** Chat, `/models check
  --live`, and exec were checked against the loopback stub only. No
  Moonshot or operator endpoint was authenticated. Offline scripted
  backends still cover the wire client in the test suite.
- **Still open:** a 24-hour soak, a five-user trial, a real authenticated
  smoke for both fx1 and fx1-lite, and the macOS/Windows matrix. Those
  are not implied by the local stub runs above.
