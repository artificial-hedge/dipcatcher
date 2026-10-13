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
> sealed receipts. `/data` attaches a CSV or Parquet that already sits
> under `<cwd>/data`. `/runs` reads the job ledger and does not start
> work. `/models check` is offline unless `--live` is set. Sessions persist
> under `$FX1_SESSION_DIR`, else `$FX1_CONFIG_DIR/sessions`, else
> `~/.fx1/sessions`, including dataset attachments. A job that was still
> `running` is marked `interrupted` on resume and is not started again.
> The chat process treats SIGHUP as exit 129 and SIGTERM as exit 143.
> `dipcatcher exec --help` maps SIGINT and SIGHUP to 130 and SIGTERM to
> 143. Two automated journey suites, a relevance suite, chaos tests, and
> red-team tests exist. They do not close gates P1–P7. Live authenticated
> smoke for both models, a five-user trial, a 24-hour soak, the
> macOS/Windows matrix, and signed release artifacts are open. See
> [the product contract](DIPCATCHER_PRODUCT_CONTRACT.md) and
> [the release evidence table](DIPCATCHER_RELEASE_EVIDENCE.md).

## Honesty contract

Research results are proper scores (pinball, CRPS, PIT, QLIKE, Brier, ECE,
Kupiec, HMM likelihood). The honesty gate withholds an answer that
headlines Sharpe, Sortino, Calmar, P&L, or NAV, that makes a
live-performance claim, or that presents synthetic evidence with a number
and no `SYNTHETIC` label. SYNTHETIC results are correctness tests, always
labeled, and are not market evidence. There is no broker connectivity and
no live-trading claim. Receipts are immutable evidence; a verification
claim needs a receipt the harness checked. The lexical gate does not prove
that a cited receipt is the right one for the sentence. A decimal glued to
the token (`sharpe3.2`, `sharpe_3.2`, `NAV1.9`, `pnl4.2`) is a headline. A
bare integer glued on (`sharpe2`) is not. Spelled-out numbers ("sharpe of
two point one"), a token glued onto a prefix ("mySharpe 2.1"), and a bare
metric name with no number stay outside the gate. A compromised endpoint
can still return false but well-formed prose that this gate does not catch.

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
dipcatcher exec --data vol.csv "ping"   # file must already be under ./data
dipcatcher models list
dipcatcher models check     # offline; does not open a socket
dipcatcher models check --live   # one identity call per configured model
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
| `dipcatcher exec ...` | One turn, no prompts. Plain mode prints the answer on stdout and the footer on stderr. `--json` prints one `dipcatcher.exec.v1` document and, in the run below, also wrote the footer on stderr. `--data` is repeatable and uses the same resolver as `/data`. |
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
JSON `warnings` array. In plain mode the answer is the only stdout line.
A `--json` run in this pass also wrote the footer line on stderr.

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
`/model` on the TTY it contained `["/model"]`. A line whose first token is
`/keys`, and secret-shaped text, are not stored. Secrets from `/keys set`
are not stored in history.

With `PYTHONIOENCODING=ascii` the banner was
`dipcatcher | fx1 @ unconfigured | unset` and the prompt was `fx1 > `.
The same run printed `warning: credentials directory permissions were
tightened to 0700`. Unicode stdout keeps the `·` banner and the `›`
prompt. While a background job is active on a POSIX tty, the line reader
is cbreak (`tty_guard.read_cbreak`), so that line has no readline history
and no arrow keys. This pass did not leave a half-typed line on screen.
`guard_tty` restores termios and turns bracketed paste off
(`\x1b[?2004l`) on SIGTERM and SIGQUIT. Model text, tool output, dataset
names, and exports go through `sanitize_for_terminal`. OSC, CSI other than
SGR, C1, bidi, and zero-width format characters are removed. With color
on, SGR keeps color and intensity; conceal, blink, and reverse are
dropped. Those paths are what the terminal and red-team tests lock. A
hostile sequence was not pasted into the live chat for this page.

On Windows the console uses blocking `input()` because `select` is not
used on console handles. Queued background progress appears after you
submit a line. The notice string is `On Windows, background progress
updates after you submit a line.` Windows was not executed here. Treat
that console as a compatibility floor.

`/use fx1-lite` keeps the conversation. Observed:
`active model 'fx1-lite'; conversation kept`.

### Long conversations

The saved session always keeps every message. What is sent to the model is
fitted to a character ceiling (`FX1_CONTEXT_MAX_CHARS`, default 400000,
floor 4000) by dropping the oldest whole turns first. A turn is a user
message with the assistant and tool messages that follow it, so a tool call
is never separated from its result, and the newest turn is always sent. When
turns are dropped the system prompt says that the earliest N messages are
absent, and chat prints one `context:` line per trimming episode. No context
size is assumed for either model: if a server answers that a prompt is too
long, the ceiling for that model drops to half of what was just sent and the
call is retried (up to three times, never when only one turn remains). The
lowered ceiling lasts for the process. This is a client-side bound, not a
claim about either model's real window.

## Slash commands

`/help` prints `HELP_TEXT`. The wording below is that text.

| | |
|---|---|
| **Chat** | |
| plain text | talk to the active model (flash, web, harness) |
| `/use <model>` | switch fx1 \| fx1-lite; keep this conversation |
| `/model` | show the active model id |
| `/models` | status table; check [--live] [fx1\|fx1-lite] |
| **Research** | |
| `/superpower <goal>` | plan and run capabilities (--sync runs inline; help lists them) |
| `/runs [id]` | persisted jobs, newest first |
| `/research <goal>` | background web research |
| `/status` | job state and evidence status |
| `/cancel` | cancel the background job |
| `/redirect <goal>` | retarget background research |
| `/evidence [E1]` | ledger items, or one item's detail |
| `/budget` | ceilings; tokens are not a spend cap |
| **Session** | |
| `/session` | id, model, job state, and turn count |
| `/sessions [N]` | newest sessions |
| `/resume [id]` | prefix, last, or id; a job is not restarted |
| `/export [md\|json]` | write a redacted transcript |
| `/new` | fresh session; keep the active model |
| **Lab** | |
| `/data ...` | attach \<csv-or-parquet\> \| list \| remove \<n\> \| inspect \<n\> |
| `/flash ...` | add\|list\|show\|remove\|correct\|search\|refresh |
| `/harness ...` | list\|describe\|run registered commands |
| `/keys ...` | list\|set\|remove endpoints (presence-only) |
| `/approvals` | decisions; /approvals revoke [key] |
| `/diagnostics` | local report; --bundle PATH writes a zip |
| `/help` | this help |
| `/exit` | leave |

`/superpower help` lists capabilities the planner may select. A live web
search was not run for `/research`. Idle `/status` is `no background job`.
Idle `/cancel` is `no running background job`. An empty `/redirect` is
`usage: /redirect <new goal>`. Idle `/evidence` is `no receipt paths on the
last job`. `/budget max-steps N` also accepts `max-seconds`,
`max-families`, `max-harness`, and `max-retries`. Observed
`budget max_steps=3`. An export wrote a mode-`0600` markdown file under
`exports/`. `/new` refuses while a job is running. An empty flash store
prints `(no flash entries)`. `/harness` is the 32 registered lab commands.
Idle `/approvals` is `no approvals recorded`. `/approvals revoke` drops one
grant, or all of them. The diagnostics bundle is a mode-`0600` zip and is
not uploaded.

An unknown slash command with no close match prints
`unknown command /nope; try /help` (observed). A close match adds a
suggestion. Observed: `unknown command /superpwer; did you mean /superpower? try /help`.
The cutoff is `difflib.get_close_matches` at 0.6.

`/budget` with no arguments printed:

```text
budget max_steps=12 max_seconds=3600 max_families=4 max_harness=4 max_retries=1; no price catalog is configured; this budget is time/step limits only, not a dollar cap
wall_s 0/180 (not started); steps 0/6; model_calls 0/4; retries 0 (max 1 per step); tokens 0/24000 (tokens only; unit price is unknown so no spend cap is implied); 0 calls reported no usage
```

The first line is the run ceiling. The second is the chat-turn usage
ceiling. Token counts are not a price.

A turn that crosses a usage ceiling says `turn stopped by usage ceiling.`
and then that usage line. Status is `budget`.

## /data

A dataset is a regular `.csv` or `.parquet` file inside `<cwd>/data`.
Copy the file there. The command does not fetch one and does not invent a
column. At most 8 are kept (`dataset limit is 8; /data remove <n> first`);
this pass did not fill that cap. A bare name is resolved from the cwd and
then from `data/`. A path that leaves `data/`, including a symlink that
does, is refused. The sha256 is recorded at attach and checked again
immediately before a run. A changed file is not scored.

`score-forecasts --inspect` supplies a profile when it exits 0 and the
digest is unchanged. Without that profile the file is still attached, and
the line is `dataset profile unavailable; attached without a profile`.
Profile text is untrusted: it is redacted and passed through
`sanitize_for_terminal` before display. The issue-time column is bound
only when the profile's `available_time` role has confidence `exact_name`
and that column is in the file header. Otherwise the answer says
point-in-time validity was not verified.

Attachments are stored on the session as `record['attachments']` (the
profile fields) and `record['datasets']` (path, kind, label for rows that
are not stale). `/resume` restores them. `/new` clears them. After a file
change, `datasets` on the saved record was an empty list and
`attachments` kept the row with `stale` true.

Captured on a pseudo-terminal, `NO_COLOR=1`, loopback stub, work directory
`/tmp/hdocs-b2rgcmxk/work`. `data/vol.csv` had a header
`timestamp,realized,forecast_var` and two data rows.

```text
/data attach vol.csv
attached dataset /tmp/hdocs-b2rgcmxk/work/data/vol.csv
profile rows=2 columns=timestamp, realized, forecast_var
warning: point-in-time validity will NOT be verified
/data list
1 csv vol.csv rows=2 sha=a79d3ca5 /tmp/hdocs-b2rgcmxk/work/data/vol.csv
/data inspect 1
profile rows=2 columns=timestamp, realized, forecast_var
warning: point-in-time validity will NOT be verified
/data attach outside.csv
dataset path must stay inside data/
/data attach escape.csv
dataset path must stay inside data/
```

`outside.csv` was in the cwd, not under `data/`. `data/escape.csv` was a
symlink to a file outside `data/`. After the CSV gained one row,
`/superpower score qlike --sync` printed
`dataset changed since attach; /data inspect 1 then re-attach` and did not
start a bench. `/data list` then showed the same row with `stale`. `/new`
printed `new session on model fx1`. The next `/data list` was
`no datasets attached`.

A second session attached the same shape of file, exited, and
`dipcatcher chat --resume` printed
`resumed session 49b96544a22c4b91b4c1eee9028c5f24 on model fx1`. `/data list`
showed the file again, not stale. That record had one `attachments` row
and one `datasets` row (`kind` `csv`, `label` `file`). `dipcatcher exec
--data vol.csv --json ping` from that directory exited 0 with status `ok`
and answer `stub fx1 reply`. `dipcatcher exec --data missing.csv ping`
exited 2 and printed `dataset file was not found under data/`. Those
timings are the stub. They are not a hosted-model measurement. The profile
on this file did not bind an issue-time column (`has_available_time`
false).

## Approvals

Consequential harness commands (`train`, `optimize`, `paper`) and
private-data sharing ask before they run. On the console the prompt is:

```text
approval required — <question> [y]es once / [a]lways this session / [N]o:
```

`y` or `yes` is once. `a` or `always` is this session, and only for the
same action key and the same bounds. A session grant stores
`ActionBounds`, including the argv that will run. A later argv for the
same capability does not reuse it. Idempotent replay of a succeeded step
does not ask again. Anything else, EOF, or Ctrl-C denies. A denial is
recorded and not retried.

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
   typed. A step missing a required argument is excluded. Observed on an
   earlier pass: `excluded score-forecasts: invalid_args: missing_required:--data`.
   An attached file under `data/` supplies that argument. A goal the
   planner classifies as unsafe is refused before selection, with
   exclusion code `unsafe` (verifier tampering, a live trade, a forbidden
   headline, or a prompt-injection directive).
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
   The plan does not invent a dataset path. It uses files already attached
   with `/data` or `exec --data`. A digest that no longer matches is a
   refusal, not a run. Re-checked on this tree: `dipcatcher capabilities
   list --json` still printed 190 rows, 11 qualified and executable, 179
   unqualified and not executable.
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

The earlier pass on this page planned goals and did not execute benches.
The `/data` capture above sent one `/superpower` after the file changed;
it was refused before a bench. No new receipt was produced by
`/superpower` for this page.

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

`/resume` inside chat uses the same rule and restores dataset
attachments. `/new` starts a new id, keeps the model, and clears
attachments. `/sessions` lists through `SessionStore.list`, which
quarantines a corrupt session file and reports it. A newer schema is
reported and left in place. This pass did not plant a corrupt file.

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
dipcatcher exec --data vol.csv "ping"    # repeatable; same resolver as /data
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
| 130 | `cancelled`, SIGINT, or SIGHUP | Interrupted. `dipcatcher exec --help` says `interrupted (SIGINT or SIGHUP)`. |
| 143 | SIGTERM | Terminated. `exec --help` says `terminated (SIGTERM)`. |

The chat console is a different process. `install_terminal_stop` arms
SIGHUP as exit 129 and SIGTERM as exit 143 when those signals still have
the default disposition. Exec installs its own handlers first, so exec
keeps 130 for SIGHUP. This pass did not signal a live chat or a live exec.
A terminal-stop signal cancels the in-flight step and does not start a
replacement: `terminate_all(stop=True)` latches the process registry, and
the orchestrator does not retry while that latch is set. A plain
`terminate_all()` reaps and does not latch.

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
| On Linux the direct child is started through a one-shot wrapper that arms `PR_SET_PDEATHSIG` (`SIGKILL`) so it dies if this process is killed. The signal is not inherited by grandchildren. A descendant that has double-forked into a new session and been reparented is not visible, and the result does not claim that pid was killed. | `fx1.interactive.procs` | `tests/fx1/test_chaos_children.py`, `tests/fx1/test_chaos_kill.py` |
| SIGTERM and SIGHUP on the chat cancel the in-flight step and do not start a replacement. | `fx1.interactive.procs.install_terminal_stop`, `fx1.interactive.orchestration` | `tests/fx1/test_chaos_signals.py`, `tests/fx1/test_review_r2_stop.py` |
| `/sessions` quarantines a corrupt session file. A symlink session file is moved aside on load. | `fx1.interactive.sessions` | `tests/fx1/test_chaos_store.py` |
| Session approval for a superpower step or a chat tool is the key plus `ActionBounds.argv`. | `fx1.interactive.approvals` | `tests/fx1/test_review_r2_orchestration_approvals.py`, `tests/fx1/test_approvals_scoped.py` |
| Proxy environment values copied to a child keep scheme, host, and port. Userinfo, path, and query are dropped. | `fx1.interactive.procs._proxy_authority` | `tests/fx1/test_chaos_children.py` |
| Redaction matches secrets after NFKC fold. Combining marks and unpaired surrogates are skipped so they cannot hide a character inside a token. | `fx1.interactive.redact` | `tests/fx1/test_redteam_secrets.py`, `tests/fx1/test_redteam_chat_secrets.py` |
| TCP, a proxy `CONNECT`, and the TLS handshake share one connect budget of `min(read timeout, 15 s)`. The default read timeout stays 90 s and is restored on the socket after connect. | `fx1.interactive.model_client`, `fx1.interactive.model_dial` | `tests/fx1/test_chaos_model.py` |
| The planner refuses an unsafe goal before selection. Exclusion code `unsafe`. | `fx1.interactive.planner` | `tests/fx1/relevance/` |

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
| dataset path | `dataset path must stay inside data/` or `dataset file was not found under data/`. Exec exits 2 for a missing `--data` file. | Copy a regular CSV or Parquet into `<cwd>/data` and pass that name. A symlink that resolves outside `data/` is refused. |
| dataset changed | `dataset changed since attach; /data inspect 1 then re-attach` | The sha256 at attach no longer matches. Inspect, then attach again. The run does not start. |
| symlink in the config path | `dipcatcher models check` with `FX1_CONFIG_DIR` set to a symlink exited 2 and printed `credentials directory must not be a symlink`. `ensure_private_dir` on a directory under that link raised `refusing to follow symlink directory` plus the path that was requested. | Point `FX1_CONFIG_DIR` and `FX1_SESSION_DIR` at the resolved real directory. This is the failure when an ancestor such as macOS `/tmp`, `/var`, or a symlinked home is a link. Those OS paths were not executed here. |
| blackholed endpoint | A connect that does not complete is a timeout: `timed out (TimeoutError) — The model server did not answer before the timeout; retry.` The connect budget is `min(read timeout, 15 s)`, not the 90 s read timeout. | This page did not time a dropped handshake. A connection refused to a closed loopback port is the `network` row, and it returns without waiting out the read timeout. |

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
- `tests/fx1/test_context_window.py` — outbound fitting, the omission note, and the too-long retry.
- `tests/fx1/test_docs_drift.py` — `fx1` commands, `make` targets, and `fx1` imports cited from `README.md` and `docs/FX1*.md` exist.
- `tests/fx1/test_ux_help.py`, `tests/fx1/test_ux_terminal.py`, `tests/fx1/test_ux_portable.py` — `/help` matches the dispatcher, terminal sanitizing, portable layout.
- `tests/fx1/journeys/` — YAML journey engine. `make dipcatcher-journeys`. Freeze file present. Not the five-user trial.
- `tests/fx1/journeys_pty/` — second 100-journey suite, JSON, in-process and pty. `scripts/dipcatcher_journeys_pty.py`. `make dipcatcher-journeys-pty`. Notes in `docs/DIPCATCHER_JOURNEYS_PTY.md`.
- `tests/fx1/relevance/` — `scripts/dipcatcher_relevance.py`, `make dipcatcher-relevance`, `docs/DIPCATCHER_RELEVANCE.md`.
- `tests/fx1/test_chaos_*.py` and `tests/fx1/chaos_support.py` — owned children, signals, store, model connect, kill.
- `tests/fx1/test_redteam_*.py` and `tests/fx1/redteam_support.py` — two authors. `test_redteam_chat_*` drive the chat.
- `tests/fx1/test_review_r1_*.py`, `tests/fx1/test_review_r2_*.py` — review fixes for attach, redaction, connect budget, approvals, and the terminal stop.
- `tests/unit/cli/test_dipcatcher_contract_*.py` — launcher, exec, models, sessions.
- `scripts/dipcatcher_latency.py`, `scripts/dipcatcher_soak.py`, `scripts/dipcatcher_release_smoke.py` — `make dipcatcher-evidence`. `make dipcatcher-gate` runs lint, strict types, `dipcatcher-test`, both journey targets, and relevance.

This page's command transcript is from a local run, not from that suite.
Measured pass counts that exist are in
[the release evidence table](DIPCATCHER_RELEASE_EVIDENCE.md). They do not
close a gate.

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
  smoke for both fx1 and fx1-lite, the macOS/Windows matrix, signed
  release artifacts, a rollback drill, and the corpus-epoch re-stamp
  (it needs the signing key). Those are not implied by the local stub
  runs above. Gates P1–P7 stay open. The partial numbers are in
  [the release evidence table](DIPCATCHER_RELEASE_EVIDENCE.md).
- **Honesty gate boundary.** Spelled-out numbers, a bare metric name with
  no number, a bare integer glued on (`sharpe2`), and a forbidden token
  glued onto a prefix ("mySharpe 2.1") are outside the lexical gate. A
  compromised endpoint can return false but well-formed prose.
- **Plain text is a conservative router.** `intent.looks_like_orchestration`
  is a verb next to an object, plus a closed list of diagnostic noun
  phrases. On the 64-phrase labeled set, a local recount is precision 1.0
  (tp 32, fp 0) and recall 1.0 (fn 0, tn 32). An earlier tree recorded
  recall 0.625 (fn 12, `P021`–`P032`). How-to questions stay on the chat
  path on purpose. P3 stays open. Use `/superpower` when the sentence is
  not one of those phrases.
- **Datasets live in `<cwd>/data`.** Copy the file there. Anywhere else,
  including a symlink out of that directory, is refused.
- **Symlink ancestors.** `ensure_private_dir` refuses a symlink anywhere
  in the path it was given. The error names that path
  (`refusing to follow symlink directory ...`). When the config directory
  itself is the symlink, `dipcatcher models check` printed
  `credentials directory must not be a symlink` and exited 2. Set
  `FX1_CONFIG_DIR` and `FX1_SESSION_DIR` to the resolved real path. macOS
  `/tmp` and `/var`, and a symlinked home, were not executed here.
- **Owned-process residual.** On Linux the direct child dies with the chat
  via `PR_SET_PDEATHSIG`. A grandchild that double-forks and calls
  `setsid` is not killed by that signal and is not found by the descendant
  sweep. `procs.py` states that limit. It was not re-measured for this
  page.
- **Windows and macOS code paths** in this tree were exercised by
  monkeypatched tests. They were not run on those operating systems.
- **Expert half of P3** (at least 90% rated by a domain expert) is open.
  The automated relevance suite is necessary and is not that rating.
- **McCabe ratchet is red** on pre-existing lab entries. None of the names
  are dipcatcher modules. `scripts/check_mccabe_ratchet.py` on this tree
  exited 1 with:
  `src/quant_fund/backtest/carry_engine.py:run_carry_backtest` (57 > baseline 52),
  `src/quant_fund/hedge_lab/lightspeed_book.py:run_lightspeed_file_book` (baseline 10, now 9),
  `src/quant_fund/hedge_lab/runner.py:run_hedge_lab` (20 < baseline 22),
  `src/quant_fund/leakage/ast_scan.py:_check_lh011` (24 > baseline 15),
  `src/quant_fund/leakage/ast_scan.py:_check_lh013` (23 > baseline 14),
  `src/quant_fund/leakage/ast_scan.py:_receiver_price_like` (12 > baseline 11),
  `src/quant_fund/monitoring/drift.py:model_health_report` (baseline 12, now 9),
  `src/quant_fund/pipeline/forecast/history.py:history_for_calibration` (14 < baseline 15),
  `src/quant_fund/research/catalog/rates.py:northset_all_floor_unit_honesty_errors` (baseline 10, now 9),
  `src/quant_fund/research/cost_allocation.py:allocate` (baseline 17, missing),
  `src/quant_fund/research/reality_sweep.py:run_sweep` (baseline 10, now 9),
  `src/quant_fund/research/receipt_v2.py:_verify_v1` (baseline 19, missing),
  `src/quant_fund/schemas/receipt_v2_leaf.py:_verify_v1` (19, new).
- **Crown-jewel and docs epoch pins are red.** `dipcatcher crown-jewels
  --check` exited 1 with `jewel_mutated` on 39 paths, including lab modules
  and also `Makefile`, `README.md`, `CHANGELOG.md`, `.gitignore`, and
  `AGENTS.md`. The list is in the release evidence table. `corpus-epoch`
  on `docs/*` was already drifting before this page was edited, including
  this file. Re-stamping needs the signing key and was not done.
