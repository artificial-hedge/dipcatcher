# fxi — interactive CLI for fx-1

`dipcatcher` now opens the **concierge console** in an interactive terminal;
`fxi` remains a compatible launcher for the same conversation with the **fx1** or
**fx1-lite** chat ids. No fine-tuned weights are published. The console includes `/superpower` command planning, background
web research, and persistent flash-context memory. Research-family suitability
and durable session/job recovery remain open. See [the CLI guide](DIP_CONCIERGE.md)
for the working-tree status and full workflow; this page covers endpoints,
security, and the classic shell.
**fx1 and fx1-lite are model names** — each has one *endpoint* here: an API
key plus a base URL, stored locally. Hosted chat and `fx1 eval` receive the
credential via an explicit child `env=` mapping. Lab harness subprocesses
(`dipcatcher`, `quant`) inherit a scrubbed environment (no API keys in
`os.environ`). Use `/use` or `fxi setup` without exporting env vars by hand.

No live model was authenticated while this page was checked. Calls that
needed a model used a loopback HTTP stub. The only built-in endpoint
constant is `https://api.moonshot.ai/v1/chat/completions`. An operator
endpoint is the URL you store or export.

**fx1 and fx1-lite are model names.** Each has one endpoint: an API key
plus a completion URL, stored locally.

`fxi setup` runs the wizard (API key not echoed, base URL, model) and
then enters the console. Bare `dipcatcher` on a terminal opens chat
directly; `/keys set fx1` stores the key and URL there. Redirected bare
`dipcatcher` prints help and does not read stdin. `dipcatcher chat`
requires a terminal. With stdin closed, bare `fxi` prints the concierge
banner and exits on EOF; it does not print the lab help.

`/data`, `/runs`, and `/models check` are chat commands. `dipcatcher exec
--data` attaches a CSV or Parquet for one turn, using the same resolver as
`/data` (the file has to already be under `./data`). Exit codes, the
SIGHUP difference between chat (129) and exec (130), and the open gates are
in the CLI guide. With `PYTHONIOENCODING=ascii` the banner observed here
was `dipcatcher | fx1 @ unconfigured | unset` and the prompt was `fx1 > `.

## Model endpoints

| Model | Environment when nothing is stored | Shared fallback |
|---|---|---|
| `fx1` | `DIPCATCHER_FX1_API_KEY`, `DIPCATCHER_FX1_BASE_URL` | `MOONSHOT_API_KEY` and `FX1_BASE_URL`, else the Moonshot URL constant |
| `fx1-lite` | `DIPCATCHER_FX1_LITE_API_KEY`, `DIPCATCHER_FX1_LITE_BASE_URL` | the same shared pair |

Resolution order for each model is stored profile, then that model's
environment, then the shared environment. A per-model key with no
per-model URL uses `FX1_BASE_URL` if set, otherwise the Moonshot
constant. A per-model URL without a per-model key does not replace the
shared URL.

The `model` field sent to the API is `fx1` or `fx1-lite`.

The classic shell's `use` command injects `MOONSHOT_API_KEY` and
`FX1_BASE_URL` into that shell process. Hosted chat and `fx1 eval`
receive the credential via an explicit child `env=` mapping. Owned
concierge harness children do not inherit those variables; they get the
scrubbed environment described in the CLI guide. A shell child was not
launched in this check.

## Security model

- Keys live in `$FX1_CONFIG_DIR/credentials.json` (default
  `~/.fx1/credentials.json`), mode `0600` on POSIX, written atomically.
  The config directory observed here was mode `0700`. On Windows the
  store is restricted with `icacls`; that was not executed here.
- Key prompts use `getpass`. Input is not echoed. `--value` exists for
  scripts and lands in shell history.
- Listings are presence-only. A stored key shows a last-four fingerprint
  (`...wxyz`). An env-backed `dipcatcher models list` row shows
  fingerprint `-`. The chat banner prints the resolved fingerprint.
- Fail-closed: a missing endpoint names the model and `/keys set`, and
  does not invent a reply.
- Pasted URLs are normalized to a chat-completions route. Plain HTTP is
  accepted only for a literal loopback address. Userinfo and query
  strings are rejected. Details and the error strings are in the CLI
  guide.

## Thinking orbs

Animations follow [rareformlabs' thinking-orbs](https://rareformlabs.github.io/thinking-orbs/)
(MIT) — dotted monochrome orbs with a distinct choreography per state —
ported to the terminal as Braille pixels:

- `Agent listening…` — idle banner / shell startup
- `Composing…` — while the model writes a reply
- `Searching…` — while the wizard checks your host
- `Working…`, `Solving…`, `Shaping…` — harness/eval states

Ink auto-detects light/dark from `COLORFGBG`. Animated orbs are off when
stdout is not a TTY, when `NO_COLOR` is set (any value), or when
`FXI_ORBS=off`. With `NO_COLOR=1` on a pseudo-terminal the startup banner
was plain text (`dipcatcher · fx1 @ unconfigured · unset`), not the
braille box. The scripted non-TTY console still printed the static
braille banner. Bracketed paste stays on for the idle TTY line editor.
While a background job runs, that line uses the cbreak reader (no history,
no arrow keys). A fatal tty signal restores the terminal and turns
bracketed paste off.

## Install

### curl

There is no `v*` release tag, and `release.yml` does not create a GitHub
Release, so this curl path has no wheel to install. Install this checkout
with `uv sync --frozen --all-groups --all-extras`. The block below is the
installer contract for a wheel you supply with `--wheel` or `FXI_INSTALL_URL`.

```sh
curl -fsSL https://raw.githubusercontent.com/artificial-hedge/dipcatcher/main/install.sh | sh
```

The installer needs Python ≥ 3.12 (set `FXI_PYTHON=/path/to/python` to pick
one), creates a venv at `~/.local/share/fxi`, installs the fx-1 wheel from
the matching GitHub release, and symlinks `fxi` (plus `fx1`, `dipcatcher`,
`quant`, `verify-ledger`, `mc-engine`) into `~/.local/bin`. It never uses
sudo. Overrides: `FXI_VERSION`, `FXI_INSTALL_URL` (wheel URL), `FXI_HOME`,
`FXI_BIN_DIR`; flags `--wheel dist/....whl` (local dev install),
`--uninstall`. A remote wheel (`http://`/`https://`, the default included)
installs only with `FXI_WHEEL_SHA256=<hex>`: the installer refuses an
unpinned download and verifies the checksum **before** pip install. Pass
`--requirements FILE` (or `FXI_REQUIREMENTS`) with a hashed requirements file
to lock the dependencies (`--require-hashes`, wheel added `--no-deps`,
`pip check` refuses a missing pin); without it pip resolves them from the
index, unhashed, and the installer warns. The installer refuses destructive
`FXI_HOME` values (empty, `/`, or `$HOME`), and `sh install.sh --help` works
even under `curl | sh`.

### Homebrew (tap)

Not installable. There is no `v*` release, and the formula `sha256` is still
`FILL_AT_RELEASE`. The supported install is a clone plus
`uv sync --frozen --all-groups --all-extras`. The formula source below is the
private-tap text for a future release, not a command that works today:

```sh
brew tap artificial-hedge/tap
brew install fxi
```

Formula source: `packaging/homebrew/fxi.rb` (copy into the tap repo at
`Formula/fxi.rb`). `url`/`sha256` are filled at release time — see the
release checklist below.

## Quickstart

```sh
dipcatcher              # chat; /keys set fx1 configures the endpoint inside it
dipcatcher chat --model fx1-lite  # fx1-lite chat id; no released weights
fxi setup               # optional standalone setup wizard → the chat
fxi                     # compatibility entry for the same chat
fxi keys list           # presence-only endpoint table
fxi doctor              # fx-1 state, endpoints, store, API reachability
fxi chat                # multi-turn chat on the active chat id (orb while it thinks)
fxi eval                # the fx-1 eval bank (proper scores, honesty gate)
```

`fxi --help` also lists `shell`, `doctor`, `eval`, `chat`, `verify`,
`run`, `superpower`, `research`, `harness`, and `flash`. `fxi doctor`
and `fxi eval` were not run against a live endpoint in this check.

With no endpoint stored, the classic shell prompt is the Moonshot host
from the built-in URL, not a configured endpoint:

```text
fxi · fx1@api.moonshot.ai ›
fxi · fx1-lite@api.moonshot.ai ›
```

One-shot equivalents: `fxi keys …` (including `fxi keys test [model]`, an
authenticated identity probe that exits 1 on failure), `fxi doctor`,
`fxi eval [--out PATH]`, `fxi chat -m "…"`, `fxi harness list|describe|run`,
`fxi verify PATH`, `fxi run TOOL ARGS…`, `fxi setup` (re-run the wizard),
`fxi sessions [--limit N]` and `fxi resume SESSION` (the durable session
journal), `fxi superpower GOAL --sync [--approve-consequential]` and
`fxi research GOAL --sync`. One-shot runs never wait for an approval:
consequential harness commands are denied unless `--approve-consequential`
states the policy. The global `--model fx1|fx1-lite` flag selects the model
for one invocation; `dipcatcher --version` prints the distribution version.

## Uninstall

```sh
curl -fsSL https://raw.githubusercontent.com/artificial-hedge/dipcatcher/main/install.sh | sh -s -- --uninstall
# or, if install.sh is checked out:  ./install.sh --uninstall
rm -rf ~/.fx1   # also delete stored endpoints
```

`brew uninstall fxi` for the Homebrew install.

## Release checklist (curl + brew)

1. Bump `fx1.__version__` per `docs/FX1_API_STABILITY.md` (additive change →
   minor).
2. `uv build` → `dist/fx_1-<version>-py3-none-any.whl`; smoke-test:
   `uv pip install --no-deps dist/*.whl` then `fxi --version`.
3. Attach the wheel to the GitHub release `v<version>` — the install script
   and the formula both download from there.
4. `sha256sum dist/fx_1-<version>-py3-none-any.whl`; update `url` + `sha256`
   + `FXI_VERSION` in `packaging/homebrew/fxi.rb` (in the tap repo) and the
   `FXI_VERSION` default in `install.sh`.
5. Update the table in `README.md` if entry points changed.
