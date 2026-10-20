# fxi — interactive CLI for fx-1

`dipcatcher` now opens the **concierge console** in an interactive terminal;
`fxi` remains a compatible launcher for the same conversation with the **fx1** or
**fx1-lite** chat ids. No fine-tuned weights are published. The console includes `/superpower` command planning, background
web research, and persistent flash-context memory. Research-family suitability
and durable session/job recovery remain open. See [the CLI guide](DIP_CONCIERGE.md)
for the working-tree status and full workflow; this page covers endpoints,
security, and the classic shell.
**fx1 and fx1-lite are model names** — each has one *endpoint* here: an API
key plus a base URL, stored locally. The chat client reads them from the
store; lab harness subprocesses inherit a scrubbed environment (no API keys
in `os.environ`). Use `fxi eval` / hosted chat with `/use` or `fxi setup`
without exporting env vars by hand.

`fxi setup` runs the onboarding wizard: enter your API key (not echoed), a
base URL (default: Moonshot's hosted endpoint), pick a model — then it drops
you into the chat. Bare `dipcatcher` opens chat directly; `/keys set fx1`
configures both the masked key and endpoint URL there. Bare redirected
`dipcatcher` prints help without consuming stdin. `dipcatcher chat --model
fx1-lite` requires terminal stdin/stdout; use existing explicit lab or `fxi`
one-shot commands for scripts.

## Model endpoints

| Model | Stored per model | Powers |
|---|---|---|
| `fx1` | API key + base URL | `fxi chat`, `fxi eval`, hosted judges |
| `fx1-lite` | API key + base URL | the same surfaces, via the light model |

Notes:

- The `model` field sent to the API is the model name itself (`fx1` or
  `fx1-lite`).
- A stored endpoint wins over the process environment; with nothing stored,
  fxi falls back to `MOONSHOT_API_KEY` / `FX1_BASE_URL` env vars.
- Applying a model (via `use` / `--model` / the wizard) injects
  `MOONSHOT_API_KEY` and `FX1_BASE_URL` into the session, so every child
  process (`fx1 eval`, `dipcatcher …`) inherits the credential.

## Security model

- Keys live in `~/.fx1/credentials.json` (override the directory with
  `FX1_CONFIG_DIR`), file mode `0600`, written atomically.
- Key prompts use `getpass` — input is not echoed. `--value` exists for
  scripts but lands in shell history; prefer the prompt.
- All listings are presence-only: you see `set`/`env`/`unset` and a last-4
  fingerprint (`...wxyz`), never the key. Mirrors `fx1 doctor`.
- Keys reach child processes as environment variables, never as argv.
- Fail-closed: a missing endpoint names the model and the setup path, never
  a silent fallback.

## Thinking orbs

Animations follow [rareformlabs' thinking-orbs](https://rareformlabs.github.io/thinking-orbs/)
(MIT) — dotted monochrome orbs with a distinct choreography per state —
ported to the terminal as Braille pixels:

- `Agent listening…` — idle banner / shell startup
- `Composing…` — while the model writes a reply
- `Searching…` — while the wizard checks your host
- `Working…`, `Solving…`, `Shaping…` — harness/eval states

Ink auto-detects light/dark from `COLORFGBG`. Orbs are automatically
disabled for non-TTY output, when `NO_COLOR` is set, or when
`FXI_ORBS=off` — pipes and CI get clean static output.

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

Inside the classic `fxi shell` (prompt shows `model@host`):

```
fxi · fx1@api.moonshot.ai › keys              # endpoint table
fxi · fx1@api.moonshot.ai › use fx1-lite      # switch model (injects its endpoint)
fxi · fx1@fx-lite.example.test › harness list
fxi · fx1@fx-lite.example.test › harness run doctor
fxi · fx1@fx-lite.example.test › verify receipts/coherence_2dd641ab766a536a.json
fxi · fx1@fx-lite.example.test › run fx1 modelcard
fxi · fx1@fx-lite.example.test › !ls data/fx1 # raw shell passthrough
fxi · fx1@fx-lite.example.test › exit
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
