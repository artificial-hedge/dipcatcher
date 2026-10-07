# fxi — interactive CLI for fx-1

`fxi` is the interactive front door to fx-1 and the dipcatcher harness.
Bare `fxi` opens the **concierge console** — conversational orchestration
with `/superpower` capability planning, background deep web research, and
persistent flash-context memory. See `docs/DIP_CONCIERGE.md` for the full
workflow; this page covers endpoints, security, and the classic shell.
**fx1 and fx1-lite are model names** — each has one *endpoint* here: an API
key plus a base URL, stored locally and injected into harness commands, so
`fx1 eval`, hosted chat, and verification runs pick up your credentials
without exporting env vars by hand.

Typing `dipcatcher` with no arguments starts the onboarding wizard: enter
your API key (not echoed), a base URL (default: Moonshot's hosted endpoint),
pick a model — then it drops you into the interactive harness shell.

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

```sh
curl -fsSL https://raw.githubusercontent.com/artificial-hedra/dipcatcher/main/install.sh | sh
```

The installer needs Python ≥ 3.12 (set `FXI_PYTHON=/path/to/python` to pick
one), creates a venv at `~/.local/share/fxi`, installs the fx-1 wheel from
the matching GitHub release, and symlinks `fxi` (plus `fx1`, `dipcatcher`,
`quant`, `verify-ledger`, `mc-engine`) into `~/.local/bin`. It never uses
sudo. Overrides: `FXI_VERSION`, `FXI_INSTALL_URL` (wheel URL), `FXI_HOME`,
`FXI_BIN_DIR`; flags `--wheel dist/....whl` (local dev install),
`--uninstall`.

### Homebrew (tap)

The distribution is proprietary, so the formula lives in a private tap:

```sh
brew tap artificial-hedge/tap
brew install fxi
```

Formula source: `packaging/homebrew/fxi.rb` (copy into the tap repo at
`Formula/fxi.rb`). `url`/`sha256` are filled at release time — see the
release checklist below.

## Quickstart

```sh
# First run launches a wizard (API key, base URL) → harness shell
dipcatcher
fxi keys list           # presence-only endpoint table
fxi doctor              # fx-1 state, endpoints, store, API reachability
fxi chat                # multi-turn chat with the active model (orb while it thinks)
fxi eval                # the fx-1 eval bank (proper scores, honesty gate)
```

Inside the shell (prompt shows `model@host`):

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

One-shot equivalents: `fxi keys …`, `fxi doctor`, `fxi eval [--out PATH]`,
`fxi chat -m "…"`, `fxi harness list|describe|run`, `fxi verify PATH`,
`fxi run TOOL ARGS…`, `fxi setup` (re-run the wizard). The global
`--model fx1|fx1-lite` flag selects the model for one invocation.

## Uninstall

```sh
curl -fsSL https://raw.githubusercontent.com/artificial-hedra/dipcatcher/main/install.sh | sh -s -- --uninstall
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
