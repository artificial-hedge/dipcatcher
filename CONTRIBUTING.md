# Contributing

**fx-1** is the model; **dipcatcher** (`src/quant_fund`) is the harness that
builds, evaluates, and verifies it. Read `AGENTS.md` first — it documents the
gates, the honesty contract, and the lab's work-tracking conventions.

> **Proprietary repository.** This codebase is licensed under the
> [`LICENSE`](LICENSE) at the repository root, not an open-source license.
> All rights are reserved by Advaith Vaithianathan (founder, Artificial
> Hedge). The notes below describe the gates a change must pass; they are
> not an invitation to fork, redistribute, or reuse this code. Any
> contribution is accepted only from parties explicitly authorized in
> writing by the copyright holder, and any accepted contribution is
> understood to be assigned to the Owner under the same license.

## Gates (run before committing)

```bash
make sync   # locked env first: uv sync --frozen --all-groups --all-extras
```

| Gate | Command | What it covers |
|---|---|---|
| Lint | `make lint` | `ruff check` + `ruff format --check` on `src`/`tests`, plus the mypy strict-allowlist ratchet |
| Types | `make typecheck` | `mypy src/quant_fund` (public modules strict); fx-1 work also runs `uv run mypy src/fx1` |
| Lab tests | `make test` | PR gate: not network, not slow. `make test-full` adds slow |
| fx-1 suite | `make fx1-test` | the separate fx-1 lane; `make fx1-gate` = lint + types + tests + honesty + corpus smoke |

Nothing merges red. Do not weaken a gate, a fail-closed default, or a
honesty check to make a run pass — fix the code or report the blocker.

## Honesty contract (hard rules — do not weaken)

1. Research claims are **proper scores** only (pinball, CRPS, PIT, QLIKE,
   Brier, ECE, Kupiec, HMM likelihood) — never Sharpe/Sortino/Calmar/
   P&L/NAV headlines. `FORBIDDEN_RESEARCH_METRIC_KEYS`
   (`quant_fund.research.catalog`) is mirrored by
   `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`; change them together and let
   `tests/fx1/test_honesty_inheritance.py` confirm.
2. SYNTHETIC results are correctness tests, always labeled SYNTHETIC, and
   never market evidence.
3. **No live-trading claims.** This is research infrastructure: no broker
   connectivity, no order placement, nothing presented as live performance.
4. `receipts/` and all signed state are immutable evidence — never edit,
   never delete; every research claim should be reproducible from a
   receipt hash.

## Non-negotiables

- The honesty contract above is enforced, not advisory
  (`tests/fx1/test_honesty_inheritance.py` blocks token-set drift).
- `uv.lock` is authoritative — regenerate via `uv lock` after touching
  `[project]`; `uv lock --check` must pass.
- Don't commit derived artifacts (`data/fx1/`, coverage, `mlruns/`) or
  secrets (see `.env.example` for the env-var surface).
- Every tracked env var the code reads is documented in `.env.example`;
  add the name there in the same change that introduces the read.
