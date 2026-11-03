# Kill-audit receipt reconciliation

`receipts/kill_audit.json` is a sealed `kill_audit.v1` receipt. Its `claim`
and `interpretation` both record `n_checks` 57 and `n_violations` 0.
`tests/unit/monitoring/test_kill_audit.py` expects the live matrix to
report 63 checks. Both numbers are right for the tree they describe. The
receipt was not rewritten when the matrix grew. This note does not edit
`receipts/`.

The receipt is SYNTHETIC. It is not a market-evidence score and it is not
a live-trading claim. `FLATTEN` is the paper-file token. `may_flatten`
stays false for that state. `dipcatcher live kill` still writes
`HALT_NEW_ORDERS` and does not flatten the paper book.

## Why 57 and 63

The seal was written from `kill_audit_bench()` in the working tree that
became `94836f167`. `git_revision` on the receipt is
`cb3c2e76d4ebc1d484d3fd59a2044cbd47820190`, the parent of that commit,
because the bench reads HEAD before the commit exists. The interpretation
stores `n_checks`, `n_violations`, `verdict`, and `violations`. It does
not store check names. The names below are the diff of
`src/quant_fund/monitoring/kill_audit.py` from `94836f167` to `ca9f305b4`.

The sealed matrix is 57:

- 4 declared states, each with `auto` false and true, and each of those
  with orders, cancel, and flatten for human false and true:
  4 × 2 × 4 = 32.
- 7 unknown tokens (`BOGUS`, empty, `enabled`, `HALT`, `FLATTEN`, `0`,
  `None`), each with orders, cancel+flatten, and `set_state`: 7 × 3 = 21.
  `FLATTEN` was an unknown token.
- 4 transitions from `ENABLED` to each declared state: 4.
- 32 + 21 + 4 = 57. `n_violations` was 0.

`ca9f305b4` declared `FLATTEN` as a state and removed it from the unknown
list. The same test file changed its expected count from 57 to 63. The
live matrix is 63:

- 5 declared states × 2 × 4 = 40.
- 6 unknown tokens × 3 = 18.
- 5 transitions = 5.
- 40 + 18 + 5 = 63. The new cells pass, so `n_violations` is still 0.

Nine checks were added and three were removed. 57 - 3 + 9 = 63. That net
of six is the gap between the sealed receipt and the test.

## Checks added

These nine labels did not exist in the sealed matrix:

```text
FLATTEN|auto=False:orders
FLATTEN|auto=False:cancel
FLATTEN|auto=False:flatten(human=False)
FLATTEN|auto=False:flatten(human=True)
FLATTEN|auto=True:orders
FLATTEN|auto=True:cancel
FLATTEN|auto=True:flatten(human=False)
FLATTEN|auto=True:flatten(human=True)
transition:ENABLED->FLATTEN
```

## Checks removed

These three labels were counted in the 57 and are not counted in the 63,
because `FLATTEN` is no longer an unknown token:

```text
unknown['FLATTEN']:orders
unknown['FLATTEN']:cancel+flatten
set_state['FLATTEN']
```

## Owner command to reseal

The key holder regenerates the receipt from the bench. Do not hand-edit
`receipts/kill_audit.json` or its `receipt_sha256`. From the repo root:

```bash
uv run python -c 'import json; from pathlib import Path; from quant_fund.monitoring.kill_audit import kill_audit_bench; Path("receipts/kill_audit.json").write_text(json.dumps(kill_audit_bench(), indent=2) + "\n", encoding="utf-8")'
```

That write drifts `receipts/*.json`. Restamp and re-sign in this order.
Do not reorder the steps. The checkpoint step is last because it binds the
signature just written:

```bash
make stamp-epochs
make sign-pins
make anchor-pins
make checkpoint
```

`make sign-pins` and `make checkpoint` need `GATE_SIGNING_KEY` or a key
file. `make anchor-pins` needs the timestamp network. This slice does not
run those commands and does not change the sealed receipt. Until the owner
reseals, the receipt stays at 57 and the test stays at 63.
