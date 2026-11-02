# End-to-end smoke

Offline correctness check for one SYNTHETIC book. It is not market evidence
and it is not a live-trading gate.

The run walks a bounded fixture through the research path:

1. ingest
2. features and labels
3. ridge ranking train
4. forecast, including conformal intervals
5. causal weights, listing exit, and the reference backtest
6. portfolio component risk and a pre-trade check
7. historical VaR/ES and the causal book gate
8. a SYNTHETIC report and tearsheet

`configs/e2e_smoke.yaml` pins the fixture: 6 names, 100 sessions, seed 13.
The runner refuses a non-synthetic source, a hosted ranker, and any panel
above 12 names or 160 sessions. A NaN, an infinity, or an empty stage raises
and the process exits non-zero.

Two runs with the same seed write the same receipt digest. The digest is
SHA-256 of the canonical receipt. Paths, MLflow run ids, and artifact
filenames stay out of that body. The receipt records rank IC and historical
VaR/ES. The tearsheet file is a research diagnostic; the receipt stores its
hash.

A published listing end is flattened on the last decision that can still
fill on the name's last print. That keeps the event-loop backtest from
holding a name after its bars have stopped. An end outside the sample, or a
name that is already flat, is left unchanged. A held name whose end was not
published in time to exit fails closed.

## Run it

```bash
make e2e-smoke
uv run pytest -m e2e_smoke -q
uv run python -m quant_fund.pipeline.e2e_smoke --root /tmp/e2e-smoke
```

`make e2e-smoke` selects the `e2e_smoke` pytest marker and pins one BLAS
thread. The direct module entry prints the receipt JSON, including `digest`.
Repeat it with a second `--root` and the same `--seed` to compare digests.
A laptop run is well under five minutes.

## Paper sessions

The same fixture continues into a simulated paper book. Four session dates,
two names, one share each, filled at that session's close. The book is
checkpointed halfway and resumed. After the fills:

- an order above the quantity cap is rejected
- the kill switch rejects a later submit
- cancel of a working order still succeeds while new orders are halted
- a fresh session opened from the checkpoint reproduces positions and cash

A NaN, an infinity, or an empty book raises. Nothing in this path opens a
socket. The receipt schema is `e2e_paper.v1`, labeled SYNTHETIC, with
`live_pnl_claim` false. Two roots and the same seed share a digest.

```bash
uv run python -m quant_fund.simtest.e2e_paper --root /tmp/e2e-paper
```

`make e2e-smoke` collects this stage with the ingest-to-report smoke.

## Release dry-run

CI's package job builds the sdist and wheel and runs the installed
`dipcatcher --help` and `fx1 --help`. The local check does the same offline:
one wheel, one sdist, both entrypoints exit 0 with a Usage line, and the
checkout environment stays on the source tree.

```bash
make release-dry-run
uv run pytest -m release_dry_run -q
```

