# CLI command index

This list is generated from the registered commands. `dipcatcher` and `quant`
share the dipcatcher list (`quant --version` prints `dipcatcher <version>`).
Argparse tools are `verify-ledger`, `data-quality`
(`python -m quant_fund.data.quality`), `parity` (`python -m quant_fund.parity`),
`market-sim` (`python -m quant_fund.market_sim`), and `robustness`
(`python -m quant_fund.robustness`). `--help` and `--version` / `-V` work on
every command below. Exit `0` is success, exit `1` is an operational failure,
and exit `2` is usage. A user mistake prints one `error:` line on stderr and
does not print a traceback.

The README is byte-pinned, so the checked copy of this index lives here.

<!-- cli-index:start -->

### dipcatcher

```text
admit
admit-batch
allocation
allocation compare
allocation engines
allocation verify-receipt
allocation walk-forward
anchor-timestamp
api
attest-release
audit-record
audit-trace
backtest
basis-carry
book-panel
build-features
build-labels
calibration-eval
candle-book
capacity
checkpoint
checkpoint-chain
coherence
collect
compare
concordance
corpus
corpus-absence
corpus-consistency
corpus-epoch
corpus-proof
cost-calibration
crown-jewels
custody
doctor
epoch-delta
epoch-position
evidence-export
execution-sensitivity
expert-mixture
explain
explain attach-sidecar
explain demo
explain report
explain verify-sidecar
fleet
fleet-monitor
forecast
forward-shadow
forward-shadow decide
forward-shadow dump
forward-shadow freeze
forward-shadow miss
forward-shadow plan
forward-shadow reconcile
forward-shadow settle
fuzz-drill
fuzz-receipts
graph
hmm
hmm eisner
hmm train
ingest
kronos-forecast
kyle-ofi
lab
lake
lake import
lake quality
lane-power
lattice
leakage
leakage scan
leakage watch
lineage
lineage show
lineage verify
live
live arm
live cancel
live disarm
live flatten
live kill
live mark
live poll
live reconcile
live reset
live status
live submit
ls
ls book
ls confirm
ls demo
ls hunt
ls race
ls specs
mcs
membership-coverage
monitor
multih-fleet
net-tournament
net-tournament prepare
net-tournament run
northset
online-fdr
optimize
ots-stamp
ots-upgrade
pairs
paper
pit
pit init
pit restate
pit stats
pit verify
promote-bars
proof
proof chain-head
proof replay
proof run
proof verify
proofcore
proofcore chain-head
proofcore export
proofcore log
proofcore query
prospective-sota
prospective-sota commitment
prospective-sota forecast
prospective-sota interrupt
prospective-sota prepare
prospective-sota settle
prospective-sota verify
qm
qm bs-price
qm gex-decide
qm greeks
qm hrp
qm iv
qm validate-greeks
quorum-init
quorum-rotate
quorum-sign
race
ranker-probability
ranker-probability run
rankic
real-benchmark
real-benchmark prepare
real-benchmark score
reality
reality ledger-gate
reality preflight
reality survivorship
reality sweep
reality trial-report
regime-performance
replay
replay-all
report
research
research100
research100 benchmark
research100 catalog
research100 verify
rotate-key
serial-watch
session-book
sign-pins
sim-live
sota
sota block-sensitivity
sota effects
sota run
sota show
sota validate-bars
stress
stress crises
stress report
suite-health
tamper-drill
tape-pin
tape-verify
tearsheet
tombstone
total-return
trade
train
train alpha
train calibration
train covariance
train distribution
train liquidity
train ranking
train regime
train reinforcement
train tail
train volatility
validate
vendor-book-map
verdict
verify-all
verify-bundle
verify-checkpoint
verify-identities
verify-ledger
verify-receipt
verify-release
verify-repo
verify-research
verify-rotations
verify-witness
vol-bench
witness-bundle
witness-checkpoint
witness-scan
xvenue-basis
```

### fx1

```text
attestation
backtest
capability-eval
contamination-audit
corpus
corpus build
corpus build-full
corpus ingest-source
curriculum
dipbench
doctor
dpo
eval
ext-bench-eval
harness
harness batch
harness batch-cancel
harness batch-output
harness batch-run
harness batch-status
harness batch-submit
harness batch-wait
harness batches
harness bench
harness cancel
harness capabilities
harness chat-delete
harness chat-get
harness chat-list
harness chat-messages
harness chat-update
harness check-text
harness commands
harness compat
harness complete
harness completion
harness completions
harness conv-create
harness conv-delete
harness conv-get
harness conv-item
harness conv-items
harness conv-items-add
harness conv-items-delete
harness conv-update
harness describe-operation
harness doctor
harness drain
harness embed
harness eval
harness eval-cancel
harness eval-diff
harness eval-run
harness eval-run-cancel
harness eval-run-delete
harness eval-run-get
harness eval-run-items
harness eval-run-list
harness eval-spec-create
harness eval-spec-delete
harness eval-spec-get
harness eval-spec-list
harness eval-spec-update
harness eval-status
harness eval-wait
harness evals
harness execute-operation
harness file-content
harness file-delete
harness file-upload
harness files
harness ft-cancel
harness ft-checkpoints
harness ft-create
harness ft-events
harness ft-jobs
harness ft-pause
harness ft-resume
harness ft-status
harness ft-wait
harness health
harness job
harness jobs
harness key-create
harness key-get
harness key-patch
harness key-revoke
harness key-rotate
harness key-usage
harness keys
harness list
harness message
harness message-batch
harness message-batch-cancel
harness message-batch-delete
harness message-batch-results
harness message-batch-status
harness message-batch-wait
harness message-batches
harness message-tokens
harness metrics
harness model
harness model-delete
harness models
harness moderate
harness operations
harness probe
harness ready
harness receipt
harness receipts
harness respond
harness response-cancel
harness response-delete
harness response-get
harness response-input-items
harness response-replay
harness run
harness score
harness self
harness selftest
harness serve
harness submit
harness submit-batch
harness text-completion
harness upload
harness upload-cancel
harness usage
harness verify
harness version
harness vs-batch-cancel
harness vs-batch-create
harness vs-batch-files
harness vs-batch-get
harness vs-create
harness vs-delete
harness vs-file-add
harness vs-file-content
harness vs-file-delete
harness vs-file-get
harness vs-files
harness vs-get
harness vs-list
harness vs-search
harness vs-update
harness wait
harness watch
infer
maskedaEval
modelcard
mrm
options-reasoning-eval
redteam
sbom
sign
sources
sources describe
sources fetch
sources list
sources probe
sources route
sources scenarios
train
train manifest
```

### fxi

```text
chat
doctor
eval
flash
flash add
flash list
flash refresh
flash remove
flash search
flash show
harness
harness describe
harness list
harness run
keys
keys list
keys remove
keys set
research
run
setup
shell
superpower
verify
```

### mc-engine

```text
bench
help-interface
resume
run
```

### verify-ledger

```text
verify-ledger
```

### data-quality

```text
data-quality
```

### parity

```text
smoke
```

### market-sim

```text
bench
report
scenarios
stress
stylized
```

### robustness

```text
smoke
```

<!-- cli-index:end -->
