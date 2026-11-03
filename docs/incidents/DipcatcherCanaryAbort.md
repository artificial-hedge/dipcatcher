# DipcatcherCanaryAbort

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_canary_abort == 1 for 1 minute.

## Impact

The canary probe failed or the roll did not finish. With auto-rollback on, the previous image tags are put back and the fleet is recreated.

## First 5 minutes

1. Read the canary JSON: action, checks, and rollback_ok.
2. Confirm dipcatcher:local and the research tag point at the previous image.
3. Check both TLS health listeners.
4. Confirm the paper ledger digest did not change across the rollback.
5. Leave the broken image tag unused. Do not roll it forward.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback is the same retag and recreate when the canary did not finish it. The operator kill file is a separate switch and stays put.
