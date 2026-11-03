# DipcatcherNotReady

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_ready == 0 for 1 minute.

## Impact

The lab readiness gate is closed. Clients that require a ready replica stop making progress. Research and simulated paper keep their last durable state.

## First 5 minutes

1. Open the lab metrics scrape and confirm dipcatcher_ready is 0 on which replica.
2. Read the lab-a and lab-b logs for the readiness error.
3. Check the data directory owner and the write-owner lock.
4. Check DipcatcherKillSwitchBlocking before treating this as a separate outage.
5. Decide whether to restore readiness in place or roll back the image.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback restores the previous image and waits for health. Leave the operator kill file as it is until the book is allowed to take orders again.
