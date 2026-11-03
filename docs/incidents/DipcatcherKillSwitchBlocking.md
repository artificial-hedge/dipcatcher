# DipcatcherKillSwitchBlocking

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_kill_switch_blocking == 1 for 1 minute.

## Impact

The operator kill file is blocking new orders. Cancels can still be allowed. This is simulated paper. Nothing here is a live broker.

## First 5 minutes

1. Read the kill file state. Do not print the rest of the data directory.
2. Confirm who set HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, or FLATTEN.
3. Check open paper orders and whether they are already resting.
4. Tell the desk the book is shut for new orders.
5. Keep the file in place until the cause is written down.

## Rollback and kill switch

Rollback does not clear the kill file. make -f deploy/prod.mk prod-rollback only restores the previous image. Clear the file, back to ENABLED, only after the reason for the halt is understood.
