# DipcatcherDiskPressure

Research and simulated paper. This is not a live-trading incident.

## Trigger

A non-temporary filesystem has under 15 percent free and is not read-only, for 5 minutes.

## Impact

Writes can fail. Backups, logs, and the paper ledger are the first files to watch. The 0.15 ratio is the rule in alerts.yml.

## First 5 minutes

1. Name the mount from the node-exporter labels.
2. See whether logs or a backup copy filled it.
3. Run the log-retain job if the log directory is the consumer.
4. Do not delete a ledger or a receipt to make space.
5. Stop a rollout that is still writing into the full disk.

## Rollback and kill switch

Free space before expecting a rollback to help. make -f deploy/prod.mk prod-rollback still needs room to recreate containers. Trip the kill file if the ledger can no longer be written safely.
