# DipcatcherBackupAge

Research and simulated paper. This is not a live-trading incident.

## Trigger

The last backup success timestamp is more than 86400 seconds behind Prometheus time, for 5 minutes.

## Impact

A verified paper-state backup is older than one day, or the gauge is still 0 because no backup has been published. Restore would use that older copy.

## First 5 minutes

1. Read dipcatcher_backup_last_success_timestamp.
2. Check the backup timer and the ship log. Do not print credentials from the ship environment.
3. Run make -f deploy/prod.mk prod-backup-ship when the copy target is configured.
4. Confirm the textfile gauge moves to the new unix time.
5. If the ship keeps failing, keep the previous archive. Do not delete it.

## Rollback and kill switch

A rollback does not create a backup. Take a fresh ship before a risky rollback when the age is already high. The kill file is unchanged by the ship.
