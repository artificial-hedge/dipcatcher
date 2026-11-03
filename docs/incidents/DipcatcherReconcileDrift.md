# DipcatcherReconcileDrift

Research and simulated paper. This is not a live-trading incident.

## Trigger

dipcatcher_reconcile_drift == 1 for 1 minute.

## Impact

A pinned corpus epoch does not match the working tree. owner_restamp would list that corpus. The gauge is read-only. It does not stamp or sign.

## First 5 minutes

1. Run python scripts/ops_signals.py reconcile and read which corpus drifted. Do not stamp.
2. Compare that path with the last reviewed change.
3. Leave the pins as they are. Re-sign is the key holder's step.
4. Do not treat the drift as market evidence.
5. Keep shipping research only after the owner accepts the tree.

## Rollback and kill switch

Rollback restores an image. It does not restore epoch pins. Do not run stamp-epochs, sign-pins, anchor-pins, or checkpoint from this alert.
