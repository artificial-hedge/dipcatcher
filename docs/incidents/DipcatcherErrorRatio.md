# DipcatcherErrorRatio

Research and simulated paper. This is not a live-trading incident.

## Trigger

The instrumented error ratio is at least 0.01 for 5 minutes. otel_export is excluded.

## Impact

More than one in a hundred instrumented stage events is an error. This is a process signal, not a research score.

## First 5 minutes

1. Read dipcatcher_errors_total by stage, excluding otel_export.
2. Open one error log line and confirm it has no token in it.
3. See whether the errors started at the last rollout.
4. If one replica is the source, leave it out of the proxy pool.
5. If both replicas are erroring, roll back.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback restores the previous image. Use the operator kill file when new orders should stop while the errors are still happening.
