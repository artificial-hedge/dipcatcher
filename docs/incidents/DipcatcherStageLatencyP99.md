# DipcatcherStageLatencyP99

Research and simulated paper. This is not a live-trading incident.

## Trigger

Stage latency p99 is above 2 seconds for 5 minutes.

## Impact

Instrumented stages are slow. Requests are still answered, and the error budget is a separate alert.

## First 5 minutes

1. Identify the stage label on dipcatcher_stage_latency_seconds.
2. Compare lab-a and lab-b. A single slow replica can be drained by the proxy.
3. Check disk pressure and scrape freshness before chasing the stage.
4. Look at the recent deploy. A new image is a candidate to roll back.
5. If the slowdown is the data directory, stop the rollout.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback returns the previous image. Trip the operator kill file if the slow path is still accepting orders that should wait.
