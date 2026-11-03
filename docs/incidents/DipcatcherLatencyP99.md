# DipcatcherLatencyP99

Research and simulated paper. This is not a live-trading incident.

## Trigger

HTTP p99 is above 0.25 seconds for 5 minutes.

## Impact

Clients are waiting. The stage-latency alert is the inside of the process. This one is the edge the proxy records.

## First 5 minutes

1. Read dipcatcher:http_latency:p99_5m by service.
2. Compare it with DipcatcherStageLatencyP99.
3. Check CPU, disk, and whether a soak or a load test is running on this host.
4. See whether the last image raised the latency.
5. Roll back if the previous image was inside 0.25 seconds.

## Rollback and kill switch

make -f deploy/prod.mk prod-rollback recreates the previous image. The kill file does not lower latency. Use it only when new orders should stop.
