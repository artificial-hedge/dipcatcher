# Live order gateway

`dipcatcher live` without a gateway file is the operator desk in
[LIVE_DESK.md](LIVE_DESK.md): shadow, an in-process loopback, or a drop
folder. That path does not open a broker socket.

This page is the other book. A config with `starting_cash` and
`pretrade_config` selects the gateway (`--config configs/live.yaml`, or
`--actor` / `--state-dir`). It stays disarmed until you arm it. A fill is
broker accounting. It is not a research receipt, not a proper score, and not a
claim that a strategy made money. `live_pnl_claim` stays false on every
gateway record.

The paper and simulated brokers are unchanged: configuring a live endpoint on
that path still raises `LiveEndpointRefused`. Institutional conditions 1–5 in
[Institutional readiness](INSTITUTIONAL_READINESS.md) are still required before
anyone may describe a result as live evidence. This gateway does not satisfy
those conditions by existing.

## Venues

| `venue` in the config | What it does |
|---|---|
| `loopback` | Fills inside this process at the limit price. Default. No network. |
| `alpaca-paper` | Alpaca paper host `paper-api.alpaca.markets`. Needs API keys. |
| `alpaca-live` | Host `api.alpaca.markets` stays configured. Submission is refused. The honesty contract in `AGENTS.md` forbids live broker connectivity, so the route is not deleted and it does not send. The confirmation phrase below is still required to arm, and arming does not lift the refusal. |

The host is fixed by the venue. The gateway will not send credentials to any
other host, and it refuses HTTP redirects.

Keys are read from the environment and are never written to the state file or
the audit log:

```bash
export ALPACA_API_KEY_ID=...
export ALPACA_API_SECRET_KEY=...
```

`APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` are accepted as aliases.

## Arming

`configs/live.yaml` selects the venue. Copy it if you do not want to change
the tracked default (`loopback`).

```bash
dipcatcher live arm --actor YOUR_NAME --config configs/live.yaml
```

For `alpaca-live` the same command must include:

```bash
--confirm-live-capital "I AUTHORIZE LIVE CAPITAL"
```

That phrase is a confirmation, not a password. The API keys are the secret.
`--at` is refused for `alpaca-live` so the session clock cannot be backdated.
`live disarm` removes the arm file and does not close positions.

Changing the pre-trade file or the calendar makes an existing arm stale. Arm
again after that change.

## What has to pass before an order is sent

1. The arm file matches the config venue and the current pre-trade digest.
2. Network venues have both credentials set.
3. The pre-trade engine allows the order: kill switch, session clock, size,
   collar, buying power, concentration, and the other checks in
   [Pre-trade risk](PRETRADE_RISK.md).
4. A repeated `client_order_id` returns the stored result and does not send
   a second order.

`calendar: us_rth` uses the pre-trade session (New York regular hours,
weekdays). `calendar: continuous` uses a UTC day that stays open on weekends.
Alpaca equity orders are still day-limit orders; the broker can reject one
that arrives outside its own session.

## Commands

```bash
dipcatcher live status
dipcatcher live submit --symbol AAPL --side buy --qty 1 --limit 190 --mark 190 --actor YOUR_NAME
dipcatcher live cancel --client-order-id ORDER_ID --actor YOUR_NAME
dipcatcher live reconcile
dipcatcher live flatten --actor YOUR_NAME
dipcatcher live kill --actor YOUR_NAME --reason "halt"
dipcatcher live reset --actor YOUR_NAME --reason "cleared"
```

`--mark` is the reference price for the collar. It is not taken from a market
data feed. `flatten` posts a limit at the stored mark for each open position.
It does not send a market order. `kill` latches this gateway; it does not
reach into the broker and cancel working orders. Use `cancel` for that, then
`flatten` if you want the position reduced.

`reconcile` pulls the latest fill page and compares cash and positions to the
broker snapshot. A drift alarm means the books differ. Idempotent re-delivery
of a fill id does not change cash or position.

State lives under `state_dir` (default `data/live/default`, gitignored): arm
file, integrity key, order state, and a hash-chained audit log. Those files
are mode `0600`. Do not commit them.

## What this is not

- Not a promotion of a research forecast into a strategy.
- Not evidence for the five institutional conditions.
- Not a profit, NAV, Sharpe, or other performance report.
- Not able to trade on Saturday through `us_rth`. That denial is intentional.
