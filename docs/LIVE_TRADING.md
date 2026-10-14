# Operator order routing

`dipcatcher live` sends orders to the operator's own Alpaca account. It is
off until you arm it. Research commands (`research`, `backtest`, `paper`,
`sim-live`) never call it, and the model harness cannot call it either.

This is not an institutional live-trading claim. `would_promote_live` and
`live_pnl_claim` stay false on every routing record. The five conditions in
[INSTITUTIONAL_READINESS.md](INSTITUTIONAL_READINESS.md) are still required
before any research result can be described as live evidence. A filled
broker order is an account fact, not a research score.

`runtime.allow_live: true` still raises. Paper and backtest stay on the
simulated broker, which refuses a live endpoint with `LiveEndpointRefused`.

## What has to be true before an order is sent

1. An authorization file (`configs/live_authorization.example.json` is the
   shape) whose `acknowledge` text is exact, whose lifetime is at most 90
   days, and whose `max_order_notional` is within `configs/pretrade_risk.yaml`.
2. `dipcatcher live arm` for that file. Changing the file disarms the match;
   arm again.
3. Broker key, secret, and the pre-trade HMAC key in the environment. They
   are never written to the audit log. See `.env.example`.
4. For `environment: live`, the process also needs
   `DIPCATCHER_LIVE_CONFIRM=ROUTE_LIVE_ORDERS`. Paper does not.
5. The pre-trade engine allows the order: session, size, buying power,
   concentration, duplicate, and the loss latch. A deny does not reach the
   broker.
6. The kill switch is clear. `dipcatcher live kill` is local and does not
   need the broker. It survives disarm. `reset-kill` clears it and does not
   cancel resting orders.

Paper uses `https://paper-api.alpaca.markets`. Live uses
`https://api.alpaca.markets`. Any other host is refused before a socket
opens. Redirects are refused.

## Commands

```text
dipcatcher live preflight --authorization configs/live_authorization.example.json
dipcatcher live arm --authorization configs/live_authorization.example.json --state-dir data/live
dipcatcher live status --state-dir data/live
dipcatcher live sync --state-dir data/live
dipcatcher live order --symbol AAPL --side buy --qty 1 --limit-price 100 --state-dir data/live
dipcatcher live cancel --broker-order-id ORDER_ID --state-dir data/live
dipcatcher live cancel-all --state-dir data/live
dipcatcher live flatten --confirm FLATTEN --state-dir data/live
dipcatcher live kill --actor YOUR_NAME --reason halt --state-dir data/live
dipcatcher live reconcile --state-dir data/live
```

Limit orders are the default. `--market` is explicit. `data/live/` is the
state directory and is gitignored. Do not point `--state-dir` at `receipts/`.

`flatten` submits reduce-only orders for open positions and then latches the
kill switch. It still has to pass the session clock and buying-power checks.
If the session is closed, cancel resting orders and flatten on the next open.
A broker timeout on a submit or cancel latches the kill switch, because the
order may have reached the broker. Reconcile before resetting it.

## What this desk does not do

- It does not promote a research receipt to live capital.
- It does not headline Sharpe, Sortino, Calmar, P&L, or NAV.
- It does not reconstruct tax lots across process restarts, so the
  good-faith check is only as good as the current pre-trade snapshot.
- It does not ingest a halt or short-locate feed. A cash account cannot
  open a short. Closes of positions the broker already reports are allowed.
- It is single-writer. One operator, one state directory.

Start on the paper environment. Live routing needs the confirmation
variable, a current authorization file, and an account you are allowed to
trade. You can lose money.
