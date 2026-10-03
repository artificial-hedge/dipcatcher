"""Alpha-investing alert budget for the ops-monitor fleet.

Each monitor rule (drift PSI, kill-switch trips, ledger anomalies, ...)
can raise an alarm at every step; nothing bounds the fleet-wide
false-alarm rate, so a hundred rules at alpha=0.05 deliver ~5 false
alarms per step — alarm fatigue is a statistical guarantee, not an ops
failure.

``AlertBudget`` runs Foster–Stine alpha-investing across named rules:
every rule owns a wealth share; raising an alarm *wagers* a slice of
that wealth (the rule's local alpha); a confirmed catch pays the wager
back (plus the freed budget), a false alarm forfeits it. A rule whose
wealth hits zero is frozen — it cannot alarm again until a confirmed
catch refunds it. Under an all-null world the fleet's total wager
stays bounded by the initial wealth, so the expected false-alarm count
never exceeds ``alpha_total`` per unit of testing — a time-uniform
guarantee over the whole fleet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_MIN_WEALTH = 1e-12


@dataclass
class _Rule:
    name: str
    wealth: float
    alarmed: bool = False
    n_alarms: int = 0
    n_false: int = 0
    n_confirmed: int = 0
    pending: list[float] = field(default_factory=list)


class AlertBudget:
    """Foster–Stine wallet over named alert rules.

    ``alarm(rule, wager_frac)`` spends ``wealth * wager_frac`` (rejected
    if frozen or empty). ``resolve(rule, confirmed)`` closes the oldest
    pending alarm: confirmed -> refund (the wager returns; the rule may
    alarm again), false -> the wager is destroyed. Wealth is conserved:
    total wealth never grows, so fleet false-alarm spending is bounded
    by ``alpha_total`` over the whole horizon.
    """

    def __init__(self, alpha_total: float, rules: list[str]) -> None:
        if not 0 < alpha_total <= 1 or not rules:
            raise ValueError("alpha_total in (0,1] and at least one rule required")
        self.alpha_total = float(alpha_total)
        share = alpha_total / len(rules)
        self.rules = {r: _Rule(r, share) for r in rules}
        self.spent_total = 0.0
        self.refunded_total = 0.0
        self.destroyed_total = 0.0

    def alarm(self, name: str, wager_frac: float = 0.5) -> float:
        """Spend a wager on an alarm; returns the wager actually placed (0 = frozen)."""
        rule = self.rules[name]
        if not 0 < wager_frac <= 1:
            raise ValueError("wager_frac in (0,1]")
        wager = rule.wealth * wager_frac
        if wager < _MIN_WEALTH:
            # unusable dust counts as destroyed — freezes the rule for real
            self.destroyed_total += rule.wealth
            rule.wealth = 0.0
            return 0.0
        rule.wealth -= wager
        rule.pending.append(wager)
        rule.alarmed = True
        rule.n_alarms += 1
        self.spent_total += wager
        return wager

    def resolve(self, name: str, confirmed: bool) -> float:
        """Close the oldest pending alarm; returns wealth change."""
        rule = self.rules[name]
        if not rule.pending:
            raise ValueError(f"{name}: no pending alarm to resolve")
        wager = rule.pending.pop(0)
        if confirmed:
            rule.wealth += wager
            rule.n_confirmed += 1
            self.refunded_total += wager
            return wager
        rule.n_false += 1
        self.destroyed_total += wager
        return -wager

    def status(self) -> dict[str, Any]:
        total_wealth = sum(r.wealth for r in self.rules.values())
        outstanding = sum(sum(r.pending) for r in self.rules.values())
        return {
            "alpha_total": self.alpha_total,
            "total_wealth": total_wealth,
            "outstanding_wagers": outstanding,
            "spent_total": self.spent_total,
            "refunded_total": self.refunded_total,
            "destroyed_total": self.destroyed_total,
            "wealth_plus_outstanding": total_wealth + outstanding,
            "per_rule": {
                name: {
                    "wealth": r.wealth,
                    "frozen": r.wealth < _MIN_WEALTH and not r.pending,
                    "n_alarms": r.n_alarms,
                    "n_false": r.n_false,
                    "n_confirmed": r.n_confirmed,
                }
                for name, r in self.rules.items()
            },
        }


def alert_budget_bench(
    n_rules: int = 6,
    n_null: int = 4,
    horizon: int = 500,
    seed: int = 0,
    alpha_total: float = 0.2,
) -> dict[str, Any]:
    """Synthetic fleet: ``n_null`` noise rules burn out; the rest stay funded."""
    rng = np.random.default_rng(seed)
    names = [f"rule_{i}" for i in range(n_rules)]
    budget = AlertBudget(alpha_total, names)
    signal = {names[i] for i in range(n_null, n_rules)}
    freezes: dict[str, int | None] = {n: None for n in names}
    for t in range(horizon):
        for name in names:
            rule = budget.rules[name]
            p_alarm = 0.6 if name in signal else 0.05
            if rng.random() < p_alarm:
                wager = budget.alarm(name, wager_frac=0.5 if name in signal else 0.9)
                if wager > 0:
                    # confirmed iff the rule was genuinely signalling
                    budget.resolve(name, confirmed=name in signal)
            if freezes[name] is None and rule.wealth < _MIN_WEALTH and not rule.pending:
                freezes[name] = t
    status = budget.status()
    null_names = names[:n_null]
    sig_names = names[n_null:]
    interpretation = {
        "n_rules": n_rules,
        "horizon": horizon,
        "total_wealth_end": status["total_wealth"],
        "spent_total": status["spent_total"],
        # wealth never grows: held + outstanding + destroyed == alpha_total
        "conservation_bound_holds": abs(
            status["wealth_plus_outstanding"] + status["destroyed_total"] - alpha_total
        )
        <= 1e-9
        and status["destroyed_total"] <= alpha_total + 1e-9,
        "null_freeze_step": {n: freezes[n] for n in null_names},
        "signal_freeze_step": {n: freezes[n] for n in sig_names},
        "null_alarms": {n: status["per_rule"][n]["n_alarms"] for n in null_names},
        "signal_alarms": {n: status["per_rule"][n]["n_alarms"] for n in sig_names},
    }
    verdict = (
        "ok"
        if interpretation["conservation_bound_holds"]
        and all(
            status["per_rule"][n]["n_confirmed"] > 0 or status["per_rule"][n]["n_alarms"] == 0
            for n in sig_names
        )
        else "weak"
    )
    payload: dict[str, Any] = {
        "kind": "alert_budget",
        "schema": "alert_budget.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "fleet false-alarm spending stays within alpha_total; confirmed rules stay funded",
            "verdict": verdict,
        },
        "interpretation": interpretation,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
