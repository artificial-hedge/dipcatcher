"""Edge-triggered SLO and kill-switch alerts.

Uses the in-process SLO evaluator and ``AlertBudget``. A rule alarms when it
enters ``fail`` and resolves when it returns to ``pass``. ``no_data`` does not
alarm and does not clear a pending failure. The kill-switch file is one rule.
Draining is shutdown, not an alert.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from quant_fund.api.lifecycle import ManagedRuntime, acceptance_block
from quant_fund.monitoring.alert_budget import AlertBudget
from quant_fund.observe.logging import log_event
from quant_fund.observe.slo import Slo, current_slo_snapshot, evaluate, load_slos


def plan_alert_actions(
    previous: Mapping[str, str], statuses: Mapping[str, str]
) -> list[tuple[str, str]]:
    """Return ``(rule, action)`` with action ``alarm`` or ``resolve``."""
    actions: list[tuple[str, str]] = []
    for rule, status in statuses.items():
        if status == "no_data":
            continue
        prior = previous.get(rule)
        if status == "fail" and prior != "fail":
            actions.append((rule, "alarm"))
        elif status == "pass" and prior == "fail":
            actions.append((rule, "resolve"))
    return actions


class AlertMonitor:
    """Background evaluator. ``tick`` is the test seam."""

    def __init__(self, settings: ManagedRuntime) -> None:
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous: dict[str, str] = {}
        self._budget: AlertBudget | None = None
        self._slo_problem_logged = False
        self._rules: tuple[str, ...] = ()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="dipcatcher-alerts", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2)

    def tick(self) -> list[tuple[str, str]]:
        statuses = self._statuses()
        actions = plan_alert_actions(self._previous, statuses)
        budget = self._budget
        if budget is None:
            return []
        for rule, action in actions:
            self._apply(budget, rule, action, statuses.get(rule, ""))
        for rule, status in statuses.items():
            if status != "no_data":
                self._previous[rule] = status
        return actions

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except (OSError, ValueError, KeyError, TypeError):
                log_event("alert_tick_failed", level="error")
            self._stop.wait(self._settings.alert_interval_s)

    def _statuses(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        slos = self._slos()
        if slos:
            for result in evaluate(current_slo_snapshot(), slos):
                statuses[result.id] = result.status
        reason = acceptance_block()
        if reason != "draining":
            statuses["kill_switch"] = "pass" if reason is None else "fail"
        self._ensure_budget(tuple(statuses))
        return statuses

    def _slos(self) -> list[Slo]:
        raw = self._settings.slo_path
        if not raw:
            return []
        path = Path(raw)
        if path.is_symlink() or not path.is_file():
            self._note_slo_problem("missing")
            return []
        try:
            return load_slos(path)
        except (OSError, ValueError, KeyError, TypeError):
            self._note_slo_problem("invalid")
            return []

    def _note_slo_problem(self, status: str) -> None:
        if self._slo_problem_logged:
            return
        self._slo_problem_logged = True
        log_event("slo_unavailable", status=status)

    def _ensure_budget(self, keys: tuple[str, ...]) -> None:
        if self._budget is not None:
            return
        rules = list(dict.fromkeys((*keys, "kill_switch")))
        if not rules:
            rules = ["kill_switch"]
        self._rules = tuple(rules)
        self._budget = AlertBudget(1.0, rules)

    def _apply(self, budget: AlertBudget, rule: str, action: str, status: str) -> None:
        if rule not in budget.rules:
            return
        if action == "alarm":
            wager = budget.alarm(rule, 0.5)
            logged = "alarm" if wager > 0 else "frozen"
            log_event("alert", rule=rule, action=logged, status=status)
            return
        if action == "resolve" and budget.rules[rule].pending:
            budget.resolve(rule, True)
            log_event("alert", rule=rule, action="resolve", status=status)
