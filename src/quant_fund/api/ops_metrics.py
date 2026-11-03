"""Scrape-time readiness, drain, and kill gauges for the lab process.

``observe`` does not import ``api``. The managed server registers this sampler
before it binds, and clears it when the process stops. Each ``/metrics``
scrape reads the kill file again. The label is the stable block code.
"""

from __future__ import annotations

from quant_fund.api.lifecycle import acceptance_block, is_draining
from quant_fund.observe.metrics import register_ops_sampler
from quant_fund.observe.write_halt import write_halt_reason
from quant_fund.utils.operator_kill import orders_blocked_by_env


def sample_operator_gauges() -> tuple[float, float, float, str]:
    """Return ready, draining, kill-blocking, and the kill code."""
    block = orders_blocked_by_env()
    durable = write_halt_reason()
    ready = 0.0 if acceptance_block() is not None else 1.0
    draining = 1.0 if is_draining() else 0.0
    if durable is not None:
        return ready, draining, 1.0, f"durable_{durable}"
    if block is None:
        return ready, draining, 0.0, "enabled"
    return ready, draining, 1.0, block


def install_ops_sampler() -> None:
    """Publish lab operator gauges on the next Prometheus exposition."""
    register_ops_sampler(sample_operator_gauges)
