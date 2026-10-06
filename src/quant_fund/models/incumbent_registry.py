"""Registry of pretrained incumbent heads (foundation-model challengers).

One declarative table of every pretrained time-series incumbent the fleet
scores, with the facts that decide whether a claim about it is honest:

- which module/class implements the fleet contract (``fit`` /  ``predict`` /
  ``predict_from_history``), or that no adapter exists yet;
- what the *upstream* output contract is (a native quantile grid, sampled
  paths, a ready quantile row, or a labelled channel layout) — the thing an
  adapter can silently get wrong. The TimesFM channel-5 incident was exactly
  this class of bug (see ``scripts/splice_timesfm_fix.py``): upstream
  ``TimesFM_2p5`` ``full_forecast`` channels are ``[q50, q10, q20, q30, q40,
  point, q60, q70, q80, q90]``, and the legacy adapter scored ``q[:9]`` by
  *position* — dropping q90 and injecting the point forecast as a
  pseudo-quantile — with no error, handicapping TimesFM in every
  remote-produced shard until timesfm-only reruns were spliced in. Quantile
  levels must be mapped by *label*, never by column position;
- whether the upstream dependency resolves into ``uv.lock`` (so CI can run
  the real model) or is deliberately kept out (the adapter then fails closed
  and is exercised only against stubs).

The registry imports nothing heavy: classes are resolved lazily through
:func:`load_head_class`. ``tests/unit/models/test_incumbent_registry.py``
holds the per-incumbent output-contract tests (level fidelity, fail-closed
width/finite handling, window causality) and cross-checks this table against
``FLEET_HEAD_REGISTRY`` and ``uv.lock`` so it cannot drift.

Research-only. SYNTHETIC stub tests prove contract correctness, never market
evidence; ``not_implemented``, ``script_only`` and ``out_of_lock`` entries
must never be reported as scored fleet incumbents.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

OutputContract = Literal[
    "native_quantile_grid", "sample_paths", "quantile_row", "labelled_channels"
]
LockStatus = Literal["in_uv_lock", "out_of_lock"]
AdapterStatus = Literal["wired", "script_only", "not_implemented"]

# Upstream ``TimesFM_2p5`` ``full_forecast`` channel layout, by label. ``None``
# marks the point forecast, which is NOT a quantile. Source of truth for the
# layout is the corrected-contract note in ``scripts/splice_timesfm_fix.py``.
TIMESFM_2P5_CHANNELS: tuple[float | None, ...] = (
    0.5,
    0.1,
    0.2,
    0.3,
    0.4,
    None,
    0.6,
    0.7,
    0.8,
    0.9,
)


def labelled_channel_indices(
    taus: Sequence[float],
    layout: Sequence[float | None],
    *,
    tol: float = 1e-9,
) -> tuple[int, ...]:
    """Map each requested quantile level to its output channel *by label*.

    ``layout[j]`` is the quantile level carried by channel ``j`` (``None`` for
    a non-quantile channel such as a point forecast). A tau with no matching
    labelled channel raises — never approximated, interpolated or taken by
    position — so a requested level can only ever be served by the channel
    that actually carries it.
    """
    out: list[int] = []
    for t in taus:
        hits = [j for j, level in enumerate(layout) if level is not None and abs(level - t) <= tol]
        if len(hits) != 1:
            carried = sorted(level for level in layout if level is not None)
            raise ValueError(
                f"tau={t:g} is not carried by exactly one labelled channel "
                f"(native levels: {carried}); score on native levels only"
            )
        out.append(hits[0])
    return tuple(out)


@dataclass(frozen=True)
class IncumbentSpec:
    """Facts about one pretrained incumbent. Immutable evidence, not config."""

    key: str
    """Fleet head name (key into ``FLEET_HEAD_REGISTRY`` when wired)."""
    upstream: str
    """Upstream package / repository the weights come from."""
    weights_id: str | None
    output_contract: OutputContract | None
    """What the upstream predictor emits per window; ``None`` if unwired."""
    lock_status: LockStatus
    lock_dist: str | None
    """Distribution name that must be present in ``uv.lock`` iff in lock."""
    status: AdapterStatus
    module: str | None
    cls: str | None
    reason: str
    """Honest one-line status: why it is (not) runnable here."""

    @property
    def wired(self) -> bool:
        return self.status == "wired"


INCUMBENTS: tuple[IncumbentSpec, ...] = (
    IncumbentSpec(
        key="tirex2",
        upstream="tirex-2 (NX-AI)",
        weights_id="NX-AI/TiRex-2",
        output_contract="native_quantile_grid",
        lock_status="in_uv_lock",
        lock_dist="tirex-2",
        status="wired",
        module="quant_fund.models.tirex2",
        cls="Tirex2Distribution",
        reason=(
            "installable via the `nn` extra; emits a fixed native quantile grid, "
            "so requested taus must map to native columns by level, never by "
            "position (non-native taus fail closed)"
        ),
    ),
    IncumbentSpec(
        key="toto2",
        upstream="toto-2 (Datadog Toto-2.0)",
        weights_id=None,
        output_contract="quantile_row",
        lock_status="out_of_lock",
        lock_dist="toto-2",
        status="wired",
        module="quant_fund.models.toto2",
        cls="Toto2Distribution",
        reason=(
            "unresolvable in uv.lock: toto-2 requires gluonts[torch]>=0.16 whose "
            "toolz~=0.10 pin conflicts with pinned exchange-calendars "
            "(toolz>=1); adapter fails closed, exercised against stubs only"
        ),
    ),
    IncumbentSpec(
        key="moirai2",
        upstream="uni2ts (Salesforce) — Moirai-2.0",
        weights_id="Salesforce/moirai-2.0-R-small",
        output_contract="sample_paths",
        lock_status="out_of_lock",
        lock_dist="uni2ts",
        status="wired",
        module="quant_fund.models.moirai2",
        cls="Moirai2Distribution",
        reason=(
            "unresolvable in uv.lock: every uni2ts release pins scipy<1.12 and "
            "numpy~=1.26 against this repo's scipy>=1.14 / numpy>=2.0; adapter "
            "fails closed and its uni2ts call sequence is unverified against "
            "the real package"
        ),
    ),
    IncumbentSpec(
        key="sundial",
        upstream="Sundial (THU-ML)",
        weights_id="thuml/sundial-base-128m",
        output_contract="sample_paths",
        lock_status="out_of_lock",
        lock_dist="transformers",
        status="wired",
        module="quant_fund.models.sundial",
        cls="SundialDistribution",
        reason=(
            "upstream requires transformers==4.40.1 exactly and "
            "trust_remote_code=True (model-repo code executes on import); kept "
            "out of uv.lock, adapter fails closed"
        ),
    ),
    IncumbentSpec(
        key="tabpfn_ts",
        upstream="tabpfn-time-series (PriorLabs)",
        weights_id=None,
        output_contract="quantile_row",
        lock_status="out_of_lock",
        lock_dist="tabpfn-time-series",
        status="wired",
        module="quant_fund.models.tabpfn_ts",
        cls="TabpfnTsDistribution",
        reason=(
            "not in uv.lock (gluonts toolz<1 conflict); adapter fails closed, "
            "exercised against stubs only"
        ),
    ),
    IncumbentSpec(
        key="timesfm",
        upstream="timesfm (Google) — TimesFM 2.5 200M",
        weights_id="timesfm-2.5-200m-pytorch",
        output_contract="labelled_channels",
        lock_status="out_of_lock",
        lock_dist="timesfm",
        status="script_only",
        module=None,
        cls=None,
        reason=(
            "SCRIPT-ONLY: scored through scripts/sota_eval_native.py (point path, "
            "channel 5), not a fleet head. Not in uv.lock. full_forecast channels "
            "are labelled (TIMESFM_2P5_CHANNELS): channel 5 is the POINT forecast, "
            "not a quantile — quantile levels must be taken by label"
        ),
    ),
    IncumbentSpec(
        key="lag_llama",
        upstream="lag-llama (time-series-foundation-models)",
        weights_id="time-series-foundation-models/Lag-Llama",
        output_contract=None,
        lock_status="out_of_lock",
        lock_dist="lag-llama",
        status="not_implemented",
        module=None,
        cls=None,
        reason=(
            "NO ADAPTER. Not published on PyPI (git install only) and "
            "requirements.txt pins gluonts[torch]<=0.14.4; the gluonts 0.14.x "
            "line is the one the moirai2 adapter records as unresolvable against "
            "this lock, so it is expected (not verified) to be uv.lock-"
            "incompatible too. Must not be reported as a scored incumbent."
        ),
    ),
)


def incumbent_keys(*, wired_only: bool = False) -> tuple[str, ...]:
    """Keys of all registered incumbents (optionally only wired adapters)."""
    return tuple(s.key for s in INCUMBENTS if s.wired or not wired_only)


def get_incumbent(key: str) -> IncumbentSpec:
    """Look up one incumbent by key; unknown keys raise ``KeyError``."""
    for spec in INCUMBENTS:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown incumbent {key!r}; registered: {', '.join(incumbent_keys())}")


def load_head_class(key: str) -> Any:
    """Import and return the head class for a wired incumbent.

    Raises ``RuntimeError`` for an unwired incumbent so a ``not_implemented``
    entry can never be mistaken for a runnable head.
    """
    spec = get_incumbent(key)
    if not spec.wired or spec.module is None or spec.cls is None:
        raise RuntimeError(f"incumbent {key!r} has no adapter: {spec.reason}")
    return getattr(importlib.import_module(spec.module), spec.cls)
