"""Wave-85 optional scorecard families — Hansen (2006)
CMA-ES (mu/mu_w,lambda) with CSA step-size control,
streaming sketches (Dunning-Ertl 2019 t-digest,
Flajolet 2007 HyperLogLog, Cormode-Muthukrishnan 2005
Count-Min, Greenwald-Khanna 2001 quantiles), Schmidt
(1986) root-MUSIC + Roy-Kailath (1989) TLS-ESPRIT
subspace spectral estimation, chain-ladder reserving
(Mack 1993 variance + Bornhuetter-Ferguson 1972 +
England-Verrall 1999 ODP bootstrap), Erlang (1917)
queueing (B/C/A, Pollaczek-Khinchine, Allen-Cunneen
G/G/c, Jackson 1957 networks), and Atkinson-Shorrocks-
Foster inequality measurement (Gini/Theil/Atkinson/
FGT/Lorenz). Emitted only when the corresponding
module's `bench_*` self-check completes on its
SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_CMA_ES_SEED = 20261231 + 498
_SKETCHES_SEED = 20261231 + 499
_MUSIC_ESPRIT_SEED = 20261231 + 500
_CHAIN_LADDER_SEED = 20261231 + 501
_ERLANG_QUEUEING_SEED = 20261231 + 502
_INEQUALITY_INDICES_SEED = 20261231 + 503

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(raw: object) -> dict[str, float]:
    """Float-coerce a lane blob, dropping non-numeric entries and
    any key carrying a forbidden headline metric token."""
    if not isinstance(raw, dict):
        return {}
    return _finite_blob(
        {
            k: float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and _FORBIDDEN.isdisjoint(k.lower().split("_"))
        }
    )


def bench_cma_es() -> dict[str, float]:
    try:
        from quant_fund.models.cma_es import bench_cma_es as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CMA_ES_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_sketches() -> dict[str, float]:
    try:
        from quant_fund.models.sketches import bench_sketches as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SKETCHES_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_music_esprit() -> dict[str, float]:
    try:
        from quant_fund.models.music_esprit import (
            bench_music_esprit as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MUSIC_ESPRIT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_chain_ladder() -> dict[str, float]:
    try:
        from quant_fund.models.chain_ladder import bench_chain_ladder as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CHAIN_LADDER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_erlang_queueing() -> dict[str, float]:
    try:
        from quant_fund.models.erlang_queueing import (
            bench_erlang_queueing as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ERLANG_QUEUEING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_inequality_indices() -> dict[str, float]:
    try:
        from quant_fund.models.inequality_indices import (
            bench_inequality_indices as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_INEQUALITY_INDICES_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
