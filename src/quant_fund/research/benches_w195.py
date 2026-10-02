"""Wave-126 adapters: exec-summary ensemble-adaptive MCMC canon — pcn_sampler,
emcee_stretch, rjmcmc, de_mcmc, dram, indep_mh —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.de_mcmc import bench_de_mcmc
from quant_fund.models.dram import bench_dram
from quant_fund.models.emcee_stretch import bench_emcee_stretch
from quant_fund.models.indep_mh import bench_indep_mh
from quant_fund.models.pcn_sampler import bench_pcn_sampler
from quant_fund.models.rjmcmc import bench_rjmcmc

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_pcn_sampler_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pcn_sampler", bench_pcn_sampler(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pcn_sampler bench failed: {exc}") from exc


def bench_emcee_stretch_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("emcee_stretch", bench_emcee_stretch(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"emcee_stretch bench failed: {exc}") from exc


def bench_rjmcmc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rjmcmc", bench_rjmcmc(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rjmcmc bench failed: {exc}") from exc


def bench_de_mcmc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("de_mcmc", bench_de_mcmc(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"de_mcmc bench failed: {exc}") from exc


def bench_dram_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dram", bench_dram(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dram bench failed: {exc}") from exc


def bench_indep_mh_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("indep_mh", bench_indep_mh(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"indep_mh bench failed: {exc}") from exc
