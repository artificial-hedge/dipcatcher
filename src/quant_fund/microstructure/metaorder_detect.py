"""metaorder_detect — recover metaorders from the fill stream.

Split parent orders arrive as bursts of same-sign children separated
by short gaps. A detector on exec data: merge consecutive same-sign
fills with inter-fill gap <= gap_s into one metaorder candidate.

On the sim we know ground truth (SplitFlow's `_remaining` flag marks
burst-child execs), so the detector's precision/recall are measurable.
On real tape it reports discovered structure — metaorder count, size,
and duration distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, EXECUTION_HIDDEN, parse_messages
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def cluster_metaorders(
    times: list[float], signs: list[int], *, gap_s: float = 1.0, min_len: int = 3
) -> list[dict[str, Any]]:
    """Group same-sign runs whose internal gaps are all <= gap_s.

    A run breaks on sign flip OR a gap > gap_s; runs shorter than
    min_len are singletons/noise, not metaorders.
    """
    meta: list[dict[str, Any]] = []
    if not times:
        return meta
    start = 0
    for i in range(1, len(times)):
        if signs[i] != signs[i - 1] or (times[i] - times[i - 1]) > gap_s:
            if i - start >= min_len:
                meta.append(
                    {
                        "sign": signs[start],
                        "n_fills": i - start,
                        "i_start": start,
                        "t_start": times[start],
                        "t_end": times[i - 1],
                    }
                )
            start = i
    if len(times) - start >= min_len:
        meta.append(
            {
                "sign": signs[start],
                "n_fills": len(times) - start,
                "i_start": start,
                "t_start": times[start],
                "t_end": times[-1],
            }
        )
    return meta


def _meta_stats(meta: list[dict[str, Any]]) -> dict[str, Any]:
    if not meta:
        return {"n_metaorders": 0}
    sizes = np.asarray([m["n_fills"] for m in meta], dtype=float)
    durs = np.asarray([m["t_end"] - m["t_start"] for m in meta], dtype=float)
    return {
        "n_metaorders": int(len(meta)),
        "mean_fills_per_meta": round(float(sizes.mean()), 2),
        "p95_fills_per_meta": round(float(np.percentile(sizes, 95)), 2),
        "max_fills_per_meta": int(sizes.max()),
        "mean_duration_s": round(float(durs.mean()), 3),
        "buy_share": round(float(np.mean([1.0 if m["sign"] > 0 else 0.0 for m in meta])), 4),
    }


def lobster_metaorders(
    tape_dir: Path, ticker: str = "AMZN", *, gap_s: float = 1.0, min_len: int = 3
) -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    times: list[float] = []
    signs: list[int] = []
    for ev in parse_messages(msg):
        if ev.event_type in (EXECUTION, EXECUTION_HIDDEN):
            times.append(ev.time_s)
            signs.append(-ev.direction)
    meta = cluster_metaorders(times, signs, gap_s=gap_s, min_len=min_len)
    n_clustered = sum(m["n_fills"] for m in meta)
    out = _meta_stats(meta)
    out["n_execs"] = len(times)
    out["clustered_fill_share"] = round(n_clustered / max(1, len(times)), 4)
    return out


def sim_metaorders(
    *,
    horizon: int = 20000,
    seed: int = 7,
    gap_s: float = 1.0,
    min_len: int = 3,
    p_start: float = 0.10,
) -> dict[str, Any]:
    """Detector precision/recall against SplitFlow's ground truth.

    `flow._remaining > 0` sampled before each step marks burst-active
    windows; an exec in such a step is a true metaorder child.
    """
    flow = SplitFlow(
        p_start=p_start,
        size_tail=1.2,
        k_min=10,
        k_max=600,
        intensity_mult=3.0,
        seed=seed + 2,
    )
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    times: list[float] = []
    signs: list[int] = []
    true_child: list[bool] = []
    for _ in range(horizon):
        burst_active = flow._remaining > 0
        n_tr = len(sim.trades)
        sim.step()
        for tr in sim.trades[n_tr:]:
            times.append(tr.t)
            signs.append(1 if tr.aggressor == "buy" else -1)
            true_child.append(burst_active)
    meta = cluster_metaorders(times, signs, gap_s=gap_s, min_len=min_len)
    clustered_flags: list[bool] = [False] * len(times)
    for m in meta:
        for j in range(m["i_start"], min(m["i_start"] + m["n_fills"], len(times))):
            clustered_flags[j] = True
    n_true = sum(true_child)
    tp = sum(1 for f, t in zip(clustered_flags, true_child, strict=True) if f and t)
    n_clustered = sum(clustered_flags)
    precision = tp / n_clustered if n_clustered else None
    recall = tp / n_true if n_true else None
    out = _meta_stats(meta)
    out.update(
        {
            "n_execs": len(times),
            "n_true_children": n_true,
            "clustered_fill_share": round(n_clustered / max(1, len(times)), 4),
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
        }
    )
    return out


def metaorder_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Metaorder structure: detector scores on sim + real distribution. Sealed."""
    real = lobster_metaorders(tape_dir, ticker)
    sim = sim_metaorders(seed=seed)
    divergences = []
    if real.get("clustered_fill_share") is not None and sim.get("clustered_fill_share") is not None:
        gap = real["clustered_fill_share"] - sim["clustered_fill_share"]
        if abs(gap) > 0.15:
            divergences.append(f"clustered_fill_share_gap_{gap:+.2f}")
    payload: dict[str, Any] = {
        "kind": "metaorder_detect",
        "schema": "metaorder_detect.v1",
        "ticker": ticker,
        "detector": {"gap_s": 1.0, "min_len": 3},
        "real": real,
        "sim": sim,
        "divergences": divergences,
        "claim": "metaorder_structure_detected",
        "interpretation": (
            "Gap-clustered same-sign runs proxy metaorders. On sim the "
            "detector's precision/recall are ground-truth-scored against "
            "SplitFlow's burst flags; on the tape the recovered count "
            "and size distribution are discovered structure."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
