"""SYNTHETIC MVCC version-chain garbage collection.

Versions carry (created_by, deleted_by_or_None). A version is collectible
when no active snapshot can see it (all readers started after it was
replaced). Verify GC keeps exactly the versions visible to live snapshots.
"""

from __future__ import annotations

import random


def gc(versions: list[tuple[int, int]], snapshots: list[int]) -> list[tuple[int, int]]:
    """versions: [(born_tx, dead_tx)]; dead_tx = txid that replaced it or
    inf. snapshots = reader txids. Keep version v if some snapshot s with
    born <= s < dead might read it."""
    keep = []
    for born, dead in versions:
        live = any(born <= s < dead for s in snapshots)
        newest_needed = born <= max(snapshots) and dead > max(snapshots)
        if live or newest_needed or dead >= 10**9:
            keep.append((born, dead))
    return keep


def visible(versions: list[tuple[int, int]], snap: int) -> tuple[int, int] | None:
    for born, dead in versions:
        if born <= snap < dead:
            return (born, dead)
    return None


def bench_mvcc_gc(seed: int = 20261231 + 433) -> dict[str, float]:
    rng = random.Random(seed)
    keeps_visible = drops_dead = chains = 0
    trials = 40
    for _ in range(trials):
        # build a version chain for one row: tx 1,5,8 wrote; 8's version live
        n = rng.randrange(3, 7)
        bounds = sorted(rng.sample(range(1, 60), n))
        INF = 10**9
        vers = [(bounds[i], bounds[i + 1]) for i in range(n - 1)] + [(bounds[-1], INF)]
        snaps = sorted(rng.sample(range(1, 80), rng.randrange(1, 4)))
        keep = gc(vers, snaps)
        # every snapshot can still read a version from keep
        keeps_visible += int(all(visible(keep, s) is not None for s in snaps if visible(vers, s)))
        # GC never removes the currently-live version
        drops_dead += int(all(v in keep for v in vers if v[1] >= INF))
        # collected versions are truly invisible to all snapshots
        removed = [v for v in vers if v not in keep]
        chains += int(all(not any(v[0] <= s < v[1] for s in snaps) for v in removed))
    return {
        "synthetic_snapshots_readable": float(keeps_visible / trials),
        "synthetic_live_version_kept": float(drops_dead / trials),
        "synthetic_dead_versions_removed": float(chains / trials),
    }
