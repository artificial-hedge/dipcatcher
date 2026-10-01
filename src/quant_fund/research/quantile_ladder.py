"""Anytime-valid calibration audit across the full quantile vector.

The sibling lanes audit one coverage claim at a time: ``coverage_watch``
asks "does the 90% band cover 90%", ``tail_watch`` asks about one nested
tail cell, ``calibration_eprocess`` bets on PIT uniformity. A
distributional forecaster can pass all of them while being wrong — e.g.
a shape whose 90% band covers exactly 90% because a too-wide mid region
and too-thin tails cancel: every individual quantile level is
miscalibrated, but any single marginal audit sees nominal.

``QuantileLadder`` audits the whole vector at once. At every level
``tau`` the breach indicator ``b_t = 1{y_t <= q_tau,t}`` is a Bernoulli
stream whose null rate is exactly ``tau`` (the PIT coverage event —
pinball's gradient channel). A per-level e-process tracks it, and the
ladder merges the family two ways, reported as **separate claims**:

- ``merged_evalue``: the arithmetic mean of the per-level e-values is
  itself an e-value under ARBITRARY cross-level dependence (Vovk & Wang
  2020, "Combining e-values") — and the levels are always dependent,
  sharing every realization. The merged claim alarms at ``1/alpha``.
- ``alarmed_levels``: the Bonferroni alternative — a level alarms when
  its own e-value ever reaches ``n_levels/alpha``. "Any level alarmed"
  is a level-alpha family claim by the union bound.

They are never folded into one ``alarmed`` flag: that would silently
double the delivered level to ``2*alpha``.

The per-level bet ports the corrected ``LossEProcess`` sign-channel
design (magnitude bets break validity under skewed streams). The bet is
on the centered indicator ``w_t = (b_t - tau)/s_tau``,
``s_tau = max(tau, 1 - tau)`` — under the calibrated null
``E[w_t | F_{t-1}] = 0`` exactly, so ``e_t = 1 + lam_t * w_t`` is an
e-factor for ANY predictable stake in ``(-1, 1)``; it lies in
``[1 - lam, 1 + lam]``. The stake is the exact Kelly plug-in for this
asymmetric payoff, fitted from strict history:

    lam_t = clip((p_hat_tau - tau) * s_tau / (tau (1 - tau)), -lam, lam)

which at the symmetric level ``tau = 0.5`` reduces to
``clip(2*p_hat - 1, -lam, lam)`` — the ``LossEProcess`` ``[0, lam]``
sign bet plus its mirror. The symmetric clip is deliberate: a level that
over-covers (breach rate below tau — a vacuous claim dressed as rigor)
is a broken calibration claim exactly as an under-covering level is, so
both directions must compound evidence. The plug-in center is the null
rate tau, not 0.5 — a breach-rate excursion at ``tau = 0.05`` is
miscalibration at 0.10, not at 0.50.

``p_hat`` is Laplace-smoothed toward the null rate:
``(n_breach + tau) / (n + 1)`` — one pseudo-observation at rate tau, so
the stake starts at 0 and must be earned from data.

Fail closed throughout: a missing/non-finite realization or quantile,
a non-monotone quantile vector, or a level absent from the update map
raises — a malformed vector is inconclusive, never evidence.

Receipt: ``quantile_ladder.v1``, sealed by ``write_ladder_receipt`` and
contract-checked by ``_quantile_ladder_errors`` (dispatched from
``receipt_v2._verify_v1``).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUANTILE_LADDER_SCHEMA = "quantile_ladder.v1"
QUANTILE_LADDER_KINDS = frozenset({"quantile_ladder", QUANTILE_LADDER_SCHEMA})
DEFAULT_LEVELS: tuple[float, ...] = tuple(i / 20 for i in range(1, 20))


def _levels_or_raise(levels: Iterable[float]) -> tuple[float, ...]:
    out = tuple(float(t) for t in levels)
    if not out:
        raise ValueError("levels must be nonempty")
    if any(not math.isfinite(t) or not (0.0 < t < 1.0) for t in out):
        raise ValueError("levels must lie in (0, 1)")
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("levels must be strictly increasing")
    return out


@dataclass(frozen=True)
class LadderState:
    """Snapshot of the ladder after one evaluation row.

    ``merged_alarmed`` is the current merged claim (merged e-value at or
    above ``1/alpha`` this step); ``alarmed_levels`` is the ever-crossed
    Bonferroni family — a level stays alarmed once crossed. Two separate
    claims, never one flag.
    """

    origin: int
    merged_evalue: float
    merged_alarmed: bool
    alarmed_levels: tuple[float, ...]
    newly_alarmed: tuple[float, ...]


@dataclass
class _LevelWatch:
    """One level's e-process on its breach stream (see module docstring)."""

    tau: float
    lam: float
    scale: float
    n: int = 0
    n_breach: int = 0
    log_e: float = 0.0
    alarmed: bool = False
    alarm_origin: int | None = None

    def p_hat(self) -> float:
        """Smoothed breach rate, prior centered at the null rate tau."""
        return (self.n_breach + self.tau) / (self.n + 1.0)

    def _predictable_lam(self) -> float:
        """Kelly plug-in on the centered-breach channel; in [-lam, lam]."""
        edge = self.p_hat() - self.tau
        kelly = edge * self.scale / (self.tau * (1.0 - self.tau))
        return min(max(kelly, -self.lam), self.lam)

    def update(self, breach: bool) -> float:
        lam_t = self._predictable_lam()
        w = (float(breach) - self.tau) / self.scale
        e_factor = 1.0 + lam_t * w
        # lam_t in [-lam, lam] and |w| <= 1 keep e_factor in [1-lam, 1+lam].
        self.log_e += math.log(e_factor)
        self.n += 1
        self.n_breach += int(breach)
        return self.evalue

    @property
    def evalue(self) -> float:
        return float(math.exp(min(self.log_e, 700.0)))


@dataclass
class QuantileLadder:
    """Per-level breach e-processes merged over the quantile vector.

    ``update(realized, quantiles)`` folds one evaluation row: the
    realized value and the forecaster's quantile map keyed by level.
    The map must cover every declared level with finite values and be
    nondecreasing in level — anything else raises (fail closed).
    """

    levels: tuple[float, ...] = DEFAULT_LEVELS
    alpha: float = 0.05
    lam: float = 0.5
    _watches: dict[float, _LevelWatch] = field(default_factory=dict, init=False)
    _n: int = field(default=0, init=False)
    _merged_alarm_origin: int | None = field(default=None, init=False)
    _states: list[LadderState] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.levels = _levels_or_raise(self.levels)
        if not (math.isfinite(self.alpha) and 0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not (math.isfinite(self.lam) and 0.0 < self.lam < 1.0):
            raise ValueError("lam must lie in (0, 1)")
        self._watches = {
            tau: _LevelWatch(tau=tau, lam=self.lam, scale=max(tau, 1.0 - tau))
            for tau in self.levels
        }

    @staticmethod
    def _resolve_quantiles(
        quantiles: Mapping[float, float], levels: tuple[float, ...]
    ) -> list[float]:
        """Coerce the quantile map and return values in declared-level order."""
        try:
            q_map = {float(k): float(v) for k, v in quantiles.items()}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"quantiles keys/values must be numeric: {exc}") from exc
        try:
            ordered = [q_map[tau] for tau in levels]
        except KeyError as exc:
            raise ValueError(f"quantiles missing declared level {exc.args[0]}") from exc
        if any(not math.isfinite(q) for q in ordered):
            raise ValueError("quantiles must be finite")
        if any(b < a for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("quantile vector must be nondecreasing in level")
        return ordered

    def update(self, realized: float, quantiles: Mapping[float, float]) -> LadderState:
        """Fold one (realized, quantile-vector) row; return the new state."""
        try:
            y = float(realized)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"realized must be numeric: {realized!r}") from exc
        if not math.isfinite(y):
            raise ValueError("realized must be finite")
        if not isinstance(quantiles, Mapping):
            raise ValueError("quantiles must be a level -> quantile mapping")
        ordered = self._resolve_quantiles(quantiles, self.levels)
        t = self._n
        family_threshold = len(self.levels) / self.alpha
        newly: list[float] = []
        evalues: list[float] = []
        for tau, q in zip(self.levels, ordered, strict=True):
            watch = self._watches[tau]
            e = watch.update(y <= q)
            evalues.append(e)
            if not watch.alarmed and e >= family_threshold:
                watch.alarmed = True
                watch.alarm_origin = t
                newly.append(tau)
        merged = sum(evalues) / len(evalues)
        merged_alarmed = merged >= 1.0 / self.alpha
        if merged_alarmed and self._merged_alarm_origin is None:
            self._merged_alarm_origin = t
        state = LadderState(
            origin=t,
            merged_evalue=float(merged),
            merged_alarmed=bool(merged_alarmed),
            alarmed_levels=tuple(sorted(tau for tau, w in self._watches.items() if w.alarmed)),
            newly_alarmed=tuple(newly),
        )
        self._states.append(state)
        self._n = t + 1
        return state

    @property
    def n_origins(self) -> int:
        return self._n

    @property
    def level_evalues(self) -> dict[float, float]:
        return {tau: w.evalue for tau, w in self._watches.items()}

    @property
    def merged_evalue(self) -> float:
        evalues = self.level_evalues
        return sum(evalues.values()) / len(evalues)

    @property
    def alarmed_levels(self) -> tuple[float, ...]:
        """Levels whose e-value ever reached the Bonferroni threshold."""
        return tuple(sorted(tau for tau, w in self._watches.items() if w.alarmed))

    @property
    def alarm_origins(self) -> dict[float, int]:
        return {
            tau: w.alarm_origin for tau, w in self._watches.items() if w.alarm_origin is not None
        }

    @property
    def merged_alarm_origin(self) -> int | None:
        return self._merged_alarm_origin

    @property
    def merged_alarmed(self) -> bool:
        return self.merged_evalue >= 1.0 / self.alpha

    @property
    def states(self) -> list[LadderState]:
        return list(self._states)


def _level_key(tau: float) -> str:
    """repr gives an exact float round-trip for the receipt's level keys."""
    return repr(float(tau))


def quantile_ladder_report(
    rows: Iterable[tuple[float, Mapping[float, float]]],
    *,
    levels: Iterable[float] = DEFAULT_LEVELS,
    alpha: float = 0.05,
    lam: float = 0.5,
    data_label: str = "UNKNOWN",
) -> dict[str, Any]:
    """Receipt-shaped ladder verdict over an evaluation stream.

    ``rows`` yields ``(realized, quantiles)`` pairs — the realized value
    and the forecaster's quantile map for that origin. Fails closed on an
    empty stream. ``data_label`` stamps provenance; bare evaluation rows
    carry none, so the default is ``UNKNOWN`` (callers must opt into
    ``SYNTHETIC``/real labels).
    """
    if not isinstance(data_label, str) or not data_label.strip():
        raise ValueError("data_label must be a nonempty string")
    ladder = QuantileLadder(levels=_levels_or_raise(levels), alpha=alpha, lam=lam)
    n = 0
    for realized, quantiles in rows:
        ladder.update(realized, quantiles)
        n += 1
    if n == 0:
        raise ValueError("evaluation stream must be nonempty")

    per_level = {
        _level_key(tau): {
            "e": ladder._watches[tau].evalue,
            "p_hat": ladder._watches[tau].p_hat(),
            "n": ladder._watches[tau].n,
            "alarmed": ladder._watches[tau].alarmed,
        }
        for tau in ladder.levels
    }
    merged = ladder.merged_evalue
    return {
        "kind": QUANTILE_LADDER_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": data_label,
        "alpha": float(alpha),
        "lam": float(lam),
        "levels": [float(t) for t in ladder.levels],
        "n_levels": len(ladder.levels),
        "n_origins": n,
        "per_level": per_level,
        "alarmed_levels": [float(t) for t in ladder.alarmed_levels],
        "alarm_origins": {
            _level_key(tau): int(origin) for tau, origin in ladder.alarm_origins.items()
        },
        "merged_evalue": float(merged),
        "merged_alarmed": bool(ladder.merged_alarmed),
        "merged_alarm_origin": ladder.merged_alarm_origin,
        "evidence": [
            "ville_inequality",
            "bernoulli_rate_per_level",
            "kelly_plugin_centered_breach",
            "mean_evalue_merger_arbitrary_dependence",
            "bonferroni_per_level_family",
            "separate_family_and_merged_claims",
            "anytime_valid",
        ],
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish a receipt without exposing partial bytes or replacing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"receipt path is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"receipt already exists with different content: {path}")
        return
    temporary_path: Path | None = None
    try:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"receipt already exists with different content: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()


def write_ladder_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal a quantile_ladder receipt into ``quantile_ladder_<hash>.json``.

    Filename digest = sha256 of the canonical payload, embedded as
    ``receipt_sha256`` (fleet_eval seal convention). Atomic and immutable:
    an existing file with different content raises.
    """
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    if (
        receipt.get("kind") != QUANTILE_LADDER_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("per_level"), Mapping)
        or not isinstance(receipt.get("alarmed_levels"), list)
        or not isinstance(receipt.get("merged_alarmed"), bool)
    ):
        raise ValueError("quantile_ladder receipt violates its contract")
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    path = Path(receipts_dir) / f"quantile_ladder_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _num(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _finite(value: object) -> bool:
    v = _num(value)
    return v is not None and math.isfinite(v)


def _parse_level_key(key: object) -> float | None:
    if not isinstance(key, str):
        return None
    try:
        return float(key)
    except ValueError:
        return None


def _quantile_ladder_errors(payload: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``quantile_ladder.v1`` receipt.

    Re-derives the merged-claim arithmetic from the sealed per-level map:
    the merged e-value must equal the mean of the level e-values (and in
    any case stay positive and within ``n_levels * max(e_level)`` — the
    dependence-robust merger bound), the merged alarm must sit at or
    above ``1/alpha`` when raised, and the Bonferroni family members must
    be declared levels with in-range alarm origins.
    """
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    data_label = payload.get("data_label")
    if not isinstance(data_label, str) or not data_label.strip():
        errors.append("data_label_not_nonempty_str")
    alpha = _num(payload.get("alpha"))
    if alpha is None or not (0.0 < alpha < 1.0):
        errors.append("alpha_out_of_unit_interval")
        alpha = None
    lam = _num(payload.get("lam"))
    if lam is None or not (0.0 < lam < 1.0):
        errors.append("lam_out_of_unit_interval")
    levels_raw = payload.get("levels")
    declared: set[float] = set()
    if not isinstance(levels_raw, list) or not levels_raw:
        errors.append("levels_not_nonempty_list")
        levels_raw = []
    else:
        parsed = [_num(t) for t in levels_raw]
        ordered = [t for t in parsed if t is not None]
        if len(ordered) != len(parsed) or any(not (0.0 < t < 1.0) for t in ordered):
            errors.append("levels_out_of_unit_interval")
        else:
            if any(b <= a for a, b in zip(ordered, ordered[1:], strict=False)):
                errors.append("levels_not_strictly_increasing")
            declared = set(ordered)
    n_levels = payload.get("n_levels")
    if not isinstance(n_levels, int) or isinstance(n_levels, bool) or n_levels < 1:
        errors.append("n_levels_not_positive_int")
        n_levels = None
    elif declared and n_levels != len(declared):
        errors.append("n_levels_mismatches_levels")
    n_origins = payload.get("n_origins")
    if not isinstance(n_origins, int) or isinstance(n_origins, bool) or n_origins < 1:
        errors.append("n_origins_not_positive_int")
        n_origins = None

    per_level = payload.get("per_level")
    if not isinstance(per_level, Mapping) or not per_level:
        errors.append("per_level_not_nonempty_mapping")
        per_level = {}
    parsed_keys: set[float] = set()
    evalues: list[float] = []
    for key, entry in per_level.items():
        tau = _parse_level_key(key)
        if tau is None or (declared and tau not in declared):
            errors.append(f"per_level_key_not_declared:{key}")
            continue
        parsed_keys.add(tau)
        if not isinstance(entry, Mapping):
            errors.append(f"per_level_entry_not_mapping:{key}")
            continue
        e = _num(entry.get("e"))
        if e is None or e <= 0.0 or not math.isfinite(e):
            errors.append(f"per_level_e_not_positive_finite:{key}")
        else:
            evalues.append(e)
        p_hat = _num(entry.get("p_hat"))
        if p_hat is None or not (0.0 <= p_hat <= 1.0):
            errors.append(f"per_level_p_hat_out_of_bounds:{key}")
        n_entry = entry.get("n")
        if not isinstance(n_entry, int) or isinstance(n_entry, bool) or n_entry < 1:
            errors.append(f"per_level_n_not_positive_int:{key}")
        elif n_origins is not None and n_entry != n_origins:
            errors.append(f"per_level_n_not_n_origins:{key}")
        if not isinstance(entry.get("alarmed"), bool):
            errors.append(f"per_level_alarmed_not_bool:{key}")
    if declared and parsed_keys != declared:
        errors.append("per_level_keys_not_all_declared_levels")

    merged = _num(payload.get("merged_evalue"))
    if merged is None or merged <= 0.0 or not math.isfinite(merged):
        errors.append("merged_evalue_not_positive_finite")
    elif evalues and len(evalues) == len(per_level):
        max_e = max(evalues)
        mean_e = sum(evalues) / len(evalues)
        bound = (n_levels if n_levels is not None else len(evalues)) * max_e
        tolerance = 1e-9 * max(1.0, bound)
        if merged > bound + tolerance:
            errors.append("merged_evalue_exceeds_merger_bound")
        elif not math.isclose(merged, mean_e, rel_tol=1e-9, abs_tol=1e-9):
            errors.append("merged_evalue_not_mean_of_levels")

    merged_alarmed = payload.get("merged_alarmed")
    if not isinstance(merged_alarmed, bool):
        errors.append("merged_alarmed_not_bool")
    elif merged_alarmed and alpha is not None and merged is not None and merged < 1.0 / alpha:
        errors.append("merged_alarmed_below_threshold")

    alarmed_levels = payload.get("alarmed_levels")
    alarmed_set: set[float] = set()
    if not isinstance(alarmed_levels, list):
        errors.append("alarmed_levels_not_list")
    else:
        for value in alarmed_levels:
            tau = _num(value)
            if tau is None or tau not in declared:
                errors.append(f"alarmed_level_not_declared:{value}")
            else:
                alarmed_set.add(tau)
    alarm_origins = payload.get("alarm_origins")
    origin_set: set[float] = set()
    if not isinstance(alarm_origins, Mapping):
        errors.append("alarm_origins_not_mapping")
    else:
        for key, origin in alarm_origins.items():
            tau = _parse_level_key(key)
            if tau is None or (declared and tau not in declared):
                errors.append(f"alarm_origin_key_not_declared:{key}")
                continue
            origin_set.add(tau)
            if (
                not isinstance(origin, int)
                or isinstance(origin, bool)
                or origin < 0
                or (n_origins is not None and origin >= n_origins)
            ):
                errors.append(f"alarm_origin_out_of_range:{key}")
    if alarmed_set != origin_set:
        errors.append("alarmed_levels_mismatch_alarm_origins")

    merged_origin = payload.get("merged_alarm_origin")
    if merged_origin is not None and (
        not isinstance(merged_origin, int)
        or isinstance(merged_origin, bool)
        or merged_origin < 0
        or (n_origins is not None and merged_origin >= n_origins)
    ):
        errors.append("merged_alarm_origin_out_of_range")
    if merged_alarmed is True and merged_origin is None:
        errors.append("merged_alarmed_without_origin")

    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or "anytime_valid" not in evidence:
        errors.append("evidence_missing_anytime_valid")
    return errors


__all__ = [
    "DEFAULT_LEVELS",
    "LadderState",
    "QUANTILE_LADDER_KINDS",
    "QUANTILE_LADDER_SCHEMA",
    "QuantileLadder",
    "quantile_ladder_report",
    "write_ladder_receipt",
]
