"""Classical CPU simulated annealing for a disclosed binary portfolio QUBO.

Primary method references (no borrowed code or device implementation):
Kirkpatrick, Gelatt & Vecchi (1983), https://doi.org/10.1126/science.220.4598.671;
Metropolis et al. (1953), https://doi.org/10.1063/1.1699114 and
https://www.osti.gov/biblio/4390578;
Glover, Kochenberger & Du, https://arxiv.org/abs/1811.11538.

For binary x and exact cardinality k, weights are w=(budget/k)x. The minimized
objective f(x)=risk_aversion*w'Cov*w - expected_return'w is an optimization
criterion, not a proper forecasting score or realized performance evidence.
The QUBO is f(x)+P(sum(x)-k)^2. For every binary x, |f(x)|<=B, with
B=sum(abs(Q_base)). P>2B guarantees every global QUBO minimizer has cardinality k:
an infeasible energy is >=P-B>B, while every feasible energy is <=B.
This mathematical penalty does not guarantee finite annealing finds a feasible
state or optimum. All trials, including failed/infeasible endpoints, are retained.

``qubo_bit_flip`` really anneals the unconstrained penalized QUBO with symmetric
single-bit proposals, Metropolis acceptance and geometric cooling. The separate
``cardinality_swap`` variant restricts proposals to the feasible cardinality
slice; it is identified as constrained classical annealing, not unconstrained
QUBO search. Neither method is quantum annealing or evidence of quantum advantage.

Exact-small enumeration proves only exhaustion of the finite float64-scored
cardinality set; it runs only within both asset and combination limits. Greedy
has the same objective and constraints, but is not an optimality certificate.
Runtime is measured locally and workload counts are reported; budgets are not
claimed compute-matched. Caller source hashes and synthetic declarations require
independent audit. JSON receipts bind data/clocks/config/code and replay solver
behavior, but hash integrity does not establish source authenticity or entitlements.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Bits = tuple[int, ...]
Mode = Literal["qubo_bit_flip", "cardinality_swap"]
TrialStatus = Literal["FEASIBLE_FOUND", "NO_FEASIBLE_FOUND", "INVALID_SWAP_INITIAL_STATE"]
BaselineStatus = Literal[
    "GREEDY_COMPLETE", "EXHAUSTIVE_FLOAT64", "SKIPPED_ASSET_LIMIT", "SKIPPED_COMBINATION_LIMIT"
]
_MAX_ASSETS = 128
_MAX_PROPOSALS = 2_000_000
_MAX_EXACT_COMBINATIONS = 250_000
_SCHEMA = "classical_qubo_selection.v1"


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _implementation_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _name(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a nonempty bounded string")


def _sha(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA256")


def _number(value: float, field: str, *, bound: float = 1e12) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or abs(value) > bound
    ):
        raise ValueError(f"{field} must be finite with absolute value <={bound}")


def _count(value: int, field: str, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low},{high}]")


def _clock(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _assets(values: tuple[str, ...]) -> None:
    if (
        not isinstance(values, tuple)
        or not 1 <= len(values) <= _MAX_ASSETS
        or any(not isinstance(value, str) for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("assets must be a unique bounded immutable tuple")
    for value in values:
        _name(value, "asset")


def _bits(values: Bits, n: int) -> None:
    if (
        not isinstance(values, tuple)
        or len(values) != n
        or any(type(v) is not int or v not in (0, 1) for v in values)
    ):
        raise ValueError("selection must be an immutable length-n tuple of integer binary bits")


def _horizon(value: timedelta) -> None:
    if not isinstance(value, timedelta) or not timedelta(0) < value <= timedelta(days=365):
        raise ValueError("horizon must be positive and no greater than 365 days")


@dataclass(frozen=True, slots=True)
class ForecastEvidence:
    """Expected simple returns over horizon, with a declared input cutoff/publication."""

    assets: tuple[str, ...]
    expected_returns: tuple[float, ...]
    horizon: timedelta
    asof: datetime
    available_time: datetime
    source_id: str
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        _assets(self.assets)
        if not isinstance(self.expected_returns, tuple) or len(self.expected_returns) != len(
            self.assets
        ):
            raise ValueError("expected returns must be an immutable aligned tuple")
        for value in self.expected_returns:
            _number(value, "expected simple return", bound=1e6)
            if value < -1:
                raise ValueError("expected simple returns cannot be below -1")
        _horizon(self.horizon)
        asof, available = (
            _clock(self.asof, "forecast asof"),
            _clock(self.available_time, "forecast availability"),
        )
        if asof > available:
            raise ValueError("forecast asof must precede publication")
        object.__setattr__(self, "asof", asof)
        object.__setattr__(self, "available_time", available)
        _name(self.source_id, "forecast source_id")
        _sha(self.source_sha256, "forecast source_sha256")
        if not isinstance(self.synthetic, bool):
            raise ValueError("forecast synthetic declaration must be boolean")


@dataclass(frozen=True, slots=True)
class CovarianceEvidence:
    assets: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    horizon: timedelta
    asof: datetime
    available_time: datetime
    source_id: str
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        _assets(self.assets)
        n = len(self.assets)
        if (
            not isinstance(self.values, tuple)
            or len(self.values) != n
            or any(not isinstance(row, tuple) or len(row) != n for row in self.values)
        ):
            raise ValueError("covariance must be an immutable aligned square tuple")
        for row in self.values:
            for value in row:
                _number(value, "covariance", bound=1e6)
        matrix = np.array(self.values, dtype=float)
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        if float(np.linalg.eigvalsh(matrix).min()) < -1e-12 or np.any(np.diag(matrix) < 0):
            raise ValueError("covariance must be positive semidefinite")
        # Symmetrize within the declared tolerance so the actual QUBO stays symmetric.
        object.__setattr__(self, "values", _tuple_matrix((matrix + matrix.T) / 2))
        _horizon(self.horizon)
        asof, available = (
            _clock(self.asof, "covariance asof"),
            _clock(self.available_time, "covariance availability"),
        )
        if asof > available:
            raise ValueError("covariance asof must precede publication")
        object.__setattr__(self, "asof", asof)
        object.__setattr__(self, "available_time", available)
        _name(self.source_id, "covariance source_id")
        _sha(self.source_sha256, "covariance source_sha256")
        if not isinstance(self.synthetic, bool):
            raise ValueError("covariance synthetic declaration must be boolean")


@dataclass(frozen=True, slots=True)
class SelectionProblem:
    forecast: ForecastEvidence
    covariance: CovarianceEvidence
    decision_time: datetime
    cardinality: int
    risk_aversion: float = 1.0
    budget: float = 1.0
    problem_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.forecast, ForecastEvidence) or not isinstance(
            self.covariance, CovarianceEvidence
        ):
            raise ValueError("typed forecast and covariance evidence required")
        if (
            self.forecast.assets != self.covariance.assets
            or self.forecast.horizon != self.covariance.horizon
        ):
            raise ValueError("forecast/covariance asset ordering and horizon must match")
        decision = _clock(self.decision_time, "decision_time")
        if max(self.forecast.available_time, self.covariance.available_time) > decision:
            raise ValueError("input evidence unpublished at decision cutoff")
        object.__setattr__(self, "decision_time", decision)
        _count(self.cardinality, "cardinality", 1, len(self.forecast.assets))
        _number(self.risk_aversion, "risk_aversion", bound=1e6)
        _number(self.budget, "budget", bound=1)
        if self.risk_aversion < 1e-9 or not 1e-9 <= self.budget <= 1:
            raise ValueError("risk_aversion and budget must be >=1e-9, budget<=1")
        if float(np.abs(_base_matrix(self)).sum()) > 1e12:
            raise ValueError("objective magnitude exceeds numerical resource bound")
        object.__setattr__(self, "problem_sha256", _hash(_problem_payload(self)))

    @property
    def assets(self) -> tuple[str, ...]:
        return self.forecast.assets

    @property
    def synthetic(self) -> bool:
        return self.forecast.synthetic or self.covariance.synthetic

    def objective(self, bits: Bits, *, require_feasible: bool = True) -> float:
        _bits(bits, len(self.assets))
        if require_feasible and sum(bits) != self.cardinality:
            raise ValueError("objective requires exact cardinality")
        x = np.array(bits, dtype=float)
        return float(x @ _base_matrix(self) @ x)

    def weights(self, bits: Bits) -> tuple[float, ...]:
        _bits(bits, len(self.assets))
        if sum(bits) != self.cardinality:
            raise ValueError("research weights require exact cardinality")
        return tuple(self.budget / self.cardinality * value for value in bits)


def _tuple_matrix(matrix: Array) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in matrix)


def _base_matrix(problem: SelectionProblem) -> Array:
    scale = problem.budget / problem.cardinality
    return problem.risk_aversion * scale**2 * np.array(problem.covariance.values) - np.diag(
        scale * np.array(problem.forecast.expected_returns)
    )


def _problem_payload(problem: SelectionProblem) -> dict[str, Any]:
    def evidence(value: ForecastEvidence | CovarianceEvidence) -> dict[str, Any]:
        payload = asdict(value)
        payload["horizon"] = value.horizon.total_seconds()
        payload["asof"] = value.asof.isoformat()
        payload["available_time"] = value.available_time.isoformat()
        return payload

    return {
        "forecast": evidence(problem.forecast),
        "covariance": evidence(problem.covariance),
        "decision_time": problem.decision_time.isoformat(),
        "cardinality": problem.cardinality,
        "risk_aversion": problem.risk_aversion,
        "budget": problem.budget,
    }


@dataclass(frozen=True, slots=True)
class QUBOModel:
    problem: SelectionProblem
    penalty: float | None = None
    matrix: tuple[tuple[float, ...], ...] = dataclass_field(init=False)
    offset: float = dataclass_field(init=False)
    objective_absolute_bound: float = dataclass_field(init=False)
    model_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, SelectionProblem):
            raise ValueError("typed selection problem required")
        base = _base_matrix(self.problem)
        bound = float(np.abs(base).sum())
        penalty = self.penalty
        if penalty is None:
            penalty = 2 * bound + max(1.0, bound * 1e-6)
        _number(penalty, "cardinality penalty", bound=3e12)
        if penalty <= 2 * bound + 1e-12 * max(1.0, bound):
            raise ValueError("penalty must exceed the proven 2B bound with numeric margin")
        n, k = len(self.problem.assets), self.problem.cardinality
        matrix = base + penalty * np.ones((n, n)) - 2 * penalty * k * np.eye(n)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "matrix", _tuple_matrix(matrix))
        object.__setattr__(self, "offset", float(penalty * k**2))
        object.__setattr__(self, "objective_absolute_bound", bound)
        object.__setattr__(
            self,
            "model_sha256",
            _hash(
                {
                    "problem_sha256": self.problem.problem_sha256,
                    "penalty": penalty,
                    "matrix": self.matrix,
                    "offset": self.offset,
                    "objective_absolute_bound": bound,
                    "implementation_sha256": _implementation_hash(),
                }
            ),
        )

    def array(self) -> Array:
        """Return an independently mutable copy of immutable QUBO coefficients."""
        return np.array(self.matrix, dtype=float)

    def energy(self, bits: Bits) -> float:
        _bits(bits, len(self.problem.assets))
        # This algebraically identical evaluation avoids cancellation of P*k**2.
        assert self.penalty is not None
        return (
            self.problem.objective(bits, require_feasible=False)
            + self.penalty * (sum(bits) - self.problem.cardinality) ** 2
        )


@dataclass(frozen=True, slots=True)
class AnnealConfig:
    seeds: tuple[int, ...] = (11, 23, 47, 83)
    sweeps: int = 100
    initial_temperature: float = 10.0
    final_temperature: float = 0.001
    mode: Mode = "qubo_bit_flip"
    max_proposals: int = 250_000
    exact_max_assets: int = 20
    exact_max_combinations: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, tuple) or not 1 <= len(self.seeds) <= 32:
            raise ValueError("seeds must be a unique bounded immutable tuple")
        for seed in self.seeds:
            _count(seed, "seed", 0, 2**32 - 1)
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        _count(self.sweeps, "sweeps", 1, 10_000)
        _count(self.max_proposals, "max_proposals", 1, _MAX_PROPOSALS)
        _count(self.exact_max_assets, "exact_max_assets", 1, 20)
        _count(self.exact_max_combinations, "exact_max_combinations", 1, _MAX_EXACT_COMBINATIONS)
        for name in ("initial_temperature", "final_temperature"):
            _number(getattr(self, name), name, bound=1e15)
        if not 1e-12 <= self.final_temperature <= self.initial_temperature:
            raise ValueError("temperatures must be positive and nonincreasing")
        if self.mode not in ("qubo_bit_flip", "cardinality_swap"):
            raise ValueError("unknown classical annealing mode")

    @property
    def config_sha256(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True)
class AnnealTrial:
    seed: int
    initial_bits: Bits
    final_bits: Bits
    best_energy_bits: Bits
    best_feasible_bits: Bits | None
    initial_energy: float
    final_energy: float
    best_energy: float
    best_feasible_objective: float | None
    proposals: int
    accepted: int
    accepted_uphill: int
    feasible_visits: int
    sweep_energy: tuple[float, ...]
    elapsed_seconds: float
    status: TrialStatus
    failure_reason: str | None

    def __post_init__(self) -> None:
        _count(self.seed, "trial seed", 0, 2**32 - 1)
        n = len(self.initial_bits)
        _count(n, "trial assets", 1, _MAX_ASSETS)
        for bits in (self.initial_bits, self.final_bits, self.best_energy_bits):
            _bits(bits, n)
        if self.best_feasible_bits is not None:
            _bits(self.best_feasible_bits, n)
        if (self.best_feasible_bits is None) != (self.best_feasible_objective is None):
            raise ValueError("feasible bits/objective must be present together")
        for value in (self.initial_energy, self.final_energy, self.best_energy):
            _number(value, "trial energy", bound=1e18)
        if self.best_feasible_objective is not None:
            _number(self.best_feasible_objective, "feasible objective")
        for name in ("proposals", "accepted", "accepted_uphill", "feasible_visits"):
            _count(getattr(self, name), name, 0, _MAX_PROPOSALS + 1)
        if not self.accepted_uphill <= self.accepted <= self.proposals:
            raise ValueError("trial acceptance counts are inconsistent")
        if not isinstance(self.sweep_energy, tuple) or not 1 <= len(self.sweep_energy) <= 10001:
            raise ValueError("sweep trace must be a bounded immutable tuple")
        for value in self.sweep_energy:
            _number(value, "sweep energy", bound=1e18)
        _number(self.elapsed_seconds, "elapsed_seconds")
        if self.elapsed_seconds < 0:
            raise ValueError("runtime must be nonnegative")
        if self.status not in ("FEASIBLE_FOUND", "NO_FEASIBLE_FOUND", "INVALID_SWAP_INITIAL_STATE"):
            raise ValueError("unknown trial status")
        if (self.status == "FEASIBLE_FOUND") != (self.best_feasible_bits is not None):
            raise ValueError("trial status/feasible output mismatch")
        if self.status == "FEASIBLE_FOUND":
            if self.failure_reason is not None:
                raise ValueError("successful trial cannot have a failure reason")
        else:
            if self.failure_reason is None:
                raise ValueError("failed trial requires a failure reason")
            _name(self.failure_reason, "failure_reason")


@dataclass(frozen=True, slots=True)
class BaselineResult:
    method: Literal["greedy", "exact_enumeration"]
    status: BaselineStatus
    bits: Bits | None
    objective: float | None
    evaluated_candidates: int
    feasible_candidate_count: int
    elapsed_seconds: float
    optimality_proven: bool

    def __post_init__(self) -> None:
        if self.method not in ("greedy", "exact_enumeration") or self.status not in (
            "GREEDY_COMPLETE",
            "EXHAUSTIVE_FLOAT64",
            "SKIPPED_ASSET_LIMIT",
            "SKIPPED_COMBINATION_LIMIT",
        ):
            raise ValueError("unknown baseline identity/status")
        _count(self.evaluated_candidates, "evaluated_candidates", 0, _MAX_EXACT_COMBINATIONS)
        # Candidate count can be huge for skipped enumerations but has a hard n<=128 bound.
        _count(self.feasible_candidate_count, "feasible_candidate_count", 1, 2**128)
        _number(self.elapsed_seconds, "baseline runtime")
        if self.elapsed_seconds < 0 or not isinstance(self.optimality_proven, bool):
            raise ValueError("baseline runtime/proof identity invalid")
        if (self.bits is None) != (self.objective is None):
            raise ValueError("baseline bits/objective must be present together")
        if self.bits is not None:
            _bits(self.bits, len(self.bits))
            if self.objective is None:
                raise ValueError("completed baseline requires an objective")
            _number(self.objective, "baseline objective")
        exhaustive = self.method == "exact_enumeration" and self.status == "EXHAUSTIVE_FLOAT64"
        if self.optimality_proven != exhaustive or (
            exhaustive and self.evaluated_candidates != self.feasible_candidate_count
        ):
            raise ValueError("only exhaustive completed enumeration can prove optimality")
        skipped = self.status.startswith("SKIPPED")
        if skipped != (self.bits is None) or (skipped and self.evaluated_candidates != 0):
            raise ValueError("skipped enumeration cannot fabricate a solution")
        if self.method == "greedy" and self.status != "GREEDY_COMPLETE":
            raise ValueError("greedy status mismatch")


def greedy_selection(problem: SelectionProblem) -> BaselineResult:
    """Deterministic forward greedy selection under the exact same cardinality."""
    started = time.perf_counter()
    bits = [0] * len(problem.assets)
    count = 0
    for _ in range(problem.cardinality):
        candidates = []
        for i, selected in enumerate(bits):
            if selected == 0:
                proposal = bits.copy()
                proposal[i] = 1
                candidates.append((problem.objective(tuple(proposal), require_feasible=False), i))
                count += 1
        _, i = min(candidates)
        bits[i] = 1
    return BaselineResult(
        "greedy",
        "GREEDY_COMPLETE",
        tuple(bits),
        problem.objective(tuple(bits)),
        count,
        math.comb(len(bits), problem.cardinality),
        time.perf_counter() - started,
        False,
    )


def exact_selection(
    problem: SelectionProblem, *, max_assets: int = 20, max_combinations: int = 100_000
) -> BaselineResult:
    """Exhaustively enumerate every k-subset only within both hard limits."""
    _count(max_assets, "exact max_assets", 1, 20)
    _count(max_combinations, "exact max_combinations", 1, _MAX_EXACT_COMBINATIONS)
    n = len(problem.assets)
    candidates = math.comb(n, problem.cardinality)
    if n > max_assets or candidates > max_combinations:
        status: BaselineStatus = (
            "SKIPPED_ASSET_LIMIT" if n > max_assets else "SKIPPED_COMBINATION_LIMIT"
        )
        return BaselineResult("exact_enumeration", status, None, None, 0, candidates, 0.0, False)
    started = time.perf_counter()
    best_bits, best_objective = None, math.inf
    count = 0
    for selected in itertools.combinations(range(n), problem.cardinality):
        bits = tuple(int(i in selected) for i in range(n))
        objective = problem.objective(bits)
        count += 1
        if objective < best_objective:
            best_bits, best_objective = bits, objective
    return BaselineResult(
        "exact_enumeration",
        "EXHAUSTIVE_FLOAT64",
        best_bits,
        best_objective,
        count,
        candidates,
        time.perf_counter() - started,
        True,
    )


def _default_initial(model: QUBOModel, config: AnnealConfig, seed: int) -> Bits:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    n, k = len(model.problem.assets), model.problem.cardinality
    if config.mode == "cardinality_swap":
        selected = set(rng.choice(n, size=k, replace=False).tolist())
        return tuple(int(i in selected) for i in range(n))
    return tuple(int(v) for v in rng.integers(0, 2, size=n))


def _anneal_trial(model: QUBOModel, config: AnnealConfig, seed: int, initial: Bits) -> AnnealTrial:
    started = time.perf_counter()
    problem, n = model.problem, len(model.problem.assets)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    base = _base_matrix(problem)
    assert model.penalty is not None
    bits = initial
    energy = model.energy(bits)
    initial_energy = energy
    best_bits, best_energy = bits, energy
    feasible = bits if sum(bits) == problem.cardinality else None
    feasible_objective = problem.objective(bits) if feasible is not None else None
    visits = int(feasible is not None)
    accepted = uphill = proposals = 0
    trace = [energy]
    if config.mode == "cardinality_swap" and feasible is None:
        return AnnealTrial(
            seed,
            initial,
            bits,
            best_bits,
            None,
            initial_energy,
            energy,
            best_energy,
            None,
            0,
            0,
            0,
            0,
            tuple(trace),
            time.perf_counter() - started,
            "INVALID_SWAP_INITIAL_STATE",
            "supplied_swap_state_has_wrong_cardinality",
        )
    for sweep in range(config.sweeps):
        fraction = sweep / max(1, config.sweeps - 1)
        temperature = math.exp(
            (1 - fraction) * math.log(config.initial_temperature)
            + fraction * math.log(config.final_temperature)
        )
        for _ in range(n):
            proposal = list(bits)
            if config.mode == "qubo_bit_flip":
                i = int(rng.integers(n))
                proposal[i] = 1 - proposal[i]
            else:
                selected = [i for i, v in enumerate(bits) if v]
                unselected = [i for i, v in enumerate(bits) if not v]
                if not unselected:
                    break  # k=n has exactly one feasible state.
                i, j = int(rng.choice(selected)), int(rng.choice(unselected))
                proposal[i], proposal[j] = 0, 1
            candidate = tuple(proposal)
            x = np.array(candidate, dtype=float)
            candidate_objective = float(x @ base @ x)
            candidate_energy = (
                candidate_objective + model.penalty * (sum(candidate) - problem.cardinality) ** 2
            )
            delta = candidate_energy - energy
            proposals += 1
            if delta <= 0 or float(rng.random()) < math.exp(-delta / temperature):
                bits, energy = candidate, candidate_energy
                accepted += 1
                uphill += int(delta > 0)
                if energy < best_energy:
                    best_bits, best_energy = bits, energy
                if sum(bits) == problem.cardinality:
                    visits += 1
                    objective = candidate_objective
                    if feasible_objective is None or objective < feasible_objective:
                        feasible, feasible_objective = bits, objective
        trace.append(energy)
    status: TrialStatus = "FEASIBLE_FOUND" if feasible is not None else "NO_FEASIBLE_FOUND"
    return AnnealTrial(
        seed,
        initial,
        bits,
        best_bits,
        feasible,
        initial_energy,
        energy,
        best_energy,
        feasible_objective,
        proposals,
        accepted,
        uphill,
        visits,
        tuple(trace),
        time.perf_counter() - started,
        status,
        None if feasible is not None else "finite_proposal_budget_found_no_feasible_state",
    )


@dataclass(frozen=True, slots=True)
class AnnealReport:
    model: QUBOModel
    config: AnnealConfig
    trials: tuple[AnnealTrial, ...]
    greedy: BaselineResult
    exact: BaselineResult
    chosen_bits: Bits | None
    implementation_sha256: str
    receipt_sha256: str = dataclass_field(init=False)
    deterministic_sha256: str = dataclass_field(init=False)
    research_only: bool = dataclass_field(init=False, default=True)
    market_evidence: bool = dataclass_field(init=False, default=False)
    quantum_device: bool = dataclass_field(init=False, default=False)
    quantum_advantage: bool = dataclass_field(init=False, default=False)

    def __post_init__(self) -> None:
        _sha(self.implementation_sha256, "implementation_sha256")
        if not isinstance(self.trials, tuple) or len(self.trials) != len(self.config.seeds):
            raise ValueError("all configured trial outcomes must be retained")
        n, k = len(self.model.problem.assets), self.model.problem.cardinality
        if len(self.config.seeds) * self.config.sweeps * n > self.config.max_proposals:
            raise ValueError("reported annealing exceeds proposal resource budget")
        for trial, seed in zip(self.trials, self.config.seeds, strict=True):
            if not isinstance(trial, AnnealTrial) or trial.seed != seed:
                raise ValueError("trial ordering/seed identity mismatch")
            for bits in (trial.initial_bits, trial.final_bits, trial.best_energy_bits):
                _bits(bits, n)
            if trial.best_feasible_bits is not None:
                _bits(trial.best_feasible_bits, n)
                if sum(trial.best_feasible_bits) != k or trial.best_feasible_objective != (
                    self.model.problem.objective(trial.best_feasible_bits)
                ):
                    raise ValueError("trial feasible objective/cardinality mismatch")
            if (
                trial.initial_energy != self.model.energy(trial.initial_bits)
                or trial.final_energy != self.model.energy(trial.final_bits)
                or trial.best_energy != self.model.energy(trial.best_energy_bits)
            ):
                raise ValueError("trial energy/state mismatch")
        available = [t for t in self.trials if t.best_feasible_bits is not None]
        best = (
            min(
                available,
                key=lambda t: (
                    t.best_feasible_objective if t.best_feasible_objective is not None else math.inf
                ),
            )
            if available
            else None
        )
        expected = best.best_feasible_bits if best is not None else None
        if self.chosen_bits != expected:
            raise ValueError("reported anneal selection must retain the actual best feasible trial")
        if self.greedy.method != "greedy" or self.exact.method != "exact_enumeration":
            raise ValueError("reported baseline method identity mismatch")
        if self.exact.optimality_proven and (
            n > self.config.exact_max_assets or math.comb(n, k) > self.config.exact_max_combinations
        ):
            raise ValueError("exact proof exceeds configured enumeration limits")
        for baseline in (self.greedy, self.exact):
            if baseline.bits is not None and (
                len(baseline.bits) != n
                or sum(baseline.bits) != k
                or baseline.objective != self.model.problem.objective(baseline.bits)
            ):
                raise ValueError("baseline objective/cardinality mismatch")
            if baseline.feasible_candidate_count != math.comb(n, k):
                raise ValueError("baseline candidate-count mismatch")
        body = self.payload()
        object.__setattr__(self, "receipt_sha256", _hash(body))
        object.__setattr__(self, "deterministic_sha256", _hash(_without_runtime(body)))

    @property
    def weights(self) -> tuple[float, ...] | None:
        return (
            self.model.problem.weights(self.chosen_bits) if self.chosen_bits is not None else None
        )

    @property
    def cash_weight(self) -> float | None:
        return 1.0 - self.model.problem.budget if self.chosen_bits is not None else None

    def diagnostics(self) -> dict[str, Any]:
        problem = self.model.problem
        objective = problem.objective(self.chosen_bits) if self.chosen_bits is not None else None
        reference = self.exact.objective
        return {
            "anneal_status": "FEASIBLE_FOUND"
            if self.chosen_bits is not None
            else "NO_FEASIBLE_SOLUTION",
            "anneal_objective": objective,
            "anneal_objective_gap": objective - reference
            if objective is not None and reference is not None
            else None,
            "greedy_objective_gap": self.greedy.objective - reference
            if self.greedy.objective is not None and reference is not None
            else None,
            "exact_optimality_proven": self.exact.optimality_proven,
            "feasible_trials": sum(t.best_feasible_bits is not None for t in self.trials),
            "failed_trials": sum(t.best_feasible_bits is None for t in self.trials),
            "infeasible_final_trials": sum(
                sum(t.final_bits) != problem.cardinality for t in self.trials
            ),
            "proposal_count": sum(t.proposals for t in self.trials),
            "accepted_uphill": sum(t.accepted_uphill for t in self.trials),
            "anneal_elapsed_seconds": sum(t.elapsed_seconds for t in self.trials),
            "greedy_elapsed_seconds": self.greedy.elapsed_seconds,
            "exact_elapsed_seconds": self.exact.elapsed_seconds,
            "constraints_matched": True,
            "compute_budget_matched": False,
            "objective_is_proper_score": False,
            "forecast_quality_evaluated": False,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "problem": _problem_payload(self.model.problem),
            "problem_sha256": self.model.problem.problem_sha256,
            "qubo": {
                "matrix": self.model.matrix,
                "offset": self.model.offset,
                "penalty": self.model.penalty,
                "objective_absolute_bound": self.model.objective_absolute_bound,
                "model_sha256": self.model.model_sha256,
            },
            "config": asdict(self.config),
            "config_sha256": self.config.config_sha256,
            "trials": [asdict(t) for t in self.trials],
            "greedy": asdict(self.greedy),
            "exact": asdict(self.exact),
            "chosen_bits": self.chosen_bits,
            "weights": self.weights,
            "cash_weight": self.cash_weight,
            "objective_definition": "risk_aversion*(budget/k)^2*x'Cov*x-(budget/k)*mu'x",
            "cardinality_constraint": "binary_x_and_sum_x_equals_k",
            "penalty_bound_definition": "B=sum(abs(Q_base)); P>2B",
            "diagnostics": self.diagnostics(),
            "implementation_sha256": self.implementation_sha256,
            "arithmetic": "numpy_float64",
            "numpy_version": np.__version__,
            "rng_algorithm": "numpy_PCG64_SeedSequence_seed_and_stream",
            "temperature_schedule": "geometric_in_sweeps",
            "synthetic": self.model.problem.synthetic,
            "data_label": "SYNTHETIC"
            if self.model.problem.forecast.synthetic and self.model.problem.covariance.synthetic
            else "MIXED"
            if self.model.problem.synthetic
            else "SUPPLIED_NON_SYNTHETIC",
            "research_only": True,
            "market_evidence": False,
            "quantum_device": False,
            "quantum_advantage": False,
            "method": "classical_unconstrained_qubo_simulated_annealing"
            if self.config.mode == "qubo_bit_flip"
            else "classical_cardinality_constrained_swap_annealing",
        }


def _without_runtime(body: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(body)
    for trial in payload["trials"]:
        trial.pop("elapsed_seconds")
    for name in ("greedy", "exact"):
        payload[name].pop("elapsed_seconds")
    for name in ("anneal_elapsed_seconds", "greedy_elapsed_seconds", "exact_elapsed_seconds"):
        payload["diagnostics"].pop(name)
    return payload


def solve_selection(
    problem: SelectionProblem,
    config: AnnealConfig | None = None,
    *,
    initial_states: tuple[Bits, ...] | None = None,
    penalty: float | None = None,
) -> AnnealReport:
    if not isinstance(problem, SelectionProblem):
        raise ValueError("typed selection problem required")
    config = AnnealConfig() if config is None else config
    if not isinstance(config, AnnealConfig):
        raise ValueError("typed AnnealConfig required")
    n = len(problem.assets)
    if len(config.seeds) * config.sweeps * n > config.max_proposals:
        raise ValueError("configured annealing proposal resource budget exceeded")
    model = QUBOModel(problem, penalty)
    if initial_states is None:
        initial_states = tuple(_default_initial(model, config, seed) for seed in config.seeds)
    if not isinstance(initial_states, tuple) or len(initial_states) != len(config.seeds):
        raise ValueError("one immutable initial state per configured seed required")
    for initial in initial_states:
        _bits(initial, n)
    trials = tuple(
        _anneal_trial(model, config, seed, initial)
        for seed, initial in zip(config.seeds, initial_states, strict=True)
    )
    available = [t for t in trials if t.best_feasible_bits is not None]
    best = (
        min(
            available,
            key=lambda t: (
                t.best_feasible_objective if t.best_feasible_objective is not None else math.inf
            ),
        )
        if available
        else None
    )
    return AnnealReport(
        model,
        config,
        trials,
        greedy_selection(problem),
        exact_selection(
            problem,
            max_assets=config.exact_max_assets,
            max_combinations=config.exact_max_combinations,
        ),
        best.best_feasible_bits if best is not None else None,
        _implementation_hash(),
    )


def save_receipt(report: AnnealReport, path: Path) -> None:
    """Write once; existing receipts are never overwritten. No pickle/code execution."""
    payload = {
        "body": report.payload(),
        "receipt_sha256": report.receipt_sha256,
        "deterministic_sha256": report.deterministic_sha256,
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)


def _keys(payload: dict[str, Any], expected: set[str]) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("receipt field schema mismatch")


def _restore_problem(payload: dict[str, Any]) -> SelectionProblem:
    _keys(
        payload,
        {"forecast", "covariance", "decision_time", "cardinality", "risk_aversion", "budget"},
    )
    forecast, covariance = dict(payload["forecast"]), dict(payload["covariance"])
    for data, kind in ((forecast, ForecastEvidence), (covariance, CovarianceEvidence)):
        _keys(data, {f.name for f in fields(kind)})
        data["assets"] = tuple(data["assets"])
        data["horizon"] = timedelta(seconds=data["horizon"])
        for name in ("asof", "available_time"):
            data[name] = datetime.fromisoformat(data[name])
    forecast["expected_returns"] = tuple(forecast["expected_returns"])
    covariance["values"] = tuple(tuple(row) for row in covariance["values"])
    return SelectionProblem(
        ForecastEvidence(**forecast),
        CovarianceEvidence(**covariance),
        datetime.fromisoformat(payload["decision_time"]),
        payload["cardinality"],
        payload["risk_aversion"],
        payload["budget"],
    )


def restore_receipt(path: Path) -> AnnealReport:
    """Validate schema/hashes and replay all deterministic solver/baseline outcomes.

    Local timing fields are retained and checked for finite nonnegative values;
    replay does not authenticate the original runtime or external data provenance.
    """
    if path.stat().st_size > 8_000_000:
        raise ValueError("receipt exceeds resource bound")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _keys(payload, {"body", "receipt_sha256", "deterministic_sha256"})
        _sha(payload["receipt_sha256"], "receipt_sha256")
        _sha(payload["deterministic_sha256"], "deterministic_sha256")
        body = payload["body"]
        if (
            _hash(body) != payload["receipt_sha256"]
            or _hash(_without_runtime(body)) != payload["deterministic_sha256"]
        ):
            raise ValueError("receipt content hash mismatch")
        if body["schema"] != _SCHEMA or body["implementation_sha256"] != _implementation_hash():
            raise ValueError("receipt schema/implementation mismatch")
        problem = _restore_problem(body["problem"])
        config_payload = dict(body["config"])
        _keys(config_payload, {f.name for f in fields(AnnealConfig)})
        config_payload["seeds"] = tuple(config_payload["seeds"])
        config = AnnealConfig(**config_payload)
        if not isinstance(body["trials"], list) or len(body["trials"]) != len(config.seeds):
            raise ValueError("receipt must retain exactly the configured trial count")
        trials = []
        for data in body["trials"]:
            data = dict(data)
            _keys(data, {f.name for f in fields(AnnealTrial)})
            for name in ("initial_bits", "final_bits", "best_energy_bits", "sweep_energy"):
                data[name] = tuple(data[name])
            if data["best_feasible_bits"] is not None:
                data["best_feasible_bits"] = tuple(data["best_feasible_bits"])
            trials.append(AnnealTrial(**data))
        baselines = []
        for name in ("greedy", "exact"):
            data = dict(body[name])
            _keys(data, {f.name for f in fields(BaselineResult)})
            if data["bits"] is not None:
                data["bits"] = tuple(data["bits"])
            baselines.append(BaselineResult(**data))
        report = AnnealReport(
            QUBOModel(problem, body["qubo"]["penalty"]),
            config,
            tuple(trials),
            baselines[0],
            baselines[1],
            tuple(body["chosen_bits"]) if body["chosen_bits"] is not None else None,
            body["implementation_sha256"],
        )
        if _hash(report.payload()) != payload["receipt_sha256"]:
            raise ValueError("receipt derived fields or honesty identity mismatch")
        replayed = solve_selection(
            problem,
            config,
            initial_states=tuple(t.initial_bits for t in report.trials),
            penalty=report.model.penalty,
        )
        if replayed.deterministic_sha256 != report.deterministic_sha256:
            raise ValueError("receipt solver replay mismatch")
        return report
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid receipt schema") from exc
