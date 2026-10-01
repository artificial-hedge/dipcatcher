"""SYNTHETIC clipped-PPO maker on the existing unit-lot FIFO ZI book.

Schulman et al. (2017), https://arxiv.org/abs/1707.06347, Eqs. 7, 9, 11-12:
old-policy likelihood ratios, pessimistic clipped surrogate, squared value
error, entropy exploration and GAE. This small discrete actor/critic is an
original adaptation, not a reproduction of the robotics benchmarks or C51.

Orders interact with the existing synthetic FIFO engine. The wrapper knows
only the current book, own queues, and completed trades; it never exposes
future events or the simulator's hidden flow regime. Buys require cash;
sells require owned units. One resting unit per side, fees, quote limits and
an inventory cap are enforced. Unchanged quotes retain queue priority.
Episode completion means the event horizon ended, not that inventory was
liquidated. Terminal inventory is retained and marked at the last known mid.

Reward and spread/wealth diagnostics are simulator-internal learning signals,
not proper forecast scores, headline performance or market evidence. ZI
order flow is exogenously parametrized; its mechanical response to orders
does not establish realistic endogenous strategic competitors, latency,
hidden liquidity, financing, impact or queue calibration. The naive control
uses the same wrapper/seeds/constraints; policy-dependent event clocks mean
the realized tapes differ. C51's state/reward/financing are different and its
matched comparison is unavailable. No broker or live-trading interface exists.

Torch is optional and lazy, CPU float64, with local generators and no global
RNG/thread changes. Frozen JSON supports NumPy inference and exact seeded
action/ledger replay under bound source/dependency versions. Hashes prove
consistency, not original execution authenticity or empirical data rights.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    Side,
    ZILobConfig,
    ZILobSimulator,
)

type Array = NDArray[np.float64]
type Regime = Literal["stationary", "buy_bias", "sell_bias", "switching"]
_REGIMES = ("stationary", "buy_bias", "sell_bias", "switching")
ACTION_GRID: tuple[tuple[int, int] | None, ...] = (
    None,
    (0, 0),
    (-1, 0),
    (0, -1),
    (-1, -1),
    (1, 0),
    (0, 1),
)
N_FEATURES, N_ACTIONS = 13, len(ACTION_GRID)
_TAG = "SYNTHETIC_PPO_MAKER"
_SCHEMA = "synthetic_ppo_maker.v1"
_MAX_BYTES = 32_000_000
_SIDES: tuple[Side, ...] = ("buy", "sell")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _code_bindings() -> dict[str, str]:
    return {
        "policy_gradient_maker": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "zi_lob_simulator": hashlib.sha256(
            Path(inspect.getfile(ZILobSimulator)).read_bytes()
        ).hexdigest(),
    }


def _number(value: float, name: str, low: float, high: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite in [{low},{high}]")


def _count(value: int, name: str, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def _name(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("episode identifier must be bounded and nonempty")


@dataclass(frozen=True)
class MakerConfig:
    decisions: int = 32
    events_per_decision: int = 4
    inventory_cap: int = 8
    initial_inventory: int = 4
    target_inventory: int = 4
    initial_cash: float = 1000.0
    quote_floor: float = 1.0
    quote_ceiling: float = 1000.0
    maker_fee_bps: float = 0.5
    inventory_penalty: float = 0.01

    def __post_init__(self) -> None:
        for name, low, high in (
            ("decisions", 2, 256),
            ("events_per_decision", 1, 16),
            ("inventory_cap", 1, 64),
        ):
            _count(getattr(self, name), name, low, high)
        for name in ("initial_inventory", "target_inventory"):
            _count(getattr(self, name), name, 0, self.inventory_cap)
        _number(self.initial_cash, "initial_cash", 0, 1e6)
        _number(self.quote_floor, "quote_floor", 1e-6, 1e6)
        _number(self.quote_ceiling, "quote_ceiling", self.quote_floor, 1e6)
        _number(self.maker_fee_bps, "maker_fee_bps", 0, 100)
        _number(self.inventory_penalty, "inventory_penalty", 0, 10)


@dataclass(frozen=True)
class MakerScenario:
    episode_id: str
    book: ZILobConfig
    regime: Regime = "stationary"

    def __post_init__(self) -> None:
        _name(self.episode_id)
        if not isinstance(self.book, ZILobConfig) or self.regime not in _REGIMES:
            raise ValueError("typed synthetic book and known regime required")
        book = self.book
        for name, low, high in (
            ("s0", 1, 1e5),
            ("tick", 1e-6, 10),
            ("lam", 1e-5, 100),
            ("mu", 1e-5, 100),
            ("theta_cxl", 1e-5, 10),
            ("p_buy", 0, 1),
            ("density_exponent", 0, 4),
            ("ref_halflife", 0, 1e5),
        ):
            _number(getattr(book, name), name, low, high)
        for name, low, high in (
            ("band", 1, 16),
            ("init_levels", 1, 16),
            ("init_depth", 1, 64),
            ("seed", 0, 2**32 - 1),
        ):
            _count(getattr(book, name), name, low, high)

    @property
    def scenario_sha256(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class MakerObservation:
    episode_id: str
    index: int
    observed_time: float
    best_bid: float | None
    best_ask: float | None
    last_known_mid: float
    inventory: int
    cash: float
    features: tuple[float, ...]
    action_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        _name(self.episode_id)
        _count(self.index, "decision index", 0, 256)
        _number(self.observed_time, "observation clock", 0, 1e12)
        _number(self.last_known_mid, "known mid", 1e-9, 1e6)
        _count(self.inventory, "inventory", 0, 64)
        _number(self.cash, "cash", 0, 1e9)
        if not isinstance(self.features, tuple) or len(self.features) != N_FEATURES:
            raise ValueError("observation feature schema mismatch")
        for v in self.features:
            _number(v, "feature", -8, 8)
        if (
            not isinstance(self.action_mask, tuple)
            or len(self.action_mask) != N_ACTIONS
            or any(not isinstance(v, bool) for v in self.action_mask)
            or self.action_mask[0] is not True
        ):
            raise ValueError("action mask schema mismatch")


class MakerEnvironment:
    """Current-state wrapper; actual fills/cancellations come from ZILobSimulator."""

    def __init__(self, scenario: MakerScenario, config: MakerConfig) -> None:
        if not isinstance(scenario, MakerScenario) or not isinstance(config, MakerConfig):
            raise ValueError("typed scenario/config required")
        # A conservative walk bound keeps every engine price positive, including
        # prices of exogenous orders; this is not a claim about market limits.
        walk = (
            (config.decisions * config.events_per_decision + config.decisions)
            * (scenario.book.band + 2)
            + scenario.book.init_levels
            + 2
        )
        if scenario.book.s0 / scenario.book.tick <= walk:
            raise ValueError("book price/tick too small for bounded positive-price episode")
        flow = None
        if scenario.regime != "stationary":
            probabilities = {
                "buy_bias": (0.8, 0.8),
                "sell_bias": (0.2, 0.2),
                "switching": (0.8, 0.2),
            }[scenario.regime]
            flow = MarkovRegimeFlow(
                tuple(RegimeState(f"hidden{i}", 1.0, p) for i, p in enumerate(probabilities)),
                (0.9, 0.9) if scenario.regime == "switching" else (1.0, 1.0),
                seed=(scenario.book.seed + 104729) % 2**32,
            )
        self.scenario, self.config = scenario, config
        self.book = ZILobSimulator(scenario.book, flow=flow)
        self.inventory, self.cash = config.initial_inventory, config.initial_cash
        self.index, self._cursor = 0, 0
        self._mid = scenario.book.s0
        self._flow_balance = 0.0
        self._orders: dict[Side, tuple[int, int] | None] = {"buy": None, "sell": None}
        self._posted: dict[int, dict[str, Any]] = {}
        self.initial_wealth = self.cash + self.inventory * self._mid
        if self.initial_wealth <= 0:
            raise ValueError("positive initial cash/asset endowment required")

    @property
    def done(self) -> bool:
        return self.index == self.config.decisions

    def _levels(self, action: int) -> tuple[int | None, int | None] | None:
        pair = ACTION_GRID[action]
        if pair is None:
            return None, None
        bb, ba = self.book.best_bid_level, self.book.best_ask_level
        if bb is None or ba is None:
            return None
        bid, ask = bb - pair[0], ba + pair[1]
        bid_price, ask_price = self.book.level_to_price(bid), self.book.level_to_price(ask)
        bid_ok = self.inventory < self.config.inventory_cap and self.cash >= bid_price * (
            1 + self.config.maker_fee_bps / 10000
        )
        ask_ok = self.inventory > 0
        for price, enabled in ((bid_price, bid_ok), (ask_price, ask_ok)):
            if enabled and not self.config.quote_floor <= price <= self.config.quote_ceiling:
                return None
        if (bid_ok and bid >= ba) or (ask_ok and ask <= bb) or (bid_ok and ask_ok and bid >= ask):
            return None
        if not bid_ok and not ask_ok:
            return None
        return bid if bid_ok else None, ask if ask_ok else None

    def observation(self) -> MakerObservation:
        bb, ba = self.book.best_bid_level, self.book.best_ask_level
        if self.book.mid is not None:
            self._mid = self.book.mid
        two_sided = bb is not None and ba is not None
        bd = self.book.depth_at("buy", bb) if bb is not None else 0
        ad = self.book.depth_at("sell", ba) if ba is not None else 0
        b1 = self.book.depth_at("buy", bb - 1) if bb is not None else 0
        a1 = self.book.depth_at("sell", ba + 1) if ba is not None else 0
        opportunities = []
        for side in _SIDES:
            order = self._orders[side]
            ahead = self.book.queue_position(order[0]) if order is not None else None
            opportunities.append(0.0 if ahead is None else 1 / (1 + ahead))
        features = (
            min((ba - bb) / 8, 8.0) if two_sided and bb is not None and ba is not None else 0.0,
            min(bd / 32, 8.0),
            min(ad / 32, 8.0),
            min(b1 / 32, 8.0),
            min(a1 / 32, 8.0),
            (bd - ad) / max(bd + ad, 1),
            (self.inventory - self.config.target_inventory) / self.config.inventory_cap,
            min(self.cash / self.initial_wealth, 8.0),
            *opportunities,
            (self.config.decisions - self.index) / self.config.decisions,
            self._flow_balance,
            float(two_sided),
        )
        return MakerObservation(
            self.scenario.episode_id,
            self.index,
            self.book.t,
            self.book.best_bid,
            self.book.best_ask,
            self._mid,
            self.inventory,
            self.cash,
            tuple(features),
            tuple(self._levels(a) is not None for a in range(N_ACTIONS)),
        )

    def _post(self, side: Side, level: int | None, decision: MakerObservation) -> None:
        old = self._orders[side]
        if old is not None and (not self.book.order_alive(old[0]) or old[1] != level):
            self.book.cancel_order(old[0])
            self._orders[side] = None
        if level is not None and self._orders[side] is None:
            oid = self.book.submit_limit_order(side, self.book.level_to_price(level), _TAG)
            self._orders[side] = (oid, level)
            self._posted[oid] = {
                "decision_time": decision.observed_time,
                "decision_mid": decision.last_known_mid,
                "limit_price": self.book.level_to_price(level),
            }

    def step(self, action: int) -> dict[str, Any]:
        _count(action, "action", 0, N_ACTIONS - 1)
        if self.done:
            raise RuntimeError("episode is complete")
        before = self.observation()
        levels = self._levels(action)
        if levels is None:
            raise ValueError("action violates current quote/inventory/cash mask")
        wealth_before = self.cash + self.inventory * before.last_known_mid
        self._post("buy", levels[0], before)
        self._post("sell", levels[1], before)
        fills = []
        for _ in range(self.config.events_per_decision):
            self.book.step()
            while self._cursor < len(self.book.trades):
                trade = self.book.trades[self._cursor]
                self._cursor += 1
                self._flow_balance = 0.9 * self._flow_balance + 0.1 * (
                    1 if trade.aggressor == "buy" else -1
                )
                if trade.maker_tag != _TAG:
                    continue
                posted = self._posted[trade.maker_order_id]
                if (
                    trade.price <= 0
                    or trade.price != posted["limit_price"]
                    or trade.t <= posted["decision_time"]
                ):
                    raise RuntimeError("engine fill violates posted limit/causal clock")
                fee = trade.price * self.config.maker_fee_bps / 10000
                dq = 1 if trade.maker_side == "buy" else -1
                self.inventory += dq
                self.cash -= dq * trade.price + fee
                if not 0 <= self.inventory <= self.config.inventory_cap or self.cash < -1e-9:
                    raise RuntimeError("engine fill violates cash/inventory constraints")
                self.cash = max(self.cash, 0.0)
                fills.append(
                    {
                        **asdict(trade),
                        **posted,
                        "fee": fee,
                        "sim_internal_spread_capture": dq * (posted["decision_mid"] - trade.price),
                    }
                )
            for side in _SIDES:
                old = self._orders[side]
                if old is not None and not self.book.order_alive(old[0]):
                    self._orders[side] = None
        self.index += 1
        after = self.observation()
        wealth_delta = self.cash + self.inventory * after.last_known_mid - wealth_before
        penalty = (
            self.config.inventory_penalty
            * ((self.inventory - self.config.target_inventory) / self.config.inventory_cap) ** 2
        )
        result = {
            "observation": asdict(before),
            "action": action,
            "next_observation": asdict(after),
            "fills": fills,
            "sim_internal_wealth_delta": wealth_delta,
            "sim_internal_inventory_penalty": penalty,
            "sim_internal_reward": wealth_delta - penalty,
            "done": self.done,
            "event_counts": self.book.event_counts(),
        }
        if self.done:
            for side in _SIDES:
                self._post(side, None, after)
        return result


def generalized_advantages(
    rewards: Array, values: Array, terminated: tuple[bool, ...], *, gamma: float, lam: float
) -> tuple[Array, Array]:
    """GAE recursion; values has T+1 entries and true terminal values are ignored."""
    _number(gamma, "gamma", 0, 1)
    _number(lam, "lambda", 0, 1)
    r, v = np.asarray(rewards, dtype=float), np.asarray(values, dtype=float)
    if (
        r.ndim != 1
        or not 1 <= len(r) <= 8192
        or v.shape != (len(r) + 1,)
        or not np.isfinite(r).all()
        or not np.isfinite(v).all()
        or not isinstance(terminated, tuple)
        or len(terminated) != len(r)
        or any(not isinstance(t, bool) for t in terminated)
    ):
        raise ValueError("GAE arrays/terminal schema mismatch")
    if np.abs(r).max() > 1e9 or np.abs(v).max() > 1e9:
        raise ValueError("GAE numeric resource bound exceeded")
    a = np.zeros_like(r)
    running = 0.0
    for t in range(len(r) - 1, -1, -1):
        continuation = float(not terminated[t])
        delta = r[t] + gamma * continuation * v[t + 1] - v[t]
        running = delta + gamma * lam * continuation * running
        a[t] = running
    return a, a + v[:-1]


def clipped_surrogate(
    new_log_prob: Array, old_log_prob: Array, advantages: Array, *, clip: float
) -> float:
    """Independent NumPy PPO Eq. 7 value, maximized by the policy."""
    _number(clip, "clip", 0.01, 0.4)
    new, old, a = (np.asarray(v, dtype=float) for v in (new_log_prob, old_log_prob, advantages))
    if (
        new.ndim != 1
        or not 1 <= len(new) <= 8192
        or new.shape != old.shape
        or new.shape != a.shape
        or any(not np.isfinite(v).all() for v in (new, old, a))
        or np.abs(new - old).max() > 40
    ):
        raise ValueError("PPO likelihood/advantage arrays exceed bounds")
    ratio = np.exp(new - old)
    return float(np.minimum(ratio * a, np.clip(ratio, 1 - clip, 1 + clip) * a).mean())


@dataclass(frozen=True)
class PPOConfig:
    rounds: int = 4
    epochs: int = 4
    minibatch: int = 32
    hidden: int = 16
    learning_rate: float = 0.003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        for name, low, high in (
            ("rounds", 1, 64),
            ("epochs", 1, 8),
            ("minibatch", 4, 256),
            ("hidden", 4, 64),
            ("seed", 0, 2**32 - 1),
        ):
            _count(getattr(self, name), name, low, high)
        for rate_name, rate_low, rate_high in (
            ("learning_rate", 1e-6, 0.03),
            ("gamma", 0, 1),
            ("gae_lambda", 0, 1),
            ("clip", 0.01, 0.4),
            ("value_coefficient", 0, 10),
            ("entropy_coefficient", 0, 1),
            ("gradient_clip", 0.1, 100),
        ):
            _number(getattr(self, rate_name), rate_name, rate_low, rate_high)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PPO fitting requires the optional nn extra (torch)") from exc
    return torch


def _shapes(hidden: int) -> dict[str, tuple[int, ...]]:
    return {
        "trunk": (N_FEATURES, hidden),
        "bias": (hidden,),
        "actor": (hidden, N_ACTIONS),
        "actor_bias": (N_ACTIONS,),
        "critic": (hidden,),
        "critic_bias": (1,),
    }


def _forward(parameters: dict[str, Any], x: Any, torch: Any) -> tuple[Any, Any]:
    h = torch.tanh(x @ parameters["trunk"] + parameters["bias"])
    return h @ parameters["actor"] + parameters["actor_bias"], h @ parameters[
        "critic"
    ] + parameters["critic_bias"][0]


def _ppo_objective(
    logits: Any,
    values: Any,
    masks: Any,
    actions: Any,
    old_log: Any,
    advantages: Any,
    targets: Any,
    config: PPOConfig,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    logp = torch.log_softmax(logits.masked_fill(~masks, float("-inf")), dim=1)
    selected = logp.gather(1, actions[:, None]).squeeze(1)
    ratio = torch.exp(selected - old_log)
    surrogate = torch.minimum(
        ratio * advantages, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * advantages
    ).mean()
    value_error = ((values - targets) ** 2).mean()
    entropy = -(torch.exp(logp) * logp.masked_fill(~masks, 0.0)).sum(1).mean()
    loss = (
        -surrogate + config.value_coefficient * value_error - config.entropy_coefficient * entropy
    )
    return loss, {
        "surrogate": surrogate,
        "value_error": value_error,
        "entropy": entropy,
        "clip_fraction": ((ratio - 1).abs() > config.clip).to(torch.float64).mean(),
    }


@dataclass(frozen=True)
class FrozenMakerPolicy:
    hidden: int
    parameters: tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]
    metadata_json: str
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _count(self.hidden, "hidden", 4, 64)
        shapes = _shapes(self.hidden)
        if not isinstance(self.parameters, tuple) or len(self.parameters) != len(shapes):
            raise ValueError("frozen parameter schema mismatch")
        names = set()
        for name, shape, values in self.parameters:
            if (
                name in names
                or name not in shapes
                or shape != shapes[name]
                or not isinstance(values, tuple)
                or len(values) != math.prod(shape)
            ):
                raise ValueError("frozen parameter architecture mismatch")
            names.add(name)
            for value in values:
                _number(value, "policy parameter", -1e6, 1e6)
        if not isinstance(self.metadata_json, str) or len(self.metadata_json.encode()) > _MAX_BYTES:
            raise ValueError("policy metadata resource mismatch")
        metadata = json.loads(self.metadata_json)
        if (
            not isinstance(metadata, dict)
            or metadata.get("synthetic") is not True
            or metadata.get("market_evidence") is not False
            or metadata.get("research_only") is not True
        ):
            raise ValueError("policy evidence honesty mismatch")
        object.__setattr__(
            self,
            "policy_sha256",
            _hash({"hidden": self.hidden, "parameters": self.parameters, "metadata": metadata}),
        )

    def distribution(self, observation: MakerObservation) -> tuple[Array, float]:
        if not isinstance(observation, MakerObservation):
            raise ValueError("typed current observation required")
        p = {name: np.array(values).reshape(shape) for name, shape, values in self.parameters}
        h = np.tanh(np.array(observation.features) @ p["trunk"] + p["bias"])
        logits = h @ p["actor"] + p["actor_bias"]
        mask = np.array(observation.action_mask)
        logits[~mask] = -np.inf
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        return probabilities, float(h @ p["critic"] + p["critic_bias"][0])

    def action(self, observation: MakerObservation) -> int:
        return int(np.argmax(self.distribution(observation)[0]))

    def metadata(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.metadata_json))

    @property
    def parameter_sha256(self) -> str:
        return _hash(self.parameters)

    def save(self, path: Path) -> None:
        encoded = json.dumps(
            {
                "schema": _SCHEMA,
                "hidden": self.hidden,
                "parameters": self.parameters,
                "metadata": self.metadata(),
                "policy_sha256": self.policy_sha256,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
        if len(encoded) > _MAX_BYTES:
            raise ValueError("policy artifact exceeds resource bound")
        with path.open("xb") as handle:
            handle.write(encoded)

    @classmethod
    def load(cls, path: Path) -> FrozenMakerPolicy:
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("policy artifact exceeds resource bound")
        with path.open("rb") as handle:
            encoded = handle.read(_MAX_BYTES + 1)
        if len(encoded) > _MAX_BYTES:
            raise ValueError("policy artifact exceeds resource bound")
        try:
            body = json.loads(encoded)
            if (
                set(body) != {"schema", "hidden", "parameters", "metadata", "policy_sha256"}
                or body["schema"] != _SCHEMA
            ):
                raise ValueError("policy schema mismatch")
            result = cls(
                body["hidden"],
                tuple((p[0], tuple(p[1]), tuple(p[2])) for p in body["parameters"]),
                json.dumps(body["metadata"], sort_keys=True),
            )
            if (
                result.policy_sha256 != body["policy_sha256"]
                or result.metadata()["code_bindings"] != _code_bindings()
                or result.metadata()["numpy_version"] != np.__version__
            ):
                raise ValueError("policy hash/source/dependency mismatch")
            _validate_trained_metadata(result)
            return result
        except (KeyError, TypeError, IndexError, AttributeError, RecursionError) as exc:
            raise ValueError("invalid policy artifact") from exc


def _parameter_state(
    parameters: dict[str, Any],
) -> tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]:
    return tuple(
        (name, tuple(p.shape), tuple(float(v) for v in p.detach().numpy().flatten()))
        for name, p in parameters.items()
    )


def _freeze(
    parameters: dict[str, Any], config: PPOConfig, metadata: dict[str, Any]
) -> FrozenMakerPolicy:
    states = _parameter_state(parameters)
    return FrozenMakerPolicy(
        config.hidden, states, json.dumps(metadata, sort_keys=True, allow_nan=False)
    )


def run_maker_episode(
    scenario: MakerScenario,
    config: MakerConfig,
    *,
    policy: FrozenMakerPolicy | None = None,
    actions: tuple[int, ...] | None = None,
    sampling_seed: int | None = None,
) -> dict[str, Any]:
    if policy is not None and actions is not None:
        raise ValueError("choose policy or action replay")
    if actions is not None and (not isinstance(actions, tuple) or len(actions) != config.decisions):
        raise ValueError("exact complete replay action tuple required")
    if sampling_seed is not None:
        _count(sampling_seed, "sampling seed", 0, 2**32 - 1)
        if policy is None:
            raise ValueError("sampling requires policy")
    rng = np.random.default_rng(sampling_seed)
    env = MakerEnvironment(scenario, config)
    transitions = []
    while not env.done:
        observation = env.observation()
        probability = value = None
        if actions is not None:
            action = actions[env.index]
        elif policy is None:
            action = 1 if observation.action_mask[1] else 0
        else:
            p, value = policy.distribution(observation)
            action = (
                int(rng.choice(N_ACTIONS, p=p)) if sampling_seed is not None else int(np.argmax(p))
            )
            probability = float(p[action])
        transition = env.step(action)
        transition.update({"behavior_probability": probability, "behavior_value": value})
        transitions.append(transition)
    fills = [fill for t in transitions for fill in t["fills"]]
    result = {
        "schema": _SCHEMA,
        "synthetic": True,
        "market_evidence": False,
        "research_only": True,
        "proper_forecast_score": False,
        "scenario": asdict(scenario),
        "environment": asdict(config),
        "code_bindings": _code_bindings(),
        "numpy_version": np.__version__,
        "policy_sha256": policy.policy_sha256 if policy is not None else None,
        "sampling_seed": sampling_seed,
        "transitions": transitions,
        "complete": env.done,
        "terminal_inventory": env.inventory,
        "terminal_inventory_liquidated": False,
        "terminal_mark_is_last_known_mid": True,
        "sim_internal_terminal_cash": env.cash,
        "sim_internal_wealth_change": env.cash
        + env.inventory * env.observation().last_known_mid
        - env.initial_wealth,
        "sim_internal_reward_sum": sum(t["sim_internal_reward"] for t in transitions),
        "sim_internal_spread_capture": sum(f["sim_internal_spread_capture"] for f in fills),
        "fees_paid": sum(f["fee"] for f in fills),
        "fills": len(fills),
        "max_inventory": max(t["next_observation"]["inventory"] for t in transitions),
        "inventory_mean_square_deviation": float(
            np.mean(
                [
                    (t["next_observation"]["inventory"] - config.target_inventory) ** 2
                    for t in transitions
                ]
            )
        ),
        "final_book_conservation": env.book.event_counts(),
    }
    result["episode_sha256"] = _hash(result)
    return result


def replay_maker_episode(
    result: dict[str, Any], *, policy: FrozenMakerPolicy | None = None
) -> dict[str, bool]:
    """Re-execute seeded engine/actions; optional policy verifies behavior decisions."""
    try:
        body = dict(result)
        config = MakerConfig(**body["environment"])
        if (
            not isinstance(body["transitions"], list)
            or len(body["transitions"]) != config.decisions
        ):
            raise ValueError("replay transition resource/schema mismatch")
        digest = body.pop("episode_sha256")
        if (
            _hash(body) != digest
            or body["code_bindings"] != _code_bindings()
            or body["numpy_version"] != np.__version__
        ):
            raise ValueError("episode hash/source/dependency mismatch")
        scenario_payload = dict(body["scenario"])
        scenario_payload["book"] = ZILobConfig(**scenario_payload["book"])
        scenario = MakerScenario(**scenario_payload)
        if policy is not None:
            if body["policy_sha256"] != policy.policy_sha256:
                raise ValueError("episode policy identity mismatch")
            replayed = run_maker_episode(
                scenario, config, policy=policy, sampling_seed=body["sampling_seed"]
            )
        else:
            replayed = run_maker_episode(
                scenario, config, actions=tuple(t["action"] for t in body["transitions"])
            )
            for actual, expected in zip(replayed["transitions"], body["transitions"], strict=True):
                actual["behavior_probability"] = expected["behavior_probability"]
                actual["behavior_value"] = expected["behavior_value"]
            replayed["policy_sha256"], replayed["sampling_seed"] = (
                body["policy_sha256"],
                body["sampling_seed"],
            )
            replayed.pop("episode_sha256")
            replayed["episode_sha256"] = _hash(replayed)
        if _hash(replayed) != _hash(result):
            raise ValueError("episode action/ledger replay mismatch")
        return {
            "ledger_replayed": True,
            "policy_replayed": policy is not None,
            "training_replayed": False,
            "execution_authenticity_verified": False,
            "market_evidence": False,
        }
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("invalid episode replay schema") from exc


def train_ppo_maker(
    scenarios: tuple[MakerScenario, ...], environment: MakerConfig, config: PPOConfig | None = None
) -> FrozenMakerPolicy:
    config = PPOConfig() if config is None else config
    if (
        not isinstance(environment, MakerConfig)
        or not isinstance(config, PPOConfig)
        or not isinstance(scenarios, tuple)
        or not 1 <= len(scenarios) <= 64
        or any(not isinstance(v, MakerScenario) for v in scenarios)
    ):
        raise ValueError("typed bounded training scenarios/config required")
    if len({s.episode_id for s in scenarios}) != len(scenarios) or len(
        {s.book.seed for s in scenarios}
    ) != len(scenarios):
        raise ValueError("training episode identities/seeds must be distinct")
    if config.rounds * len(scenarios) * environment.decisions > 8192:
        raise ValueError("training transition resource budget exceeded")
    for scenario in scenarios:
        MakerEnvironment(scenario, environment)
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    parameters = {
        name: (
            torch.zeros(shape, dtype=torch.float64, device="cpu")
            if "bias" in name
            else torch.randn(shape, generator=generator, dtype=torch.float64, device="cpu") * 0.1
        ).requires_grad_(True)
        for name, shape in _shapes(config.hidden).items()
    }
    optimizer = torch.optim.Adam(tuple(parameters.values()), lr=config.learning_rate)
    initial_hash = _hash(_parameter_state(parameters))
    metadata: dict[str, Any] = {
        "synthetic": True,
        "market_evidence": False,
        "research_only": True,
        "algorithm": "PPO_CLIP_GAE",
        "official_reproduction": False,
        "proper_forecast_score": False,
        "environment": asdict(environment),
        "config": asdict(config),
        "training_scenarios": [asdict(v) for v in scenarios],
        "training_scenario_sha256": _hash([asdict(v) for v in scenarios]),
        "code_bindings": _code_bindings(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "initial_parameter_sha256": initial_hash,
        "device": "cpu",
        "data_class": "SYNTHETIC_ZI_GENERATOR",
        "empirical_data_rights": "UNVERIFIED_NO_EMPIRICAL_INPUT",
        "training_replay_available": False,
    }
    behavior_metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    losses, rollout_hashes, rounds, training_rollouts, round_parameters = [], [], [], [], []
    rng = np.random.default_rng(config.seed)
    updates = 0
    for round_index in range(config.rounds):
        old_policy = _freeze(parameters, config, metadata)
        round_parameters.append(old_policy.parameters)
        observations, masks, actions, log_prob, advantages, targets = [], [], [], [], [], []
        episode_summaries = []
        for scenario in scenarios:
            sampling_seed = int(rng.integers(0, 2**32))
            episode = run_maker_episode(
                scenario, environment, policy=old_policy, sampling_seed=sampling_seed
            )
            training_rollouts.append(episode)
            rollout_hashes.append(episode["episode_sha256"])
            rewards = np.array([t["sim_internal_reward"] for t in episode["transitions"]])
            values = np.array([t["behavior_value"] for t in episode["transitions"]] + [0.0])
            adv, ret = generalized_advantages(
                rewards,
                values,
                tuple(t["done"] for t in episode["transitions"]),
                gamma=config.gamma,
                lam=config.gae_lambda,
            )
            for t, a, target in zip(episode["transitions"], adv, ret, strict=True):
                observations.append(t["observation"]["features"])
                masks.append(t["observation"]["action_mask"])
                actions.append(t["action"])
                log_prob.append(math.log(t["behavior_probability"]))
                advantages.append(float(a))
                targets.append(float(target))
            episode_summaries.append(
                {
                    "episode_id": scenario.episode_id,
                    "seed": scenario.book.seed,
                    "sampling_seed": sampling_seed,
                    "episode_sha256": episode["episode_sha256"],
                    "complete": episode["complete"],
                    "sim_internal_reward_sum": episode["sim_internal_reward_sum"],
                    "terminal_inventory": episode["terminal_inventory"],
                    "fills": episode["fills"],
                }
            )
        adv_array = np.array(advantages)
        adv_array = (adv_array - adv_array.mean()) / max(float(adv_array.std()), 1e-8)
        arrays = (
            torch.tensor(np.array(observations), dtype=torch.float64, device="cpu"),
            torch.tensor(np.array(masks), dtype=torch.bool, device="cpu"),
            torch.tensor(actions, dtype=torch.long, device="cpu"),
            torch.tensor(log_prob, dtype=torch.float64, device="cpu"),
            torch.tensor(adv_array, dtype=torch.float64, device="cpu"),
            torch.tensor(targets, dtype=torch.float64, device="cpu"),
        )
        for _ in range(config.epochs):
            order = rng.permutation(len(actions))
            for start in range(0, len(order), config.minibatch):
                index = torch.tensor(
                    order[start : start + config.minibatch], dtype=torch.long, device="cpu"
                )
                logits, predicted_values = _forward(parameters, arrays[0][index], torch)
                loss, diagnostic = _ppo_objective(
                    logits,
                    predicted_values,
                    arrays[1][index],
                    arrays[2][index],
                    arrays[3][index],
                    arrays[4][index],
                    arrays[5][index],
                    config,
                    torch,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite PPO learning objective")
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    tuple(parameters.values()), config.gradient_clip
                )
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("nonfinite PPO gradients")
                optimizer.step()
                losses.append(
                    {
                        "round": round_index,
                        "loss": float(loss.detach()),
                        **{name: float(value.detach()) for name, value in diagnostic.items()},
                    }
                )
                updates += 1
        rounds.append(
            {
                "round": round_index,
                "old_policy_sha256": old_policy.policy_sha256,
                "episodes": episode_summaries,
            }
        )
    metadata.update(
        {
            "rounds": rounds,
            "loss_history": losses,
            "updates": updates,
            "training_transitions": config.rounds * len(scenarios) * environment.decisions,
            "training_rollout_sha256": _hash(rollout_hashes),
            "rollout_hashes": rollout_hashes,
            "training_seed_reuse_across_rounds": True,
            "training_data_replay_available": True,
            "training_rollouts": training_rollouts,
            "behavior_policy_metadata_json": behavior_metadata_json,
            "round_policy_parameters": round_parameters,
        }
    )
    return _freeze(parameters, config, metadata)


def _validate_trained_metadata(policy: FrozenMakerPolicy) -> None:
    metadata = policy.metadata()
    if (
        metadata["algorithm"] != "PPO_CLIP_GAE"
        or metadata["official_reproduction"] is not False
        or metadata["proper_forecast_score"] is not False
        or metadata["device"] != "cpu"
        or metadata["data_class"] != "SYNTHETIC_ZI_GENERATOR"
        or metadata["training_replay_available"] is not False
        or metadata["training_data_replay_available"] is not True
        or metadata["training_seed_reuse_across_rounds"] is not True
    ):
        raise ValueError("trained-policy method/evidence contract mismatch")
    config = PPOConfig(**metadata["config"])
    environment = MakerConfig(**metadata["environment"])
    if config.hidden != policy.hidden:
        raise ValueError("trained-policy configuration/architecture mismatch")
    scenarios = []
    for row in metadata["training_scenarios"]:
        fields = dict(row)
        fields["book"] = ZILobConfig(**fields["book"])
        scenarios.append(MakerScenario(**fields))
    if (
        not 1 <= len(scenarios) <= 64
        or len({s.episode_id for s in scenarios}) != len(scenarios)
        or len({s.book.seed for s in scenarios}) != len(scenarios)
    ):
        raise ValueError("trained-policy episode identities mismatch")
    transitions = config.rounds * len(scenarios) * environment.decisions
    updates = (
        config.rounds
        * config.epochs
        * math.ceil(len(scenarios) * environment.decisions / config.minibatch)
    )
    if (
        transitions > 8192
        or metadata["training_transitions"] != transitions
        or metadata["updates"] != updates
        or _hash(metadata["training_scenarios"]) != metadata["training_scenario_sha256"]
    ):
        raise ValueError("trained-policy data/budget/count mismatch")
    if (
        len(metadata["loss_history"]) != updates
        or len(metadata["rounds"]) != config.rounds
        or len(metadata["round_policy_parameters"]) != config.rounds
        or len(metadata["training_rollouts"]) != config.rounds * len(scenarios)
    ):
        raise ValueError("trained-policy training-history counts mismatch")
    for entry in metadata["loss_history"]:
        _count(entry["round"], "training round", 0, config.rounds - 1)
        for key in ("loss", "surrogate", "value_error", "entropy", "clip_fraction"):
            _number(entry[key], key, -1e20, 1e20)
    rollouts = metadata["training_rollouts"]
    hashes = []
    for round_index in range(config.rounds):
        states = tuple(
            (p[0], tuple(p[1]), tuple(p[2]))
            for p in metadata["round_policy_parameters"][round_index]
        )
        old_policy = FrozenMakerPolicy(
            policy.hidden, states, metadata["behavior_policy_metadata_json"]
        )
        if metadata["rounds"][round_index]["old_policy_sha256"] != old_policy.policy_sha256:
            raise ValueError("old rollout-policy identity mismatch")
        for j, scenario in enumerate(scenarios):
            rollout = rollouts[round_index * len(scenarios) + j]
            if rollout["scenario"] != asdict(scenario) or rollout["environment"] != asdict(
                environment
            ):
                raise ValueError("training rollout scenario/config binding mismatch")
            replay_maker_episode(rollout, policy=old_policy)
            hashes.append(rollout["episode_sha256"])
    if hashes != metadata["rollout_hashes"] or _hash(hashes) != metadata["training_rollout_sha256"]:
        raise ValueError("training rollout data hash mismatch")


def evaluate_ppo_maker(
    policy: FrozenMakerPolicy, scenarios: tuple[MakerScenario, ...], environment: MakerConfig
) -> dict[str, Any]:
    if (
        not isinstance(policy, FrozenMakerPolicy)
        or not isinstance(scenarios, tuple)
        or not 1 <= len(scenarios) <= 64
        or any(not isinstance(v, MakerScenario) for v in scenarios)
    ):
        raise ValueError("bounded typed policy/evaluation scenarios required")
    metadata = policy.metadata()
    training = metadata["training_scenarios"]
    if environment != MakerConfig(**metadata["environment"]):
        raise ValueError("evaluation constraints must match frozen training environment")
    if (
        len({s.episode_id for s in scenarios}) != len(scenarios)
        or len({s.book.seed for s in scenarios}) != len(scenarios)
        or {s.episode_id for s in scenarios}.intersection(v["episode_id"] for v in training)
        or {s.book.seed for s in scenarios}.intersection(v["book"]["seed"] for v in training)
    ):
        raise ValueError("evaluation seeds/episodes overlap training or each other")
    outcomes = [
        {
            "scenario": asdict(s),
            "ppo": run_maker_episode(s, environment, policy=policy),
            "naive": run_maker_episode(s, environment),
        }
        for s in scenarios
    ]
    result = {
        "synthetic": True,
        "market_evidence": False,
        "research_only": True,
        "proper_forecast_score": False,
        "policy_sha256": policy.policy_sha256,
        "outcomes": outcomes,
        "comparison": {
            "environment_constraints_matched": True,
            "initial_seeds_matched": True,
            "identical_realized_tapes": False,
            "compute_matched": False,
            "naive_training_updates": 0,
            "ppo_training_updates": metadata["updates"],
            "benefit_asserted": False,
            "c51_status": "UNAVAILABLE_STATE_REWARD_FINANCING_MISMATCH",
            "heldout_regimes": sorted(
                {s.regime for s in scenarios} - {v["regime"] for v in training}
            ),
            "heldout_seeds": True,
        },
        "limitations": [
            "uncalibrated synthetic exogenous flow",
            "no strategic competitors/latency/hidden liquidity",
            "terminal inventory retained, not liquidated",
            "no empirical market evidence or proper forecast score",
            "unmatched C51 comparison remains open",
        ],
    }
    result["evaluation_sha256"] = _hash(result)
    return result
