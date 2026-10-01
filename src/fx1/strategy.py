"""Model-generated research strategies with a bounded, causal interpreter.

This is the first strategy-code lane, not a trained QuantCode reproduction.
An actual caller-supplied inference backend generates Python; no template is
substituted for model inference. The supported Python subset is deliberately
explicit: a single ``strategy(history)`` function returns a dictionary with
``probability_up`` and ``target_weight`` expressions. No Python code is executed.
The AST interpreter has no imports, attributes, indexing, loops, mutation,
filesystem or network access. General Backtrader programs need a separate
isolated runtime and are rejected rather than executed with host privileges.

Replay uses only observations available at the decision time and scores the
next observable close direction with the proper Brier and binary log scores.
It reports costed turnover, never a live performance claim. An independent
specification judge is required to claim semantic correctness. Its absence
cannot be rescued by syntax success or a successful replay.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from fx1.serve.backends import InferenceBackend

_MAX_CODE_BYTES = 16_384
_MAX_NODES = 256
_MAX_DEPTH = 32
_MAX_REPLAY_ROWS = 100_000
_MAX_REPLAY_WORK = 2_000_000
_MAX_REPLAY_DECISIONS = 10_000
_MAX_INTERPRETER_WORK = 20_000_000
_CONTRACT = "fx1_strategy_expression_v1"
_CALLS = frozenset({"last", "mean", "std", "minimum", "maximum", "return_over", "clip", "abs"})
_OUTPUTS = frozenset({"probability_up", "target_weight"})


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class StrategySpec:
    id: str
    title: str
    description: str
    parameters: Mapping[str, float | int | str]

    def __post_init__(self) -> None:
        if any(
            not isinstance(x, str) or not x.strip() for x in (self.id, self.title, self.description)
        ):
            raise ValueError("strategy id, title and description must be nonempty strings")
        if len(self.description.encode()) > _MAX_CODE_BYTES:
            raise ValueError("strategy description is too large")
        # Copy the mapping before freezing the serialized specification hash.
        params = dict(self.parameters)
        if not all(isinstance(k, str) and k.strip() for k in params):
            raise ValueError("parameter names must be nonempty strings")
        if any(
            isinstance(v, bool) or not isinstance(v, (str, float, int)) for v in params.values()
        ):
            raise ValueError("parameters must be scalar strings or numbers")
        _hash(params)  # reject non-finite numeric parameters
        from types import MappingProxyType

        object.__setattr__(self, "parameters", MappingProxyType(params))

    def payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    @property
    def sha256(self) -> str:
        return _hash(self.payload())


@dataclass(frozen=True)
class GeneratedCode:
    spec_id: str
    spec_sha256: str
    code_text: str
    code_sha256: str
    generated_at: str
    backend_id: str
    generation_seconds: float
    contract: str = _CONTRACT

    def __post_init__(self) -> None:
        if not self.spec_id.strip() or not self.backend_id.strip() or self.contract != _CONTRACT:
            raise ValueError("generated identity, backend and supported contract are required")
        if len(self.spec_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.spec_sha256
        ):
            raise ValueError("spec_sha256 must be a SHA-256 hex digest")
        if hashlib.sha256(self.code_text.encode()).hexdigest() != self.code_sha256:
            raise ValueError("generated code hash mismatch")
        _aware(datetime.fromisoformat(self.generated_at), "generated_at")
        if not math.isfinite(self.generation_seconds) or self.generation_seconds < 0:
            raise ValueError("generation_seconds must be finite and nonnegative")


@dataclass(frozen=True)
class ReplayBar:
    event_time: datetime
    available_time: datetime
    close: float

    def __post_init__(self) -> None:
        event = _aware(self.event_time, "event_time")
        available = _aware(self.available_time, "available_time")
        if available < event:
            raise ValueError("available_time must not precede event_time")
        if isinstance(self.close, bool) or not math.isfinite(self.close) or self.close <= 0:
            raise ValueError("close must be finite and positive")
        object.__setattr__(self, "event_time", event)
        object.__setattr__(self, "available_time", available)
        object.__setattr__(self, "close", float(self.close))


@dataclass(frozen=True)
class StrategyDecision:
    probability_up: float
    target_weight: float


@dataclass(frozen=True)
class BacktestResult:
    spec_id: str
    code_sha256: str
    data_sha256: str
    source_bars: tuple[tuple[str, str, float], ...]
    decision_times: tuple[str, ...]
    probabilities_up: tuple[float, ...]
    target_weights: tuple[float, ...]
    realized_up: tuple[int, ...]
    brier_score: float
    binary_log_score: float
    turnover: float
    transaction_cost_bps: float
    cost_bps: float
    data_source: str
    synthetic: bool
    success_flag: bool = True
    research_only: bool = True
    live_pnl_claim: bool = False

    def __post_init__(self) -> None:
        for name in ("decision_times", "probabilities_up", "target_weights", "realized_up"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "source_bars", tuple(tuple(row) for row in self.source_bars))
        n = len(self.decision_times)
        if n == 0 or any(
            len(values) != n
            for values in (self.probabilities_up, self.target_weights, self.realized_up)
        ):
            raise ValueError("replay arrays must be nonempty and aligned")
        p = np.asarray(self.probabilities_up, dtype=float)
        weights = np.asarray(self.target_weights, dtype=float)
        labels = np.asarray(self.realized_up)
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("replay probabilities must be finite in [0,1]")
        if not np.all(np.isfinite(weights)) or np.any(np.abs(weights) > 1):
            raise ValueError("replay target weights must be finite in [-1,1]")
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("replay labels must be binary")
        likelihood = np.where(labels == 1, p, 1 - p)
        if np.any(likelihood == 0):
            raise ValueError("replay binary log score must be finite")
        expected_brier = float(np.mean((p - labels) ** 2))
        expected_log = float(-np.mean(np.log(likelihood)))
        expected_turnover = float(np.sum(np.abs(np.diff(np.r_[0.0, weights]))))
        for name, actual, expected in (
            ("brier_score", self.brier_score, expected_brier),
            ("binary_log_score", self.binary_log_score, expected_log),
            ("turnover", self.turnover, expected_turnover),
        ):
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"replay {name} does not agree with its observations")
        if not math.isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0:
            raise ValueError("transaction costs must be finite and nonnegative")
        if (
            not math.isfinite(self.cost_bps)
            or self.cost_bps < 0
            or not math.isclose(
                self.transaction_cost_bps,
                self.cost_bps * self.turnover,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("transaction cost does not agree with declared rate and turnover")
        if self.research_only is not True or self.live_pnl_claim is not False:
            raise ValueError("replay must retain research-only honesty flags")
        if not isinstance(self.success_flag, bool):
            raise ValueError("success_flag must be an explicit boolean")
        if not isinstance(self.synthetic, bool) or not self.data_source.strip():
            raise ValueError("replay requires an explicit source and synthetic label")
        if "synthetic" in self.data_source.lower() and not self.synthetic:
            raise ValueError("synthetic source cannot be labeled empirical")
        expected_data_hash = _hash(
            {"bars": self.source_bars, "source": self.data_source, "synthetic": self.synthetic}
        )
        if self.data_sha256 != expected_data_hash:
            raise ValueError("replay source data hash mismatch")

    @property
    def sha256(self) -> str:
        """Bind the full replay, including clocks, costs and scores, not only input bars."""
        return _hash(asdict(self))


@dataclass(frozen=True)
class StrategyEvaluation:
    syntax_passed: bool
    replay_passed: bool
    specification_correct: bool | None
    judge_id: str | None
    failures: tuple[str, ...]
    spec_sha256: str
    code_sha256: str
    data_sha256: str
    replay_sha256: str
    research_only: bool = True
    live_pnl_claim: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.syntax_passed, bool) or not isinstance(self.replay_passed, bool):
            raise ValueError("syntax and replay results must be explicit booleans")
        if self.specification_correct is not None and not isinstance(
            self.specification_correct, bool
        ):
            raise ValueError("specification correctness must be boolean or unmeasured")
        if self.research_only is not True or self.live_pnl_claim is not False:
            raise ValueError("evaluation must retain research-only honesty flags")
        if self.specification_correct is True and (not self.judge_id or self.failures):
            raise ValueError("a correct specification requires an identified judge and no failures")

    @property
    def judge_passed(self) -> bool:
        return self.syntax_passed and self.replay_passed and self.specification_correct is True


SpecJudge = Callable[[StrategySpec, GeneratedCode, BacktestResult], bool]


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((_depth(child) for child in children), default=0)


def parse_strategy(code: str) -> ast.Dict:
    """Validate the complete program before any interpretation, including dead branches."""
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > _MAX_CODE_BYTES:
        raise ValueError("code must be nonempty and at most 16 KiB")
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid strategy syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ValueError("strategy exceeds the AST node budget")
    if _depth(tree) > _MAX_DEPTH:
        raise ValueError("strategy exceeds the expression depth budget")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("only a single strategy(history) function is supported")
    fn = tree.body[0]
    args = fn.args
    if (
        fn.name != "strategy"
        or fn.decorator_list
        or fn.returns is not None
        or fn.type_params
        or len(args.args) != 1
        or args.args[0].arg != "history"
        or args.args[0].annotation is not None
        or args.posonlyargs
        or args.kwonlyargs
        or args.defaults
        or args.kw_defaults
        or args.vararg
        or args.kwarg
        or len(fn.body) != 1
        or not isinstance(fn.body[0], ast.Return)
        or not isinstance(fn.body[0].value, ast.Dict)
    ):
        raise ValueError("expected undecorated strategy(history) returning one decision dictionary")
    result = fn.body[0].value
    keys = [key.value if isinstance(key, ast.Constant) else None for key in result.keys]
    if len(keys) != 2 or set(keys) != _OUTPUTS:
        raise ValueError("decision requires exactly probability_up and target_weight")
    for value in result.values:
        _validate_expression(value)
    return result


def _validate_expression(node: ast.AST) -> None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (float, int)):
            raise ValueError("only finite numeric constants are allowed in expressions")
        if not math.isfinite(float(node.value)) or abs(node.value) > 1e12:
            raise ValueError("constant is outside the finite numeric budget")
    elif isinstance(node, ast.Name):
        if node.id != "history":
            raise ValueError(f"unknown name: {node.id}")
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        _validate_expression(node.left)
        _validate_expression(node.right)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_expression(node.operand)
    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        if not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
            raise ValueError("unsupported comparison")
        _validate_expression(node.left)
        _validate_expression(node.comparators[0])
    elif isinstance(node, ast.IfExp):
        _validate_expression(node.test)
        _validate_expression(node.body)
        _validate_expression(node.orelse)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALLS:
        if node.keywords:
            raise ValueError("keyword arguments are unsupported")
        name = node.func.id
        expected = 1 if name in {"last", "abs"} else 3 if name == "clip" else 2
        if len(node.args) != expected:
            raise ValueError(f"{name} requires {expected} arguments")
        if name not in {"clip", "abs"}:
            if not isinstance(node.args[0], ast.Name) or node.args[0].id != "history":
                raise ValueError(f"{name} requires history as its first argument")
            if len(node.args) == 2 and (
                not isinstance(node.args[1], ast.Constant)
                or isinstance(node.args[1].value, bool)
                or not isinstance(node.args[1].value, int)
                or not 1 <= node.args[1].value <= 10_000
            ):
                raise ValueError("lookback must be a literal integer in [1, 10000]")
        for arg in node.args:
            _validate_expression(arg)
    else:
        raise ValueError(f"unsupported expression: {type(node).__name__}")


def _scalar(value: Any) -> float:
    if isinstance(value, np.ndarray):
        raise ValueError("history is valid only as an indicator argument")
    out = float(value)
    if not math.isfinite(out) or abs(out) > 1e12:
        raise ValueError("expression produced a non-finite or oversized scalar")
    return out


def _interpret(node: ast.AST, history: np.ndarray) -> float | np.ndarray:
    if isinstance(node, ast.Constant):
        return float(cast(float, node.value))
    if isinstance(node, ast.Name):
        return history
    if isinstance(node, ast.IfExp):
        branch = node.body if _scalar(_interpret(node.test, history)) else node.orelse
        return _interpret(branch, history)
    if isinstance(node, ast.UnaryOp):
        value = _scalar(_interpret(node.operand, history))
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _scalar(_interpret(node.left, history))
        right = _scalar(_interpret(node.right, history))
        if isinstance(node.op, ast.Add):
            return _scalar(left + right)
        if isinstance(node.op, ast.Sub):
            return _scalar(left - right)
        if isinstance(node.op, ast.Mult):
            return _scalar(left * right)
        if right == 0:
            raise ValueError("division by zero in strategy")
        return _scalar(left / right)
    if isinstance(node, ast.Compare):
        left = _scalar(_interpret(node.left, history))
        right = _scalar(_interpret(node.comparators[0], history))
        op = node.ops[0]
        comparisons = {
            ast.Lt: left < right,
            ast.LtE: left <= right,
            ast.Gt: left > right,
            ast.GtE: left >= right,
            ast.Eq: left == right,
            ast.NotEq: left != right,
        }
        return float(comparisons[type(op)])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name == "abs":
            return abs(_scalar(_interpret(node.args[0], history)))
        if name == "clip":
            value, lo, hi = (_scalar(_interpret(arg, history)) for arg in node.args)
            if lo > hi:
                raise ValueError("clip lower bound exceeds upper bound")
            return float(np.clip(value, lo, hi))
        if name == "last":
            return float(history[-1])
        width = cast(int, cast(ast.Constant, node.args[1]).value)  # validated literal
        required = width + 1 if name == "return_over" else width
        if len(history) < required:
            raise ValueError(f"{name} needs {required} available observations")
        window = history[-width:]
        if name == "mean":
            return float(np.mean(window))
        if name == "std":
            return float(np.std(window))
        if name == "minimum":
            return float(np.min(window))
        if name == "maximum":
            return float(np.max(window))
        return float(history[-1] / history[-width - 1] - 1.0)
    raise ValueError("strategy was not validated")


def run_strategy(code: str, history: Sequence[float]) -> StrategyDecision:
    expression = parse_strategy(code)
    return _run_expression(expression, history)


def _run_expression(expression: ast.Dict, history: Sequence[float]) -> StrategyDecision:
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0 or values.size > _MAX_REPLAY_ROWS:
        raise ValueError("history must be a nonempty bounded vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("history prices must be finite and positive")
    decision = {
        key.value: _scalar(_interpret(value, values))
        for key, value in zip(expression.keys, expression.values, strict=True)
        if isinstance(key, ast.Constant)
    }
    if not 0 <= decision["probability_up"] <= 1:
        raise ValueError("probability_up must be in [0, 1]")
    if not -1 <= decision["target_weight"] <= 1:
        raise ValueError("target_weight must be in [-1, 1]")
    return StrategyDecision(decision["probability_up"], decision["target_weight"])


def text_to_code(
    spec: StrategySpec, backend: InferenceBackend, *, backend_id: str
) -> GeneratedCode:
    if not backend_id.strip():
        raise ValueError("backend_id is required for provenance")
    messages = [
        {
            "role": "system",
            "content": (
                "Generate research strategy Python only, with no Markdown fences. "
                "Return exactly one undecorated def strategy(history): function with one return. "
                "Return {'probability_up': expression, 'target_weight': expression}. "
                "probability_up is next-close up probability in [0,1], target_weight in [-1,1]. "
                "history contains positive close prices available at the decision time. "
                "Allowed: numeric constants, + - * /, comparisons, conditional expressions, "
                "last(history), mean/std/minimum/maximum(history, literal_lookback), "
                "return_over(history, literal_lookback), abs(x), clip(x,lo,hi). "
                "No imports, attributes, indexing, other functions, assignments or loops. "
                "No market, live, profitability or SOTA claims."
            ),
        },
        {"role": "user", "content": json.dumps(spec.payload(), sort_keys=True, allow_nan=False)},
    ]
    start = time.perf_counter()
    code = backend.complete(messages)
    latency = time.perf_counter() - start
    parse_strategy(code)
    return GeneratedCode(
        spec_id=spec.id,
        spec_sha256=spec.sha256,
        code_text=code,
        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        generated_at=datetime.now(UTC).isoformat(),
        backend_id=backend_id,
        generation_seconds=latency,
    )


def backtest(
    generated: GeneratedCode,
    bars: Sequence[ReplayBar],
    *,
    decision_times: Sequence[datetime],
    data_source: str,
    synthetic: bool,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """Replay a single instrument; delayed observations cannot enter history."""
    expression = parse_strategy(generated.code_text)
    if hashlib.sha256(generated.code_text.encode()).hexdigest() != generated.code_sha256:
        raise ValueError("generated code hash mismatch")
    if not isinstance(synthetic, bool) or not data_source.strip():
        raise ValueError("explicit data source and synthetic label are required")
    if "synthetic" in data_source.lower() and not synthetic:
        raise ValueError("synthetic source cannot be labeled empirical")
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    if not 2 <= len(bars) <= _MAX_REPLAY_ROWS or not decision_times:
        raise ValueError("bounded bars and nonempty decisions are required")
    if len(decision_times) > _MAX_REPLAY_DECISIONS:
        raise ValueError("replay exceeds the decision count budget")
    if any(
        left.event_time >= right.event_time for left, right in zip(bars, bars[1:], strict=False)
    ):
        raise ValueError("bars must have strictly increasing unique event times")
    clocks = [_aware(t, "decision_time") for t in decision_times]
    if any(a >= b for a, b in zip(clocks, clocks[1:], strict=False)):
        raise ValueError("decisions must have strictly increasing unique times")
    if len(bars) * len(clocks) > _MAX_REPLAY_WORK:
        raise ValueError("replay exceeds the bounded observation work budget")
    if len(bars) * len(clocks) * sum(1 for _ in ast.walk(expression)) > _MAX_INTERPRETER_WORK:
        raise ValueError("replay exceeds the bounded interpreter work budget")
    probabilities: list[float] = []
    weights: list[float] = []
    labels: list[int] = []
    turnover = 0.0
    previous_weight = 0.0
    for clock in clocks:
        visible = [bar for bar in bars if bar.event_time <= clock and bar.available_time <= clock]
        future = next((bar for bar in bars if bar.event_time > clock), None)
        if not visible or future is None:
            raise ValueError("each decision requires visible history and a subsequent label bar")
        last_visible = visible[-1]
        # The target is next-event versus the latest decision-event close,
        # rather than a skipped latent close; delayed current bars make the
        # decision unscoreable instead of silently changing the horizon.
        current = next((bar for bar in reversed(bars) if bar.event_time <= clock), None)
        if current is not last_visible:
            raise ValueError("current event close is unavailable at the decision clock")
        decision = _run_expression(expression, [bar.close for bar in visible])
        probabilities.append(decision.probability_up)
        weights.append(decision.target_weight)
        labels.append(int(future.close > last_visible.close))
        turnover += abs(decision.target_weight - previous_weight)
        previous_weight = decision.target_weight
    p = np.asarray(probabilities)
    y = np.asarray(labels)
    # Exact endpoint probability contradicting the event has infinite log
    # score. Refuse that result rather than quietly clipping the forecast.
    likelihood = np.where(y == 1, p, 1 - p)
    if np.any(likelihood == 0):
        raise ValueError("zero event probability produces an infinite log score")
    data_payload = [
        (bar.event_time.isoformat(), bar.available_time.isoformat(), float(bar.close))
        for bar in bars
    ]
    return BacktestResult(
        spec_id=generated.spec_id,
        code_sha256=generated.code_sha256,
        data_sha256=_hash({"bars": data_payload, "source": data_source, "synthetic": synthetic}),
        source_bars=tuple(data_payload),
        decision_times=tuple(t.isoformat() for t in clocks),
        probabilities_up=tuple(probabilities),
        target_weights=tuple(weights),
        realized_up=tuple(labels),
        brier_score=float(np.mean((p - y) ** 2)),
        binary_log_score=float(-np.mean(np.log(likelihood))),
        turnover=turnover,
        transaction_cost_bps=cost_bps * turnover,
        cost_bps=cost_bps,
        data_source=data_source,
        synthetic=synthetic,
    )


def evaluate(
    spec: StrategySpec,
    generated: GeneratedCode,
    replay: BacktestResult,
    *,
    judge: SpecJudge | None = None,
    judge_id: str | None = None,
) -> StrategyEvaluation:
    if generated.spec_id != spec.id or generated.spec_sha256 != spec.sha256:
        raise ValueError("generated code is not bound to this specification")
    if replay.spec_id != spec.id or replay.code_sha256 != generated.code_sha256:
        raise ValueError("replay is not bound to this generated code")
    parse_strategy(generated.code_text)
    if hashlib.sha256(generated.code_text.encode()).hexdigest() != generated.code_sha256:
        raise ValueError("generated code hash mismatch")
    if judge is None:
        return StrategyEvaluation(
            True,
            replay.success_flag,
            None,
            None,
            ("specification_judge_missing",),
            spec.sha256,
            generated.code_sha256,
            replay.data_sha256,
            replay.sha256,
        )
    if not judge_id or not judge_id.strip():
        raise ValueError("judge_id is required when a judge is provided")
    correct = judge(spec, generated, replay)
    if not isinstance(correct, bool):
        raise ValueError("specification judge must return an explicit boolean")
    return StrategyEvaluation(
        True,
        replay.success_flag,
        correct,
        judge_id,
        () if correct else ("specification_mismatch",),
        spec.sha256,
        generated.code_sha256,
        replay.data_sha256,
        replay.sha256,
    )


def build_receipt(
    spec: StrategySpec,
    generated: GeneratedCode,
    replay: BacktestResult,
    evaluation: StrategyEvaluation,
) -> dict[str, Any]:
    """Build a hash-bound feedback trace for HTTP and immutable file outputs."""
    evaluate(spec, generated, replay)  # independently recheck identity binding
    if (
        evaluation.spec_sha256,
        evaluation.code_sha256,
        evaluation.data_sha256,
        evaluation.replay_sha256,
    ) != (
        spec.sha256,
        generated.code_sha256,
        replay.data_sha256,
        replay.sha256,
    ):
        raise ValueError("evaluation is not bound to this specification, code and replay")
    payload = {
        "schema_version": "fx1_strategy_receipt_v1",
        "spec": spec.payload(),
        "generated": asdict(generated),
        "replay": asdict(replay),
        "evaluation": asdict(evaluation),
        "specification_judge_passed": evaluation.judge_passed,
        "research_only": True,
        "live_pnl_claim": False,
        "synthetic": replay.synthetic,
        "trained_checkpoint": False,
        "market_evidence": False,
        "generation_identity_independently_verified": False,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            "bounded_expression_contract_only",
            "no_fill_or_market_impact_model",
            "no_trained_strategy_checkpoint_evidence",
            "not_a_verify_research_receipt",
        ],
    }
    digest = _hash(payload)
    payload["receipt_sha256"] = digest
    return payload


def write_receipt(
    path: str | Path,
    spec: StrategySpec,
    generated: GeneratedCode,
    replay: BacktestResult,
    evaluation: StrategyEvaluation,
) -> str:
    """Write an immutable feedback trace. This is not a verify-research receipt."""
    payload = build_receipt(spec, generated, replay, evaluation)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return str(payload["receipt_sha256"])


def verify_strategy_receipt(path: str | Path) -> dict[str, object]:
    """Verify content/identity/honesty and reproduce the bounded historical replay.

    A receipt preserves a judge's declared verdict but cannot independently
    prove the competence or truthfulness of an external semantic judge.
    """
    source = Path(path)
    if source.stat().st_size > 32_000_000:
        raise ValueError("strategy receipt exceeds the bounded input size")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "fx1_strategy_receipt_v1":
        raise ValueError("unsupported strategy receipt schema")
    digest = payload.pop("receipt_sha256", None)
    if digest != _hash(payload):
        raise ValueError("strategy receipt content hash mismatch")
    if payload.get("research_only") is not True or payload.get("live_pnl_claim") is not False:
        raise ValueError("strategy receipt honesty flags invalid")
    if (
        payload.get("trained_checkpoint") is not False
        or payload.get("market_evidence") is not False
    ):
        raise ValueError("strategy receipt cannot certify a checkpoint or market evidence")
    spec = StrategySpec(**payload["spec"])
    generated = GeneratedCode(**payload["generated"])
    result = BacktestResult(**payload["replay"])
    assessment = StrategyEvaluation(**payload["evaluation"])
    evaluate(spec, generated, result)
    if (
        assessment.spec_sha256,
        assessment.code_sha256,
        assessment.data_sha256,
        assessment.replay_sha256,
    ) != (
        spec.sha256,
        generated.code_sha256,
        result.data_sha256,
        result.sha256,
    ):
        raise ValueError("strategy receipt evaluation binding mismatch")
    if (
        payload.get("synthetic") is not result.synthetic
        or payload.get("specification_judge_passed") is not assessment.judge_passed
    ):
        raise ValueError("strategy receipt evidence status mismatch")
    bars = [
        ReplayBar(datetime.fromisoformat(event), datetime.fromisoformat(available), price)
        for event, available, price in result.source_bars
    ]
    reproduced = backtest(
        generated,
        bars,
        decision_times=[datetime.fromisoformat(clock) for clock in result.decision_times],
        data_source=result.data_source,
        synthetic=result.synthetic,
        cost_bps=result.cost_bps,
    )
    if reproduced.sha256 != result.sha256:
        raise ValueError("strategy receipt replay did not reproduce")
    return {
        "valid": True,
        "receipt_sha256": digest,
        "replay_sha256": result.sha256,
        "replay_reproduced": True,
        "synthetic": result.synthetic,
        "specification_judge_recorded_pass": assessment.judge_passed,
        "semantic_judge_independently_verified": False,
        "generation_identity_independently_verified": False,
        "implementation_matches_current": payload.get("implementation_sha256")
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "research_only": True,
        "live_pnl_claim": False,
    }
