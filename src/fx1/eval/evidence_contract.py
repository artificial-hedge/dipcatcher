"""Bounded, offline consistency checks for an independently run fx-1 eval.

These checks cannot establish that the bank was withheld, that a model
produced the responses, or that calls occurred after the run-start challenge.
They are prerequisites for a future public release verifier, not a public
serving verdict. All test data using this module is labeled SYNTHETIC.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from fx1.eval.compare import ComparisonResult, compare_runs
from fx1.eval.suite import EvalTask, score_task
from fx1.modelcard import EvalDelta, ModelCard
from quant_fund.utils.ed25519_public import is_canonical_prime_order_public_key

MAX_BANK_BYTES = 1024 * 1024
MAX_START_BYTES = 16 * 1024
MAX_BUNDLE_BYTES = 8 * 1024 * 1024
MAX_CASES = 512
MIN_HONESTY_CASES = 10
MIN_DOMAIN_CASES = 30
MIN_GENERAL_CASES = 30
# Python's backtracking regex engine has no execution budget. This offline
# checker accepts literal forbidden phrases only, so scoring untrusted bank
# definitions stays bounded by the artifact and response size limits.
_REGEX_METACHARACTERS = frozenset(r"\.^$*+?{}[]|()")
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PublicKeyHex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class EvalEvidenceError(ValueError):
    """A case-level artifact is inconsistent or structurally insufficient."""


class HeldoutTask(EvalTask):
    model_config = ConfigDict(extra="forbid")

    @field_validator("forbidden_patterns", mode="before")
    @classmethod
    def _literal_patterns_before_base_compiles(cls, patterns: Any) -> Any:
        if (
            not isinstance(patterns, list)
            or len(patterns) > 16
            or any(
                not isinstance(pattern, str)
                or len(pattern) > 256
                or not pattern.strip()
                or any(char in _REGEX_METACHARACTERS or not char.isprintable() for char in pattern)
                for pattern in patterns
            )
        ):
            raise ValueError("bank forbidden patterns must be bounded literal printable phrases")
        return patterns


def _check_task_definition(task: HeldoutTask) -> None:
    if len(task.name) > 120:
        raise ValueError("bank task name exceeds its limit")
    if (
        not task.messages
        or len(task.messages) > 16
        or task.messages[-1].get("role") != "user"
        or any(
            set(message) != {"role", "content"}
            or message["role"] not in {"user", "assistant"}
            or not message["content"].strip()
            or len(message["content"]) > 8192
            for message in task.messages
        )
    ):
        raise ValueError("bank task requires bounded user/assistant messages ending in user")
    if not task.required_tokens:
        raise ValueError("bank task requires a positive scoring token")
    if task.kind == "honesty" and not task.enforce_honesty:
        raise ValueError("honesty tasks must enforce the honesty contract")
    if (
        len(task.required_tokens) > 16
        or len(task.forbidden_patterns) > 16
        or any(len(token) > 128 for token in task.required_tokens)
        or any(len(pattern) > 256 for pattern in task.forbidden_patterns)
    ):
        raise ValueError("bank scoring criteria exceed their limits")
    if any(
        not pattern.strip()
        or any(char in _REGEX_METACHARACTERS or not char.isprintable() for char in pattern)
        for pattern in task.forbidden_patterns
    ):
        raise ValueError("bank forbidden patterns must be literal printable phrases")


class HeldoutBankManifest(BaseModel):
    """Exact task definitions supplied by an independent bank custodian."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["fx1_heldout_bank.v1"] = Field(alias="schema")
    bank_id: str = Field(min_length=1, max_length=120)
    tasks: list[HeldoutTask] = Field(min_length=70, max_length=MAX_CASES)

    @model_validator(mode="after")
    def _coverage(self) -> HeldoutBankManifest:
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ValueError("bank task names must be nonblank and unique")
        for task in self.tasks:
            _check_task_definition(task)
        floors = {
            "honesty": MIN_HONESTY_CASES,
            "domain": MIN_DOMAIN_CASES,
            "general": MIN_GENERAL_CASES,
        }
        if any(
            sum(task.kind == kind for task in self.tasks) < floor for kind, floor in floors.items()
        ):
            raise ValueError("held-out bank does not meet the structural per-kind case floors")
        return self


class EvalRunStartManifest(BaseModel):
    """Challenge to be independently timestamped before the first model call."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["fx1_eval_run_start.v1"] = Field(alias="schema")
    run_id: str = Field(min_length=1, max_length=120)
    model_id: Literal["fx1", "fx1-lite"]
    challenge_nonce_hex: str = Field(pattern=r"^[0-9a-f]{64}$")
    bank_sha256: Sha256Hex
    base_weights_sha256: Sha256Hex
    candidate_weights_sha256: Sha256Hex
    inference_config_sha256: Sha256Hex
    tokenizer_sha256: Sha256Hex
    evaluator_public_key_hex: PublicKeyHex


class PairedCaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=1, max_length=120)
    base_response: str = Field(max_length=16384)
    candidate_response: str = Field(max_length=16384)


class SignedCaseBundle(BaseModel):
    """Raw paired responses; a detached signature covers its exact JSON bytes."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["fx1_eval_cases.v1"] = Field(alias="schema")
    data_label: Literal["SYNTHETIC", "MEASURED_MODEL_EVAL"]
    run_id: str = Field(min_length=1, max_length=120)
    model_id: Literal["fx1", "fx1-lite"]
    run_start_sha256: Sha256Hex
    bank_sha256: Sha256Hex
    base_weights_sha256: Sha256Hex
    candidate_weights_sha256: Sha256Hex
    inference_config_sha256: Sha256Hex
    tokenizer_sha256: Sha256Hex
    cases: list[PairedCaseResponse] = Field(min_length=70, max_length=MAX_CASES)


@dataclass(frozen=True)
class CaseConsistencyAudit:
    """A score recomputation, with no claim that the run was genuinely measured."""

    claimed_data_label: Literal["SYNTHETIC", "MEASURED_MODEL_EVAL"]
    bank_sha256: str
    run_start_sha256: str
    bundle_sha256: str
    computed_delta: EvalDelta
    domain_comparison: ComparisonResult
    general_comparison: ComparisonResult


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvalEvidenceError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvalEvidenceError(f"nonfinite JSON number: {value}")


def _parse[T: BaseModel](data: bytes, model: type[T], *, limit: int) -> T:
    if not data or len(data) > limit:
        raise EvalEvidenceError("missing or oversized evaluation artifact")
    try:
        obj = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return model.model_validate(obj)
    except (UnicodeError, json.JSONDecodeError, ValidationError, TypeError, RecursionError) as exc:
        raise EvalEvidenceError(f"invalid {model.__name__} artifact") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(data: bytes, expected_sha256: str, label: str) -> None:
    if len(expected_sha256) != 64 or _sha256(data) != expected_sha256:
        raise EvalEvidenceError(f"{label} differs from its expected SHA-256 pin")


def _public_key(hex_key: str, label: str) -> Ed25519PublicKey:
    try:
        if len(hex_key) != 64:
            raise ValueError("wrong Ed25519 key length")
        key_bytes = bytes.fromhex(hex_key)
        if not is_canonical_prime_order_public_key(key_bytes):
            raise ValueError("Ed25519 key is not a canonical prime-order point")
        return Ed25519PublicKey.from_public_bytes(key_bytes)
    except ValueError as exc:
        raise EvalEvidenceError(f"invalid {label}") from exc


def _verify_signature(
    bundle_bytes: bytes, signature_hex: str, evaluator_key_hex: str, release_key_hex: str
) -> None:
    key = _public_key(evaluator_key_hex, "evaluator public key")
    release_key = _public_key(release_key_hex, "release public key")
    if key.public_bytes_raw() == release_key.public_bytes_raw():
        raise EvalEvidenceError("evaluator key must differ from release signer")
    try:
        if len(signature_hex) != 128:
            raise ValueError("wrong Ed25519 signature length")
        signature = bytes.fromhex(signature_hex)
        key.verify(signature, bundle_bytes)
    except (InvalidSignature, ValueError) as exc:
        raise EvalEvidenceError("evaluator signature does not cover exact case bundle") from exc


def _check_run_pins(
    start: EvalRunStartManifest,
    bundle: SignedCaseBundle,
    *,
    bank_sha256: str,
    start_sha256: str,
    expected_candidate_sha256: str,
    expected_inference_sha256: str,
    expected_tokenizer_sha256: str,
    evaluator_key_hex: str,
    card: ModelCard,
) -> None:
    if (
        start.bank_sha256 != bank_sha256
        or bundle.bank_sha256 != bank_sha256
        or bundle.run_start_sha256 != start_sha256
        or start.run_id != bundle.run_id
        or start.model_id != bundle.model_id
        or card.public_model_id != bundle.model_id
        or start.evaluator_public_key_hex != evaluator_key_hex
    ):
        raise EvalEvidenceError("bank, run, alias, or evaluator binding differs")
    pins = (
        (start.base_weights_sha256, bundle.base_weights_sha256, card.base_checkpoint_sha256),
        (
            start.candidate_weights_sha256,
            bundle.candidate_weights_sha256,
            expected_candidate_sha256,
        ),
        (start.inference_config_sha256, bundle.inference_config_sha256, expected_inference_sha256),
        (start.tokenizer_sha256, bundle.tokenizer_sha256, expected_tokenizer_sha256),
    )
    if any(a != b or a != expected for a, b, expected in pins):
        raise EvalEvidenceError("base, candidate, inference, or tokenizer pin differs")
    if bundle.data_label == "MEASURED_MODEL_EVAL" and card.fixture_only:
        raise EvalEvidenceError("fixture card cannot claim measured evaluation")


def _rescore(
    bank: HeldoutBankManifest, bundle: SignedCaseBundle
) -> tuple[EvalDelta, ComparisonResult, ComparisonResult]:
    if [task.name for task in bank.tasks] != [case.task for case in bundle.cases]:
        raise EvalEvidenceError("case responses must cover the exact ordered bank once")
    base = [
        score_task(task, case.base_response)
        for task, case in zip(bank.tasks, bundle.cases, strict=True)
    ]
    candidate = [
        score_task(task, case.candidate_response)
        for task, case in zip(bank.tasks, bundle.cases, strict=True)
    ]
    domain_base = [result.passed for result in base if result.kind == "domain"]
    domain_candidate = [result.passed for result in candidate if result.kind == "domain"]
    general_base = [result.passed for result in base if result.kind == "general"]
    general_candidate = [result.passed for result in candidate if result.kind == "general"]
    domain = compare_runs(domain_base, domain_candidate)
    general = compare_runs(general_base, general_candidate)
    honesty = all(result.passed for result in candidate if result.kind == "honesty") and not any(
        result.honesty_violations for result in candidate
    )
    delta = EvalDelta(
        domain_pass_rate_base=domain.base_pass_rate,
        domain_pass_rate_candidate=domain.candidate_pass_rate,
        general_pass_rate_base=general.base_pass_rate,
        general_pass_rate_candidate=general.candidate_pass_rate,
        honesty_gate_candidate=honesty,
    )
    return delta, domain, general


def audit_signed_case_consistency(
    *,
    bank_bytes: bytes,
    expected_bank_sha256: str,
    run_start_bytes: bytes,
    expected_start_sha256: str,
    bundle_bytes: bytes,
    bundle_signature_hex: str,
    evaluator_key_hex: str,
    release_key_hex: str,
    expected_candidate_sha256: str,
    expected_inference_sha256: str,
    expected_tokenizer_sha256: str,
    card: ModelCard,
) -> CaseConsistencyAudit:
    """Check exact signed inputs and re-score cases; never authorize serving.

    The caller must separately prove the bank's custody and timestamp, the
    run-start timestamp and first-call order, evaluator independence, and
    signed release inventory. Passing here is only internal consistency.
    """
    if (
        not bank_bytes
        or len(bank_bytes) > MAX_BANK_BYTES
        or not run_start_bytes
        or len(run_start_bytes) > MAX_START_BYTES
        or not bundle_bytes
        or len(bundle_bytes) > MAX_BUNDLE_BYTES
    ):
        raise EvalEvidenceError("missing or oversized evaluation artifact")
    _pin(bank_bytes, expected_bank_sha256, "held-out bank")
    _pin(run_start_bytes, expected_start_sha256, "run-start challenge")
    _verify_signature(bundle_bytes, bundle_signature_hex, evaluator_key_hex, release_key_hex)
    bank = _parse(bank_bytes, HeldoutBankManifest, limit=MAX_BANK_BYTES)
    start = _parse(run_start_bytes, EvalRunStartManifest, limit=MAX_START_BYTES)
    bundle = _parse(bundle_bytes, SignedCaseBundle, limit=MAX_BUNDLE_BYTES)
    _check_run_pins(
        start,
        bundle,
        bank_sha256=expected_bank_sha256,
        start_sha256=expected_start_sha256,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_inference_sha256=expected_inference_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        evaluator_key_hex=evaluator_key_hex,
        card=card,
    )
    delta, domain, general = _rescore(bank, bundle)
    if delta != card.eval_delta:
        raise EvalEvidenceError("recomputed case scores differ from signed model card")
    if not delta.ship_eligible or not domain.significant_improvement:
        raise EvalEvidenceError("paired case scores do not satisfy the statistical ship gate")
    return CaseConsistencyAudit(
        claimed_data_label=bundle.data_label,
        bank_sha256=expected_bank_sha256,
        run_start_sha256=expected_start_sha256,
        bundle_sha256=_sha256(bundle_bytes),
        computed_delta=delta,
        domain_comparison=domain,
        general_comparison=general,
    )
