"""fx-1 model cards — versioned, evidence-bound checkpoint metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SYNTHETIC_FIXTURE_ENV = "FX1_SYNTHETIC_FIXTURE"

_FIXTURE_PLACEHOLDER_EVAL: dict[str, object] = {
    "domain_pass_rate_base": 0.5,
    "domain_pass_rate_candidate": 0.51,
    "general_pass_rate_base": 0.9,
    "general_pass_rate_candidate": 0.9,
    "honesty_gate_candidate": True,
}


class EvalDelta(BaseModel):
    """Eval comparison of a candidate checkpoint against its base."""

    domain_pass_rate_base: float = Field(ge=0, le=1)
    domain_pass_rate_candidate: float = Field(ge=0, le=1)
    general_pass_rate_base: float = Field(ge=0, le=1)
    general_pass_rate_candidate: float = Field(ge=0, le=1)
    honesty_gate_candidate: bool

    @property
    def is_placeholder(self) -> bool:
        """True for the labelled SYNTHETIC fixture delta, not a measured pair."""
        return self.model_dump() == _FIXTURE_PLACEHOLDER_EVAL

    @property
    def ship_eligible(self) -> bool:
        """Ship gate: honesty native, domain better, general not regressed."""
        if self.is_placeholder:
            return False
        return (
            self.honesty_gate_candidate
            and self.domain_pass_rate_candidate > self.domain_pass_rate_base
            and self.general_pass_rate_candidate >= self.general_pass_rate_base
        )


def synthetic_fixture_from_env() -> bool:
    """True when the process explicitly opted into SYNTHETIC fixture serving."""
    return os.environ.get(SYNTHETIC_FIXTURE_ENV) == "1"


def eval_fixture_placeholder_delta() -> EvalDelta:
    """Synthetic pass-rate placeholders on the committed tiny-LM fixture card."""
    return EvalDelta.model_validate(_FIXTURE_PLACEHOLDER_EVAL)


class ModelCard(BaseModel):
    """Versioned fx-1 checkpoint card."""

    version: str = Field(pattern=r"^fx-1\.v\d+\.\d+$")
    base_model: str = "moonshotai/Kimi-K3"
    base_checkpoint_sha256: str | None = None
    corpus_sha256: str = Field(min_length=64, max_length=64)
    corpus_receipt_range: str = Field(
        description="First..last receipt SHA-256 prefix covered by the corpus"
    )
    training_manifest_sha256: str = Field(min_length=64, max_length=64)
    eval_delta: EvalDelta
    license_tier: str = "internal_research"
    known_limits: list[str] = Field(default_factory=list)
    public_model_id: Literal["fx1", "fx1-lite"] | None = None
    fixture_only: bool = False
    live_pnl_claim: bool = False
    research_only: bool = True

    @model_validator(mode="after")
    def _never_live(self) -> ModelCard:
        if self.live_pnl_claim or not self.research_only:
            raise ValueError(
                "fx-1 model cards are research-scoped by construction: "
                "live_pnl_claim must be false and research_only true"
            )
        return self

    @property
    def is_synthetic_placeholder(self) -> bool:
        """True for the labelled SYNTHETIC fixture card, not a measured release."""
        if self.fixture_only or self.eval_delta.is_placeholder:
            return True
        return any(
            "synthetic ship-gate placeholders" in limit.lower() for limit in self.known_limits
        )

    @property
    def ship_eligible(self) -> bool:
        if self.is_synthetic_placeholder:
            return False
        return self.eval_delta.ship_eligible

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
