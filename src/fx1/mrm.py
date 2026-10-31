"""Model-risk evidence pack — the regulatory moat, compiled.

SR 26-2 (April 2026) explicitly excluded generative/agentic AI from model-risk
guidance, leaving institutions to build their own frameworks around the
surviving expectations: documented development, independent validation,
outcomes analysis, ongoing monitoring, governance. ``fx1 mrm`` compiles a
checkpoint's existing artifacts into that five-activity dossier structure —
fx-1 is the model a validation function can approve cheapest, because the
evidence already exists.

Every dossier section cites the artifact hashes it was compiled from; a
dossier referencing a missing artifact fails closed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from fx1.modelcard import ModelCard
from quant_fund.utils.atomicio import atomic_write_text

FIVE_ACTIVITIES = (
    "development",
    "implementation",
    "validation",
    "monitoring",
    "governance",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DossierSection(BaseModel):
    activity: str
    artifact_hashes: dict[str, str] = Field(
        description="artifact path -> sha256 backing this activity"
    )
    summary: str


class MRMDossier(BaseModel):
    """The compiled five-activity model-risk dossier."""

    model_version: str
    compiled_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_model: str
    sections: list[DossierSection]
    contamination_flagged: bool
    ship_eligible: bool
    disclaimer: str = (
        "Research-scoped evidence pack. fx-1 and the dipcatcher harness "
        "produce research, backtest, or simulated evidence only; this dossier "
        "supports — and does not replace — the institution's independent "
        "validation. Not investment advice."
    )

    @property
    def complete(self) -> bool:
        """True only when every activity carries activity-specific evidence.

        The model card is added to every section's backing, so an activity
        is evidenced iff its section pins more than the card alone — except
        ``governance``, whose evidence *is* the signed card.
        """
        return all(len(s.artifact_hashes) > 1 or s.activity == "governance" for s in self.sections)


def compile_dossier(
    *,
    modelcard_path: str | Path,
    artifacts: dict[str, str | Path],
    out_path: str | Path,
) -> MRMDossier:
    """Compile the dossier. Fail-closed on missing artifacts or activities.

    *artifacts* maps activity name -> artifact path (JSON/MD). The model card
    itself always backs ``governance``. A contamination report, if provided
    under key ``contamination_report``, is parsed for its overall flag.
    """
    card = ModelCard.load(modelcard_path)
    hashes: dict[str, dict[str, str]] = {}
    contamination_flagged = False
    for activity, artifact in artifacts.items():
        path = Path(artifact)
        if not path.exists():
            raise FileNotFoundError(f"dossier artifact for {activity!r} missing: {path}")
        report_activity = "validation" if activity == "contamination_report" else activity
        hashes.setdefault(report_activity, {})[str(path)] = _sha(path)
        declared = activity == "contamination_report" or "contamination" in path.name
        report: object = None
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            report = None
        if isinstance(report, dict) and "overall_flagged" in report:
            # A dict carrying the field is contamination evidence wherever
            # it sits — renaming the file or the activity cannot launder it.
            contamination_flagged = contamination_flagged or bool(report["overall_flagged"])
        elif declared:
            # A declared contamination artifact that fails to parse, or a
            # valid JSON non-dict ("[]", "null") with no overall_flagged
            # field, cannot certify the corpus clean.
            contamination_flagged = True
    sections: list[DossierSection] = []
    summaries = {
        "development": (
            f"fx-1 {card.version} fine-tuned from {card.base_model}; corpus "
            f"sha256 {card.corpus_sha256[:16]}…, training manifest "
            f"{card.training_manifest_sha256[:16]}…. Provenance-hashed corpus "
            "with documented quality gates."
        ),
        "implementation": (
            "Serving path enforces model-card ship gate and (when configured) "
            "release signatures; backends fail closed."
        ),
        "validation": (
            f"Ship gate: domain {card.eval_delta.domain_pass_rate_base:.2f} → "
            f"{card.eval_delta.domain_pass_rate_candidate:.2f}, general "
            f"{card.eval_delta.general_pass_rate_base:.2f} → "
            f"{card.eval_delta.general_pass_rate_candidate:.2f}, honesty "
            f"gate {'passed' if card.eval_delta.honesty_gate_candidate else 'FAILED'}. "
            f"Contamination audit flagged: {contamination_flagged}."
        ),
        "monitoring": (
            "Red-team suites, shadow/leaderboard harness surfaces, and "
            "telemetry hooks provide ongoing-monitoring inputs."
        ),
        "governance": (
            f"Model card {card.version}, license tier {card.license_tier}, "
            "research-scoped by construction (live_pnl_claim=false)."
        ),
    }
    for activity in FIVE_ACTIVITIES:
        backing = dict(hashes.get(activity, {}))
        backing[str(modelcard_path)] = _sha(Path(modelcard_path))
        sections.append(
            DossierSection(
                activity=activity,
                artifact_hashes=backing,
                summary=summaries[activity],
            )
        )
    is_complete = all(
        activity in hashes or activity == "governance" for activity in FIVE_ACTIVITIES
    )
    dossier = MRMDossier(
        model_version=card.version,
        base_model=card.base_model,
        sections=sections,
        contamination_flagged=contamination_flagged,
        # A flagged contamination audit invalidates ship eligibility even
        # when the card's eval delta passed — and so does an incomplete
        # dossier: the five-activity pack cannot certify a checkpoint on
        # evidence it never pinned.
        ship_eligible=(card.eval_delta.ship_eligible and not contamination_flagged and is_complete),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, dossier.model_dump_json(indent=2))
    return dossier
