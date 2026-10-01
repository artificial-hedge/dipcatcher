"""Offline causal narrative learner and observed-proxy covariance diagnostics.

Method: Say, Echo, Do, https://arxiv.org/abs/2609.38545, Section 4.2.
Official MIT reference: https://github.com/AliAtiah/say-echo-do/tree/
4a201e77f8b50c423e9aae156125bea386984d50. The published detection AUC is from
simulated markets. This original small token/projection learner implements the
return-aligned neighbour cross entropy, with an additional direction head. It
does not reproduce the entire paper, its TF-IDF experiment, Hawkes parentage,
signatures, pretrained language models, trading or an empirical performance claim.

Echo parentage is supplied, timestamped and independently auditable, never inferred
as fact by similarity. Latest *available* text/link revisions are selected at each
decision. Vocabulary and outcome scales use training only; calibration/holdout
are disjoint later episodes with target-window purge and embargo. Full-batch
contrastive learning is bounded, uses CPU torch lazily, and leaves global RNG and
thread settings untouched. Baselines share windows/vocabulary/update counts;
bag-of-words has different capacity and binary-only supervision; no-echo omits a
modality. These differences are disclosed, not a compute/richness-matched claim.

Say/Echo--Do covariance requires supplied tone observations and explicitly linked,
independently timed position/disclosure proxies. Missing Do is unavailable.
Disclosed holdings changes are not actual transactions or hidden intent. Neither
the theory's sign identities nor deception/ownership claims are applied to these
proxies. Source authenticity, independent labels, linkage accuracy and data rights
remain unverified. Originals must be retained externally to reproduce training;
model JSON binds their content/source/clock hashes without acquiring any data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

type Array = NDArray[np.float64]
Voice = Literal["say", "echo"]
Arm = Literal["narrative", "no_echo", "bow"]
_ARMS = ("narrative", "no_echo", "bow")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_SCHEMA = "causal_narrative.v1"
_PIN = "4a201e77f8b50c423e9aae156125bea386984d50"
_MODEL_BYTES = 8_000_000


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _name(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a bounded nonempty string")


def _sha(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA256")


def _number(value: float, field: str, bound: float = 1e9) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or abs(value) > bound
    ):
        raise ValueError(f"{field} must be a bounded finite number")


def _count(value: int, field: str, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low},{high}]")


def _clock(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _provenance(value: Any) -> None:
    _name(value.source_id, "source_id")
    _sha(value.source_sha256, "source_sha256")
    if not isinstance(value.synthetic, bool) or not isinstance(value.derived, bool):
        raise ValueError("synthetic and derived declarations must be boolean")


def _public_clocks(value: Any, event: str | None = None) -> None:
    published = _clock(value.published_at, "published_at")
    ingested = _clock(value.ingested_at, "ingested_at")
    if ingested < published:
        raise ValueError("ingestion must follow publication")
    if event is not None:
        time = _clock(getattr(value, event), event)
        if time > published:
            raise ValueError("event must precede publication")
        object.__setattr__(value, event, time)
    object.__setattr__(value, "published_at", published)
    object.__setattr__(value, "ingested_at", ingested)


@dataclass(frozen=True, slots=True)
class TextRevision:
    document_id: str
    revision_id: str
    entity_id: str
    speaker_id: str
    voice: Voice
    text: str
    event_time: datetime
    published_at: datetime
    ingested_at: datetime
    source_id: str
    source_sha256: str
    synthetic: bool
    derived: bool = False

    def __post_init__(self) -> None:
        for name in ("document_id", "revision_id", "entity_id", "speaker_id"):
            _name(getattr(self, name), name)
        if self.voice not in ("say", "echo"):
            raise ValueError("voice must be say or echo")
        if (
            not isinstance(self.text, str)
            or not 1 <= len(self.text) <= 8192
            or not _TOKEN.findall(self.text)
        ):
            raise ValueError("original text must contain bounded tokens")
        _public_clocks(self, "event_time")
        _provenance(self)


@dataclass(frozen=True, slots=True)
class EchoLink:
    link_id: str
    echo_document_id: str
    statement_document_id: str
    entity_id: str
    institution_id: str
    published_at: datetime
    ingested_at: datetime
    source_id: str
    source_sha256: str
    synthetic: bool
    derived: bool = False

    def __post_init__(self) -> None:
        for name in (
            "link_id",
            "echo_document_id",
            "statement_document_id",
            "entity_id",
            "institution_id",
        ):
            _name(getattr(self, name), name)
        if self.echo_document_id == self.statement_document_id:
            raise ValueError("echo cannot be its own statement parent")
        _public_clocks(self)
        _provenance(self)


@dataclass(frozen=True, slots=True)
class NarrativeWindow:
    window_id: str
    episode_id: str
    entity_id: str
    institution_id: str
    decision_time: datetime
    texts: tuple[TextRevision, ...]
    links: tuple[EchoLink, ...] = ()

    def __post_init__(self) -> None:
        for name in ("window_id", "episode_id", "entity_id", "institution_id"):
            _name(getattr(self, name), name)
        object.__setattr__(self, "decision_time", _clock(self.decision_time, "decision_time"))
        for values, kind, limit in ((self.texts, TextRevision, 256), (self.links, EchoLink, 256)):
            if (
                not isinstance(values, tuple)
                or len(values) > limit
                or any(not isinstance(v, kind) for v in values)
            ):
                raise ValueError("window records must be bounded immutable typed tuples")
            if any(v.entity_id != self.entity_id for v in values):
                raise ValueError("window entity alignment mismatch")
        if len({(v.document_id, v.revision_id) for v in self.texts}) != len(self.texts):
            raise ValueError("duplicate document revision")
        if len({v.link_id for v in self.links}) != len(self.links):
            raise ValueError("duplicate echo link_id")
        if any(v.voice == "say" and v.speaker_id != self.institution_id for v in self.texts) or any(
            v.institution_id != self.institution_id for v in self.links
        ):
            raise ValueError("statement/echo institution alignment mismatch")


@dataclass(frozen=True, slots=True)
class ReturnExample:
    window: NarrativeWindow
    target_return: float
    target_event_time: datetime
    target_available_time: datetime
    source_id: str
    source_sha256: str
    synthetic: bool
    derived: bool = False
    constructed_from_text: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.window, NarrativeWindow):
            raise ValueError("typed narrative window required")
        _number(self.target_return, "target simple return", 1e3)
        if self.target_return < -1:
            raise ValueError("simple returns cannot be below -1")
        event, available = (
            _clock(self.target_event_time, "target event"),
            _clock(self.target_available_time, "target availability"),
        )
        if event <= self.window.decision_time or available < event:
            raise ValueError("target clocks must satisfy decision<event<=availability")
        object.__setattr__(self, "target_event_time", event)
        object.__setattr__(self, "target_available_time", available)
        _provenance(self)
        if not isinstance(self.constructed_from_text, bool):
            raise ValueError("label independence declaration must be boolean")


@dataclass(frozen=True, slots=True)
class NarrativeConfig:
    horizon: timedelta = timedelta(days=1)
    lookback: timedelta = timedelta(days=30)
    publication_delay: timedelta = timedelta(0)
    embargo: timedelta = timedelta(0)
    embedding_dim: int = 8
    max_vocab: int = 256
    max_tokens: int = 64
    max_documents: int = 16
    max_rows: int = 128
    steps: int = 60
    learning_rate: float = 0.02
    temperature: float = 0.3
    outcome_bandwidth: float = 1.0
    contrastive_weight: float = 1.0
    seed: int = 7
    pair_operation_budget: int = 10_000_000

    def __post_init__(self) -> None:
        for name in ("horizon", "lookback"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or not timedelta(0) < value <= timedelta(days=365):
                raise ValueError("horizon/lookback must be positive bounded durations")
        for name in ("publication_delay", "embargo"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or not timedelta(0) <= value <= timedelta(days=365):
                raise ValueError("delay/embargo must be bounded nonnegative durations")
        for name, low, high in (
            ("embedding_dim", 2, 32),
            ("max_vocab", 8, 512),
            ("max_tokens", 1, 256),
            ("max_documents", 1, 64),
            ("max_rows", 6, 256),
            ("steps", 1, 300),
            ("seed", 0, 2**32 - 1),
            ("pair_operation_budget", 1, 20_000_000),
        ):
            _count(getattr(self, name), name, low, high)
        for name in ("learning_rate", "temperature", "outcome_bandwidth", "contrastive_weight"):
            _number(getattr(self, name), name)
        if (
            not 1e-5 <= self.learning_rate <= 0.2
            or not 0.01 <= self.temperature <= 10
            or not 0.05 <= self.outcome_bandwidth <= 10
            or not 0.01 <= self.contrastive_weight <= 10
        ):
            raise ValueError("training hyperparameters exceed numeric bounds")


@dataclass(frozen=True, slots=True)
class WindowSnapshot:
    window_id: str
    episode_id: str
    entity_id: str
    institution_id: str
    decision_time: datetime
    say: tuple[TextRevision, ...]
    echo: tuple[TextRevision, ...]
    links: tuple[EchoLink, ...]
    excluded: tuple[tuple[str, str], ...]
    snapshot_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        # Excluded future suffixes are diagnostics, not part of a causal input hash.
        object.__setattr__(
            self,
            "snapshot_sha256",
            _hash(
                {
                    name: getattr(self, name)
                    for name in (
                        "window_id",
                        "episode_id",
                        "entity_id",
                        "institution_id",
                        "decision_time",
                        "say",
                        "echo",
                        "links",
                    )
                }
            ),
        )

    @property
    def synthetic(self) -> bool:
        return any(v.synthetic for v in self.say + self.echo + self.links)

    @property
    def derived(self) -> bool:
        return any(v.derived for v in self.say + self.echo + self.links)


def snapshot_window(window: NarrativeWindow, config: NarrativeConfig) -> WindowSnapshot:
    if not isinstance(window, NarrativeWindow) or not isinstance(config, NarrativeConfig):
        raise ValueError("typed window/config required")
    cutoff, excluded = window.decision_time, []
    documents: dict[str, TextRevision] = {}
    for doc in sorted(window.texts, key=lambda d: (d.published_at, d.ingested_at, d.revision_id)):
        if max(doc.published_at + config.publication_delay, doc.ingested_at) > cutoff:
            excluded.append((doc.revision_id, "unpublished_or_uningested"))
            continue
        if not cutoff - config.lookback <= doc.event_time <= cutoff:
            excluded.append((doc.revision_id, "outside_trailing_window"))
            continue
        previous_doc = documents.get(doc.document_id)
        if previous_doc is not None and (previous_doc.published_at, previous_doc.ingested_at) == (
            doc.published_at,
            doc.ingested_at,
        ):
            raise ValueError("ambiguous same-clock document revisions")
        if previous_doc is not None and (
            previous_doc.voice != doc.voice
            or previous_doc.speaker_id != doc.speaker_id
            or previous_doc.event_time != doc.event_time
        ):
            raise ValueError("revision cannot change document speaker/voice/original event")
        documents[doc.document_id] = doc
    links: dict[str, EchoLink] = {}
    for link in sorted(window.links, key=lambda v: (v.published_at, v.ingested_at, v.link_id)):
        if max(link.published_at + config.publication_delay, link.ingested_at) > cutoff:
            continue
        previous_link = links.get(link.echo_document_id)
        if previous_link is not None and (
            previous_link.published_at,
            previous_link.ingested_at,
        ) == (
            link.published_at,
            link.ingested_at,
        ):
            raise ValueError("ambiguous same-clock echo links")
        links[link.echo_document_id] = link
    say = tuple(d for _, d in sorted(documents.items()) if d.voice == "say")
    echoes, used = [], []
    for _, doc in sorted(documents.items()):
        if doc.voice != "echo":
            continue
        used_link = links.get(doc.document_id)
        parent = documents.get(used_link.statement_document_id) if used_link else None
        if parent is None:
            excluded.append((doc.document_id, "no_available_explicit_statement_link"))
            continue
        # A correction can be published after an echo of the original statement.
        # Require an observed earlier revision, not a future correction as its parent.
        earliest_publication = min(
            revision.published_at
            for revision in window.texts
            if revision.document_id == parent.document_id
            and max(revision.published_at + config.publication_delay, revision.ingested_at)
            <= cutoff
        )
        if parent.voice != "say" or earliest_publication > doc.published_at:
            raise ValueError("echo parent must be an earlier institutional statement")
        assert used_link is not None
        echoes.append(doc)
        used.append(used_link)
    if not say:
        raise ValueError("no visible institutional statement")
    if len(say) + len(echoes) > config.max_documents:
        raise ValueError("visible document resource bound exceeded")
    return WindowSnapshot(
        window.window_id,
        window.episode_id,
        window.entity_id,
        window.institution_id,
        cutoff,
        say,
        tuple(echoes),
        tuple(used),
        tuple(excluded),
    )


@dataclass(frozen=True, slots=True)
class AlignmentLoss:
    mean_cross_entropy: float
    mean_entropy_bound: float
    per_anchor_excess_kl: tuple[float, ...]


def contrastive_alignment(
    embeddings: Array, outcomes: Array, *, temperature: float, bandwidth: float
) -> AlignmentLoss:
    """Mean per-anchor soft neighbour CE (paper uses a sum; only scale differs)."""
    z, y = np.asarray(embeddings, dtype=float), np.asarray(outcomes, dtype=float)
    _number(temperature, "temperature")
    _number(bandwidth, "bandwidth")
    if not 0.01 <= temperature <= 10 or not 0.05 <= bandwidth <= 10:
        raise ValueError("contrastive bandwidth/temperature bounds")
    if (
        z.ndim != 2
        or y.ndim != 2
        or not 3 <= len(z) <= 512
        or len(y) != len(z)
        or not 1 <= z.shape[1] <= 128
        or not 1 <= y.shape[1] <= 16
        or not np.isfinite(z).all()
        or not np.isfinite(y).all()
        or np.abs(z).max() > 1e6
        or np.abs(y).max() > 1e6
    ):
        raise ValueError("contrastive arrays must have bounded finite aligned shapes")
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("contrastive embeddings must have nonzero norm")
    z = z / norms
    n = len(z)
    mask: NDArray[np.bool_] = ~np.eye(n, dtype=bool)
    distance = ((y[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    q_logits = (-distance / (2 * bandwidth**2))[mask].reshape(n, n - 1)
    logq = q_logits - logsumexp(q_logits, axis=1, keepdims=True)
    q = np.exp(logq)
    logits = (z @ z.T / temperature)[mask].reshape(n, n - 1)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    ce, entropy = -(q * logp).sum(1), -(q * logq).sum(1)
    return AlignmentLoss(
        float(ce.mean()), float(entropy.mean()), tuple(float(v) for v in ce - entropy)
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("narrative learning requires the optional nn extra (torch)") from exc
    return torch


def _alignment_torch(z: Any, y: Any, config: NarrativeConfig, torch: Any) -> Any:
    n = len(z)
    mask = ~torch.eye(n, dtype=torch.bool, device="cpu")
    distances = ((y[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    target = torch.softmax(
        (-distances / (2 * config.outcome_bandwidth**2))[mask].reshape(n, n - 1), dim=1
    )
    log_neighbors = torch.log_softmax((z @ z.T / config.temperature)[mask].reshape(n, n - 1), dim=1)
    return -(target * log_neighbors).sum(1).mean()


def _config_payload(config: NarrativeConfig) -> dict[str, Any]:
    result = asdict(config)
    for name in ("horizon", "lookback", "publication_delay", "embargo"):
        result[name] = getattr(config, name).total_seconds()
    return result


def _row_payload(row: ReturnExample, snapshot: WindowSnapshot) -> dict[str, Any]:
    docs = []
    for doc in snapshot.say + snapshot.echo:
        payload = asdict(doc)
        payload["text_sha256"] = hashlib.sha256(payload.pop("text").encode()).hexdigest()
        docs.append(payload)
    return {
        "window_id": row.window.window_id,
        "episode_id": row.window.episode_id,
        "entity_id": row.window.entity_id,
        "institution_id": row.window.institution_id,
        "decision_time": row.window.decision_time,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "documents": docs,
        "links": [asdict(v) for v in snapshot.links],
        "target_return": row.target_return,
        "target_event_time": row.target_event_time,
        "target_available_time": row.target_available_time,
        "label_source_id": row.source_id,
        "label_source_sha256": row.source_sha256,
        "label_synthetic": row.synthetic,
        "label_derived": row.derived,
        "label_constructed_from_text": row.constructed_from_text,
    }


def _validate_target_partition(
    rows: tuple[ReturnExample, ...], config: NarrativeConfig, cutoff: datetime, minimum: int
) -> None:
    if (
        not isinstance(rows, tuple)
        or not minimum <= len(rows) <= config.max_rows
        or any(not isinstance(r, ReturnExample) for r in rows)
    ):
        raise ValueError("examples must be bounded immutable typed rows")
    if len({r.window.window_id for r in rows}) != len(rows) or len(
        {r.window.episode_id for r in rows}
    ) != len(rows):
        raise ValueError("windows/episodes must be unique and disjoint")
    for row in rows:
        if (
            row.target_event_time - row.window.decision_time != config.horizon
            or row.target_available_time > cutoff
        ):
            raise ValueError("target horizon mismatch or label unpublished at cutoff")
        if row.constructed_from_text:
            raise ValueError("return labels cannot be constructed from text")
    groups: dict[str, list[ReturnExample]] = {}
    for row in rows:
        groups.setdefault(row.window.entity_id, []).append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda r: r.window.decision_time)
        if any(
            a.target_event_time + config.embargo > b.window.decision_time
            for a, b in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError("target intervals overlap or violate embargo")


def _validate_examples(
    rows: tuple[ReturnExample, ...], config: NarrativeConfig, cutoff: datetime, minimum: int
) -> tuple[WindowSnapshot, ...]:
    _validate_target_partition(rows, config, cutoff, minimum)
    return tuple(snapshot_window(r.window, config) for r in rows)


def _persisted_clock(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("persisted clock must be an ISO timestamp")
    return _clock(datetime.fromisoformat(value), "persisted clock")


def _persisted_rows(
    payloads: Any, config: NarrativeConfig, cutoff: datetime, minimum: int
) -> tuple[ReturnExample, ...]:
    """Validate redacted provenance; originals are needed to verify text/hash truth.

    Latest statement revisions can postdate an echo of an earlier revision. The
    redacted artifact cannot prove that earlier publication or echo linkage, so
    this check does not assert original-source authenticity or replay training.
    """
    if not isinstance(payloads, list) or not minimum <= len(payloads) <= config.max_rows:
        raise ValueError("persisted row partition resource mismatch")
    row_fields = {
        "window_id",
        "episode_id",
        "entity_id",
        "institution_id",
        "decision_time",
        "snapshot_sha256",
        "documents",
        "links",
        "target_return",
        "target_event_time",
        "target_available_time",
        "label_source_id",
        "label_source_sha256",
        "label_synthetic",
        "label_derived",
        "label_constructed_from_text",
    }
    rows = []
    for payload in payloads:
        if not isinstance(payload, dict) or set(payload) != row_fields:
            raise ValueError("persisted row schema mismatch")
        decision = _persisted_clock(payload["decision_time"])
        _sha(payload["snapshot_sha256"], "persisted snapshot_sha256")
        documents, links = payload["documents"], payload["links"]
        if (
            not isinstance(documents, list)
            or not 1 <= len(documents) <= config.max_documents
            or not isinstance(links, list)
            or len(links) > config.max_documents
        ):
            raise ValueError("persisted document/link resource mismatch")
        texts = []
        for document in documents:
            values = dict(document)
            _sha(values.pop("text_sha256"), "persisted text_sha256")
            for name in ("event_time", "published_at", "ingested_at"):
                values[name] = _persisted_clock(values[name])
            text = TextRevision(text="REDACTED ORIGINAL REQUIRED", **values)
            if (
                max(text.published_at + config.publication_delay, text.ingested_at) > decision
                or not decision - config.lookback <= text.event_time <= decision
            ):
                raise ValueError("persisted document unavailable at decision")
            texts.append(text)
        if len({d.document_id for d in texts}) != len(texts) or not any(
            d.voice == "say" for d in texts
        ):
            raise ValueError("persisted document identities mismatch")
        loaded_links = []
        for link in links:
            values = dict(link)
            for name in ("published_at", "ingested_at"):
                values[name] = _persisted_clock(values[name])
            loaded_link = EchoLink(**values)
            if (
                max(loaded_link.published_at + config.publication_delay, loaded_link.ingested_at)
                > decision
            ):
                raise ValueError("persisted echo link unavailable at decision")
            loaded_links.append(loaded_link)
        text_by_id = {d.document_id: d for d in texts}
        echoes = {d.document_id for d in texts if d.voice == "echo"}
        if (
            {v.echo_document_id for v in loaded_links} != echoes
            or len(loaded_links) != len(echoes)
            or any(
                v.statement_document_id not in text_by_id
                or text_by_id[v.statement_document_id].voice != "say"
                for v in loaded_links
            )
        ):
            raise ValueError("persisted echo parent identity mismatch")
        window = NarrativeWindow(
            payload["window_id"],
            payload["episode_id"],
            payload["entity_id"],
            payload["institution_id"],
            decision,
            tuple(texts),
            tuple(loaded_links),
        )
        rows.append(
            ReturnExample(
                window,
                payload["target_return"],
                _persisted_clock(payload["target_event_time"]),
                _persisted_clock(payload["target_available_time"]),
                payload["label_source_id"],
                payload["label_source_sha256"],
                payload["label_synthetic"],
                payload["label_derived"],
                payload["label_constructed_from_text"],
            )
        )
    result = tuple(rows)
    _validate_target_partition(result, config, cutoff, minimum)
    return result


def _features(
    snapshots: tuple[WindowSnapshot, ...], vocabulary: tuple[str, ...], config: NarrativeConfig
) -> Array:
    lookup, v = {word: i for i, word in enumerate(vocabulary)}, len(vocabulary)
    result = np.zeros((len(snapshots), 2 * v + 2))
    for j, snapshot in enumerate(snapshots):
        for channel, documents in enumerate((snapshot.say, snapshot.echo)):
            for doc in documents:
                for token in _TOKEN.findall(doc.text.lower())[: config.max_tokens]:
                    result[j, channel * v + lookup.get(token, 0)] += 1
            total = float(result[j, channel * v : (channel + 1) * v].sum())
            if total:
                result[j, channel * v : (channel + 1) * v] /= total
            result[j, 2 * v + channel] = float(bool(documents))
    return result


def _shapes(v: int, d: int, arm: Arm) -> dict[str, tuple[int, ...]]:
    if arm == "bow":
        return {"head": (2 * v + 2,), "bias": (1,)}
    return {
        "tokens": (v, d),
        "projection": (2 * d + 2, d),
        "projection_bias": (d,),
        "head": (d,),
        "bias": (1,),
    }


def _initial(v: int, config: NarrativeConfig, arm: Arm, torch: Any) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    return {
        name: (
            torch.zeros(shape, dtype=torch.float64, device="cpu")
            if "bias" in name
            else torch.randn(shape, dtype=torch.float64, device="cpu", generator=generator) * 0.1
        ).requires_grad_(True)
        for name, shape in _shapes(v, config.embedding_dim, arm).items()
    }


def _forward(parameters: dict[str, Any], x: Any, v: int, arm: Arm, torch: Any) -> tuple[Any, Any]:
    if arm == "bow":
        return x, x @ parameters["head"] + parameters["bias"][0]
    say = x[:, :v] @ parameters["tokens"]
    echo = x[:, v : 2 * v] @ parameters["tokens"] if arm == "narrative" else torch.zeros_like(say)
    presence = (
        x[:, 2 * v :]
        if arm == "narrative"
        else torch.stack((x[:, 2 * v], torch.zeros_like(x[:, 2 * v])), dim=1)
    )
    combined = torch.cat((say, echo, presence), dim=1)
    latent = torch.tanh(combined @ parameters["projection"] + parameters["projection_bias"])
    latent = torch.nn.functional.normalize(latent, dim=1, eps=1e-12)
    return latent, latent @ parameters["head"] + parameters["bias"][0]


@dataclass(frozen=True, slots=True)
class TensorState:
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _name(self.name, "parameter name")
        if not isinstance(self.shape, tuple) or not 1 <= len(self.shape) <= 2:
            raise ValueError("parameter shape must be immutable")
        for size in self.shape:
            _count(size, "parameter size", 1, 2048)
        if (
            not isinstance(self.values, tuple)
            or len(self.values) != math.prod(self.shape)
            or len(self.values) > 32768
        ):
            raise ValueError("parameter values/shape resource mismatch")
        for value in self.values:
            _number(value, "parameter", 1e6)


def _freeze(parameters: dict[str, Any]) -> tuple[TensorState, ...]:
    return tuple(
        TensorState(
            name, tuple(p.shape), tuple(float(v) for v in p.detach().cpu().flatten().tolist())
        )
        for name, p in parameters.items()
    )


def _thaw(states: tuple[TensorState, ...], torch: Any) -> dict[str, Any]:
    return {
        state.name: torch.tensor(
            np.array(state.values).reshape(state.shape), dtype=torch.float64, device="cpu"
        )
        for state in states
    }


def _proper(probabilities: Array, labels: Array) -> dict[str, float]:
    p = np.clip(probabilities, 1e-9, 1 - 1e-9)
    ece = 0.0
    for bin_index in range(10):
        mask = (probabilities >= bin_index / 10) & (
            (probabilities < (bin_index + 1) / 10) if bin_index < 9 else (probabilities <= 1)
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(probabilities[mask].mean() - labels[mask].mean()))
    return {
        "brier": float(((probabilities - labels) ** 2).mean()),
        "binary_log_score": float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()),
        "ece": ece,
    }


@dataclass(frozen=True, slots=True)
class DirectionForecast:
    window_ids: tuple[str, ...]
    probability_up: tuple[float, ...]
    raw_logits: tuple[float, ...]
    embeddings: tuple[tuple[float, ...], ...]
    arm: Arm
    calibration_status: str
    model_sha256: str
    input_sha256: str
    synthetic: bool
    derived: bool
    forecast_sha256: str = dataclass_field(init=False)
    market_evidence: bool = dataclass_field(init=False, default=False)

    def __post_init__(self) -> None:
        n = len(self.window_ids)
        if not n or any(
            not isinstance(value, tuple) or len(value) != n
            for value in (self.window_ids, self.probability_up, self.raw_logits, self.embeddings)
        ):
            raise ValueError("forecast fields must be aligned immutable tuples")
        for value in self.probability_up:
            _number(value, "probability", 1)
            if not 0 <= value <= 1:
                raise ValueError("probability must lie in [0,1]")
        for value in self.raw_logits:
            _number(value, "logit", 1e6)
        for row in self.embeddings:
            if not isinstance(row, tuple) or not row:
                raise ValueError("embedding must be an immutable vector")
            for value in row:
                _number(value, "embedding", 1e6)
        _sha(self.model_sha256, "model_sha256")
        _sha(self.input_sha256, "input_sha256")
        if (
            self.arm not in _ARMS
            or not isinstance(self.synthetic, bool)
            or not isinstance(self.derived, bool)
        ):
            raise ValueError("forecast identity mismatch")
        object.__setattr__(
            self,
            "forecast_sha256",
            _hash(
                {
                    f: getattr(self, f)
                    for f in (
                        "window_ids",
                        "probability_up",
                        "raw_logits",
                        "embeddings",
                        "arm",
                        "calibration_status",
                        "model_sha256",
                        "input_sha256",
                        "synthetic",
                        "derived",
                    )
                }
            ),
        )


class NarrativeLearner:
    def __init__(self, config: NarrativeConfig | None = None) -> None:
        self._config = NarrativeConfig() if config is None else config
        if not isinstance(self._config, NarrativeConfig):
            raise ValueError("typed NarrativeConfig required")
        self._body: dict[str, Any] = {}
        self._states: dict[str, tuple[TensorState, ...]] = {}
        self._vocabulary: tuple[str, ...] = ()
        self._model_sha256: str | None = None

    @property
    def config(self) -> NarrativeConfig:
        return self._config

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._vocabulary

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    def parameters(self, arm: Arm = "narrative") -> tuple[TensorState, ...]:
        if arm not in self._states:
            raise RuntimeError("fitted arm required")
        return self._states[arm]

    def fit(
        self,
        training: tuple[ReturnExample, ...],
        *,
        fit_cutoff: datetime,
        calibration: tuple[ReturnExample, ...] = (),
        calibration_cutoff: datetime | None = None,
    ) -> NarrativeLearner:
        if self._states:
            raise RuntimeError("fitted model is frozen; use a fresh learner")
        fit_cutoff = _clock(fit_cutoff, "fit_cutoff")
        snapshots = _validate_examples(training, self.config, fit_cutoff, 6)
        labels = np.array([float(r.target_return > 0) for r in training])
        if len(set(labels)) != 2:
            raise ValueError("training needs both return directions")
        if 2 * self.config.steps * len(training) ** 2 > self.config.pair_operation_budget:
            raise ValueError("contrastive pair operation resource budget exceeded")
        calibration_snapshots: tuple[WindowSnapshot, ...] = ()
        if calibration:
            if calibration_cutoff is None:
                raise ValueError("explicit calibration cutoff required")
            calibration_cutoff = _clock(calibration_cutoff, "calibration_cutoff")
            calibration_snapshots = _validate_examples(
                calibration, self.config, calibration_cutoff, 1
            )
            if (
                calibration_cutoff <= fit_cutoff
                or min(r.window.decision_time for r in calibration)
                < fit_cutoff + self.config.embargo
            ):
                raise ValueError("calibration must follow training cutoff and embargo")
            self._check_disjoint(training, calibration, snapshots, calibration_snapshots)
        elif calibration_cutoff is not None:
            raise ValueError("calibration cutoff needs calibration rows")
        words = Counter(
            token
            for snapshot in snapshots
            for doc in snapshot.say + snapshot.echo
            for token in _TOKEN.findall(doc.text.lower())[: self.config.max_tokens]
        )
        vocabulary = ("<UNK>",) + tuple(
            word
            for word, _ in sorted(words.items(), key=lambda v: (-v[1], v[0]))[
                : self.config.max_vocab - 1
            ]
        )
        x = _features(snapshots, vocabulary, self.config)
        targets = np.array([r.target_return for r in training])
        target_mean, target_scale = float(targets.mean()), float(targets.std())
        if target_scale < 1e-8:
            raise ValueError("target scale too small for return-aligned training")
        torch = _torch()
        tx = torch.tensor(x, dtype=torch.float64, device="cpu")
        ty = torch.tensor(labels, dtype=torch.float64, device="cpu")
        outcomes = torch.tensor(
            ((targets - target_mean) / target_scale)[:, None], dtype=torch.float64, device="cpu"
        )
        states: dict[str, tuple[TensorState, ...]] = {}
        histories: dict[str, Any] = {}
        initial_hashes: dict[str, str] = {}
        for arm_name in _ARMS:
            arm = cast(Arm, arm_name)
            params = _initial(len(vocabulary), self.config, arm, torch)
            initial_hashes[arm] = _hash([asdict(v) for v in _freeze(params)])
            optimizer = torch.optim.Adam(tuple(params.values()), lr=self.config.learning_rate)
            history = []
            for _ in range(self.config.steps):
                optimizer.zero_grad()
                z, logits = _forward(params, tx, len(vocabulary), arm, torch)
                binary = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty)
                alignment = (
                    _alignment_torch(z, outcomes, self.config, torch)
                    if arm != "bow"
                    else torch.tensor(0.0, dtype=torch.float64, device="cpu")
                )
                loss = binary + self.config.contrastive_weight * alignment
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite narrative training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tuple(params.values()), 10.0)
                optimizer.step()
                history.append(
                    {
                        "total": float(loss.detach()),
                        "return_aligned_ce": float(alignment.detach()) if arm != "bow" else None,
                        "binary_log_score": float(binary.detach()),
                    }
                )
            states[arm], histories[arm] = _freeze(params), history
        calibration_states: dict[str, Any] = {}
        for arm_name in _ARMS:
            arm = cast(Arm, arm_name)
            status, a, b = "UNAVAILABLE_NO_CALIBRATION", 1.0, 0.0
            if calibration:
                y_cal = np.array([float(r.target_return > 0) for r in calibration])
                status = "UNAVAILABLE_TOO_FEW_OR_SINGLE_CLASS"
                if len(calibration) >= 4 and len(set(y_cal)) == 2:
                    with torch.inference_mode():
                        _, logits = _forward(
                            _thaw(states[arm], torch),
                            torch.tensor(
                                _features(calibration_snapshots, vocabulary, self.config),
                                dtype=torch.float64,
                                device="cpu",
                            ),
                            len(vocabulary),
                            arm,
                            torch,
                        )
                    values = logits.numpy()

                    def objective(ab: Array, values: Array = values, y_cal: Array = y_cal) -> float:
                        scores = ab[0] * values + ab[1]
                        return float(
                            (np.logaddexp(0, scores) - y_cal * scores).mean() + 0.001 * (ab @ ab)
                        )

                    def gradient(ab: Array, values: Array = values, y_cal: Array = y_cal) -> Array:
                        residual = expit(ab[0] * values + ab[1]) - y_cal
                        return (
                            np.array([float((residual * values).mean()), float(residual.mean())])
                            + 0.002 * ab
                        )

                    fit = minimize(
                        objective,
                        np.array([1.0, 0.0]),
                        jac=gradient,
                        method="BFGS",
                        options={"maxiter": 200, "gtol": 1e-7},
                    )
                    if fit.success and np.isfinite(fit.x).all():
                        a, b, status = (
                            float(fit.x[0]),
                            float(fit.x[1]),
                            "PLATT_CALIBRATED_LATER_SPLIT",
                        )
                    else:
                        status = "UNAVAILABLE_OPTIMIZATION_FAILED"
            calibration_states[arm] = {"status": status, "slope": a, "intercept": b}
        row_payloads = [_row_payload(r, s) for r, s in zip(training, snapshots, strict=True)]
        cal_payloads = [
            _row_payload(r, s) for r, s in zip(calibration, calibration_snapshots, strict=True)
        ]
        synthetic = any(
            r.synthetic or s.synthetic
            for r, s in zip(training + calibration, snapshots + calibration_snapshots, strict=True)
        )
        derived = any(
            r.derived or s.derived
            for r, s in zip(training + calibration, snapshots + calibration_snapshots, strict=True)
        )
        body = {
            "schema": _SCHEMA,
            "reference": "https://arxiv.org/abs/2609.38545",
            "official_code_revision": _PIN,
            "config": _config_payload(self.config),
            "fit_cutoff": fit_cutoff.isoformat(),
            "calibration_cutoff": calibration_cutoff.isoformat() if calibration_cutoff else None,
            "vocabulary": vocabulary,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "training_rows": row_payloads,
            "calibration_rows": cal_payloads,
            "training_data_sha256": _hash(row_payloads),
            "calibration_data_sha256": _hash(cal_payloads),
            "initial_parameter_sha256": initial_hashes,
            "states": {arm: [asdict(v) for v in values] for arm, values in states.items()},
            "loss_history": histories,
            "calibration": calibration_states,
            "synthetic": synthetic,
            "derived_annotations": derived,
            "research_only": True,
            "market_evidence": False,
            "pretrained": False,
            "official_reproduction": False,
            "data_rights": "UNVERIFIED",
            "independent_labels": "SUPPLIED_DECLARATION_UNVERIFIED",
            "device": "cpu",
            "torch_version": torch.__version__,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "comparison": {
                "windows_matched": True,
                "vocabulary_train_only": True,
                "optimizer_steps_per_arm": self.config.steps,
                "compute_matched": False,
                "capacity_matched": False,
                "supervision_richness_matched": False,
                "bow_limitation": "linear token-count head; binary-only supervision",
                "no_echo_limitation": "same trainable architecture and objective; echo modality withheld",
                "benefit_asserted": False,
            },
        }
        self._body, self._states, self._vocabulary, self._model_sha256 = (
            body,
            states,
            vocabulary,
            _hash(body),
        )
        return self

    @staticmethod
    def _check_disjoint(
        first: tuple[ReturnExample, ...],
        second: tuple[ReturnExample, ...],
        a: tuple[WindowSnapshot, ...],
        b: tuple[WindowSnapshot, ...],
    ) -> None:
        for field in ("window_id", "episode_id"):
            if {getattr(r.window, field) for r in first}.intersection(
                getattr(r.window, field) for r in second
            ):
                raise ValueError("training/calibration/holdout episode partitions overlap")
        if {d.document_id for s in a for d in s.say + s.echo}.intersection(
            d.document_id for s in b for d in s.say + s.echo
        ):
            raise ValueError("document groups cross training/calibration/holdout partitions")

    def _inference_snapshots(
        self, windows: tuple[NarrativeWindow, ...], asof: datetime
    ) -> tuple[WindowSnapshot, ...]:
        if not self._states or self.model_sha256 is None:
            raise RuntimeError("narrative learner is not fitted")
        asof = _clock(asof, "prediction asof")
        if (
            not isinstance(windows, tuple)
            or not 1 <= len(windows) <= self.config.max_rows
            or any(not isinstance(w, NarrativeWindow) for w in windows)
        ):
            raise ValueError("windows must be bounded immutable typed tuples")
        visible = tuple(w for w in windows if w.decision_time <= asof)
        if not visible:
            raise ValueError("no visible prediction window")
        cutoff = datetime.fromisoformat(
            self._body["calibration_cutoff"] or self._body["fit_cutoff"]
        )
        if any(w.decision_time < cutoff + self.config.embargo for w in visible):
            raise ValueError("prediction precedes frozen model cutoff/embargo")
        if len({w.window_id for w in visible}) != len(visible) or len(
            {w.episode_id for w in visible}
        ) != len(visible):
            raise ValueError("prediction episodes/window IDs overlap")
        snapshots = tuple(snapshot_window(w, self.config) for w in visible)
        past = self._body["training_rows"] + self._body["calibration_rows"]
        if {w.episode_id for w in visible}.intersection(r["episode_id"] for r in past) or {
            w.window_id for w in visible
        }.intersection(r["window_id"] for r in past):
            raise ValueError("holdout episode overlaps fitted partitions")
        if {d.document_id for s in snapshots for d in s.say + s.echo}.intersection(
            d["document_id"] for r in past for d in r["documents"]
        ):
            raise ValueError("holdout document groups overlap fitted partitions")
        return snapshots

    def predict(
        self, windows: tuple[NarrativeWindow, ...], *, asof: datetime, arm: Arm = "narrative"
    ) -> DirectionForecast:
        if arm not in _ARMS:
            raise ValueError("unknown narrative baseline arm")
        snapshots = self._inference_snapshots(windows, asof)
        torch = _torch()
        with torch.inference_mode():
            z, logits = _forward(
                _thaw(self._states[arm], torch),
                torch.tensor(
                    _features(snapshots, self.vocabulary, self.config),
                    dtype=torch.float64,
                    device="cpu",
                ),
                len(self.vocabulary),
                arm,
                torch,
            )
        calibration = self._body["calibration"][arm]
        values = logits.numpy()
        probabilities = expit(calibration["slope"] * values + calibration["intercept"])
        assert self.model_sha256 is not None
        return DirectionForecast(
            tuple(s.window_id for s in snapshots),
            tuple(float(v) for v in probabilities),
            tuple(float(v) for v in values),
            tuple(tuple(float(v) for v in row) for row in z.numpy()),
            arm,
            calibration["status"],
            self.model_sha256,
            _hash([s.snapshot_sha256 for s in snapshots]),
            self._body["synthetic"] or any(s.synthetic for s in snapshots),
            self._body["derived_annotations"] or any(s.derived for s in snapshots),
        )

    def evaluate(self, holdout: tuple[ReturnExample, ...], *, asof: datetime) -> dict[str, Any]:
        snapshots = _validate_examples(holdout, self.config, _clock(asof, "evaluation asof"), 1)
        labels = np.array([float(r.target_return > 0) for r in holdout])
        outcomes = []
        for name in _ARMS:
            forecast = self.predict(
                tuple(r.window for r in holdout), asof=asof, arm=cast(Arm, name)
            )
            if len(forecast.window_ids) != len(holdout):
                raise ValueError("evaluation cannot omit unavailable targets/windows")
            outcomes.append(
                {
                    "arm": name,
                    **_proper(np.array(forecast.probability_up), labels),
                    "calibration_status": forecast.calibration_status,
                    "forecast_sha256": forecast.forecast_sha256,
                }
            )
        result = {
            "outcomes": outcomes,
            "holdout_data_sha256": _hash(
                [_row_payload(r, s) for r, s in zip(holdout, snapshots, strict=True)]
            ),
            "model_sha256": self.model_sha256,
            "rows": len(holdout),
            "synthetic": self._body["synthetic"]
            or any(r.synthetic or s.synthetic for r, s in zip(holdout, snapshots, strict=True)),
            "derived_annotations": self._body["derived_annotations"]
            or any(r.derived or s.derived for r, s in zip(holdout, snapshots, strict=True)),
            "market_evidence": False,
            "research_only": True,
            "ece_is_proper_score": False,
            "comparison": self._body["comparison"],
        }
        return cast(dict[str, Any], json.loads(json.dumps(result)))

    def metadata(self) -> dict[str, Any]:
        return {
            **json.loads(json.dumps(self._body, default=str)),
            "model_sha256": self.model_sha256,
            "market_evidence": False,
            "pretrained": False,
            "official_reproduction": False,
        }

    def save(self, path: Path) -> None:
        if self.model_sha256 is None:
            raise RuntimeError("model is not fitted")
        encoded = json.dumps(
            {"body": self._body, "model_sha256": self.model_sha256},
            sort_keys=True,
            default=str,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > _MODEL_BYTES:
            raise ValueError("model artifact exceeds resource bound")
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded.decode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> NarrativeLearner:
        if path.stat().st_size > _MODEL_BYTES:
            raise ValueError("model artifact exceeds resource bound")
        try:
            with path.open("rb") as handle:
                encoded = handle.read(_MODEL_BYTES + 1)
            if len(encoded) > _MODEL_BYTES:
                raise ValueError("model artifact exceeds resource bound")
            payload = json.loads(encoded)
            body = payload["body"]
            if (
                body["schema"] != _SCHEMA
                or _hash(body) != payload["model_sha256"]
                or body["implementation_sha256"]
                != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            ):
                raise ValueError("model artifact schema/hash/implementation mismatch")
            if (
                body["market_evidence"] is not False
                or body["pretrained"] is not False
                or body["official_reproduction"] is not False
                or body["research_only"] is not True
                or body["device"] != "cpu"
                or body["data_rights"] != "UNVERIFIED"
                or body["independent_labels"] != "SUPPLIED_DECLARATION_UNVERIFIED"
                or not isinstance(body["synthetic"], bool)
                or not isinstance(body["derived_annotations"], bool)
            ):
                raise ValueError("persisted model honesty mismatch")
            config_payload = dict(body["config"])
            for field in ("horizon", "lookback", "publication_delay", "embargo"):
                config_payload[field] = timedelta(seconds=config_payload[field])
            model = cls(NarrativeConfig(**config_payload))
            if body["config"] != _config_payload(model.config):
                raise ValueError("persisted configuration fields mismatch")
            vocabulary = tuple(body["vocabulary"])
            if (
                not 1 <= len(vocabulary) <= model.config.max_vocab
                or vocabulary[0] != "<UNK>"
                or len(set(vocabulary)) != len(vocabulary)
            ):
                raise ValueError("vocabulary schema mismatch")
            for token in vocabulary[1:]:
                if (
                    not isinstance(token, str)
                    or not 1 <= len(token) <= 8192
                    or token != token.lower()
                    or _TOKEN.fullmatch(token) is None
                ):
                    raise ValueError("persisted vocabulary token mismatch")
            states = {
                arm: tuple(
                    TensorState(v["name"], tuple(v["shape"]), tuple(v["values"])) for v in values
                )
                for arm, values in body["states"].items()
            }
            if set(states) != set(_ARMS) or any(
                len(values)
                != len(_shapes(len(vocabulary), model.config.embedding_dim, cast(Arm, arm)))
                or {v.name: v.shape for v in values}
                != _shapes(len(vocabulary), model.config.embedding_dim, cast(Arm, arm))
                for arm, values in states.items()
            ):
                raise ValueError("persisted parameter architecture mismatch")
            fit_cutoff = _persisted_clock(body["fit_cutoff"])
            training = _persisted_rows(body["training_rows"], model.config, fit_cutoff, 6)
            calibration: tuple[ReturnExample, ...] = ()
            if body["calibration_cutoff"] is not None:
                calibration_cutoff = _persisted_clock(body["calibration_cutoff"])
                calibration = _persisted_rows(
                    body["calibration_rows"], model.config, calibration_cutoff, 1
                )
                if (
                    calibration_cutoff <= fit_cutoff
                    or min(r.window.decision_time for r in calibration)
                    < fit_cutoff + model.config.embargo
                ):
                    raise ValueError("persisted calibration precedes training cutoff/embargo")
                for field in ("window_id", "episode_id"):
                    if {getattr(r.window, field) for r in training}.intersection(
                        getattr(r.window, field) for r in calibration
                    ):
                        raise ValueError("persisted training/calibration partitions overlap")
                if {d.document_id for r in training for d in r.window.texts}.intersection(
                    d.document_id for r in calibration for d in r.window.texts
                ):
                    raise ValueError("persisted training/calibration document groups overlap")
            elif body["calibration_rows"] != []:
                raise ValueError("persisted calibration rows require explicit cutoff")
            _number(body["target_mean"], "target mean")
            _number(body["target_scale"], "target scale")
            if body["target_scale"] < 1e-8:
                raise ValueError("invalid persisted target scale")
            targets = np.array([r.target_return for r in training])
            if (
                not (np.any(targets > 0) and np.any(targets <= 0))
                or not math.isclose(body["target_mean"], float(targets.mean()), abs_tol=1e-12)
                or not math.isclose(body["target_scale"], float(targets.std()), abs_tol=1e-12)
            ):
                raise ValueError("persisted training target scales/directions mismatch")
            if body["synthetic"] != any(
                r.synthetic or any(v.synthetic for v in r.window.texts + r.window.links)
                for r in training + calibration
            ) or body["derived_annotations"] != any(
                r.derived or any(v.derived for v in r.window.texts + r.window.links)
                for r in training + calibration
            ):
                raise ValueError("persisted synthetic/derived provenance mismatch")
            for arm in _ARMS:
                _number(body["calibration"][arm]["slope"], "calibration slope", 1e6)
                _number(body["calibration"][arm]["intercept"], "calibration intercept", 1e6)
                status = body["calibration"][arm]["status"]
                if not calibration:
                    statuses = {"UNAVAILABLE_NO_CALIBRATION"}
                elif len(calibration) < 4 or len({r.target_return > 0 for r in calibration}) < 2:
                    statuses = {"UNAVAILABLE_TOO_FEW_OR_SINGLE_CLASS"}
                else:
                    statuses = {"PLATT_CALIBRATED_LATER_SPLIT", "UNAVAILABLE_OPTIMIZATION_FAILED"}
                if status not in statuses or (
                    status.startswith("UNAVAILABLE")
                    and (
                        body["calibration"][arm]["slope"] != 1.0
                        or body["calibration"][arm]["intercept"] != 0.0
                    )
                ):
                    raise ValueError("persisted calibration status contradicts partition")
            if (
                _hash(body["training_rows"]) != body["training_data_sha256"]
                or _hash(body["calibration_rows"]) != body["calibration_data_sha256"]
            ):
                raise ValueError("persisted row provenance mismatch")
            model._body, model._states, model._vocabulary, model._model_sha256 = (
                body,
                states,
                vocabulary,
                payload["model_sha256"],
            )
            return model
        except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
            raise ValueError("invalid model artifact schema") from exc


@dataclass(frozen=True, slots=True)
class VoiceObservation:
    document_id: str
    revision_id: str
    entity_id: str
    institution_id: str
    voice: Voice
    tone_proxy: float
    measure_id: str
    published_at: datetime
    ingested_at: datetime
    source_id: str
    source_sha256: str
    synthetic: bool
    derived: bool = False

    def __post_init__(self) -> None:
        for name in ("document_id", "revision_id", "entity_id", "institution_id", "measure_id"):
            _name(getattr(self, name), name)
        _number(self.tone_proxy, "tone proxy", 1)
        if self.voice not in ("say", "echo"):
            raise ValueError("observed voice must be say or echo")
        _public_clocks(self)
        _provenance(self)


@dataclass(frozen=True, slots=True)
class PositionProxy:
    proxy_id: str
    statement_document_id: str
    entity_id: str
    institution_id: str
    period_start: datetime
    period_end: datetime
    published_at: datetime
    ingested_at: datetime
    position_change: float
    units: str
    kind: Literal["disclosed_holdings_change", "supplied_position_change_proxy"]
    source_id: str
    source_sha256: str
    synthetic: bool
    derived: bool = False
    constructed_from_text: bool = False

    def __post_init__(self) -> None:
        for name in ("proxy_id", "statement_document_id", "entity_id", "institution_id", "units"):
            _name(getattr(self, name), name)
        start, end = (
            _clock(self.period_start, "position period_start"),
            _clock(self.period_end, "position period_end"),
        )
        if start >= end:
            raise ValueError("position period must have positive duration")
        object.__setattr__(self, "period_start", start)
        object.__setattr__(self, "period_end", end)
        _public_clocks(self, "period_end")
        _number(self.position_change, "position proxy")
        _provenance(self)
        if self.kind not in (
            "disclosed_holdings_change",
            "supplied_position_change_proxy",
        ) or not isinstance(self.constructed_from_text, bool):
            raise ValueError("Do must be an identified position/disclosure proxy")


@dataclass(frozen=True, slots=True)
class VoiceDoDiagnostic:
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    say_do_covariance: float | None
    echo_do_covariance: float | None
    say_pairs: int
    echo_pairs: int
    reasons: tuple[str, ...]
    asof: datetime
    input_sha256: str
    synthetic: bool
    derived: bool
    diagnostic_sha256: str = dataclass_field(init=False)
    intent: str = dataclass_field(init=False, default="UNKNOWN")
    actual_trades: bool = dataclass_field(init=False, default=False)
    market_evidence: bool = dataclass_field(init=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostic_sha256",
            _hash(
                {
                    name: getattr(self, name)
                    for name in (
                        "status",
                        "say_do_covariance",
                        "echo_do_covariance",
                        "say_pairs",
                        "echo_pairs",
                        "reasons",
                        "asof",
                        "input_sha256",
                        "synthetic",
                        "derived",
                    )
                }
            ),
        )


def voice_do_covariance(
    windows: tuple[NarrativeWindow, ...],
    observations: tuple[VoiceObservation, ...],
    proxies: tuple[PositionProxy, ...],
    *,
    asof: datetime,
    config: NarrativeConfig | None = None,
    min_pairs: int = 3,
) -> VoiceDoDiagnostic:
    """Trailing covariance of observed tone and linked position proxies at asof.

    Tone/link/text must be known at the original window decision; position proxies
    can be disclosed later, but must be known at this diagnostic's explicit asof.
    The result cannot be moved back to an earlier decision clock.
    """
    config = NarrativeConfig() if config is None else config
    asof = _clock(asof, "diagnostic asof")
    _count(min_pairs, "minimum pairs", 2, 256)
    for values, kind in (
        (windows, NarrativeWindow),
        (observations, VoiceObservation),
        (proxies, PositionProxy),
    ):
        if (
            not isinstance(values, tuple)
            or len(values) > 512
            or any(not isinstance(v, kind) for v in values)
        ):
            raise ValueError("diagnostic inputs must be bounded immutable typed tuples")
    visible = tuple(w for w in windows if asof - config.lookback <= w.decision_time <= asof)
    if len({w.window_id for w in visible}) != len(visible) or len(
        {w.episode_id for w in visible}
    ) != len(visible):
        raise ValueError("diagnostic episodes must be unique")
    groups = {(w.entity_id, w.institution_id) for w in visible}
    if len(groups) > 1:
        raise ValueError("covariance must pair one entity/institution group")
    observation_map: dict[tuple[str, str], VoiceObservation] = {}
    for observation in observations:
        key = (observation.document_id, observation.revision_id)
        if max(observation.published_at, observation.ingested_at) > asof:
            continue
        if key in observation_map:
            raise ValueError("ambiguous observed tone annotations")
        observation_map[key] = observation
    available_proxies: dict[str, PositionProxy] = {}
    reasons = []
    for proxy in proxies:
        if max(proxy.published_at, proxy.ingested_at) > asof:
            continue
        if proxy.constructed_from_text:
            reasons.append("text_constructed_position_proxy_excluded")
            continue
        if proxy.statement_document_id in available_proxies:
            raise ValueError("ambiguous independently supplied position proxies for statement")
        available_proxies[proxy.statement_document_id] = proxy
    say_pairs: list[tuple[float, float]] = []
    echo_pairs: list[tuple[float, float]] = []
    evidence: list[Any] = []
    seen_statements: set[str] = set()
    seen_proxies: set[str] = set()
    synthetic_used = derived_used = False
    units, measures = set(), set()
    for window in visible:
        snapshot = snapshot_window(window, config)
        synthetic_used = synthetic_used or snapshot.synthetic
        derived_used = derived_used or snapshot.derived
        for statement in snapshot.say:
            paired_proxy = available_proxies.get(statement.document_id)
            if paired_proxy is None:
                reasons.append("missing_independently_timed_linked_Do_proxy")
                continue
            if (paired_proxy.entity_id, paired_proxy.institution_id) != (
                window.entity_id,
                window.institution_id,
            ):
                raise ValueError("Do proxy entity/institution alignment mismatch")
            if statement.document_id in seen_statements or paired_proxy.proxy_id in seen_proxies:
                raise ValueError("covariance pairs reuse statement/proxy observations")
            seen_statements.add(statement.document_id)
            seen_proxies.add(paired_proxy.proxy_id)

            def observed_value(
                doc: TextRevision, current: NarrativeWindow = window
            ) -> VoiceObservation | None:
                value = observation_map.get((doc.document_id, doc.revision_id))
                if (
                    value is None
                    or max(value.published_at + config.publication_delay, value.ingested_at)
                    > current.decision_time
                ):
                    return None
                if (
                    value.voice != doc.voice
                    or (value.entity_id, value.institution_id)
                    != (current.entity_id, current.institution_id)
                    or value.published_at < doc.published_at
                ):
                    raise ValueError("tone proxy identity/revision/publication mismatch")
                return value

            observed = observed_value(statement)
            if observed is not None:
                say_pairs.append((observed.tone_proxy, paired_proxy.position_change))
                evidence.append((snapshot.snapshot_sha256, asdict(observed), asdict(paired_proxy)))
                measures.add(observed.measure_id)
                units.add(paired_proxy.units)
                synthetic_used = synthetic_used or observed.synthetic or paired_proxy.synthetic
                derived_used = derived_used or observed.derived or paired_proxy.derived
            echo_observed = [
                value
                for doc in snapshot.echo
                if any(
                    link.echo_document_id == doc.document_id
                    and link.statement_document_id == statement.document_id
                    for link in snapshot.links
                )
                if (value := observed_value(doc)) is not None
            ]
            if echo_observed:
                echo_pairs.append(
                    (
                        float(np.mean([v.tone_proxy for v in echo_observed])),
                        paired_proxy.position_change,
                    )
                )
                evidence.append(
                    (
                        snapshot.snapshot_sha256,
                        [asdict(v) for v in echo_observed],
                        asdict(paired_proxy),
                    )
                )
                measures.update(v.measure_id for v in echo_observed)
                units.add(paired_proxy.units)
                synthetic_used = (
                    synthetic_used
                    or paired_proxy.synthetic
                    or any(v.synthetic for v in echo_observed)
                )
                derived_used = (
                    derived_used or paired_proxy.derived or any(v.derived for v in echo_observed)
                )
    if len(units) > 1 or len(measures) > 1:
        raise ValueError("position units and tone measures must be comparable")

    def covariance(pairs: list[tuple[float, float]]) -> float | None:
        return float(np.cov(np.array(pairs).T, ddof=1)[0, 1]) if len(pairs) >= min_pairs else None

    say_cov, echo_cov = covariance(say_pairs), covariance(echo_pairs)
    if say_cov is None and echo_cov is None:
        reasons.append("insufficient_independent_observed_pairs")
    return VoiceDoDiagnostic(
        "AVAILABLE" if say_cov is not None or echo_cov is not None else "UNAVAILABLE",
        say_cov,
        echo_cov,
        len(say_pairs),
        len(echo_pairs),
        tuple(sorted(set(reasons))),
        asof,
        _hash(
            {
                "evidence": evidence,
                "asof": asof,
                "config": _config_payload(config),
                "min_pairs": min_pairs,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        ),
        synthetic_used,
        derived_used,
    )
