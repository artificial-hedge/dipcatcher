"""Architecture for the repository byte language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RepoLMConfig:
    """Decoder-only byte transformer.

    ``vocab_size`` is 256 because the corpus is a raw byte stream. File
    boundaries are length-prefixed frames inside that stream, not extra
    vocabulary ids.
    """

    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    seq_len: int = 128
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1 or self.d_ff < 1:
            raise ValueError("model dimensions must be positive")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, int | bool]) -> RepoLMConfig:
        return cls(
            vocab_size=int(payload["vocab_size"]),
            d_model=int(payload["d_model"]),
            n_heads=int(payload["n_heads"]),
            n_layers=int(payload["n_layers"]),
            d_ff=int(payload["d_ff"]),
            seq_len=int(payload["seq_len"]),
            tie_embeddings=bool(payload["tie_embeddings"]),
        )
