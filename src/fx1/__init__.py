"""fx-1 — harness, evaluation program, and training pipeline.

fx-1 (always lowercase) is the model *program*, not a released checkpoint.
The declared base is Kimi K3 (``fx1.BASE_MODEL``). No fine-tuned weights
are published, and ``fx1.__version__`` is the harness API, not a model
release. This package is the machinery around that program: corpus
construction (``fx1.data``), evaluation (``fx1.eval``), training
(``fx1.train``), serving (``fx1.serve``), the flagship bench
(``fx1.bench``), the reward model (``fx1.reward``), and the harness
bridge (``fx1.harness``). See ``docs/FX1.md`` and the status block in
``README.md``.

``fx1.forecast`` is a separate inference/evaluation harness for an external
price/return model that is not implemented in this package. See
``docs/fx1_harness.md``.
"""

__version__ = "0.4.0"

BASE_MODEL = "moonshotai/Kimi-K3"
