"""Generated feature wrapper for 'planted_signal'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="planted_signal",
    references=owner_references("feature", "planted_signal"),
    module=__name__,
)
