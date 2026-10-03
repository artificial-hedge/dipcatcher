"""Generated feature wrapper for 'high_52w_prox'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="high_52w_prox",
    references=owner_references("feature", "high_52w_prox"),
    module=__name__,
)
