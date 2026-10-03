"""Generated feature wrapper for 'log_price'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="log_price",
    references=owner_references("feature", "log_price"),
    module=__name__,
)
