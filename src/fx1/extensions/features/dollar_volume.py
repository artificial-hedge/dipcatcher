"""Generated feature wrapper for 'dollar_volume'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="dollar_volume",
    references=owner_references("feature", "dollar_volume"),
    module=__name__,
)
