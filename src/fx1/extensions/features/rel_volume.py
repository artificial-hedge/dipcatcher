"""Generated feature wrapper for 'rel_volume'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="rel_volume",
    references=owner_references("feature", "rel_volume"),
    module=__name__,
)
