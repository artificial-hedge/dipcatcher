"""Generated feature wrapper for 'ret_overnight_20'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="ret_overnight_20",
    references=owner_references("feature", "ret_overnight_20"),
    module=__name__,
)
