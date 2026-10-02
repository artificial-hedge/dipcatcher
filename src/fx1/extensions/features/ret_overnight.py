"""Generated feature wrapper for 'ret_overnight'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="ret_overnight",
    references=owner_references("feature", "ret_overnight"),
    module=__name__,
)
