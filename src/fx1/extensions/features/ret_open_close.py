"""Generated feature wrapper for 'ret_open_close'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="ret_open_close",
    references=owner_references("feature", "ret_open_close"),
    module=__name__,
)
