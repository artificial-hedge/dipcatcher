"""Generated feature wrapper for 'log_ret_1'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="log_ret_1",
    references=owner_references("feature", "log_ret_1"),
    module=__name__,
)
