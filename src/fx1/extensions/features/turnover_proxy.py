"""Generated feature wrapper for 'turnover_proxy'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="turnover_proxy",
    references=owner_references("feature", "turnover_proxy"),
    module=__name__,
)
