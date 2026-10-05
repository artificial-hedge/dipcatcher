"""Generated feature wrapper for 'volume_vol'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="volume_vol",
    references=owner_references("feature", "volume_vol"),
    module=__name__,
)
