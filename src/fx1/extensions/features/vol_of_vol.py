"""Generated feature wrapper for 'vol_of_vol'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="vol_of_vol",
    references=owner_references("feature", "vol_of_vol"),
    module=__name__,
)
