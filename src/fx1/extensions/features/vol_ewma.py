"""Generated feature wrapper for 'vol_ewma'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="vol_ewma",
    references=owner_references("feature", "vol_ewma"),
    module=__name__,
)
