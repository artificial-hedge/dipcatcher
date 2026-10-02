"""Generated feature wrapper for 'vol_parkinson'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="vol_parkinson",
    references=owner_references("feature", "vol_parkinson"),
    module=__name__,
)
