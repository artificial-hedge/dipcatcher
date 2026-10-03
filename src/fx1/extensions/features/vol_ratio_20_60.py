"""Generated feature wrapper for 'vol_ratio_20_60'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="vol_ratio_20_60",
    references=owner_references("feature", "vol_ratio_20_60"),
    module=__name__,
)
