"""Generated feature wrapper for 'z_vs_ma20'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="z_vs_ma20",
    references=owner_references("feature", "z_vs_ma20"),
    module=__name__,
)
