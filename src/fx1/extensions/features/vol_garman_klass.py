"""Generated feature wrapper for 'vol_garman_klass'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import FeatureExtension

MODULE = FeatureExtension(
    kind="feature",
    owner="vol_garman_klass",
    references=owner_references("feature", "vol_garman_klass"),
    module=__name__,
)
