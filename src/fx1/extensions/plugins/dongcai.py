"""Generated plugin wrapper for 'dongcai'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="dongcai",
    references=owner_references("plugin", "dongcai"),
    module=__name__,
)
