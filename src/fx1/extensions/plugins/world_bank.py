"""Generated plugin wrapper for 'world_bank'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="world_bank",
    references=owner_references("plugin", "world_bank"),
    module=__name__,
)
