"""Generated plugin wrapper for 'caixin'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="caixin",
    references=owner_references("plugin", "caixin"),
    module=__name__,
)
