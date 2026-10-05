"""Generated plugin wrapper for 'wind'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="wind",
    references=owner_references("plugin", "wind"),
    module=__name__,
)
