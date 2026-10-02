"""Generated plugin wrapper for 'imf'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="imf",
    references=owner_references("plugin", "imf"),
    module=__name__,
)
