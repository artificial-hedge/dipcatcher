"""Generated plugin wrapper for 'ifind'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="ifind",
    references=owner_references("plugin", "ifind"),
    module=__name__,
)
