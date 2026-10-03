"""Generated plugin wrapper for 'xhcj'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="xhcj",
    references=owner_references("plugin", "xhcj"),
    module=__name__,
)
