"""Generated plugin wrapper for 'finenter'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="finenter",
    references=owner_references("plugin", "finenter"),
    module=__name__,
)
