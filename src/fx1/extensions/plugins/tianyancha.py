"""Generated plugin wrapper for 'tianyancha'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="tianyancha",
    references=owner_references("plugin", "tianyancha"),
    module=__name__,
)
