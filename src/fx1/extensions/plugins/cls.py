"""Generated plugin wrapper for 'cls'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="cls",
    references=owner_references("plugin", "cls"),
    module=__name__,
)
