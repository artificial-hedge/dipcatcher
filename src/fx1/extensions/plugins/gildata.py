"""Generated plugin wrapper for 'gildata'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="gildata",
    references=owner_references("plugin", "gildata"),
    module=__name__,
)
