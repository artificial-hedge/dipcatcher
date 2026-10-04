"""Generated plugin wrapper for 'yahoo_finance'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="yahoo_finance",
    references=owner_references("plugin", "yahoo_finance"),
    module=__name__,
)
