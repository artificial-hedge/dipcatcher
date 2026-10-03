"""Generated plugin wrapper for 'sp_data'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="sp_data",
    references=owner_references("plugin", "sp_data"),
    module=__name__,
)
