"""Generated plugin wrapper for 'igo_open_data'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="igo_open_data",
    references=owner_references("plugin", "igo_open_data"),
    module=__name__,
)
