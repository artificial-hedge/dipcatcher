"""Generated plugin wrapper for 'sec_edgar'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="sec_edgar",
    references=owner_references("plugin", "sec_edgar"),
    module=__name__,
)
