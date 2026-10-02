"""Generated plugin wrapper for 'finance_fetch'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="finance_fetch",
    references=owner_references("plugin", "finance_fetch"),
    module=__name__,
)
