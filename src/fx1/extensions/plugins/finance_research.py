"""Generated plugin wrapper for 'finance_research'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="finance_research",
    references=owner_references("plugin", "finance_research"),
    module=__name__,
)
