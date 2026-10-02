"""Generated plugin wrapper for 'binance_crypto'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import PluginExtension

MODULE = PluginExtension(
    kind="plugin",
    owner="binance_crypto",
    references=owner_references("plugin", "binance_crypto"),
    module=__name__,
)
