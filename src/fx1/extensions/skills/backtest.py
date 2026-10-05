"""Generated skill wrapper for 'backtest'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="backtest",
    references=owner_references("skill", "backtest"),
    module=__name__,
)
