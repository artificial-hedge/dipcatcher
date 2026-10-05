"""Generated skill wrapper for 'candle-book'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="candle-book",
    references=owner_references("skill", "candle-book"),
    module=__name__,
)
