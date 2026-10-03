"""Generated skill wrapper for 'kronos-forecast'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="kronos-forecast",
    references=owner_references("skill", "kronos-forecast"),
    module=__name__,
)
