"""Generated skill wrapper for 'forecast'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="forecast",
    references=owner_references("skill", "forecast"),
    module=__name__,
)
