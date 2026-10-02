"""Generated skill wrapper for 'monitor'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="monitor",
    references=owner_references("skill", "monitor"),
    module=__name__,
)
