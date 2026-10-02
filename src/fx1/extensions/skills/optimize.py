"""Generated skill wrapper for 'optimize'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="optimize",
    references=owner_references("skill", "optimize"),
    module=__name__,
)
