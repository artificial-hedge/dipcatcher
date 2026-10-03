"""Generated skill wrapper for 'collect'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="collect",
    references=owner_references("skill", "collect"),
    module=__name__,
)
