"""Generated skill wrapper for 'paper'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="paper",
    references=owner_references("skill", "paper"),
    module=__name__,
)
