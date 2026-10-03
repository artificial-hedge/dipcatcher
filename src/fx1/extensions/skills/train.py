"""Generated skill wrapper for 'train'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="train",
    references=owner_references("skill", "train"),
    module=__name__,
)
