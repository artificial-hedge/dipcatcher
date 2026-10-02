"""Generated skill wrapper for 'report'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="report",
    references=owner_references("skill", "report"),
    module=__name__,
)
