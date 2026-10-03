"""Generated skill wrapper for 'build-labels'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="build-labels",
    references=owner_references("skill", "build-labels"),
    module=__name__,
)
