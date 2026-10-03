"""Generated skill wrapper for 'build-features'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="build-features",
    references=owner_references("skill", "build-features"),
    module=__name__,
)
