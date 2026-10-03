"""Generated skill wrapper for 'vendor-book-map'; binds to the existing registry."""

from fx1.capabilities import owner_references
from fx1.extensions.contracts import SkillExtension

MODULE = SkillExtension(
    kind="skill",
    owner="vendor-book-map",
    references=owner_references("skill", "vendor-book-map"),
    module=__name__,
)
