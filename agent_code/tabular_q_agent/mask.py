"""The action mask: which actions the agent may choose from at all.

``core.safety.assess_actions`` grades every legal action by the strongest
opponent model it survives (tier 4 down to 0, see ``Assessment.tier``). The
mask turns those grades into the allowed set. The same mask must be used for
exploration, for greedy selection and for the maximum in the Q-learning target,
or the learned values describe a different policy from the one that is played.

The mask never picks an action; it only removes actions. It is never empty:
``legal_actions`` always contains ``WAIT``, and when no action in the requested
tiers exists, every variant falls back to ``safest``, which returns the
longest-surviving actions.
"""

from collections.abc import Sequence
from typing import Literal, get_args

from .core.safety import Assessment, safest

# Which safety tiers the agent may choose from: the best tier
# available, anything that survives static opponents (tier >= 2), anything with
# some known escape (tier >= 1), or every legal action.
MaskVariant = Literal["best_tier", "min_tier_2", "any_escape", "legal"]
MASK_VARIANTS: tuple[MaskVariant, ...] = get_args(MaskVariant)


def allowed_actions(
    assessments: Sequence[Assessment], variant: MaskVariant
) -> list[str]:
    """The actions ``variant`` allows, in the order of ``assessments``."""
    match variant:
        case "legal":
            return [a.action for a in assessments]
        case "best_tier":
            return [a.action for a in safest(assessments)]
        case "min_tier_2":
            floor = 2
        case "any_escape":
            floor = 1
    wide = [a.action for a in assessments if a.tier >= floor]
    return wide or [a.action for a in safest(assessments)]
