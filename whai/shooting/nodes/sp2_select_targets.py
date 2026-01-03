"""SP2: select target unit(s) for the shooter."""

from .base import ShootingNode
from ..specs.shooting_steps import SP2_SELECT_TARGETS


class SP2SelectTargetsNode(ShootingNode):
    def __init__(self):
        super().__init__("sp2_select_targets", SP2_SELECT_TARGETS)

    def select_action(self, obs, legal_actions, rng):
        if not legal_actions:
            return None
        return legal_actions[0]
