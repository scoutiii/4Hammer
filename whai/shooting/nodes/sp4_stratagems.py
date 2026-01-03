"""SP4: stratagem/ability timing during shooting."""

from .base import ShootingNode
from ..specs.shooting_steps import SP4_STRATAGEMS


class SP4StratagemsNode(ShootingNode):
    def __init__(self):
        super().__init__("sp4_stratagems", SP4_STRATAGEMS)

    def select_action(self, obs, legal_actions, rng):
        if not legal_actions:
            return None
        return legal_actions[0]
