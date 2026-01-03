"""SP3: split fire / allocate attacks across targets."""

from .base import ShootingNode
from ..specs.shooting_steps import SP3_SPLIT_FIRE


class SP3SplitFireNode(ShootingNode):
    def __init__(self):
        super().__init__("sp3_split_fire", SP3_SPLIT_FIRE)

    def select_action(self, obs, legal_actions, rng):
        if not legal_actions:
            return None
        return legal_actions[0]
