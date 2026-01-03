"""SP1: select which friendly unit shoots next (activation order)."""

from .base import ShootingNode
from ..specs.shooting_steps import SP1_SELECT_SHOOTER_UNIT


class SP1SelectShooterNode(ShootingNode):
    def __init__(self):
        super().__init__("sp1_select_shooter", SP1_SELECT_SHOOTER_UNIT)

    def select_action(self, obs, legal_actions, rng):
        if not legal_actions:
            return None
        return legal_actions[0]
