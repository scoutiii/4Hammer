"""SP5: resolve attacks (usually deterministic/no-op)."""

from .base import ShootingNode
from ..specs.shooting_steps import SP5_RESOLVE_ATTACKS


class SP5ResolveAttacksNode(ShootingNode):
    def __init__(self):
        super().__init__("sp5_resolve_attacks", SP5_RESOLVE_ATTACKS)

    def select_action(self, obs, legal_actions, rng):
        return None
