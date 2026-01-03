"""Minimal controller scaffold for composing shooting steps."""

from ..specs import shooting_steps
from .registry import build_node

DEFAULT_CONFIG = {
    shooting_steps.SP1_SELECT_SHOOTER_UNIT: "heuristic",
    shooting_steps.SP2_SELECT_TARGETS: "ppo",
    shooting_steps.SP3_SPLIT_FIRE: "heuristic",
    shooting_steps.SP4_STRATAGEMS: "heuristic",
    shooting_steps.SP5_RESOLVE_ATTACKS: "heuristic",
}


class ShootingController:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.nodes = {
            step_id: build_node(step_id, backend)
            for step_id, backend in self.config.items()
        }

    def select_action(self, step_id, obs, legal_actions, rng):
        node = self.nodes.get(step_id)
        if node is None:
            return None
        return node.select_action(obs, legal_actions, rng)
