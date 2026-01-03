"""Base interface for Shooting phase decision nodes."""


class ShootingNode:
    def __init__(self, name, step_id):
        self.name = name
        self.step_id = step_id

    def select_action(self, obs, legal_actions, rng):
        """Select an action index or structured decision.

        Subclasses should override this.
        """
        raise NotImplementedError
