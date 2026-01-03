"""Definitions for Shooting phase sub-steps (SP1..SP5).

SP steps are used to split the phase into deterministic rules and learned decisions.
"""

SP1_SELECT_SHOOTER_UNIT = "SP1_SELECT_SHOOTER_UNIT"
"""Select which friendly unit activates to shoot (activation order)."""

SP2_SELECT_TARGETS = "SP2_SELECT_TARGETS"
"""Select target unit(s) for the shooter. Learned in early curricula."""

SP3_SPLIT_FIRE = "SP3_SPLIT_FIRE"
"""Allocate weapons/attacks across multiple targets (split fire decisions)."""

SP4_STRATAGEMS = "SP4_STRATAGEMS"
"""Timing/selection of stratagems and abilities during shooting."""

SP5_RESOLVE_ATTACKS = "SP5_RESOLVE_ATTACKS"
"""Deterministic rules resolution for attacks (usually heuristic/no-op)."""
