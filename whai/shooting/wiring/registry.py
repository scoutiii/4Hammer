"""Registry for shooting step implementations."""

from ..specs.shooting_steps import (
    SP1_SELECT_SHOOTER_UNIT,
    SP2_SELECT_TARGETS,
    SP3_SPLIT_FIRE,
    SP4_STRATAGEMS,
    SP5_RESOLVE_ATTACKS,
)
from ..nodes.sp1_select_shooter import SP1SelectShooterNode
from ..nodes.sp2_select_targets import SP2SelectTargetsNode
from ..nodes.sp3_split_fire import SP3SplitFireNode
from ..nodes.sp4_stratagems import SP4StratagemsNode
from ..nodes.sp5_resolve_attacks import SP5ResolveAttacksNode

REGISTRY = {
    SP1_SELECT_SHOOTER_UNIT: {"heuristic": SP1SelectShooterNode},
    SP2_SELECT_TARGETS: {"ppo": SP2SelectTargetsNode, "heuristic": SP2SelectTargetsNode},
    SP3_SPLIT_FIRE: {"heuristic": SP3SplitFireNode},
    SP4_STRATAGEMS: {"heuristic": SP4StratagemsNode},
    SP5_RESOLVE_ATTACKS: {"heuristic": SP5ResolveAttacksNode},
}


def build_node(step_id, backend):
    if step_id not in REGISTRY or backend not in REGISTRY[step_id]:
        raise KeyError(f"No node registered for {step_id} with backend {backend}")
    return REGISTRY[step_id][backend]()
