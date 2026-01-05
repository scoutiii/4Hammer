import argparse
import os
import random

from rlc import Program, compile


def compile_rules(rules_path, include_dir):
    include_dirs = [include_dir]
    try:
        return compile([rules_path], rlc_includes=include_dirs)
    except TypeError:
        pass
    try:
        return compile([rules_path], include_dirs=include_dirs)
    except TypeError:
        pass
    try:
        return compile([rules_path], include_paths=include_dirs)
    except TypeError:
        pass
    try:
        return compile([rules_path], include=include_dirs)
    except TypeError:
        pass
    os.environ.setdefault("RLC_INCLUDE_PATH", os.pathsep.join(include_dirs))
    return compile([rules_path])


def get_current_player(rules, state):
    module = rules.module if hasattr(rules, "module") else rules
    game_state = state.state if hasattr(state, "state") else state
    if hasattr(module, "get_current_player"):
        return int(module.get_current_player(game_state))
    if hasattr(state, "get_current_player"):
        return int(state.get_current_player())
    return 0


def format_action(rules, action):
    try:
        return str(action)
    except Exception:
        return str(action)


def list_legal_actions(state, rules):
    legal_indices = list(state.legal_actions_indicies)
    actions = state.actions
    print(f"Legal actions: {len(legal_indices)}")
    for idx, action_index in enumerate(legal_indices):
        action = actions[action_index]
        print(f"{idx:>4}: {format_action(rules, action)}")
    return legal_indices, actions


def pick_user_action(legal_indices, actions, rules, rng):
    while True:
        raw = input("Select action index (or 'r' for random): ").strip()
        if raw.lower() in {"r", "rand", "random"}:
            action_index = rng.choice(legal_indices)
            return actions[action_index]
        try:
            selection = int(raw)
        except ValueError:
            print("Enter a number from the list or 'r'.")
            continue
        if 0 <= selection < len(legal_indices):
            action_index = legal_indices[selection]
            return actions[action_index]
        print("Invalid selection.")


def pick_random_action(legal_indices, actions, rng):
    action_index = rng.choice(legal_indices)
    return actions[action_index]


def main():
    parser = argparse.ArgumentParser(
        description="Interactive single-round demo: player 0 selects actions, player 1 is random."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random player actions.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules_path = os.path.join(root_dir, "examples", "interactive_single_round.rl")
    include_dir = os.path.join(root_dir, "src")

    print(f"Loading rules from: {rules_path}")
    rules = compile_rules(rules_path, include_dir)
    program = rules if isinstance(rules, Program) else Program(rules)
    state = program.start()

    step = 0
    while not state.is_done():
        step += 1
        player = get_current_player(rules, state)
        print(f"\nStep {step} | current player: {player}")

        legal_indices, actions = list_legal_actions(state, rules)
        if not legal_indices:
            print("No legal actions; stopping.")
            break

        if player == 0:
            action = pick_user_action(legal_indices, actions, rules, rng)
            print(f"Player 0 chose: {format_action(rules, action)}")
        else:
            action = pick_random_action(legal_indices, actions, rng)
            print(f"Player 1/random chose: {format_action(rules, action)}")

        state.step(action)

    print("\nGame complete.")


if __name__ == "__main__":
    main()
