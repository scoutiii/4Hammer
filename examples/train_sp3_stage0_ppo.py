import argparse
import os
import random
import re
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import rlc
from ml.env import SingleRLCEnvironment, exit_on_invalid_env

SCENARIO_PATH = Path(__file__).with_name("sp3_stage0_weapon_choice.rl")
SRC_DIR = Path(__file__).resolve().parent.parent / "src"

SHOOTER_UNIT_INDEX = 0
TARGET_UNIT_INDEX = 1

WEAPON_RE = re.compile(r"select_weapon\s*\{\s*weapon_id:\s*(\d+)\s*\}")
WEAPON_ID_FALLBACK_RE = re.compile(r"weapon_id\s*:\s*(\d+)")
SANITY_CHECKED = False


def find_rlc_compiler(explicit_path=None):
    if explicit_path:
        return explicit_path
    env_path = os.environ.get("RLC_COMPILER")
    if env_path:
        return env_path
    path = shutil.which("rlc")
    if path:
        return path
    candidate = Path(sys.executable).parent / "rlc"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("Could not find rlc. Set RLC_COMPILER or install rlc in the active env.")


def rlc_string_to_py(sobj):
    chars = []
    for i in range(sobj.size()):
        ch_ptr = sobj.get(i)
        val = ch_ptr.contents.value
        if val < 0:
            val += 256
        chars.append(chr(val))
    return "".join(chars)


def build_action_weapon_id_map(program, actions):
    mapping = {}
    for idx, action in enumerate(actions):
        s = rlc_string_to_py(program.module.to_string(action)).strip()
        match = WEAPON_RE.search(s)
        if not match:
            match = WEAPON_ID_FALLBACK_RE.search(s)
        mapping[idx] = int(match.group(1)) if match else None
    return mapping


def stat_mean(stat):
    if stat.resume_index == 0:
        return float(stat._content._alternative0.value)
    sides = stat._content._alternative1.value
    return (float(sides) + 1.0) / 2.0


def profile_features(profile):
    inv = profile.invuln_save()
    inv = 0 if inv >= 10 else inv
    return [
        float(profile.thoughness()),
        float(profile.save()),
        float(inv),
        float(profile.wounds()),
        float(profile.leadership()),
        float(profile.control()),
        float(profile.movement()),
    ]


def encode_state(state):
    board = state.board
    shooter = board.units.get(SHOOTER_UNIT_INDEX).contents
    target = board.units.get(TARGET_UNIT_INDEX).contents

    distance = float(shooter.distance(target))

    shooter_models = float(shooter.models.size())
    shooter_can_shoot = float(shooter.can_shoot)
    shooter_has_shoot = float(shooter.has_shoot)

    if shooter.models.size() > 0:
        shooter_profile = shooter.models.get(0).contents.profile
        shooter_profile_feats = profile_features(shooter_profile)
    else:
        shooter_profile_feats = [0.0] * 7

    target_models = float(target.models.size())
    total_wounds = 0.0
    for i in range(target.models.size()):
        total_wounds += float(target.models.get(i).contents.wounds_left())

    if target.models.size() > 0:
        target_profile = target.models.get(0).contents.profile
        target_profile_feats = profile_features(target_profile)
    else:
        target_profile_feats = [0.0] * 7

    return np.array(
        [
            distance,
            shooter_models,
            shooter_can_shoot,
            shooter_has_shoot,
            *shooter_profile_feats,
            target_models,
            total_wounds,
            *target_profile_feats,
        ],
        dtype=np.float32,
    )


def rule_kind(module, name, fallback=None):
    kind = module.WeaponRuleKind
    if hasattr(kind, name):
        return getattr(kind, name)()
    if fallback and hasattr(kind, fallback):
        return getattr(kind, fallback)()
    raise AttributeError(f"WeaponRuleKind missing {name} (fallback {fallback})")


def encode_action(weapon, module):
    flag_torrent = float(weapon.has_rule(rule_kind(module, "torrent")))
    flag_sustained = float(weapon.has_rule(rule_kind(module, "sustained_hit")))
    flag_lethal = float(weapon.has_rule(rule_kind(module, "lethal_hits", "letal_hits")))
    flag_dev = float(weapon.has_rule(rule_kind(module, "devastating_wounds")))
    flag_hazard = float(weapon.has_rule(rule_kind(module, "hazardous")))
    flag_blast = float(weapon.has_rule(rule_kind(module, "blast")))
    flag_melta = float(weapon.has_rule(rule_kind(module, "melta")))
    flag_rapid = float(weapon.has_rule(rule_kind(module, "rapid_fire")))
    flag_ignore_cover = float(weapon.has_rule(rule_kind(module, "ignore_cover")))

    param = 0.0
    for rk in [
        rule_kind(module, "rapid_fire"),
        rule_kind(module, "melta"),
        rule_kind(module, "sustained_hit"),
        rule_kind(module, "blast"),
    ]:
        val = weapon.get_rule_parameter(rk)
        if val != 0:
            param = float(val)
            break

    return np.array(
        [
            float(weapon.range()),
            stat_mean(weapon.attacks()),
            float(weapon.skill()),
            float(weapon.strenght()),
            float(weapon.penetration()),
            stat_mean(weapon.damage()),
            flag_torrent,
            flag_sustained,
            flag_lethal,
            flag_dev,
            flag_hazard,
            flag_blast,
            flag_melta,
            flag_rapid,
            flag_ignore_cover,
            param,
        ],
        dtype=np.float32,
    )


def select_weapon_actions(env, state, action_weapon_id, module):
    mask = env.get_action_mask()
    legal_indices = np.where(mask == 1)[0]
    board = state.board
    shooter = board.units.get(SHOOTER_UNIT_INDEX).contents
    model = shooter.models.get(0).contents

    action_indices = []
    action_features = []
    for idx in legal_indices:
        weapon_id = action_weapon_id.get(idx)
        if weapon_id is None:
            continue
        weapon = model.weapons.get(weapon_id).contents
        action_indices.append(idx)
        action_features.append(encode_action(weapon, module))
    return action_indices, action_features


def choose_auto_action(env, rng):
    mask = env.get_action_mask()
    legal_indices = np.where(mask == 1)[0]
    if len(legal_indices) == 1:
        return int(legal_indices[0])
    return int(rng.choice(legal_indices))


def is_weapon_state(state, module):
    return state.board.current_state.value == module.CurrentStateDescription.select_weapon().value


def target_total_wounds(board):
    target = board.units.get(TARGET_UNIT_INDEX).contents
    total = 0.0
    for i in range(target.models.size()):
        total += float(target.models.get(i).contents.wounds_left())
    return total


def ensure_weapon_actions(action_indices):
    global SANITY_CHECKED
    if SANITY_CHECKED:
        return
    if len(action_indices) < 2:
        raise RuntimeError(
            f"Expected >=2 weapon actions at first weapon selection, found {len(action_indices)}"
        )
    SANITY_CHECKED = True


@dataclass
class Transition:
    state: np.ndarray
    action_features: np.ndarray
    action_index: int
    old_logprob: float
    value: float
    next_value: float
    reward: float
    done: bool


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class WeaponPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.state_mlp = MLP(state_dim, hidden_dim, hidden_dim)
        self.action_mlp = MLP(action_dim, hidden_dim, hidden_dim)
        self.logit_mlp = MLP(hidden_dim * 2, hidden_dim, 1)
        self.value_mlp = MLP(hidden_dim, hidden_dim, 1)

    def action_logits(self, state_tensor, action_tensor):
        hs = self.state_mlp(state_tensor)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        ha = self.action_mlp(action_tensor)
        hs_rep = hs.unsqueeze(0).expand(ha.size(0), -1)
        logits = self.logit_mlp(torch.cat([hs_rep, ha], dim=-1)).squeeze(-1)
        value = self.value_mlp(hs).squeeze(-1)
        return logits, value

    def value(self, state_tensor):
        hs = self.state_mlp(state_tensor)
        return self.value_mlp(hs).squeeze(-1)


def advance_until_weapon_or_done(env, module, action_weapon_id, rng, last_target_wounds):
    reward = 0.0
    while not env.is_done_underling():
        state = env.state.state
        if is_weapon_state(state, module):
            action_indices, _ = select_weapon_actions(env, state, action_weapon_id, module)
            if action_indices:
                break
        auto_action = choose_auto_action(env, rng)
        env.step(auto_action)
        new_target_wounds = target_total_wounds(env.state.state.board)
        reward += last_target_wounds - new_target_wounds
        last_target_wounds = new_target_wounds
    return reward, last_target_wounds


def run_episode(env, policy, action_weapon_id, module, rng, train=True):
    env.reset()
    last_target_wounds = target_total_wounds(env.state.state.board)
    done = env.is_done_underling()

    transitions = []
    episode_reward = 0.0

    while not done:
        state = env.state.state
        if is_weapon_state(state, module):
            action_indices, action_feats = select_weapon_actions(
                env, state, action_weapon_id, module
            )
            if action_indices:
                ensure_weapon_actions(action_indices)

                state_vec = encode_state(state)
                feats = np.stack(action_feats, axis=0)

                with torch.no_grad():
                    st = torch.tensor(state_vec, dtype=torch.float32)
                    at = torch.tensor(feats, dtype=torch.float32)
                    logits, value = policy.action_logits(st, at)
                    dist = torch.distributions.Categorical(logits=logits)
                    if train:
                        chosen = dist.sample()
                    else:
                        chosen = torch.argmax(logits)
                    logprob = dist.log_prob(chosen)

                action_idx = action_indices[int(chosen.item())]
                step_reward = 0.0

                env.step(action_idx)
                new_target_wounds = target_total_wounds(env.state.state.board)
                step_reward += last_target_wounds - new_target_wounds
                last_target_wounds = new_target_wounds

                auto_reward, last_target_wounds = advance_until_weapon_or_done(
                    env, module, action_weapon_id, rng, last_target_wounds
                )
                step_reward += auto_reward

                done = env.is_done_underling()
                episode_reward += step_reward

                if train:
                    if done:
                        next_value = 0.0
                    else:
                        next_state_vec = encode_state(env.state.state)
                        with torch.no_grad():
                            next_value = float(
                                policy.value(torch.tensor(next_state_vec, dtype=torch.float32)).item()
                            )
                    transitions.append(
                        Transition(
                            state=state_vec,
                            action_features=feats,
                            action_index=int(chosen.item()),
                            old_logprob=float(logprob.item()),
                            value=float(value.item()),
                            next_value=next_value,
                            reward=float(step_reward),
                            done=done,
                        )
                    )
                continue

        auto_action = choose_auto_action(env, rng)
        env.step(auto_action)
        new_target_wounds = target_total_wounds(env.state.state.board)
        episode_reward += last_target_wounds - new_target_wounds
        last_target_wounds = new_target_wounds
        done = env.is_done_underling()

    return transitions, episode_reward


def run_episode_eval(env, policy, action_weapon_id, module, rng, mode="greedy"):
    env.reset()
    last_target_wounds = target_total_wounds(env.state.state.board)
    done = env.is_done_underling()

    episode_reward = 0.0
    weapon_counts = {}

    while not done:
        state = env.state.state
        if is_weapon_state(state, module):
            action_indices, action_feats = select_weapon_actions(
                env, state, action_weapon_id, module
            )
            if action_indices:
                ensure_weapon_actions(action_indices)

                if mode == "random":
                    chosen_idx = int(rng.integers(len(action_indices)))
                else:
                    state_vec = encode_state(state)
                    feats = np.stack(action_feats, axis=0)
                    with torch.no_grad():
                        st = torch.tensor(state_vec, dtype=torch.float32)
                        at = torch.tensor(feats, dtype=torch.float32)
                        logits, _ = policy.action_logits(st, at)
                        chosen_idx = int(torch.argmax(logits).item())

                action_idx = action_indices[chosen_idx]
                weapon_id = action_weapon_id.get(action_idx)
                if weapon_id is not None:
                    weapon_counts[weapon_id] = weapon_counts.get(weapon_id, 0) + 1

                env.step(action_idx)
                new_target_wounds = target_total_wounds(env.state.state.board)
                episode_reward += last_target_wounds - new_target_wounds
                last_target_wounds = new_target_wounds

                auto_reward, last_target_wounds = advance_until_weapon_or_done(
                    env, module, action_weapon_id, rng, last_target_wounds
                )
                episode_reward += auto_reward
                done = env.is_done_underling()
                continue

        auto_action = choose_auto_action(env, rng)
        env.step(auto_action)
        new_target_wounds = target_total_wounds(env.state.state.board)
        episode_reward += last_target_wounds - new_target_wounds
        last_target_wounds = new_target_wounds
        done = env.is_done_underling()

    return episode_reward, weapon_counts


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, transitions, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4, minibatch_size=256):
    rewards = np.array([t.reward for t in transitions], dtype=np.float32)
    values = np.array([t.value for t in transitions], dtype=np.float32)
    next_values = np.array([t.next_value for t in transitions], dtype=np.float32)
    dones = np.array([t.done for t in transitions], dtype=np.bool_)

    advantages, returns = compute_gae(rewards, values, next_values, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    indices = np.arange(len(transitions))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch_idx = indices[start : start + minibatch_size]

            policy_losses = []
            value_losses = []
            entropies = []

            for idx in batch_idx:
                t = transitions[idx]
                st = torch.tensor(t.state, dtype=torch.float32)
                at = torch.tensor(t.action_features, dtype=torch.float32)
                logits, value = policy.action_logits(st, at)
                dist = torch.distributions.Categorical(logits=logits)

                action_tensor = torch.tensor(t.action_index, dtype=torch.long)
                logprob = dist.log_prob(action_tensor)
                entropy = dist.entropy()

                old = torch.tensor(t.old_logprob, dtype=torch.float32)
                ratio = torch.exp(logprob - old)
                adv = torch.tensor(advantages[idx], dtype=torch.float32)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                policy_losses.append(-torch.min(surr1, surr2))

                ret = torch.tensor(returns[idx], dtype=torch.float32)
                value_losses.append((ret - value) ** 2)
                entropies.append(entropy)

            loss = (
                torch.stack(policy_losses).mean()
                + value_coef * torch.stack(value_losses).mean()
                - entropy_coef * torch.stack(entropies).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(policy, env, action_weapon_id, module, episodes=20, seed=0, mode="greedy"):
    rng = np.random.default_rng(seed)
    rewards = []
    weapon_counts = {}
    if policy is not None:
        policy.eval()

    for _ in range(episodes):
        ep_reward, ep_counts = run_episode_eval(
            env, policy, action_weapon_id, module, rng, mode=mode
        )
        rewards.append(ep_reward)
        for weapon_id, count in ep_counts.items():
            weapon_counts[weapon_id] = weapon_counts.get(weapon_id, 0) + count

    if policy is not None:
        policy.train()

    return float(np.mean(rewards)), float(np.std(rewards)), weapon_counts


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)

    rlc_path = find_rlc_compiler()
    program = rlc.compile(
        [str(SCENARIO_PATH)],
        rlc_compiler=rlc_path,
        rlc_includes=[str(SRC_DIR)],
    )
    exit_on_invalid_env(program, forced_one_player=False, needs_score=True)

    env = SingleRLCEnvironment(program)
    action_weapon_id = build_action_weapon_id_map(program, env.actions())

    state_dim = len(encode_state(env.state.state))
    action_dim = len(
        encode_action(
            env.state.state.board.units.get(SHOOTER_UNIT_INDEX).contents.models.get(0).contents.weapons.get(0).contents,
            program.module,
        )
    )

    policy = WeaponPolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    ckpt_dir = Path(__file__).with_name("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "sp3_stage0_ppo.pt"

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)

    if args.eval:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        data = torch.load(ckpt_path, map_location="cpu")
        policy.load_state_dict(data["model"])
        mean_r, std_r, counts = evaluate(
            policy,
            env,
            action_weapon_id,
            program.module,
            args.eval_episodes,
            args.seed,
            mode="greedy",
        )
        print(f"Eval greedy | mean {mean_r:.3f} | std {std_r:.3f} | weapon_counts {counts}")
        return

    last_eval_greedy_mean, last_eval_greedy_std, last_eval_greedy_counts = evaluate(
        policy,
        env,
        action_weapon_id,
        program.module,
        args.eval_episodes,
        args.seed,
        mode="greedy",
    )
    last_eval_random_mean, last_eval_random_std, last_eval_random_counts = evaluate(
        None,
        env,
        action_weapon_id,
        program.module,
        args.eval_episodes,
        args.seed + 1,
        mode="random",
    )

    rng = np.random.default_rng(args.seed)
    for update in range(1, args.updates + 1):
        transitions = []
        episode_rewards = []

        while len(transitions) < args.rollout_steps:
            ep_transitions, ep_reward = run_episode(
                env, policy, action_weapon_id, program.module, rng, train=True
            )
            transitions.extend(ep_transitions)
            episode_rewards.append(ep_reward)

        ppo_update(policy, optimizer, transitions)

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        do_eval = update % args.save_every == 0 or update == args.updates
        if do_eval:
            last_eval_greedy_mean, last_eval_greedy_std, last_eval_greedy_counts = evaluate(
                policy,
                env,
                action_weapon_id,
                program.module,
                args.eval_episodes,
                args.seed + update,
                mode="greedy",
            )
            last_eval_random_mean, last_eval_random_std, last_eval_random_counts = evaluate(
                None,
                env,
                action_weapon_id,
                program.module,
                args.eval_episodes,
                args.seed + update + 1,
                mode="random",
            )
            print(
                f"Eval greedy | mean {last_eval_greedy_mean:.3f} | std {last_eval_greedy_std:.3f} | weapon_counts {last_eval_greedy_counts}"
            )
            print(
                f"Eval random | mean {last_eval_random_mean:.3f} | std {last_eval_random_std:.3f} | weapon_counts {last_eval_random_counts}"
            )

        print(
            "Update {:03d} | avg_ep_reward {:.3f} | eval_greedy {:.3f} | eval_random {:.3f}".format(
                update, avg_reward, last_eval_greedy_mean, last_eval_random_mean
            )
        )

        if update % args.save_every == 0 or update == args.updates:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
