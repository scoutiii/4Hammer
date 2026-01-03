import argparse
import csv
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import rlc
from ml.env import SingleRLCEnvironment, exit_on_invalid_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parents[5]
SCENARIO_PATH = Path(__file__).with_name("sp2_stage1_target_choice.rl")
SRC_DIR = REPO_ROOT / "src"
RUNS_DIR = REPO_ROOT / "runs" / "sp2_stage1"

SHOOTER_UNIT_INDEX = 0
MAX_TARGETS = 6
TARGET_ID_START = 1
TARGET_ID_RANGE = list(range(TARGET_ID_START, TARGET_ID_START + MAX_TARGETS))
TARGET_FEATURES = 9
ACTION_FEATURES = 8

TARGET_RE = re.compile(r"select_target\s*\{\s*target_id:\s*(\d+)\s*\}")
TARGET_ID_FALLBACK_RE = re.compile(r"target_id\s*:\s*(\d+)")


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


class TargetPolicy(nn.Module):
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


def safe_to_string(module, obj):
    try:
        return rlc_string_to_py(module.to_string(obj))
    except Exception:
        return None


def default_outdir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / timestamp


def ensure_outdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_action_target_id_map(program, actions):
    mapping = {}
    for idx, action in enumerate(actions):
        s = rlc_string_to_py(program.module.to_string(action)).strip()
        if "select_target" not in s:
            mapping[idx] = None
            continue
        match = TARGET_RE.search(s)
        if not match:
            match = TARGET_ID_FALLBACK_RE.search(s)
        mapping[idx] = int(match.group(1)) if match else None
    return mapping


def profile_basic_features(profile):
    return [
        float(profile.thoughness()),
        float(profile.save()),
        float(profile.invuln_save()),
        float(profile.wounds()),
    ]


def profile_shooter_features(profile):
    return [
        float(profile.thoughness()),
        float(profile.save()),
        float(profile.wounds()),
    ]


def unit_total_wounds(unit):
    total = 0.0
    for i in range(unit.models.size()):
        total += float(unit.models.get(i).contents.wounds_left())
    return total


def encode_state(state, value_weights, max_targets=MAX_TARGETS):
    board = state.board
    shooter = board.units.get(SHOOTER_UNIT_INDEX).contents

    shooter_models = float(shooter.models.size())
    shooter_can_shoot = float(shooter.can_shoot)
    shooter_has_shoot = float(shooter.has_shoot)
    if shooter.models.size() > 0:
        shooter_profile = shooter.models.get(0).contents.profile
        shooter_feats = profile_shooter_features(shooter_profile)
    else:
        shooter_feats = [0.0, 0.0, 0.0]

    target_feats = []
    pad = [0.0] * TARGET_FEATURES
    for slot in range(max_targets):
        target_id = TARGET_ID_START + slot
        if board.units.size() <= target_id:
            target_feats.extend(pad)
            continue
        target = board.units.get(target_id).contents
        if target.models.size() == 0 or not target.owned_by_player1:
            target_feats.extend(pad)
            continue
        distance = float(shooter.distance(target))
        models = float(target.models.size())
        total_wounds = unit_total_wounds(target)
        if target.models.size() > 0:
            profile_feats = profile_basic_features(target.models.get(0).contents.profile)
        else:
            profile_feats = [0.0, 0.0, 0.0, 0.0]
        value_weight = float(value_weights.get(target_id, 1.0))
        slot_mask = 1.0
        target_feats.extend(
            [
                distance,
                models,
                total_wounds,
                *profile_feats,
                value_weight,
                slot_mask,
            ]
        )

    return np.array(
        [
            shooter_models,
            shooter_can_shoot,
            shooter_has_shoot,
            *shooter_feats,
            *target_feats,
        ],
        dtype=np.float32,
    )


def encode_action(shooter, target, value_weight):
    distance = float(shooter.distance(target))
    models = float(target.models.size())
    total_wounds = unit_total_wounds(target)
    if target.models.size() > 0:
        profile_feats = profile_basic_features(target.models.get(0).contents.profile)
    else:
        profile_feats = [0.0, 0.0, 0.0, 0.0]
    return np.array(
        [distance, models, total_wounds, *profile_feats, float(value_weight)],
        dtype=np.float32,
    )


def collect_target_actions(env, state, action_target_id):
    mask = env.get_action_mask()
    legal_indices = np.where(mask == 1)[0]
    board = state.board

    action_indices = []
    target_ids = []
    for idx in legal_indices:
        target_id = action_target_id.get(idx)
        if target_id is None:
            continue
        if target_id < TARGET_ID_START or target_id >= TARGET_ID_START + MAX_TARGETS:
            continue
        if board.units.size() <= target_id:
            continue
        target = board.units.get(target_id).contents
        if target.models.size() == 0 or not target.owned_by_player1:
            continue
        action_indices.append(idx)
        target_ids.append(target_id)
    return action_indices, target_ids


def build_action_features(shooter, board, target_ids, value_weights):
    feats = []
    for target_id in target_ids:
        target = board.units.get(target_id).contents
        value_weight = float(value_weights.get(target_id, 1.0))
        feats.append(encode_action(shooter, target, value_weight))
    return np.stack(feats, axis=0)


def choose_auto_action(env, rng):
    mask = env.get_action_mask()
    legal_indices = np.where(mask == 1)[0]
    if len(legal_indices) == 1:
        return int(legal_indices[0])
    return int(rng.choice(legal_indices))


def is_target_state(env, target_action_indices):
    mask = env.get_action_mask()
    for idx in target_action_indices:
        if idx < len(mask) and mask[idx] == 1:
            return True
    return False


def target_total_wounds(board, target_id):
    target = board.units.get(target_id).contents
    return unit_total_wounds(target)


def advance_until_target_or_done(env, rng, target_action_indices):
    while not env.is_done_underling():
        if is_target_state(env, target_action_indices):
            break
        auto_action = choose_auto_action(env, rng)
        env.step(auto_action)


def advance_until_done(env, rng, target_action_indices):
    while not env.is_done_underling():
        auto_action = choose_auto_action(env, rng)
        env.step(auto_action)


def compute_base_reward(target_wounds_before, target_wounds_after, target_models_after, alpha, kill_bonus, overkill_penalty):
    damage = target_wounds_before - target_wounds_after
    kill = 1.0 if target_models_after == 0 else 0.0
    overkill = max(0.0, damage - target_wounds_before)
    base = alpha * damage + kill_bonus * kill - overkill_penalty * overkill
    return base, damage, kill, overkill


def assign_value_weights(target_ids, rng, weight_prob, weight_mult):
    weights = {target_id: 1.0 for target_id in target_ids}
    if target_ids and rng.random() < weight_prob:
        chosen = int(rng.choice(target_ids))
        weights[chosen] = float(weight_mult)
    return weights


def run_episode(
    env,
    policy,
    action_target_id,
    target_action_indices,
    module,
    rng,
    device,
    alpha,
    kill_bonus,
    overkill_penalty,
    value_weight_prob,
    value_weight_mult,
    train=True,
):
    env.reset()
    done = env.is_done_underling()
    transitions = []
    episode_reward = 0.0
    value_weights = None

    while not done:
        advance_until_target_or_done(env, rng, target_action_indices)
        done = env.is_done_underling()
        if done:
            break

        state = env.state.state
        action_indices, target_ids = collect_target_actions(env, state, action_target_id)
        if not action_indices:
            auto_action = choose_auto_action(env, rng)
            env.step(auto_action)
            done = env.is_done_underling()
            continue

        if value_weights is None:
            value_weights = assign_value_weights(target_ids, rng, value_weight_prob, value_weight_mult)

        board = state.board
        shooter = board.units.get(SHOOTER_UNIT_INDEX).contents
        state_vec = encode_state(state, value_weights)
        feats = build_action_features(shooter, board, target_ids, value_weights)

        with torch.no_grad():
            st = torch.tensor(state_vec, dtype=torch.float32, device=device)
            at = torch.tensor(feats, dtype=torch.float32, device=device)
            logits, value = policy.action_logits(st, at)
            dist = torch.distributions.Categorical(logits=logits)
            chosen = dist.sample() if train else torch.argmax(logits)
            logprob = dist.log_prob(chosen)

        chosen_idx = int(chosen.item())
        action_idx = action_indices[chosen_idx]
        chosen_target_id = target_ids[chosen_idx]

        target_wounds_before = target_total_wounds(state.board, chosen_target_id)
        target_models_before = state.board.units.get(chosen_target_id).contents.models.size()

        env.step(action_idx)
        advance_until_done(env, rng, target_action_indices)
        done = env.is_done_underling()

        target_wounds_after = target_total_wounds(env.state.state.board, chosen_target_id)
        target_models_after = env.state.state.board.units.get(chosen_target_id).contents.models.size()

        base, damage, kill, overkill = compute_base_reward(
            target_wounds_before,
            target_wounds_after,
            target_models_after,
            alpha,
            kill_bonus,
            overkill_penalty,
        )
        reward = base * float(value_weights.get(chosen_target_id, 1.0))
        episode_reward += reward

        if train:
            next_value = 0.0
            transitions.append(
                Transition(
                    state=state_vec,
                    action_features=feats,
                    action_index=chosen_idx,
                    old_logprob=float(logprob.item()),
                    value=float(value.item()),
                    next_value=next_value,
                    reward=float(reward),
                    done=True,
                )
            )
        break

    return transitions, episode_reward


def run_episode_eval(
    env,
    policy,
    action_target_id,
    target_action_indices,
    module,
    rng,
    device,
    alpha,
    kill_bonus,
    overkill_penalty,
    value_weight_prob,
    value_weight_mult,
    mode="greedy",
):
    env.reset()
    done = env.is_done_underling()
    episode_reward = 0.0
    target_counts = {}
    chosen_target_id = None
    value_weights = None

    while not done:
        advance_until_target_or_done(env, rng, target_action_indices)
        done = env.is_done_underling()
        if done:
            break

        state = env.state.state
        action_indices, target_ids = collect_target_actions(env, state, action_target_id)
        if not action_indices:
            auto_action = choose_auto_action(env, rng)
            env.step(auto_action)
            done = env.is_done_underling()
            continue

        if value_weights is None:
            value_weights = assign_value_weights(target_ids, rng, value_weight_prob, value_weight_mult)

        if mode == "random" or policy is None:
            chosen_idx = int(rng.integers(len(action_indices)))
        else:
            board = state.board
            shooter = board.units.get(SHOOTER_UNIT_INDEX).contents
            state_vec = encode_state(state, value_weights)
            feats = build_action_features(shooter, board, target_ids, value_weights)
            with torch.no_grad():
                st = torch.tensor(state_vec, dtype=torch.float32, device=device)
                at = torch.tensor(feats, dtype=torch.float32, device=device)
                logits, _ = policy.action_logits(st, at)
                chosen_idx = int(torch.argmax(logits).item())

        action_idx = action_indices[chosen_idx]
        chosen_target_id = target_ids[chosen_idx]
        target_counts[chosen_target_id] = target_counts.get(chosen_target_id, 0) + 1

        target_wounds_before = target_total_wounds(state.board, chosen_target_id)
        target_models_before = state.board.units.get(chosen_target_id).contents.models.size()

        env.step(action_idx)
        advance_until_done(env, rng, target_action_indices)
        done = env.is_done_underling()

        target_wounds_after = target_total_wounds(env.state.state.board, chosen_target_id)
        target_models_after = env.state.state.board.units.get(chosen_target_id).contents.models.size()

        base, damage, kill, overkill = compute_base_reward(
            target_wounds_before,
            target_wounds_after,
            target_models_after,
            alpha,
            kill_bonus,
            overkill_penalty,
        )
        reward = base * float(value_weights.get(chosen_target_id, 1.0))
        episode_reward += reward
        break

    return episode_reward, target_counts, chosen_target_id


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


def ppo_update(
    policy,
    optimizer,
    transitions,
    device,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
    minibatch_size=256,
):
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
                st = torch.tensor(t.state, dtype=torch.float32, device=device)
                at = torch.tensor(t.action_features, dtype=torch.float32, device=device)
                logits, value = policy.action_logits(st, at)
                dist = torch.distributions.Categorical(logits=logits)

                action_tensor = torch.tensor(t.action_index, dtype=torch.long, device=device)
                logprob = dist.log_prob(action_tensor)
                entropy = dist.entropy()

                old = torch.tensor(t.old_logprob, dtype=torch.float32, device=device)
                ratio = torch.exp(logprob - old)
                adv = torch.tensor(advantages[idx], dtype=torch.float32, device=device)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                policy_losses.append(-torch.min(surr1, surr2))

                ret = torch.tensor(returns[idx], dtype=torch.float32, device=device)
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


def summarize_rewards(rewards):
    if not rewards:
        return 0.0, 0.0
    arr = np.array(rewards, dtype=np.float32)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


def paired_diff_stats(diffs):
    if not diffs:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(diffs, dtype=np.float32)
    mean_d = float(np.mean(arr))
    sd_d = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se_d = sd_d / math.sqrt(len(arr)) if len(arr) > 0 else 0.0
    t_stat = mean_d / se_d if se_d > 0 else 0.0
    ci_low = mean_d - 1.96 * se_d
    ci_high = mean_d + 1.96 * se_d
    return mean_d, sd_d, se_d, t_stat, ci_low, ci_high


def aggregate_counts(dest, src):
    for key, val in src.items():
        dest[key] = dest.get(key, 0) + val


def counts_to_freqs(counts, target_ids):
    total = float(sum(counts.values()))
    freqs = {}
    for tid in target_ids:
        if total > 0.0:
            freqs[tid] = float(counts.get(tid, 0)) / total
        else:
            freqs[tid] = 0.0
    return freqs


def evaluate_paired(
    policy,
    env,
    action_target_id,
    target_action_indices,
    module,
    device,
    alpha,
    kill_bonus,
    overkill_penalty,
    value_weight_prob,
    value_weight_mult,
    episodes=20,
    seed=0,
):
    greedy_rewards = []
    random_rewards = []
    diffs = []
    greedy_counts = {}
    random_counts = {}

    if policy is not None:
        policy.eval()

    for i in range(episodes):
        seed_i = seed + i
        greedy_rng = np.random.default_rng(seed_i)
        greedy_reward, greedy_ep_counts, _ = run_episode_eval(
            env,
            policy,
            action_target_id,
            target_action_indices,
            module,
            greedy_rng,
            device,
            alpha,
            kill_bonus,
            overkill_penalty,
            value_weight_prob,
            value_weight_mult,
            mode="greedy",
        )
        random_rng = np.random.default_rng(seed_i)
        random_reward, random_ep_counts, _ = run_episode_eval(
            env,
            None,
            action_target_id,
            target_action_indices,
            module,
            random_rng,
            device,
            alpha,
            kill_bonus,
            overkill_penalty,
            value_weight_prob,
            value_weight_mult,
            mode="random",
        )
        greedy_rewards.append(greedy_reward)
        random_rewards.append(random_reward)
        diffs.append(greedy_reward - random_reward)
        aggregate_counts(greedy_counts, greedy_ep_counts)
        aggregate_counts(random_counts, random_ep_counts)

    if policy is not None:
        policy.train()

    greedy_mean, greedy_std = summarize_rewards(greedy_rewards)
    random_mean, random_std = summarize_rewards(random_rewards)
    diff_mean, diff_std, diff_se, diff_t, diff_ci_low, diff_ci_high = paired_diff_stats(
        diffs
    )

    return (
        greedy_mean,
        greedy_std,
        random_mean,
        random_std,
        diff_mean,
        diff_std,
        diff_se,
        diff_t,
        diff_ci_low,
        diff_ci_high,
        greedy_counts,
        random_counts,
    )


def run_forced_episode(
    env,
    action_target_id,
    target_action_indices,
    module,
    setup_rng,
    rollout_rng,
    forced_target_id,
    alpha,
    kill_bonus,
    overkill_penalty,
    value_weights,
):
    env.reset()
    advance_until_target_or_done(env, setup_rng, target_action_indices)
    if env.is_done_underling():
        return None

    state = env.state.state
    action_indices, target_ids = collect_target_actions(env, state, action_target_id)
    if forced_target_id not in target_ids:
        return None

    chosen_idx = target_ids.index(forced_target_id)
    action_idx = action_indices[chosen_idx]

    target_wounds_before = target_total_wounds(state.board, forced_target_id)
    target_models_before = state.board.units.get(forced_target_id).contents.models.size()

    env.step(action_idx)
    advance_until_done(env, rollout_rng, target_action_indices)

    target_wounds_after = target_total_wounds(env.state.state.board, forced_target_id)
    target_models_after = env.state.state.board.units.get(forced_target_id).contents.models.size()

    base, damage, kill, overkill = compute_base_reward(
        target_wounds_before,
        target_wounds_after,
        target_models_after,
        alpha,
        kill_bonus,
        overkill_penalty,
    )
    reward = base * float(value_weights.get(forced_target_id, 1.0))
    return reward


def make_rollout_seed(episode_seed, target_id, sim_index):
    return (episode_seed * 1000003 + target_id * 9176 + sim_index) % (2**32 - 1)


def evaluate_oracle(
    policy,
    env,
    action_target_id,
    target_action_indices,
    module,
    device,
    alpha,
    kill_bonus,
    overkill_penalty,
    value_weight_prob,
    value_weight_mult,
    oracle_sims,
    oracle_eval_episodes,
    oracle_max_targets,
    seed=0,
):
    accuracy = []
    regrets = []
    greedy_rewards = []
    best_mean_rewards = []
    best_target_counts = {}

    if policy is not None:
        policy.eval()

    start = time.perf_counter()
    progress_interval = max(1, oracle_eval_episodes // 5)

    for i in range(oracle_eval_episodes):
        episode_seed = seed + i

        greedy_rng = np.random.default_rng(episode_seed)
        greedy_reward, _, greedy_target = run_episode_eval(
            env,
            policy,
            action_target_id,
            target_action_indices,
            module,
            greedy_rng,
            device,
            alpha,
            kill_bonus,
            overkill_penalty,
            value_weight_prob,
            value_weight_mult,
            mode="greedy",
        )
        if greedy_target is None:
            continue

        setup_rng = np.random.default_rng(episode_seed)
        env.reset()
        advance_until_target_or_done(env, setup_rng, target_action_indices)
        if env.is_done_underling():
            continue

        state = env.state.state
        action_indices, target_ids = collect_target_actions(env, state, action_target_id)
        if not target_ids:
            continue

        value_weights = assign_value_weights(target_ids, setup_rng, value_weight_prob, value_weight_mult)
        if oracle_max_targets and len(target_ids) > oracle_max_targets:
            target_ids = target_ids[:oracle_max_targets]

        mean_rewards = {}
        for target_id in target_ids:
            rewards = []
            for sim_idx in range(oracle_sims):
                setup_rng_sim = np.random.default_rng(episode_seed)
                rollout_rng = np.random.default_rng(
                    make_rollout_seed(episode_seed, target_id, sim_idx)
                )
                reward = run_forced_episode(
                    env,
                    action_target_id,
                    target_action_indices,
                    module,
                    setup_rng_sim,
                    rollout_rng,
                    target_id,
                    alpha,
                    kill_bonus,
                    overkill_penalty,
                    value_weights,
                )
                if reward is None:
                    continue
                rewards.append(reward)
            mean_rewards[target_id] = float(np.mean(rewards)) if rewards else 0.0

        best_target = max(mean_rewards, key=mean_rewards.get)
        best_reward = mean_rewards[best_target]
        best_mean_rewards.append(best_reward)
        greedy_rewards.append(greedy_reward)
        accuracy.append(1.0 if greedy_target == best_target else 0.0)
        regrets.append(best_reward - greedy_reward)
        best_target_counts[best_target] = best_target_counts.get(best_target, 0) + 1

        if (i + 1) % progress_interval == 0:
            elapsed = time.perf_counter() - start
            per_ep = elapsed / (i + 1)
            remaining = per_ep * (oracle_eval_episodes - i - 1)
            print(
                "Oracle eval {}/{} | elapsed {:.1f}s | ETA {:.1f}s".format(
                    i + 1, oracle_eval_episodes, elapsed, remaining
                )
            )

    if policy is not None:
        policy.train()

    used_episodes = len(regrets)
    accuracy_mean = float(np.mean(accuracy)) if accuracy else 0.0
    regret_mean = float(np.mean(regrets)) if regrets else 0.0
    regret_std = float(np.std(regrets, ddof=1)) if len(regrets) > 1 else 0.0
    best_mean_reward = float(np.mean(best_mean_rewards)) if best_mean_rewards else 0.0
    greedy_mean_reward = float(np.mean(greedy_rewards)) if greedy_rewards else 0.0

    return (
        accuracy_mean,
        regret_mean,
        regret_std,
        best_mean_reward,
        greedy_mean_reward,
        used_episodes,
        best_target_counts,
    )


def write_metrics_header(path, fieldnames):
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_metrics_row(path, fieldnames, row):
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def plot_metrics(rows, target_ids, outpath):
    if not rows:
        return
    import matplotlib.pyplot as plt

    updates = [row["update"] for row in rows]
    greedy_mean = [row["eval_greedy_mean"] for row in rows]
    greedy_se = [
        (row["eval_greedy_std"] / math.sqrt(row["eval_episodes"]))
        if row["eval_episodes"] > 0
        else 0.0
        for row in rows
    ]
    random_mean = [row["eval_random_mean"] for row in rows]
    random_se = [
        (row["eval_random_std"] / math.sqrt(row["eval_episodes"]))
        if row["eval_episodes"] > 0
        else 0.0
        for row in rows
    ]
    diff_mean = [row["paired_diff_mean"] for row in rows]
    diff_se = [row["paired_diff_se"] for row in rows]

    oracle_enabled = any("oracle_accuracy" in row for row in rows)
    num_plots = 5 if oracle_enabled else 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    axes[0].errorbar(updates, greedy_mean, yerr=greedy_se, label="greedy")
    axes[0].errorbar(updates, random_mean, yerr=random_se, label="random")
    axes[0].set_ylabel("Eval reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(updates, diff_mean, yerr=diff_se, label="greedy-random")
    axes[1].set_ylabel("Paired diff")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for tid in target_ids:
        key = f"greedy_target_freq_{tid}"
        series = [row.get(key, 0.0) for row in rows]
        axes[2].plot(updates, series, label=f"target {tid}")
    axes[2].set_xlabel("Update")
    axes[2].set_ylabel("Greedy target freq")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if oracle_enabled:
        oracle_accuracy = [row.get("oracle_accuracy", 0.0) for row in rows]
        oracle_regret = [row.get("oracle_regret_mean", 0.0) for row in rows]
        oracle_regret_se = [
            (row.get("oracle_regret_std", 0.0) / math.sqrt(row.get("oracle_eval_episodes", 1)))
            if row.get("oracle_eval_episodes", 0) > 1
            else 0.0
            for row in rows
        ]

        axes[3].plot(updates, oracle_accuracy, label="oracle_accuracy")
        axes[3].set_ylabel("Oracle accuracy")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        axes[4].errorbar(updates, oracle_regret, yerr=oracle_regret_se, label="oracle_regret")
        axes[4].set_xlabel("Update")
        axes[4].set_ylabel("Oracle regret")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_debug_episode(
    env,
    action_target_id,
    target_action_indices,
    module,
    rng,
    alpha,
    kill_bonus,
    overkill_penalty,
    value_weight_prob,
    value_weight_mult,
):
    env.reset()
    done = env.is_done_underling()

    while not done:
        advance_until_target_or_done(env, rng, target_action_indices)
        done = env.is_done_underling()
        if done:
            break

        state = env.state.state
        action_indices, target_ids = collect_target_actions(env, state, action_target_id)
        print("legal select_target actions:")
        actions = env.actions()
        for idx in action_indices:
            action_str = safe_to_string(module, actions[idx]) or "<unavailable>"
            print(f"  {idx} | {action_str}")

        if not action_indices:
            auto_action = choose_auto_action(env, rng)
            env.step(auto_action)
            done = env.is_done_underling()
            continue

        value_weights = assign_value_weights(target_ids, rng, value_weight_prob, value_weight_mult)
        for target_id in target_ids:
            print(f"  target {target_id} value_weight {value_weights.get(target_id, 1.0):.1f}")

        chosen_idx = int(rng.integers(len(action_indices)))
        action_idx = action_indices[chosen_idx]
        chosen_target_id = target_ids[chosen_idx]
        target_wounds_before = target_total_wounds(state.board, chosen_target_id)
        target_models_before = state.board.units.get(chosen_target_id).contents.models.size()

        print(f"chosen target id: {chosen_target_id}")
        print(
            f"target before: wounds {target_wounds_before:.1f} | models {target_models_before}"
        )

        env.step(action_idx)
        advance_until_done(env, rng, target_action_indices)
        done = env.is_done_underling()

        target_wounds_after = target_total_wounds(env.state.state.board, chosen_target_id)
        target_models_after = env.state.state.board.units.get(chosen_target_id).contents.models.size()

        base, damage, kill, overkill = compute_base_reward(
            target_wounds_before,
            target_wounds_after,
            target_models_after,
            alpha,
            kill_bonus,
            overkill_penalty,
        )
        reward = base * float(value_weights.get(chosen_target_id, 1.0))
        print(
            "reward breakdown: damage {:.1f} | kill {} | overkill {:.1f} | base {:.3f} | reward {:.3f}".format(
                damage, int(kill), overkill, base, reward
            )
        )
        print(
            f"target after: wounds {target_wounds_after:.1f} | models {target_models_after}"
        )
        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--debug-episode", action="store_true")
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kill-bonus", type=float, default=5.0)
    parser.add_argument("--overkill-penalty", type=float, default=0.5)
    parser.add_argument("--value-weight-prob", type=float, default=0.5)
    parser.add_argument("--value-weight-mult", type=float, default=2.0)
    parser.add_argument("--oracle-eval", action="store_true")
    parser.add_argument("--oracle-sims", type=int, default=200)
    parser.add_argument("--oracle-max-targets", type=int, default=6)
    parser.add_argument("--oracle-eval-episodes", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed

    print(DEVICE)

    rlc_path = find_rlc_compiler()
    program = rlc.compile(
        [str(SCENARIO_PATH)],
        rlc_compiler=rlc_path,
        rlc_includes=[str(SRC_DIR)],
    )
    exit_on_invalid_env(program, forced_one_player=False, needs_score=True)

    env = SingleRLCEnvironment(program)
    actions = env.actions()
    action_target_id = build_action_target_id_map(program, actions)
    target_action_indices = [idx for idx, tid in action_target_id.items() if tid is not None]
    target_ids = sorted({tid for tid in action_target_id.values() if tid is not None})

    if args.debug_episode:
        rng = np.random.default_rng(args.seed)
        run_debug_episode(
            env,
            action_target_id,
            target_action_indices,
            program.module,
            rng,
            args.alpha,
            args.kill_bonus,
            args.overkill_penalty,
            args.value_weight_prob,
            args.value_weight_mult,
        )
        return

    outdir = Path(args.outdir) if args.outdir else default_outdir()
    outdir = ensure_outdir(outdir)
    metrics_path = outdir / "metrics.csv"

    metrics_fields = [
        "update",
        "avg_ep_reward",
        "eval_greedy_mean",
        "eval_greedy_std",
        "eval_random_mean",
        "eval_random_std",
        "paired_diff_mean",
        "paired_diff_std",
        "paired_diff_se",
        "paired_diff_t",
        "paired_diff_ci_low",
        "paired_diff_ci_high",
        "eval_episodes",
        "eval_seed",
    ]
    for tid in target_ids:
        metrics_fields.append(f"greedy_target_freq_{tid}")
    for tid in target_ids:
        metrics_fields.append(f"random_target_freq_{tid}")

    if args.oracle_eval:
        metrics_fields.extend(
            [
                "oracle_accuracy",
                "oracle_regret_mean",
                "oracle_regret_std",
                "oracle_best_mean_reward",
                "oracle_greedy_mean_reward",
                "oracle_eval_episodes",
            ]
        )
        for tid in target_ids:
            metrics_fields.append(f"oracle_best_target_freq_{tid}")

    write_metrics_header(metrics_path, metrics_fields)
    metrics_rows = []

    state_dim = len(encode_state(env.state.state, {}))
    action_dim = ACTION_FEATURES

    policy = TargetPolicy(state_dim, action_dim)
    policy.to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "sp2_stage1_ppo.pt"

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)

    oracle_eval_episodes = min(args.oracle_eval_episodes, args.eval_episodes)

    if args.eval:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        data = torch.load(ckpt_path, map_location=DEVICE)
        policy.load_state_dict(data["model"])

        (
            greedy_mean,
            greedy_std,
            random_mean,
            random_std,
            diff_mean,
            diff_std,
            diff_se,
            diff_t,
            diff_ci_low,
            diff_ci_high,
            greedy_counts,
            random_counts,
        ) = evaluate_paired(
            policy,
            env,
            action_target_id,
            target_action_indices,
            program.module,
            DEVICE,
            args.alpha,
            args.kill_bonus,
            args.overkill_penalty,
            args.value_weight_prob,
            args.value_weight_mult,
            episodes=args.eval_episodes,
            seed=eval_seed,
        )

        greedy_freqs = counts_to_freqs(greedy_counts, target_ids)
        random_freqs = counts_to_freqs(random_counts, target_ids)
        print(
            f"Eval greedy | mean {greedy_mean:.3f} | std {greedy_std:.3f} | target_counts {greedy_counts}"
        )
        print(
            f"Eval random | mean {random_mean:.3f} | std {random_std:.3f} | target_counts {random_counts}"
        )
        print(
            "Paired diff (greedy-random): mean {:.3f} | SE {:.3f} | t {:.2f} | 95% CI [{:.3f}, {:.3f}] | n {}".format(
                diff_mean,
                diff_se,
                diff_t,
                diff_ci_low,
                diff_ci_high,
                args.eval_episodes,
            )
        )

        row = {
            "update": 0,
            "avg_ep_reward": 0.0,
            "eval_greedy_mean": greedy_mean,
            "eval_greedy_std": greedy_std,
            "eval_random_mean": random_mean,
            "eval_random_std": random_std,
            "paired_diff_mean": diff_mean,
            "paired_diff_std": diff_std,
            "paired_diff_se": diff_se,
            "paired_diff_t": diff_t,
            "paired_diff_ci_low": diff_ci_low,
            "paired_diff_ci_high": diff_ci_high,
            "eval_episodes": args.eval_episodes,
            "eval_seed": eval_seed,
        }
        for tid in target_ids:
            row[f"greedy_target_freq_{tid}"] = greedy_freqs.get(tid, 0.0)
        for tid in target_ids:
            row[f"random_target_freq_{tid}"] = random_freqs.get(tid, 0.0)

        if args.oracle_eval:
            (
                oracle_accuracy,
                oracle_regret_mean,
                oracle_regret_std,
                oracle_best_mean_reward,
                oracle_greedy_mean_reward,
                oracle_used_episodes,
                oracle_best_counts,
            ) = evaluate_oracle(
                policy,
                env,
                action_target_id,
                target_action_indices,
                program.module,
                DEVICE,
                args.alpha,
                args.kill_bonus,
                args.overkill_penalty,
                args.value_weight_prob,
                args.value_weight_mult,
                args.oracle_sims,
                oracle_eval_episodes,
                args.oracle_max_targets,
                seed=eval_seed,
            )
            oracle_freqs = counts_to_freqs(oracle_best_counts, target_ids)
            print(
                "Oracle | accuracy {:.3f} | regret mean {:.3f} std {:.3f} | episodes {}".format(
                    oracle_accuracy, oracle_regret_mean, oracle_regret_std, oracle_used_episodes
                )
            )
            print(f"Oracle best target counts: {oracle_best_counts}")
            row.update(
                {
                    "oracle_accuracy": oracle_accuracy,
                    "oracle_regret_mean": oracle_regret_mean,
                    "oracle_regret_std": oracle_regret_std,
                    "oracle_best_mean_reward": oracle_best_mean_reward,
                    "oracle_greedy_mean_reward": oracle_greedy_mean_reward,
                    "oracle_eval_episodes": oracle_used_episodes,
                }
            )
            for tid in target_ids:
                row[f"oracle_best_target_freq_{tid}"] = oracle_freqs.get(tid, 0.0)

        append_metrics_row(metrics_path, metrics_fields, row)
        return

    (
        greedy_mean,
        greedy_std,
        random_mean,
        random_std,
        diff_mean,
        diff_std,
        diff_se,
        diff_t,
        diff_ci_low,
        diff_ci_high,
        greedy_counts,
        random_counts,
    ) = evaluate_paired(
        policy,
        env,
        action_target_id,
        target_action_indices,
        program.module,
        DEVICE,
        args.alpha,
        args.kill_bonus,
        args.overkill_penalty,
        args.value_weight_prob,
        args.value_weight_mult,
        episodes=args.eval_episodes,
        seed=eval_seed,
    )

    greedy_freqs = counts_to_freqs(greedy_counts, target_ids)
    random_freqs = counts_to_freqs(random_counts, target_ids)

    last_eval = {
        "greedy_mean": greedy_mean,
        "greedy_std": greedy_std,
        "random_mean": random_mean,
        "random_std": random_std,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_se": diff_se,
        "diff_t": diff_t,
        "diff_ci_low": diff_ci_low,
        "diff_ci_high": diff_ci_high,
        "greedy_freqs": greedy_freqs,
        "random_freqs": random_freqs,
        "eval_seed": eval_seed,
        "oracle_accuracy": 0.0,
        "oracle_regret_mean": 0.0,
        "oracle_regret_std": 0.0,
        "oracle_best_mean_reward": 0.0,
        "oracle_greedy_mean_reward": 0.0,
        "oracle_eval_episodes": 0,
        "oracle_best_freqs": {tid: 0.0 for tid in target_ids},
    }

    if args.oracle_eval:
        (
            oracle_accuracy,
            oracle_regret_mean,
            oracle_regret_std,
            oracle_best_mean_reward,
            oracle_greedy_mean_reward,
            oracle_used_episodes,
            oracle_best_counts,
        ) = evaluate_oracle(
            policy,
            env,
            action_target_id,
            target_action_indices,
            program.module,
            DEVICE,
            args.alpha,
            args.kill_bonus,
            args.overkill_penalty,
            args.value_weight_prob,
            args.value_weight_mult,
            args.oracle_sims,
            oracle_eval_episodes,
            args.oracle_max_targets,
            seed=eval_seed,
        )
        oracle_freqs = counts_to_freqs(oracle_best_counts, target_ids)
        last_eval.update(
            {
                "oracle_accuracy": oracle_accuracy,
                "oracle_regret_mean": oracle_regret_mean,
                "oracle_regret_std": oracle_regret_std,
                "oracle_best_mean_reward": oracle_best_mean_reward,
                "oracle_greedy_mean_reward": oracle_greedy_mean_reward,
                "oracle_eval_episodes": oracle_used_episodes,
                "oracle_best_freqs": oracle_freqs,
            }
        )

    rng = np.random.default_rng(args.seed)
    for update in range(1, args.updates + 1):
        transitions = []
        episode_rewards = []

        while len(transitions) < args.rollout_steps:
            ep_transitions, ep_reward = run_episode(
                env,
                policy,
                action_target_id,
                target_action_indices,
                program.module,
                rng,
                DEVICE,
                args.alpha,
                args.kill_bonus,
                args.overkill_penalty,
                args.value_weight_prob,
                args.value_weight_mult,
                train=True,
            )
            transitions.extend(ep_transitions)
            episode_rewards.append(ep_reward)

        ppo_update(policy, optimizer, transitions, DEVICE)

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        do_eval = update % args.save_every == 0 or update == args.updates
        if do_eval:
            (
                greedy_mean,
                greedy_std,
                random_mean,
                random_std,
                diff_mean,
                diff_std,
                diff_se,
                diff_t,
                diff_ci_low,
                diff_ci_high,
                greedy_counts,
                random_counts,
            ) = evaluate_paired(
                policy,
                env,
                action_target_id,
                target_action_indices,
                program.module,
                DEVICE,
                args.alpha,
                args.kill_bonus,
                args.overkill_penalty,
                args.value_weight_prob,
                args.value_weight_mult,
                episodes=args.eval_episodes,
                seed=eval_seed + update,
            )

            greedy_freqs = counts_to_freqs(greedy_counts, target_ids)
            random_freqs = counts_to_freqs(random_counts, target_ids)

            print(
                f"Eval greedy | mean {greedy_mean:.3f} | std {greedy_std:.3f} | target_counts {greedy_counts}"
            )
            print(
                f"Eval random | mean {random_mean:.3f} | std {random_std:.3f} | target_counts {random_counts}"
            )
            print(
                "Paired diff (greedy-random): mean {:.3f} | SE {:.3f} | t {:.2f} | 95% CI [{:.3f}, {:.3f}] | n {}".format(
                    diff_mean,
                    diff_se,
                    diff_t,
                    diff_ci_low,
                    diff_ci_high,
                    args.eval_episodes,
                )
            )

            last_eval = {
                "greedy_mean": greedy_mean,
                "greedy_std": greedy_std,
                "random_mean": random_mean,
                "random_std": random_std,
                "diff_mean": diff_mean,
                "diff_std": diff_std,
                "diff_se": diff_se,
                "diff_t": diff_t,
                "diff_ci_low": diff_ci_low,
                "diff_ci_high": diff_ci_high,
                "greedy_freqs": greedy_freqs,
                "random_freqs": random_freqs,
                "eval_seed": eval_seed + update,
                "oracle_accuracy": last_eval.get("oracle_accuracy", 0.0),
                "oracle_regret_mean": last_eval.get("oracle_regret_mean", 0.0),
                "oracle_regret_std": last_eval.get("oracle_regret_std", 0.0),
                "oracle_best_mean_reward": last_eval.get("oracle_best_mean_reward", 0.0),
                "oracle_greedy_mean_reward": last_eval.get("oracle_greedy_mean_reward", 0.0),
                "oracle_eval_episodes": last_eval.get("oracle_eval_episodes", 0),
                "oracle_best_freqs": last_eval.get("oracle_best_freqs", {tid: 0.0 for tid in target_ids}),
            }

            if args.oracle_eval:
                (
                    oracle_accuracy,
                    oracle_regret_mean,
                    oracle_regret_std,
                    oracle_best_mean_reward,
                    oracle_greedy_mean_reward,
                    oracle_used_episodes,
                    oracle_best_counts,
                ) = evaluate_oracle(
                    policy,
                    env,
                    action_target_id,
                    target_action_indices,
                    program.module,
                    DEVICE,
                    args.alpha,
                    args.kill_bonus,
                    args.overkill_penalty,
                    args.value_weight_prob,
                    args.value_weight_mult,
                    args.oracle_sims,
                    oracle_eval_episodes,
                    args.oracle_max_targets,
                    seed=eval_seed + update,
                )
                oracle_freqs = counts_to_freqs(oracle_best_counts, target_ids)
                print(
                    "Oracle | accuracy {:.3f} | regret mean {:.3f} std {:.3f} | episodes {}".format(
                        oracle_accuracy,
                        oracle_regret_mean,
                        oracle_regret_std,
                        oracle_used_episodes,
                    )
                )
                print(f"Oracle best target counts: {oracle_best_counts}")
                last_eval.update(
                    {
                        "oracle_accuracy": oracle_accuracy,
                        "oracle_regret_mean": oracle_regret_mean,
                        "oracle_regret_std": oracle_regret_std,
                        "oracle_best_mean_reward": oracle_best_mean_reward,
                        "oracle_greedy_mean_reward": oracle_greedy_mean_reward,
                        "oracle_eval_episodes": oracle_used_episodes,
                        "oracle_best_freqs": oracle_freqs,
                    }
                )

        print(
            "Update {:03d} | avg_ep_reward {:.3f} | eval_greedy {:.3f} | eval_random {:.3f}".format(
                update,
                avg_reward,
                last_eval["greedy_mean"],
                last_eval["random_mean"],
            )
        )

        row = {
            "update": update,
            "avg_ep_reward": avg_reward,
            "eval_greedy_mean": last_eval["greedy_mean"],
            "eval_greedy_std": last_eval["greedy_std"],
            "eval_random_mean": last_eval["random_mean"],
            "eval_random_std": last_eval["random_std"],
            "paired_diff_mean": last_eval["diff_mean"],
            "paired_diff_std": last_eval["diff_std"],
            "paired_diff_se": last_eval["diff_se"],
            "paired_diff_t": last_eval["diff_t"],
            "paired_diff_ci_low": last_eval["diff_ci_low"],
            "paired_diff_ci_high": last_eval["diff_ci_high"],
            "eval_episodes": args.eval_episodes,
            "eval_seed": last_eval["eval_seed"],
        }
        for tid in target_ids:
            row[f"greedy_target_freq_{tid}"] = last_eval["greedy_freqs"].get(tid, 0.0)
        for tid in target_ids:
            row[f"random_target_freq_{tid}"] = last_eval["random_freqs"].get(tid, 0.0)

        if args.oracle_eval:
            row.update(
                {
                    "oracle_accuracy": last_eval.get("oracle_accuracy", 0.0),
                    "oracle_regret_mean": last_eval.get("oracle_regret_mean", 0.0),
                    "oracle_regret_std": last_eval.get("oracle_regret_std", 0.0),
                    "oracle_best_mean_reward": last_eval.get("oracle_best_mean_reward", 0.0),
                    "oracle_greedy_mean_reward": last_eval.get("oracle_greedy_mean_reward", 0.0),
                    "oracle_eval_episodes": last_eval.get("oracle_eval_episodes", 0),
                }
            )
            for tid in target_ids:
                row[f"oracle_best_target_freq_{tid}"] = last_eval["oracle_best_freqs"].get(
                    tid, 0.0
                )

        append_metrics_row(metrics_path, metrics_fields, row)
        metrics_rows.append(row)

        if update % args.save_every == 0 or update == args.updates:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                },
                ckpt_path,
            )

    if args.plot:
        plot_metrics(metrics_rows, target_ids, outdir / "plots.png")


if __name__ == "__main__":
    main()
