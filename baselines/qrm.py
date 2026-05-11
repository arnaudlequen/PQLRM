"""
qrm.py — Self-contained Q-Learning for Reward Machines (QRM)
=============================================================
Based on the original work by Toro Icarte et al. (ICML 2018).
Reference: https://bitbucket.org/RToroIcarte/qrm

Usage
-----
1.  Define your reward machines using the builder functions at the bottom.
2.  Implement `get_events(state, action, next_state) -> set[str]` for your env.
3.  Instantiate QRMAgent and call `select_action` / `update` each step.

Key design decisions vs. the original:
- No file-based RM loading; RMs are built programmatically.
- All RM logic, reward functions, and QRM agent in one file.
- Pure NumPy tabular Q-tables (no TensorFlow).
- Clean Python 3.10+ type hints throughout.
"""

from __future__ import annotations

import random
import numpy as np

from environments.reward_machines.reward_machine import RewardMachine
from environments.reward_machines.reward_functions import RewardFunction, ConstantRewardFunction, SumRewardFunction

# ---------------------------------------------------------------------------
# QRMAgent  —  single-task learner (one Q-table per RM state)
# ---------------------------------------------------------------------------

class QRMAgent:
    """
    Q-Learning for Reward Machines (tabular, single task).

    Holds one Q-table per RM state:  Q[u][s, a] → float.

    The key QRM idea: at every environment transition (s, a, s', props)
    we perform a Q-learning update for *every* RM state u, not just the
    one currently active.  This is the counterfactual / off-policy
    multi-update trick that makes QRM sample-efficient.

    Parameters
    ----------
    rm       : the reward machine for this task
    n_states : number of discrete environment states (0-indexed)
    n_actions: number of discrete actions (0-indexed)
    alpha    : learning rate
    gamma    : discount factor
    epsilon  : ε-greedy exploration probability
    """

    def __init__(
        self,
        rm: RewardMachine,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.5,
    ) -> None:
        self.rm = rm
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q[u] is a (n_states × n_actions) table for RM state u
        self.Q: dict[int, np.ndarray] = {u: np.zeros((n_states, n_actions)) for u in rm.U}
        self.Q[rm.terminal_u] = np.zeros((n_states, n_actions))  # never trained; avoids KeyErrors

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def select_action(self, s: int, u: int) -> int:
        """ε-greedy action selection conditioned on the current RM state."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[u][s]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        s: int,
        a: int,
        props: list[str],
        s2: int,
        env_done: bool = False,
        s_info: dict = None
    ) -> float:
        """
        Off-policy QRM update over all RM states.

        For every RM state u1 we compute the hypothetical transition
        (u1 → u2) under `true_props`, compute the TD target, and update
        Q[u1][s, a].  This reuses a single environment transition to
        train every sub-policy simultaneously.

        Returns the reward actually received by the *active* RM state.
        """

        for u1 in self.rm.U:
            u2, r, rm_done = self.rm.step(u1, props, env_done=env_done, s_info=s_info)
            # TD target: bootstrap from u2 unless terminal
            if u2 != self.rm.terminal_u and not env_done:
                td_target = r + self.gamma * float(np.max(self.Q[u2][s2]))
            else:
                td_target = r

            self.Q[u1][s, a] += self.alpha * (td_target - self.Q[u1][s, a])

        return

    def get_policy(self) -> dict[int, np.ndarray]:
        """
        Extract a greedy deterministic policy from the current Q-tables.

        Returns
        -------
        policy : dict mapping each RM state u to a 1-D int array of shape
                 (n_states,), where policy[u][s] is the greedy action in
                 environment state s when the RM is in state u.

        The result is also stored as self.policy for later inspection.
        """
        self.policy: dict[int, np.ndarray] = {
            u: np.argmax(q_table, axis=1).astype(np.int32)
            for u, q_table in self.Q.items()
            if u != self.rm.terminal_u  # terminal state has no meaningful policy
        }
        return self.policy

# ---------------------------------------------------------------------------
# MultiTaskQRMTrainer
# ---------------------------------------------------------------------------

class MultiTaskQRMTrainer:
    def __init__(
        self,
        agents: list[QRMAgent],
        env_factories: list,          # NEW: one () -> Env per task
        max_steps_per_episode: int = 50,
    ) -> None:
        if not agents:
            raise ValueError("Need at least one agent.")
        if len(agents) != len(env_factories):
            raise ValueError("One env_factory per agent required.")
        self.agents = agents
        self.env_factories = env_factories
        self.max_steps_per_episode = max_steps_per_episode
        self._current_task: int = -1
        self.total_steps: int = 0
        self.episode_count: int = 0
        self.reward_history: list[list[tuple[int, float]]] = [[] for _ in agents]

    def run_episode(self, shared_env) -> tuple[int, float]:
        task_id = (self._current_task + 1) % len(self.agents)
        self._current_task = task_id
        active_agent = self.agents[task_id]

        # Active env: the single-RM env for this task
        env = self.env_factories[task_id]()
        s, _ = env.reset()
        episode_reward = 0.0

        for _ in range(self.max_steps_per_episode):
            u = env.get_rm_states()[0]         # always index 0 in single-RM env
            a = active_agent.select_action(s, u)

            new_s, reward, terminated, truncated, info = env.step(a)
            props = info.get("props", [])      # always a list now
            env_done = info.get("env_done", False)

            episode_reward += float(np.asarray(reward).flat[0])

            # Simultaneous update: all agents learn from the same (s,a,props,s2)
            # Props come from the active env — they are environment propositions,
            # not RM-specific, so they are valid for every agent's RM
            for agent in self.agents:
                agent.update(s, a, props, new_s, env_done=env_done, s_info=info)

            self.total_steps += 1
            s = new_s

            if terminated or truncated:
                break

        env.close()
        self.episode_count += 1
        self.reward_history[task_id].append((self.episode_count, episode_reward))
        return task_id, episode_reward

    def train(self, n_episodes: int, print_every: int = 0) -> None:
        for ep in range(n_episodes):
            task_id, ep_reward = self.run_episode(None)
            if print_every > 0 and (ep + 1) % print_every == 0:
                print(
                    f"  ep={ep+1:>4}  task={task_id}  "
                    f"reward={ep_reward:.2f}  "
                    f"total_steps={self.total_steps}"
                )

    def eval_agent(
            self,
            task_id: int,
            env_factory,  # ← NEW: callable () -> Env, not a pre-built env
            n_eval_episodes: int = 10,
    ) -> float:
        """
        Evaluate a single agent greedily (ε=0) over several episodes.

        Parameters
        ----------
        task_id     : index of the task / agent to evaluate
        env_factory : zero-argument callable that returns a *fresh* single-RM
                      environment wrapping only this task's reward machine.
                      Example:
                          lambda: PBSTEnv(reward_sources=[rm_treasure])
                      Using a single-RM env guarantees:
                        • reward is a scalar (not a 3-vector)
                        • env.get_rm_states()[0] is always the right RM state
        """
        agent = self.agents[task_id]
        saved_eps = agent.epsilon
        agent.epsilon = 0.0

        total = 0.0
        max = -1000
        for _ in range(n_eval_episodes):
            env = env_factory()  # fresh env, RM reset to u0
            s, _ = env.reset()
            ep_r = 0.0
            for _ in range(self.max_steps_per_episode):
                u = env.get_rm_states()[0]  # always index 0 in a single-RM env
                a = agent.select_action(s, u)
                s, r, terminated, truncated, info = env.step(a)
                ep_r += float(np.asarray(r).flat[0])  # scalar, safe for any shape
                if terminated or truncated:
                    break
            total += ep_r
            if ep_r > max:
                max = ep_r
            env.close()

        agent.epsilon = saved_eps
        return (total / n_eval_episodes), max


# ---------------------------------------------------------------------------
# Example Reward Machines (Office / Grid World)
# ---------------------------------------------------------------------------

def rm_get_coffee_wo_deco() -> RewardMachine:
    """
    Task A — Coffee without decoration.

    Pick up coffee (prop: ``coffee``), then reach the office (``office``).
    Touching the decoration (``decoration``) at any point fails the task.
    Reward of 1 on reaching the office with coffee.
    """
    rm = RewardMachine()
    rm.set_initial_state(0)
    rm.add_transition(0, 0,             "!coffee&!decoration",  ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration",           ConstantRewardFunction(0))
    rm.add_transition(0, 1,             "coffee&!decoration",   ConstantRewardFunction(0))
    rm.add_transition(1, 1,             "!office&!decoration",  ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "decoration",           ConstantRewardFunction(0))
    rm.add_transition(1, 2,             "office&!decoration",   ConstantRewardFunction(1))
    rm.add_transition(2, rm.terminal_u, "True",                 ConstantRewardFunction(0))
    rm.finalize()
    return rm


def rm_patrol_ab() -> RewardMachine:
    """
    Task B — Patrol A then B.

    Visit location A (prop: ``a``), then location B (``b``).
    Reward of 1 when reaching B after A.
    """
    rm = RewardMachine()
    rm.set_initial_state(0)
    rm.add_transition(0, 0, "!a", ConstantRewardFunction(0))
    rm.add_transition(0, 1, "a",  ConstantRewardFunction(0))
    rm.add_transition(1, 1, "!b", ConstantRewardFunction(0))
    rm.add_transition(1, 2, "b",  ConstantRewardFunction(1))
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    return rm


def rm_avoid_hazard() -> RewardMachine:
    """
    Task C — Reach goal, avoid hazard.

    Reach the goal (prop: ``goal``) for reward +1; touching a hazard
    (``hazard``) yields reward -1 and ends the episode.
    """
    rm = RewardMachine()
    rm.set_initial_state(0)
    rm.add_transition(0, 0,             "!goal&!hazard", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "hazard",        ConstantRewardFunction(-1))
    rm.add_transition(0, 1,             "goal&!hazard",  ConstantRewardFunction(1))
    rm.add_transition(1, rm.terminal_u, "True",          ConstantRewardFunction(0))
    rm.finalize()
    return rm


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _make_toy_env(n_states: int = 16, n_actions: int = 4, seed: int = 0):
    """
    Minimal toy environment for testing.

    The grid has ``n_states`` cells (linear).  Actions move left/right/stay
    or wrap.  Special cells emit propositions used by the three tasks:

        cell 4  → "coffee"
        cell 8  → "office"  (also "a" for patrol task)
        cell 12 → "b"       (patrol task)
        cell 15 → "goal"
        cell 2  → "hazard"
        cell 6  → "decoration"

    The environment never terminates on its own (env_done always False).
    """
    rng = random.Random(seed)

    PROP_MAP: dict[int, list[str]] = {
        4:  ["coffee"],
        8:  ["office", "a"],
        12: ["b"],
        15: ["goal"],
        2:  ["hazard"],
        6:  ["decoration"],
    }

    def reset() -> int:
        return rng.randint(0, n_states - 1)

    def step(s: int, a: int) -> tuple[int, bool]:
        # 0=left, 1=right, 2=stay, 3=random
        if a == 0:
            s2 = max(0, s - 1)
        elif a == 1:
            s2 = min(n_states - 1, s + 1)
        elif a == 2:
            s2 = s
        else:
            s2 = rng.randint(0, n_states - 1)
        return s2, False  # env never terminates by itself

    def get_props(s: int, a: int, s2: int) -> list[str]:
        return PROP_MAP.get(s2, [])

    return reset, step, get_props


if __name__ == "__main__":
    print("=" * 60)
    print("QRM Multi-Task Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build the three reward machines
    # ------------------------------------------------------------------
    rm_coffee  = rm_get_coffee_wo_deco()
    rm_patrol  = rm_patrol_ab()
    rm_hazard  = rm_avoid_hazard()

    """
    print("\n── Reward Machines ──────────────────────────────────────────")
    for name, rm in [("Coffee", rm_coffee), ("Patrol", rm_patrol), ("Hazard", rm_hazard)]:
        print(f"\n[{name}]  states={rm.U}  u0={rm.u0}")
        rm.pretty_print()
    """

    # ------------------------------------------------------------------
    # 2. Build one QRMAgent per task
    # ------------------------------------------------------------------
    N_STATES  = 16
    N_ACTIONS = 4

    agent_coffee = QRMAgent(rm_coffee, N_STATES, N_ACTIONS, alpha=0.1, gamma=0.9, epsilon=0.8)
    agent_patrol = QRMAgent(rm_patrol, N_STATES, N_ACTIONS, alpha=0.1, gamma=0.9, epsilon=0.8)
    agent_hazard = QRMAgent(rm_hazard, N_STATES, N_ACTIONS, alpha=0.1, gamma=0.9, epsilon=0.8)

    agents = [agent_coffee, agent_patrol, agent_hazard]
    task_names = ["Coffee", "Patrol", "Hazard"]

    # ------------------------------------------------------------------
    # 3. Build the multi-task trainer
    # ------------------------------------------------------------------
    trainer = MultiTaskQRMTrainer(agents, max_steps_per_episode=100)

    # ------------------------------------------------------------------
    # 4. Toy environment
    # ------------------------------------------------------------------
    env_reset, env_step, get_props = _make_toy_env(N_STATES, N_ACTIONS, seed=42)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    N_EPISODES = 1_000
    PRINT_EVERY = 50

    print(f"\n── Training  ({N_EPISODES} episodes, round-robin across {len(agents)} tasks) ──")
    trainer.train(N_EPISODES, env_reset, env_step, get_props, print_every=PRINT_EVERY)

    # ------------------------------------------------------------------
    # 6. Per-task reward curves (last-k average)
    # ------------------------------------------------------------------
    print("\n── Per-task reward (last 20 training episodes each) ─────────")
    for i, name in enumerate(task_names):
        history = trainer.reward_history[i]
        last = [r for _, r in history[-20:]]
        if last:
            avg = sum(last) / len(last)
            print(f"  [{name:7s}]  episodes_played={len(history):>3}  "
                  f"avg_last20={avg:+.3f}  "
                  f"max={max(last):+.3f}")
        else:
            print(f"  [{name:7s}]  no data")

    # ------------------------------------------------------------------
    # 7. Greedy evaluation
    # ------------------------------------------------------------------
    print("\n── Greedy evaluation (10 episodes each, ε=0) ────────────────")
    for i, name in enumerate(task_names):
        mean_r = trainer.eval_agent(i, env_reset, env_step, get_props, n_eval_episodes=10)
        print(f"  [{name:7s}]  mean_return={mean_r:+.3f}")

    # ------------------------------------------------------------------
    # 8. Simultaneous-update sanity check
    # ------------------------------------------------------------------
    print("\n── Simultaneous-update sanity check ─────────────────────────")
    print("  Verifying that a single transition updates Q-tables of ALL agents.")

    # Snapshot Q sums before
    def _q_sum(agent: QRMAgent) -> float:
        return sum(float(np.sum(q)) for q in agent.Q.values())

    before = [_q_sum(a) for a in agents]

    # Force one known transition
    for a in agents:
        a.reset()
    s_test, a_test = 7, 1         # move right: 7 → 8 (emits "office", "a")
    props_test = get_props(s_test, a_test, 8)
    for agent in agents:
        agent.update(s_test, a_test, props_test, 8, env_done=False)

    after = [_q_sum(a) for a in agents]

    for i, name in enumerate(task_names):
        changed = "CHANGED ✓" if abs(after[i] - before[i]) > 1e-12 else "unchanged ✗"
        print(f"  [{name:7s}]  ΔQ_sum={after[i]-before[i]:+.6f}  → {changed}")
