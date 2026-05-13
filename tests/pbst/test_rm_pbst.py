from __future__ import annotations

import random
from typing import Any

import numpy as np
from environments.reward_machines.reward_machine import RewardMachine
from environments.reward_machines.reward_functions import ConstantRewardFunction,RewardFunction

# PBST environment
from environments.pbst.pressurizedBountifulSeaTreasure import PBSTEnv

# ============================================================================
#  PBST  Reward Machines
# ============================================================================

def _build_constant_rm(reward_value: float) -> RewardMachine:
    """
    Build a single-state RM that emits `reward_value` on every step and
    never terminates.

    Structure:
        u0 --[True]--> u0   reward = reward_value   (self-loop forever)
    """
    rm = RewardMachine()
    rm.set_initial_state(0)
    rm.add_transition(0, 0, "True", ConstantRewardFunction(reward_value))
    rm.finalize()
    return rm

def build_pbst_rm_time(time_penalty: float = 1.0) -> RewardMachine:
    """
    RM_time for PBST: emits -time_penalty on every step, never terminates.

    Propositions used: none (constant reward regardless of props).

    States
    ------
    0  (initial, non-terminal, self-loop)
    """
    return _build_constant_rm(-time_penalty)


def build_pbst_rm_treasure(treasures: dict[tuple, float]) -> RewardMachine:
    """
    RM_treasure for PBST: emits the treasure value when the agent lands on a
    treasure cell (proposition "goal"), then terminates; emits 0 otherwise.

    Propositions used
    -----------------
    "goal"   — agent is on a treasure cell (emitted by PBSTEnv._get_true_props)
    "!goal"  — complement

    States
    ------
    0  initial / exploring (self-loop on !goal)
    → TERMINAL on "goal" with reward = treasure_value extracted from s_info

    Because the treasure value varies by cell we use a special RewardFunction
    that reads the current position from s_info.
    """

    class TreasureRewardFunction(RewardFunction):
        """Reads the treasure value from s_info['position']."""

        def __init__(self, treasure_map: dict[tuple, float]):
            self.treasure_map = treasure_map

        def get_reward(self, s_info: dict[str, Any] | None) -> float:
            if s_info is None:
                return 0.0
            pos = tuple(s_info.get("position", (-1, -1)))
            return float(self.treasure_map.get(pos, 0.0))

    rm = RewardMachine()
    rm.set_initial_state(0)
    # Stay in u0 while not on a treasure cell
    rm.add_transition(0, 0, "!goal", ConstantRewardFunction(0.0))

    # Reach terminal when landing on a treasure cell → collect reward
    rm.add_transition(0, 1, "goal",
                      TreasureRewardFunction(treasures))
    rm.add_transition(1, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    return rm


def build_pbst_rm_pressure() -> RewardMachine:
    """
    RM_pressure for PBST: emits -depth (row index) every step, where depth
    is read from s_info['state'][0].

    Propositions used: none (reward comes entirely from s_info).

    States
    ------
    0  (initial, non-terminal, self-loop)
    """

    class PressureRewardFunction(RewardFunction):
        """Returns -(row index) from s_info['state']."""

        def get_reward(self, s_info: dict[str, Any] | None) -> float:
            if s_info is None:
                return 0.0
            state = s_info.get("position", None)
            if state is None:
                return 0.0
            depth = int(state[0])
            return -depth

    rm = RewardMachine()
    rm.set_initial_state(0)
    rm.add_transition(0, 0, "True", PressureRewardFunction())
    rm.finalize()
    return rm


# ============================================================================
#  Test
# ============================================================================

def main(n_steps: int = 200, seed: int = 42) -> None:
    """
    Run the PBST environment twice — once with the default callable reward
    function and once with three RewardMachines — and verify that the
    accumulated rewards match.
    """
    print("=" * 70)
    print("PBST ENVIRONMENT — smoke test")
    print("  Replacing [r_time, r_treasure, r_pressure] with 3 RMs")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------ #
    # 1. Reference run: default callable reward function
    # ------------------------------------------------------------------ #
    env_ref = PBSTEnv(render_mode=None)  # uses default reward_function
    obs, _ = env_ref.reset(seed=seed)

    ref_cumulative = np.zeros(3, dtype=np.float64)
    ref_actions: list[int] = []

    done = False
    for _ in range(n_steps):
        a = env_ref.action_space.sample()
        ref_actions.append(a)
        obs, r, terminated, truncated, _ = env_ref.step(a)
        ref_cumulative += np.asarray(r, dtype=np.float64).flatten()[:3]
        done = terminated or truncated
        if done:
            break

    print(f"\n[Reference] steps={len(ref_actions)}, done={done}")
    print(f"  cumulative r_time     = {ref_cumulative[0]:>8.2f}")
    print(f"  cumulative r_treasure = {ref_cumulative[1]:>8.2f}")
    print(f"  cumulative r_pressure = {ref_cumulative[2]:>8.2f}")

    # ------------------------------------------------------------------ #
    # 2. RM run: three separate RewardMachines
    # ------------------------------------------------------------------ #
    rm_time = build_pbst_rm_time(time_penalty=1.0)
    rm_treasure = build_pbst_rm_treasure(env_ref._treasure)
    rm_pressure = build_pbst_rm_pressure()

    print(f"\n[RM_time]     {rm_time}")
    print(f"[RM_treasure] {rm_treasure}")
    print(f"[RM_pressure] {rm_pressure}")

    env_rm = PBSTEnv(
        render_mode=None,
        reward_sources=[rm_time, rm_treasure, rm_pressure],
    )
    obs, _ = env_rm.reset(seed=seed)

    rm_cumulative = np.zeros(3, dtype=np.float64)

    for step_i, a in enumerate(ref_actions):
        obs, r, terminated, truncated, info = env_rm.step(a)
        r_arr = np.asarray(r, dtype=np.float64).flatten()
        rm_cumulative += r_arr[:3]
        if terminated or truncated:
            break

    print(f"\n[RM]        steps={step_i + 1}")
    print(f"  cumulative r_time     = {rm_cumulative[0]:>8.2f}")
    print(f"  cumulative r_treasure = {rm_cumulative[1]:>8.2f}")
    print(f"  cumulative r_pressure = {rm_cumulative[2]:>8.2f}")

    # ------------------------------------------------------------------ #
    # 3. Comparison
    # ------------------------------------------------------------------ #
    match = np.allclose(ref_cumulative, rm_cumulative, atol=1e-6)
    print(f"\n{'✓ PASS' if match else '✗ FAIL'} — RM rewards {'match' if match else 'DIFFER FROM'} reference")
    if not match:
        print(f"  Δ = {rm_cumulative - ref_cumulative}")

    env_ref.close()
    env_rm.close()
    print()

if __name__ == "__main__":
    main()