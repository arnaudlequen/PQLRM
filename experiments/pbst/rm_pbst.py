from __future__ import annotations

import random
from typing import Any
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from environments.reward_machines.reward_machine import RewardMachine
from environments.reward_machines.reward_functions import ConstantRewardFunction,RewardFunction

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
            pos = tuple(s_info.get("position_xy", (-1, -1)))
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

    class PressureRewardFunction(RewardFunction):  # imported RewardFunction
        def get_reward(self, s_info):
            if s_info is None:
                return 0.0
            pos = s_info.get("position_xy", None)
            if pos is None:
                return 0.0
            return -float(pos[0])

    rm = RewardMachine()
    rm.set_initial_state(0)
    rm.add_transition(0, 0, "!goal", ConstantRewardFunction(0.0))  # imported
    rm.add_transition(0, 1, "goal", PressureRewardFunction())
    rm.add_transition(1, 1, "True", ConstantRewardFunction(0.0))   # imported
    rm.finalize()
    return rm

def build_pbst_rm_pressure_v2() -> RewardMachine:
    """
    RM_pressure_v2: penalty grows with consecutive DOWN actions.

    Streak states:
        0 — no consecutive downs (or just reset)
        1 — 1 consecutive down  → reward -1
        2 — 2 consecutive downs → reward -3 (cumulative: -4)
        3 — 3 consecutive downs → reward -5 (cumulative: -9)
        4 — 4+ consecutive downs → reward -7 (cumulative: -16, -23, ...)

    Any non-down action resets streak to state 0 (reward 0).

    Propositions used: "down" / "!down", "goal" / "!goal"
    """

    class StreakRewardFunction(RewardFunction):
        def __init__(self, penalty: float):
            self.penalty = penalty
        def get_reward(self, s_info):
            return self.penalty

    rm = RewardMachine()
    rm.set_initial_state(0)

    # ── Non-down action from any streak state: reset to 0, no reward ──
    for u in range(4):
        rm.add_transition(u, 0, "!down&!goal", ConstantRewardFunction(0.0))

    # ── Down actions: advance streak, emit penalty ──
    # streak 0 → 1: first down, penalty -1
    rm.add_transition(0, 1, "down&!goal", StreakRewardFunction(-1.0))
    # streak 1 → 2: second consecutive down, penalty -3
    rm.add_transition(1, 2, "down&!goal", StreakRewardFunction(-3.0))
    # streak 2 → 3: third consecutive down, penalty -5
    rm.add_transition(2, 3, "down&!goal", StreakRewardFunction(-5.0))
    # streak 3 → 3: fourth+ consecutive down, penalty -7 (self-loop)
    rm.add_transition(3, 3, "down&!goal", StreakRewardFunction(-7.0))

    # ── Goal reached from any streak state: terminal, no extra reward ──
    for u in range(4):
        rm.add_transition(u, 4, "goal", ConstantRewardFunction(50.0))
    
    rm.add_transition(4, rm.terminal_u, "True", ConstantRewardFunction(0))

    rm.finalize()
    return rm

def build_pbst_rm_pressure_v3() -> RewardMachine:
    class StreakRewardFunction(RewardFunction):
        def __init__(self, penalty: float):
            self.penalty = penalty
        def get_reward(self, s_info):
            return self.penalty

    rm = RewardMachine()
    rm.set_initial_state(0)

    # ── Non-down: decrease streak by 1, floor at 0 ──
    rm.add_transition(0, 0, "!down&!goal", ConstantRewardFunction(0.0))
    rm.add_transition(1, 0, "!down&!goal", ConstantRewardFunction(0.0))
    rm.add_transition(2, 1, "!down&!goal", ConstantRewardFunction(0.0))
    rm.add_transition(3, 2, "!down&!goal", ConstantRewardFunction(0.0))

    # ── Down: advance streak, emit growing penalty ──
    rm.add_transition(0, 1, "down&!goal", StreakRewardFunction(-1.0))
    rm.add_transition(1, 2, "down&!goal", StreakRewardFunction(-3.0))
    rm.add_transition(2, 3, "down&!goal", StreakRewardFunction(-5.0))
    rm.add_transition(3, 3, "down&!goal", StreakRewardFunction(-7.0))

    # ── Goal from any state: terminal ──
    for u in range(4):
        rm.add_transition(u, rm.terminal_u, "goal", ConstantRewardFunction(0.0))

    rm.finalize()
    return rm
