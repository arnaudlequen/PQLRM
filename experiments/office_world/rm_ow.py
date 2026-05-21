import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.reward_machines.reward_machine import RewardMachine,ConstantRewardFunction

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

def build_ow_rm_time(time_penalty: float = 1.0) -> RewardMachine:
    """
    RM_time for PBST: emits -time_penalty on every step, never terminates.

    Propositions used: none (constant reward regardless of props).

    States
    ------
    0  (initial, non-terminal, self-loop)
    """
    return _build_constant_rm(-time_penalty)

def rm_get_coffee_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!coffee&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "coffee&!decoration", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!office&!decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(1, 2, "office&!decoration", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    #rm.pretty_print()
    return rm

def rm_get_coffee():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!coffee", ConstantRewardFunction(0))
    rm.add_transition(0, 1, "coffee", ConstantRewardFunction(1)) # using 1 increases the number of trade-off policies found
    rm.add_transition(1, 1, "!office", ConstantRewardFunction(0))
    rm.add_transition(1, 2, "office", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    rm.pretty_print()
    return rm

def rm_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!office&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, 1, "office", ConstantRewardFunction(1))
    rm.add_transition(0, 2, "decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    rm.pretty_print()
    return rm

def rm_get_mail_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!mail&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "mail&!decoration", ConstantRewardFunction(1)) # using 1 increases the number of trade-off policies found
    rm.add_transition(1, 1, "!office&!decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(1, 2, "office&!decoration", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def rm_get_mail():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!mail", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "mail", ConstantRewardFunction(1)) 
    rm.add_transition(1, 1, "!office", ConstantRewardFunction(0)) 
    rm.add_transition(1, 2, "office", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def rm_patrol():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!A", ConstantRewardFunction(0))
    rm.add_transition(0, 1, "A", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!B", ConstantRewardFunction(0)) 
    rm.add_transition(1, 2, "B", ConstantRewardFunction(0))  
    rm.add_transition(2, 2, "!C", ConstantRewardFunction(0)) 
    rm.add_transition(2, 3, "C", ConstantRewardFunction(0))
    rm.add_transition(3, 3, "!D", ConstantRewardFunction(0))   
    rm.add_transition(3, 4, "D", ConstantRewardFunction(0))
    rm.add_transition(4, 4, "!office", ConstantRewardFunction(0))
    rm.add_transition(4, 5, "office", ConstantRewardFunction(1)) # necessary to end the episode
    rm.add_transition(5, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def rm_patrol_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!A&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "A&!decoration", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!B&!decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(1, 2, "B&!decoration", ConstantRewardFunction(0))  
    rm.add_transition(2, 2, "!C&!decoration", ConstantRewardFunction(0))
    rm.add_transition(2, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(2, 3, "C&!decoration", ConstantRewardFunction(0))
    rm.add_transition(3, rm.terminal_u, "decoration", ConstantRewardFunction(0))
    rm.add_transition(3, 3, "!D&!decoration", ConstantRewardFunction(0))   
    rm.add_transition(3, 4, "D&!decoration", ConstantRewardFunction(0))
    rm.add_transition(4, 5, "office", ConstantRewardFunction(1)) # necessary to end the episode
    rm.add_transition(4, rm.terminal_u, "decoration", ConstantRewardFunction(0))
    rm.add_transition(5, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

