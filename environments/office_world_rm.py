from contextlib import closing
from io import StringIO
from os import path
from typing import Any, Sequence, Union
from collections.abc import Callable

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reward_machines.reward_machine import RewardMachine
import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled

# Import shared map constants
from environments.office_world_maps import UP, RIGHT, DOWN, LEFT, POSITION_MAPPING, MAPS

def read_decorators(map, shape, decorator: str):
    loc = np.zeros(shape, dtype=bool)
    for x in range(shape[0]):
        for y in range(shape[1]):
            if MAPS[map][x][y] == decorator:
                loc[x][y] = True
    return loc

def time_penalty(env: Env, current_state: int, new_state: int, new_position: tuple[int, int]) -> float:
    return -1

def go_to_office(env: Env, current_state: int, new_state: int, new_position: tuple[int, int]) -> float:
    if env.map[new_position[0]][new_position[1]] == "s":
        return 1
    else:
        return 0

class OfficeWorldRM(Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: str | None = "ansi",
                 map: str = "default_office",
                 reward_sources: Callable | RewardMachine | list[Callable | RewardMachine] | None = None,
                 policy = None): # TODO replace Callable by RewardFunction

        self.shape = (11, 15)
        self.map = MAPS[map]

        self.start_position_index = 137  # just out of the office : 50
        self._coffee_found = False
        self._mail_found = False

        # Reward
        if reward_sources is None:
            raise ValueError("No reward defined")
        elif isinstance(reward_sources, (list, tuple)): #TODO test this condition
            self.reward_sources = list(reward_sources)
        else:
            self.reward_sources = [reward_sources]
        
        # Track RM internal states
        self._rm_states: list[int| None ] = []

        for src in reward_sources:
            if hasattr(src, "reset"):
                self._rm_states.append(src.reset())
            else:
                self._rm_states.append(None)

        # nb RM states
        self.totalRMStates = 1
        for src in reward_sources:
            if hasattr(src, "step"):
                self.totalRMStates *= len(src.get_states())

        # Already learnt policy
        self.policy = policy

        # Wall location
        self._wall = read_decorators(map, self.shape, "W")

        # Decoration location
        self._decoration = read_decorators(map, self.shape, "X")
        self.hit_decoration = False

        # Coffee location
        self._coffee = read_decorators(map, self.shape, "c")

        # All other decorators appear only a single time on the map

        self.desc = np.asarray(self.map, dtype="c")
        self.base_nS = int(np.prod(self.shape))
        self.nS = self.base_nS * 2 * 2 * self.totalRMStates # for coffee and mail and RMs states
        self.nA = 4
        

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.render_mode = render_mode

        self.start_state_index = self.encode_state(self.start_position_index,
                                                   coffee_found=False,
                                                   mail_found=False)
        self.s = self.start_state_index

        # pygame utils -> TODO

    def encode_state(self, position_state: int, coffee_found: bool, mail_found: bool) -> int:
        index_state = position_state + self.base_nS * int(coffee_found) + self.base_nS * 2 * int(mail_found)
        c = self.base_nS * 2 * 2
        for i in range(len(self.reward_sources)):
            if self._rm_states[i] != None:
                index_state += c * self._rm_states[i]
                c *= len(self.reward_sources[i].get_states())
        return index_state

    def decode_state(self, state: int) -> tuple[int, bool]:
        remaining = int(state)
        position = remaining % self.base_nS
        remaining //= self.base_nS
        coffee_found = bool(remaining % 2)
        remaining //= 2
        mail_found = bool(remaining % 2)
        return {'position': position,
                'coffee_found': coffee_found,
                'mail_found': mail_found}


    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def step(self, a):
        # ----- 1. Environment dynamics -----
        full_state = self.decode_state(self.s)
        current_pos = np.unravel_index(full_state['position'], self.shape)

        delta = POSITION_MAPPING[a] # does not take into account slippery surface

        new_position = np.array(current_pos) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        if self.is_wall(new_position):
            new_position = np.array(current_pos)
        #print('current_pos:', current_pos, "new_position:", new_position)
        new_position_state = int(np.ravel_multi_index(tuple(new_position), self.shape))
        next_coffee_found = full_state['coffee_found'] or self.is_coffee(new_position)
        next_mail_found = full_state['mail_found'] or self.is_mail(new_position)

        env_done = self.is_office(new_position) #False #

        # ----- 2. Reward machines dynamics -----
        rewards, new_configuration, rm_done = self._evaluate_rewards(self.s, self._rm_states, a)
        self._rm_states = new_configuration.copy()
        #print('env_done:', env_done)
        #print('rm_done:', rm_done)

        # ----- 3. Termination -----
        terminated = env_done# or rm_done
        truncated = False

        # ----- 4. State action update -----
        self._coffee_found = next_coffee_found
        self._mail_found = next_mail_found
        new_state = self.encode_state(new_position_state, next_coffee_found, next_mail_found)
        self.s = new_state
        self.lastaction = a

        if self.render_mode == "human":
            self.render()

        return int(new_state), rewards, terminated, truncated, {"prob": 1.0}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # reset RMs
        self._rm_states = []
        for src in self.reward_sources:
            if hasattr(src, "reset"):
                self._rm_states.append(src.reset())
            else:
                self._rm_states.append(None)

        self.s = self.start_state_index #categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.hit_decoration = False

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def set_state(self, state, info=None):
        self.s = state
        return

    def get_rm_states(self):
        return self._rm_states
    
    def get_reward_sources(self):
        return self.reward_sources

    def is_wall(self, position):
        return self._wall[tuple(position)]
    
    #def is_decoration(self, position):
    #    if self._decoration[tuple(position)] and not self.hit_decoration:
    #        self.hit_decoration = True
        
    #    return self.hit_decoration
    
    def is_decoration(self, position):
        return self._decoration[tuple(position)]

    def is_coffee(self, position):
        return self._coffee[tuple(position)]
    
    def is_office(self, position):
        return self.map[position[0]][position[1]] == "s"
    
    def is_mail(self, position):
        return self.map[position[0]][position[1]] == "m"
    
    def is_A(self, position):
        return self.map[position[0]][position[1]] == "A"
    
    def is_B(self, position):
        return self.map[position[0]][position[1]] == "B"
    
    def is_C(self, position):
        return self.map[position[0]][position[1]] == "C"
    
    def is_D(self, position):
        return self.map[position[0]][position[1]] == "D"
    

    def _get_true_props(self, next_position, current_state, action=None):
        props = []
        policy_props = []
        allchecks = [self.is_wall, self.is_decoration, self.is_coffee, 
                     self.is_office, self.is_mail,
                     self.is_A, self.is_B, self.is_C, self.is_D]
        
        for f in allchecks:
            p = str(f).split("_")[1].split(" ")[0]
            if f(next_position):
                props.append(p)
            else:
                props.append("!"+p)

        # if an RM is dedicated to the policy previously learnt
        if self.policy is not None:
            agent_action = self.policy.predict(current_state)
            #print(f"Check action taken : {agent_action, action, current_state}")
            if agent_action == action:
                policy_props.append("policy")
            else:
                policy_props.append("!policy")

        #print("props: ", props)
        return props, policy_props

    def _evaluate_rewards(self, current_state: int, current_configuration: list, action = None):
        rewards = []
        next_configuration = current_configuration.copy()
        done_flags = []
        full_state = self.decode_state(self.s)
        current_pos = np.unravel_index(full_state['position'], self.shape)
        delta = POSITION_MAPPING[action]
        new_position = np.array(current_pos) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = int(np.ravel_multi_index(tuple(new_position), self.shape))

        #print('rm_states before update:', self._rm_states)

        # Compute true propositions of :
        # - the state we are after following action
        # - the policy
        props, policy_props = self._get_true_props(new_position, current_state, action)

        for i, src in enumerate(self.reward_sources):
            # --- Reward Machine ---
            if hasattr(src, "step"):
                u1 = current_configuration[i] #self._rm_states[i]
                # By convention, the RM related to self.policy is always the first one in reward_sources
                true_props = policy_props if (self.policy is not None and i == 0) else props
                u2, r, rm_done = src.step(u1,true_props,
                    s_info={
                        "state": current_pos,
                        "position": current_pos,
                    },
                    env_done="office" in props, #self.is_office(new_position),
                ) # TODO: s_info for RewardFunction related to an RM state. To be implemented !

                #self._rm_states[i] = u2
                #current_configuration[i] = u2 # !!! check side effect
                next_configuration[i] = u2
                rewards.append(r)
                done_flags.append(rm_done)

            # --- Callable reward function ---
            else:
                r = src(self, current_state, new_state, new_position) # TODO: replace by a RewardFunction src.get_reward(s_info)
                rewards.append(r)
                done_flags.append(False)

        # Aggregate rewards and dones
        reward = rewards[0] if len(rewards) == 1 else np.asarray(rewards, dtype=float)
        #print('rm_states after update:', self._rm_states)
        #print('done_flags:', done_flags)
        rm_done = all(done_flags) # TODO : any ?

        return reward, next_configuration, rm_done