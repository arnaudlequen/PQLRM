import gymnasium
import numpy as np
import pygame
from gymnasium import Env, spaces
from collections.abc import Callable

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reward_machines.reward_machine import RewardMachine



# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {
    UP: [-1, 0],
    RIGHT: [0, 1],
    DOWN: [1, 0],
    LEFT: [0, -1],
}

class DiscreteObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Discrete(np.prod(env.shape))

    #def observation(self, obs):
        # example: (1, 1) in a grid of (10, 11) -> 1 * 11 + 1 = 12
    #    return int(np.ravel_multi_index(obs, self.env.unwrapped.shape))

    def _flatten_state(self, s):
        if isinstance(s, (int, np.integer)): return int(s)
        return int(np.ravel_multi_index(s, self.env.unwrapped.shape))

    def _unflatten_state(self, state_index):
        """
        Convertit un index scalaire -> [row, col]
        """
        # Si c'est déjà un tableau ou une liste, on le renvoie tel quel
        if not isinstance(state_index, (int, np.integer)):
            return state_index

        # unravel_index renvoie un tuple (row, col)
        # On utilise la forme de la grille de l'environnement
        coords = np.unravel_index(state_index, self.env.unwrapped.shape)

        return np.array(coords, dtype=int)

    def reset(self, **kwargs):
        s, info = self.env.reset(**kwargs)
        return self._flatten_state(s), info

    def step(self, action):
        s, r, term, trunc, info = self.env.step(action)
        info["state"] = self._unflatten_state(s)
        info["position"] = tuple(self._unflatten_state(s))
        return self._flatten_state(s), r, term, trunc, info

    def __getattr__(self, name):
        """Redirect method calls."""
        return getattr(self.env, name)

class PBSTEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None,
                 reward_sources: Callable | RewardMachine | list[Callable | RewardMachine] | None = None):
        super().__init__()

        # Grid
        self.shape = (10, 11)  # (rows, cols)
        self.start_state = (0, 0)
        self.start_state_index = 0

        # Action/Observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.shape[0] - 1, self.shape[1] - 1]),
            dtype=np.int32,
        )

        # Treasure map (PBST-style increasing rewards)
        self._treasure = {
            (1, 0): 5,
            (2, 1): 80,
            (3, 2): 120,
            (4, 3): 140,
            (4, 4): 145,
            (4, 5): 150,
            (7, 6): 163,
            (7, 7): 166,
            (9, 8): 173,
            (10, 9): 175,
        }

        self.state = self.start_state_index

        # --- Reward sources (RM or callable) ---
        if reward_sources is None:
            self.reward_sources = [self.reward_function]
        elif isinstance(reward_sources, (list, tuple)):
            self.reward_sources = list(reward_sources)
        else:
            self.reward_sources = [reward_sources]

        # Track RM states
        self._rm_states = []

        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 50
        self.window_size = (
            self.shape[1] * self.cell_size,
            self.shape[0] * self.cell_size,
        )

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    # -------------------------
    # Transition function
    # -------------------------
    def transition_function(self, state:int, action) -> int:
        """Deterministic transition"""
        current_pos = np.unravel_index(state, self.shape)

        delta = POSITION_MAPPING[action]

        new_position = np.array(current_pos) + np.array(delta)
        #print(state, current_pos, delta, new_position)
        # Clip to grid
        new_position = self._limit_coordinates(new_position).astype(int)

        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        #print(f"state : {state}, position : {current_pos}, action : {action}, new position : {new_position}, new_state : {new_state}")
        return new_state

    # -------------------------
    # Reward function (vector)
    # -------------------------
    def reward_function(self, state, action, new_state):
        """
        Returns a 3D reward vector:
        [time, treasure, pressure]
        """
        next_position = np.unravel_index(new_state, self.shape)

        # Time penalty
        r_time = -1

        # Pressure penalty (depth = row index)
        depth = next_position[0]
        r_pressure = -depth

        # Treasure reward
        #pos = tuple(next_state)
        r_treasure = self._treasure.get(next_position, 0)

        return np.array([r_time, r_treasure, r_pressure], dtype=np.float32)
        #return np.array([r_time, r_treasure], dtype=np.float32)

    # -------------------------
    # Step
    # -------------------------
    def step(self, a):
        # ----- 1. Environment dynamics -----
        current_state = self.state
        new_state = self.transition_function(current_state, a)
        env_done = self.is_treasure(new_state)

        # ----- 2. Reward machines dynamics -----
        rewards, new_configuration, rm_done, true_props = self._evaluate_rewards(current_state, self._rm_states, new_state, a)
        self._rm_states = new_configuration.copy()

        # ----- 3. Termination -----
        terminated = env_done #or rm_done
        truncated = False

        # ----- 4. State update -----
        self.state = new_state

        if self.render_mode == "human":
            self.render()


        return new_state, rewards, terminated, truncated, {"prob": 1.0, "props": true_props, "env_done": env_done}

    # -------------------------
    # Reset
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state_index

        # Reset RM states
        self._rm_states = []
        for src in self.reward_sources:
            if hasattr(src, "reset"):
                self._rm_states.append(src.reset())
            else:
                self._rm_states.append(None)

        if self.render_mode == "human":
            self.render()

        return self.state, {"prob": 1.0}

    # -------------------------
    # Render
    # -------------------------
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("PBST Environment")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 105, 148))  # ocean blue

        # Draw grid + treasures
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

                if (i, j) in self._treasure:
                    pygame.draw.circle(
                        canvas,
                        (255, 215, 0),  # gold
                        rect.center,
                        self.cell_size // 4,
                    )

                    font = pygame.font.SysFont(None, 20)
                    text = font.render(
                        str(self._treasure[(i, j)]), True, (255, 255, 255)
                    )
                    canvas.blit(text, (rect.x + 5, rect.y + 5))

        # Draw agent (submarine)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (
                self.state[1] * self.cell_size + self.cell_size // 2,
                self.state[0] * self.cell_size + self.cell_size // 2,
            ),
            self.cell_size // 3,
        )

        # Display
        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    # -------------------------
    # Close
    # -------------------------
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

    def _get_true_props(self, state, is_treasure):
        props = []
        position = np.unravel_index(state, self.shape)
        # Treasure proposition
        if is_treasure:
            props.append("goal")
        else:
            props.append("!goal")

        # Depth proposition (pressure-related)
        if position[0] >= self.shape[0] // 2:
            props.append("deep")
        else:
            props.append("!deep")

        return props

    def _evaluate_rewards(self, current_state, current_configuration, new_state, action):
        rewards = []
        next_configuration = current_configuration.copy()
        done_flags = []
        is_rm = False
        is_treasure = self.is_treasure(new_state)

        # Compute true propositions
        props = self._get_true_props(new_state, is_treasure)
        
        for i, src in enumerate(self.reward_sources):
            # --- Reward Machine ---
            if hasattr(src, "step"):
                u1 = current_configuration[i]
                is_rm = True
                u2, r, rm_done = src.step(
                    u1,
                    props,
                    s_info={
                        "state": new_state,
                        "position": np.unravel_index(new_state, self.shape), # is it used ?
                    },
                    env_done=is_treasure,
                )

                next_configuration[i] = u2
                rewards.append(r)
                done_flags.append(rm_done)

            # --- Callable reward ---
            else:
                if len(self.reward_sources) == 1:
                    rewards = src(current_state, action, new_state)
                else:
                    r = src(current_state, action, new_state)
                    rewards.append(r)
                done_flags.append(False)

        # Aggregate rewards and dones
        reward = rewards[0] if len(rewards) == 1 else np.asarray(rewards, dtype=float)
        rm_done = any(done_flags)
        return reward, next_configuration, rm_done, props

    def is_treasure(self, state) -> bool:
        return np.unravel_index(state, self.shape) in self._treasure

    def get_rm_states(self):
        return self._rm_states

    def get_reward_sources(self):
        if self.reward_sources == [self.reward_function]:
            return [self.reward_function, self.reward_function, self.reward_function] #self.reward_function
        else:
            return self.reward_sources

    def get_successor_states(self, s, a):
        new_state = self.transition_function(s, a)
        return [tuple(new_state)]

    def get_successor_rewards(self, rm_configuration, s, a, info=None):
        new_state = self.transition_function(s, a)

        # Rebuild full RM configuration
        extended_configuration = self._rm_states.copy()
        current_rm_idx = 0

        for i in range(len(extended_configuration)):
            if extended_configuration[i] is not None:
                extended_configuration[i] = rm_configuration[current_rm_idx]
                current_rm_idx += 1

        rewards, new_configuration, _, _ = self._evaluate_rewards(
            s,
            extended_configuration,
            new_state,
            a
        )

        return rewards, tuple(new_configuration)

    def set_state(self, state, info=None):
        self.state = state
        return
