"""Pareto Q-Learning."""

import numbers
from typing import Callable, List, Optional
import time
import copy
import gymnasium as gym
import numpy as np
from itertools import product
import rich

from baselines.common.evaluation import log_all_multi_policy_metrics
from baselines.common.morl_algorithm import MOAgent
from baselines.common.pareto import get_non_dominated
from baselines.common.performance_indicators import hypervolume
from baselines.common.utils import linearly_decaying_value
from environments.reward_machines.reward_machine import RewardMachine


class PQLRM(MOAgent):
    """Pareto Q-learning.

    Tabular method relying on pareto pruning.
    Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
        self,
        env,
        ref_point: np.ndarray,
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay_steps: int = 100000,
        final_epsilon: float = 0.1,
        seed: Optional[int] = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Pareto Q-Learning with RM",
        output_file: str = None,
        log: bool = True,

    ):
        """Initialize the Pareto Q-learning algorithm.

        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            log: Whether to log or not.
        """
        super().__init__(env, seed=seed)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.ref_point = ref_point

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("PQL only supports (multi)discrete action spaces.")

        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.env_shape = (self.env.observation_space.n,)
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.env_shape = self.env.observation_space.nvec
        elif (
            isinstance(self.env.observation_space, gym.spaces.Box)
            and self.env.observation_space.is_bounded(manner="both")
            and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("PQL only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)
        self.num_objectives = len(self.env.get_reward_sources())
        
        states = [src.get_states() + [-1] for src in self.env.get_reward_sources() if hasattr(src, "step")]
        self.terminal_states = []
        self.num_rms = len(states)
        configurations = product(*states)
        # get all possible configurations from reward machines
        self.rm_configurations = []
        for s in configurations:
            self.rm_configurations.append(s)

        self.counts = np.zeros((self.num_states, self.num_actions))
        self.counts_rmd = np.zeros((len(self.rm_configurations), self.num_states, self.num_actions))
        self.non_dominated = {
             c:[
                [{} for _ in range(self.num_actions)] for _ in range(self.num_states)
            ] for c in self.rm_configurations
        }



        self.avg_reward = {c:np.zeros((self.num_states, self.num_actions, self.num_objectives)) for c in self.rm_configurations}

        # Logging
        
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.output_file = output_file
        

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "env_id": self.env.unwrapped.spec.id,
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def score_pareto_cardinality(self, rm_configuration: tuple, state: int):
        """Compute the action scores based upon the Pareto cardinality metric.

        Args:
            state (int): The current state.
            rm_configuration (tuple) : The current state of each reward machine.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(rm_configuration, state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        if len(candidates) == 0:
            non_dominated = {}
        else:
            non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def score_hypervolume(self, rm_configuration: tuple, state: int):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.
            rm_configuration (tuple) : The current state of each reward machine.

        Returns:
            ndarray: A score per action.
        """
        q_sets = [self.get_q_set(rm_configuration, state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores

    def get_q_set(self, rm_configuration : tuple, state: int, action: int):
        """Compute the Q-set for a given state-action pair.

        Args:
            state (int): The current state.
            rm_configuration (tuple) : The current state of each reward machine.
            action (int): The action.

        Returns:
            A set of Q vectors.
        """
        nd_array = np.array(list(self.non_dominated[rm_configuration][state][action]))

        reward = self.avg_reward[rm_configuration][state, action]
        if len(nd_array) == 0 and (np.zeros(self.num_objectives)-reward == 0).all() and not state in self.terminal_states:
            return {}
        elif len(nd_array) == 0:
            return {tuple(reward)}
        else:
            q_array = reward + self.gamma * nd_array
            return {tuple(vec) for vec in q_array}

    def select_action(self, rm_configuration : tuple, state: int, score_func: Callable):
        """Select an action in the current state.

        Args:
            state (int): The current state.
            rm_configuration (tuple) : The current state of each reward machine.
            score_func (callable): A function that returns a score per action.

        Returns:
            int: The selected action.
        """
        if self.np_random.uniform(0, 1) < self.epsilon:
            action = self.np_random.integers(self.num_actions)
            return action
        else:
            action_scores = score_func(state, rm_configuration)
            action = self.np_random.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())
            return action

    def calc_non_dominated(self, rm_configuration: tuple, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.
            rm_configuration (tuple) : The current state of each reward machine.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        candidates = set().union(*[self.get_q_set(rm_configuration, state, action) for action in range(self.num_actions)])
        
        if len(candidates) == 0:
            return {}
        else:
            non_dominated = get_non_dominated(candidates)
            return non_dominated

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        max_local_steps: Optional[int] = 50,
        optimization: Optional[str] = "Qsets+RI",
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        log_every: Optional[int] = 100,
        action_eval: Optional[str] = "hypervolume",
        convergence_callback: Optional[Callable] = None,
    ):
        """Learn the Pareto front.

        Args:
            total_timesteps (int, optional): The number of episodes to train for.
            max_local_steps: (int): the maximal length of an episode.
            optimization: either Qsets+RI, Qsets, RI, or None. Default is Qsets+RI; None is adviced with large qsets, but slower convergence
            eval_env (gym.Env): The environment to evaluate the policies on.
            eval_ref_point (ndarray, optional): The reference point for the hypervolume metric during evaluation. If none, use the same ref point as training.
            known_pareto_front (List[ndarray], optional): The optimal Pareto front, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            log_every (int, optional): Log the results every number of timesteps. (Default value = 1000)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            Set: The final Pareto front.
        """
        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")
        
        if ref_point is None:
            ref_point = self.ref_point

        callback_env = eval_env if eval_env is not None else self.env
        if convergence_callback is not None and callback_env is self.env:
            # Convergence logging calls track_policy(), which resets the env.
            # Use an isolated copy to avoid mutating the live training episode.
            callback_env = copy.deepcopy(self.env)

        episode_idx = 0
        while self.global_step < total_timesteps:
            episode_idx += 1
            
            
            local_step = 0

            state, _ = self.env.reset()
            #state = int(np.ravel_multi_index(state, self.env_shape))
            
            # initialize the state vector of RMs
            initial_configuration = tuple(src.reset() for src in self.env.get_reward_sources() if hasattr(src, "step"))
            rm_configuration = initial_configuration

            terminated = False
            truncated = False
            last_state = None
            last_action = None
            rich.print(f"Episode : {self.global_step}, Policies : {self.get_local_pcs(rm_configuration=initial_configuration, state=self.env.start_state_index)}")
            #rich.print(f"Episode : {self.global_step}, Policies : {len(self.get_local_pcs(rm_configuration=initial_configuration, state=self.env.start_state_index))}")
            # clean_policy = list(map(lambda x: (x[0], x[1]), self.get_local_pcs(rm_configuration=initial_configuration, state=self.env.start_state_index)))
            # print(f"Episode : {self.global_step}, Policies : {' -> '.join(clean_policy)}")
            while not (terminated or truncated) and self.global_step < total_timesteps and local_step < max_local_steps:
                self.global_step += 1
                local_step += 1
                # choose action from state and current rm_state
                action = self.select_action(state, rm_configuration, score_func)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                next_rm_configuration = tuple(s for s in self.env.get_rm_states() if s is not None)

                self.counts[state, action] += 1

                # Optimization methods
                if optimization == "Qsets+RI":
                    for configuration in self.rm_configurations:
                        if -1 not in configuration:
                            rewards, next_configuration = self.env.get_successor_rewards(configuration, state, action, info)
                            next_configuration = tuple(s for s in next_configuration if s is not None)
                            
                            # self.counts is not modified from pql since all counts are updated for all possible configuration of rms
                            #print(f"configuration : {configuration}, state : {state}, action : {action}, rewards : {rewards}, next_state : {next_state}, next_configuration : {next_configuration}")
                            self.non_dominated[configuration][state][action] = self.calc_non_dominated(next_configuration, next_state)
                            self.avg_reward[configuration][state, action] += (rewards - self.avg_reward[configuration][state, action]) / self.counts[state, action]

                elif optimization == "Qsets":
                    for configuration in self.rm_configurations:
                        if -1 not in configuration:
                            rewards, next_configuration = self.env.get_successor_rewards(configuration, state, action,
                                                                                         info)
                            next_configuration = tuple(s for s in next_configuration if s is not None)

                            self.non_dominated[configuration][state][action] = self.calc_non_dominated(
                                next_configuration, next_state)

                    # Counts are now dependent on the RM states
                    self.counts_rmd[rm_configuration, state, action] += 1

                    self.avg_reward[rm_configuration][state, action] += (reward - self.avg_reward[rm_configuration][state, action]) / self.counts_rmd[rm_configuration, state, action]
                elif optimization == "RI":
                    self.non_dominated[rm_configuration][state][action] = self.calc_non_dominated(next_rm_configuration, next_state)

                    for configuration in self.rm_configurations:
                        if -1 not in configuration:
                            rewards, next_configuration = self.env.get_successor_rewards(configuration, state, action,
                                                                                         info)
                            self.avg_reward[configuration][state, action] += (rewards - self.avg_reward[configuration][state, action]) / self.counts[state, action]

                else:
                    self.non_dominated[rm_configuration][state][action] = self.calc_non_dominated(next_rm_configuration, next_state)
                    self.avg_reward[rm_configuration][state, action] += (reward - self.avg_reward[rm_configuration][state, action]) / self.counts[state, action]
                            
                rm_configuration = next_rm_configuration
                state = next_state

                if terminated and not next_state in self.terminal_states:
                    self.terminal_states.append(next_state)
                    for a in range(self.num_actions):
                        for configuration in self.rm_configurations:
                            self.non_dominated[configuration][next_state][a] = {tuple(np.zeros(self.num_objectives))}

                if convergence_callback is not None and self.global_step % log_every == 0:
                    convergence_callback(
                        agent=self,
                        env=callback_env,
                        initial_configuration=initial_configuration,
                        step=self.global_step,
                        episode=episode_idx,
                    )

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        return self.get_local_pcs(rm_configuration=initial_configuration, state=self.env.start_state_index)

    def _eval_all_policies(self, env: gym.Env) -> List[np.ndarray]:
        """Evaluate all learned policies by tracking them."""
        pf = []
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env))

        return pf

    def track_policy(self, vec, env: gym.Env, tol=1e-3, max_steps=float('inf')):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
            max_steps (int, optional): The maximum number of steps taken in the output plan

        Returns:
            List of tuples (state, action, candidate_vector, rm_configuration, immediate_reward) for each step.
        """
        actions_followed = []
        target = np.array(vec)
        state, _ = env.reset()
        rm_configuration = tuple(src.reset() for src in env.get_reward_sources() if hasattr(src, "step"))
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)
        current_gamma = 1.0

        policy_steps = 0
        while not (terminated or truncated)\
                and policy_steps < max_steps:
            policy_steps += 1
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target
            candidate_vector = None
            selected_im_rew = None

            for action in range(self.num_actions):
                im_rew = self.avg_reward[rm_configuration][state, action]
                non_dominated_set = self.non_dominated[rm_configuration][state][action]
                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    #print(f"ACTION : {action}, REWARD : {im_rew}, NON-DOMINATED : {q}, DISTANCE : {dist}")
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q
                        candidate_vector = self.gamma * q + im_rew
                        selected_im_rew = im_rew

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            # Include RM configuration and immediate reward in the step tuple
            actions_followed.append((state, closest_action, candidate_vector, rm_configuration, selected_im_rew, dist))
            state, reward, terminated, truncated, _ = env.step(closest_action)
            # To follow actions
            #env.render()
            rm_configuration = tuple(s for s in env.get_rm_states() if s is not None)
            
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            target = new_target

        return actions_followed

    def get_local_pcs(self, rm_configuration: tuple, state: int = 0, ):
        """Collect the local PCS in a given state.

        Args:
            state (int): The state to get a local PCS for. (Default value = 0)
            rm_configuration (tuple): The current state of each reward machine.

        Returns:
            Set: A set of Pareto optimal vectors.
        """
        q_sets = [self.get_q_set(rm_configuration, state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        if len(candidates) == 0:
            return {}
        else:
            return get_non_dominated(candidates)

    def episode_end(self, rm_configuration: tuple) -> np.ndarray:
        """Compute the rewards that RMs would grant for a "last" transition.

        This simulates taking a transition labelled "last" in each reward machine
        and returns the rewards that would be granted.

        Args:
            rm_configuration (tuple): The current state of each reward machine.

        Returns:
            np.ndarray: Array of rewards from each RM for the "last" transition.
        """
        rewards = []
        last_props = ["last"]  # The proposition indicating episode end

        rm_idx = 0
        for src in self.env.get_reward_sources():
            if hasattr(src, "step"):
                # Get current RM state from configuration
                u1 = rm_configuration[rm_idx]
                # Simulate step with "last" proposition
                _, reward, _ = src.step(u1, last_props, s_info={}, env_done=True)
                rewards.append(reward)
                rm_idx += 1
            else:
                # Non-RM reward source: no additional reward for "last"
                rewards.append(0.0)

        return np.array(rewards)
