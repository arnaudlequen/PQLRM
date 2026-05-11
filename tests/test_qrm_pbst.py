import numpy as np

from environments.pressurizedBountifulSeaTreasure import PBSTEnv, DiscreteObservationWrapper
from reward_machines.reward_machine import RewardMachine
from reward_machines.reward_functions import ConstantRewardFunction
from rl_agents.qrm import QRMAgent, MultiTaskQRMTrainer
from tests.test_pbst_rm import build_pbst_rm_time, build_pbst_rm_treasure, build_pbst_rm_pressure

def test_qrm_pbst():

    # ------------------------------------------------------------------
    # 1. Build reward machines
    # ------------------------------------------------------------------
    env_ref = PBSTEnv(render_mode=None)
    rm_time = build_pbst_rm_time(time_penalty=1.0)
    rm_treasure = build_pbst_rm_treasure(env_ref._treasure)
    rm_pressure = build_pbst_rm_pressure()
    env_ref = DiscreteObservationWrapper(env_ref)

    # ------------------------------------------------------------------
    # 2. Build one environment per task
    # ------------------------------------------------------------------
    env_factories = [
        lambda: DiscreteObservationWrapper(PBSTEnv(reward_sources=[rm_time])),
        lambda: DiscreteObservationWrapper(PBSTEnv(reward_sources=[rm_treasure])),
        lambda: DiscreteObservationWrapper(PBSTEnv(reward_sources=[rm_pressure])),
    ]

    # ------------------------------------------------------------------
    # 3. Build one QRMAgent per task
    # ------------------------------------------------------------------
    N_states  = env_ref.observation_space.n
    N_actions = env_ref.action_space.n

    agent_time = QRMAgent(rm_time, N_states, N_actions, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent_treasure = QRMAgent(rm_treasure, N_states, N_actions, alpha=0.1, gamma=0.99, epsilon=1.0)
    agent_pressure = QRMAgent(rm_pressure, N_states, N_actions, alpha=0.1, gamma=0.9, epsilon=0.1)

    agents = [agent_time, agent_treasure, agent_pressure]
    task_names = ["Time", "Treasure", "Pressure"]

    # ------------------------------------------------------------------
    # 4. Build the multi-task trainer
    # ------------------------------------------------------------------
    trainer = MultiTaskQRMTrainer(agents, env_factories, max_steps_per_episode=500)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    N_EPISODES = 100_000
    PRINT_EVERY = 10_000

    print(f"\n── Training  ({N_EPISODES} episodes, round-robin across {len(agents)} tasks) ──")
    trainer.train(N_EPISODES, print_every=PRINT_EVERY)

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
                  f"avg_last20={avg}  "
                  f"max={max(last)}")
        else:
            print(f"  [{name:7s}]  no data")

    # ------------------------------------------------------------------
    # 7. Greedy evaluation
    # ------------------------------------------------------------------
    print("\n── Greedy evaluation (100 episodes each, ε=0) ────────────────")

    for i, name in enumerate(task_names):
        mean_r, max_r = trainer.eval_agent(i, env_factories[i], n_eval_episodes=100)
        print(f"  [{name:7s}]  mean_return={mean_r}, max_return={max_r}")

    # ------------------------------------------------------------------
    # 8. Display policies
    # ------------------------------------------------------------------
    print("\n── Policies ────────────────")

    # Print policies
    for i, (name, agent) in enumerate(zip(task_names, trainer.agents)):
        policy = agent.get_policy()
        print(f"\n  [{name}] policy (u0 → greedy action per state):")
        print(f"    {policy[agent.rm.u0]}")

if __name__ == "__main__":
    test_qrm_pbst()
