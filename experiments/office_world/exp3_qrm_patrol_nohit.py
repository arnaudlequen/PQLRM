"""
test_qrm_office.py — QRM multi-task test for the OfficeWorld environment
=========================================================================
Mirrors the structure of test_qrm_pbst.py.

Tasks
-----
Task 0 — Time       : -1 per step (constant penalty, single-state RM)
Task 1 — Coffee     : pick up coffee (prop "coffee"), then reach office
                      (prop "office") without hitting decoration ("decoration")
Task 2 — Mail       : pick up mail (prop "mail"), then reach office
Task 3 — CoffeeAndMail : coffee AND mail before office, no decoration

Propositions emitted by OfficeWorldRM._get_true_props
------------------------------------------------------
"wall", "decoration", "coffee", "office", "mail", "A", "B", "C", "D"
(or their negations "!wall", "!coffee", etc.)
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#from environments.office_world.office_world_rm import OfficeWorldRM
from environments.office_world.office_world import OfficeWorld
from baselines.qrm import QRMAgent, MultiTaskQRMTrainer, SharedEnvTrainer
from experiments.office_world.rm_ow import rm_get_mail, rm_get_coffee, rm_patrol, rm_no_hit_deco

def test_qrm_office(map_name: str = "default_office") -> None:

    # ------------------------------------------------------------------
    # 1. Build reward machines
    # ------------------------------------------------------------------
    rm_ptrl         = rm_patrol()
    rm_no_deco  = rm_no_hit_deco()

    rms        = [rm_no_deco, rm_ptrl]#, rm_ptrl, rm_no_deco]
    task_names = ["Office_noDeco", "Patrol"]#, "Patrol", "No-deco"]

    # ------------------------------------------------------------------
    # 2. One env factory per task
    #    Each factory creates a fresh OfficeWorldRM with a single RM so
    #    that reward is always a scalar and rm_states[0] is unambiguous.
    # ------------------------------------------------------------------
    def make_factory(rm):
        return lambda: OfficeWorld(
            render_mode=None,
            map=map_name,
            reward_sources=[rm],
        )

    env_factories = [make_factory(rm) for rm in rms]

    # Probe one env to get state/action space sizes
    _probe = env_factories[0]()
    N_states  = _probe.observation_space.n
    N_actions = _probe.action_space.n
    _probe.close() if hasattr(_probe, "close") else None

    print(f"OfficeWorld ({map_name})")

    # ------------------------------------------------------------------
    # 3. One QRMAgent per task
    # ------------------------------------------------------------------
    agents = []
    for i, rm in enumerate(rms):
        _probe = env_factories[i]()
        n_s = _probe.observation_space.n
        n_a = _probe.action_space.n
        #print(n_s, n_a, rm.__str__())
        agents.append(QRMAgent(rm, n_s, n_a, alpha=0.1, gamma=0.95, epsilon=0.5))

    # ------------------------------------------------------------------
    # 4. Trainer
    # ------------------------------------------------------------------
    trainer = MultiTaskQRMTrainer(agents, env_factories, max_steps_per_episode=500)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    N_EPISODES  = 2_000
    PRINT_EVERY = 100

    print(f"\n── Training  ({N_EPISODES} episodes, round-robin across {len(agents)} tasks) ──")
    trainer.train(N_EPISODES, print_every=PRINT_EVERY)

    # ------------------------------------------------------------------
    # 6. Per-task reward curves (last 20 episodes each)
    # ------------------------------------------------------------------
    print("\n── Per-task reward (last 20 training episodes each) ─────────")
    for i, name in enumerate(task_names):
        history = trainer.reward_history[i]
        last = [r for _, r in history[-20:]]
        if last:
            avg = sum(last) / len(last)
            print(f"  [{name:12s}]  episodes_played={len(history):>5}  "
                  f"avg_last20={avg:+.3f}  max={max(last):+.3f}")
        else:
            print(f"  [{name:12s}]  no data")

    # ------------------------------------------------------------------
    # 7. Greedy evaluation
    # ------------------------------------------------------------------
    print("\n── Greedy evaluation (50 episodes each, ε=0) ────────────────")
    for i, name in enumerate(task_names):
        mean_r, max_r = trainer.eval_agent(i, env_factories[i], n_eval_episodes=50)
        print(f"  [{name:12s}]  mean_return={mean_r:+.3f}  max_return={max_r:+.3f}")

    # ------------------------------------------------------------------
    # 8. Policies
    # ------------------------------------------------------------------
    print("\n── Policies (u0, first 30 states) ───────────────────────────")
    for i, (name, agent) in enumerate(zip(task_names, agents)):
        policy = agent.get_policy()
        u0 = agent.rm.u0
        print(f"  [{name:12s}]  {list(policy[u0][:30])}…")


if __name__ == "__main__":
    test_qrm_office()