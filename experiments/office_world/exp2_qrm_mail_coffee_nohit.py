"""
Exp 2, QRM: get_mail + get_coffee + no_hit_decoration (one QRMAgent per task).

Trains MultiTaskQRMTrainer in round-robin and writes one convergence trace
per seed into dataviz/data/exp2/qrm_mail_coffee_nohit.csv. The convergence
metric is the sum across tasks of V(s0, u0) = max_a Q[u0][s0][a], analogous
to avg_return in the Q-learning baselines of resources/CFXRL.
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from environments.office_world.office_world import OfficeWorld
from baselines.qrm import QRMAgent, MultiTaskQRMTrainer
from experiments.office_world.rm_ow import rm_get_mail, rm_get_coffee, rm_no_hit_deco
from experiments.office_world.convergence_logging import (
    init_qsum_csv, make_qsum_logger, csv_path_for,
)


EXP_ID = 2
ALGO = "qrm"
DESCRIPTOR = "mail_coffee_nohit"


def make_env_factories(map_name: str):
    rms = [rm_no_hit_deco(), rm_get_mail(), rm_get_coffee()]
    task_names = ["NoDeco", "Mail", "Coffee"]

    def make_factory(rm):
        return lambda: OfficeWorld(render_mode=None, map=map_name, reward_sources=[rm])

    return [make_factory(rm) for rm in rms], rms, task_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-nbr", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=2_000)
    parser.add_argument("--max-local-steps", type=int, default=500)
    parser.add_argument("--log-every-steps", type=int, default=1_000,
                        help="Step interval at which to log convergence.")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--map", type=str, default="default_office")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.5)
    args = parser.parse_args()

    csv_path = csv_path_for(EXP_ID, ALGO, DESCRIPTOR, REPO_ROOT)
    init_qsum_csv(csv_path)
    print(f"Writing convergence trace -> {csv_path}")

    for seed in range(args.seed_start, args.seed_start + args.seed_nbr):
        env_factories, rms, task_names = make_env_factories(args.map)

        probe = env_factories[0]()
        n_s = probe.observation_space.n
        n_a = probe.action_space.n
        if hasattr(probe, "close"):
            probe.close()

        agents = [
            QRMAgent(rm, n_s, n_a,
                     alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                     seed=seed)
            for rm in rms
        ]

        trainer = MultiTaskQRMTrainer(
            agents, env_factories, max_steps_per_episode=args.max_local_steps,
        )

        logger = make_qsum_logger(seed=seed, csv_path=csv_path, env_factories=env_factories)

        print(f"\n[seed={seed}] Training QRM ({args.n_episodes} episodes) on tasks: {task_names}")
        trainer.train(
            n_episodes=args.n_episodes,
            print_every=args.print_every,
            convergence_callback=logger,
            log_every=args.log_every_steps,
        )
        print(f"[seed={seed}] total_steps={trainer.total_steps}")


if __name__ == "__main__":
    main()
