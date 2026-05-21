"""
PBST exp1, QRM: time + treasure + pressure (v2), one QRMAgent per task.

Writes one convergence trace per seed into
dataviz/data/pbst_exp1/qrm_time_treasure_pressure.csv (q-value sum at start state).
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from environments.pbst.pressurizedBountifulSeaTreasure import PBSTEnv, DiscreteObservationWrapper
from baselines.qrm import QRMAgent, MultiTaskQRMTrainer
from experiments.pbst.rm_pbst import (
    build_pbst_rm_time, build_pbst_rm_treasure, build_pbst_rm_pressure_v2,
)
from experiments.office_world.convergence_logging import (
    init_qsum_csv, make_qsum_logger, csv_path_for,
)


EXP_ID = 1
ALGO = "qrm"
DESCRIPTOR = "time_treasure_pressure"
ENV_TAG = "pbst"


def make_env_factories():
    env_ref = PBSTEnv(render_mode=None)
    rm_time = build_pbst_rm_time(time_penalty=1.0)
    rm_treasure = build_pbst_rm_treasure(env_ref._treasure)
    rm_pressure = build_pbst_rm_pressure_v2()
    rms = [rm_time, rm_treasure, rm_pressure]
    task_names = ["Time", "Treasure", "Pressure"]

    def make_factory(rm):
        return lambda: DiscreteObservationWrapper(PBSTEnv(reward_sources=[rm]))

    return [make_factory(rm) for rm in rms], rms, task_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-nbr", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=10_000)
    parser.add_argument("--max-local-steps", type=int, default=300)
    parser.add_argument("--log-every-steps", type=int, default=2_000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=0.5)
    args = parser.parse_args()

    csv_path = csv_path_for(EXP_ID, ALGO, DESCRIPTOR, REPO_ROOT, env=ENV_TAG)
    init_qsum_csv(csv_path)
    print(f"Writing convergence trace -> {csv_path}")

    for seed in range(args.seed_start, args.seed_start + args.seed_nbr):
        env_factories, rms, task_names = make_env_factories()

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
