"""
Exp 2, PQLRM: get_mail + get_coffee + no_hit_decoration.

Trains for `--seed-nbr` seeds and writes one convergence trace per seed into
dataviz/data/exp2/pqlrm_mail_coffee_nohit.csv (hypervolume + cardinality).
"""

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from baselines.pql_rm import PQLRM
from environments.office_world.office_world import OfficeWorld, time_penalty
from tests.office_world.common import track_and_save_policies
from experiments.office_world.rm_ow import rm_get_mail, rm_get_coffee, rm_no_hit_deco
from experiments.office_world.convergence_logging import (
    init_hv_csv, make_hv_logger, csv_path_for,
)


EXP_ID = 2
ALGO = "pqlrm"
DESCRIPTOR = "mail_coffee_nohit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-nbr", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--max-local-steps", type=int, default=500)
    parser.add_argument("--save-policies", action="store_true")
    args = parser.parse_args()

    env_map = "default_office"
    ref_point = np.array([0, 0, 0])
    csv_path = csv_path_for(EXP_ID, ALGO, DESCRIPTOR, REPO_ROOT)
    init_hv_csv(csv_path)
    print(f"Writing convergence trace -> {csv_path}")

    last_agent = last_env = last_pf = None
    for seed in range(args.seed_start, args.seed_start + args.seed_nbr):
        task_coffee = rm_get_coffee()
        task_mail = rm_get_mail()
        task_no_hit = rm_no_hit_deco()
        env = OfficeWorld(
            map=env_map,
            reward_sources=[task_no_hit, task_coffee, task_mail],
            render_mode="ansi",
        )

        agent = PQLRM(
            env, ref_point,
            gamma=0.95,
            initial_epsilon=1.0,
            epsilon_decay_steps=args.total_steps,
            final_epsilon=0.1,
            seed=seed,
            output_file=None,
            log=False,
        )

        logger = make_hv_logger(seed=seed, csv_path=csv_path, max_steps=args.max_local_steps)

        print(f"\n[seed={seed}] Training PQLRM for {args.total_steps} steps ...")
        pf = agent.train(
            total_timesteps=args.total_steps,
            action_eval="pareto_cardinality",
            ref_point=ref_point,
            eval_env=env,
            log_every=args.log_every,
            max_local_steps=args.max_local_steps,
            optimization="Qsets+RI",
            convergence_callback=logger,
        )
        print(f"[seed={seed}] |PF| = {len(pf)}")
        last_agent, last_env, last_pf = agent, env, pf

    if args.save_policies and last_pf is not None:
        out_json = Path(__file__).with_suffix(".json")
        track_and_save_policies(
            last_agent, last_env, last_pf,
            output_file=str(out_json),
            map_shape="Default",
            include_rewards=True,
            reward_index=1,
            max_steps=100,
        )


if __name__ == "__main__":
    main()
