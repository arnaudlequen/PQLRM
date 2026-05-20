import numpy as np
import os

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.pbst.pressurizedBountifulSeaTreasure import PBSTEnv, DiscreteObservationWrapper
from environments.pbst.pbst_rm import PBSTEnv_rm
from experiments.pbst.rm_pbst import build_pbst_rm_time, build_pbst_rm_treasure, build_pbst_rm_pressure, build_pbst_rm_pressure_v2, build_pbst_rm_pressure_v3

from baselines.pql_rm import PQLRM

from tests.pbst.common import (
    track_and_save_policies,
)

def main():
    # -- Build RMs --
    env_ref = PBSTEnv(render_mode=None)
    rm_time = build_pbst_rm_time(time_penalty=1.0)
    rm_treasure = build_pbst_rm_treasure(env_ref._treasure)
    rm_pressure = build_pbst_rm_pressure_v2()

    print(f"\n[RM_time]     {rm_time}")
    print(f"[RM_treasure] {rm_treasure}")
    print(f"[RM_pressure] {rm_pressure}")

    env_id = "pressurised-bountiful-sea-treasure"
    filename = __file__.split(".")[0]

    EPSILON_MAX = 1
    EPSILON_STEPS = 50_000
    EPSILON_MIN = 0.1
    GAMMA = 0.999
    TRAINING_STEPS = 50_000
    EPISODE_LENGTH = 200

    # all policies obtained with gamma = 1, 20000 steps for epsilon and training
    # Bug:
    # When gamma < 1 and the episode length = 5, we generate rewards higher than -5 for time_penalty
    # Maybe because the reward is continuously propagated through the different episodes generating an infinite reward loop?
    # Check the evolution of the q_set for shorter episodes

    # -- Logs --

    ref_point = np.array([-25, -1, -11])

    log = True
    if log:
        outputPath = os.path.join("Results", env_id)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        outputFile = os.path.join(outputPath, "result.txt")
        with open(outputFile, 'w') as of:
            line = "agent;run;step;hv;card\n"
            of.write(line)

    # -- Training for each agent type (nbofruns times) --

        env = PBSTEnv(render_mode=None,
                        reward_sources=[rm_time, rm_treasure, rm_pressure]) # rm_time,rm_pressure,rm_treasure
        env = DiscreteObservationWrapper(env)
        agent = PQLRM(
            env,
            ref_point,
            gamma=GAMMA,
            initial_epsilon=EPSILON_MAX,
            epsilon_decay_steps=EPSILON_STEPS,
            final_epsilon=EPSILON_MIN,
            seed=1,
            output_file=outputFile,
            log=log,
        )

        pf = agent.train(total_timesteps=TRAINING_STEPS,
                            action_eval="pareto_cardinality",
                            ref_point=ref_point,
                            eval_env=env,
                            max_local_steps=EPISODE_LENGTH,
                            log_every=2000)

        print(f'Total of {len(pf)} policies')
        output_file = filename + ".json"
        track_and_save_policies(
            agent,
            env,
            pf,
            output_file=output_file,
            map_shape="Default",
            include_rewards=True,
            reward_index=1,
            max_steps=50
    )


if __name__ == "__main__":
    main()

