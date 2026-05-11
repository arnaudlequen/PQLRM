import numpy as np
import os

from environments.pressurizedBountifulSeaTreasure import PBSTEnv, DiscreteObservationWrapper
from tests.test_rm_pbst import build_pbst_rm_time, build_pbst_rm_treasure, build_pbst_rm_pressure

from baselines.pql_rm import PQLRM
from baselines.pql import PQL

def main():
    # -- Build RMs --
    env_ref = PBSTEnv(render_mode=None)
    rm_time = build_pbst_rm_time(time_penalty=1.0)
    rm_treasure = build_pbst_rm_treasure(env_ref._treasure)
    rm_pressure = build_pbst_rm_pressure()

    print(f"\n[RM_time]     {rm_time}")
    print(f"[RM_treasure] {rm_treasure}")
    print(f"[RM_pressure] {rm_pressure}")

    env_id = "pressurised-bountiful-sea-treasure"
    agents_to_test = ["PQL", "PQLRM"]
    nbofruns = 1

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

    for run in range(1, nbofruns + 1):
        for agent_id in agents_to_test:
            print(f"-----------{agent_id}-----------")
            if agent_id == "PQL":
                env = PBSTEnv(render_mode=None)
                env = DiscreteObservationWrapper(env)
                agent = PQL(
                    env,
                    ref_point,
                    gamma=1.0,
                    initial_epsilon=1.0,
                    epsilon_decay_steps=5000,
                    final_epsilon=0.2,
                    seed=run,
                    output_file=outputFile,
                    log=log,
                )

            elif agent_id == "PQLRM":
                env = PBSTEnv(render_mode=None,
                                reward_sources=[rm_time, rm_treasure, rm_pressure])
                env = DiscreteObservationWrapper(env)
                agent = PQLRM(
                    env,
                    ref_point,
                    gamma=1.0,
                    initial_epsilon=1.0,
                    epsilon_decay_steps=5000,
                    final_epsilon=0.2,
                    seed=run,
                    output_file=outputFile,
                    log=log,
                )

            pf = agent.train(total_timesteps=20_000,
                             action_eval="pareto_cardinality",
                             ref_point=ref_point,
                             eval_env=env,
                             log_every=2000)

            print(f'Total of {len(pf)} policies')
            for target in pf:
                target = np.array(target)
                policy = agent.track_policy(target, env=env)
                # assert np.all(tracked == target)
                print(f"Policy : {policy}")


if __name__ == "__main__":
    main()

