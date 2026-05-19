import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.pql_rm import PQLRM

from environments.office_world.office_world import OfficeWorld, time_penalty


from tests.office_world.common import (
    track_and_save_policies,
)

from experiments.office_world.rm_ow import rm_get_mail, rm_get_coffee, rm_no_hit_deco


def main():
    env_id = "office_world"
    env_map = "default_office"
    filename = __file__.split(".")[0]

    # -- Init Environment --

    ref_point = np.array([-25, 0, 0])

    log = False
    if log:
        outputPath = os.path.join("Results", env_id, env_map)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        outputFile = os.path.join(outputPath, "result.txt")
        with open(outputFile, 'w') as of:
            line = "agent;run;step;hv;card\n"
            of.write(line)
    else:
        outputFile = None

        task_coffee = rm_get_coffee()
        task_mail = rm_get_mail()
        task_no_hit = rm_no_hit_deco()

        env = OfficeWorld(map=env_map, reward_sources=[task_no_hit, task_coffee, task_mail], render_mode='ansi')

            
        agent = PQLRM(
                env,
                ref_point,
                gamma=0.95,
                initial_epsilon=1.0,
                epsilon_decay_steps=100000,
                final_epsilon=0.1,
                seed=1,
                output_file=outputFile,
                log=log,                
                )
        
        pf = agent.train(total_timesteps=100000, 
                    action_eval="pareto_cardinality", 
                    ref_point=ref_point, 
                    eval_env=env,
                    log_every=1000,
                    max_local_steps=500,
                    optimization = "Qsets+RI") # with "None" --> many many policies != pql

        assert len(pf) > 0
        print(f"Pareto front : {pf}")
        output_file = filename + ".json"
        track_and_save_policies(
            agent,
            env,
            pf,
            output_file=output_file,
            map_shape="Default",
            include_rewards=True,
            reward_index=1,
            max_steps=100
        )




if __name__ == "__main__":
    main()

