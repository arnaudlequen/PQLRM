import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.pql import PQL

from environments.reward_machines.reward_machine import RewardMachine,ConstantRewardFunction
from environments.office_world.office_world_rm import OfficeWorldRM, time_penalty

from common import (
    track_and_save_policies,
)

def rm_get_coffee_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!coffee&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "coffee&!decoration", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!office&!decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(1, 2, "office&!decoration", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    #rm.pretty_print()
    return rm

def rm_get_coffee():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!coffee", ConstantRewardFunction(0))
    rm.add_transition(0, 1, "coffee", ConstantRewardFunction(1)) 
    rm.add_transition(1, 1, "!office", ConstantRewardFunction(0))
    rm.add_transition(1, 2, "office", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    rm.pretty_print()
    return rm

def rm_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!office&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, 1, "office", ConstantRewardFunction(1))
    rm.add_transition(0, 2, "decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    rm.pretty_print()
    return rm

def rm_get_mail_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!mail&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "mail&!decoration", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!office&!decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(1, 2, "office&!decoration", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def rm_get_mail():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!mail", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "mail", ConstantRewardFunction(1)) 
    rm.add_transition(1, 1, "!office", ConstantRewardFunction(0)) 
    rm.add_transition(1, 2, "office", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def rm_patrol():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!A", ConstantRewardFunction(0))
    rm.add_transition(0, 1, "A", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!B", ConstantRewardFunction(0)) 
    rm.add_transition(1, 2, "B", ConstantRewardFunction(0))  
    rm.add_transition(2, 2, "!C", ConstantRewardFunction(0)) 
    rm.add_transition(2, 3, "C", ConstantRewardFunction(0))
    rm.add_transition(3, 3, "!D", ConstantRewardFunction(0))   
    rm.add_transition(3, 4, "D", ConstantRewardFunction(0))
    rm.add_transition(4, 4, "!office", ConstantRewardFunction(0))
    rm.add_transition(4, 5, "office", ConstantRewardFunction(1)) # necessary to end the episode
    rm.add_transition(5, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def rm_patrol_no_hit_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!A&!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "A&!decoration", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!B&!decoration", ConstantRewardFunction(0))
    rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(1, 2, "B&!decoration", ConstantRewardFunction(0))  
    rm.add_transition(2, 2, "!C&!decoration", ConstantRewardFunction(0))
    rm.add_transition(2, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(2, 3, "C&!decoration", ConstantRewardFunction(0))
    rm.add_transition(3, rm.terminal_u, "decoration", ConstantRewardFunction(0))
    rm.add_transition(3, 3, "!D&!decoration", ConstantRewardFunction(0))   
    rm.add_transition(3, 4, "D&!decoration", ConstantRewardFunction(0))
    rm.add_transition(4, 5, "office", ConstantRewardFunction(1)) # necessary to end the episode
    rm.add_transition(4, rm.terminal_u, "decoration", ConstantRewardFunction(0))
    rm.add_transition(5, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def main():
    env_id = "office_world"
    env_map = "default_office"
    nbofruns = 1
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
        task_coffee_no_hit = rm_get_coffee_no_hit_deco()
        task_mail = rm_get_mail()
        task_mail_no_hit = rm_get_mail_no_hit_deco
        
        task_no_hit = rm_no_hit_deco()
        task_patrol = rm_patrol()
        env = OfficeWorldRM(map=env_map, reward_sources=[task_no_hit, task_coffee, task_mail], render_mode='ansi')

            
        agent = PQL(
                env,
                ref_point,
                gamma=0.95,
                initial_epsilon=1.0,
                epsilon_decay_steps=400000,
                final_epsilon=0.1,
                seed=1,
                output_file=outputFile,
                log=log,                
                )
        
        pf = agent.train(total_timesteps=400000, 
                    action_eval="pareto_cardinality", 
                    ref_point=ref_point, 
                    eval_env=env,
                    log_every=1000,
                    max_local_steps=200)

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
            max_steps=50
        )




if __name__ == "__main__":
    main()

