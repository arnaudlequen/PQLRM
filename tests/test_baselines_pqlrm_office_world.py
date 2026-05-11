import numpy as np
import os

from baselines.pql_rm import PQLRM
from baselines.pql import PQL

from reward_machines.reward_machine import RewardMachine,ConstantRewardFunction
from environments.office_world import OfficeWorld, time_penalty, go_to_office
from utils.plan_utils import save_plans_to_json

def rm_get_coffee_wo_deco():
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
    rm.add_transition(0, 1, "coffee", ConstantRewardFunction(0)) 
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
    rm.add_transition(0, 0, "!decoration", ConstantRewardFunction(0))
    rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(-10)) 
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    rm.add_reward_shaping(0.9, 0.9)
    #rm.pretty_print()
    return rm

def rm_get_mail_wo_deco():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!mail&!decoration", ConstantRewardFunction(0))
    #rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "mail&!decoration", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!office&!decoration", ConstantRewardFunction(0))
    #rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
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
    rm.add_transition(0, 1, "mail", ConstantRewardFunction(0)) 
    rm.add_transition(1, 1, "!office", ConstantRewardFunction(0)) 
    rm.add_transition(1, 2, "office", ConstantRewardFunction(1))  
    rm.add_transition(2, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def rm_get_mail_and_coffee():
    pass

def rm_patrol():
    # -- Create RewardMachine --
    rm = RewardMachine()
    # Initial state
    rm.set_initial_state(0)
    # Transitions
    rm.add_transition(0, 0, "!A&!decoration", ConstantRewardFunction(0))
    #rm.add_transition(0, rm.terminal_u, "decoration", ConstantRewardFunction(0)) 
    rm.add_transition(0, 1, "A&!decoration", ConstantRewardFunction(10)) 
    rm.add_transition(1, 1, "!B&!decoration", ConstantRewardFunction(0))
    #rm.add_transition(1, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(1, 2, "B&!decoration", ConstantRewardFunction(10))  
    rm.add_transition(2, 2, "!C&!decoration", ConstantRewardFunction(0))
    #rm.add_transition(2, rm.terminal_u, "decoration", ConstantRewardFunction(0))   
    rm.add_transition(2, 3, "C&!decoration", ConstantRewardFunction(10))
    #rm.add_transition(3, rm.terminal_u, "decoration", ConstantRewardFunction(0))
    rm.add_transition(3, 3, "!D&!decoration", ConstantRewardFunction(0))   
    rm.add_transition(3, 4, "D&!decoration", ConstantRewardFunction(10))  
    rm.add_transition(4, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.add_transition(rm.terminal_u, rm.terminal_u, "True", ConstantRewardFunction(0))
    rm.finalize()
    #rm.add_reward_shaping(0.9, 0.9)
    return rm

def main():
    env_id = "office_world"
    env_map = "default_office"
    agents_to_test = ["PQLRM"]
    nbofruns = 1

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
    for run in range(1, nbofruns+1):
        for agent_id in agents_to_test:
            print(f"-----------{agent_id}-----------")
            task1 = rm_get_coffee()
            task1b = rm_get_coffee_wo_deco()
            task2 = rm_get_mail()
            task3 = rm_no_hit_deco()
            task4 = rm_patrol()
            env = OfficeWorld(map="default_office", reward_sources=[time_penalty, task1, task2], render_mode='ansi')
            
            if agent_id == "PQLRM":
                
                agent = PQLRM(
                    env,
                    ref_point,
                    gamma=0.9,
                    initial_epsilon=1.0,
                    epsilon_decay_steps=50000,
                    final_epsilon=0.1,
                    seed=2,
                    output_file=outputFile,
                    log=log,
                )
            
            elif agent_id == "PQL":

                agent = PQL(
                    env,
                    ref_point,
                    gamma = 0.8,
                    initial_epsilon=1.0,
                    epsilon_decay_steps=50000,
                    final_epsilon=0.7,
                    seed=run,
                    output_file=outputFile,
                    log=log,
                )

            pf = agent.train(total_timesteps=100000, 
                        action_eval="hypervolume", 
                        ref_point=ref_point, 
                        eval_env=env,
                        log_every=1000)

            # Collect all plans for this agent
            all_plans = []
            for i, target in enumerate(pf):
                target = np.array(target)
                policy = agent.track_policy(target, env=env, max_steps=40)
                #assert np.all(tracked == target)
                clean_policy = list(map(lambda x: str((x[0], x[1])), policy))
                print(f"Policy : {' -> '.join(clean_policy)}")
                print()

                # Store the plan
                all_plans.append({
                    'target': target,
                    'steps': policy,
                    'name': f"Target {target.tolist()}"
                })

            # Save all plans to JSON
            save_plans_to_json(all_plans, info={'map_name': 'default_office', 'env': 'office_world'})


if __name__ == "__main__":
    main()

