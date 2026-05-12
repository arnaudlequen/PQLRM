import argparse
import numpy as np
import sys
from pathlib import Path
from collections.abc import Callable
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from environments.reward_machines.reward_machine import RewardMachine, ConstantRewardFunction
from baselines.common.performance_indicators import hypervolume
from utils.plan_utils import save_plans_to_json

def track_and_save_policies(
    agent,
    env,
    pareto_front,
    output_file,
    map_shape,
    select_policies=None,
    max_steps=None,
    include_rewards=False,
    reward_index=1,
):
    """Track policies for selected Pareto points and save them to JSON."""
    def _decode_policy_states(policy_steps):
        if not hasattr(env, "decode_state"):
            return policy_steps
        decoded = []
        for step in policy_steps:
            if len(step) == 0:
                decoded.append(step)
                continue
            detailed_state = env.decode_state(step[0])
            pos_state = detailed_state['position']
            decoded.append((int(pos_state),) + tuple(step[1:]))
        return decoded

    selected_policies = pareto_front if select_policies is None else select_policies(pareto_front)

    all_plans = []
    for target in selected_policies:
        target = np.array(target)
        if max_steps is None:
            policy = agent.track_policy(target, env=env)
        else:
            policy = agent.track_policy(target, env=env, max_steps=max_steps)
        policy = _decode_policy_states(policy)

        clean_policy = list(map(lambda x: str((x[0], x[1])), policy))
        print(f"Policy : {' -> '.join(clean_policy)}")
        print()

        all_plans.append({
            "target": target,
            "steps": policy,
            "name": f"Target {target.tolist()}",
        })

    kwargs = {
        "include_rm_states": True,
        "rm_index": 0,
        "info": {"map_shape": map_shape, "env": "office_world"},
        "filepath": output_file,
    }
    if include_rewards:
        kwargs["include_rewards"] = True
        kwargs["reward_index"] = reward_index

    save_plans_to_json(all_plans, **kwargs)
    return all_plans