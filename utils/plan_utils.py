"""
Utility functions for saving and loading plans.
"""

import json
from typing import List, Dict, Any, Optional


def save_plans_to_json(plans: List[Dict[str, Any]], filepath: str = "last_plan.json",
                       include_rm_states: bool = False, rm_index: Optional[int] = None,
                       include_rewards: bool = False, reward_index: Optional[int] = None,
                       info: Optional[Dict[str, Any]] = None):
    """
    Save a list of plans to a JSON file.

    Args:
        plans: List of plans, where each plan is a dict with:
            - 'target': the target vector (list of floats)
            - 'steps': list of (state, action, vector, rm_config, im_reward) tuples
            - 'name': optional name/description
        filepath: Path to save the JSON file
        include_rm_states: If True, include RM states in the output
        rm_index: If specified, only include the RM state at this index (e.g., 0 for policy RM)
        include_rewards: If True, include immediate rewards in the output
        reward_index: If specified, only include the reward at this index (e.g., 1 for task RM)
        info: Optional dict with general info (e.g., {'map_shape': [3, 9], 'env': 'cliffwalking'})
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_plans = []
    for plan in plans:
        target = plan.get('target')
        if target is not None:
            try:
                target = [float(x) for x in target]
            except (TypeError, ValueError):
                target = None

        steps = plan['steps']

        # Build step data
        if (include_rm_states or include_rewards) and len(steps) > 0 and len(steps[0]) >= 4:
            # track_policy returns (state, action, vector, rm_config, im_reward) tuples
            serialized_steps = []
            for step in steps:
                step_data = {
                    'state': int(step[0]),
                    'action': int(step[1]),
                }
                # Extract RM states
                if include_rm_states and len(step) >= 4 and step[3] is not None:
                    rm_config = step[3]
                    if rm_index is not None and rm_index < len(rm_config):
                        step_data['rm_state'] = int(rm_config[rm_index])
                    else:
                        step_data['rm_states'] = [int(s) for s in rm_config]

                # Extract immediate rewards
                if include_rewards and len(step) >= 5 and step[4] is not None:
                    im_reward = step[4]
                    try:
                        if reward_index is not None and reward_index < len(im_reward):
                            step_data['reward'] = float(im_reward[reward_index])
                        else:
                            step_data['rewards'] = [float(r) for r in im_reward]
                    except (TypeError, IndexError):
                        # Single reward value
                        step_data['reward'] = float(im_reward)

                serialized_steps.append(step_data)
        else:
            # Simple format: just state and action
            serialized_steps = [[int(step[0]), int(step[1])] for step in steps]

        serializable_plan = {
            'target': target,
            'steps': serialized_steps,
            'name': plan.get('name', f"Plan {len(serializable_plans) + 1}")
        }
        serializable_plans.append(serializable_plan)

    # Build output structure
    if info is not None:
        output = {
            'info': info,
            'plans': serializable_plans
        }
    else:
        # Backwards compatible: just the list of plans
        output = serializable_plans

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(plans)} plans to {filepath}")


def load_plans_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load plans from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dict with:
        - 'plans': List of plan dicts with 'target', 'steps', 'name' keys
        - 'info': Optional dict with general info (e.g., map_shape)

        Steps can be either:
        - Simple format: list of [state, action] pairs
        - Extended format: list of {'state': int, 'action': int, 'rm_state': int, 'reward': float} dicts
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    info = None
    plans = []

    # Handle new format with info field
    if isinstance(data, dict) and 'plans' in data:
        info = data.get('info')
        plan_list = data['plans']
    elif isinstance(data, list):
        plan_list = data
    else:
        return {'plans': [], 'info': None}

    if isinstance(plan_list, list):
        if len(plan_list) == 0:
            return {'plans': [], 'info': info}
        elif isinstance(plan_list[0], dict):
            # New format: list of plan dicts
            for plan_dict in plan_list:
                raw_steps = plan_dict.get('steps', [])
                # Handle both simple and extended step formats
                if raw_steps and isinstance(raw_steps[0], dict):
                    # Extended format with rm_state and/or reward
                    steps = []
                    for step in raw_steps:
                        step_tuple = (step['state'], step['action'])
                        # Add RM state
                        if 'rm_state' in step:
                            step_tuple = step_tuple + (step['rm_state'],)
                        elif 'rm_states' in step:
                            step_tuple = step_tuple + (tuple(step['rm_states']),)
                        # Add reward
                        if 'reward' in step:
                            step_tuple = step_tuple + (step['reward'],)
                        elif 'rewards' in step:
                            step_tuple = step_tuple + (tuple(step['rewards']),)
                        steps.append(step_tuple)
                else:
                    # Simple format: list of [state, action]
                    steps = [tuple(item) for item in raw_steps]

                plans.append({
                    'steps': steps,
                    'target': plan_dict.get('target'),
                    'name': plan_dict.get('name', f'Plan {len(plans) + 1}')
                })
        else:
            # Old format: single plan as list of [state, action]
            plans = [{
                'steps': [tuple(item) for item in plan_list],
                'target': None,
                'name': 'Plan 1'
            }]

    return {'plans': plans, 'info': info}
