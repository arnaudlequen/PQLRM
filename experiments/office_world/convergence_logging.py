"""
Convergence logging utilities for OfficeWorld experiments.

Two CSV formats are produced, both intended for dataviz/plot_results.py:

  - PQL / PQLRM (multi-objective): per checkpoint, hypervolume of the local
    PCS at the env's start state.

  Header: seed;step;episode;run;hypervolume;PFcardinality;plan_lengths;computation_time

  - QRM (one tabular Q-table per RM state, per task): per checkpoint, the
    sum across tasks of V(s0, u0) = max_a Q[u0][s0][a]. Analogous to
    avg_return in the Q-learning baselines under resources/CFXRL.

  Header: seed;step;episode;run;q_sum;per_task_q;computation_time

The `make_*` factories return callbacks compatible with the agent training
loops we patched: PQL/PQLRM.train(convergence_callback=...) calls with
kwargs (agent, env, initial_configuration, step, episode);
MultiTaskQRMTrainer.train(convergence_callback=...) calls with
kwargs (agents, step, episode).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


HV_HEADER = "seed;step;episode;run;hypervolume;PFcardinality;plan_lengths;computation_time\n"
QSUM_HEADER = "seed;step;episode;run;q_sum;per_task_q;computation_time\n"


def init_hv_csv(csv_path: str | Path) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(HV_HEADER)


def init_qsum_csv(csv_path: str | Path) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(QSUM_HEADER)


def make_hv_logger(
    seed: int,
    csv_path: str | Path,
    max_steps: int = 100,
) -> Callable:
    """Logger for PQL / PQLRM.

    Works with both agents because PQLRM passes `initial_configuration=<tuple>`
    while PQL passes `initial_configuration=None`. We branch internally on
    which `get_local_pcs` overload to call.
    """
    from baselines.common.performance_indicators import hypervolume

    start_time = time.perf_counter()

    def _logger(agent, env, initial_configuration, step: int, episode: int) -> None:
        if initial_configuration is None:
            pcs = list(agent.get_local_pcs(state=env.start_state_index))
        else:
            pcs = list(agent.get_local_pcs(
                rm_configuration=initial_configuration,
                state=env.start_state_index,
            ))

        plan_lengths: list[int] = []
        true_pcs = []
        for vec in pcs:
            try:
                tracked = agent.track_policy(np.array(vec), env=env, max_steps=max_steps)
                if len(tracked) < max_steps:
                    plan_lengths.append(len(tracked))
                    true_pcs.append(vec)
            except Exception:
                plan_lengths.append(-1)

        hv = float(hypervolume(agent.ref_point, true_pcs)) if pcs else 0.0
        cardinality = len(true_pcs)

        with open(csv_path, "a") as f:
            f.write(
                f"{seed};{step};{episode};run1;{hv};{cardinality};"
                f"{json.dumps(plan_lengths)};{time.perf_counter() - start_time}\n"
            )

    return _logger


def make_qsum_logger(
    seed: int,
    csv_path: str | Path,
    env_factories,
) -> Callable:
    """Logger for QRM (MultiTaskQRMTrainer).

    For each task i, computes V_i(s0, u0_i) = max_a Q[u0_i][s0_i, a]
    where s0_i is the start state of env_factories[i]() and u0_i is
    agent.rm.u0. The CSV records both per-task values and their sum.
    """
    # Resolve start states once
    start_states: list[int] = []
    for factory in env_factories:
        e = factory()
        s0 = getattr(e, "start_state_index", None)
        if s0 is None:
            s0, _ = e.reset()
        start_states.append(int(s0))
        if hasattr(e, "close"):
            e.close()

    start_time = time.perf_counter()

    def _logger(agents, step: int, episode: int) -> None:
        per_task: list[float] = []
        for agent, s0 in zip(agents, start_states):
            u0 = agent.rm.u0
            q_row = agent.Q[u0][s0]
            per_task.append(float(np.max(q_row)))
        q_sum = float(sum(per_task))

        with open(csv_path, "a") as f:
            f.write(
                f"{seed};{step};{episode};run1;{q_sum};{json.dumps(per_task)};"
                f"{time.perf_counter() - start_time}\n"
            )

    return _logger


def csv_path_for(
    exp_id: int,
    algo: str,
    descriptor: str,
    root: str | Path,
    env: str | None = None,
) -> Path:
    """Standard path:
        <root>/dataviz/data/exp<N>/<algo>_<descriptor>.csv               (env=None)
        <root>/dataviz/data/<env>_exp<N>/<algo>_<descriptor>.csv         (env given)
    The env prefix keeps results from different environments (office_world,
    pbst, ...) in distinct subfolders so the shared plot script picks
    them up as separate experiments.
    """
    subfolder = f"exp{exp_id}" if env is None else f"{env}_exp{exp_id}"
    return Path(root) / "dataviz" / "data" / subfolder / f"{algo}_{descriptor}.csv"
