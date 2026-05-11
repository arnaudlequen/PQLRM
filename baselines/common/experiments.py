"""Common experiment utilities."""

import argparse
import numpy as np

from morl_agents.pareto_q_learning.pql import PQL

ALGOS = {
    "pql": PQL
}

ENVS_WITH_KNOWN_PARETO_FRONT = [
    "deep-sea-treasure-concave-v0",
    "deep-sea-treasure-v0",
    "minecart-v0",
    "minecart-deterministic-v0",
    "resource-gathering-v0",
    "fruit-tree-v0",
]


class StoreDict(argparse.Action):
    """Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}

    From RL Baselines3 Zoo.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        """Init."""
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Convert list of strings to a dict."""
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            print(value)
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)
