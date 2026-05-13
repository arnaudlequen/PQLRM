from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.reward_machines.reward_functions import RewardFunction, ConstantRewardFunction, SumRewardFunction
from environments.reward_machines.reward_machine_utils import evaluate_dnf, simplify_formula_to_dnf, value_iteration

from graphviz import Digraph

class RewardMachine:
    def __init__(self, file: str | None = None) -> None:
        # <U,u0,delta_u,delta_r>
        self.U: list[int] = []         # list of non-terminal RM states
        self.u0: int | None = None     # initial state
        self.delta_u: dict[int, dict[int, str]] = {}    # state-transition function
        self.delta_r: dict[int, dict[int, RewardFunction]] = {}    # reward-transition function
        self.terminal_u: int = -1  # All terminal states are sent to the same terminal state with id *-1*
        self.known_transitions: dict[tuple[int, tuple[str, ...]], int] = {} # Auxiliary variable to speed up computation of the next RM state

        self.add_rs: bool = False
        self.accepting_states: set[int] = set()

        if file is not None:
            self._load_reward_machine(file)

    # Public methods -----------------------------------

    def add_reward_shaping(self, gamma: float, rs_gamma: float) -> None:
        """
        It computes the potential values for shaping the reward function:
            - gamma(float):    this is the gamma from the environment
            - rs_gamma(float): this gamma that is used in the value iteration that compute the shaping potentials
        """
        self.gamma: float = gamma
        self.potentials: dict[int, float] = value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_u, rs_gamma)
        self.add_rs = True

        for u in self.potentials:
            self.potentials[u] = -self.potentials[u]

    def reset(self) -> int | None:
        # Returns the initial state
        return self.u0

    def get_next_state(self, u1: int, true_props: Iterable[str]) -> int:
        """
        Return the state that results of a transition where true_props are the true propositions
        """
        tp = tuple(sorted(true_props))
        if (u1, tp) not in self.known_transitions:
            u2 = self._compute_next_state(u1, tp)
            self.known_transitions[(u1, tp)] = u2
        return self.known_transitions[(u1, tp)]

    def step(self, u1: int, true_props: Iterable[str], s_info: dict[str, Any] | None, env_done: bool = False) -> tuple[int, float, bool]:
        """
        Emulates a step on the reward machine from state u1 when observing true_props.
        The rest of the parameters are for computing the reward when working with non-simple RMs.

        Args:
            u1: Current state of the reward machine.
            true_props: Set of propositions that are true in the current environment state.
            s_info: Extra state information used to compute the reward (for non-simple RMs).
            env_done: Whether the environment has reached a terminal state.

        Returns:
            A tuple (u2, reward, done) where:
                - u2: The next state of the reward machine.
                - reward: The reward for this transition.
                - done: Whether the reward machine has reached a terminal state.
        """

        # Computing the next state in the RM and checking if the episode is done
        #assert u1 != self.terminal_u, "the RM was set to a terminal state!"
        u2 = self.get_next_state(u1, true_props)
        done = (u2 == self.terminal_u)
        # Getting the reward
        reward = self._get_reward(u1, u2, s_info, env_done)

        return u2, reward, done

    def get_states(self) -> list[int]:
        return self.U

    def get_useful_transitions(self, u1: int) -> list[list[str]]:
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]

    def set_initial_state(self, u0: int) -> None:
        """Sets the initial state of the reward machine."""
        self.u0 = u0
        self._add_state([u0])

    def add_transition(self, u1: int, u2: int, dnf_formula: str, reward: float | RewardFunction) -> None:
        """
        Adds a transition to the reward machine.

        Args:
            u1: Source state
            u2: Destination state (use self.terminal_u for terminal)
            dnf_formula: DNF formula for the transition condition
            reward: Reward value (float) or RewardFunction instance
        """
        # Add states
        self._add_state([u1])
        if u2 != self.terminal_u:
            self._add_state([u2])

        # Add state-transition to delta_u
        if u1 not in self.delta_u:
            self.delta_u[u1] = {}
        self.delta_u[u1][u2] = dnf_formula

        # Add reward-transition to delta_r
        if u1 not in self.delta_r:
            self.delta_r[u1] = {}
        if isinstance(reward, RewardFunction):
            self.delta_r[u1][u2] = reward
        else:
            self.delta_r[u1][u2] = ConstantRewardFunction(reward)

    def finalize(self) -> None:
        """Sorts states after all transitions have been added."""
        self.U = sorted(self.U)

    def simplify_transition_formulas(self, simplify: bool = True) -> None:
        """Convert all transition labels to DNF, optionally minimizing them."""
        for u1 in list(self.delta_u.keys()):
            for u2 in list(self.delta_u[u1].keys()):
                self.delta_u[u1][u2] = simplify_formula_to_dnf(self.delta_u[u1][u2], simplify=simplify)

    def remove_unreachable_states(self) -> None:
        """Remove states that are not reachable from the initial state."""
        if self.u0 is None:
            return

        def _is_false_edge(formula: str) -> bool:
            try:
                return simplify_formula_to_dnf(formula, simplify=False) == "False"
            except Exception:
                return formula.strip() == "False"

        reachable: set[int] = set()
        queue: list[int] = [self.u0]
        while queue:
            u = queue.pop()
            if u in reachable:
                continue
            reachable.add(u)
            for v, formula in self.delta_u.get(u, {}).items():
                if _is_false_edge(formula):
                    continue
                if v != self.terminal_u and v not in reachable:
                    queue.append(v)

        self.U = sorted([u for u in self.U if u in reachable])
        self.accepting_states &= reachable

        new_delta_u: dict[int, dict[int, str]] = {}
        new_delta_r: dict[int, dict[int, RewardFunction]] = {}
        for u1 in reachable:
            outgoing_u = self.delta_u.get(u1, {})
            outgoing_r = self.delta_r.get(u1, {})
            for u2, formula in outgoing_u.items():
                if u2 != self.terminal_u and u2 not in reachable:
                    continue
                new_delta_u.setdefault(u1, {})[u2] = formula
                if u2 in outgoing_r:
                    new_delta_r.setdefault(u1, {})[u2] = outgoing_r[u2]

        self.delta_u = new_delta_u
        self.delta_r = new_delta_r
        self.known_transitions = {
            (u1, tp): u2
            for (u1, tp), u2 in self.known_transitions.items()
            if u1 in reachable and (u2 == self.terminal_u or u2 in reachable)
        }

        if hasattr(self, "potentials"):
            self.potentials = {u: v for u, v in self.potentials.items() if u in reachable or u == self.terminal_u}

    def pretty_print(self) -> None:
        """
        Prints the reward machine in a compact, colorful format using rich.
        """
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console()

        # Header with states info
        console.print(f"[bold blue]Reward Machine[/] | "
                      f"[green]u0=[/][bold]{self.u0}[/] | "
                      f"[dim]states=[/]{self.U} | "
                      f"[red]term=[/][bold]{self.terminal_u}[/]")

        # Transitions table
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("From", style="cyan", justify="center")
        table.add_column("", justify="center")  # Arrow
        table.add_column("To", justify="center")
        table.add_column("Condition", style="yellow")
        table.add_column("Reward", justify="right")

        for u1 in sorted(self.delta_u.keys()):
            for u2 in sorted(self.delta_u[u1].keys()):
                formula = self.delta_u[u1][u2]

                # Get reward info
                reward_str = "?"
                if u1 in self.delta_r and u2 in self.delta_r[u1]:
                    r = self._get_reward(u1, u2, None, False)
                    reward_str = f"[bold green]{r}[/]" if r > 0 else f"[dim]{r}[/]"

                # Format destination state
                if u2 == self.terminal_u:
                    dest = Text("TERM", style="bold red")
                elif u1 == u2:
                    dest = Text(str(u2), style="cyan dim")
                else:
                    dest = Text(str(u2), style="bold white")

                # Arrow style
                arrow = "↺" if u1 == u2 else "→"
                arrow_style = "dim" if u1 == u2 else "green"

                table.add_row(str(u1), f"[{arrow_style}]{arrow}[/]", dest, formula, reward_str)

        console.print(table)

    def visualize(self, filename: str | None = None, view: bool = True, format: str = "png") -> str:
        """
        Visualizes the reward machine as a directed graph using graphviz.

        Args:
            filename: Output filename (without extension). If None, uses a temp file.
            view: If True, opens the rendered image.
            format: Output format ('png', 'pdf', 'svg', etc.)

        Returns:
            Path to the rendered file.
        """

        dot = Digraph(comment="Reward Machine")
        dot.attr(rankdir="LR")  # Left to right layout
        has_terminal_edges = any(
            u2 == self.terminal_u
            for u1 in self.delta_u
            for u2 in self.delta_u[u1]
        )

        # Add states as nodes
        for u in self.U:
            is_initial = (u == self.u0)
            is_accepting = (u in self.accepting_states)
            shape = "doublecircle" if is_accepting else "circle"
            dot.node(str(u), str(u), shape=shape)

        # Add terminal state only if needed.
        if has_terminal_edges:
            dot.node("TERM", "⊗", shape="doublecircle", color="red", style="bold")

        # Add initial state's arrow
        dot.node("start", "", shape="none", width="0", height="0")
        dot.edge("start", str(self.u0), style="bold", color="green")

        # Add transitions as edges
        for u1 in sorted(self.delta_u.keys()):
            for u2 in sorted(self.delta_u[u1].keys()):
                formula = self.delta_u[u1][u2]

                # Get reward
                reward_str = ""
                if u1 in self.delta_r and u2 in self.delta_r[u1]:
                    r = self._get_reward(u1, u2, None, False)
                    if r != 0:
                        reward_str = f"\nr={r}"

                label = f"{formula}{reward_str}"
                if u2 == self.terminal_u:
                    if not has_terminal_edges:
                        continue
                    dest = "TERM"
                else:
                    dest = str(u2)

                # Style: positive reward edges in green/bold
                attrs: dict[str, str] = {}
                if reward_str and "r=" in reward_str and "r=0" not in reward_str:
                    attrs = {"color": "green", "style": "bold", "fontcolor": "green"}

                dot.edge(str(u1), dest, label=label, **attrs)

        # Render
        if filename is None:
            import tempfile
            filename = tempfile.mktemp(prefix="reward_machine_")

        output_path = dot.render(filename, format=format, view=view, cleanup=True)
        return output_path

    def __str__(self) -> str:
        """Returns a compact string representation of the reward machine."""
        lines = [
            f"RewardMachine(u0={self.u0}, states={self.U}, terminal={self.terminal_u})",
            "Transitions:"
        ]
        for u1 in sorted(self.delta_u.keys()):
            for u2 in sorted(self.delta_u[u1].keys()):
                formula = self.delta_u[u1][u2]
                dest = "TERM" if u2 == self.terminal_u else str(u2)
                lines.append(f"  {u1} --[{formula}]--> {dest}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Returns a technical representation of the reward machine."""
        return f"RewardMachine(u0={self.u0}, U={self.U}, terminal_u={self.terminal_u})"

    def merge_with(
        self,
        other: "RewardMachine",
        left_weight: float = 1.0,
        right_weight: float = 1.0,
        simplify_formulas: bool = True,
    ) -> "RewardMachine":
        """
        Build the synchronous product of two reward machines.

        The combined reward is a weighted sum of transition rewards from both RMs.
        A product transition goes to terminal if either component transitions to terminal.
        """
        if self.u0 is None or other.u0 is None:
            raise ValueError("Both reward machines must have an initial state set before merging.")

        def _and_formula(f1: str, f2: str) -> str:
            if f1 == "True":
                return f2
            if f2 == "True":
                return f1
            return f"({f1})&({f2})"

        def _merge_or(formulas: list[str]) -> str:
            if not formulas:
                return "False"
            if "True" in formulas:
                return "True"
            if len(formulas) == 1:
                return formulas[0]
            return "|".join(f"({f})" for f in formulas)

        def _get_outgoing(rm: "RewardMachine", u: int) -> list[tuple[int, str]]:
            # Mirror RM runtime semantics: missing transitions imply terminal on any proposition.
            if u not in rm.delta_u:
                return [(rm.terminal_u, "True")]
            return list(rm.delta_u[u].items())

        def _get_reward_fn(rm: "RewardMachine", u1: int, u2: int) -> RewardFunction:
            if u1 in rm.delta_r and u2 in rm.delta_r[u1]:
                return rm.delta_r[u1][u2]
            return ConstantRewardFunction(0.0)

        product = RewardMachine()

        pair_to_state: dict[tuple[int, int], int] = {}
        next_state_id = 0

        def _state_of(pair: tuple[int, int]) -> int:
            nonlocal next_state_id
            if pair not in pair_to_state:
                pair_to_state[pair] = next_state_id
                next_state_id += 1
            return pair_to_state[pair]

        initial_pair = (self.u0, other.u0)
        product.set_initial_state(_state_of(initial_pair))

        # Optional accepting-state propagation for visualization.
        left_acc = set(getattr(self, "accepting_states", set()))
        right_acc = set(getattr(other, "accepting_states", set()))
        product_accepting: set[int] = set()

        # BFS over reachable product states.
        from collections import deque, defaultdict
        queue = deque([initial_pair])
        seen: set[tuple[int, int]] = set()

        # Aggregate transitions by (src,dst) so conditions can be OR-merged.
        conds_by_edge: dict[tuple[int, int], list[str]] = defaultdict(list)
        reward_by_edge: dict[tuple[int, int], RewardFunction] = {}

        while queue:
            p = queue.popleft()
            if p in seen:
                continue
            seen.add(p)

            u1, v1 = p
            src_state = _state_of(p)

            if u1 in left_acc and v1 in right_acc:
                product_accepting.add(src_state)

            out1 = _get_outgoing(self, u1)
            out2 = _get_outgoing(other, v1)

            for u2, f1 in out1:
                for v2, f2 in out2:
                    combined_formula = _and_formula(f1, f2)
                    if u2 == self.terminal_u or v2 == other.terminal_u:
                        dst_state = product.terminal_u
                    else:
                        pair2 = (u2, v2)
                        dst_state = _state_of(pair2)
                        if pair2 not in seen:
                            queue.append(pair2)

                    edge = (src_state, dst_state)
                    conds_by_edge[edge].append(combined_formula)

                    # Reward depends only on component transitions, so it is consistent per edge.
                    r_left = _get_reward_fn(self, u1, u2)
                    r_right = _get_reward_fn(other, v1, v2)
                    reward_by_edge[edge] = SumRewardFunction(
                        r_left, r_right, left_weight=left_weight, right_weight=right_weight
                    )

        for (src, dst), conds in conds_by_edge.items():
            product.add_transition(src, dst, _merge_or(conds), reward_by_edge[(src, dst)])

        product.accepting_states = product_accepting
        product.simplify_transition_formulas(simplify=simplify_formulas)
        product.finalize()
        return product

    # Private methods -----------------------------------

    def _load_reward_machine(self, file: str) -> None:
        """
        Example:
            0      # initial state
            [2]    # terminal state
            (0,0,'!e&!n',ConstantRewardFunction(0))
            (0,1,'e&!g&!n',ConstantRewardFunction(0))
            (0,2,'e&g&!n',ConstantRewardFunction(1))
            (1,1,'!g&!n',ConstantRewardFunction(0))
            (1,2,'g&!n',ConstantRewardFunction(1))
        """
        # Reading the file
        f = open(file)
        lines: list[str] = [l.rstrip() for l in f]
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        terminal_states: list[int] = eval(lines[1])
        # adding transitions
        for e in lines[2:]:
            # Reading the transition
            u1: int
            u2: int
            dnf_formula: str
            reward_function: RewardFunction
            u1, u2, dnf_formula, reward_function = eval(e)
            # terminal states
            if u1 in terminal_states:
                continue
            if u2 in terminal_states:
                u2 = self.terminal_u
            # Adding machine state
            self._add_state([u1, u2])
            # Adding state-transition to delta_u
            if u1 not in self.delta_u:
                self.delta_u[u1] = {}
            self.delta_u[u1][u2] = dnf_formula
            # Adding reward-transition to delta_r
            if u1 not in self.delta_r:
                self.delta_r[u1] = {}
            self.delta_r[u1][u2] = reward_function
        # Sorting self.U... just because...
        self.U = sorted(self.U)

    def _compute_next_state(self, u1: int, true_props: Iterable[str]) -> int:
        # If state has no transitions defined, default to terminal

        if u1 not in self.delta_u:
            return self.terminal_u
        for u2 in self.delta_u[u1]:
            if evaluate_dnf(self.delta_u[u1][u2], true_props):
                return u2
        return self.terminal_u # no transition is defined for true_props


    def _get_reward(self, u1: int, u2: int, s_info: dict[str, Any] | None, env_done: bool) -> float:
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward: float = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            reward += self.delta_r[u1][u2].get_reward(s_info)
        # Adding the reward shaping (if needed)
        rs: float = 0.0
        if self.add_rs:
            un = self.terminal_u if env_done else u2 # If the env reached a terminal state, we have to use the potential from the terminal RM state to keep RS optimality guarantees
            rs = self.gamma * self.potentials[un] - self.potentials[u1]
        # Returning final reward
        return reward + rs

    def _add_state(self, u_list: list[int]) -> None:
        for u in u_list:
            if u not in self.U and u != self.terminal_u:
                self.U.append(u)
