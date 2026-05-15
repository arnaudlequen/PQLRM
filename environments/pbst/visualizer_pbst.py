#!/usr/bin/env python3
"""
Terminal-based visualizer for the Pressurized Bountiful Sea Treasure (PBST) environment.

Usage:
    python3 visualizers/pbst.py [--plan PLAN_FILE]

Controls:
    Right Arrow / l : Step forward in the plan
    Left Arrow / h  : Step backward in the plan
    Up Arrow / k    : Switch to next plan
    Down Arrow / j  : Switch to previous plan
    r               : Reset to start of current plan
    q / Esc         : Quit
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

import numpy as np

# ---------------------------------------------------------------------------
# Grid constants (mirror pressurizedBountifulSeaTreasure.py)
# ---------------------------------------------------------------------------
SHAPE = (11, 10)  # (rows, cols)

UP    = 0
RIGHT = 1
DOWN  = 2
LEFT  = 3

ACTION_NAMES = {
    UP:    'UP',
    RIGHT: 'RIGHT',
    DOWN:  'DOWN',
    LEFT:  'LEFT',
}

ACTION_ARROWS = {
    UP:    '\u2191',
    RIGHT: '\u2192',
    DOWN:  '\u2193',
    LEFT:  '\u2190',
}

# Treasure positions and values (from the environment)
TREASURE_MAP: Dict[Tuple[int, int], int] = {
    (1, 0):  5,
    (2, 1):  80,
    (3, 2):  120,
    (4, 3):  140,
    (4, 4):  145,
    (4, 5):  150,
    (7, 6):  163,
    (7, 7):  166,
    (9, 8):  173,
    (10, 9): 175,
}

# Depth threshold (rows >= this are considered "deep")
DEEP_THRESHOLD = SHAPE[0] // 2  # 5


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _is_rock(row: int, col: int) -> bool:
    for x,y in TREASURE_MAP.keys():
        if y == col:
            return row > x
    return False

def _cell_symbol(row: int, col: int) -> str:
    """Return the background symbol for a grid cell."""
    if (row, col) in TREASURE_MAP:
        return '\u2605'   # ★  treasure
    if _is_rock(row, col):
        return '\u2591'   # rock
    if row >= DEEP_THRESHOLD:
        return '\u2248'   # ≈  deep water

    return '~'            # shallow water


def _cell_name(row: int, col: int) -> str:
    """Human-readable cell description."""
    if (row, col) in TREASURE_MAP:
        return f'Treasure ({TREASURE_MAP[(row, col)]})'
    if _is_rock(row, col):
        return 'Rock'   # rock
    if row >= DEEP_THRESHOLD:
        return 'Deep water'
    return 'Shallow water'


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------
class PBSTVisualizer:
    """Terminal-based visualizer for the PBST environment."""

    def __init__(self, shape: Tuple[int, int] = SHAPE):
        self.shape = shape
        self.plans: List[Dict[str, Any]] = []
        self.current_plan_idx = 0
        self.current_step = 0

    # ------------------------------------------------------------------
    # State / position helpers
    # ------------------------------------------------------------------
    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat state index to (row, col) position."""
        return tuple(np.unravel_index(state, self.shape))

    def pos_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) position to flat state index."""
        return int(np.ravel_multi_index((row, col), self.shape))

    # ------------------------------------------------------------------
    # Plan management
    # ------------------------------------------------------------------
    @property
    def current_plan(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.current_plan_idx < len(self.plans):
            return self.plans[self.current_plan_idx]
        return None

    @property
    def current_steps(self) -> List[Tuple]:
        plan = self.current_plan
        if plan is not None:
            return plan.get('steps', [])
        return []

    def load_plans(self, plans: List[Dict[str, Any]]):
        """Load a list of plan dicts (each with 'steps', 'target', 'name')."""
        self.plans = plans
        self.current_plan_idx = 0
        self.current_step = 0

    def load_plan(self, plan: List[Tuple], name: str = "Plan 1"):
        """Load a single plan as a list of (state, action) tuples."""
        self.plans = [{'steps': plan, 'target': None, 'name': name}]
        self.current_plan_idx = 0
        self.current_step = 0

    def load_plans_from_file(self, filepath: str):
        """Load plans from a JSON file (supports both old and new format)."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        info = None
        plan_list = None

        if isinstance(data, dict) and 'plans' in data:
            info = data.get('info')
            plan_list = data['plans']
            # Update shape if stored in info
            if info and 'map_shape' in info:
                s = info['map_shape']
                if s:
                    self.shape = tuple(s)
        elif isinstance(data, list):
            plan_list = data

        if plan_list is None or len(plan_list) == 0:
            self.plans = []
        elif isinstance(plan_list[0], dict):
            self.plans = []
            for plan_dict in plan_list:
                raw_steps = plan_dict.get('steps', [])
                steps = self._parse_steps(raw_steps)
                self.plans.append({
                    'steps': steps,
                    'target': plan_dict.get('target'),
                    'name': plan_dict.get('name', f'Plan {len(self.plans) + 1}'),
                })
        else:
            # Old format: single list of [state, action]
            self.plans = [{
                'steps': [tuple(item) for item in plan_list],
                'target': None,
                'name': 'Plan 1',
            }]

        self.current_plan_idx = 0
        self.current_step = 0

    def _parse_steps(self, raw_steps: list) -> list:
        """Parse steps from JSON, handling both list and dict formats."""
        if not raw_steps:
            return []

        steps = []
        for item in raw_steps:
            if isinstance(item, dict):
                # Extended format: {'state': int, 'action': int, 'rm_state': int, 'reward': ...}
                step = (item['state'], item['action'])
                if 'rm_state' in item:
                    step = step + (item['rm_state'],)
                elif 'rm_states' in item:
                    step = step + (tuple(item['rm_states']),)
                if 'reward' in item:
                    step = step + (item['reward'],)
                elif 'rewards' in item:
                    step = step + (tuple(item['rewards']),)
                steps.append(step)
            else:
                steps.append(tuple(item))
        return steps

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def get_current_state(self) -> Optional[Tuple]:
        steps = self.current_steps
        if 0 <= self.current_step < len(steps):
            return steps[self.current_step]
        return None

    def step_forward(self):
        if self.current_step < len(self.current_steps) - 1:
            self.current_step += 1

    def step_backward(self):
        if self.current_step > 0:
            self.current_step -= 1

    def next_plan(self):
        if self.current_plan_idx < len(self.plans) - 1:
            self.current_plan_idx += 1
            self.current_step = 0

    def prev_plan(self):
        if self.current_plan_idx > 0:
            self.current_plan_idx -= 1
            self.current_step = 0

    def reset(self):
        self.current_step = 0

    # ------------------------------------------------------------------
    # Curses rendering helpers
    # ------------------------------------------------------------------
    def _safe_addstr(self, stdscr, row: int, col: int, text: str, attr=0):
        max_y, max_x = stdscr.getmaxyx()
        if row < max_y - 1 and col < max_x:
            try:
                available_width = max_x - col - 1
                if len(text) > available_width:
                    text = text[:available_width]
                stdscr.addstr(row, col, text, attr)
            except curses.error:
                pass

    def _render_map_safe(self, stdscr, start_row: int, max_y: int, max_x: int) -> int:
        """Render the PBST grid with bounds checking."""
        current = self.get_current_state()
        agent_pos = None
        current_action = None

        if current is not None:
            state, action = current[0], current[1]
            agent_pos = self.state_to_pos(state)
            current_action = action

        border_h = '+' + '-' * (self.shape[1] * 2 + 1) + '+'
        self._safe_addstr(stdscr, start_row, 0, border_h)

        for row_idx in range(self.shape[0]):
            if start_row + row_idx + 1 >= max_y - 1:
                break
            line = '| '
            for col_idx in range(self.shape[1]):
                if agent_pos is not None and (row_idx, col_idx) == agent_pos:
                    line += ACTION_ARROWS.get(current_action, '@')
                else:
                    line += _cell_symbol(row_idx, col_idx)
                line += ' '
            line += '|'
            self._safe_addstr(stdscr, start_row + row_idx + 1, 0, line)

        self._safe_addstr(stdscr, start_row + self.shape[0] + 1, 0, border_h)
        return start_row + self.shape[0] + 3

    def _render_plan_selector_safe(self, stdscr, start_row: int, max_y: int) -> int:
        if start_row >= max_y - 1:
            return start_row
        if len(self.plans) > 1:
            plan = self.current_plan
            selector = f"Plan: [{self.current_plan_idx + 1}/{len(self.plans)}]"
            name = plan.get('name', '') if plan else ''
            self._safe_addstr(stdscr, start_row, 0, f"{selector}  {name}", curses.A_BOLD)

            target = plan.get('target') if plan else None
            if target is not None:
                self._safe_addstr(stdscr, start_row + 1, 0, f"Target: {target}")
                return start_row + 3
            return start_row + 2
        return start_row

    def _render_info_safe(self, stdscr, start_row: int, max_y: int) -> int:
        if start_row >= max_y - 1:
            return start_row
        current = self.get_current_state()
        steps = self.current_steps

        if steps:
            progress = f"Step: {self.current_step + 1}/{len(steps)}"
            self._safe_addstr(stdscr, start_row, 0, progress)

            bar_width = 30
            filled = int((self.current_step + 1) / len(steps) * bar_width)
            bar = '[' + '=' * filled + ' ' * (bar_width - filled) + ']'
            self._safe_addstr(stdscr, start_row + 1, 0, bar)

            if current is not None:
                state, action = current[0], current[1]
                pos = self.state_to_pos(state)

                info = (
                    f"State: {state} | Pos: {pos} | "
                    f"Depth: {pos[0]} | Cell: {_cell_name(*pos)}"
                )
                self._safe_addstr(stdscr, start_row + 3, 0, info)

                action_str = f"Action: {ACTION_ARROWS.get(action, '?')} {ACTION_NAMES.get(action, 'Unknown')}"
                self._safe_addstr(stdscr, start_row + 4, 0, action_str)

                extra_row = start_row + 5
                if len(current) >= 3 and extra_row < max_y - 1:
                    self._safe_addstr(stdscr, extra_row, 0, f"RM State: {current[2]}")
                    extra_row += 1
                if len(current) >= 4 and extra_row < max_y - 1:
                    self._safe_addstr(stdscr, extra_row, 0, f"Reward: {current[3]}")
                    extra_row += 1

                return extra_row + 1
        else:
            self._safe_addstr(stdscr, start_row, 0, "No plan loaded")
            return start_row + 2

        return start_row + 3

    def _render_controls_safe(self, stdscr, start_row: int, max_y: int):
        if start_row >= max_y - 1:
            return
        controls = "h/l:Step  j/k:Plan  r:Reset  q:Quit"
        self._safe_addstr(stdscr, start_row, 0, controls)

    # ------------------------------------------------------------------
    # Curses main loop
    # ------------------------------------------------------------------
    def run(self, stdscr):
        """Main visualization loop using curses."""
        curses.curs_set(0)
        stdscr.clear()

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()

            title = "PBST Visualizer  (Pressurized Bountiful Sea Treasure)"
            self._safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)

            row = self._render_map_safe(stdscr, start_row=2, max_y=max_y, max_x=max_x)
            row = self._render_plan_selector_safe(stdscr, start_row=row, max_y=max_y)
            row = self._render_info_safe(stdscr, start_row=row, max_y=max_y)
            self._render_controls_safe(stdscr, start_row=row, max_y=max_y)

            stdscr.refresh()

            key = stdscr.getch()
            if key == ord('q') or key == 27:
                break
            elif key == curses.KEY_RIGHT or key == ord('l'):
                self.step_forward()
            elif key == curses.KEY_LEFT or key == ord('h'):
                self.step_backward()
            elif key == curses.KEY_UP or key == ord('k'):
                self.next_plan()
            elif key == curses.KEY_DOWN or key == ord('j'):
                self.prev_plan()
            elif key == ord('r'):
                self.reset()

    # ------------------------------------------------------------------
    # Simple (no-curses) rendering
    # ------------------------------------------------------------------
    def run_simple(self):
        """Simple text-based visualization loop (no curses required)."""
        import os
        try:
            import termios
            import tty

            def getch():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'C': return 'RIGHT'
                            elif ch3 == 'D': return 'LEFT'
                            elif ch3 == 'A': return 'UP'
                            elif ch3 == 'B': return 'DOWN'
                        return 'ESC'
                    return ch
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except ImportError:
            def getch():
                return input("Command (h/l/j/k/r/q): ").strip()

        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            self._print_simple()
            key = getch()

            if key in ('q', 'Q', 'ESC'):
                break
            elif key in ('l', 'RIGHT'):
                self.step_forward()
            elif key in ('h', 'LEFT'):
                self.step_backward()
            elif key in ('k', 'UP'):
                self.next_plan()
            elif key in ('j', 'DOWN'):
                self.prev_plan()
            elif key == 'r':
                self.reset()

    def _print_simple(self):
        """Print the visualization without curses."""
        print("=" * 56)
        print("PBST Visualizer  (Pressurized Bountiful Sea Treasure)")
        print("=" * 56)
        print()

        current = self.get_current_state()
        agent_pos = None
        current_action = None

        if current is not None:
            state, action = current[0], current[1]
            agent_pos = self.state_to_pos(state)
            current_action = action

        # ---- Draw grid ----
        # Column index header
        col_header = '    ' + ' '.join(f'{c:1d}' for c in range(self.shape[1]))
        print(col_header)

        border_h = '+' + '-' * (self.shape[1] * 2 + 1) + '+'
        print(border_h)

        for row_idx in range(self.shape[0]):
            line = '| '
            for col_idx in range(self.shape[1]):
                if agent_pos is not None and (row_idx, col_idx) == agent_pos:
                    line += ACTION_ARROWS.get(current_action, '@')
                else:
                    line += _cell_symbol(row_idx, col_idx)
                line += ' '
            line += f'| {row_idx}'
            print(line)

        print(border_h)
        print()

        # ---- Legend ----
        print(f"Legend:  ~=shallow  \u2248=deep(row\u2265{DEEP_THRESHOLD})  \u2605=treasure  @/arrow=agent")
        print()

        # ---- Plan selector ----
        if len(self.plans) > 1:
            plan = self.current_plan
            print(f"Plan: [{self.current_plan_idx + 1}/{len(self.plans)}] {plan.get('name', '')}")
            target = plan.get('target') if plan else None
            if target is not None:
                print(f"Target: {target}")
            print()

        # ---- Step info ----
        steps = self.current_steps
        if steps:
            print(f"Step: {self.current_step + 1}/{len(steps)}")
            bar_width = 30
            filled = int((self.current_step + 1) / len(steps) * bar_width)
            bar = '[' + '=' * filled + ' ' * (bar_width - filled) + ']'
            print(bar)
            print()

            if current is not None:
                state, action = current[0], current[1]
                pos = self.state_to_pos(state)
                print(
                    f"State: {state} | Position: {pos} | "
                    f"Depth: {pos[0]} | Cell: {_cell_name(*pos)}"
                )
                print(f"Action: {ACTION_ARROWS.get(action, '?')} {ACTION_NAMES.get(action, 'Unknown')}")
                if len(current) >= 3:
                    print(f"RM State: {current[2]}")
                if len(current) >= 4:
                    print(f"Reward: {current[3]}")
        else:
            print("No plan loaded")
        print()

        # ---- Controls ----
        print("Controls: h/LEFT=Back  l/RIGHT=Forward  j/DOWN=PrevPlan  k/UP=NextPlan  r=Reset  q=Quit")


# ---------------------------------------------------------------------------
# Demo plan
# ---------------------------------------------------------------------------
def demo_plan() -> List[Tuple[int, int]]:
    """Generate a short demo plan (diagonal path toward a shallow treasure)."""
    # Starting at (0,0), move down then right toward (2,1) [treasure=80]
    plan = [
        (0,  DOWN),   # (0,0) -> (1,0)
        (10, RIGHT),   # (1,0) -> (1,1)
        (11, DOWN),  # (1,1) -> (2,1)  ← treasure 80
    ]
    return plan


# ---------------------------------------------------------------------------
# Default plan path
# ---------------------------------------------------------------------------
def get_default_plan_path() -> Path:
    project_root = Path(__file__).parent.parent
    return project_root / "last_plan_pbst.json"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='PBST Visualizer')
    parser.add_argument('--plan',   type=str,  help='Path to plan JSON file (default: last_plan_pbst.json)')
    parser.add_argument('--demo',   action='store_true', help='Run with demo plan')
    parser.add_argument('--simple', action='store_true', help='Use simple text mode (no curses)')
    args = parser.parse_args()

    visualizer = PBSTVisualizer()

    if args.demo:
        visualizer.load_plan(demo_plan())
    elif args.plan:
        visualizer.load_plans_from_file(args.plan)
    else:
        default_path = get_default_plan_path()
        if default_path.exists():
            print(f"Loading plans from {default_path}")
            visualizer.load_plans_from_file(str(default_path))
        else:
            print(f"No plan file found at {default_path}. Running with demo plan.")
            print("Use --plan FILE to load a custom plan, or run your experiment first.")
            visualizer.load_plan(demo_plan())
        print("Press Enter to continue...")
        input()

    if args.simple or not CURSES_AVAILABLE:
        visualizer.run_simple()
    else:
        curses.wrapper(visualizer.run)


if __name__ == '__main__':
    main()