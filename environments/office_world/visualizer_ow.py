#!/usr/bin/env python3
"""
Terminal-based visualizer for the Office World environment.

Usage:
    python3 visualizers/office_world.py [--plan PLAN_FILE]

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

# Add parent directory to path for imports
#sys.path.insert(0, str(Path(__file__).parent))

try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

import numpy as np

# Import shared map constants
from office_world_maps import UP, RIGHT, DOWN, LEFT, MAPS

# Visual representations for map elements
SYMBOLS = {
    'o': ' ',      # Open space
    'W': '\u2588', # Wall (full block)
    'X': '\u2715', # Decoration (X mark)
    'c': 'c', # Coffee (hot beverage) # \u2615
    's': '\u2302', # Office/Start (house)
    'm': '\u2709', # Mail (envelope)
    'A': 'A',      # Office A
    'B': 'B',      # Office B
    'C': 'C',      # Office C
    'D': 'D',      # Office D
}

# Action names
ACTION_NAMES = {
    UP: 'UP',
    RIGHT: 'RIGHT',
    DOWN: 'DOWN',
    LEFT: 'LEFT'
}

# Action arrows for display
ACTION_ARROWS = {
    UP: '\u2191',    # Up arrow
    RIGHT: '\u2192', # Right arrow
    DOWN: '\u2193',  # Down arrow
    LEFT: '\u2190'   # Left arrow
}


class OfficeWorldVisualizer:
    """Terminal-based visualizer for Office World environment."""

    def __init__(self, map_name: str = "default_office"):
        self.map_name = map_name
        self.map_data = MAPS[map_name]
        self.shape = (len(self.map_data), len(self.map_data[0]))
        self.plans: List[Dict[str, Any]] = []  # List of plan dicts
        self.current_plan_idx = 0
        self.current_step = 0
        self.rm_type = None  # 'policy' if rm_state tracks policy adherence

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat state index to (row, col) position."""
        return tuple(np.unravel_index(state, self.shape))

    def pos_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) position to flat state index."""
        return int(np.ravel_multi_index((row, col), self.shape))

    @property
    def current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current plan dict."""
        if 0 <= self.current_plan_idx < len(self.plans):
            return self.plans[self.current_plan_idx]
        return None

    @property
    def current_steps(self) -> List[Tuple[int, int]]:
        """Get the steps of the current plan."""
        plan = self.current_plan
        if plan is not None:
            return plan.get('steps', [])
        return []

    def load_plans(self, plans: List[Dict[str, Any]]):
        """
        Load multiple plans.

        Args:
            plans: List of plan dicts with 'steps', 'target', 'name' keys
        """
        self.plans = plans
        self.current_plan_idx = 0
        self.current_step = 0

    def load_plan(self, plan: List[Tuple[int, int]], name: str = "Plan 1"):
        """
        Load a single plan as a list of (state, action) tuples.

        Args:
            plan: List of (state, action) tuples representing the plan
            name: Name for this plan
        """
        self.plans = [{'steps': plan, 'target': None, 'name': name}]
        self.current_plan_idx = 0
        self.current_step = 0

    def load_plans_from_file(self, filepath: str):
        """Load plans from a JSON file (supports both old and new format)."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        info = None
        plan_list = None

        # Handle new format with info field
        if isinstance(data, dict) and 'plans' in data:
            info = data.get('info')
            plan_list = data['plans']
            # Update map if provided in info
            if info and 'map_name' in info:
                new_map = info['map_name']
                if new_map in MAPS:
                    self.map_name = new_map
                    self.map_data = MAPS[new_map]
                    self.shape = (len(self.map_data), len(self.map_data[0]))
            # Load rm_type from info
            if info and 'rm_type' in info:
                self.rm_type = info['rm_type']
            else:
                self.rm_type = None
        elif isinstance(data, list):
            plan_list = data

        if plan_list is None:
            self.plans = []
        elif len(plan_list) == 0:
            self.plans = []
        elif isinstance(plan_list[0], dict):
            # New format: list of plan dicts
            self.plans = []
            for plan_dict in plan_list:
                raw_steps = plan_dict.get('steps', [])
                steps = self._parse_steps(raw_steps)
                self.plans.append({
                    'steps': steps,
                    'target': plan_dict.get('target'),
                    'name': plan_dict.get('name', f'Plan {len(self.plans) + 1}')
                })
        else:
            # Old format: single plan as list of [state, action]
            self.plans = [{
                'steps': [tuple(item) for item in plan_list],
                'target': None,
                'name': 'Plan 1'
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
                # Extended format: {'state': int, 'action': int, 'rm_state': int, 'reward': float}
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
                # Simple format: [state, action]
                steps.append(tuple(item))
        return steps

    def get_current_state(self) -> Optional[Tuple[int, int]]:
        """Get the current (state, action) from the plan."""
        steps = self.current_steps
        if 0 <= self.current_step < len(steps):
            return steps[self.current_step]
        return None

    def step_forward(self):
        """Move one step forward in the plan."""
        if self.current_step < len(self.current_steps) - 1:
            self.current_step += 1

    def step_backward(self):
        """Move one step backward in the plan."""
        if self.current_step > 0:
            self.current_step -= 1

    def next_plan(self):
        """Switch to the next plan."""
        if self.current_plan_idx < len(self.plans) - 1:
            self.current_plan_idx += 1
            self.current_step = 0

    def prev_plan(self):
        """Switch to the previous plan."""
        if self.current_plan_idx > 0:
            self.current_plan_idx -= 1
            self.current_step = 0

    def reset(self):
        """Reset to the beginning of the current plan."""
        self.current_step = 0

    def render_map(self, stdscr, start_row: int = 2) -> int:
        """
        Render the map to the curses screen.

        Returns:
            The row number after the map
        """
        current = self.get_current_state()
        agent_pos = None
        current_action = None

        if current is not None:
            state, action = current[0], current[1]
            agent_pos = self.state_to_pos(state)
            current_action = action

        # Draw border
        border_h = '+' + '-' * (self.shape[1] * 2 + 1) + '+'
        stdscr.addstr(start_row, 0, border_h)

        for row_idx, row in enumerate(self.map_data):
            line = '| '
            for col_idx, cell in enumerate(row):
                if agent_pos is not None and (row_idx, col_idx) == agent_pos:
                    # Draw agent with action arrow
                    if current_action is not None:
                        line += ACTION_ARROWS.get(current_action, '@')
                    else:
                        line += '@'
                else:
                    line += SYMBOLS.get(cell, cell)
                line += ' '
            line += '|'
            stdscr.addstr(start_row + row_idx + 1, 0, line)

        stdscr.addstr(start_row + self.shape[0] + 1, 0, border_h)

        return start_row + self.shape[0] + 3

    def render_plan_selector(self, stdscr, start_row: int) -> int:
        """Render plan selector."""
        if len(self.plans) > 1:
            selector = f"Plan: [{self.current_plan_idx + 1}/{len(self.plans)}]"
            stdscr.addstr(start_row, 0, selector, curses.A_BOLD)

            plan = self.current_plan
            if plan is not None:
                name = plan.get('name', '')
                stdscr.addstr(start_row, len(selector) + 2, name)

                target = plan.get('target')
                if target is not None:
                    target_str = f"Target: {target}"
                    stdscr.addstr(start_row + 1, 0, target_str)
                    return start_row + 3

            return start_row + 2
        return start_row

    def render_info(self, stdscr, start_row: int) -> int:
        """Render plan information."""
        current = self.get_current_state()
        steps = self.current_steps

        # Plan progress
        if steps:
            progress = f"Step: {self.current_step + 1}/{len(steps)}"
            stdscr.addstr(start_row, 0, progress)

            # Progress bar
            bar_width = 30
            filled = int((self.current_step + 1) / len(steps) * bar_width)
            bar = '[' + '=' * filled + ' ' * (bar_width - filled) + ']'
            stdscr.addstr(start_row + 1, 0, bar)

            # Current state info
            if current is not None:
                state, action = current[0], current[1]
                pos = self.state_to_pos(state)
                cell = self.map_data[pos[0]][pos[1]]
                cell_name = self._get_cell_name(cell)

                info = f"State: {state} | Position: ({pos[0]}, {pos[1]}) | Cell: {cell_name}"
                stdscr.addstr(start_row + 3, 0, info)

                action_str = f"Action: {ACTION_ARROWS.get(action, '?')} {ACTION_NAMES.get(action, 'Unknown')}"
                stdscr.addstr(start_row + 4, 0, action_str)

                return start_row + 6
        else:
            stdscr.addstr(start_row, 0, "No plan loaded")
            return start_row + 2

        return start_row + 3

    def _get_cell_name(self, cell: str) -> str:
        """Get a human-readable name for a cell type."""
        names = {
            'o': 'Open',
            'W': 'Wall',
            'X': 'Decoration',
            'c': 'Coffee',
            's': 'Office',
            'm': 'Mail',
            'A': 'Office A',
            'B': 'Office B',
            'C': 'Office C',
            'D': 'Office D',
        }
        return names.get(cell, cell)

    def render_legend(self, stdscr, start_row: int) -> int:
        """Render the map legend."""
        stdscr.addstr(start_row, 0, "Legend:")
        legend_items = [
            (SYMBOLS['W'], 'Wall'),
            (SYMBOLS['X'], 'Decoration'),
            (SYMBOLS['c'], 'Coffee'),
            (SYMBOLS['s'], 'Office'),
            (SYMBOLS['m'], 'Mail'),
            ('@', 'Agent'),
        ]

        col = 0
        for symbol, name in legend_items:
            item = f"  {symbol}={name}"
            stdscr.addstr(start_row + 1, col, item)
            col += len(item) + 2

        return start_row + 3

    def render_controls(self, stdscr, start_row: int):
        """Render control instructions."""
        stdscr.addstr(start_row, 0, "Controls:")
        controls_line1 = "  \u2190/h: Back  \u2192/l: Forward  r: Reset  q: Quit"
        stdscr.addstr(start_row + 1, 0, controls_line1)

        if len(self.plans) > 1:
            controls_line2 = "  \u2191/k: Next plan  \u2193/j: Prev plan"
            stdscr.addstr(start_row + 2, 0, controls_line2)

    def _safe_addstr(self, stdscr, row: int, col: int, text: str, attr=0):
        """Safely add string to screen, ignoring if out of bounds."""
        max_y, max_x = stdscr.getmaxyx()
        if row < max_y - 1 and col < max_x:
            try:
                # Truncate text if too long
                available_width = max_x - col - 1
                if len(text) > available_width:
                    text = text[:available_width]
                stdscr.addstr(row, col, text, attr)
            except curses.error:
                pass  # Ignore curses errors

    def run(self, stdscr):
        """Main visualization loop using curses."""
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()

            # Title
            title = "Office World Visualizer"
            self._safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)

            # Render components with bounds checking
            row = self._render_map_safe(stdscr, start_row=2, max_y=max_y, max_x=max_x)
            row = self._render_plan_selector_safe(stdscr, start_row=row, max_y=max_y)
            row = self._render_info_safe(stdscr, start_row=row, max_y=max_y)
            self._render_controls_safe(stdscr, start_row=row, max_y=max_y)

            stdscr.refresh()

            # Handle input
            key = stdscr.getch()

            if key == ord('q') or key == 27:  # q or Escape
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

    def _render_map_safe(self, stdscr, start_row: int, max_y: int, max_x: int) -> int:
        """Render map with bounds checking."""
        current = self.get_current_state()
        agent_pos = None
        current_action = None

        if current is not None:
            state, action = current[0], current[1]
            agent_pos = self.state_to_pos(state)
            current_action = action

        border_h = '+' + '-' * (self.shape[1] * 2 + 1) + '+'
        self._safe_addstr(stdscr, start_row, 0, border_h)

        for row_idx, row in enumerate(self.map_data):
            if start_row + row_idx + 1 >= max_y - 1:
                break
            line = '| '
            for col_idx, cell in enumerate(row):
                if agent_pos is not None and (row_idx, col_idx) == agent_pos:
                    if current_action is not None:
                        line += ACTION_ARROWS.get(current_action, '@')
                    else:
                        line += '@'
                else:
                    line += SYMBOLS.get(cell, cell)
                line += ' '
            line += '|'
            self._safe_addstr(stdscr, start_row + row_idx + 1, 0, line)

        self._safe_addstr(stdscr, start_row + self.shape[0] + 1, 0, border_h)
        return start_row + self.shape[0] + 3

    def _render_plan_selector_safe(self, stdscr, start_row: int, max_y: int) -> int:
        """Render plan selector with bounds checking."""
        if start_row >= max_y - 1:
            return start_row
        if len(self.plans) > 1:
            plan = self.current_plan
            selector = f"Plan: [{self.current_plan_idx + 1}/{len(self.plans)}]"
            name = plan.get('name', '') if plan else ''
            self._safe_addstr(stdscr, start_row, 0, f"{selector}  {name}", curses.A_BOLD)
            return start_row + 2
        return start_row

    def _render_info_safe(self, stdscr, start_row: int, max_y: int) -> int:
        """Render step info with bounds checking."""
        if start_row >= max_y - 1:
            return start_row
        current = self.get_current_state()
        steps = self.current_steps

        if steps:
            progress = f"Step: {self.current_step + 1}/{len(steps)}"
            self._safe_addstr(stdscr, start_row, 0, progress)

            if current is not None and start_row + 1 < max_y - 1:
                state, action = current[0], current[1]
                pos = self.state_to_pos(state)
                info = f"State: {state} | Pos: {pos} | Action: {ACTION_ARROWS.get(action, '?')} {ACTION_NAMES.get(action, '?')}"
                self._safe_addstr(stdscr, start_row + 1, 0, info)

                extra_row = start_row + 2
                # Display RM state if available
                if len(current) >= 3 and extra_row < max_y - 1:
                    rm_state = current[2]
                    rm_info = f"RM State: {rm_state}"
                    self._safe_addstr(stdscr, extra_row, 0, rm_info)
                    extra_row += 1

                # Display reward if available
                if len(current) >= 4 and extra_row < max_y - 1:
                    reward = current[3]
                    reward_info = f"Task RM Reward: {reward}"
                    self._safe_addstr(stdscr, extra_row, 0, reward_info)
                    extra_row += 1

                return extra_row + 1
        return start_row + 2

    def _render_controls_safe(self, stdscr, start_row: int, max_y: int):
        """Render controls with bounds checking."""
        if start_row >= max_y - 1:
            return
        controls = "h/l:Step  j/k:Plan  r:Reset  q:Quit"
        self._safe_addstr(stdscr, start_row, 0, controls)

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
                    # Handle arrow keys (escape sequences)
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
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')

            # Print visualization
            self._print_simple()

            # Get input
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
        print("=" * 40)
        print("Office World Visualizer")
        print("=" * 40)
        print()

        current = self.get_current_state()
        agent_pos = None
        current_action = None

        if current is not None:
            state, action = current[0], current[1]
            agent_pos = self.state_to_pos(state)
            current_action = action

        # Draw map
        border_h = '+' + '-' * (self.shape[1] * 2 + 1) + '+'
        print(border_h)

        for row_idx, row in enumerate(self.map_data):
            line = '| '
            for col_idx, cell in enumerate(row):
                if agent_pos is not None and (row_idx, col_idx) == agent_pos:
                    if current_action is not None:
                        line += ACTION_ARROWS.get(current_action, '@')
                    else:
                        line += '@'
                else:
                    line += SYMBOLS.get(cell, cell)
                line += ' '
            line += '|'
            print(line)

        print(border_h)
        print()

        # Plan selector
        if len(self.plans) > 1:
            plan = self.current_plan
            print(f"Plan: [{self.current_plan_idx + 1}/{len(self.plans)}] {plan.get('name', '')}")
            target = plan.get('target') if plan else None
            if target is not None:
                print(f"Target: {target}")
            print()

        # Step info
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
                cell = self.map_data[pos[0]][pos[1]]
                print(f"State: {state} | Position: {pos} | Cell: {self._get_cell_name(cell)}")
                print(f"Action: {ACTION_ARROWS.get(action, '?')} {ACTION_NAMES.get(action, 'Unknown')}")
                # Display RM state if available
                if len(current) >= 3:
                    print(f"RM State: {current[2]}")
                # Display reward if available
                if len(current) >= 4:
                    print(f"Task RM Reward: {current[3]}")
        else:
            print("No plan loaded")
        print()

        # Controls
        print("Controls: h/LEFT=Back  l/RIGHT=Forward  j/DOWN=PrevPlan  k/UP=NextPlan  r=Reset  q=Quit")


def demo_plan() -> List[Tuple[int, int]]:
    """Generate a demo plan for testing."""
    # Start from state 137 (position 9,2 - near office A)
    # Move around to demonstrate
    plan = [
        (137, UP),      # Move up
        (122, UP),      # Move up
        (107, RIGHT),   # Move right
        (108, RIGHT),   # Move right
        (109, UP),      # Move up
        (94, UP),       # Move up
        (79, UP),       # Move up
        (64, UP),       # Move up
        (49, UP),       # Move up
        (34, UP),       # Move up
        (19, LEFT),     # Move left
        (18, LEFT),     # Move left
        (17, LEFT),     # Move left - arrive at coffee
    ]
    return plan


def get_default_plan_path() -> Path:
    """Get the default path to last_plan.json."""
    # Look in the project root (parent of visualizers)
    project_root = Path(__file__).parent.parent
    return project_root / "last_plan.json"


def main():
    parser = argparse.ArgumentParser(description='Office World Visualizer')
    parser.add_argument('--plan', type=str, help='Path to plan JSON file (default: last_plan.json)')
    parser.add_argument('--demo', action='store_true', help='Run with demo plan')
    parser.add_argument('--simple', action='store_true', help='Use simple text mode (no curses)')
    parser.add_argument('--map', type=str, default='default_office', help='Map name to use')
    args = parser.parse_args()

    visualizer = OfficeWorldVisualizer(map_name=args.map)

    if args.demo:
        visualizer.load_plan(demo_plan())
    elif args.plan:
        visualizer.load_plans_from_file(args.plan)
    else:
        # Try to load last_plan.json by default
        default_path = get_default_plan_path()
        if default_path.exists():
            print(f"Loading plans from {default_path}")
            visualizer.load_plans_from_file(str(default_path))
        else:
            # Fall back to demo if no plan file exists
            print(f"No plan file found at {default_path}. Running with demo plan.")
            print("Use --plan FILE to load a custom plan, or run your experiment first.")
            visualizer.load_plan(demo_plan())
        print("Press Enter to continue...")
        input()

    # Run the visualization
    if args.simple or not CURSES_AVAILABLE:
        visualizer.run_simple()
    else:
        curses.wrapper(visualizer.run)


if __name__ == '__main__':
    main()
