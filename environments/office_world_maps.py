"""
Office World map definitions and constants.

This module is shared between the environment and the visualizer
to avoid code duplication and keep maps in sync.
"""

# Action constants
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {UP: [-1, 0], RIGHT: [0, 1], DOWN: [1, 0], LEFT: [0, -1]}

# Map definitions
MAPS = {
    "default_office": [
        "oooWcooWoooWooo",
        "oDoooXoooXoooCo",
        "oooWoooWoooWooo",
        "WoWWWoWWWoWWWoW",
        "oooWoooWoooWooo",
        "oXoWosoWomoWoXo",
        "oooWoooWoooWooo",
        "WoWWWWWWWWWWWoW",
        "oooWoooWoocWooo",
        "oAoooXoooXoooBo",
        "oooWoooWoooWooo"
    ]
}
