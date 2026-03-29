"""Recycling demo — sort a random mix of objects into bins.

Pool of boxes, cans, and bottles. 5 random objects spawned each run. Try:
  sort_all()
  robot.pickup()
  robot.place("recycle_bin")
  chat('clear the table')
"""

import random

# Pre-allocate 3 of each type (pool). Only `spawn_count` are activated per run.
scene = {
    "objects": {
        "can": 3,
        "spam_can": 3,
        "cracker_box": 3,
        "gelatin_box": 3,
        "sugar_box": 3,
        "pocky_box": 3,
        "fuze_bottle": 3,
        "pop_tarts_case": 3,
        "plastic_glass": 3,
        "recycle_bin": 2,
    },
    "fixtures": {"recycle_bin": [[0.85, -0.35, 0.01], [-0.85, -0.35, 0.01]]},
    "spawn_count": 5,
}

# Object types that can be spawned (everything except fixtures)
_GRASPABLE_TYPES = [
    k for k in scene["objects"] if k != "recycle_bin"
]


def sort_all():
    """Pick up and place every object into a recycle bin."""
    while robot.pickup():
        if not robot.place("recycle_bin"):
            robot.go_home()
    robot.go_home()
