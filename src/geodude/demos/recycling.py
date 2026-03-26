"""Recycling demo — sort objects into bins.

4 objects on the worktop, 2 recycle bins. Try:
  sort_all()
  robot.pickup()
  robot.place("recycle_bin")
  chat('clear the table')
"""

scene = {
    "objects": {"can": 3, "potted_meat_can": 1, "recycle_bin": 2},
    "fixtures": {"recycle_bin": [[0.85, -0.35, 0.01], [-0.85, -0.35, 0.01]]},
}


def sort_all():
    """Pick up and place every object into a recycle bin."""
    while robot.pickup():  # noqa: F821 — robot is injected at load time
        if not robot.place("recycle_bin"):  # noqa: F821
            robot.go_home()  # noqa: F821
    robot.go_home()  # noqa: F821
