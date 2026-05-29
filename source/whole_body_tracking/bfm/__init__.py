"""Whole-body tracking package.

Import ``bfm.tasks`` explicitly when Gym task registration is needed. Keeping
the package root light lets config modules be imported before IsaacSim/AppLauncher
has initialized the simulator runtime.
"""
