"""Simulation environment: parameters, terrain, victims, drones, perception, main loop.

This package contains everything that constitutes the simulated world:
configuration constants (``config``), the discretised terrain
(``grid_map``), victim trajectories (``target``), UAV state (``uav``),
the onboard detection model (``perception``), per-cell observation
history (``memory``), and the search-cycle main loop (``simulator``).

Agent policies live in the sibling ``agent`` package; experiment
drivers live in ``experiments``.
"""
