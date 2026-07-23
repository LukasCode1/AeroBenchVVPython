"""
f16_mothership.py
------------------
Mothership vehicle backed by AeroBenchVV's full 13-state nonlinear F-16
model (code/aerobench), replacing the reduced-order energy-state Vehicle
model that used to live in aircraft.py for this role.

F16Mothership exposes the same minimal read-only surface the rest of the
sim already consumes off a "Vehicle" (`.x`, `.y`, `.z`, `.heading`, `.v`,
`.alive`, `.pos()`, `.log()`/`.history`), so swarm.py and missile.py --
which only ever read position/velocity off the mothership -- work
unmodified.

Units: this sim (engagement.py/swarm.py/missile.py) works in meters,
meters/sec and radians. AeroBenchVV works in feet, feet/sec and radians.
All conversion happens at the boundary in this file.

Frame: world x <-> AeroBenchVV's POSN (north), world y <-> POSE (east).
This lines up exactly with AeroBenchVV's psi/POSN/POSE convention (psi=0
flies in +POSN, psi=+pi/2 flies in +POSE), so `heading` here is precisely
the F-16's yaw state -- no angle remapping needed.
"""

import os
import sys

_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np

from aerobench.run_f16_sim import F16SimState, SimModelError
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot
from aerobench.util import StateIndex

M_TO_FT = 3.280839895
FT_TO_M = 1.0 / M_TO_FT


class F16Mothership:
    """Straight/level ingress F-16, simulated with AeroBenchVV's nonlinear model.

    Steering is handled internally by AeroBenchVV's own WaypointAutopilot,
    aimed at a single far waypoint placed `ingress_distance_m` ahead of the
    start point along the initial heading -- this reproduces the old
    Vehicle's "always chase a point way out in front" straight-ingress
    behavior, but as a fixed world-frame waypoint (which is what
    AeroBenchVV's autopilot expects).
    """

    def __init__(self, name, x, y, z, heading, v, ingress_distance_m=150_000.0, step=1 / 30):
        self.name = name
        self.step_size = step

        alt_ft = z * M_TO_FT
        vt_ft = v * M_TO_FT

        # WaypointAutopilot wants (east, north, alt) tuples -- see get_waypoint_data()
        # in aerobench/examples/waypoint/waypoint_autopilot.py, confirmed against
        # run_waypoint.py's usage -- i.e. (POSE, POSN, ALT), the reverse of our
        # (x <-> POSN, y <-> POSE) mapping.
        waypoint_ft = (
            y * M_TO_FT + ingress_distance_m * M_TO_FT * np.sin(heading),
            x * M_TO_FT + ingress_distance_m * M_TO_FT * np.cos(heading),
            alt_ft,
        )

        self.ap = WaypointAutopilot([waypoint_ft], gain_str='old')
        self.ap.cfg_airspeed = vt_ft

        # Rough trim guess (small AoA, ~level pitch); the autopilot's Nz/roll/
        # airspeed loops close the gap from here over the first couple seconds.
        alpha_trim = np.deg2rad(1.8)

        init_state = np.zeros(13)
        init_state[StateIndex.VEL] = vt_ft
        init_state[StateIndex.ALPHA] = alpha_trim
        init_state[StateIndex.THETA] = alpha_trim
        init_state[StateIndex.PSI] = heading
        init_state[StateIndex.POSN] = x * M_TO_FT
        init_state[StateIndex.POSE] = y * M_TO_FT
        init_state[StateIndex.ALT] = alt_ft
        init_state[StateIndex.POW] = 8.0  # engine power level, 0-10

        self.sim = F16SimState(init_state, self.ap, step=step, extended_states=False,
                                print_errors=False)
        self.sim.init_simulation()

        self.alive = True
        self.history = []  # list of (t, x, y, z, heading, roll, pitch, v), same shape as Vehicle.log
        self._t = 0.0

    @property
    def _state(self):
        return self.sim.states[-1]

    @property
    def x(self):
        return self._state[StateIndex.POSN] * FT_TO_M

    @property
    def y(self):
        return self._state[StateIndex.POSE] * FT_TO_M

    @property
    def z(self):
        return self._state[StateIndex.ALT] * FT_TO_M

    @property
    def heading(self):
        return float(self._state[StateIndex.PSI])

    @property
    def roll(self):
        return float(self._state[StateIndex.PHI])

    @property
    def pitch(self):
        return float(self._state[StateIndex.THETA])

    @property
    def v(self):
        return self._state[StateIndex.VEL] * FT_TO_M

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def step(self, dt, *_ignored_autopilot_cmds):
        """Advance the F-16 by dt seconds.

        Accepts (and ignores) extra positional args so this is a drop-in
        replacement at call sites that used to pass reduced-order
        (bank, gamma, accel) commands -- AeroBenchVV's own waypoint
        autopilot drives the aircraft internally instead.
        """
        if not self.alive:
            return

        self._t += dt

        try:
            self.sim.simulate_to(self._t)
        except SimModelError:
            self.alive = False
            return

        if self.sim.integrator.status not in ('running', 'autopilot finished'):
            self.alive = False

        if self.z <= 0.0:
            self.alive = False

    def log(self, t):
        self.history.append((t, self.x, self.y, self.z, self.heading, self.roll, self.pitch, self.v))
