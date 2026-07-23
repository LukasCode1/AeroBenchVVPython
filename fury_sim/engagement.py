"""
engagement.py
-------------
Ties the mothership (flown with AeroBenchVV's actual 13-state nonlinear
F-16 model, see f16_mothership.py), drone swarm, and a generic SAM threat
together into a single timestepped engagement, and records outcome metrics
for research/statistics.
"""

import numpy as np
from f16_mothership import F16Mothership
from swarm import Drone, swarm_policy_manual, command_drone, jam_probability_fn
from missile import GenericSAM


class EngagementResult:
    def __init__(self):
        self.mothership_survived = None
        self.n_drones_lost = 0
        self.time_to_resolution = None
        self.min_missile_miss_distance = None
        self.roles_over_time = []


def run_engagement(n_drones=3, sam_range=20000.0, sam_offset_angle=0.0,
                    dt=0.05, t_max=90.0, policy=swarm_policy_manual,
                    denial_radius=1200.0, seed=None, log_trajectories=False):
    """
    Runs one engagement:
      - mothership flies a straight/level ingress
      - a SAM launches from ground, sam_range meters away, offset by
        sam_offset_angle radians from nose-on
      - n_drones escort drones execute the swarm policy each tick
    Returns an EngagementResult.
    """
    if seed is not None:
        np.random.seed(seed)

    mothership = F16Mothership("F16", x=0.0, y=0.0, z=6000.0, heading=0.0, v=230.0, step=dt)

    drones = [Drone(f"drone_{i}", x=-200.0 - 50 * i, y=(-1)**i * 150.0, z=6000.0,
                     heading=0.0, v=230.0) for i in range(n_drones)]

    sam_x = sam_range * np.cos(sam_offset_angle)
    sam_y = sam_range * np.sin(sam_offset_angle)
    sam = GenericSAM("sam_1", x=sam_x, y=sam_y, z=0.0)
    # initial launch velocity aimed roughly at mothership
    aim = mothership.pos() - sam.pos
    sam.vel = aim / np.linalg.norm(aim) * sam.speed

    result = EngagementResult()
    t = 0.0
    min_miss = np.inf

    while t < t_max:
        # 1. mothership: straight-and-level ingress, flown by AeroBenchVV's own
        # 13-state nonlinear F-16 model + waypoint autopilot (see f16_mothership.py)
        mothership.step(dt)

        # 2. swarm role assignment + commands
        threats = [sam] if sam.alive and not sam.detonated else []
        roles = policy(drones, mothership, threats)
        result.roles_over_time.append(dict(roles))

        threat = threats[0] if threats else None
        for i, d in enumerate(drones):
            if not d.alive:
                continue
            role = roles.get(d.name, "ESCORT")
            b, g, a = command_drone(d, role, mothership, threat, i, n_drones)
            d.step(dt, b, g, a)

        # 3. SAM seeker acquisition with countermeasures
        candidates = [mothership] + [d for d in drones if d.alive]
        jam_fn_builder = jam_probability_fn(drones, roles, denial_radius=denial_radius)

        def jam_prob_fn(cand):
            return jam_fn_builder(cand, sam.pos)

        if sam.alive and not sam.detonated:
            sam.acquire(candidates, jam_prob_fn=jam_prob_fn)
            sam.step(dt)
            if sam.miss_distance is not None:
                min_miss = min(min_miss, sam.miss_distance)

        # 4. INTERCEPT drones can also "body-block" the missile directly:
        # if an intercept drone gets within lethal-radius proxy of missile, treat as hard-kill.
        for d in drones:
            if not d.alive:
                continue
            if roles.get(d.name) == "INTERCEPT" and sam.alive and not sam.detonated:
                if np.linalg.norm(d.pos() - sam.pos) < 20.0:
                    sam.alive = False
                    sam.detonated = True

        if log_trajectories:
            mothership.log(t)
            for d in drones:
                d.log(t)
            sam.log(t)

        t += dt

        if sam.detonated or not sam.alive:
            break
        if not mothership.alive:
            break

    result.mothership_survived = mothership.alive
    result.n_drones_lost = sum(1 for d in drones if not d.alive)
    result.time_to_resolution = t
    result.min_missile_miss_distance = None if min_miss == np.inf else min_miss
    return result, mothership, drones, sam
