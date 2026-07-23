"""
missile.py
----------
Generic surface-to-air missile (SAM) model using Proportional Navigation (PN),
the standard, publicly-published guidance law:

    a_cmd = N' * Vc * lambda_dot

where lambda_dot is the line-of-sight rotation rate, Vc is closing velocity,
and N' is the navigation constant (typically 3-5). This is textbook material
(see Zarchan, "Tactical and Strategic Missile Guidance") and is NOT modeling
any specific real-world system's seeker, airframe, or classified parameters.

The missile also has a generic "seeker" that can be defeated probabilistically
by drone countermeasures (decoy lure, RF/IR jamming, or a physical body-block
that spoofs closing geometry). These are abstracted as probability modifiers,
not real electronic-warfare techniques.
"""

import numpy as np

G = 9.81


class GenericSAM:
    def __init__(self, name, x, y, z, v=900.0, nav_constant=4.0,
                 max_accel_g=25.0, seeker_fov=np.radians(60), seeker_range=25000.0):
        self.name = name
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.speed = v
        self.N = nav_constant
        self.max_accel = max_accel_g * G
        self.seeker_fov = seeker_fov
        self.seeker_range = seeker_range

        self.target = None          # currently locked Vehicle-like object
        self.locked = False
        self.alive = True
        self.detonated = False
        self.miss_distance = None
        self.history = []

        self._prev_los = None

    def acquire(self, candidates, jam_prob_fn=None):
        """
        candidates: list of vehicles (aircraft/drones) with .pos() and alive
        jam_prob_fn: optional fn(candidate) -> probability this candidate's
                     jamming/decoy denies lock this tick
        Chooses nearest valid target within seeker range/FOV; can be spoofed
        onto a decoy drone if jamming succeeds.
        """
        best = None
        best_rng = np.inf
        for c in candidates:
            if not getattr(c, "alive", True):
                continue
            rel = c.pos() - self.pos
            rng = np.linalg.norm(rel)
            if rng > self.seeker_range:
                continue
            los_dir = rel / max(rng, 1e-6)
            fwd = self.vel / max(np.linalg.norm(self.vel), 1e-6) if np.linalg.norm(self.vel) > 1 else los_dir
            ang = np.arccos(np.clip(np.dot(los_dir, fwd), -1, 1))
            if ang > self.seeker_fov / 2:
                continue
            if rng < best_rng:
                best_rng = rng
                best = c

        if best is None:
            self.locked = False
            self.target = None
            return

        # Countermeasure roll: jam/decoy can deny or redirect lock
        if jam_prob_fn is not None:
            p_deny = jam_prob_fn(best)
            if np.random.rand() < p_deny:
                self.locked = False
                self.target = None
                return

        self.target = best
        self.locked = True

    def step(self, dt):
        if not self.alive or self.detonated:
            return
        if not self.locked or self.target is None or not getattr(self.target, "alive", True):
            # ballistic / lost-lock coast
            self.pos += self.vel * dt
            return

        rel = self.target.pos() - self.pos
        rng = np.linalg.norm(rel)
        los = rel / max(rng, 1e-6)

        if self._prev_los is None:
            self._prev_los = los

        # Angular rate of LOS vector (finite-difference)
        los_rate_vec = (los - self._prev_los) / max(dt, 1e-3)
        self._prev_los = los

        closing_speed = self.speed + getattr(self.target, "v", 0.0)
        # PN command: lateral accel proportional to LOS rotation rate * closing velocity
        accel_cmd = self.N * closing_speed * los_rate_vec
        accel_cmd = np.clip(accel_cmd, -self.max_accel, self.max_accel)

        # Update velocity: turn toward target while holding speed roughly constant
        vel_dir = self.vel / max(np.linalg.norm(self.vel), 1e-6) if np.linalg.norm(self.vel) > 1 else los
        new_vel_dir = vel_dir + accel_cmd * dt / self.speed
        new_vel_dir = new_vel_dir / max(np.linalg.norm(new_vel_dir), 1e-6)
        self.vel = new_vel_dir * self.speed

        self.pos += self.vel * dt

        self.miss_distance = rng
        if rng < 15.0:  # generic lethal-radius proxy
            self.detonated = True
            self.target.alive = False

    def log(self, t):
        self.history.append((t, *self.pos, self.locked, self.target.name if self.target else None))
