"""
aircraft.py
-----------
Reduced-order 3D flight dynamics for the escort drones (see swarm.py). This
uses the standard "energy-state / bank-to-turn" model found in aircraft
performance textbooks (turn rate driven by bank angle and speed, climb rate
driven by pitch/flight-path angle, speed driven by a simple thrust-minus-drag
term).

The mothership used to be modeled with this same reduced-order Vehicle class,
but has since been swapped out for AeroBenchVV's actual 13-state nonlinear
F-16 model -- see f16_mothership.py. Drones stay on this reduced-order model.

State vector per vehicle:
    x, y, z       - position (m), z positive up
    heading (psi) - rad
    gamma         - flight path angle (rad)
    v             - true airspeed (m/s)
    roll (phi)    - bank angle (rad)
    pitch (theta) - approximate pitch attitude (rad), derived from gamma
"""

import numpy as np

G = 9.81


class Vehicle:
    def __init__(self, name, x, y, z, heading, v,
                 max_bank=np.radians(75), max_g=9.0,
                 roll_rate_limit=np.radians(180), max_accel=15.0,
                 min_speed=80.0, max_speed=650.0):
        self.name = name
        self.x, self.y, self.z = x, y, z
        self.heading = heading      # rad
        self.gamma = 0.0            # flight path angle, rad
        self.v = v                  # m/s
        self.roll = 0.0             # rad (bank angle)
        self.pitch = 0.0            # rad

        self.max_bank = max_bank
        self.max_g = max_g
        self.roll_rate_limit = roll_rate_limit
        self.max_accel = max_accel
        self.min_speed = min_speed
        self.max_speed = max_speed

        self.alive = True
        self.history = []  # list of (t, x, y, z, heading, roll, pitch, v)

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def velocity_vector(self):
        vx = self.v * np.cos(self.gamma) * np.cos(self.heading)
        vy = self.v * np.cos(self.gamma) * np.sin(self.heading)
        vz = self.v * np.sin(self.gamma)
        return np.array([vx, vy, vz])

    def step(self, dt, cmd_roll, cmd_gamma, cmd_accel):
        """
        cmd_roll   : commanded bank angle (rad), clipped to max_bank
        cmd_gamma  : commanded flight path angle (rad)
        cmd_accel  : commanded forward acceleration (m/s^2), clipped to max_accel
        """
        if not self.alive:
            return

        cmd_roll = np.clip(cmd_roll, -self.max_bank, self.max_bank)
        cmd_accel = np.clip(cmd_accel, -self.max_accel, self.max_accel)

        # Roll dynamics: first-order lag toward commanded bank, rate-limited
        roll_err = cmd_roll - self.roll
        roll_rate = np.clip(roll_err / max(dt, 1e-3), -self.roll_rate_limit, self.roll_rate_limit)
        self.roll += roll_rate * dt

        # Turn rate from coordinated bank-to-turn relation: psi_dot = g*tan(phi)/V
        # (standard result from level, coordinated-turn flight dynamics)
        turn_rate = G * np.tan(self.roll) / max(self.v, 1.0)
        # Respect g-limit on turn rate indirectly by capping bank angle already done
        self.heading += turn_rate * dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        # Flight path angle: first-order lag toward commanded gamma
        gamma_rate_limit = np.radians(60)
        gamma_err = cmd_gamma - self.gamma
        gamma_rate = np.clip(gamma_err / max(dt, 1e-3), -gamma_rate_limit, gamma_rate_limit)
        self.gamma += gamma_rate * dt
        self.pitch = self.gamma  # approximate: pitch ~= flight path angle + small AoA (ignored)

        # Speed dynamics
        self.v += cmd_accel * dt
        self.v = np.clip(self.v, self.min_speed, self.max_speed)

        vel = self.velocity_vector()
        self.x += vel[0] * dt
        self.y += vel[1] * dt
        self.z += vel[2] * dt
        self.z = max(self.z, 0.0)

    def log(self, t):
        self.history.append((t, self.x, self.y, self.z, self.heading, self.roll, self.pitch, self.v))


class Autopilot:
    """Simple waypoint/behavior autopilot producing (roll, gamma, accel) commands."""

    @staticmethod
    def fly_to_point(vehicle, target_xyz, desired_speed=None, max_bank=None):
        dx = target_xyz[0] - vehicle.x
        dy = target_xyz[1] - vehicle.y
        dz = target_xyz[2] - vehicle.z
        rng = np.hypot(dx, dy)

        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle.heading + np.pi) % (2 * np.pi) - np.pi

        bank_cmd = np.clip(2.5 * heading_err, -(max_bank or vehicle.max_bank), (max_bank or vehicle.max_bank))

        target_gamma = np.arctan2(dz, max(rng, 1.0))
        target_gamma = np.clip(target_gamma, np.radians(-30), np.radians(30))

        speed_target = desired_speed if desired_speed is not None else vehicle.v
        accel_cmd = np.clip((speed_target - vehicle.v) * 0.5, -vehicle.max_accel, vehicle.max_accel)

        return bank_cmd, target_gamma, accel_cmd

    @staticmethod
    def intercept_lead(vehicle, target_pos, target_vel, closing_gain=3.0):
        """Simple pursuit with lead: aim at predicted intercept point."""
        rel = target_pos - vehicle.pos()
        rng = np.linalg.norm(rel)
        t_go = rng / max(vehicle.v, 1.0)
        predicted = target_pos + target_vel * t_go * 0.5
        return Autopilot.fly_to_point(vehicle, predicted, desired_speed=vehicle.max_speed, max_bank=vehicle.max_bank)
