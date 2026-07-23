# Fury Style Escort Swarm Engagement Simulation

A simulation testbed for studying whether an autonomous escort drone swarm
improves the survivability of a manned strike aircraft against a single
surface to air missile shot, and how that effect scales with swarm size and
role allocation. The manned aircraft (the "mothership") is flown with
AeroBenchVV's actual nonlinear F 16 flight dynamics model, so the ingress
trajectory this study is built on is the same six degree of freedom aircraft
model used elsewhere in this repository, not a simplified stand in.

## Research question

Given a fixed missile threat and a fixed ingress profile, does adding escort
drones in defined roles (decoy, jammer, hard kill interceptor) change the
probability that the mothership survives, and is there a point of
diminishing or negative returns as swarm size grows?

## Animated example

![Engagement animation](engagement_anim3d.gif)

The clip above is produced by `animate_engagement.py`. Black marker and
trail: the mothership. Colored markers: escort drones. Red triangle: the
inbound SAM. The default scenario ends with the drone closing to intercept
range and neutralizing the missile before it reaches the mothership; the
text overlay in the final frames states the actual outcome explicitly
(see "Reading the animation" below for why that label is necessary).

## Model summary

| Entity          | Dynamics model                                                   | Fidelity |
|------------------|-------------------------------------------------------------------|----------|
| Mothership       | AeroBenchVV 13 state nonlinear F 16 model, LQR inner loop, waypoint outer loop autopilot | Full nonlinear ODE, integrated with `scipy.integrate.RK45` |
| Escort drones    | Energy state / bank to turn point mass model (turn rate from bank angle and speed, climb rate from flight path angle, first order roll and speed lag) | Reduced order, closed form update |
| SAM              | Proportional navigation guidance (Zarchan, *Tactical and Strategic Missile Guidance*), generic seeker with field of view, range, and lock/jam probability | Reduced order, textbook guidance law |

The mothership gets the higher fidelity treatment deliberately: this study
is about the survivability of that one aircraft, so its trajectory should
come from the same validated nonlinear model used for verification work
elsewhere in this repository. The drones and the missile use reduced order
models so that thousands of Monte Carlo trials remain computationally
tractable.

None of these models represent the seeker, airframe, or guidance parameters
of any specific real world system. The SAM guidance law and aerodynamic
model are both standard, publicly published textbook material.

## Repository layout

| File                     | Role |
|--------------------------|------|
| `f16_mothership.py`      | Wraps `code/aerobench` (AeroBenchVV) as the mothership. Converts between this project's meters and AeroBenchVV's feet, and exposes position, heading, and velocity in the form the rest of the simulation expects. |
| `aircraft.py`            | Reduced order `Vehicle` flight model, used only for the escort drones. |
| `swarm.py`               | `Drone` class and swarm role logic: escort, decoy, jammer, interceptor. Role assignment is a pluggable policy function. |
| `missile.py`             | `GenericSAM`: proportional navigation guidance, seeker acquisition, and a probabilistic countermeasure denial model. |
| `engagement.py`          | `run_engagement()`: ties one mothership, one swarm, and one SAM together into a single timestepped run and returns outcome statistics. |
| `run_experiment.py`      | Monte Carlo harness. Sweeps swarm size, aggregates survival statistics, and writes CSVs and summary plots. |
| `animate_engagement.py`  | Renders one engagement as an animated 3D GIF, with an explicit outcome label (see below). |

## Requirements

Python 3 with `numpy`, `scipy`, `matplotlib`, `pandas`, and `Pillow`. No
separate install step is needed for the AeroBenchVV dependency:
`f16_mothership.py` adds `code/` (relative to this directory) to `sys.path`
at import time.

## Running the Monte Carlo experiment

```
cd fury_sim
python run_experiment.py
```

This runs 60 trials at each of several swarm sizes (0, 1, 2, 3, 4, 6, and 8
drones), with the SAM launched from 20 km at a random offset angle each
trial. It writes `output/experiment_results.csv` (one row per trial),
`output/experiment_summary.csv` (aggregated by swarm size), and two plots.

The script prints nothing until the full sweep finishes. Because the
mothership is now a genuine nonlinear ODE simulation rather than a closed
form update, a full sweep (420 trials) takes a few minutes on a typical
laptop; this is expected, not a hang.

### Sample results

From a 60 trial per swarm size run against a SAM launched at 20 km, offset
uniformly between -25 and +25 degrees off nose on:

| Escort drones | Survival rate | Mean miss distance (m) |
|---------------|---------------|--------------------------|
| 0             | 0.82          | 31.6 |
| 1             | 0.85          | 28.4 |
| 2             | 0.72          | 27.3 |
| 3             | 0.82          | 63.0 |
| 4             | 0.82          | 64.3 |
| 6             | 0.83          | 61.0 |
| 8             | 0.93          | 75.7 |

No drone was lost in any trial in this run. These numbers will vary between
runs since trial seeds are not fixed in the sweep; treat this table as
illustrative of the kind of output the harness produces, not as a final
result.

![Sample trajectory](output/sample_trajectory.png)
![Survival vs swarm size](output/survival_vs_swarm_size.png)

## Rendering a 3D engagement animation

```
cd fury_sim
python animate_engagement.py
```

Produces `engagement_anim3d.gif`. Useful options:

| Flag | Meaning | Default |
|------|---------|---------|
| `--n-drones` | escort swarm size | 1 |
| `--seed` | trial seed | 1 |
| `--sam-offset-deg` | SAM launch angle off nose on | 0 |
| `--max-frames` | rendered frame budget (lower renders faster) | 200 |
| `--hold-seconds` | how long the final frame freezes before the loop restarts | 2.5 |
| `--elev`, `--azim` | 3D camera angle | 25, -60 |
| `--fps` | playback frame rate | 20 |

The default `n_drones=1, seed=1` scenario was chosen because it actually
resolves, with the drone reaching intercept range and neutralizing the SAM
around t=30s. With three or more escort drones, the SAM frequently loses
lock under sustained jamming and decoy pressure and coasts, unresolved, for
the rest of the 90 second engagement window. That is a real property of the
current SAM model, not a rendering bug, but it produces a far less
legible animation, so it was not used as the default.

## Reading the animation

At the scale of this simulation, the SAM closes over tens of kilometers
while the escort formation holds a spacing of only a few hundred meters.
That means an intercepted SAM and an actual hit on the mothership look
almost identical: both end with the SAM marker converging on the same small
cluster of aircraft markers. `animate_engagement.py` resolves this
ambiguity directly rather than leaving it to marker proximity. Once an
engagement ends, it checks which entity, if any, actually has its `alive`
flag cleared and prints and overlays one of:

- `Mothership HIT, destroyed`
- `<drone name> HIT by SAM, mothership safe`
- `SAM neutralized (intercepted), no casualties`
- `engagement timed out, SAM never resolved, mothership safe`

whichever entity was actually destroyed is marked with a red X for the
remaining frames.

## Known modeling limitations

- The SAM's countermeasure denial model (decoy pull, jamming) is a tunable
  probability function, not a model of a real seeker, waveform, or IR
  signature.
- An interceptor drone that body blocks the SAM destroys it at no modeled
  cost to the drone itself. This is a deliberate simplification, not an
  oversight.
- Once the SAM's seeker loses lock, it coasts ballistically on its last
  known heading and keeps attempting reacquisition, but in practice rarely
  succeeds once the geometry has diverged. Survival statistics for larger
  swarm sizes should be read with this in mind: many of those trials end in
  an unresolved time out rather than a defeated missile.
- The escort drones use a reduced order flight model, so their maneuvering
  limits are approximate, not derived from a specific airframe.

## Attribution

This project is built on top of AeroBenchVV, the F 16 verification
benchmark in the rest of this repository. See the top level `README.md` and
`LICENSE` for citation information and license terms.
