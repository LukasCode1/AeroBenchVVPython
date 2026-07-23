"""
animate_engagement.py
----------------------
Renders a 3D animated GIF of one engagement: the mothership (flown with
AeroBenchVV's actual F-16 model), the escort drone swarm, and a generic SAM
threat, all moving together over time.

Reuses engagement.run_engagement() (with log_trajectories=True) rather than
re-implementing the tick loop, so this stays in sync with whatever
engagement.py/swarm.py/missile.py actually do. The per-tick position
histories it records are all this script needs: the engagement loop always
breaks on the same tick any entity is killed (see run_engagement), so there
are no stale "frozen after death" frames to filter out -- the last rendered
frame is simply the moment of impact/resolution.

Usage:
    python animate_engagement.py [output.gif] [--n-drones N] [--seed S]
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from matplotlib import animation

from engagement import run_engagement

MOTHERSHIP_COLOR = "black"
SAM_COLOR = "tab:red"
DRONE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown",
                "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("filename", nargs="?", default="engagement_anim3d.gif",
                    help="output .gif path (default: engagement_anim3d.gif)")
    # defaults picked to land on a scenario that actually resolves (drone_0
    # intercepts/body-blocks the SAM around t=30s) rather than the more
    # common case where the SAM loses lock and just coasts for the full
    # engagement -- see missile.py's "ballistic / lost-lock coast" branch.
    # That's a real characteristic of the existing SAM model, not a bug here,
    # but it makes for a badly-scaled, less illustrative 3D animation.
    p.add_argument("--n-drones", type=int, default=1)
    p.add_argument("--sam-range", type=float, default=20000.0)
    p.add_argument("--sam-offset-deg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=200,
                    help="rendered frame budget; the tick history is downsampled to roughly this many frames")
    p.add_argument("--trail-seconds", type=float, default=8.0,
                    help="how many seconds of recent trail to draw behind each moving marker")
    p.add_argument("--elev", type=float, default=25.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--hold-seconds", type=float, default=2.5,
                    help="how long to freeze on the final frame (with the outcome label) before the gif loops")
    return p.parse_args()


def determine_outcome(mothership, drones, sam):
    '''
    Figure out, in plain terms, what actually happened -- and who (if anyone)
    was actually destroyed -- so the animation can say so explicitly instead
    of leaving the viewer to guess from marker proximity. At this sim's
    scale (SAM closing over tens of km, formation spacing a few hundred m),
    an intercept/near-miss and an actual hit look nearly identical -- both
    end with the SAM marker converging on the mothership/drone cluster.
    '''

    if not mothership.alive:
        return "OUTCOME: Mothership HIT -- destroyed", "mothership"

    dead_drones = [d.name for d in drones if not d.alive]
    if dead_drones:
        return f"OUTCOME: {', '.join(dead_drones)} HIT by SAM -- mothership safe", dead_drones[0]

    if sam.detonated or not sam.alive:
        return "OUTCOME: SAM neutralized (intercepted) -- no casualties", None

    return "OUTCOME: engagement timed out, SAM never resolved -- mothership safe", None


def simulate(args):
    result, mothership, drones, sam = run_engagement(
        n_drones=args.n_drones,
        sam_range=args.sam_range,
        sam_offset_angle=np.radians(args.sam_offset_deg),
        seed=args.seed,
        log_trajectories=True,
    )

    outcome_text, destroyed_name = determine_outcome(mothership, drones, sam)

    print(f"survived={result.mothership_survived}, drones_lost={result.n_drones_lost}, "
          f"t_res={result.time_to_resolution:.2f}s, min_miss={result.min_missile_miss_distance}")
    print(outcome_text)

    mh = np.array(mothership.history, dtype=float)  # t, x, y, z, heading, roll, pitch, v
    dhs = [np.array(d.history, dtype=float) for d in drones]
    drone_names = [d.name for d in drones]

    sh_raw = sam.history  # (t, x, y, z, locked, target_name) -- mixed dtypes
    sh_t = np.array([row[0] for row in sh_raw], dtype=float)
    sh_xyz = np.array([row[1:4] for row in sh_raw], dtype=float)

    return mh, dhs, drone_names, sh_t, sh_xyz, result, outcome_text, destroyed_name


def downsample_indices(n, max_frames):
    step = max(1, n // max_frames)
    return np.arange(0, n, step)


def main():
    args = parse_args()
    mh, dhs, drone_names, sh_t, sh_xyz, result, outcome_text, destroyed_name = simulate(args)

    idx = downsample_indices(len(mh), args.max_frames)
    n_real_frames = len(idx)
    hold_frames = int(args.hold_seconds * args.fps)
    n_frames = n_real_frames + hold_frames

    dt = mh[1, 0] - mh[0, 0] if len(mh) > 1 else 0.05
    trail_len = max(1, int(args.trail_seconds / dt) // max(1, idx[1] - idx[0] if len(idx) > 1 else 1))

    all_xyz = [mh[:, 1:4]] + [dh[:, 1:4] for dh in dhs] + [sh_xyz]
    all_xyz = np.vstack(all_xyz)
    pad = 0.05 * max(1.0, np.ptp(all_xyz[:, 0]), np.ptp(all_xyz[:, 1]))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad)
    ax.set_ylim(all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad)
    ax.set_zlim(0, max(all_xyz[:, 2].max() * 1.1, 100))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title("Fury-style escort engagement")

    mh_point, = ax.plot([], [], [], "o", color=MOTHERSHIP_COLOR, markersize=8, label="Mothership (F-16)")
    mh_trail, = ax.plot([], [], [], "-", color=MOTHERSHIP_COLOR, lw=2)

    drone_points, drone_trails = [], []
    for i in range(len(dhs)):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        pt, = ax.plot([], [], [], "o", color=color, markersize=5, label=f"drone_{i}")
        tr, = ax.plot([], [], [], "-", color=color, lw=1)
        drone_points.append(pt)
        drone_trails.append(tr)

    sam_point, = ax.plot([], [], [], "^", color=SAM_COLOR, markersize=7, label="SAM")
    sam_trail, = ax.plot([], [], [], ":", color=SAM_COLOR, lw=1.5)

    ax.legend(loc="upper left", fontsize=8)

    outcome_artist = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=12,
                                color="darkred", weight="bold")

    def frame_at(arr_t, arr_xyz, t):
        'nearest-index lookup by time (arrays may end early if that entity died sooner)'
        j = min(np.searchsorted(arr_t, t), len(arr_t) - 1)
        return j

    def update(frame_num):
        real_frame = min(frame_num, n_real_frames - 1)
        resolved = frame_num >= n_real_frames - 1

        i = idx[real_frame]
        t = mh[i, 0]
        start = idx[max(0, real_frame - trail_len)]

        mh_point.set_data([mh[i, 1]], [mh[i, 2]])
        mh_point.set_3d_properties([mh[i, 3]])
        mh_trail.set_data(mh[start:i + 1, 1], mh[start:i + 1, 2])
        mh_trail.set_3d_properties(mh[start:i + 1, 3])

        for dh, pt, tr in zip(dhs, drone_points, drone_trails):
            j = frame_at(dh[:, 0], dh[:, 1:4], t)
            j_start = frame_at(dh[:, 0], dh[:, 1:4], mh[start, 0])
            pt.set_data([dh[j, 1]], [dh[j, 2]])
            pt.set_3d_properties([dh[j, 3]])
            tr.set_data(dh[j_start:j + 1, 1], dh[j_start:j + 1, 2])
            tr.set_3d_properties(dh[j_start:j + 1, 3])

        js = frame_at(sh_t, sh_xyz, t)
        js_start = frame_at(sh_t, sh_xyz, mh[start, 0])
        sam_point.set_data([sh_xyz[js, 0]], [sh_xyz[js, 1]])
        sam_point.set_3d_properties([sh_xyz[js, 2]])
        sam_trail.set_data(sh_xyz[js_start:js + 1, 0], sh_xyz[js_start:js + 1, 1])
        sam_trail.set_3d_properties(sh_xyz[js_start:js + 1, 2])

        # Only once the engagement has actually resolved, call out in plain
        # text who (if anyone) was actually hit -- at this sim's scale, an
        # intercept/near-miss and a real hit both look like the SAM marker
        # merging into the aircraft cluster, so proximity alone is ambiguous.
        if resolved:
            outcome_artist.set_text(outcome_text)

            if destroyed_name == "mothership":
                mh_point.set_marker("X")
                mh_point.set_color("red")
                mh_point.set_markersize(14)
            elif destroyed_name in drone_names:
                di = drone_names.index(destroyed_name)
                drone_points[di].set_marker("X")
                drone_points[di].set_color("red")
                drone_points[di].set_markersize(12)
        else:
            outcome_artist.set_text("")

        ax.set_title(f"Fury-style escort engagement (t={t:.1f}s, "
                     f"survived={result.mothership_survived})")

        return ([mh_point, mh_trail, sam_point, sam_trail, outcome_artist]
                + drone_points + drone_trails)

    anim_obj = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)

    print(f"Rendering {n_frames} frames to '{args.filename}'...")
    anim_obj.save(args.filename, writer="pillow", fps=args.fps, dpi=90)
    plt.close(fig)
    print(f"Saved '{args.filename}'")


if __name__ == "__main__":
    main()
