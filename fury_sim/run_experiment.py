"""
run_experiment.py
-----------------
Research harness: runs Monte Carlo trials across swarm sizes (and, optionally,
threat geometry) and reports mothership survival rate, drone attrition, and
missile miss-distance statistics -- the core numbers for a "does a Fury-style
escort swarm help the mothership survive a SAM shot" study.

Usage:
    python run_experiment.py
Produces:
    output/experiment_results.csv
    output/survival_vs_swarm_size.png
    output/sample_trajectory.png   (one illustrative 3-drone engagement)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engagement import run_engagement

N_TRIALS_PER_CONFIG = 60
SWARM_SIZES = [0, 1, 2, 3, 4, 6, 8]
SAM_RANGE = 20000.0


def run_all():
    rows = []
    for n_drones in SWARM_SIZES:
        for trial in range(N_TRIALS_PER_CONFIG):
            offset_angle = np.radians(np.random.uniform(-25, 25))
            result, mothership, drones, sam = run_engagement(
                n_drones=n_drones,
                sam_range=SAM_RANGE,
                sam_offset_angle=offset_angle,
                seed=None,
            )
            rows.append({
                "n_drones": n_drones,
                "trial": trial,
                "mothership_survived": result.mothership_survived,
                "n_drones_lost": result.n_drones_lost,
                "time_to_resolution": result.time_to_resolution,
                "min_miss_distance": result.min_missile_miss_distance,
            })
    return pd.DataFrame(rows)


def summarize_and_plot(df):
    summary = df.groupby("n_drones").agg(
        survival_rate=("mothership_survived", "mean"),
        mean_drones_lost=("n_drones_lost", "mean"),
        mean_miss_distance=("min_miss_distance", "mean"),
        n_trials=("trial", "count"),
    ).reset_index()
    print(summary.to_string(index=False))

    summary.to_csv("output/experiment_summary.csv", index=False)
    df.to_csv("output/experiment_results.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(summary["n_drones"], summary["survival_rate"] * 100, "o-", color="tab:blue", label="Mothership survival rate")
    ax1.set_xlabel("Escort swarm size (number of drones)")
    ax1.set_ylabel("Mothership survival rate (%)", color="tab:blue")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(f"Mothership survival vs escort swarm size\n({N_TRIALS_PER_CONFIG} Monte Carlo trials per size, generic PN-guided SAM)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(summary["n_drones"], summary["mean_drones_lost"], "s--", color="tab:red", label="Mean drones lost")
    ax2.set_ylabel("Mean drones lost per engagement", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    fig.savefig("output/survival_vs_swarm_size.png", dpi=150)
    plt.close(fig)
    return summary


def plot_sample_trajectory():
    result, mothership, drones, sam = run_engagement(
        n_drones=3, sam_range=SAM_RANGE, sam_offset_angle=0.0, seed=1, log_trajectories=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    mh = np.array(mothership.history)
    ax.plot(mh[:, 1], mh[:, 2], color="black", lw=2, label="Mothership (F-16)")

    for d in drones:
        if len(d.history) == 0:
            continue
        dh = np.array(d.history)
        ax.plot(dh[:, 1], dh[:, 2], lw=1.2, label=d.name)

    sh = np.array(sam.history, dtype=object)
    if len(sh) > 0:
        sx = np.array([row[1] for row in sh], dtype=float)
        sy = np.array([row[2] for row in sh], dtype=float)
        ax.plot(sx, sy, color="red", lw=1.5, linestyle=":", label="SAM")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Sample engagement (survived={result.mothership_survived}, "
                 f"drones_lost={result.n_drones_lost})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig("output/sample_trajectory.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = run_all()
    summarize_and_plot(df)
    plot_sample_trajectory()
    print("\nDone. See output/ for CSVs and plots.")
