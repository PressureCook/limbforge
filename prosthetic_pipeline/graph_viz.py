"""Plots and animation for graph-engine runs (SPEC §7.4)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from .topology import get_topology
from .visualize import ACCENT, ACCENT2, GRAY, MAX_GIF_FRAMES, _skeleton_segments

ACT_COLORS = ["#3a6ea5", "#4caf7d", "#7a5cc4"]


def plot_transmission(table, F_max, path, title="Evolved transmission"):
    """Per-actuator transmission ratio, reduced inertia, and load vs capacity."""
    deg = np.degrees(table.theta)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i in range(table.n_act):
        unit = 1000.0 if table.actuator_kinds[i] != "rotary" else 1.0
        lab = "mm" if table.actuator_kinds[i] != "rotary" else "ratio"
        axes[0].plot(deg, table.r[i] * unit, color=ACT_COLORS[i % 3], lw=2,
                     label=f"a{i} ({table.actuator_kinds[i]}, {lab})")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("transmission ratio r(theta)")
    axes[0].legend(fontsize=8)

    axes[1].plot(deg, table.I_eff * 1000, color=ACCENT, lw=2)
    axes[1].set_title("reduced inertia I_eff (g m^2)")

    load = np.abs(table.G + table.S)
    cap = sum(np.abs(table.r[i]) * F_max[i] for i in range(table.n_act))
    axes[2].plot(deg, cap, color=ACCENT2, lw=2, label="torque capacity")
    axes[2].plot(deg, 1.3 * load, ls="--", color=GRAY, label="1.3 x (gravity+spring)")
    axes[2].set_title("capacity vs load (N m)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("flexion (deg)")
        ax.grid(alpha=0.3)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_tracking_graph(final, path):
    n_act = final["u"].shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2]})
    t = final["t"]
    axes[0].plot(t, final["theta_target_deg"], color=GRAY, lw=2, label="recorded")
    axes[0].plot(t, final["theta_sim_deg"], color=ACCENT, lw=1.6, label="prosthesis")
    axes[0].set_ylabel("flexion (deg)")
    axes[0].set_title(f"Tracking - RMSE {final['rmse_deg']:.2f} deg (QP allocation)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for i in range(n_act):
        axes[1].plot(t, final["u"][i], color=ACT_COLORS[i % 3], lw=1.1,
                     label=f"a{i}")
        axes[1].axhline(final["F_max"][i], ls="--", color=ACT_COLORS[i % 3], lw=0.7)
        axes[1].axhline(-final["F_max"][i], ls="--", color=ACT_COLORS[i % 3], lw=0.7)
    axes[1].set_ylabel("actuator effort")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8, ncol=n_act)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_search(history_best, history_species, history_parts, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(history_best, color=ACCENT, lw=2)
    axes[0].set_yscale("log")
    axes[0].set_title("best design cost")
    axes[1].plot(history_species, color=ACCENT2, lw=2)
    axes[1].set_title("species count (diversity)")
    axes[2].plot(history_parts, color="#7a5cc4", lw=2)
    axes[2].set_title("mean parts per design (bloat check)")
    for ax in axes:
        ax.set_xlabel("generation")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_gif_graph(pose, genome, ctx, final, path, title_extra=""):
    """Skeleton vs evolved mechanism, drawn by the topology itself."""
    topo = get_topology(genome.topology)
    n = len(pose.t)
    step = max(1, int(np.ceil(n / MAX_GIF_FRAMES)))
    frame_idx = np.arange(0, n, step)
    times = pose.t[frame_idx]
    th_at = np.interp(times, final["t"], np.deg2rad(final["theta_sim_deg"]))

    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.0], hspace=0.32, wspace=0.12)
    ax_sk = fig.add_subplot(gs[0, 0])
    ax_me = fig.add_subplot(gs[0, 1])
    ax_tl = fig.add_subplot(gs[1, :])
    fig.suptitle(f"Movement -> evolved {genome.topology} mechanism   "
                 f"({pose.side} {pose.joint}{title_extra})", fontsize=12)

    ax_sk.set_title("observed movement", fontsize=10)
    lc = LineCollection([], colors=GRAY, linewidths=2)
    ax_sk.add_collection(lc)
    ar = pose.frame_wh[0] / max(pose.frame_wh[1], 1)
    pts = pose.norm_landmarks[frame_idx][:, :, :2].reshape(-1, 2)
    ok = ~np.isnan(pts[:, 0])
    xs = pts[ok, 0] * ar if ok.any() else np.array([0, 1])
    ys = 1 - pts[ok, 1] if ok.any() else np.array([0, 1])
    ax_sk.set_xlim(xs.min() - 0.05, xs.max() + 0.05)
    ax_sk.set_ylim(ys.min() - 0.05, ys.max() + 0.05)
    ax_sk.set_aspect("equal")
    ax_sk.axis("off")

    lim = 1.15 * max(ctx.Lp, ctx.Ld)
    ax_tl.plot(pose.t, pose.theta_deg, color=GRAY, lw=2, label="observed")
    ax_tl.plot(final["t"], final["theta_sim_deg"], color=ACCENT, lw=1.4,
               label="prosthesis")
    cursor = ax_tl.axvline(0, color="k", lw=1)
    ax_tl.set_xlabel("time (s)", fontsize=9)
    ax_tl.set_ylabel("flexion (deg)", fontsize=9)
    ax_tl.legend(fontsize=8, ncol=2, loc="upper right")
    ax_tl.grid(alpha=0.3)

    def update(k):
        i = frame_idx[k]
        lc.set_segments(_skeleton_segments(pose.norm_landmarks[i], pose.frame_wh))
        ax_me.cla()
        topo.draw(ax_me, genome, ctx, float(th_at[k]))
        ax_me.set_xlim(-0.55 * lim, lim)
        ax_me.set_ylim(-0.35 * lim, lim)
        ax_me.set_aspect("equal")
        ax_me.axis("off")
        ax_me.set_title(f"evolved {genome.topology} (simulation)", fontsize=10)
        cursor.set_xdata([times[k], times[k]])
        return []

    fps = float(np.clip(len(times) / max(times[-1] - times[0], 0.5), 5, 30))
    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
