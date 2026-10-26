"""Stage 5: plots and the animated side-by-side demo (real movement vs prosthesis)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from .joints import POSE_CONNECTIONS
from .mechanism import anchor_A, anchor_B, length_and_derivative
from .simulate import gravity_torque

ACCENT = "#e05c2a"       # prosthesis / distal segment
ACCENT2 = "#1f77b4"      # actuator
GRAY = "#8a8a8a"

MAX_GIF_FRAMES = 110


def plot_convergence(history_best, history_mean, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    gens = np.arange(len(history_best))
    ax.plot(gens, history_mean, color=GRAY, label="population mean")
    ax.plot(gens, history_best, color=ACCENT, lw=2, label="best design")
    ax.set_yscale("log")
    ax.set_xlabel("generation (count)")
    ax.set_ylabel("design cost (dimensionless, log scale)")
    ax.set_title("Evolutionary search convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_mechanism_curves(mech, ctx, th_lo, th_hi, path):
    th = np.linspace(th_lo, th_hi, 200)
    L, dL, r = length_and_derivative(mech.a, mech.alpha, mech.b, mech.beta, th)
    tau_cap = mech.F_max * r
    tau_g = np.abs(gravity_torque(th, ctx))
    deg = np.degrees(th)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(deg, L * 1000, color=ACCENT2, lw=2, label="length L(θ)")
    axes[0].axhline(mech.L_ret * 1000, ls="--", color=GRAY,
                    label=f"retracted {mech.L_ret*1000:.0f} mm")
    axes[0].axhline(mech.L_ext * 1000, ls="--", color="#b0b0b0",
                    label=f"extended {mech.L_ext*1000:.0f} mm")
    axes[0].set_title("actuator length vs flexion")
    axes[0].set_ylabel("length (mm)")
    axes[0].legend(fontsize=7)

    axes[1].plot(deg, r * 1000, color=ACCENT, lw=2, label="moment arm r(θ)")
    axes[1].axhline(8, ls=":", color="k", lw=1,
                    label="min usable arm 8 mm")
    axes[1].set_title("moment arm vs flexion")
    axes[1].set_ylabel("moment arm (mm)")
    axes[1].legend(fontsize=7)

    axes[2].plot(deg, tau_cap, color=ACCENT2, lw=2, label="torque capacity F·r")
    axes[2].plot(deg, 1.3 * tau_g, ls="--", color=GRAY,
                 label="static requirement: 1.3 × gravity")
    axes[2].set_title("torque capacity vs load")
    axes[2].set_ylabel("torque (N m)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("flexion (deg)")
        ax.grid(alpha=0.3)
    fig.suptitle("Evolved mechanism characteristics  ·  flexion θ = 180° − interior angle "
                 "(0° = limb straight, increasing = flexing)", y=1.03, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_tracking(final, mech, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    t = final["t"]
    axes[0].plot(t, final["theta_target_deg"], color=GRAY, lw=2, label="your movement (target)")
    axes[0].plot(t, final["theta_sim_deg"], color=ACCENT, lw=1.6, label="prosthesis simulation")
    axes[0].set_ylabel("flexion (deg)")
    axes[0].set_title(f"Trajectory tracking - RMSE {final['rmse_deg']:.2f} deg")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, final["F"], color=ACCENT2, lw=1.2,
                 label="axial force (tension +)")
    axes[1].axhline(mech.F_max, ls="--", color=GRAY,
                    label=f"capacity +{mech.F_max:.0f} N")
    axes[1].axhline(-mech.F_max, ls="--", color="#b0b0b0",
                    label=f"capacity −{mech.F_max:.0f} N")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_ylabel("actuator force (N)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _mech_draw_data(mech, theta):
    """Geometry needed to draw the mechanism at one flexion angle."""
    Ax, Ay = anchor_A(mech.a, mech.alpha)
    Bx, By = anchor_B(mech.b, mech.beta, theta)
    A, B = np.array([Ax, Ay]), np.array([Bx, By])
    L = np.linalg.norm(B - A)
    u = (B - A) / max(L, 1e-9)
    barrel_end = A + u * min(0.85 * mech.L_ret, 0.9 * L)
    return A, B, barrel_end, L


def _skeleton_segments(norm_lm, wh):
    w, h = wh
    ar = w / max(h, 1)
    segs = []
    for i, j in POSE_CONNECTIONS:
        p, q = norm_lm[i], norm_lm[j]
        if np.isnan(p[0]) or np.isnan(q[0]):
            continue
        segs.append([(p[0] * ar, 1 - p[1]), (q[0] * ar, 1 - q[1])])
    return segs


def make_gif(pose, mech, ctx, final, th_lo, th_hi, path, title_extra=""):
    """Animated demo: observed skeleton vs the evolved prosthetic joint."""
    # time base: decimate observed frames
    n = len(pose.t)
    step = max(1, int(np.ceil(n / MAX_GIF_FRAMES)))
    frame_idx = np.arange(0, n, step)
    times = pose.t[frame_idx]
    theta_sim_at = np.interp(times, final["t"], np.deg2rad(final["theta_sim_deg"]))
    L_at = np.interp(times, final["t"], final["L"])
    F_at = np.interp(times, final["t"], final["F"])

    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.0], hspace=0.32, wspace=0.12)
    ax_sk = fig.add_subplot(gs[0, 0])
    ax_me = fig.add_subplot(gs[0, 1])
    ax_tl = fig.add_subplot(gs[1, :])
    side_joint = f"{pose.side} {pose.joint}"
    fig.suptitle(
        f"Movement -> evolved prosthetic joint   ({side_joint}{title_extra})",
        fontsize=12)

    # --- skeleton panel -------------------------------------------------------
    ax_sk.set_title("observed movement" + (" (synthetic demo)" if pose.synthetic else " (from your photos)"),
                    fontsize=10)
    lc = LineCollection([], colors=GRAY, linewidths=2)
    ax_sk.add_collection(lc)
    tri_line, = ax_sk.plot([], [], color=ACCENT, lw=3.5, solid_capstyle="round")
    tri_dots, = ax_sk.plot([], [], "o", color=ACCENT, ms=5)
    ar = pose.frame_wh[0] / max(pose.frame_wh[1], 1)
    all_pts = pose.norm_landmarks[frame_idx][:, :, :2].reshape(-1, 2)
    ok = ~np.isnan(all_pts[:, 0])
    if ok.any():
        xs, ys = all_pts[ok, 0] * ar, 1 - all_pts[ok, 1]
    else:  # no valid landmarks in the decimated frames - neutral canvas
        xs, ys = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    pad = 0.05
    ax_sk.set_xlim(xs.min() - pad, xs.max() + pad)
    ax_sk.set_ylim(ys.min() - pad, ys.max() + pad)
    ax_sk.set_aspect("equal")
    ax_sk.axis("off")

    from .joints import JOINT_TRIPLETS
    triplet = JOINT_TRIPLETS[pose.joint][pose.side]

    # --- mechanism panel ------------------------------------------------------
    ax_me.set_title("evolved actuator joint (simulation)", fontsize=10)
    Lp, Ld = ctx.Lp, ctx.Ld
    # static proximal segment + ROM ghost arcs
    ax_me.plot([0, Lp], [0, 0], color="#444444", lw=7, solid_capstyle="round", zorder=1)
    for th_g in (th_lo, th_hi):
        tip = Ld * np.array([np.cos(np.pi - th_g), np.sin(np.pi - th_g)])
        ax_me.plot([0, tip[0]], [0, tip[1]], color=GRAY, lw=1.2, ls="--", zorder=1)
    distal_line, = ax_me.plot([], [], color=ACCENT, lw=7, solid_capstyle="round", zorder=3)
    barrel_line, = ax_me.plot([], [], color=ACCENT2, lw=8, solid_capstyle="butt", zorder=4)
    rod_line, = ax_me.plot([], [], color="#5f9ed1", lw=3, zorder=4)
    bracket_a, = ax_me.plot([], [], color="#444444", lw=2, zorder=2)
    bracket_b, = ax_me.plot([], [], color=ACCENT, lw=2, zorder=2)
    anchors, = ax_me.plot([], [], "o", color="k", ms=5, zorder=5)
    ax_me.plot(0, 0, "o", color="k", ms=9, zorder=5)
    ax_me.plot(0, 0, "o", color="w", ms=4, zorder=6)
    info_txt = ax_me.text(0.02, 0.02, "", transform=ax_me.transAxes, fontsize=9,
                          va="bottom", family="monospace")
    # bounds
    ths = np.linspace(min(th_lo, 0.05), th_hi, 60)
    tips = Ld * np.stack([np.cos(np.pi - ths), np.sin(np.pi - ths)], axis=1)
    Axy = np.array(anchor_A(mech.a, mech.alpha))
    Bs = np.stack(anchor_B(mech.b, mech.beta, ths), axis=1)
    pts = np.vstack([tips, Bs, Axy[None, :], [[0, 0]], [[Lp, 0]]])
    lo = pts.min(axis=0) - 0.06
    hi = pts.max(axis=0) + 0.06
    ax_me.set_xlim(lo[0], hi[0])
    ax_me.set_ylim(lo[1], hi[1])
    ax_me.set_aspect("equal")
    ax_me.axis("off")

    # --- timeline panel -------------------------------------------------------
    ax_tl.plot(pose.t, pose.theta_deg, color=GRAY, lw=2, label="observed")
    ax_tl.plot(final["t"], final["theta_sim_deg"], color=ACCENT, lw=1.4, label="prosthesis")
    cursor = ax_tl.axvline(0, color="k", lw=1)
    ax_tl.set_xlabel("time (s)", fontsize=9)
    ax_tl.set_ylabel("flexion (deg)", fontsize=9)
    ax_tl.legend(fontsize=8, loc="upper right", ncol=2)
    ax_tl.grid(alpha=0.3)
    ax_tl.tick_params(labelsize=8)

    def update(k):
        i = frame_idx[k]
        lm = pose.norm_landmarks[i]
        lc.set_segments(_skeleton_segments(lm, pose.frame_wh))
        tx = [lm[j, 0] * ar for j in triplet]
        ty = [1 - lm[j, 1] for j in triplet]
        if not any(np.isnan(tx)):
            tri_line.set_data(tx, ty)
            tri_dots.set_data(tx, ty)

        th = theta_sim_at[k]
        tip = Ld * np.array([np.cos(np.pi - th), np.sin(np.pi - th)])
        distal_line.set_data([0, tip[0]], [0, tip[1]])
        A, B, barrel_end, L = _mech_draw_data(mech, th)
        barrel_line.set_data([A[0], barrel_end[0]], [A[1], barrel_end[1]])
        rod_line.set_data([barrel_end[0], B[0]], [barrel_end[1], B[1]])
        bracket_a.set_data([min(mech.a * np.cos(mech.alpha), Lp), A[0]], [0, A[1]])
        bone_b = mech.b * np.cos(mech.beta)
        bb = bone_b * np.array([np.cos(np.pi - th), np.sin(np.pi - th)])
        bracket_b.set_data([bb[0], B[0]], [bb[1], B[1]])
        anchors.set_data([A[0], B[0]], [A[1], B[1]])
        info_txt.set_text(
            f"flexion {np.degrees(th):6.1f} deg\nlength  {L_at[k]*1000:6.1f} mm\n"
            f"force   {F_at[k]:6.0f} N")
        cursor.set_xdata([times[k], times[k]])
        return []

    fps = float(np.clip(len(times) / max(times[-1] - times[0], 0.5), 5, 30))
    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# full-limb animation
# ---------------------------------------------------------------------------

JOINT_COLORS = {"shoulder": "#e05c2a", "elbow": "#1f9e77", "hip": "#e05c2a",
                "knee": "#1f9e77", "ankle": "#b06fc4"}
# world sense of each joint's mechanism +y (flexion) axis in the sagittal chain
MECH_FLIP = {"shoulder": 1.0, "elbow": -1.0, "hip": -1.0, "knee": 1.0, "ankle": -1.0}


def _chain_points(limb, angles_deg, seg_len):
    """World positions (y up, facing +x) of the limb chain at given flexion angles.

    Returns dict joint -> (joint_pos, prox_ref_pos, distal_tip_pos, distal_phi).
    """
    out = {}
    d = np.deg2rad
    if limb == "body":
        out = _chain_points("leg", angles_deg, seg_len)
        if "shoulder" in angles_deg:
            arm = _chain_points("arm", angles_deg, seg_len)
            sh_off = np.array([0.0, 0.52])          # shoulder above the hip
            for j, (J, P, tip, phi) in arm.items():
                out[j] = (J + sh_off, P + sh_off, tip + sh_off, phi)
        return out
    if limb == "arm":
        sh = np.array([0.0, 0.0])
        hip = sh + np.array([0.0, -0.55])
        raise_ang = 180.0 - angles_deg["shoulder"]          # measured -> raise
        phi_ua = d(-90.0 + raise_ang)
        elbow = sh + seg_len["shoulder"] * np.array([np.cos(phi_ua), np.sin(phi_ua)])
        out["shoulder"] = (sh, hip, elbow, phi_ua)
        if "elbow" in angles_deg:
            phi_fa = phi_ua + d(angles_deg["elbow"])
            wrist = elbow + seg_len["elbow"] * np.array([np.cos(phi_fa), np.sin(phi_fa)])
            out["elbow"] = (elbow, sh, wrist, phi_fa)
    else:  # leg
        hip = np.array([0.0, 0.0])
        sh = hip + np.array([0.0, 0.50])
        phi_th = d(-90.0 + angles_deg["hip"])
        knee = hip + seg_len["hip"] * np.array([np.cos(phi_th), np.sin(phi_th)])
        out["hip"] = (hip, sh, knee, phi_th)
        if "knee" in angles_deg:
            phi_sh = phi_th - d(angles_deg["knee"])
            ankle = knee + seg_len["knee"] * np.array([np.cos(phi_sh), np.sin(phi_sh)])
            out["knee"] = (knee, hip, ankle, phi_sh)
            if "ankle" in angles_deg:
                phi_ft = phi_sh + d(angles_deg["ankle"])
                toe = ankle + seg_len["ankle"] * np.array([np.cos(phi_ft), np.sin(phi_ft)])
                out["ankle"] = (ankle, knee, toe, phi_ft)
    return out


def _mech_anchors_world(mech_obj, joint, J, P, theta_rad):
    """Actuator anchor points in chain-world coordinates for one joint."""
    u = P - J
    ang = np.arctan2(u[1], u[0])
    flip = MECH_FLIP[joint]
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    Am = np.array(anchor_A(mech_obj.a, mech_obj.alpha))
    Bm = np.array(anchor_B(mech_obj.b, mech_obj.beta, theta_rad))
    Aw = J + R @ (Am * np.array([1.0, flip]))
    Bw = J + R @ (Bm * np.array([1.0, flip]))
    return Aw, Bw


def make_limb_gif(limb, side, series_map, finals, mechs, ctx_map, path):
    """Animated full-limb demo: observed skeleton vs the simulated actuator chain."""
    ref_joint = max(series_map, key=lambda j: series_map[j].t[-1])
    ref = series_map[ref_joint]
    n = len(ref.t)
    step = max(1, int(np.ceil(n / MAX_GIF_FRAMES)))
    frame_idx = np.arange(0, n, step)
    times = ref.t[frame_idx]

    joints = [j for j in ("shoulder", "elbow", "hip", "knee", "ankle") if j in series_map]
    sim_at = {j: np.interp(times, finals[j]["t"], finals[j]["theta_sim_deg"])
              for j in joints}
    seg_len = {j: ctx_map[j].Ld for j in joints}

    fig = plt.figure(figsize=(10.5, 7.2))
    n_tl = len(joints)
    gs = fig.add_gridspec(1 + n_tl, 2, height_ratios=[3.2] + [0.75] * n_tl,
                          hspace=0.45, wspace=0.12)
    ax_sk = fig.add_subplot(gs[0, 0])
    ax_ch = fig.add_subplot(gs[0, 1])
    fig.suptitle(f"Full {limb} simulation ({side}) - every joint driven by its "
                 "evolved actuator", fontsize=12)

    # observed skeleton panel
    ax_sk.set_title("observed movement" + (" (synthetic demo)" if ref.synthetic else ""),
                    fontsize=10)
    lc = LineCollection([], colors=GRAY, linewidths=2)
    ax_sk.add_collection(lc)
    ar = ref.frame_wh[0] / max(ref.frame_wh[1], 1)
    pts = ref.norm_landmarks[frame_idx][:, :, :2].reshape(-1, 2)
    ok = ~np.isnan(pts[:, 0])
    xs, ys = pts[ok, 0] * ar, 1 - pts[ok, 1]
    ax_sk.set_xlim(xs.min() - 0.05, xs.max() + 0.05)
    ax_sk.set_ylim(ys.min() - 0.05, ys.max() + 0.05)
    ax_sk.set_aspect("equal")
    ax_sk.axis("off")

    # simulated chain panel
    ax_ch.set_title("simulated prosthetic limb", fontsize=10)
    torso_line, = ax_ch.plot([], [], color="#444444", lw=8, solid_capstyle="round", zorder=1)
    seg_lines, act_lines, dots = {}, {}, {}
    for j in joints:
        seg_lines[j], = ax_ch.plot([], [], color=JOINT_COLORS[j], lw=7,
                                   solid_capstyle="round", zorder=3, label=j)
        act_lines[j], = ax_ch.plot([], [], color=ACCENT2, lw=4, zorder=4)
        dots[j], = ax_ch.plot([], [], "o", color="k", ms=7, zorder=5)
    ax_ch.legend(fontsize=8, loc="upper right")
    reach = sum(seg_len.values()) + 0.35
    if limb == "arm":
        ax_ch.set_xlim(-reach * 0.7, reach)
        ax_ch.set_ylim(-reach, reach * 0.8)
    else:
        ax_ch.set_xlim(-reach * 0.7, reach)
        ax_ch.set_ylim(-reach, reach * 0.7)
    ax_ch.set_aspect("equal")
    ax_ch.axis("off")

    # per-joint timelines
    tl_axes, cursors = {}, {}
    for k, j in enumerate(joints):
        ax = fig.add_subplot(gs[1 + k, :])
        s = series_map[j]
        ax.plot(s.t, s.theta_deg, color=GRAY, lw=1.6)
        ax.plot(finals[j]["t"], finals[j]["theta_sim_deg"], color=JOINT_COLORS[j], lw=1.2)
        cursors[j] = ax.axvline(0, color="k", lw=0.8)
        ax.set_ylabel(j, fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if k < n_tl - 1:
            ax.set_xticklabels([])
        tl_axes[j] = ax
    tl_axes[joints[-1]].set_xlabel("time (s)", fontsize=8)

    def update(k):
        i = frame_idx[k]
        lc.set_segments(_skeleton_segments(ref.norm_landmarks[i], ref.frame_wh))
        angles = {j: sim_at[j][k] for j in joints}
        chain = _chain_points(limb, angles, seg_len)
        first = joints[0]
        J0, P0, _, _ = chain[first]
        torso_line.set_data([J0[0], P0[0]], [J0[1], P0[1]])
        for j in joints:
            J, P, tip, _ = chain[j]
            seg_lines[j].set_data([J[0], tip[0]], [J[1], tip[1]])
            th_fold = np.deg2rad(sim_at[j][k])
            Aw, Bw = _mech_anchors_world(mechs[j], j, J, P, th_fold)
            act_lines[j].set_data([Aw[0], Bw[0]], [Aw[1], Bw[1]])
            dots[j].set_data([J[0]], [J[1]])
            cursors[j].set_xdata([times[k], times[k]])
        return []

    fps = float(np.clip(len(times) / max(times[-1] - times[0], 0.5), 5, 30))
    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
