"""Stage 8: real-world usage demonstration on the evolved prosthetic limb.

Leg -> a walking cycle; arm -> picking an object up off a table and placing it
on a shelf (with the object's mass carried by the actuators). Every joint is
driven by ITS evolved actuator through the task trajectory - force/speed
saturation and payload effects are physical, so this answers "could this
prosthesis actually do the job?" with tracking error and force utilization.

Task trajectories are synthetic, normative-style profiles (labeled as such),
in the pipeline's measured-flexion convention (180 - interior angle).
"""

from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .simulate import simulate_final
from .visualize import GRAY, JOINT_COLORS, MAX_GIF_FRAMES, _mech_anchors_world

PAYLOAD_KG = 1.0
FPS_TRAJ = 30.0


# ---------------------------------------------------------------------------
# task trajectories
# ---------------------------------------------------------------------------

def gait_trajectories(duration=5.0, f=0.8):
    """Walking cycle, one side (measured flexion, deg)."""
    t = np.arange(0.0, duration, 1.0 / FPS_TRAJ)
    w = 2 * np.pi * f
    hip = 17.0 - 15.0 * np.cos(w * t)
    knee = np.clip(22.0 - 18.0 * np.cos(w * t) + 12.0 * np.cos(2 * w * t - 2.5), 1.0, None)
    ankle = 90.0 + 12.0 * np.sin(w * t - 0.8) - 8.0 * np.sin(2 * w * t)
    return t, {"hip": hip, "knee": knee, "ankle": ankle}, {"cycle_s": 1.0 / f}


def _keyframe_traj(t, keys):
    """Smooth (cosine-eased) interpolation through (time, value) keyframes."""
    kt = np.array([k[0] for k in keys])
    kv = np.array([k[1] for k in keys])
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        j = np.searchsorted(kt, ti, side="right") - 1
        j = np.clip(j, 0, len(kt) - 2)
        span = kt[j + 1] - kt[j]
        s = np.clip((ti - kt[j]) / max(span, 1e-9), 0, 1)
        s = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in-out
        out[i] = kv[j] + s * (kv[j + 1] - kv[j])
    return out


def pick_place_trajectories(duration=9.0):
    """Reach -> grasp -> lift to shelf -> place -> return (measured flexion, deg)."""
    t = np.arange(0.0, duration, 1.0 / FPS_TRAJ)
    # keyframes in "raise" space for the shoulder (0 = hanging), elbow flexion direct
    raise_keys = [(0, 10), (2.0, 35), (3.0, 35), (5.5, 70), (6.5, 70), (8.0, 12), (9.0, 10)]
    elbow_keys = [(0, 15), (2.0, 55), (3.0, 55), (5.5, 40), (6.5, 40), (8.0, 18), (9.0, 15)]
    shoulder = 180.0 - _keyframe_traj(t, raise_keys)
    elbow = _keyframe_traj(t, elbow_keys)
    grasp_s, release_s = 3.0, 6.5
    payload = np.where((t >= grasp_s) & (t < release_s), PAYLOAD_KG, 0.0)
    events = {"grasp_s": grasp_s, "release_s": release_s, "payload_kg": PAYLOAD_KG}
    return t, {"shoulder": shoulder, "elbow": elbow}, payload, events


def task_trajectories_for(limb: str):
    """(t, traj_by_joint, payload_at_hand) for design-time use of the usage task."""
    if limb == "leg":
        t, traj, _ = gait_trajectories()
        return t, traj, np.zeros_like(t)
    if limb == "body":
        # full body: leg joints get the gait duty cycle; arm joints get a
        # walking arm swing so shoulder/elbow are also task-checked
        t, traj, _ = gait_trajectories()
        w = 2 * np.pi * 0.8
        traj = dict(traj)
        traj["shoulder"] = 180.0 - (12.0 + 8.0 * np.sin(w * t + np.pi))
        traj["elbow"] = 22.0 + 8.0 * np.sin(w * t)
        return t, traj, np.zeros_like(t)
    t, traj, payload, _ = pick_place_trajectories()
    return t, traj, payload


# ---------------------------------------------------------------------------
# usage simulation
# ---------------------------------------------------------------------------

def run_usage_scenario(limb: str, mechs: Dict, ctxs: Dict):
    """Simulate every evolved joint through the limb's real-world task."""
    if limb in ("leg", "body"):
        t, traj, meta = gait_trajectories()
        if limb == "body":
            w = 2 * np.pi * 0.8
            traj = dict(traj)
            traj["shoulder"] = 180.0 - (12.0 + 8.0 * np.sin(w * t + np.pi))
            traj["elbow"] = 22.0 + 8.0 * np.sin(w * t)
        payload = np.zeros_like(t)
        scenario = "walking cycle" + (" with arm swing" if limb == "body" else "")
        events = meta
    else:
        t, traj, payload, events = pick_place_trajectories()
        scenario = f"pick & place a {events['payload_kg']:.0f} kg object"

    results = {}
    for joint, theta in traj.items():
        if joint not in mechs:
            continue
        ctx = ctxs[joint]
        # the object rides at the hand = ctx.L_mass from the joint (whole arm
        # for the shoulder, forearm for the elbow), the payload default
        final = simulate_final(mechs[joint], t, theta, ctx,
                               payload_t=t, payload_kg=payload)
        F = np.abs(final["F"])
        results[joint] = {
            "final": final,
            "theta_task": theta,
            "rmse_deg": final["rmse_deg"],
            "utilization": float(F.max() / mechs[joint].F_max),
            "saturation_frac": float(np.mean(F >= 0.98 * mechs[joint].F_max)),
        }
    return {"scenario": scenario, "t": t, "traj": traj, "payload": payload,
            "events": events, "joints": results}


# ---------------------------------------------------------------------------
# scene animations
# ---------------------------------------------------------------------------

def _leg_chain(hip_deg, knee_deg, ankle_deg, L):
    d = np.deg2rad
    phi_th = -np.pi / 2 + d(hip_deg)
    phi_sh = phi_th - d(knee_deg)
    phi_ft = phi_sh + d(ankle_deg)
    hip = np.array([0.0, 0.0])
    knee = hip + L["hip"] * np.array([np.cos(phi_th), np.sin(phi_th)])
    ankle = knee + L["knee"] * np.array([np.cos(phi_sh), np.sin(phi_sh)])
    toe = ankle + L["ankle"] * np.array([np.cos(phi_ft), np.sin(phi_ft)])
    return hip, knee, ankle, toe


def _arm_chain(sh_deg_measured, el_deg, L):
    d = np.deg2rad
    phi_ua = -np.pi / 2 + d(180.0 - sh_deg_measured)
    phi_fa = phi_ua + d(el_deg)
    sh = np.array([0.0, 0.0])
    elbow = sh + L["shoulder"] * np.array([np.cos(phi_ua), np.sin(phi_ua)])
    wrist = elbow + L["elbow"] * np.array([np.cos(phi_fa), np.sin(phi_fa)])
    return sh, elbow, wrist


def make_usage_gif(limb: str, side: str, usage: dict, mechs: Dict, ctxs: Dict, path: str):
    t_all = usage["t"]
    joints = list(usage["joints"])
    n = len(t_all)
    step = max(1, int(np.ceil(n / MAX_GIF_FRAMES)))
    idx = np.arange(0, n, step)
    times = t_all[idx]
    sim = {j: np.interp(times, usage["joints"][j]["final"]["t"],
                        usage["joints"][j]["final"]["theta_sim_deg"]) for j in joints}
    L = {j: ctxs[j].Ld for j in joints}

    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    axu = fig.add_subplot(gs[1])
    fig.suptitle(f"Real-world usage: {usage['scenario']}  ({side} {limb}, "
                 "evolved actuators)", fontsize=12)

    # utilization strip
    for j in joints:
        Fn = np.interp(t_all, usage["joints"][j]["final"]["t"],
                       np.abs(usage["joints"][j]["final"]["F"])) / mechs[j].F_max
        axu.plot(t_all, 100 * Fn, color=JOINT_COLORS[j], lw=1.4, label=j)
    axu.axhline(100, ls="--", color=GRAY, lw=1)
    axu.set_ylabel("force used (%)", fontsize=8)
    axu.set_xlabel("time (s)", fontsize=8)
    axu.set_ylim(0, 118)
    axu.legend(fontsize=8, ncol=len(joints), loc="upper right")
    axu.tick_params(labelsize=7)
    axu.grid(alpha=0.25)
    cursor = axu.axvline(0, color="k", lw=0.9)

    ax.set_aspect("equal")
    ax.axis("off")

    if limb in ("leg", "body"):
        joints = [j for j in ("hip", "knee", "ankle") if j in joints]
        leg_h = L["hip"] + L["knee"] + 0.06
        ground_y = -leg_h
        ax.set_xlim(-0.8 * leg_h, 0.9 * leg_h)
        ax.set_ylim(ground_y - 0.12, 0.75)
        ax.plot([-2, 2], [ground_y, ground_y], color="#666666", lw=2, zorder=1)
        ticks = [ax.plot([], [], color="#666666", lw=1.2, zorder=1)[0] for _ in range(14)]
        torso, = ax.plot([], [], color="#444444", lw=8, solid_capstyle="round", zorder=2)
        head = plt.Circle((0, 0), 0.055, fc="#444444", zorder=2)
        ax.add_patch(head)
        other = [ax.plot([], [], color="#b9b9b9", lw=6, solid_capstyle="round", zorder=2)[0]
                 for _ in range(3)]
        pros = {j: ax.plot([], [], color=JOINT_COLORS[j], lw=7, solid_capstyle="round",
                           zorder=4)[0] for j in joints}
        acts = {j: ax.plot([], [], color="#3a6ea5", lw=3.5, zorder=5)[0] for j in joints}
        gait_t, gait_traj, meta = gait_trajectories(duration=t_all[-1] + 1.0 / FPS_TRAJ)
        half = 0.5 * meta["cycle_s"]

        def update(k):
            tk = times[k]
            bob = 0.015 * np.sin(4 * np.pi * 0.8 * tk)
            hip0 = np.array([0.0, bob])
            torso.set_data([hip0[0], hip0[0] + 0.03], [hip0[1], hip0[1] + 0.5])
            head.center = (hip0[0] + 0.035, hip0[1] + 0.57)
            # intact side: same gait half a cycle out of phase
            oth = {j: np.interp((tk + half) % gait_t[-1], gait_t, gait_traj[j])
                   for j in ("hip", "knee", "ankle")}
            pts = _leg_chain(oth["hip"], oth["knee"], oth["ankle"], L)
            for seg, (p, q) in zip(other, zip(pts[:-1], pts[1:])):
                seg.set_data([p[0] + hip0[0], q[0] + hip0[0]], [p[1] + hip0[1], q[1] + hip0[1]])
            # prosthetic side from the simulated joints
            pts = _leg_chain(sim["hip"][k], sim["knee"][k], sim["ankle"][k], L)
            named = {"hip": (pts[0], pts[1]), "knee": (pts[1], pts[2]), "ankle": (pts[2], pts[3])}
            prox_ref = {"hip": pts[0] + np.array([0, 0.4]), "knee": pts[0], "ankle": pts[1]}
            for j in joints:
                p, q = named[j]
                pros[j].set_data([p[0] + hip0[0], q[0] + hip0[0]], [p[1] + hip0[1], q[1] + hip0[1]])
                Aw, Bw = _mech_anchors_world(mechs[j], j, p, prox_ref[j],
                                             np.deg2rad(sim[j][k]))
                acts[j].set_data([Aw[0] + hip0[0], Bw[0] + hip0[0]],
                                 [Aw[1] + hip0[1], Bw[1] + hip0[1]])
            # treadmill ticks
            v = 0.55
            for i, tick in enumerate(ticks):
                x = -0.8 * leg_h + ((i * 0.14 - v * tk) % (1.7 * leg_h))
                tick.set_data([x, x + 0.05], [ground_y, ground_y])
            cursor.set_xdata([tk, tk])
            return []
    else:  # arm pick & place
        reach = L["shoulder"] + L["elbow"]
        ax.set_xlim(-0.55 * reach, 1.45 * reach)
        ax.set_ylim(-1.25 * reach, 0.55 * reach)
        torso, = ax.plot([0, -0.02], [0, -0.62], color="#444444", lw=8,
                         solid_capstyle="round", zorder=2)
        head = plt.Circle((0.01, 0.09), 0.05, fc="#444444", zorder=2)
        ax.add_patch(head)
        # table and shelf placed where the hand actually goes
        traj = usage["traj"]
        g_i = int(usage["events"]["grasp_s"] * FPS_TRAJ)
        p_i = int((usage["events"]["release_s"] - 0.2) * FPS_TRAJ)
        hand_g = _arm_chain(traj["shoulder"][g_i], traj["elbow"][g_i], L)[2]
        hand_p = _arm_chain(traj["shoulder"][p_i], traj["elbow"][p_i], L)[2]
        for pos, w in ((hand_g, 0.30), (hand_p, 0.22)):
            ax.plot([pos[0] - w / 2, pos[0] + w / 2], [pos[1] - 0.035] * 2,
                    color="#666666", lw=3, zorder=1)
            ax.plot([pos[0] - w / 2 + 0.02] * 2, [pos[1] - 0.035, -1.25 * reach],
                    color="#666666", lw=2, zorder=1, alpha=0.5)
        obj = plt.Rectangle((0, 0), 0.055, 0.055, fc="#c9a227", ec="k", zorder=6)
        ax.add_patch(obj)
        pros = {j: ax.plot([], [], color=JOINT_COLORS[j], lw=7, solid_capstyle="round",
                           zorder=4)[0] for j in joints}
        acts = {j: ax.plot([], [], color="#3a6ea5", lw=3.5, zorder=5)[0] for j in joints}
        state_txt = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=9)
        grasp_s, release_s = usage["events"]["grasp_s"], usage["events"]["release_s"]

        def update(k):
            tk = times[k]
            sh, elbow, wrist = _arm_chain(sim["shoulder"][k], sim["elbow"][k], L)
            pros["shoulder"].set_data([sh[0], elbow[0]], [sh[1], elbow[1]])
            pros["elbow"].set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
            hipref = sh + np.array([0.0, -0.5])
            Aw, Bw = _mech_anchors_world(mechs["shoulder"], "shoulder", sh, hipref,
                                         np.deg2rad(sim["shoulder"][k]))
            acts["shoulder"].set_data([Aw[0], Bw[0]], [Aw[1], Bw[1]])
            Aw, Bw = _mech_anchors_world(mechs["elbow"], "elbow", elbow, sh,
                                         np.deg2rad(sim["elbow"][k]))
            acts["elbow"].set_data([Aw[0], Bw[0]], [Aw[1], Bw[1]])
            if tk < grasp_s:
                pos, label = hand_g, "reaching"
            elif tk < release_s:
                pos, label = wrist, f"carrying {usage['events']['payload_kg']:.0f} kg"
            else:
                pos, label = hand_p, "placed - returning"
            obj.set_xy((pos[0] - 0.027, pos[1] - 0.027))
            state_txt.set_text(label)
            cursor.set_xdata([tk, tk])
            return []

    fps = float(np.clip(len(times) / max(times[-1] - times[0], 0.5), 5, 30))
    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
