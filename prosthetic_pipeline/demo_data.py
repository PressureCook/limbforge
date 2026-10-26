"""Synthetic movement data for demo mode (no camera required).

A parametric side-view stick figure articulated by joint angles powers both the
single-joint demo (elbow curls) and full-limb demos (arm raise+curl, leg squat
sequence), so every downstream stage runs exactly as on real footage.

Image coordinates: x right, y DOWN. Segment direction angles phi are measured
from +x; hanging straight down is phi = 90 deg. Chains (right side articulated):
    upper arm  phi_ua = 90 - theta_shoulder      (flexion raises arm forward)
    forearm    phi_fa = phi_ua - theta_elbow     (elbow folds forward)
    thigh      phi_th = 90 - theta_hip
    shank      phi_sh = phi_th + theta_knee      (knee folds backward)
    foot       phi_ft = phi_sh - theta_ankle     (ankle: 90 deg = foot flat)
"""

from typing import Dict

import numpy as np

from .pose import PoseSeries

FPS = 30.0

# visual segment lengths in normalized image units
L_UA, L_FA, L_TH, L_SH, L_FT = 0.155, 0.145, 0.17, 0.17, 0.055

# physical segment lengths (m) per joint: (proximal, distal)
SEG_LEN = {
    "shoulder": (0.45, 0.30),
    "elbow": (0.30, 0.27),
    "hip": (0.45, 0.42),
    "knee": (0.42, 0.43),
    "ankle": (0.43, 0.20),
}
# driven-mass reference lengths (m): whole limb for shoulder/hip
MASS_LEN = {"shoulder": 0.57, "hip": 0.85}

DEFAULT_ANGLES = {"shoulder": 55.0, "elbow": 15.0, "hip": 4.0, "knee": 4.0, "ankle": 90.0}


def _dir(phi_deg: float) -> np.ndarray:
    r = np.deg2rad(phi_deg)
    return np.array([np.cos(r), np.sin(r)])


def _stick_figure(angles: Dict[str, float]) -> np.ndarray:
    """33 BlazePose landmarks (x, y, visibility) for a right-side view."""
    a = dict(DEFAULT_ANGLES)
    a.update(angles)
    lm = np.full((33, 3), np.nan)
    vis_hi, vis_lo = 0.99, 0.85

    shoulder = np.array([0.40, 0.30])
    hip = np.array([0.42, 0.58])
    head = shoulder + np.array([0.015, -0.10])

    def put(i, p, v=vis_hi):
        lm[i] = (p[0], p[1], v)

    # head
    put(0, head)
    for i in (1, 2, 3, 4, 5, 6):
        put(i, head + np.array([0.008, -0.008]), vis_lo)
    put(7, head + np.array([-0.02, 0.0]), vis_lo)
    put(8, head + np.array([-0.015, 0.002]), vis_lo)
    put(9, head + np.array([0.015, 0.03]), vis_lo)
    put(10, head + np.array([0.02, 0.03]), vis_lo)

    # articulated right arm
    phi_ua = 90.0 - a["shoulder"]
    phi_fa = phi_ua - a["elbow"]
    elbow = shoulder + L_UA * _dir(phi_ua)
    wrist = elbow + L_FA * _dir(phi_fa)
    put(12, shoulder)
    put(14, elbow)
    put(16, wrist)
    hand = _dir(phi_fa)
    put(18, wrist + 0.030 * hand, vis_lo)
    put(20, wrist + 0.034 * hand, vis_lo)
    put(22, wrist + 0.025 * hand, vis_lo)

    # far-side arm hangs
    put(11, shoulder + np.array([-0.015, 0.005]), vis_lo)
    put(13, shoulder + np.array([-0.02, 0.15]), vis_lo)
    put(15, shoulder + np.array([-0.02, 0.28]), vis_lo)
    for i in (17, 19, 21):
        put(i, shoulder + np.array([-0.02, 0.31]), 0.6)

    # articulated right leg
    phi_th = 90.0 - a["hip"]
    phi_sh = phi_th + a["knee"]
    phi_ft = phi_sh - a["ankle"]
    knee = hip + L_TH * _dir(phi_th)
    ankle = knee + L_SH * _dir(phi_sh)
    toe = ankle + L_FT * _dir(phi_ft)
    heel = ankle + 0.025 * _dir(phi_ft + 180.0)
    put(24, hip)
    put(26, knee)
    put(28, ankle)
    put(30, heel, vis_lo)
    put(32, toe)

    # far-side leg stands straight
    knee_l = hip + np.array([-0.015, L_TH])
    ankle_l = knee_l + np.array([0.0, L_SH])
    put(23, hip + np.array([-0.015, 0.0]), vis_lo)
    put(25, knee_l, vis_lo)
    put(27, ankle_l, vis_lo)
    put(29, ankle_l + np.array([-0.03, 0.02]), 0.6)
    put(31, ankle_l + np.array([0.05, 0.02]), 0.6)
    return lm


def _series(joint: str, t: np.ndarray, theta: np.ndarray, lms: np.ndarray,
            note: str) -> PoseSeries:
    Lp, Ld = SEG_LEN[joint]
    return PoseSeries(
        joint=joint, side="right", t=t, theta_deg=theta, Lp=Lp, Ld=Ld,
        L_mass=MASS_LEN.get(joint, Ld), norm_landmarks=lms, frame_wh=(640, 640),
        n_frames=len(t), n_detected=len(t), notes=[note], synthetic=True)


def make_demo_series(seed: int = 0) -> PoseSeries:
    """Single-joint demo: three elbow curls with a hold (unchanged behavior)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 8.0, 1.0 / FPS)
    theta = 72.0 - 62.0 * np.cos(2 * np.pi * 0.40 * t)
    hold = t > 6.9
    theta[hold] = theta[np.argmax(hold)]
    theta = np.clip(theta + rng.normal(0.0, 0.35, theta.shape), 2.0, 148.0)
    lms = np.stack([_stick_figure({"elbow": td}) for td in theta])
    return _series("elbow", t, theta, lms,
                   "synthetic demo trajectory (three elbow curls at ~0.4 Hz)")


def make_limb_demo_series(limb: str, seed: int = 0) -> Dict[str, PoseSeries]:
    """Full-limb demo: all joints of the limb moving through a natural sequence."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 10.0, 1.0 / FPS)

    if limb == "body":
        figure_angles = {
            "shoulder": 20.0 - 16.0 * np.cos(2 * np.pi * 0.15 * t),
            "elbow": 55.0 - 45.0 * np.cos(2 * np.pi * 0.35 * t + 1.0),
            "hip": 32.0 - 28.0 * np.cos(2 * np.pi * 0.18 * t - 0.2),
            "knee": 45.0 - 40.0 * np.cos(2 * np.pi * 0.18 * t - 0.5),
            "ankle": 90.0 + 16.0 * np.sin(2 * np.pi * 0.18 * t - 0.7),
        }
        note = "synthetic full-body demo (arm raise+curl with slow squats)"
    elif limb == "arm":
        figure_angles = {
            "shoulder": 45.0 - 40.0 * np.cos(2 * np.pi * 0.15 * t),
            "elbow": 60.0 - 50.0 * np.cos(2 * np.pi * 0.35 * t + 1.0),
        }
        note = "synthetic arm demo (shoulder raises + elbow curls)"
    elif limb == "leg":
        figure_angles = {
            "hip": 40.0 - 35.0 * np.cos(2 * np.pi * 0.18 * t),
            "knee": 50.0 - 45.0 * np.cos(2 * np.pi * 0.18 * t - 0.3),
            "ankle": 90.0 + 20.0 * np.sin(2 * np.pi * 0.18 * t - 0.6),
        }
        note = "synthetic leg demo (slow squats with ankle motion)"
    else:
        raise ValueError(f"unknown limb: {limb}")

    for k in figure_angles:
        figure_angles[k] = figure_angles[k] + rng.normal(0.0, 0.3, len(t))

    lms = np.stack([_stick_figure({k: v[i] for k, v in figure_angles.items()})
                    for i in range(len(t))])
    # Pose extraction measures flexion = 180 - interior angle. For the shoulder
    # (hip reference on the same side as the hanging arm) the raise angle r maps
    # to a measured flexion of 180 - r; all other joints measure the raise/fold
    # angle directly. Emit series in measured space so demo == real footage.
    series_angles = {k: (180.0 - v if k == "shoulder" else v)
                     for k, v in figure_angles.items()}
    return {joint: _series(joint, t, theta, lms, note)
            for joint, theta in series_angles.items()}
