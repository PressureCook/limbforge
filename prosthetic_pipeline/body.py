"""Full-body coupled design: each joint is built WITH THE OTHERS IN MIND.

Design runs DISTAL-FIRST (elbow before shoulder; ankle -> knee -> hip), and
two real couplings flow up the chain into every proximal joint's model:

1. HARDWARE ROLLUP - a designed distal joint's actuator + linkage mass is a
   point mass riding the proximal joint's driven segment (an elbow actuator
   hangs from the shoulder; knee + ankle hardware hangs from the hip). It
   raises the proximal mass, shifts its COM and adds m*d^2 of inertia, so the
   proximal joint is sized to carry the prosthesis it is part of - not a bare
   biological limb.

2. TRAJECTORY COUPLING - the recorded DISTAL motion changes the limb's mass
   distribution over time: curling the elbow pulls the arm's COM toward the
   shoulder, flexing the knee pulls the leg's COM toward the hip. The
   time-varying first-moment correction (relative to the straight-limb
   anthropometry the base model assumes) enters the proximal simulation
   through the payload channel, so the proximal design is evolved against the
   load profile the video actually shows.

Approximations, stated: the correction matches the gravity FIRST moment
exactly; the inertia correction is applied at the L_mass reference distance
(second-moment approximation). Joint time bases are aligned by interpolation.
Full recursive chain dynamics (reaction torques from distal acceleration)
remain out of scope and are listed in the report.
"""

from dataclasses import replace
from typing import Dict

import numpy as np

from .joints import BODY_CHILDREN, SEGMENT_ANTHROPOMETRY
from .simulate import JointContext

# design order: children before parents, so rollup masses exist when needed
BODY_DESIGN_ORDER = ["elbow", "ankle", "knee", "shoulder", "hip"]

# where a child's hardware mass rides, as seen from the parent joint:
# (child -> lambda poses: distance from parent joint)
_HW_DIST = {
    ("shoulder", "elbow"): lambda p: p["shoulder"].Ld,
    ("hip", "knee"): lambda p: p["hip"].Ld,
    ("hip", "ankle"): lambda p: p["hip"].L_mass,
    ("knee", "ankle"): lambda p: p["knee"].Ld,
}

# trajectory coupling: (parent -> (child whose flexion moves the COM,
# distal-part anthropometry key))
_TRAJ_CHILD = {"shoulder": ("elbow", "elbow"), "hip": ("knee", "knee")}


def hardware_mass_kg(joint_result: dict) -> float:
    """Actuator system mass of one designed joint (spec rollup or variant)."""
    spec = joint_result.get("spec") or {}
    if "rollup" in spec:                                   # graph-engine spec
        return float(spec["rollup"].get("total_mass_kg", 0.15))
    try:
        rec = spec["variants"][spec["recommended"]]
        return float(rec["mass_est_kg"])
    except Exception:
        return 0.15


def coupled_context(joint: str, series_map: Dict, designed: Dict,
                    body_mass: float):
    """JointContext for `joint`, carrying every already-designed descendant.

    Returns (ctx, coupling_info). designed: {joint: result-dict} built so far
    (distal-first order guarantees descendants are present)."""
    pose = series_map[joint]
    ctx = JointContext.from_pose(joint, pose.Lp, pose.Ld, body_mass,
                                 L_mass=pose.L_mass)
    carried = []
    m, first, second = ctx.mass, ctx.mass * ctx.com, ctx.inertia
    for child in BODY_CHILDREN.get(joint, []):
        if child not in designed or child not in series_map:
            continue
        d_fn = _HW_DIST.get((joint, child))
        if d_fn is None:
            continue
        d = float(d_fn(series_map))
        m_hw = hardware_mass_kg(designed[child])
        m += m_hw
        first += m_hw * d
        second += m_hw * d ** 2
        carried.append({"joint": child, "mass_g": round(1000 * m_hw),
                        "at_mm": round(1000 * d)})
    ctx = replace(ctx, mass=m, com=first / max(m, 1e-9), inertia=second,
                  damping=0.02 + 0.15 * second)
    return ctx, carried


def trajectory_coupling(joint: str, series_map: Dict, body_mass: float):
    """Time-varying payload correction from recorded distal motion.

    Returns (t, payload_kg, info) or (None, None, info). payload_kg is the
    equivalent point mass at ctx.L_mass whose first moment equals the COM
    shift the child's recorded flexion produces (negative when the limb
    folds toward the joint - the load genuinely drops)."""
    if joint not in _TRAJ_CHILD or joint not in series_map:
        return None, None, {}
    child, antro_key = _TRAJ_CHILD[joint]
    if child not in series_map:
        return None, None, {}
    parent, dist = series_map[joint], series_map[child]
    a = SEGMENT_ANTHROPOMETRY[antro_key]
    m_d = a["mass_frac"] * body_mass                 # distal-part mass
    L = parent.Ld                                    # parent joint -> child joint
    c = a["com_frac"] * dist.L_mass                  # child joint -> distal COM
    th = np.deg2rad(np.interp(parent.t, dist.t, dist.theta_deg))
    d_t = np.sqrt(L ** 2 + c ** 2 + 2 * L * c * np.cos(th))
    d0 = L + c                                       # straight-limb reference
    L_ref = parent.L_mass
    payload = m_d * (d_t - d0) / max(L_ref, 1e-9)
    info = {"from": child, "distal_mass_g": round(1000 * m_d),
            "lever_shift_mm": [round(1000 * float(d_t.min() - d0)),
                               round(1000 * float(d_t.max() - d0))],
            "note": ("first-moment exact; inertia correction applied at the "
                     "L_mass reference distance (approximation)")}
    return parent.t.copy(), payload, info
