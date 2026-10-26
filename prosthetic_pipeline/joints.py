"""Joint definitions on the BlazePose 33-landmark topology.

Each supported joint is defined by a (proximal_ref, joint, distal_ref) landmark
triplet. The flexion angle is defined as 180 deg minus the interior angle at the
joint, so 0 deg = segments aligned (limb straight) and larger = more flexed.
"""

# BlazePose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

# joint -> side -> (proximal_ref, joint, distal_ref)
JOINT_TRIPLETS = {
    "elbow": {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)},
    "knee": {"left": (L_HIP, L_KNEE, L_ANKLE), "right": (R_HIP, R_KNEE, R_ANKLE)},
    "shoulder": {"left": (L_HIP, L_SHOULDER, L_ELBOW), "right": (R_HIP, R_SHOULDER, R_ELBOW)},
    "hip": {"left": (L_SHOULDER, L_HIP, L_KNEE), "right": (R_SHOULDER, R_HIP, R_KNEE)},
    "ankle": {"left": (L_KNEE, L_ANKLE, L_FOOT), "right": (R_KNEE, R_ANKLE, R_FOOT)},
}

JOINT_NAMES = list(JOINT_TRIPLETS.keys())

# Anthropometry for the *driven* mass, from Winter ("Biomechanics and Motor
# Control of Human Movement") with self-consistent COM / radius-of-gyration
# pairs (I_prox = m*(k_frac*L)^2 = I_com + m*(com_frac*L)^2 holds).
#
# mass_frac: of body mass. com_frac / k_frac: COM distance and radius of
# gyration ABOUT THE JOINT as fractions of the *mass reference length* L_mass -
# the full driven-limb length (shoulder: shoulder->wrist; hip: hip->ankle),
# NOT the first bone segment. neutral_deg: measured flexion (180 - interior)
# at which the driven mass hangs gravity-neutral below the joint - 0 for
# elbow/knee/hip, ~180 for the shoulder (arm hanging reads ~170-180), ~90 for
# the dangling foot. len_default: fallback L_mass in meters.
SEGMENT_ANTHROPOMETRY = {
    "elbow": {"mass_frac": 0.022, "com_frac": 0.682, "k_frac": 0.827,
              "neutral_deg": 0.0, "len_default": 0.28, "name": "forearm+hand"},
    "knee": {"mass_frac": 0.061, "com_frac": 0.606, "k_frac": 0.735,
             "neutral_deg": 0.0, "len_default": 0.43, "name": "shank+foot"},
    "shoulder": {"mass_frac": 0.050, "com_frac": 0.530, "k_frac": 0.645,
                 "neutral_deg": 180.0, "len_default": 0.60, "name": "whole arm"},
    "hip": {"mass_frac": 0.161, "com_frac": 0.447, "k_frac": 0.560,
            "neutral_deg": 0.0, "len_default": 0.85, "name": "whole leg"},
    "ankle": {"mass_frac": 0.0145, "com_frac": 0.50, "k_frac": 0.690,
              "neutral_deg": 90.0, "len_default": 0.20, "name": "foot"},
}

# Landmark measuring the far end of the driven mass (per side), where it is not
# simply the joint triplet's distal reference: whole arm ends at the wrist,
# whole leg at the ankle.
MASS_END_LANDMARK = {
    "shoulder": {"left": L_WRIST, "right": R_WRIST},
    "hip": {"left": L_ANKLE, "right": R_ANKLE},
}

# Full-limb presets: joints extracted together from one recording.
LIMB_PRESETS = {
    "arm": ["shoulder", "elbow"],
    "leg": ["hip", "knee", "ankle"],
    # full body: every joint one sagittal video can support, one side.
    # Design order in the runner is DISTAL-FIRST so each proximal joint is
    # sized carrying the hardware and the recorded motion of everything below.
    "body": ["shoulder", "elbow", "hip", "knee", "ankle"],
}

# joint -> its distal descendants within the body chain, with the bone route
# used to compute where the descendant's hardware mass rides
BODY_CHILDREN = {
    "shoulder": ["elbow"],
    "hip": ["knee", "ankle"],
    "knee": ["ankle"],
}

# ---------------------------------------------------------------------------
# Movement task library
#
# Functional movements a prosthetic is expected to undergo. Each task has:
#   target: {"joint": name} or {"limb": name}
#   phases: list of dicts -
#     seconds: nominal duration (timed phase) or max wait (checked phase)
#     instruction: shown live on the guided recorder / in upload instructions
#     check (optional): requirement verified live from the webcam before the
#       phase completes - {"joint": j, "above": deg} / {"joint": j, "below": deg}
#       ("visible" checks that all task joints are trackable).
#   min_rom: per-joint range of motion (deg) a recording must contain to be
#     accepted - enforced again server-side after upload.
#
# Angles are the pipeline's measured flexion (180 - interior angle). Note the
# shoulder therefore READS ~170 hanging and DECREASES as the arm raises.
# ---------------------------------------------------------------------------

MOVEMENTS = {
    # ---- arm tasks --------------------------------------------------------
    "reach_pick": {
        "label": "Reach & pick up (arm)",
        "target": {"limb": "arm"},
        "description": "Reach forward to a table, grasp, lift to a shelf, set down.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, whole arm visible, arm relaxed",
             "check": {"visible": True}},
            {"seconds": 8, "instruction": "Reach forward and down, as if to an object on a table",
             "check": {"joint": "elbow", "above": 45}},
            {"seconds": 2, "instruction": "Close your hand around the imaginary object"},
            {"seconds": 8, "instruction": "Lift it up to shoulder height, like placing it on a shelf",
             "check": {"joint": "shoulder", "below": 125}},
            {"seconds": 6, "instruction": "Lower your arm and set the object back down",
             "check": {"joint": "shoulder", "above": 150}},
            {"seconds": 8, "instruction": "One more time: reach, grasp and lift it up",
             "check": {"joint": "shoulder", "below": 125}},
            {"seconds": 4, "instruction": "And lower back down slowly"},
        ],
        "min_rom": {"shoulder": 30, "elbow": 35},
    },
    "push": {
        "label": "Push (arm)",
        "target": {"limb": "arm"},
        "description": "Push something away at chest height - straighten the arm under control.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, hand up near your chest, elbow bent",
             "check": {"joint": "elbow", "above": 75}},
            {"seconds": 6, "instruction": "Push your hand straight forward until the arm is nearly straight",
             "check": {"joint": "elbow", "below": 30}},
            {"seconds": 5, "instruction": "Pull the hand back to your chest",
             "check": {"joint": "elbow", "above": 75}},
            {"seconds": 6, "instruction": "Push forward again, slow and controlled",
             "check": {"joint": "elbow", "below": 30}},
            {"seconds": 4, "instruction": "And relax the arm down"},
        ],
        "min_rom": {"shoulder": 15, "elbow": 45},
    },
    "lift_carry": {
        "label": "Lift / carry curl (elbow)",
        "target": {"joint": "elbow"},
        "description": "Curl a carried load up and lower it - the classic elbow duty cycle.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, whole arm visible, arm hanging straight",
             "check": {"joint": "elbow", "below": 25}},
            {"seconds": 6, "instruction": "Slowly curl your hand up to full bend",
             "check": {"joint": "elbow", "above": 95}},
            {"seconds": 5, "instruction": "Slowly lower until straight",
             "check": {"joint": "elbow", "below": 25}},
            {"seconds": 6, "instruction": "Curl up again", "check": {"joint": "elbow", "above": 95}},
            {"seconds": 5, "instruction": "And lower fully", "check": {"joint": "elbow", "below": 25}},
        ],
        "min_rom": {"elbow": 60},
    },
    # ---- leg tasks --------------------------------------------------------
    "walk_in_place": {
        "label": "Walk in place (leg)",
        "target": {"limb": "leg"},
        "description": "Marching steps - the gait duty cycle for hip, knee and ankle.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, whole leg visible; hold support if needed",
             "check": {"visible": True}},
            {"seconds": 6, "instruction": "March: lift your knee up high", "check": {"joint": "knee", "above": 50}},
            {"seconds": 4, "instruction": "Foot back down", "check": {"joint": "knee", "below": 20}},
            {"seconds": 6, "instruction": "Lift the knee again", "check": {"joint": "knee", "above": 50}},
            {"seconds": 4, "instruction": "Down again", "check": {"joint": "knee", "below": 20}},
            {"seconds": 6, "instruction": "One more high step", "check": {"joint": "knee", "above": 50}},
            {"seconds": 5, "instruction": "Down, then rise onto your toes and back"},
        ],
        "min_rom": {"hip": 20, "knee": 40, "ankle": 10},
    },
    "sit_stand": {
        "label": "Sit-to-stand (leg)",
        "target": {"limb": "leg"},
        "description": "Slow squats - the sit-down / stand-up duty cycle.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, whole leg visible; hold support for balance",
             "check": {"visible": True}},
            {"seconds": 8, "instruction": "Squat down slowly as if sitting on a chair",
             "check": {"joint": "knee", "above": 60}},
            {"seconds": 6, "instruction": "Stand back up slowly", "check": {"joint": "knee", "below": 20}},
            {"seconds": 8, "instruction": "One more slow squat", "check": {"joint": "knee", "above": 60}},
            {"seconds": 6, "instruction": "And stand up", "check": {"joint": "knee", "below": 20}},
        ],
        "min_rom": {"hip": 30, "knee": 50, "ankle": 8},
    },
    # ---- single-joint range checks ---------------------------------------
    "knee_bends": {
        "label": "Knee bends (knee)",
        "target": {"joint": "knee"},
        "description": "Standing knee flexion through the full comfortable range.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, whole leg visible; hold support",
             "check": {"visible": True}},
            {"seconds": 6, "instruction": "Bend the knee, heel up behind you", "check": {"joint": "knee", "above": 75}},
            {"seconds": 5, "instruction": "Lower the foot back down", "check": {"joint": "knee", "below": 20}},
            {"seconds": 6, "instruction": "Bend it up again", "check": {"joint": "knee", "above": 75}},
            {"seconds": 5, "instruction": "And down", "check": {"joint": "knee", "below": 20}},
        ],
        "min_rom": {"knee": 55},
    },
    "shoulder_raise": {
        "label": "Shoulder raises (shoulder)",
        "target": {"joint": "shoulder"},
        "description": "Straight-arm forward raises through the full range.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, arm hanging relaxed",
             "check": {"joint": "shoulder", "above": 150}},
            {"seconds": 7, "instruction": "Raise the straight arm forward and up",
             "check": {"joint": "shoulder", "below": 115}},
            {"seconds": 6, "instruction": "Lower it back to your side", "check": {"joint": "shoulder", "above": 150}},
            {"seconds": 7, "instruction": "Raise again, slow and smooth", "check": {"joint": "shoulder", "below": 115}},
            {"seconds": 4, "instruction": "And lower slowly"},
        ],
        "min_rom": {"shoulder": 40},
    },
    "hip_raise": {
        "label": "Leg raises (hip)",
        "target": {"joint": "hip"},
        "description": "Standing straight-leg raises.",
        "phases": [
            {"seconds": 6, "instruction": "Stand sideways, whole leg visible; hold support",
             "check": {"joint": "hip", "below": 12}},
            {"seconds": 7, "instruction": "Raise the fairly-straight leg forward",
             "check": {"joint": "hip", "above": 40}},
            {"seconds": 5, "instruction": "Lower it back down", "check": {"joint": "hip", "below": 12}},
            {"seconds": 7, "instruction": "Raise again", "check": {"joint": "hip", "above": 40}},
            {"seconds": 4, "instruction": "And down slowly"},
        ],
        "min_rom": {"hip": 30},
    },
    "heel_toe": {
        "label": "Heel & toe rocks (ankle)",
        "target": {"joint": "ankle"},
        "description": "Rise onto toes, rock back onto heels - push-off and clearance.",
        "phases": [
            {"seconds": 6, "instruction": "Stand or sit sideways, whole foot and shin visible",
             "check": {"visible": True}},
            {"seconds": 5, "instruction": "Rise onto your toes (point the foot down)",
             "check": {"joint": "ankle", "below": 80}},
            {"seconds": 5, "instruction": "Rock back, lifting toes toward the shin",
             "check": {"joint": "ankle", "above": 98}},
            {"seconds": 5, "instruction": "Point down once more", "check": {"joint": "ankle", "below": 80}},
            {"seconds": 4, "instruction": "And relax flat"},
        ],
        "min_rom": {"ankle": 15},
    },
}


def movements_for(target_kind: str, name: str):
    """Movement keys applicable to a joint or limb selection."""
    out = []
    for key, mv in MOVEMENTS.items():
        t = mv["target"]
        if target_kind == "limb" and t.get("limb") == name:
            out.append(key)
        elif target_kind == "joint" and t.get("joint") == name:
            out.append(key)
    return out

SAFETY_NOTE = ("Move slowly and only within a comfortable, pain-free range. "
               "Use support for balance on standing movements. Stop if anything hurts.")

# ---------------------------------------------------------------------------
# Guided FREE capture: when no task movement is selected, the recorder still
# guides the participant through the phases a joint design actually needs -
# a neutral calibration hold (gravity/offset), two slow full-range sweeps
# (geometry + ROM), fast reps (velocity/power class) and a mid-range hold
# (static holdability). Checks use each joint's measured-flexion convention
# (shoulder READS ~170 hanging and DECREASES as the arm raises).
# ---------------------------------------------------------------------------

_FC = {   # per joint: (neutral check, flexed check, neutral name, flexed name)
    "elbow": ({"joint": "elbow", "below": 25}, {"joint": "elbow", "above": 95},
              "arm hanging straight", "fully curled"),
    "knee": ({"joint": "knee", "below": 20}, {"joint": "knee", "above": 75},
             "leg straight", "heel lifted well up"),
    "shoulder": ({"joint": "shoulder", "above": 150},
                 {"joint": "shoulder", "below": 115},
                 "arm hanging relaxed", "arm raised forward/up"),
    "hip": ({"joint": "hip", "below": 12}, {"joint": "hip", "above": 40},
            "standing straight", "leg raised forward"),
    "ankle": ({"joint": "ankle", "above": 98}, {"joint": "ankle", "below": 80},
              "toes pulled up toward the shin", "up on your toes"),
}


def free_capture_phases(joint: str):
    neutral, flexed, n_name, f_name = _FC[joint]
    return [
        {"seconds": 5, "instruction":
            "Stand sideways, whole limb visible, good light",
         "check": {"visible": True}},
        {"seconds": 5, "instruction":
            f"Calibration: hold still, {n_name}", "check": neutral},
        {"seconds": 8, "instruction":
            f"SLOW sweep: move to {f_name}, as far as comfortable",
         "check": flexed},
        {"seconds": 6, "instruction": f"Slowly return: {n_name}",
         "check": neutral},
        {"seconds": 7, "instruction": "One more slow full sweep",
         "check": flexed},
        {"seconds": 5, "instruction": f"Return again: {n_name}",
         "check": neutral},
        {"seconds": 6, "instruction":
            "Now two QUICK comfortable reps (speed data)"},
        {"seconds": 4, "instruction":
            "Hold still at MID-range (holding-force data)"},
        {"seconds": 3, "instruction": "Relax"},
    ]


FREE_CAPTURE = {j: {"label": f"guided free capture ({j})",
                    "phases": free_capture_phases(j),
                    "min_rom": {j: {"elbow": 60, "knee": 50, "shoulder": 35,
                                    "hip": 25, "ankle": 14}[j]}}
                for j in _FC}


# Standard BlazePose skeleton edges, used only for drawing.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
