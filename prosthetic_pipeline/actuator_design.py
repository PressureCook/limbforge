"""Stage 6: turn the evolved requirements into a real-world actuator design.

Takes the evolved mechanism (force/speed class, stroke, retracted length) plus
the force/velocity traces recorded while simulating YOUR movement, and produces
a buildable engineering spec from first principles:

- screw transmission sizing (lead screw vs ball screw, lead, diameter, efficiency
  from thread geometry and friction, self-locking check)
- gear ratio and motor class selection (torque/speed chain)
- electrical sizing (bus voltage, peak current) and duty analysis
- energy per recorded movement and battery-life estimate for both variants,
  computed by integrating the actual recorded force x velocity profile
- structural checks (rod buckling, clevis pin shear)
- packaging check against the retracted length the evolved geometry demands
- mass estimate and a dimensioned drawing

All heuristic constants (friction coefficients, motor classes, efficiencies)
are engineering estimates for early design exploration, labeled as such in the
output. Verify against vendor datasheets before building anything.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np

from .mechanism import DEAD_LENGTH_PARALLEL, anchor_A, anchor_B

# --- design assumptions (documented in the spec sheet) ------------------------
SF_FORCE = 1.5                 # service factor on the evolved force class
SPEED_MARGIN = 1.25            # margin on the evolved speed class
MU_LEADSCREW = 0.16 / np.cos(np.deg2rad(14.5))  # acme thread, steel/bronze, lubricated
ETA_BALLSCREW = (0.90, 0.85)   # forward, backdrive
ETA_GEAR_STAGE = 0.85
ETA_MOTOR = 0.80
N_MOTOR_MAX_RPM = 12000.0
N_SCREW_MAX_RPM = 4000.0
LEADS_MM = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
GEAR_RATIOS = [1.0, 3.7, 5.2, 14.0, 19.0]
IDLE_POWER_W = 0.35            # controller idle draw
BATTERY_WH = 24.4              # 3S li-ion, 2200 mAh (assumption for the estimate)
HOLD_SPEED_THRESH = 0.002      # m/s: below this the actuator is "holding"
E_STEEL = 200e9
RHO_STEEL = 7850.0
SHEAR_ALLOW = 210e6            # ~0.6 * yield of a 350 MPa pin steel

# (max continuous motor torque N m, class label, mass kg, motor constant N m/sqrt(W))
MOTOR_CLASSES = [
    (0.008, "22 mm BLDC class", 0.055, 0.0065),
    (0.025, "28-32 mm BLDC class", 0.110, 0.013),
    (0.060, "36 mm BLDC class", 0.210, 0.021),
    (0.120, "42 mm BLDC class", 0.360, 0.032),
]


@dataclass
class ScrewChoice:
    kind: str                  # "leadscrew" | "ballscrew"
    d_nom_mm: float
    d_pitch_mm: float
    lead_mm: float
    eta_fwd: float
    eta_back: float
    self_locking: bool


@dataclass
class DriveChoice:
    screw: ScrewChoice
    gear_ratio: float
    eta_gear: float
    tau_screw_peak_Nm: float
    tau_motor_peak_Nm: float
    tau_motor_cont_Nm: float
    n_screw_rpm: float
    n_motor_rpm: float
    motor_class: str
    motor_mass_kg: float
    motor_km: float


@dataclass
class VariantSpec:
    name: str
    drive: DriveChoice
    P_mech_peak_W: float
    P_elec_peak_W: float
    bus_voltage_V: float
    I_peak_A: float
    energy_per_recording_J: float
    avg_power_W: float
    hold_power_W: float
    battery_hours: float
    hold_fraction: float
    mass_est_kg: float
    notes: List[str] = field(default_factory=list)


def _screw_efficiency(lead_m, d_pitch_m, mu):
    lam = np.arctan(lead_m / (np.pi * d_pitch_m))
    t = np.tan(lam)
    eta_fwd = t * (1 - mu * t) / (t + mu)
    self_locking = t <= mu
    eta_back = 0.0 if self_locking else 0.9 * (t - mu) / (t * (1 + mu / max(t, 1e-9)))
    return max(eta_fwd, 0.0), max(eta_back, 0.0), self_locking


def _screw_diameter_mm(F_design):
    for tier, d in [(400, 6), (900, 8), (1600, 10)]:
        if F_design <= tier:
            return float(d)
    return 12.0



# --- custom-drive synthesis (beyond the catalogue) ---------------------------
# Motor scaling laws fitted to the catalogue classes (max fit error <= 14%):
# tau_cont ~ D^4.20, mass ~ D^2.90, km ~ D^2.46. When no catalogue combination
# fits an evolved requirement, a bespoke drive is SYNTHESIZED from these laws -
# custom high-helix screw lead, continuous (multi-stage) gear ratio, and a
# motor diameter solved from the required continuous torque. Clearly labelled
# "(custom)": buildable specifications, not shelf parts.
MOTOR_D_REF = 22.0
MOTOR_TAU_EXP, MOTOR_TAU_REF = 4.20, 0.008
MOTOR_MASS_EXP, MOTOR_MASS_REF = 2.90, 0.055
MOTOR_KM_EXP, MOTOR_KM_REF = 2.46, 0.0065
MOTOR_D_MAX = 80.0                 # mm - beyond this, report closest fit
CUSTOM_LEAD_MAX_FACTOR = 2.5       # high-helix ballscrew lead up to 2.5 x d
CUSTOM_N_BALL_RPM = 6000.0         # quality preloaded ballscrew
CUSTOM_G_MAX = 120.0               # three compound stages


def _motor_from_torque(tau_cont):
    """Synthesize a motor class for a required continuous torque."""
    D = MOTOR_D_REF * (max(tau_cont, 1e-6) / MOTOR_TAU_REF) ** (1 / MOTOR_TAU_EXP)
    mass = MOTOR_MASS_REF * (D / MOTOR_D_REF) ** MOTOR_MASS_EXP
    km = MOTOR_KM_REF * (D / MOTOR_D_REF) ** MOTOR_KM_EXP
    return D, mass, km


def _stages_eta(G):
    if G <= 1.0 + 1e-9:
        return 1.0
    n_stages = 1 if G <= 7 else (2 if G <= 49 else 3)
    return ETA_GEAR_STAGE ** n_stages


def _synthesize_drive(kind, F_design, F_rms, v_design):
    """Search the CONTINUOUS drive space for a bespoke combination that fits
    the evolved requirement: if the catalogue cannot build it, invent a drive
    that can, rather than failing the run. Fine grid over lead; gear ratio
    solved continuously per lead; motor diameter from the scaling law.
    Minimises motor mass, then ratio."""
    d_nom = _screw_diameter_mm(F_design)
    if F_design > 1600:
        d_nom = 12.0 if F_design <= 2600 else (16.0 if F_design <= 4000 else 20.0)
    d_pitch = 0.9 * d_nom
    lead_cap = (CUSTOM_LEAD_MAX_FACTOR if kind == "ballscrew" else 1.3) * d_nom
    rpm_cap = CUSTOM_N_BALL_RPM if kind == "ballscrew" else N_SCREW_MAX_RPM
    best = None
    for lead_mm in np.arange(2.0, min(lead_cap, 32.0) + 0.25, 0.5):
        lead = lead_mm / 1000.0
        if kind == "ballscrew":
            eta_fwd, eta_back = ETA_BALLSCREW
            self_lock = False
        else:
            eta_fwd, eta_back, self_lock = _screw_efficiency(
                lead, d_pitch / 1000.0, MU_LEADSCREW)
            if eta_fwd < 0.15:
                continue
        n_screw = v_design / lead * 60.0
        if n_screw > rpm_cap:
            continue
        tau_screw = 1.1 * F_design * lead / (2 * np.pi * eta_fwd)
        tau_screw_cont = 1.1 * F_rms * lead / (2 * np.pi * eta_fwd)
        G_hi = min(N_MOTOR_MAX_RPM / max(n_screw, 1e-9), CUSTOM_G_MAX)
        for G in np.geomspace(1.0, max(G_hi, 1.0), 24):
            eta_g = _stages_eta(G)
            tau_m = tau_screw / (G * eta_g)
            tau_m_cont = tau_screw_cont / (G * eta_g)
            D, mass, km = _motor_from_torque(max(tau_m_cont, tau_m / 3.5))
            if D > MOTOR_D_MAX:
                continue
            score = (mass, G)
            if best is None or score < best[0]:
                screw = ScrewChoice(kind, d_nom, d_pitch, float(lead_mm),
                                    round(eta_fwd, 3), round(eta_back, 3),
                                    self_lock)
                label = f"{D:.0f} mm BLDC (custom wind)"
                best = (score, DriveChoice(
                    screw=screw, gear_ratio=round(float(G), 1), eta_gear=eta_g,
                    tau_screw_peak_Nm=tau_screw, tau_motor_peak_Nm=tau_m,
                    tau_motor_cont_Nm=tau_m_cont, n_screw_rpm=n_screw,
                    n_motor_rpm=n_screw * G, motor_class=label,
                    motor_mass_kg=mass, motor_km=km))
    return best[1] if best else None


def _motor_diameter_mm(motor_class: str) -> float:
    """Diameter from a class label - catalogue AND custom labels."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", motor_class or "")
    return float(m.group(1)) if m else 30.0


def _pick_drive(kind, F_design, F_rms, v_design):
    """Enumerate lead x gear ratio, keep the feasible option with the smallest motor."""
    d_nom = _screw_diameter_mm(F_design)
    d_pitch = 0.9 * d_nom
    best = None
    for lead_mm in LEADS_MM:
        if lead_mm > 1.3 * d_nom:   # keep to manufacturable helix angles
            continue
        lead = lead_mm / 1000.0
        if kind == "ballscrew":
            eta_fwd, eta_back = ETA_BALLSCREW
            self_lock = False
        else:
            eta_fwd, eta_back, self_lock = _screw_efficiency(lead, d_pitch / 1000.0, MU_LEADSCREW)
            if eta_fwd < 0.15:
                continue
        n_screw = v_design / lead * 60.0
        if n_screw > N_SCREW_MAX_RPM:
            continue
        tau_screw = 1.1 * F_design * lead / (2 * np.pi * eta_fwd)
        tau_screw_cont = 1.1 * F_rms * lead / (2 * np.pi * eta_fwd)
        for G in GEAR_RATIOS:
            eta_g = 1.0 if G == 1.0 else ETA_GEAR_STAGE
            n_motor = n_screw * G
            if n_motor > N_MOTOR_MAX_RPM:
                continue
            tau_m = tau_screw / (G * eta_g)
            tau_m_cont = tau_screw_cont / (G * eta_g)
            mclass = next((mc for mc in MOTOR_CLASSES if tau_m_cont <= mc[0]), None)
            if mclass is None or tau_m > 3.5 * mclass[0]:
                continue
            score = (tau_m, G)  # smallest motor, then simplest gearing
            if best is None or score < best[0]:
                screw = ScrewChoice(kind, d_nom, d_pitch, lead_mm, round(eta_fwd, 3),
                                    round(eta_back, 3), self_lock)
                best = (score, DriveChoice(
                    screw=screw, gear_ratio=G, eta_gear=eta_g,
                    tau_screw_peak_Nm=tau_screw, tau_motor_peak_Nm=tau_m,
                    tau_motor_cont_Nm=tau_m_cont, n_screw_rpm=n_screw, n_motor_rpm=n_motor,
                    motor_class=mclass[1], motor_mass_kg=mclass[2], motor_km=mclass[3]))
    return best[1] if best else None


def _energy_analysis(drive: DriveChoice, F_tr, v_tr, dt):
    """Integrate electrical power over the recorded movement for one drive variant."""
    screw = drive.screw
    lead = screw.lead_mm / 1000.0
    eta_tot = screw.eta_fwd * drive.eta_gear * ETA_MOTOR
    # power delivered to the limb: tau*omega = -T*dL/dtheta*omega = -F*v
    # (axial tension convention, see mechanism.py: tau = -T * dL/dtheta)
    P_mech = -F_tr * v_tr
    moving = np.abs(v_tr) > HOLD_SPEED_THRESH
    holding = ~moving

    P = np.full_like(F_tr, IDLE_POWER_W)
    P[moving & (P_mech > 0)] += P_mech[moving & (P_mech > 0)] / eta_tot
    # lowering / backdriven phases: friction still costs some drive power
    lower_factor = 0.15 if screw.self_locking else 0.25
    P[moving & (P_mech <= 0)] += lower_factor * np.abs(P_mech[moving & (P_mech <= 0)])
    # holding: self-locking screws hold for free; others burn motor current
    # (load back-drives the screw, then the gear stage, to the motor shaft)
    if not screw.self_locking:
        tau_m_hold = (np.abs(F_tr[holding]) * lead * screw.eta_back * drive.eta_gear
                      / (2 * np.pi * drive.gear_ratio))
        P[holding] += (tau_m_hold / drive.motor_km) ** 2
    E = float(np.sum(P) * dt)
    T = len(F_tr) * dt
    avg_W = E / max(T, 1e-9)
    hold_W = float(np.mean(P[holding])) if holding.any() else 0.0
    return E, avg_W, hold_W, float(holding.mean())


def _mass_estimate(drive: DriveChoice, stroke, L_ext):
    d_root = 0.8 * drive.screw.d_nom_mm / 1000.0
    screw_m = RHO_STEEL * np.pi * (d_root / 2) ** 2 * (stroke + 0.04)
    rod_m = RHO_STEEL * np.pi * 0.003 ** 2 * L_ext * 0.7  # tubular rod
    housing_m = 0.055 * (stroke + DEAD_LENGTH_PARALLEL) / 0.08
    return drive.motor_mass_kg + screw_m + rod_m + housing_m + 0.020 + 0.018  # + nut + clevises


def design_actuator(mech, final, ctx):
    """Produce the real-world spec: requirements, both screw variants, recommendation."""
    F_class, v_class = mech.F_max, mech.v_max
    F_design = SF_FORCE * F_class
    v_design = SPEED_MARGIN * v_class
    t, F_tr, L_tr = final["t"], final["F"], final["L"]
    dt = float(t[1] - t[0])
    v_tr = np.gradient(L_tr, dt)
    F_peak_rec = float(np.max(np.abs(F_tr)))
    F_rms = float(np.sqrt(np.mean(F_tr ** 2)))
    P_mech_peak = float(np.max(np.abs(F_tr * v_tr)))

    variants = []
    for kind in ("leadscrew", "ballscrew"):
        drive = _pick_drive(kind, F_design, SF_FORCE * F_rms, v_design)
        synthesized = False
        if drive is None:
            # the catalogue cannot build this requirement: synthesize a custom
            # combination (high-helix screw / continuous ratio / custom motor
            # wind) instead of failing the run
            drive = _synthesize_drive(kind, F_design, SF_FORCE * F_rms, v_design)
            synthesized = drive is not None
        if drive is None:
            continue
        eta_tot = drive.screw.eta_fwd * drive.eta_gear * ETA_MOTOR
        P_elec_peak = F_design * v_design / eta_tot
        V_bus = 12.0 if P_elec_peak < 60 else 24.0
        E, avg_W, hold_W, hold_frac = _energy_analysis(drive, F_tr, v_tr, dt)
        mass = _mass_estimate(drive, mech.stroke, mech.L_ext)
        notes = []
        if drive.screw.self_locking:
            notes.append("self-locking: holds position with zero motor power and on power loss (fail-safe)")
        else:
            notes.append("backdrivable: needs motor current or a power-off brake (+~15 g) to hold")
        if synthesized:
            notes.append(
                "CUSTOM DRIVE: no catalogue screw/gear/motor combination met "
                f"this requirement; synthesized {drive.screw.d_nom_mm:.0f} x "
                f"{drive.screw.lead_mm:.1f} mm screw, {drive.gear_ratio:.1f}:1 "
                f"ratio, {drive.motor_class} from motor scaling laws "
                "(tau~D^4.2) - source as custom parts, verify thermals")
        variants.append(VariantSpec(
            name=kind, drive=drive,
            P_mech_peak_W=P_mech_peak, P_elec_peak_W=P_elec_peak,
            bus_voltage_V=V_bus, I_peak_A=P_elec_peak / V_bus,
            energy_per_recording_J=E, avg_power_W=avg_W, hold_power_W=hold_W,
            battery_hours=BATTERY_WH / max(avg_W, 1e-6), hold_fraction=hold_frac,
            mass_est_kg=mass, notes=notes))

    if not variants:
        # unreachable for any decodable requirement now that synthesis exists;
        # kept as a guard for out-of-envelope inputs
        raise RuntimeError(
            "no drive fits even with custom synthesis "
            f"(F_design {F_design:.0f} N, v_design {v_design*1000:.0f} mm/s "
            f"exceed the physical envelope: motor <= {MOTOR_D_MAX:.0f} mm, "
            f"lead <= {CUSTOM_LEAD_MAX_FACTOR:.1f} x d at "
            f"{CUSTOM_N_BALL_RPM:.0f} rpm)")

    # recommendation: energy decides, with a fail-safe bias toward self-locking
    by_name = {v.name: v for v in variants}
    if "leadscrew" in by_name and "ballscrew" in by_name:
        lead_v, ball_v = by_name["leadscrew"], by_name["ballscrew"]
        if by_name["leadscrew"].drive.screw.self_locking and \
                lead_v.energy_per_recording_J <= 1.6 * ball_v.energy_per_recording_J:
            rec, why = "leadscrew", (
                "self-locking (fail-safe hold, zero hold power) at "
                f"{lead_v.energy_per_recording_J/ball_v.energy_per_recording_J:.1f}x the "
                "ball-screw energy - a good trade for a prosthetic joint")
        else:
            rec, why = "ballscrew", (
                "meaningfully lower energy per movement; add a power-off brake for fail-safe holding")
    else:
        rec, why = variants[0].name, "only feasible variant"

    # structural checks (on the recommended variant's geometry)
    rod_d = 0.006 if F_design < 600 else 0.008
    I_rod = np.pi * rod_d ** 4 / 64
    P_cr = np.pi ** 2 * E_STEEL * I_rod / max(mech.L_ext, 1e-3) ** 2
    pin_d = 0.004 if F_design < 600 else 0.005
    pin_SF = SHEAR_ALLOW / (F_design / (2 * np.pi * pin_d ** 2 / 4))

    body_len = mech.stroke + DEAD_LENGTH_PARALLEL
    packaging_margin_mm = (mech.L_ret - body_len) * 1000

    return _to_native({
        "requirements": {
            "from_evolution": {"F_class_N": F_class, "v_class_m_s": v_class,
                               "stroke_mm": mech.stroke * 1000, "L_ret_mm": mech.L_ret * 1000,
                               "L_ext_mm": mech.L_ext * 1000},
            "from_recording": {"F_peak_N": F_peak_rec, "F_rms_N": F_rms,
                               "P_mech_peak_W": P_mech_peak,
                               "recording_s": float(t[-1] - t[0])},
            "design_targets": {"F_design_N": F_design, "v_design_m_s": v_design,
                               "service_factor": SF_FORCE},
        },
        "variants": {v.name: _variant_dict(v) for v in variants},
        "recommended": rec,
        "recommendation_reason": why,
        "structure": {
            "rod_diameter_mm": rod_d * 1000, "rod_buckling_SF": float(P_cr / F_design),
            "clevis_pin_mm": pin_d * 1000, "pin_shear_SF": float(pin_SF),
        },
        "packaging": {
            "body_length_mm": body_len * 1000,
            "required_retracted_mm": mech.L_ret * 1000,
            "margin_mm": packaging_margin_mm,
            "fits": bool(packaging_margin_mm >= -1.0),
            "layout": "motor parallel to screw (piggyback), single belt/gear stage",
        },
        "assumptions": {
            "friction_mu_leadscrew": round(MU_LEADSCREW, 3),
            "eta_ballscrew": ETA_BALLSCREW, "eta_gear_stage": ETA_GEAR_STAGE,
            "eta_motor": ETA_MOTOR, "battery_Wh": BATTERY_WH,
            "idle_power_W": IDLE_POWER_W,
            "disclaimer": ("first-principles estimates for design exploration; "
                           "verify every number against vendor datasheets before building"),
        },
    })


# ---------------------------------------------------------------------------
# requirement-driven sizing (SPEC R-18/R-19): one section per actuator, plus a
# rotary path (gearbox + motor) alongside the existing screw chain
# ---------------------------------------------------------------------------

ROTARY_RATIOS = [16.0, 25.0, 50.0, 100.0, 160.0]
ETA_PLANETARY = 0.85


def _size_rotary(req) -> dict:
    """Gearbox ratio + motor class for a rotary requirement (torque in N m)."""
    tau_design = SF_FORCE * req.peak_effort
    omega_design = SPEED_MARGIN * req.peak_rate
    best = None
    for G_r in ROTARY_RATIOS:
        n_motor = omega_design * G_r * 60 / (2 * np.pi)     # rpm
        if n_motor > N_MOTOR_MAX_RPM:
            continue
        tau_m = tau_design / (G_r * ETA_PLANETARY)
        tau_m_cont = SF_FORCE * req.rms_effort / (G_r * ETA_PLANETARY)
        mclass = next((mc for mc in MOTOR_CLASSES if tau_m_cont <= mc[0]), None)
        if mclass is None or tau_m > 3.5 * mclass[0]:
            continue
        if best is None or tau_m < best["tau_motor_peak_Nm"]:
            best = {"ratio": G_r, "n_motor_rpm": n_motor,
                    "tau_motor_peak_Nm": tau_m, "tau_motor_cont_Nm": tau_m_cont,
                    "motor_class": mclass[1], "motor_mass_kg": mclass[2]}
    if best is None:
        return {"feasible": False, "reason": "no gearbox/motor combination "
                f"reaches {tau_design:.0f} N m at {omega_design:.1f} rad/s"}
    # energy over the recording: power to the limb = -u * qdot (uniform convention)
    P_mech = -req.effort_trace * req.rate_trace
    P = np.where(P_mech > 0, P_mech / (ETA_PLANETARY * ETA_MOTOR),
                 0.25 * np.abs(P_mech)) + IDLE_POWER_W
    E = float(np.sum(P) * req.dt)
    T_rec = len(req.effort_trace) * req.dt
    mass = best["motor_mass_kg"] + 0.10 + 0.004 * tau_design   # + gearbox
    P_elec_peak = tau_design * omega_design / (ETA_PLANETARY * ETA_MOTOR)
    V = 12.0 if P_elec_peak < 60 else 24.0
    return _to_native({
        "feasible": True, "kind": "rotary",
        "tau_design_Nm": tau_design, "omega_design_rad_s": omega_design,
        "sweep_deg": float(np.degrees(req.travel)),
        "drive": best,
        "bus_voltage_V": V, "I_peak_A": P_elec_peak / V,
        "energy_per_recording_J": E, "avg_power_W": E / max(T_rec, 1e-9),
        "mass_est_kg": mass,
        "notes": ["not self-locking: needs a power-off brake for fail-safe holding",
                  "reflected rotor inertia not modelled in the simulation"],
    })


def _size_linear_req(req) -> dict:
    """Linear/tendon requirement through the existing screw chain."""
    F_design = SF_FORCE * req.peak_effort
    v_design = SPEED_MARGIN * req.peak_rate
    variants = {}
    for kind in ("leadscrew", "ballscrew"):
        drive = _pick_drive(kind, F_design, SF_FORCE * max(req.rms_effort, 1.0),
                            v_design)
        synthesized = False
        if drive is None:
            drive = _synthesize_drive(kind, F_design,
                                      SF_FORCE * max(req.rms_effort, 1.0),
                                      v_design)
            synthesized = drive is not None
        if drive is None:
            continue
        E, avg_W, hold_W, hold_frac = _energy_analysis(
            drive, req.effort_trace, req.rate_trace, req.dt)
        eta_tot = drive.screw.eta_fwd * drive.eta_gear * ETA_MOTOR
        P_elec_peak = F_design * v_design / eta_tot
        V = 12.0 if P_elec_peak < 60 else 24.0
        variants[kind] = _to_native({
            "drive": _variant_drive_dict(drive),
            "bus_voltage_V": V, "I_peak_A": P_elec_peak / V,
            "energy_per_recording_J": E, "avg_power_W": avg_W,
            "hold_power_W": hold_W, "hold_fraction": hold_frac,
            "mass_est_kg": _mass_estimate(drive, req.travel,
                                          req.q_max or req.travel + 0.05),
            "self_locking": drive.screw.self_locking,
        })
    if not variants:
        return {"feasible": False, "kind": req.kind,
                "reason": f"no screw drive reaches {F_design:.0f} N at "
                          f"{v_design*1000:.0f} mm/s"}
    if "leadscrew" in variants and "ballscrew" in variants and \
            variants["leadscrew"]["self_locking"] and \
            variants["leadscrew"]["energy_per_recording_J"] <= \
            1.6 * variants["ballscrew"]["energy_per_recording_J"]:
        rec = "leadscrew"
    else:
        rec = min(variants, key=lambda k: variants[k]["energy_per_recording_J"])
    body = req.travel + DEAD_LENGTH_PARALLEL
    return _to_native({
        "feasible": True, "kind": req.kind,
        "F_design_N": F_design, "v_design_m_s": v_design,
        "stroke_mm": req.travel * 1000, "retracted_mm": (req.q_min or 0) * 1000,
        "variants": variants, "recommended": rec,
        "packaging": {"body_length_mm": body * 1000,
                      "margin_mm": ((req.q_min or 0) - body) * 1000,
                      "fits": bool((req.q_min or 0) - body >= -0.001)},
        "notes": (["tension-only tendon: pretension or antagonist required"]
                  if req.kind == "tendon" else []),
    })


def _variant_drive_dict(drive) -> dict:
    d = asdict(drive)
    return d


def design_from_requirements(reqs, ctx) -> dict:
    """SPEC §7.2: per-actuator sizing + system-level rollup."""
    sections = []
    for req in reqs:
        s = _size_rotary(req) if req.kind == "rotary" else _size_linear_req(req)
        s["label"] = req.label
        s["peak_effort"] = req.peak_effort
        s["rms_effort"] = req.rms_effort
        sections.append(s)
    def _vals(s):  # rotary stores rollup values at top level, linear per variant
        if not s.get("feasible"):
            return 0.0, 0.0, 0.0
        v = s if s["kind"] == "rotary" else s["variants"][s["recommended"]]
        return (v.get("energy_per_recording_J", 0.0),
                v.get("mass_est_kg", 0.0), v.get("avg_power_W", 0.0))

    vals = [_vals(s) for s in sections]
    total_E = sum(v[0] for v in vals)
    total_m = sum(v[1] for v in vals)
    avg_W = sum(v[2] for v in vals)
    return _to_native({
        "schema": "requirements-v1",
        "actuators": sections,
        "rollup": {
            "n_actuators": len(sections),
            "all_feasible": all(s.get("feasible") for s in sections),
            "total_mass_kg": total_m,
            "total_energy_per_recording_J": total_E,
            "battery_hours": BATTERY_WH / max(avg_W, 1e-6),
        },
        "assumptions": {"service_factor": SF_FORCE, "battery_Wh": BATTERY_WH,
                        "disclaimer": "first-principles estimates; verify "
                                      "against vendor datasheets"},
    })


def requirements_markdown(spec: dict, topology_kind: str) -> str:
    lines = [f"# Actuator specification - {topology_kind} topology\n"]
    for s in spec["actuators"]:
        lines.append(f"## Actuator {s['label']} ({s.get('kind', '?')})")
        if not s.get("feasible"):
            lines += [f"- **NOT FEASIBLE**: {s['reason']}", ""]
            continue
        if s["kind"] == "rotary":
            d = s["drive"]
            lines += [
                f"- output: **{s['tau_design_Nm']:.1f} N m** peak at "
                f"**{s['omega_design_rad_s']:.1f} rad/s**, sweep {s['sweep_deg']:.0f} deg",
                f"- gearbox {d['ratio']:.0f}:1 planetary -> motor "
                f"{d['n_motor_rpm']:.0f} rpm, {d['motor_class']}",
                f"- electrical: {s['bus_voltage_V']:.0f} V, peak {s['I_peak_A']:.1f} A",
                f"- energy {s['energy_per_recording_J']:.1f} J per recording; "
                f"mass ~{s['mass_est_kg']*1000:.0f} g",
            ]
        else:
            rec = s["variants"][s["recommended"]]
            dr = rec["drive"]["screw"]
            lines += [
                f"- class: **{s['F_design_N']:.0f} N / {s['v_design_m_s']*1000:.0f} mm/s**, "
                f"stroke {s['stroke_mm']:.0f} mm, retracted {s['retracted_mm']:.0f} mm",
                f"- recommended: **{s['recommended']}** {dr['d_nom_mm']:.0f} x "
                f"{dr['lead_mm']:.0f} mm, {rec['drive']['gear_ratio']:.1f}:1, "
                f"{rec['drive']['motor_class']}",
                f"- electrical: {rec['bus_voltage_V']:.0f} V, peak {rec['I_peak_A']:.1f} A; "
                f"energy {rec['energy_per_recording_J']:.1f} J; mass ~{rec['mass_est_kg']*1000:.0f} g",
                f"- packaging: margin {s['packaging']['margin_mm']:.0f} mm "
                + ("**FITS**" if s["packaging"]["fits"] else "**DOES NOT FIT**"),
            ]
        for nline in s.get("notes", []):
            lines.append(f"- {nline}")
        lines.append("")
    r = spec["rollup"]
    lines += [
        "## System rollup",
        f"- {r['n_actuators']} actuator(s), all feasible: **{r['all_feasible']}**",
        f"- total actuator mass ~{r['total_mass_kg']*1000:.0f} g",
        f"- total energy {r['total_energy_per_recording_J']:.1f} J per recording; "
        f"battery ~{r['battery_hours']:.1f} h",
        "",
        f"*{spec['assumptions']['disclaimer']}*",
    ]
    return "\n".join(lines) + "\n"


def _variant_dict(v: VariantSpec) -> dict:
    d = asdict(v)
    return d


def _to_native(obj):
    """Recursively convert numpy scalars/arrays so the spec is JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# output files
# ---------------------------------------------------------------------------

def write_spec_files(spec: dict, mech, ctx, out_dir: str, th_mid: float = None):
    paths = {}
    paths["actuator_json"] = os.path.join(out_dir, "actuator_spec.json")
    with open(paths["actuator_json"], "w") as f:
        json.dump(spec, f, indent=2)
    paths["actuator_md"] = os.path.join(out_dir, "actuator_spec.md")
    with open(paths["actuator_md"], "w") as f:
        f.write(spec_markdown(spec))
    paths["actuator_drawing"] = os.path.join(out_dir, "actuator_drawing.png")
    draw_actuator(spec, mech, ctx, paths["actuator_drawing"], th_mid=th_mid)
    return paths


def spec_markdown(spec: dict) -> str:
    req, pack = spec["requirements"], spec["packaging"]
    ev, rec_d, tgt = req["from_evolution"], req["from_recording"], req["design_targets"]
    lines = [
        "# Real-world actuator specification\n",
        f"Recommended variant: **{spec['recommended']}** - {spec['recommendation_reason']}\n",
        "## Requirements (from your movement data)",
        f"- evolved class: {ev['F_class_N']:.0f} N, {ev['v_class_m_s']*1000:.0f} mm/s, "
        f"stroke {ev['stroke_mm']:.0f} mm, retracted {ev['L_ret_mm']:.0f} mm",
        f"- recorded while tracking your movement: peak {rec_d['F_peak_N']:.0f} N, "
        f"RMS {rec_d['F_rms_N']:.0f} N, peak mech power {rec_d['P_mech_peak_W']:.1f} W "
        f"over {rec_d['recording_s']:.1f} s",
        f"- design targets (with x{tgt['service_factor']:.1f} service factor): "
        f"**{tgt['F_design_N']:.0f} N**, **{tgt['v_design_m_s']*1000:.0f} mm/s**",
        "",
    ]
    for name, v in spec["variants"].items():
        d, s = v["drive"], v["drive"]["screw"]
        star = " (RECOMMENDED)" if name == spec["recommended"] else ""
        lines += [
            f"## Variant: {name}{star}",
            f"- screw: {s['d_nom_mm']:.0f} mm {s['kind']}, {s['lead_mm']:.0f} mm lead, "
            f"efficiency {s['eta_fwd']:.2f} fwd / {s['eta_back']:.2f} back"
            + (", **self-locking**" if s["self_locking"] else ""),
            f"- gearing: {d['gear_ratio']:.1f}:1 -> screw {d['n_screw_rpm']:.0f} rpm, "
            f"motor {d['n_motor_rpm']:.0f} rpm",
            f"- motor: {d['motor_class']} - peak {d['tau_motor_peak_Nm']*1000:.0f} mN m, "
            f"continuous {d['tau_motor_cont_Nm']*1000:.0f} mN m",
            f"- electrical: {v['bus_voltage_V']:.0f} V bus, peak {v['P_elec_peak_W']:.0f} W "
            f"/ {v['I_peak_A']:.1f} A",
            f"- energy for one recording of your movement: {v['energy_per_recording_J']:.1f} J "
            f"(avg {v['avg_power_W']:.2f} W, holding {100*v['hold_fraction']:.0f}% of the time "
            f"at {v['hold_power_W']:.2f} W)",
            f"- battery estimate: **{v['battery_hours']:.1f} h** of repeating this movement "
            f"on a {spec['assumptions']['battery_Wh']:.0f} Wh pack",
            f"- estimated mass: {v['mass_est_kg']*1000:.0f} g (+/-40%)",
        ] + [f"- {n}" for n in v["notes"]] + [""]
    st = spec["structure"]
    lines += [
        "## Structure and packaging",
        f"- rod {st['rod_diameter_mm']:.0f} mm: buckling safety factor {st['rod_buckling_SF']:.0f}",
        f"- clevis pins {st['clevis_pin_mm']:.0f} mm: shear safety factor {st['pin_shear_SF']:.1f}",
        f"- body (stroke + dead length): {pack['body_length_mm']:.0f} mm vs required retracted "
        f"{pack['required_retracted_mm']:.0f} mm -> margin {pack['margin_mm']:.0f} mm "
        + ("**FITS**" if pack["fits"] else "**DOES NOT FIT - re-evolve with tighter packaging**"),
        f"- layout: {pack['layout']}",
        "",
        "## Safety / integration notes",
        "- add hard end-stops plus software soft limits inside the evolved stroke",
        "- limit switch or absolute encoder for homing; current limiting in the drive",
        "- if ballscrew variant is chosen, fit a power-off brake for fail-safe holding",
        "",
        f"*{spec['assumptions']['disclaimer']}*",
    ]
    return "\n".join(lines) + "\n"


def draw_actuator(spec: dict, mech, ctx, path: str, th_mid: float = None):
    """Datasheet-style spec sheet: dimensioned views, title block, ratings table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = spec["variants"][spec["recommended"]]
    s = v["drive"]["screw"]
    L_ret, stroke, L_ext = mech.L_ret * 1000, mech.stroke * 1000, mech.L_ext * 1000
    body = spec["packaging"]["body_length_mm"]
    barrel_d = s["d_nom_mm"] + 14
    rod_d = spec["structure"]["rod_diameter_mm"]
    pin_d = spec["structure"]["clevis_pin_mm"]
    motor_d = _motor_diameter_mm(v["drive"]["motor_class"])

    fig = plt.figure(figsize=(11, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.62, 1.3], hspace=0.34)
    ax1 = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    def dim(ax, x0, x1, y, label, color="k", above=False):
        ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", lw=0.9, color=color))
        ax.text((x0 + x1) / 2, y + (2.5 if above else -6.5), label, ha="center",
                fontsize=8, color=color)

    # --- main dimensioned side view (retracted + extended rod ghost) ----------
    y0 = 0
    eye_r = max(5.0, pin_d / 2 + 3)
    ax1.add_patch(plt.Rectangle((0, y0 - barrel_d / 2), body, barrel_d,
                                fc="#c8d6e5", ec="k", lw=1.2, zorder=3))
    ax1.add_patch(plt.Rectangle((body, y0 - rod_d / 2), L_ret - body - eye_r - 2, rod_d,
                                fc="#eeeeee", ec="k", lw=1.0, zorder=2))
    ax1.add_patch(plt.Rectangle((2, y0 + barrel_d / 2), 0.55 * body, motor_d,
                                fc="#9aa5b1", ec="k", lw=1.0, zorder=3))
    ax1.text(2 + 0.27 * body, y0 + barrel_d / 2 + motor_d / 2,
             v["drive"]["motor_class"].replace(" class", ""), ha="center",
             va="center", fontsize=8, zorder=4)
    # clevis eyes with pin holes at both ends
    for cx in (-eye_r + 1, L_ret):
        ax1.add_patch(plt.Circle((cx, y0), eye_r, fc="#c8d6e5", ec="k", lw=1.2, zorder=2))
        ax1.add_patch(plt.Circle((cx, y0), pin_d / 2, fc="w", ec="k", lw=1.0, zorder=4))
    # extended-state ghost
    ext_y = y0 - barrel_d / 2 - 20
    ax1.add_patch(plt.Rectangle((body, ext_y - rod_d / 2), L_ext - body - eye_r - 2, rod_d,
                                fc="none", ec="#888888", ls="--", lw=1.0, zorder=1))
    ax1.add_patch(plt.Circle((L_ext, ext_y), eye_r, fc="none", ec="#888888", ls="--", zorder=1))
    ax1.plot([0, body], [ext_y, ext_y], color="#bbbbbb", lw=0.8)
    # dimensions
    dim(ax1, 0, body, y0 - barrel_d / 2 - 8, f"body {body:.0f}")
    dim(ax1, -eye_r + 1, L_ret, y0 + barrel_d / 2 + motor_d + 9,
        f"retracted pin-to-pin {L_ret:.0f}", above=True)
    dim(ax1, L_ret, L_ext, ext_y - 9, f"stroke {stroke:.0f}", color="#555555")
    dim(ax1, -eye_r + 1, L_ext, ext_y - 22, f"extended pin-to-pin {L_ext:.0f}", color="#555555")
    ax1.annotate(f"2x pin holes {pin_d:.1f}", (-eye_r + 1, y0 - pin_d / 2),
                 (-eye_r - 4, y0 - barrel_d / 2 - 16), fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.annotate(f"barrel {barrel_d:.0f}", (body * 0.75, y0 + barrel_d / 2),
                 (body * 0.75 + 16, y0 + barrel_d / 2 + 16), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.annotate(f"rod {rod_d:.0f}", (body + 12, y0 + rod_d / 2),
                 (body + 26, y0 + barrel_d / 2 + 8), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.set_xlim(-eye_r - 16, L_ext + 24)
    ax1.set_ylim(ext_y - 34, y0 + barrel_d / 2 + motor_d + 20)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(
        f"PROSTHETIC LINEAR ACTUATOR  -  {ctx.joint.upper()}  -  "
        f"{spec['recommended'].upper()} {s['d_nom_mm']:.0f} x {s['lead_mm']:.0f}  -  "
        "dimensions in mm", fontsize=11, weight="bold", loc="left")

    # --- ratings table ---------------------------------------------------------
    axt.axis("off")
    rows = [
        ("peak force", f"{spec['requirements']['design_targets']['F_design_N']:.0f} N",
         "rated speed", f"{v['drive']['n_screw_rpm'] * s['lead_mm'] / 60:.0f} mm/s"),
        ("stroke", f"{stroke:.0f} mm",
         "gear stage", f"{v['drive']['gear_ratio']:.1f}:1  (motor {v['drive']['n_motor_rpm']:.0f} rpm)"),
        ("supply", f"{v['bus_voltage_V']:.0f} V dc, {v['I_peak_A']:.1f} A peak",
         "motor", v["drive"]["motor_class"]),
        ("screw efficiency", f"{s['eta_fwd']:.2f} fwd / {s['eta_back']:.2f} back"
         + ("  SELF-LOCKING" if s["self_locking"] else ""),
         "mass (est.)", f"{v['mass_est_kg']*1000:.0f} g"),
        ("duty (recorded)", f"avg {v['avg_power_W']:.1f} W, {v['battery_hours']:.1f} h on "
         f"{spec['assumptions']['battery_Wh']:.0f} Wh",
         "holding", "zero-power (self-locking)" if s["self_locking"]
         else f"{v['hold_power_W']:.2f} W motor hold (or brake)"),
    ]
    table = axt.table(
        cellText=[[a, b, c, d] for a, b, c, d in rows],
        cellLoc="left", loc="center",
        colWidths=[0.16, 0.34, 0.16, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        if c in (0, 2):
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eef1f5")
    axt.set_title("ratings", fontsize=9, loc="left")

    # --- panel 2: installation on the joint, mid flexion ----------------------
    th = np.deg2rad(45.0) if th_mid is None else float(th_mid)
    Lp, Ld = ctx.Lp * 1000, ctx.Ld * 1000
    A = np.array(anchor_A(mech.a, mech.alpha)) * 1000
    B = np.array(anchor_B(mech.b, mech.beta, th)) * 1000
    tip = Ld * np.array([np.cos(np.pi - th), np.sin(np.pi - th)])
    ax2.plot([0, Lp], [0, 0], color="#444444", lw=9, solid_capstyle="round")
    ax2.plot([0, tip[0]], [0, tip[1]], color="#e05c2a", lw=9, solid_capstyle="round")
    u = (B - A) / np.linalg.norm(B - A)
    be = A + u * min(0.85 * L_ret, 0.9 * np.linalg.norm(B - A))
    ax2.plot([A[0], be[0]], [A[1], be[1]], color="#3a6ea5", lw=10, solid_capstyle="butt")
    ax2.plot([be[0], B[0]], [be[1], B[1]], color="#9bb7d4", lw=4)
    ax2.plot([A[0], B[0]], [A[1], B[1]], "ko", ms=6)
    ax2.plot(0, 0, "o", color="k", ms=11)
    ax2.plot(0, 0, "o", color="w", ms=5)
    ax2.annotate("", (0, -14), (A[0], -14), arrowprops=dict(arrowstyle="<->", lw=1))
    ax2.text(A[0] / 2, -20, f"a = {mech.a*1000:.0f} mm @ {np.degrees(mech.alpha):+.0f} deg",
             ha="center", fontsize=9)
    ax2.annotate(f"b = {mech.b*1000:.0f} mm @ {np.degrees(mech.beta):+.0f} deg",
                 B, B + np.array([12, 18]), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=1))
    if abs(A[1]) > 6:
        ax2.plot([A[0], A[0]], [0, A[1]], color="#444444", lw=2)
        ax2.annotate(f"bracket {abs(A[1]):.0f} mm", (A[0], A[1] * 0.75),
                     (A[0] + 30, A[1] * 0.75 - 18), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.set_title(
        f"installation on the {ctx.joint} (mid flexion) - proximal segment fixed, mm scale",
        fontsize=10)
    pts = np.vstack([[0, 0], [Lp, 0], tip, A, B])
    lo, hi = pts.min(axis=0) - 45, pts.max(axis=0) + 45
    ax2.set_xlim(lo[0], hi[0])
    ax2.set_ylim(lo[1], hi[1])
    ax2.set_aspect("equal")
    ax2.axis("off")

    # --- title block -----------------------------------------------------------
    fig.text(0.06, 0.015,
             f"PROSTHETIC PIPELINE - ACTUATOR SPEC SHEET   |   joint: {ctx.joint}   |   "
             f"variant: {spec['recommended']}   |   "
             f"packaging margin: {spec['packaging']['margin_mm']:.0f} mm   |   "
             "estimates for design exploration - verify against vendor datasheets",
             fontsize=7.5, color="#555555")
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return path
