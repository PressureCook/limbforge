"""Stage 4: forward dynamics of the actuator-driven joint.

The distal segment is a rigid pendulum about the joint, driven by the linear
actuator through its configuration-dependent moment arm, against gravity and
light viscous damping. A PD servo with gravity feed-forward commands actuator
force; force and speed saturate at the actuator's limits (that saturation is
exactly what separates a good evolved geometry from a bad one).

Everything is vectorized over a population axis so the evolutionary loop can
integrate all candidate designs simultaneously.
"""

from dataclasses import dataclass

import numpy as np

from .joints import SEGMENT_ANTHROPOMETRY
from .mechanism import length_and_derivative

G = 9.81
DT_SIM = 1.0 / 240.0
OMEGA_C = 18.0          # rad/s, servo bandwidth
SETTLE_SKIP_S = 0.25    # exclude from tracking error (controller settle-in)
MAX_EVAL_DURATION = 12.0


@dataclass
class JointContext:
    """Physical context of the driven joint."""
    joint: str
    Lp: float           # m, proximal segment length
    Ld: float           # m, distal BONE segment length (mechanism/anchor scale)
    L_mass: float       # m, driven-mass reference length (whole limb for shoulder/hip)
    mass: float         # kg, driven mass
    com: float          # m, driven-mass COM distance from the joint
    inertia: float      # kg m^2 about the joint (from Winter radius of gyration)
    damping: float      # N m s
    theta_neutral: float  # rad, measured flexion where the mass hangs gravity-neutral

    @classmethod
    def from_pose(cls, joint: str, Lp: float, Ld: float, body_mass: float,
                  L_mass: float = None):
        antro = SEGMENT_ANTHROPOMETRY[joint]
        L_mass = L_mass or Ld
        m = antro["mass_frac"] * body_mass
        c = antro["com_frac"] * L_mass
        inertia = m * (antro["k_frac"] * L_mass) ** 2
        return cls(joint=joint, Lp=Lp, Ld=Ld, L_mass=L_mass, mass=m, com=c,
                   inertia=inertia, damping=0.02 + 0.15 * inertia,
                   theta_neutral=np.deg2rad(antro["neutral_deg"]))


def make_target(t_obs: np.ndarray, theta_obs_deg: np.ndarray, dt: float = DT_SIM,
                max_duration: float = MAX_EVAL_DURATION):
    """Resample the observed trajectory onto the simulation grid (radians)."""
    theta = np.deg2rad(theta_obs_deg)
    t_end = min(t_obs[-1], max_duration)
    ts = np.arange(0.0, t_end + 1e-9, dt)
    th_d = np.interp(ts, t_obs, theta)
    thdot_d = np.gradient(th_d, dt)
    return ts, th_d, thdot_d


def gravity_torque(theta, ctx: JointContext, extra_lever: float = 0.0):
    """Generalized gravity torque in measured-flexion coordinates.

    Gravity restores toward the joint's neutral pose theta_neutral (where the
    driven mass hangs below the joint): 0 for elbow/knee/hip, ~180 deg for the
    shoulder (measured flexion reads ~170-180 with the arm hanging), 90 deg for
    the dangling foot. Peak load is a quarter turn from neutral, e.g. the arm
    raised horizontal.  extra_lever adds payload mass*distance (kg m).

    Note the ankle model covers only the hanging foot's weight - standing
    ankle torque is ground-reaction dominated and NOT modeled.
    """
    lever = ctx.mass * ctx.com + extra_lever
    return -G * lever * np.sin(theta - ctx.theta_neutral)


def simulate_population(params: dict, ts, th_d, thdot_d, ctx: JointContext,
                        return_traces: bool = False, payload: np.ndarray = None,
                        payload_dist: float = None):
    """Integrate all designs against the target trajectory.

    params: dict of (P,) arrays with keys a, alpha, b, beta, F_max, v_max.
    payload: optional (T,) carried point mass in kg, acting at payload_dist
    meters from the joint (default ctx.L_mass, i.e. the hand/foot) - adds both
    gravity moment (m*d) and rotational inertia (m*d^2) over time.
    Returns dict of (P,) metric arrays; with return_traces=True also per-step
    traces (P, T) for theta, actuator length, axial force.
    """
    a, alpha = params["a"], params["alpha"]
    b, beta = params["b"], params["beta"]
    F_max, v_max = params["F_max"], params["v_max"]
    P = np.broadcast(a, b).size
    dt = ts[1] - ts[0]
    T = len(ts)

    I0, bv = ctx.inertia, ctx.damping
    if payload is None:
        payload = np.zeros(T)
    p_dist = ctx.L_mass if payload_dist is None else payload_dist
    Kp, Kd = OMEGA_C ** 2, 2.0 * OMEGA_C
    th_clip = (min(th_d.min() - 0.6, -0.3), max(th_d.max() + 0.6, np.pi))

    theta = np.full(P, th_d[0])
    omega = np.full(P, np.clip(thdot_d[0], -5, 5))
    sse = np.zeros(P)
    n_err = 0
    F_peak = np.zeros(P)
    F_demand_peak = np.zeros(P)
    v_peak = np.zeros(P)
    skip = int(SETTLE_SKIP_S / dt)

    if return_traces:
        tr_theta = np.zeros((P, T))
        tr_L = np.zeros((P, T))
        tr_F = np.zeros((P, T))

    for k in range(T):
        I = I0 + payload[k] * p_dist ** 2
        L, dL, r = length_and_derivative(a, alpha, b, beta, theta)
        tau_g = gravity_torque(theta, ctx, extra_lever=payload[k] * p_dist)
        tau_cmd = I * (Kp * (th_d[k] - theta) + Kd * (thdot_d[k] - omega)) - tau_g
        denom = np.where(np.abs(dL) > 1e-4, dL, np.where(dL >= 0, 1e-4, -1e-4))
        T_need = -tau_cmd / denom
        T_ax = np.clip(T_need, -F_max, F_max)
        tau_act = -T_ax * dL

        acc = (tau_act + tau_g - bv * omega) / I
        omega = omega + dt * acc
        omega_lim = v_max / np.maximum(r, 1e-3)
        omega = np.clip(omega, -omega_lim, omega_lim)
        theta = np.clip(theta + dt * omega, th_clip[0], th_clip[1])

        if k >= skip:
            sse += (theta - th_d[k]) ** 2
            n_err += 1
        F_peak = np.maximum(F_peak, np.abs(T_ax))
        F_demand_peak = np.maximum(F_demand_peak, np.abs(T_need))
        v_peak = np.maximum(v_peak, np.abs(dL * omega))
        if return_traces:
            tr_theta[:, k] = theta
            tr_L[:, k] = L
            tr_F[:, k] = T_ax

    rmse_deg = np.degrees(np.sqrt(sse / max(n_err, 1)))
    rmse_deg = np.where(np.isfinite(rmse_deg), rmse_deg, 1e6)
    out = {"rmse_deg": rmse_deg, "F_peak": F_peak, "v_peak": v_peak,
           "F_demand_peak": F_demand_peak}
    if return_traces:
        out["traces"] = {"t": ts, "theta": tr_theta, "L": tr_L, "F": tr_F,
                         "theta_target": th_d}
    return out


def simulate_final(mech, pose_t, pose_theta_deg, ctx: JointContext,
                   payload_t: np.ndarray = None, payload_kg: np.ndarray = None,
                   payload_dist: float = None):
    """Full-duration simulation of one concrete Mechanism, with traces.

    payload_t/payload_kg: optional carried-mass profile (interpolated onto the
    simulation grid), acting at payload_dist meters from the joint (default:
    ctx.L_mass, the hand/foot).
    """
    ts, th_d, thdot_d = make_target(pose_t, pose_theta_deg, max_duration=60.0)
    payload = None
    if payload_t is not None and payload_kg is not None:
        payload = np.interp(ts, payload_t, payload_kg)
    params = {k: np.array([getattr(mech, k)]) for k in
              ("a", "alpha", "b", "beta", "F_max", "v_max")}
    res = simulate_population(params, ts, th_d, thdot_d, ctx, return_traces=True,
                              payload=payload, payload_dist=payload_dist)
    traces = res["traces"]
    return {
        "t": traces["t"],
        "theta_sim_deg": np.degrees(traces["theta"][0]),
        "theta_target_deg": np.degrees(traces["theta_target"]),
        "L": traces["L"][0],
        "F": traces["F"][0],
        "rmse_deg": float(res["rmse_deg"][0]),
        "F_peak": float(res["F_peak"][0]),
        "F_demand_peak": float(res["F_demand_peak"][0]),
        "v_peak": float(res["v_peak"][0]),
    }
