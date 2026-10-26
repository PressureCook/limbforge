"""Reduced-EOM simulator over transmission tables (SPEC §5.4, R-6/R-7/R-10).

Integrates, vectorised over a batch of same-structure candidates (a species):

    I_eff(th)·th'' + 1/2·dI/dth·th'^2 = G(th) + S(th) + sum_i r_i(th)·u_i − b·th'

Tables are (P, N) arrays gathered with manual linear interpolation (np.interp
is 1-D only). Payload enters through the table's payload columns so a carried
mass adds both gravity moment and inertia exactly as in the closed-form model.
The controller, saturation and metrics mirror simulate.simulate_population.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .simulate import G as GRAV
from .simulate import OMEGA_C, SETTLE_SKIP_S

# cheap drivetrain classes for the fitness energy term (SPEC-2 §4.1). Keyed on
# kind AND drive class so a self-locking leadscrew is distinguishable from a
# backdrivable ballscrew inside a species: (eta_fwd, backdrive_loss, hold_W/N).
# The exact figures still come from actuator_design for the winner; the
# cheap-vs-exact gap is reported there (E-7).
DRIVE_CLASSES = {
    ("linear", "lead"): (0.35, 0.15, 0.000),   # self-locking: free holding
    ("linear", "ball"): (0.72, 0.25, 0.004),
    ("tendon", "winch"): (0.55, 0.20, 0.002),
    ("rotary", "gear"): (0.65, 0.30, 0.050),
}
ETA_CHEAP = {"linear": 0.55, "tendon": 0.60, "rotary": 0.65}   # kind fallback
HOLD_W_PER_EFFORT = {"linear": 0.004, "tendon": 0.004, "rotary": 0.05}
HOLD_RATE_THRESH = 2e-3


def drive_params_for(kinds, drive_genes=None):
    """(eta, backdrive_loss, hold_coef) arrays for a candidate's actuators.

    drive_genes: optional per-actuator gene in [0,1]; for linear actuators
    < 0.5 selects the self-locking leadscrew class, else ballscrew."""
    eta, beta, hold = [], [], []
    for i, k in enumerate(kinds):
        if k == "linear":
            gd = 0.75 if drive_genes is None else float(drive_genes[i])
            key = ("linear", "lead" if gd < 0.5 else "ball")
        elif k == "tendon":
            key = ("tendon", "winch")
        else:
            key = ("rotary", "gear")
        e, b, h = DRIVE_CLASSES[key]
        eta.append(e)
        beta.append(b)
        hold.append(h)
    return np.array(eta), np.array(beta), np.array(hold)


@dataclass
class TableBatch:
    """Same-structure candidates stacked along axis 0."""
    theta: np.ndarray        # (N,) shared grid
    q: np.ndarray            # (P, n_act, N)
    r: np.ndarray            # (P, n_act, N)
    I_eff: np.ndarray        # (P, N)
    dI: np.ndarray           # (P, N)
    Gt: np.ndarray           # (P, N)
    St: np.ndarray           # (P, N)
    pay_dydth: np.ndarray    # (P, N)
    pay_dpdth2: np.ndarray   # (P, N)
    u_max: np.ndarray        # (P, n_act)
    rate_max: np.ndarray     # (P, n_act)
    unilateral: np.ndarray   # (n_act,) bool
    kinds: tuple = ()        # (n_act,) actuator kinds
    eta: np.ndarray = None   # (P, n_act) drivetrain efficiency (cheap class)
    beta: np.ndarray = None  # (P, n_act) backdrive loss factor
    hold: np.ndarray = None  # (P, n_act) holding power per N of effort

    @classmethod
    def stack(cls, tables: List, u_max: np.ndarray, rate_max: np.ndarray,
              drive_genes=None):
        t0 = tables[0]
        P = len(tables)
        eta = np.zeros((P, len(t0.actuator_kinds)))
        beta = np.zeros_like(eta)
        hold = np.zeros_like(eta)
        for p in range(P):
            dg = None if drive_genes is None else drive_genes[p]
            eta[p], beta[p], hold[p] = drive_params_for(t0.actuator_kinds, dg)
        return cls(
            theta=t0.theta,
            q=np.stack([t.q for t in tables]),
            r=np.stack([t.r for t in tables]),
            I_eff=np.stack([t.I_eff for t in tables]),
            dI=np.stack([t.dI_dtheta for t in tables]),
            Gt=np.stack([t.G for t in tables]),
            St=np.stack([t.S for t in tables]),
            pay_dydth=np.stack([t.payload_dydth for t in tables]),
            pay_dpdth2=np.stack([t.payload_dpdth2 for t in tables]),
            u_max=np.asarray(u_max, dtype=float),
            rate_max=np.asarray(rate_max, dtype=float),
            unilateral=t0.unilateral.copy(),
            kinds=tuple(t0.actuator_kinds),
            eta=eta, beta=beta, hold=hold,
        )


def _gather(tab: np.ndarray, idx: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Linear interp of (P, [n_act,] N) tables at per-candidate positions."""
    if tab.ndim == 2:
        a = np.take_along_axis(tab, idx[:, None], axis=1)[:, 0]
        b = np.take_along_axis(tab, (idx + 1)[:, None], axis=1)[:, 0]
        return a + frac * (b - a)
    a = np.take_along_axis(tab, idx[:, None, None], axis=2)[:, :, 0]
    b = np.take_along_axis(tab, (idx + 1)[:, None, None], axis=2)[:, :, 0]
    return a + frac[:, None] * (b - a)


def allocate_greedy(tau_cmd, r, u_max, unilateral):
    """Capacity-weighted effort allocation (§5.6 population policy).

    tau_cmd (P,), r (P, n_act), u_max (P, n_act). Exact for n_act == 1.
    Returns u (P, n_act) with |u_i| <= u_max_i (tendons: 0 <= u_i)."""
    cap = np.abs(r) * u_max                       # torque capacity per actuator
    total = cap.sum(axis=1, keepdims=True)
    w = np.where(total > 1e-12, cap / np.maximum(total, 1e-12),
                 1.0 / r.shape[1])
    r_safe = np.where(np.abs(r) > 1e-6, r, np.where(r >= 0, 1e-6, -1e-6))
    u = w * tau_cmd[:, None] / r_safe
    lo = np.where(unilateral[None, :], 0.0, -u_max)
    u = np.clip(u, lo, u_max)
    # one residual redistribution among unsaturated actuators
    resid = tau_cmd - (r * u).sum(axis=1)
    slack_hi = (u_max - u) * np.abs(r)
    slack_lo = (u - lo) * np.abs(r)
    slack = np.where(resid[:, None] * r >= 0, slack_hi, slack_lo)
    stot = slack.sum(axis=1, keepdims=True)
    share = np.where(stot > 1e-12, slack / np.maximum(stot, 1e-12), 0.0)
    u = np.clip(u + share * resid[:, None] / r_safe, lo, u_max)
    return u


def allocate_qp(tau_cmd, r, u_max, unilateral):
    """Exact allocation (§5.6): min sum u_i^2 s.t. sum r_i u_i = tau, box.

    KKT: u_i = clip(lambda*r_i, lo_i, hi_i) with g(lambda) = sum r_i u_i
    monotone nondecreasing - solved by bisection, fully vectorised."""
    lo = np.where(unilateral[None, :], 0.0, -u_max)
    lam_lo = np.full(tau_cmd.shape, -1e7)
    lam_hi = np.full(tau_cmd.shape, 1e7)
    lam = np.zeros_like(tau_cmd)
    for _ in range(60):
        lam = 0.5 * (lam_lo + lam_hi)
        u = np.clip(lam[:, None] * r, lo, u_max)
        g = (r * u).sum(axis=1)
        too_high = g > tau_cmd
        lam_hi = np.where(too_high, lam, lam_hi)
        lam_lo = np.where(too_high, lam_lo, lam)
    return np.clip(lam[:, None] * r, lo, u_max)


def simulate_tables(batch: TableBatch, ts, th_d, thdot_d, ctx,
                    return_traces: bool = False, payload: np.ndarray = None,
                    allocator: str = "greedy", err_mask: np.ndarray = None):
    # err_mask: optional (T,) bool - timesteps that count toward the tracking
    # error (SPEC-2 E-15 hold-out; None = all)
    """Vectorised reduced-EOM integration of a same-structure batch (R-10)."""
    P, n_act, N = batch.q.shape
    dt = ts[1] - ts[0]
    T = len(ts)
    th0, dth = batch.theta[0], batch.theta[1] - batch.theta[0]
    if payload is None:
        payload = np.zeros(T)

    Kp, Kd = OMEGA_C ** 2, 2.0 * OMEGA_C
    bv = ctx.damping
    th_clip = (min(th_d.min() - 0.6, batch.theta[0]),
               max(th_d.max() + 0.6, batch.theta[-1]))

    kinds = batch.kinds or tuple(["linear"] * n_act)
    if batch.eta is not None:
        eta_pc, beta_pc, hold_pc = batch.eta, batch.beta, batch.hold
    else:
        e, b, h = drive_params_for(kinds)
        eta_pc = np.broadcast_to(e, (P, n_act))
        beta_pc = np.broadcast_to(b, (P, n_act))
        hold_pc = np.broadcast_to(h, (P, n_act))

    theta = np.full(P, th_d[0])
    omega = np.full(P, float(np.clip(thdot_d[0], -5, 5)))
    sse = np.zeros(P)
    n_err = 0
    u_peak = np.zeros((P, n_act))
    u_demand_peak = np.zeros((P, n_act))
    rate_peak = np.zeros((P, n_act))
    energy = np.zeros(P)
    skip = int(SETTLE_SKIP_S / dt)

    if return_traces:
        tr_theta = np.zeros((P, T))
        tr_u = np.zeros((P, n_act, T))
        tr_q = np.zeros((P, n_act, T))

    for k in range(T):
        pos = np.clip((theta - th0) / dth, 0.0, N - 1 - 1e-9)
        idx = pos.astype(np.int64)
        frac = pos - idx

        r = _gather(batch.r, idx, frac)              # (P, n_act)
        I = _gather(batch.I_eff, idx, frac) + payload[k] * _gather(batch.pay_dpdth2, idx, frac)
        dI = _gather(batch.dI, idx, frac)
        Gk = _gather(batch.Gt, idx, frac) - payload[k] * GRAV * _gather(batch.pay_dydth, idx, frac)
        Sk = _gather(batch.St, idx, frac)

        tau_cmd = I * (Kp * (th_d[k] - theta) + Kd * (thdot_d[k] - omega)) - (Gk + Sk)
        if allocator == "qp":
            u = allocate_qp(tau_cmd, r, batch.u_max, batch.unilateral)
            # demand = the SAME policy with the box lifted (SPEC-2 §4.4 /
            # REVIEW-3): unbounded QP allocation is u_i = tau r_i / sum r^2
            r2 = (r * r).sum(axis=1, keepdims=True)
            u_need = tau_cmd[:, None] * r / np.maximum(r2, 1e-12)
        else:
            u = allocate_greedy(tau_cmd, r, batch.u_max, batch.unilateral)
            cap = np.abs(r) * batch.u_max
            tot = cap.sum(axis=1, keepdims=True)
            w = np.where(tot > 1e-12, cap / np.maximum(tot, 1e-12),
                         1.0 / r.shape[1])
            r_safe = np.where(np.abs(r) > 1e-6, r,
                              np.where(r >= 0, 1e-6, -1e-6))
            u_need = w * tau_cmd[:, None] / r_safe
        tau_act = (r * u).sum(axis=1)

        acc = (tau_act + Gk + Sk - bv * omega - 0.5 * dI * omega ** 2) / I
        omega = omega + dt * acc
        # zero-capacity and slack unilateral actuators must not throttle the
        # joint (SPEC-2 G5). The threshold is CAPACITY-RELATIVE (REVIEW-3 H5:
        # an 8 uN actuator is physically zero, an absolute 1e-6 N floor is a
        # knife-edge that re-imposes the rate limit one mutation later).
        cap_floor = np.maximum(1e-3, 1e-3 * batch.u_max.max(axis=1, keepdims=True))
        active = (batch.u_max > cap_floor) & (
            ~batch.unilateral[None, :] | (np.abs(u) > 1e-3 * batch.u_max))
        with np.errstate(divide="ignore"):
            lims = batch.rate_max / np.maximum(np.abs(r), 1e-4)
        lims = np.where(active, lims, np.inf)
        omega_lim = lims.min(axis=1)
        omega = np.clip(omega, -omega_lim, omega_lim)
        theta = np.clip(theta + dt * omega, th_clip[0], th_clip[1])

        if k >= skip and (err_mask is None or err_mask[k]):
            sse += (theta - th_d[k]) ** 2
            n_err += 1
        rates = np.abs(r * omega[:, None])
        u_peak = np.maximum(u_peak, np.abs(u))
        u_demand_peak = np.maximum(u_demand_peak, np.abs(u_need))
        rate_peak = np.maximum(rate_peak, rates)
        # per-actuator energy (SPEC-2 §4.1, REVIEW-3 H4): positive work costs
        # 1/eta of its class; NEGATIVE (backdriven) work costs a friction
        # fraction beta - this is what separates the old net-torque formula
        # from a real per-actuator model, and what makes tendon-antagonist
        # co-contraction and drive-class choice visible to fitness. Holding
        # costs hold_coef * |u| (zero for the self-locking leadscrew class).
        P_i = u * r * omega[:, None]
        P_cost = (np.maximum(P_i, 0.0) / eta_pc
                  + beta_pc * np.maximum(-P_i, 0.0))
        holding = rates < HOLD_RATE_THRESH
        P_hold = np.where(holding, hold_pc * np.abs(u), 0.0)
        energy += (P_cost + P_hold).sum(axis=1) * dt
        if return_traces:
            tr_theta[:, k] = theta
            tr_u[:, :, k] = u
            tr_q[:, :, k] = _gather(batch.q, idx, frac)

    rmse_deg = np.degrees(np.sqrt(sse / max(n_err, 1)))
    out = {
        "rmse_deg": np.where(np.isfinite(rmse_deg), rmse_deg, 1e6),
        "u_peak": u_peak, "u_demand_peak": u_demand_peak,
        "rate_peak": rate_peak, "energy_J": energy,
    }
    if return_traces:
        out["traces"] = {"t": ts, "theta": tr_theta, "u": tr_u, "q": tr_q,
                         "theta_target": th_d}
    return out
