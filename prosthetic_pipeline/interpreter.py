"""The generic planar mechanism interpreter (SPEC-2 §2, E-1/E-2/E-3/E-6).

One solver for every topology. A topology DECLARES its structure as a batched
GraphSpec (bodies, revolute joints, actuator attachment points); this module:

- assembles the loop-closure residual f(x, theta) = 0 and its analytic
  Jacobians, vectorised over the population axis (E-6: no per-candidate
  Python inside assembly - the only Python loops are over the joint list,
  whose length is a structural constant of the species),
- Newton-solves the theta sweep with warm starting,
- derives x' and x'' by first- and second-order implicit differentiation
  (reusing the factorised Jacobian; SPEC-2 §2.3),
- produces every table quantity ANALYTICALLY (E-3: no np.gradient):
  I_eff, dI_eff/dtheta, G, q_i, r_i from body pose derivatives (§2.4),
- reports measured conditioning (E-5) and applies branch-continuity
  invalidation (§3.1: a flip invalidates the remainder of the sweep).

Frames: GROUND fixed at (0,0,0); OUTPUT pose (0,0,pi-theta); couplers carry
(x, y, phi) unknowns. Gravity acts along g_hat = [cos(th_n), -sin(th_n)]
(the per-joint neutral convention, applied uniformly to every body).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .simulate import G as GRAV
from .topology import TransmissionTable

GROUND, OUTPUT = 0, 1          # body indices; couplers are 2, 3, ...
NEWTON_TOL = 1e-11   # polish to keep the 1e-10 oracle contract
NEWTON_ITERS = 12
COND_MAX = 1e8
BRANCH_K = 6.0
BRANCH_FLOOR = 0.15


@dataclass
class JointDecl:
    """One revolute joint: world anchors coincide. Anchors are (P, 2) local."""
    body_a: int
    anchor_a: np.ndarray
    body_b: int
    anchor_b: np.ndarray


@dataclass
class ActuatorDecl:
    kind: str                          # linear | tendon | rotary
    body_a: int = GROUND
    pt_a: Optional[np.ndarray] = None  # (P, 2) local
    body_b: int = OUTPUT
    pt_b: Optional[np.ndarray] = None


@dataclass
class GraphSpec:
    """Batched, structure-uniform graph for one species (P candidates)."""
    P: int
    n_couplers: int
    coupler_com: np.ndarray            # (P, K, 2) local COM
    coupler_mass: np.ndarray           # (P, K)
    coupler_Icom: np.ndarray           # (P, K)
    joints: List[JointDecl] = field(default_factory=list)
    actuators: List[ActuatorDecl] = field(default_factory=list)

    def kutzbach(self) -> int:
        """M = 3(n_bodies - 1) - 2 n_joints, output revolute included (E-2)."""
        n_bodies = 2 + self.n_couplers
        n_joints = 1 + len(self.joints)      # + the anatomical output revolute
        return 3 * (n_bodies - 1) - 2 * n_joints

    def is_square(self) -> bool:
        return 3 * self.n_couplers == 2 * len(self.joints)


def _rot(phi):
    c, s = np.cos(phi), np.sin(phi)
    out = np.empty(np.shape(phi) + (2, 2))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def _drot(phi):
    c, s = np.cos(phi), np.sin(phi)
    out = np.empty(np.shape(phi) + (2, 2))
    out[..., 0, 0] = -s
    out[..., 0, 1] = -c
    out[..., 1, 0] = c
    out[..., 1, 1] = -s
    return out


def _apply(M, v):
    """(..., 2, 2) @ (..., 2) -> (..., 2)."""
    return np.einsum("...ij,...j->...i", M, v)


class _Poses:
    """Body poses + first/second theta-derivatives at one grid point."""

    def __init__(self, spec: GraphSpec, theta: float, X, Xp, Xpp):
        P, K = spec.P, spec.n_couplers
        phi_out = np.pi - theta
        self.p = np.zeros((P, 2 + K, 2))
        self.phi = np.zeros((P, 2 + K))
        self.pp = np.zeros((P, 2 + K, 2))       # dp/dtheta
        self.phip = np.zeros((P, 2 + K))
        self.ppp = np.zeros((P, 2 + K, 2))      # d2p/dtheta2
        self.phipp = np.zeros((P, 2 + K))
        self.phi[:, OUTPUT] = phi_out
        self.phip[:, OUTPUT] = -1.0
        if K:
            self.p[:, 2:] = X.reshape(P, K, 3)[:, :, :2]
            self.phi[:, 2:] = X.reshape(P, K, 3)[:, :, 2]
            self.pp[:, 2:] = Xp.reshape(P, K, 3)[:, :, :2]
            self.phip[:, 2:] = Xp.reshape(P, K, 3)[:, :, 2]
            self.ppp[:, 2:] = Xpp.reshape(P, K, 3)[:, :, :2]
            self.phipp[:, 2:] = Xpp.reshape(P, K, 3)[:, :, 2]

    def point(self, body, loc):
        """World point and its first two theta-derivatives for (P,2) local loc."""
        R = _rot(self.phi[:, body])
        Rp = _drot(self.phi[:, body])
        w = self.p[:, body] + _apply(R, loc)
        wp = self.pp[:, body] + _apply(Rp, loc) * self.phip[:, body, None]
        wpp = (self.ppp[:, body]
               - _apply(R, loc) * (self.phip[:, body, None] ** 2)   # R'' = -R
               + _apply(Rp, loc) * self.phipp[:, body, None])
        return w, wp, wpp


def _assemble(spec: GraphSpec, theta: float, X):
    """Residual f (P, 2J), Jacobians J_x (P, 2J, 3K) and J_theta (P, 2J).

    Rotation matrices for ALL bodies are built once per call (H1 fix: the
    previous version rebuilt 2x2 matrices per joint side via nested np.stack).
    """
    P, K, J = spec.P, spec.n_couplers, len(spec.joints)
    poses = _Poses(spec, theta, X, np.zeros((P, 3 * K)), np.zeros((P, 3 * K)))
    Rall = _rot(poses.phi)          # (P, 2+K, 2, 2)
    Rdall = _drot(poses.phi)
    f = np.zeros((P, 2 * J))
    Jx = np.zeros((P, 2 * J, 3 * K))
    Jth = np.zeros((P, 2 * J))
    for j, jd in enumerate(spec.joints):
        rb = slice(2 * j, 2 * j + 2)
        for body, loc, s in ((jd.body_a, jd.anchor_a, 1.0),
                             (jd.body_b, jd.anchor_b, -1.0)):
            w = poses.p[:, body] + _apply(Rall[:, body], loc)
            f[:, rb] += s * w
            if body >= 2:
                k = body - 2
                Jx[:, rb, 3 * k:3 * k + 2] += s * np.eye(2)[None, :, :]
                Jx[:, 2 * j:2 * j + 2, 3 * k + 2] += \
                    s * _apply(Rdall[:, body], loc)
            elif body == OUTPUT:   # phi_out = pi - theta -> dphi/dtheta = -1
                Jth[:, rb] += -s * _apply(Rdall[:, body], loc)
    return f, Jx, Jth


def _second_rhs(spec: GraphSpec, poses: _Poses):
    """RHS of J_x x'' = -sum s R''(phi) loc (phi')^2  (SPEC-2 §2.3)."""
    P, J = spec.P, len(spec.joints)
    h = np.zeros((P, 2 * J))
    for j, jd in enumerate(spec.joints):
        rb = slice(2 * j, 2 * j + 2)
        for body, loc, s in ((jd.body_a, jd.anchor_a, 1.0),
                             (jd.body_b, jd.anchor_b, -1.0)):
            if body == GROUND:
                continue
            R = _rot(poses.phi[:, body])          # R'' = -R
            h[:, rb] += s * (-_apply(R, loc)) * (poses.phip[:, body, None] ** 2)
    return -h


def solve_tables(spec: GraphSpec, genomes, topo, ctx, theta_grid,
                 X0: Optional[np.ndarray] = None) -> List[TransmissionTable]:
    """Sweep the grid; return one TransmissionTable per candidate."""
    from .topologies import _spring_terms   # torsional springs at the output

    P, K, J = spec.P, spec.n_couplers, len(spec.joints)
    n = len(theta_grid)
    n_act = len(spec.actuators)
    if not spec.is_square():
        raise ValueError(
            f"mechanism is not single-DOF: {3*spec.n_couplers} unknowns vs "
            f"{2*len(spec.joints)} equations (Kutzbach M={spec.kutzbach()})")

    tn = ctx.theta_neutral
    g_hat = np.array([np.cos(tn), -np.sin(tn)])
    out_com = np.stack([np.full(P, ctx.com), np.zeros(P)], -1)
    m_out = ctx.mass
    I_out_com = ctx.inertia - ctx.mass * ctx.com ** 2

    q = np.zeros((P, n_act, n))
    r = np.zeros((P, n_act, n))
    I_eff = np.zeros((P, n))
    dI = np.zeros((P, n))
    Gt = np.zeros((P, n))
    valid = np.ones((P, n), dtype=bool)
    cond = np.ones((P, n))

    # geometry capture for the universal gate (V8/V9): sampled link/actuator/
    # bone segments in world coordinates, filled for EVERY topology so the
    # gate never depends on an optional per-topology hook (REVIEW-3 H2).
    geom_stride = max(1, n // 16)
    geom_idx = list(range(0, n, geom_stride))
    n_bone_actuator_segs = 2 + sum(1 for a in spec.actuators
                                   if a.kind in ("linear", "tendon"))
    coupler_seg_locs = _coupler_segment_locs(spec)     # list of (k, locA, locB)
    n_segs = n_bone_actuator_segs + len(coupler_seg_locs)
    geom_segs = np.zeros((P, len(geom_idx), n_segs, 2, 2))
    geom_pos = {gi: si for si, gi in enumerate(geom_idx)}

    X = (X0.copy() if X0 is not None else np.zeros((P, 3 * K)))
    flipped = np.zeros(P, dtype=bool)
    prev_step = np.full(P, np.nan)
    eyeK = np.eye(3 * K)[None] if K else None

    for i, th in enumerate(theta_grid):
        if K:
            # --- warm-started Newton with early exit (H1a) -------------------
            dead = np.zeros(P, dtype=bool)
            conv = np.zeros(P, dtype=bool)
            res_prev = np.full(P, np.inf)
            stall = np.zeros(P, dtype=np.int8)
            for it in range(NEWTON_ITERS):
                f, Jx, _ = _assemble(spec, th, X)
                bad = (~np.isfinite(f).all(axis=1)
                       | ~np.isfinite(Jx).all(axis=(1, 2)))
                dead |= bad
                res = np.linalg.norm(np.where(dead[:, None], 0.0, f), axis=1)
                conv = (res < NEWTON_TOL) & ~dead
                # stall detection: a candidate whose residual stops halving is
                # not going to assemble here - retire it so the batch can exit
                # early instead of burning the full iteration cap (H1a)
                if it >= 2:
                    stall = np.where(res > 0.5 * res_prev, stall + 1, 0).astype(np.int8)
                    dead |= (stall >= 2) & ~conv
                res_prev = res
                if np.all(conv | dead):
                    break
                Jx_s = np.where(dead[:, None, None], eyeK, Jx)
                f_s = np.where((dead | conv)[:, None], 0.0, f)
                try:
                    dx = np.linalg.solve(Jx_s, f_s[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    # rare path: isolate singular members, retry (H1c: the
                    # determinant screen is no longer in the hot loop)
                    with np.errstate(over="ignore", invalid="ignore"):
                        det = np.linalg.det(Jx_s)
                    sing = ~np.isfinite(det) | (np.abs(det) < 1e-15)
                    dead |= sing
                    Jx_s = np.where(dead[:, None, None], eyeK, Jx_s)
                    f_s = np.where(dead[:, None], 0.0, f_s)
                    dx = np.linalg.solve(Jx_s, f_s[..., None])[..., 0]
                X = np.clip(X - dx, -50.0, 50.0)
            f, Jx, Jth = _assemble(spec, th, X)
            bad = (~np.isfinite(f).all(axis=1)
                   | ~np.isfinite(Jx).all(axis=(1, 2)))
            dead |= bad
            res = np.where(dead, np.inf,
                           np.linalg.norm(np.where(dead[:, None], 0.0, f),
                                          axis=1))
            valid[:, i] &= res < NEWTON_TOL
            Jx_g = np.where(dead[:, None, None], eyeK, Jx)
            sv = np.linalg.svd(Jx_g, compute_uv=False)
            cond[:, i] = np.where(dead, COND_MAX * 10,
                                  sv[:, 0] / np.maximum(sv[:, -1], 1e-300))
            valid[:, i] &= cond[:, i] < COND_MAX
            Jx_safe = Jx_g + np.where(cond[:, i] >= COND_MAX, 1e-9, 0.0)[
                :, None, None] * eyeK
            Xp = np.linalg.solve(Jx_safe, -Jth[..., None])[..., 0]
            poses = _Poses(spec, th, X, Xp, np.zeros((P, 3 * K)))
            Xpp = np.linalg.solve(Jx_safe, _second_rhs(spec, poses)[..., None])[..., 0]
            poses = _Poses(spec, th, X, Xp, Xpp)
            if i:   # branch continuity (§3.1): flip poisons the remainder
                step = np.linalg.norm(X - X_prev, axis=1)
                thr = np.maximum(BRANCH_FLOOR,
                                 BRANCH_K * np.where(np.isnan(prev_step), 0.05,
                                                     prev_step))
                flipped |= step > thr
                prev_step = step
            valid[:, i] &= ~flipped
            X_prev = X.copy()
        else:
            poses = _Poses(spec, th, np.zeros((P, 0)), np.zeros((P, 0)),
                           np.zeros((P, 0)))
            X_prev = X

        # --- reduced inertia, its analytic derivative, gravity (§2.4) --------
        cw, cwp, cwpp = poses.point(OUTPUT, out_com)
        Ii = m_out * np.einsum("pi,pi->p", cwp, cwp) + I_out_com * 1.0
        dIi = 2 * m_out * np.einsum("pi,pi->p", cwp, cwpp)
        Gi = -m_out * GRAV * (cwp @ g_hat)
        for k in range(K):
            ck, ckp, ckpp = poses.point(2 + k, spec.coupler_com[:, k])
            mk, Ik = spec.coupler_mass[:, k], spec.coupler_Icom[:, k]
            phip = poses.phip[:, 2 + k]
            phipp = poses.phipp[:, 2 + k]
            Ii = Ii + mk * np.einsum("pi,pi->p", ckp, ckp) + Ik * phip ** 2
            dIi = dIi + 2 * mk * np.einsum("pi,pi->p", ckp, ckpp) \
                + 2 * Ik * phip * phipp
            Gi = Gi - mk * GRAV * (ckp @ g_hat)
        I_eff[:, i], dI[:, i], Gt[:, i] = Ii, dIi, Gi

        # --- actuator coordinates and transmission ratios --------------------
        act_endpoints = []
        for ai, ad in enumerate(spec.actuators):
            if ad.kind == "rotary":
                q[:, ai, i], r[:, ai, i] = -th, 1.0
                continue
            wa, wap, _ = poses.point(ad.body_a, ad.pt_a)
            wb, wbp, _ = poses.point(ad.body_b, ad.pt_b)
            d = wa - wb
            L = np.linalg.norm(d, axis=1)
            q[:, ai, i] = L
            r[:, ai, i] = -np.einsum("pi,pi->p", d / np.maximum(L, 1e-12)[:, None],
                                     wap - wbp)
            act_endpoints.append((wa, wb))

        # measured conditioning for coupler-free topologies (E-5): force
        # transmission conditioning = characteristic length / min |r| among
        # bilateral force-path actuators; rotary-only mechanisms are uniform.
        if K == 0:
            r_lin = [np.abs(r[:, ai, i]) for ai, ad in enumerate(spec.actuators)
                     if ad.kind in ("linear", "tendon")]
            if r_lin:
                r_min_i = np.min(np.stack(r_lin, axis=1), axis=1)
                cond[:, i] = ctx.Ld / np.maximum(r_min_i, 1e-12)
                valid[:, i] &= cond[:, i] < COND_MAX

        # --- geometry capture at sampled angles (universal gate V8/V9) -------
        if i in geom_pos:
            si = geom_pos[i]
            s_i = 0
            psi = np.pi - th
            geom_segs[:, si, s_i, 0] = 0.0
            geom_segs[:, si, s_i, 1] = (ctx.Lp, 0.0)
            s_i += 1
            geom_segs[:, si, s_i, 0] = 0.0
            geom_segs[:, si, s_i, 1] = (ctx.Ld * np.cos(psi),
                                        ctx.Ld * np.sin(psi))
            s_i += 1
            for wa, wb in act_endpoints:
                geom_segs[:, si, s_i, 0] = wa
                geom_segs[:, si, s_i, 1] = wb
                s_i += 1
            for (k, locA, locB) in coupler_seg_locs:
                pa, _, _ = poses.point(2 + k, locA)
                pb, _, _ = poses.point(2 + k, locB)
                geom_segs[:, si, s_i, 0] = pa
                geom_segs[:, si, s_i, 1] = pb
                s_i += 1

    pay_dy = ctx.L_mass * np.sin(theta_grid - tn)
    pay_d2 = np.full(n, ctx.L_mass ** 2)
    geom_thetas = theta_grid[geom_idx]
    tables = []
    for p in range(P):
        t = TransmissionTable(
            theta=theta_grid, q=q[p], r=r[p], I_eff=I_eff[p], dI_dtheta=dI[p],
            G=Gt[p], S=_spring_terms(genomes[p], ctx, theta_grid),
            valid=valid[p], cond=cond[p],
            payload_dydth=pay_dy, payload_dpdth2=pay_d2,
            actuator_kinds=[a.kind for a in spec.actuators],
            unilateral=np.array([a.kind == "tendon" for a in spec.actuators]))
        t.geometry = {"theta": geom_thetas, "segs": geom_segs[p]}
        tables.append(t)
    return tables


def _coupler_segment_locs(spec: GraphSpec):
    """Per coupler: local anchor pairs forming its drawable/checkable links.

    A coupler's attachment points are its joint anchors plus any actuator
    mount points on it; consecutive pairs form the physical link segments.
    """
    locs_by_coupler = {k: [] for k in range(spec.n_couplers)}
    for jd in spec.joints:
        if jd.body_a >= 2:
            locs_by_coupler[jd.body_a - 2].append(jd.anchor_a)
        if jd.body_b >= 2:
            locs_by_coupler[jd.body_b - 2].append(jd.anchor_b)
    for ad in spec.actuators:
        if ad.kind == "rotary":
            continue
        if ad.body_a >= 2:
            locs_by_coupler[ad.body_a - 2].append(ad.pt_a)
        if ad.body_b >= 2:
            locs_by_coupler[ad.body_b - 2].append(ad.pt_b)
    out = []
    for k, locs in locs_by_coupler.items():
        for a in range(len(locs) - 1):
            out.append((k, locs[a], locs[a + 1]))
    return out
