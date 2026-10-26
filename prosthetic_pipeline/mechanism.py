"""Geometry of the candidate actuator mechanism.

Model: a 2D revolute joint (the anatomical joint) driven by a linear actuator
that spans it -- the architecture used in powered prosthetic knees/elbows.

Frames and conventions
----------------------
- Joint center J at the origin, working in the plane of motion (sagittal).
- Proximal segment (the socket side, e.g. upper arm / thigh) is fixed along +x:
  the bone runs from J to S = (Lp, 0).
- theta = flexion angle in radians. theta = 0 means fully extended (distal
  segment collinear with proximal, pointing along -x). Increasing theta curls
  the distal segment toward +y. The distal bone direction is
      u_d(theta) = [cos(psi), sin(psi)],  psi = pi - theta.
- Actuator anchor A sits on the proximal segment at polar (a, alpha) in the
  fixed frame. Anchor B sits on the distal segment at polar (b, beta) measured
  from the distal bone axis, so in the fixed frame B = b*[cos(psi+beta), sin(psi+beta)].
- Actuator length L(theta) = |A - B(theta)|. Its derivative dL/dtheta equals
  (up to sign) the moment arm r(theta) = |A x B| / L, which converts actuator
  force to joint torque: tau = -T * dL/dtheta for axial tension T.

All functions broadcast: parameters shaped (P, 1) against theta shaped (1, T)
evaluate a whole population over a whole trajectory at once.
"""

from dataclasses import dataclass, asdict

import numpy as np

# Physical bounds used when decoding genomes (fractions of segment length, etc.)
ANCHOR_A_FRAC = (0.06, 0.70)      # a as fraction of proximal segment length
ANCHOR_B_FRAC = (0.05, 0.60)      # b as fraction of distal segment length
ANCHOR_ANGLE_MAX = np.deg2rad(35)  # |alpha|, |beta| bound (keeps mounting brackets sane)
F_MAX_RANGE = (60.0, 1500.0)       # N, log-scale decode
V_MAX_RANGE = (0.02, 0.35)         # m/s, log-scale decode
STROKE_RATIO_LIMIT = 1.8           # extended/retracted length limit (real ballscrew actuators)
MIN_RETRACTED = 0.04               # m, smallest physically plausible retracted length
# Real packaging: retracted length >= stroke + dead length (nut + end blocks, with
# the motor piggybacked parallel to the screw; an inline motor would need ~70 mm).
DEAD_LENGTH_PARALLEL = 0.028       # m
MIN_MOMENT_ARM = 0.008             # m, below this the mechanism can't transmit torque
STROKE_MARGIN = 0.003              # m of stroke margin beyond the required range

N_GENES = 6  # a, alpha, b, beta, F_max, v_max (all in [0, 1])


@dataclass
class Mechanism:
    """A decoded, concrete mechanism design (scalars, SI units)."""
    a: float          # m, proximal anchor distance from joint
    alpha: float      # rad, proximal anchor angle from proximal bone axis
    b: float          # m, distal anchor distance from joint
    beta: float       # rad, distal anchor angle from distal bone axis
    F_max: float      # N, actuator peak force
    v_max: float      # m/s, actuator peak linear speed
    L_ret: float      # m, retracted length chosen for the observed ROM
    stroke: float     # m

    @property
    def L_ext(self) -> float:
        return self.L_ret + self.stroke

    @property
    def stroke_ratio(self) -> float:
        return self.L_ext / max(self.L_ret, 1e-9)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["L_ext"] = self.L_ext
        d["stroke_ratio"] = self.stroke_ratio
        return d


def decode(genes: np.ndarray, Lp: float, Ld: float) -> dict:
    """Map genomes in [0,1]^6 to physical parameter arrays.

    genes: (..., 6). Returns dict of arrays shaped like genes[..., 0].
    """
    g = np.clip(genes, 0.0, 1.0)
    lo_a, hi_a = ANCHOR_A_FRAC
    lo_b, hi_b = ANCHOR_B_FRAC
    flo, fhi = F_MAX_RANGE
    vlo, vhi = V_MAX_RANGE
    return {
        "a": (lo_a + (hi_a - lo_a) * g[..., 0]) * Lp,
        "alpha": (2.0 * g[..., 1] - 1.0) * ANCHOR_ANGLE_MAX,
        "b": (lo_b + (hi_b - lo_b) * g[..., 2]) * Ld,
        "beta": (2.0 * g[..., 3] - 1.0) * ANCHOR_ANGLE_MAX,
        "F_max": flo * (fhi / flo) ** g[..., 4],
        "v_max": vlo * (vhi / vlo) ** g[..., 5],
    }


def anchor_A(a, alpha):
    """Fixed proximal anchor, components (Ax, Ay)."""
    return a * np.cos(alpha), a * np.sin(alpha)


def anchor_B(b, beta, theta):
    """Distal anchor at flexion theta, components (Bx, By)."""
    phi = (np.pi - theta) + beta
    return b * np.cos(phi), b * np.sin(phi)


def length_and_derivative(a, alpha, b, beta, theta):
    """Return L(theta), dL/dtheta, and moment arm r(theta) = |dL/dtheta|.

    Derivation: L^2 = |A - B|^2 with B'(theta) = b*[sin(phi), -cos(phi)],
    and B . B' = 0, giving dL/dtheta = -cross(A, B)/L.
    """
    Ax, Ay = anchor_A(a, alpha)
    Bx, By = anchor_B(b, beta, theta)
    L = np.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2)
    L = np.maximum(L, 1e-9)
    cross = Ax * By - Ay * Bx
    dL = -cross / L
    return L, dL, np.abs(dL)


def fit_stroke(a, alpha, b, beta, theta_lo, theta_hi, n=80):
    """Choose retracted length / stroke to cover the observed ROM.

    Works on parameter arrays shaped (P,). Returns (L_ret, stroke) arrays.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    grid = np.linspace(theta_lo, theta_hi, n)[None, :]
    L, _, _ = length_and_derivative(
        a[:, None], np.atleast_1d(alpha)[:, None],
        np.atleast_1d(b)[:, None], np.atleast_1d(beta)[:, None], grid,
    )
    L_lo = L.min(axis=1) - STROKE_MARGIN
    L_hi = L.max(axis=1) + STROKE_MARGIN
    return L_lo, L_hi - L_lo


def make_mechanism(genes: np.ndarray, Lp: float, Ld: float,
                   theta_lo: float, theta_hi: float) -> Mechanism:
    """Decode a single genome into a concrete Mechanism."""
    p = decode(np.asarray(genes, dtype=float), Lp, Ld)
    L_ret, stroke = fit_stroke(p["a"], p["alpha"], p["b"], p["beta"], theta_lo, theta_hi)
    return Mechanism(
        a=float(p["a"]), alpha=float(p["alpha"]),
        b=float(p["b"]), beta=float(p["beta"]),
        F_max=float(p["F_max"]), v_max=float(p["v_max"]),
        L_ret=float(L_ret[0]), stroke=float(stroke[0]),
    )
