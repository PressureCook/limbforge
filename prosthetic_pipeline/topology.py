"""Open-topology core: mechanism graph, transmission table, protocol, registry.

Implements SPEC-open-topology-evolution §5 (architecture) and §7.1/7.5
(capability interface and registration). A design is a planar mechanism graph
of rigid bodies + joints + actuators + springs, constrained to mobility 1 so
that all kinematics reduce to functions of the anatomical flexion angle theta
and dynamics collapse to the scalar reduced equation of motion (§5.4).

The TransmissionTable is the ONLY interface between a topology and the
simulator (R-6): theta-indexed arrays of actuator coordinates q_i, transmission
ratios r_i = -dq_i/dtheta, reduced inertia I_eff, dI_eff/dtheta, gravity G,
spring torque S, and a validity mask.
"""

import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

GROUND = "ground"
OUTPUT = "output"

# global innovation counter (R-11); seeded per run by the search engine
_INNOVATION = itertools.count(1)


def next_innovation() -> int:
    return next(_INNOVATION)


def reset_innovations(start: int = 1):
    global _INNOVATION
    _INNOVATION = itertools.count(start)


# ---------------------------------------------------------------------------
# genes (§5.1 / §6.1) - continuous values live in [0,1] and are decoded by the
# owning topology; structural identity is carried by innovation ids
# ---------------------------------------------------------------------------

@dataclass
class BodyGene:
    id: str
    kind: str = "coupler"            # ground | output | coupler
    genes: Dict[str, float] = field(default_factory=dict)  # e.g. length, density
    innovation: int = 0


@dataclass
class JointGene:
    id: str
    kind: str                         # revolute | prismatic
    body_a: str
    body_b: str
    genes: Dict[str, float] = field(default_factory=dict)  # anchor coords etc.
    innovation: int = 0


@dataclass
class ActuatorGene:
    id: str
    kind: str                         # linear | rotary | tendon
    mount: Dict = field(default_factory=dict)   # topology-interpreted
    genes: Dict[str, float] = field(default_factory=dict)  # capacity + geometry
    innovation: int = 0


@dataclass
class SpringGene:
    id: str
    kind: str                         # linear | torsional
    mount: Dict = field(default_factory=dict)
    genes: Dict[str, float] = field(default_factory=dict)  # k, q_free
    innovation: int = 0


@dataclass
class Genome:
    topology: str                     # registered topology kind
    bodies: List[BodyGene] = field(default_factory=list)
    joints: List[JointGene] = field(default_factory=list)
    actuators: List[ActuatorGene] = field(default_factory=list)
    springs: List[SpringGene] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    # --- serialization (R-1, R-8) -----------------------------------------
    def to_dict(self) -> dict:
        return {
            "topology": self.topology,
            "bodies": [asdict(b) for b in self.bodies],
            "joints": [asdict(j) for j in self.joints],
            "actuators": [asdict(a) for a in self.actuators],
            "springs": [asdict(s) for s in self.springs],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(
            topology=d["topology"],
            bodies=[BodyGene(**b) for b in d.get("bodies", [])],
            joints=[JointGene(**j) for j in d.get("joints", [])],
            actuators=[ActuatorGene(**a) for a in d.get("actuators", [])],
            springs=[SpringGene(**s) for s in d.get("springs", [])],
            meta=d.get("meta", {}),
        )

    def roundtrip_equal(self, other: "Genome") -> bool:
        return json.dumps(self.to_dict(), sort_keys=True) == \
            json.dumps(other.to_dict(), sort_keys=True)

    def innovations(self) -> List[int]:
        return sorted(g.innovation for g in
                      self.joints + self.actuators + self.springs if g.innovation)

    def continuous_genes(self) -> Dict[str, float]:
        """Flat {path: value} view of every continuous gene (for mutation/xover)."""
        out = {}
        for group, items in (("b", self.bodies), ("j", self.joints),
                             ("a", self.actuators), ("s", self.springs)):
            for it in items:
                for k, v in it.genes.items():
                    out[f"{group}.{it.id}.{k}"] = v
        return out

    def set_continuous(self, path: str, value: float):
        group, gid, key = path.split(".", 2)
        items = {"b": self.bodies, "j": self.joints,
                 "a": self.actuators, "s": self.springs}[group]
        for it in items:
            if it.id == gid:
                it.genes[key] = float(np.clip(value, 0.0, 1.0))
                return
        raise KeyError(path)


# ---------------------------------------------------------------------------
# transmission table (§5.3)
# ---------------------------------------------------------------------------

@dataclass
class TransmissionTable:
    theta: np.ndarray          # (N,)
    q: np.ndarray              # (n_act, N) actuator coordinates
    r: np.ndarray              # (n_act, N) transmission ratios, r_i = -dq_i/dtheta
    I_eff: np.ndarray          # (N,) reduced inertia about the output joint
    dI_dtheta: np.ndarray      # (N,)
    G: np.ndarray              # (N,) generalized gravity torque
    S: np.ndarray              # (N,) generalized passive/spring torque
    valid: np.ndarray          # (N,) bool
    cond: np.ndarray           # (N,) constraint-Jacobian condition (diagnostic)
    payload_dydth: np.ndarray = None   # (N,) d(payload height)/dtheta at carry point
    payload_dpdth2: np.ndarray = None  # (N,) |d(payload pos)/dtheta|^2 (inertia factor)
    actuator_kinds: List[str] = field(default_factory=list)
    unilateral: np.ndarray = None      # (n_act,) bool - tension-only (tendons)

    def __post_init__(self):
        n_act = self.q.shape[0]
        if self.unilateral is None:
            self.unilateral = np.zeros(n_act, dtype=bool)
        if self.payload_dydth is None:  # default: payload rides the output tip
            self.payload_dydth = np.zeros_like(self.theta)
        if self.payload_dpdth2 is None:
            self.payload_dpdth2 = np.zeros_like(self.theta)

    @property
    def n_act(self) -> int:
        return self.q.shape[0]

    def all_valid_in(self, th_lo: float, th_hi: float) -> bool:
        m = (self.theta >= th_lo - 1e-12) & (self.theta <= th_hi + 1e-12)
        return bool(np.all(self.valid[m]))


# ---------------------------------------------------------------------------
# capability protocol + registry (§7.1, §7.5)
# ---------------------------------------------------------------------------

@dataclass
class ActuatorRequirement:
    """Per-actuator sizing input for actuator_design (R-18)."""
    kind: str                  # linear | rotary | tendon
    peak_effort: float         # N or N m
    rms_effort: float
    peak_rate: float           # m/s or rad/s
    travel: float              # stroke (m) or sweep (rad)
    effort_trace: np.ndarray
    rate_trace: np.ndarray
    dt: float
    packaging_envelope: float  # available retracted length (m) or diameter
    q_min: float = 0.0         # coordinate range over the ROM
    q_max: float = 0.0
    label: str = "a0"


REQUIRED_METHODS = ("build_table", "mobility", "actuator_requirements",
                    "solid_model", "draw", "describe", "random_genome")

TOPOLOGIES: Dict[str, object] = {}


def register_topology(kind: str, smoke_ctx: Optional[Callable] = None):
    """Class decorator: verify the full capability interface (§1.3) and run a
    registration-time smoke test. A topology that cannot complete the
    downstream chain never enters the registry (R-17)."""

    def deco(cls):
        inst = cls()
        missing = [m for m in REQUIRED_METHODS if not callable(getattr(inst, m, None))]
        if missing:
            raise TypeError(f"topology '{kind}' missing methods: {missing}")
        inst.kind = kind
        _smoke_test(inst)
        TOPOLOGIES[kind] = inst
        return cls

    return deco


def _smoke_test(topo):
    """Random genome -> table -> requirements -> solid model, at import time."""
    from .simulate import JointContext
    ctx = JointContext.from_pose("knee", 0.42, 0.43, 80.0)
    rng = np.random.default_rng(0)
    genome = topo.random_genome(rng, ctx)
    grid = np.linspace(np.deg2rad(3), np.deg2rad(110), 64)
    table = topo.build_table(genome, ctx, grid)
    assert isinstance(table, TransmissionTable) and table.q.shape[1] == 64
    assert topo.mobility(genome) == 1, f"{topo.kind}: mobility != 1"
    fake = {"t": np.linspace(0, 1, 50), "dt": 0.02,
            "efforts": np.zeros((table.n_act, 50)),
            "rates": np.zeros((table.n_act, 50)),
            "theta": np.linspace(grid[0], grid[-1], 50)}
    reqs = topo.actuator_requirements(genome, ctx, table, fake)
    assert len(reqs) == table.n_act
    parts = topo.solid_model(genome, ctx, 1.0)
    assert len(parts) >= 1
    desc = topo.describe(genome)
    assert "n_actuators" in desc


def get_topology(kind: str):
    if kind not in TOPOLOGIES:
        raise KeyError(f"unknown topology '{kind}' (registered: {list(TOPOLOGIES)})")
    return TOPOLOGIES[kind]
