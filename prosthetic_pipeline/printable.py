"""Stage 7: 3D-printable model of the evolved actuator joint.

Generates STL parts for a hand-assemblable demonstrator of the evolved
mechanism at true scale (auto-scaled down if a segment is too large to print):

- link_proximal / link_distal: flat linkage plates with the joint bore and the
  anchor bracket at the exact evolved anchor position (a@alpha, b@beta)
- actuator_barrel / actuator_rod: telescoping mock of the specced actuator
  (true retracted length, stroke, barrel/rod diameters, motor pod)
- pin x3: shouldered pins for the joint and both anchor clevises
- print_plate.stl: everything laid out for one print job
- assembled_preview.stl + printable_preview.png: the assembled mechanism

Requires trimesh + manifold3d (real CSG -> watertight, slicer-friendly parts).
All dimensions in millimeters. This is a form/kinematics demonstrator, not a
load-bearing prosthetic part.
"""

import os
from typing import Dict

import numpy as np

from .mechanism import anchor_A, anchor_B

PLATE_T = 6.0        # linkage plate thickness
LUG_T = 6.0
PIN_D = 4.0
HOLE_D = 4.4         # printed clearance hole for the pins
BOSS_D = 12.0
BAR_W = 18.0
MAX_PART_MM = 180.0  # auto-scale threshold
ROD_EXTRA = 40.0     # rod overlap length inside the barrel


def _trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise RuntimeError(
            "3D model generation needs trimesh + manifold3d "
            "(pip install trimesh manifold3d)") from e


def _union(tm, meshes):
    return tm.boolean.union(meshes)


def _diff(tm, a, b):
    return tm.boolean.difference([a, b])


def _cyl(tm, d, h, at=(0, 0, 0), axis="z"):
    m = tm.creation.cylinder(radius=d / 2.0, height=h, sections=48)
    if axis == "x":
        m.apply_transform(tm.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(tm.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(at)
    return m


def _box(tm, ext, at=(0, 0, 0)):
    m = tm.creation.box(extents=ext)
    m.apply_translation(at)
    return m


def _link_plate(tm, reach_xy, tip_len, label_len):
    """Flat linkage plate in XY (thickness z): joint bore at origin, anchor
    bracket at reach_xy = (x, y), main bar continuing to label_len along +x."""
    ax, ay = reach_xy
    L = max(label_len, ax + 15.0, 40.0)
    parts = [
        _box(tm, (L + 24, BAR_W, PLATE_T), ((L - 12) / 2 + 6, 0, PLATE_T / 2)),
        _cyl(tm, BOSS_D + 6, PLATE_T, (0, 0, PLATE_T / 2)),
        _cyl(tm, BOSS_D, PLATE_T, (ax, ay, PLATE_T / 2)),
    ]
    if abs(ay) > 2.0:  # bracket arm from the bar out to the anchor point
        parts.append(_box(tm, (BAR_W * 0.8, abs(ay) + BAR_W * 0.7, PLATE_T),
                          (ax, ay / 2, PLATE_T / 2)))
    if tip_len > 0:  # pointer tip beyond the bar end
        parts.append(_box(tm, (tip_len, BAR_W * 0.55, PLATE_T),
                          (L + 6 + tip_len / 2, 0, PLATE_T / 2)))
    solid = _union(tm, parts)
    solid = _diff(tm, solid, _cyl(tm, HOLE_D, PLATE_T * 3, (0, 0, PLATE_T / 2)))
    solid = _diff(tm, solid, _cyl(tm, HOLE_D, PLATE_T * 3, (ax, ay, PLATE_T / 2)))
    return solid


def _lug(tm, at, height=18.0):
    """Vertical clevis tab with a horizontal pin hole; base sits on z=0."""
    tab = _box(tm, (14.0, LUG_T, height), (at[0], at[1], height / 2))
    hole = _cyl(tm, HOLE_D, LUG_T * 3, (at[0], at[1], height - 6.0), axis="y")
    return tab, hole


def _barrel(tm, od, bore_d, body_len, motor_d, motor_len):
    lug_h = 16.0
    body = _cyl(tm, od, body_len, (0, 0, lug_h + body_len / 2))
    tab, hole = _lug(tm, (0, 0), lug_h + 4)
    pod = _cyl(tm, motor_d, motor_len,
               (od / 2 + motor_d / 2 - 2.0, 0, lug_h + motor_len / 2))
    solid = _union(tm, [body, tab, pod])
    bore = _cyl(tm, bore_d, body_len, (0, 0, lug_h + body_len / 2 + 4.0))
    solid = _diff(tm, solid, bore)
    return _diff(tm, solid, hole)


def _rod(tm, rod_d, length):
    lug_h = 16.0
    shaft = _cyl(tm, rod_d, length, (0, 0, lug_h + length / 2 - 4))
    tab, hole = _lug(tm, (0, 0), lug_h + 4)
    return _diff(tm, _union(tm, [shaft, tab]), hole)


def _pin(tm):
    return _union(tm, [
        _cyl(tm, 8.0, 3.0, (0, 0, 1.5)),
        _cyl(tm, PIN_D, PLATE_T * 2 + LUG_T + 4, (0, 0, 3.0 + (PLATE_T * 2 + LUG_T + 4) / 2)),
    ])


def generate_printables(mech, spec, ctx, out_dir: str, th_mid: float = None) -> Dict:
    tm = _trimesh()
    os.makedirs(out_dir, exist_ok=True)

    rec = spec["variants"][spec["recommended"]]
    screw = rec["drive"]["screw"]
    od = screw["d_nom_mm"] + 14.0
    rod_d = spec["structure"]["rod_diameter_mm"]
    body_len = spec["packaging"]["body_length_mm"]
    from .actuator_design import _motor_diameter_mm
    motor_d = _motor_diameter_mm(rec["drive"]["motor_class"])
    stroke = mech.stroke * 1000
    L_ret = mech.L_ret * 1000
    # mock so the assembled retracted pin-to-pin matches the evolved L_ret
    mock_body = max(min(body_len, L_ret - 28.0), 30.0)

    A = np.array(anchor_A(mech.a, mech.alpha)) * 1000
    Bloc = np.array([mech.b * np.cos(mech.beta), mech.b * np.sin(mech.beta)]) * 1000

    # auto-scale so the largest part stays printable
    dims = [abs(A[0]) + 40, abs(Bloc[0]) + 40, mock_body + 30,
            min(ctx.Ld * 1000 * 0.6, 150) + 40]
    scale = min(1.0, MAX_PART_MM / max(dims))

    def s(v):
        return v * scale

    parts = {
        "link_proximal": _link_plate(
            tm, (s(A[0]), s(A[1])), 0, s(min(ctx.Lp * 1000 * 0.5, 130))),
        "link_distal": _link_plate(
            tm, (s(Bloc[0]), s(Bloc[1])), s(25), s(min(ctx.Ld * 1000 * 0.6, 150))),
        "actuator_barrel": _barrel(tm, s(od), s(rod_d) + 0.6, s(mock_body),
                                   s(motor_d), s(mock_body * 0.55)),
        "actuator_rod": _rod(tm, s(rod_d), s(stroke + ROD_EXTRA)),
        "pin": _pin(tm),
    }

    paths, notes = {}, []
    if scale < 1.0:
        notes.append(f"parts scaled to {scale:.2f}x to fit a printer (mock demonstrator)")
    plate_items, x_cursor = [], 0.0
    for name, mesh in parts.items():
        if not mesh.is_watertight:
            mesh.fill_holes()
            if not mesh.is_watertight:
                notes.append(f"{name}: mesh not fully watertight; run slicer repair")
        fn = os.path.join(out_dir, f"part_{name}.stl")
        mesh.export(fn)
        paths[f"part_{name}.stl"] = fn
        # lay out on the shared print plate
        count = 3 if name == "pin" else 1
        for k in range(count):
            inst = mesh.copy()
            b = inst.bounds
            inst.apply_translation((x_cursor - b[0][0], -b[0][1], -b[0][2]))
            plate_items.append(inst)
            x_cursor += (b[1][0] - b[0][0]) + 10.0

    plate = tm.util.concatenate(plate_items)
    plate_fn = os.path.join(out_dir, "print_plate.stl")
    plate.export(plate_fn)
    paths["print_plate.stl"] = plate_fn

    # assembled preview at mid flexion (not for printing)
    th = np.deg2rad(45.0) if th_mid is None else float(th_mid)
    B_world = np.array(anchor_B(mech.b, mech.beta, th)) * 1000
    prox = parts["link_proximal"].copy()
    dist = parts["link_distal"].copy()
    dist.apply_transform(tm.transformations.rotation_matrix(np.pi - th, [0, 0, 1]))
    dist.apply_translation((0, 0, PLATE_T + 1))
    act_vec = np.array([s(B_world[0]) - s(A[0]), s(B_world[1]) - s(A[1]), 0.0])
    act_len = np.linalg.norm(act_vec)
    barrel = parts["actuator_barrel"].copy()
    rod = parts["actuator_rod"].copy()
    rod.apply_translation((0, 0, act_len - s(stroke + ROD_EXTRA) - 8))
    act = tm.util.concatenate([barrel, rod])
    rot = tm.geometry.align_vectors([0, 0, 1], act_vec / act_len)
    act.apply_transform(rot)
    act.apply_translation((s(A[0]), s(A[1]), PLATE_T * 2 + 8))
    assembled = tm.util.concatenate([prox, dist, act])
    asm_fn = os.path.join(out_dir, "assembled_preview.stl")
    assembled.export(asm_fn)
    paths["assembled_preview.stl"] = asm_fn

    png = os.path.join(out_dir, "printable_preview.png")
    _render_preview(plate, assembled, png)
    paths["printable_preview.png"] = png

    return {"paths": paths, "scale": scale, "notes": notes,
            "part_count": len(parts) + 2}


def _render_preview(plate, assembled, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5))
    for i, (mesh, title, elev, azim) in enumerate([
            (plate, "print plate (all parts)", 55, -60),
            (assembled, "assembled demonstrator (mid flexion)", 28, -55)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        v, f = mesh.vertices, mesh.faces
        step = max(1, len(f) // 6000)
        ax.plot_trisurf(v[:, 0], v[:, 1], f[::step], v[:, 2],
                        color="#7fa8cc", edgecolor="none", shade=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_box_aspect((np.ptp(v[:, 0]) + 1, np.ptp(v[:, 1]) + 1, np.ptp(v[:, 2]) + 1))
        ax.axis("off")
    fig.suptitle("3D-printable actuator joint demonstrator (mm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# declarative primitive generator (SPEC §7.3, R-20): any topology composed of
# registered primitives gets STLs without bespoke code
# ---------------------------------------------------------------------------

def from_parts(parts, out_dir: str) -> Dict:
    """Turn a Topology.solid_model() primitive list into printable STLs."""
    from .topologies import Barrel, Pin, Plate, Rod
    tm = _trimesh()
    os.makedirs(out_dir, exist_ok=True)

    def plate_mesh(p: Plate):
        segs = []
        pts = [np.asarray(q, dtype=float) for q in p.outline]
        if len(pts) == 1:
            pts.append(pts[0] + np.array([30.0, 0.0]))
        for a, b in zip(pts[:-1], pts[1:]):
            mid = (a + b) / 2
            L = max(float(np.linalg.norm(b - a)), 12.0)
            ang = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
            bar = _box(tm, (L + BAR_W, BAR_W, PLATE_T), (0, 0, PLATE_T / 2))
            bar.apply_transform(tm.transformations.rotation_matrix(ang, [0, 0, 1]))
            bar.apply_translation((mid[0], mid[1], 0))
            segs.append(bar)
        for bx, by in p.bores:
            segs.append(_cyl(tm, BOSS_D, PLATE_T, (bx, by, PLATE_T / 2)))
        solid = _union(tm, segs)
        for bx, by in p.bores:
            solid = _diff(tm, solid, _cyl(tm, HOLE_D, PLATE_T * 3, (bx, by, PLATE_T / 2)))
        return solid

    meshes = {}
    counts = {}
    for p in parts:
        kind = type(p).__name__.lower()
        counts[kind] = counts.get(kind, 0) + 1
        name = f"{kind}_{counts[kind]}"
        if isinstance(p, Plate):
            meshes[name] = plate_mesh(p)
        elif isinstance(p, Pin):
            meshes[name] = _pin(tm)
        elif isinstance(p, Barrel):
            body = _cyl(tm, p.od, p.length, (0, 0, p.length / 2))
            meshes[name] = _diff(tm, body, _cyl(tm, p.bore, p.length,
                                                (0, 0, p.length / 2 + 3)))
        elif isinstance(p, Rod):
            meshes[name] = _cyl(tm, p.d, p.length, (0, 0, p.length / 2))

    paths, notes = {}, []
    plate_items, x_cursor = [], 0.0
    for name, mesh in meshes.items():
        if not mesh.is_watertight:
            mesh.fill_holes()
            if not mesh.is_watertight:
                notes.append(f"{name}: run slicer repair")
        fn = os.path.join(out_dir, f"part_{name}.stl")
        mesh.export(fn)
        paths[f"part_{name}.stl"] = fn
        inst = mesh.copy()
        b = inst.bounds
        inst.apply_translation((x_cursor - b[0][0], -b[0][1], -b[0][2]))
        plate_items.append(inst)
        x_cursor += (b[1][0] - b[0][0]) + 10.0

    plate = tm.util.concatenate(plate_items)
    plate_fn = os.path.join(out_dir, "print_plate.stl")
    plate.export(plate_fn)
    paths["print_plate.stl"] = plate_fn
    png = os.path.join(out_dir, "printable_preview.png")
    _render_preview(plate, plate, png)
    paths["printable_preview.png"] = png
    return {"paths": paths, "scale": 1.0, "notes": notes,
            "part_count": len(meshes) + 1}
