"""Scene export: one geometric source of truth per run (SPEC-2 §7, V-1/V-2).

Each run emits:
  scene.glb   - named meshes: parametric anatomical bones generated from the
                MEASURED segment lengths (V-3), a translucent soft-tissue
                envelope, and unit mechanism primitives the viewer stretches
                per frame (actuator barrel/rod, coupler link, pins)
  scene.json  - per-frame poses: flexion angle, actuator endpoints + effort,
                coupler link segments; anatomical landmark labels; the ROM;
                honesty annotations (V-4: representative-geometry caveat and
                the fixed-revolute-axis approximation note)
  scene_preview.png - offscreen matplotlib still (V-7 fallback tier)

Consumers (the WebGL viewer, the preview renderer) read these files and never
recompute a pose. The mechanism z-plane is offset from the bone axis so the
linkage reads as bracket-mounted hardware, which is how it would be built.
"""

import json
import os

import numpy as np

MAX_SCENE_FRAMES = 96
MECH_Z = 0.045            # m, mechanism plane offset from the bone axis

CAVEAT = ("Anatomy shown is representative parametric geometry scaled to the "
          "measured segment lengths - NOT imaging of the participant.")
AXIS_NOTE = ("The joint is modelled as a FIXED revolute axis. The anatomical "
             "axis (especially at the knee) migrates through flexion; this "
             "render approximates it.")

# per-joint anatomy: bone names + labelled landmarks as (name, axial fraction
# along the bone, lateral offset fraction) on proximal (P) / distal (D) bones
ANATOMY = {
    "elbow": {
        "prox": "humerus", "dist": "radius + ulna",
        "landmarks": [("medial epicondyle", "P", 0.97, 0.5),
                      ("lateral epicondyle", "P", 0.97, -0.5),
                      ("olecranon", "D", 0.04, 0.6),
                      ("radial head", "D", 0.06, -0.5),
                      ("styloid (wrist)", "D", 0.97, 0.0)],
        "twin_distal": True,
    },
    "knee": {
        "prox": "femur", "dist": "tibia + fibula",
        "landmarks": [("medial femoral condyle", "P", 0.97, 0.5),
                      ("lateral femoral condyle", "P", 0.97, -0.5),
                      ("tibial plateau", "D", 0.03, 0.0),
                      ("tibial tuberosity", "D", 0.12, 0.0),
                      ("medial malleolus", "D", 0.97, 0.4)],
        "twin_distal": True,
    },
    "shoulder": {
        "prox": "scapula / glenoid", "dist": "humerus (whole arm)",
        "landmarks": [("acromion", "P", 0.9, -0.4),
                      ("glenoid", "P", 1.0, 0.0),
                      ("humeral head", "D", 0.03, 0.0),
                      ("deltoid tuberosity", "D", 0.35, -0.4)],
        "twin_distal": False,
    },
    "hip": {
        "prox": "pelvis (ilium)", "dist": "femur (whole leg)",
        "landmarks": [("acetabulum", "P", 1.0, 0.0),
                      ("femoral head", "D", 0.02, 0.0),
                      ("greater trochanter", "D", 0.08, -0.6)],
        "twin_distal": False,
    },
    "ankle": {
        "prox": "tibia + fibula", "dist": "foot (talus/calcaneus)",
        "landmarks": [("medial malleolus", "P", 0.97, 0.5),
                      ("lateral malleolus", "P", 0.97, -0.5),
                      ("talar dome", "D", 0.05, 0.0),
                      ("calcaneus", "D", 0.25, 0.5)],
        "twin_distal": False,
    },
}

BONE_COLOR = [0.93, 0.90, 0.82, 1.0]
TISSUE_COLOR = [0.85, 0.62, 0.52, 0.22]
LINK_COLOR = [0.28, 0.52, 0.30, 1.0]
BARREL_COLOR = [0.23, 0.43, 0.65, 1.0]
ROD_COLOR = [0.62, 0.72, 0.83, 1.0]


def _tm():
    import trimesh
    return trimesh


def _mat(tm, rgba, name):
    m = tm.visual.material.PBRMaterial(
        baseColorFactor=rgba, metallicFactor=0.1, roughnessFactor=0.7,
        name=name)
    if rgba[3] < 1.0:
        m.alphaMode = "BLEND"
    return m


def _long_bone(tm, length, d_shaft, d_head, d_tail, bow=0.0):
    """Parametric long bone along +x: tapered shaft + epiphysis spheres.

    bow: anterior bow as a fraction of length (femur ~ 0.02)."""
    n = 9
    xs = np.linspace(0, length, n)
    parts = []
    for i in range(n - 1):
        f0, f1 = xs[i] / length, xs[i + 1] / length
        # shaft narrows mid-bone, widens at the metaphyses
        w = d_shaft * (1.0 + 0.55 * (abs(f0 - 0.5) * 2) ** 2)
        seg = tm.creation.cylinder(radius=w / 2, height=xs[i + 1] - xs[i],
                                   sections=20)
        seg.apply_transform(tm.transformations.rotation_matrix(
            np.pi / 2, [0, 1, 0]))
        y_off = bow * length * np.sin(np.pi * (f0 + f1) / 2)
        seg.apply_translation(((xs[i] + xs[i + 1]) / 2, y_off, 0))
        parts.append(seg)
    head = tm.creation.icosphere(subdivisions=2, radius=d_head / 2)
    head.apply_translation((0, 0, 0))
    tail = tm.creation.icosphere(subdivisions=2, radius=d_tail / 2)
    tail.apply_translation((length, 0, 0))
    return tm.util.concatenate(parts + [head, tail])


def _condyle_pair(tm, at_x, d, sep):
    a = tm.creation.icosphere(subdivisions=2, radius=d / 2)
    a.apply_translation((at_x, 0, sep / 2))
    b = tm.creation.icosphere(subdivisions=2, radius=d / 2)
    b.apply_translation((at_x, 0, -sep / 2))
    return tm.util.concatenate([a, b])


def _bone_group(tm, joint, length, is_distal):
    """Anatomy-flavoured bone solid for one segment, along +x from the joint."""
    d = max(0.018, 0.052 * length)           # robusticity ~ 5% of length
    info = ANATOMY[joint]
    twin = info["twin_distal"] and is_distal
    bow = 0.02 if joint in ("knee", "hip") and not is_distal else 0.0
    if twin:
        b1 = _long_bone(tm, length, d * 0.75, d * 1.25, d * 0.9)
        b1.apply_translation((0, 0, d * 0.55))
        b2 = _long_bone(tm, length * 0.98, d * 0.55, d * 0.8, d * 0.7)
        b2.apply_translation((0, 0, -d * 0.55))
        bone = tm.util.concatenate([b1, b2])
    else:
        bone = _long_bone(tm, length, d, d * 1.5, d * 1.2, bow=bow)
    cond = _condyle_pair(tm, length if not is_distal else 0.0,
                         d * 1.15, d * 1.1)
    return tm.util.concatenate([bone, cond])


def _tissue(tm, length, girth):
    cap = tm.creation.capsule(radius=girth / 2, height=length * 0.92)
    cap.apply_transform(tm.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    cap.apply_translation((length / 2, 0, 0))
    return cap


def build_scene_files(out_dir, ctx, joint, side, final, frames_of_theta,
                      actuator_endpoints, link_segments=None,
                      util_of_frame=None):
    """Write scene.glb + scene.json + scene_preview.png.

    frames_of_theta: (F,) flexion rad per exported frame
    actuator_endpoints: (F, n_act, 2, 2) world xy endpoints per frame
    link_segments: optional (F, S, 2, 2) coupler link segments per frame
    util_of_frame: optional (F, n_act) |u|/u_max
    """
    tm = _tm()
    os.makedirs(out_dir, exist_ok=True)
    info = ANATOMY[joint]

    scene = tm.Scene()
    prox = _bone_group(tm, joint, ctx.Lp, is_distal=False)
    prox.visual.material = _mat(tm, BONE_COLOR, "bone")
    dist = _bone_group(tm, joint, ctx.Ld, is_distal=True)
    dist.visual.material = _mat(tm, BONE_COLOR, "bone")
    tis_p = _tissue(tm, ctx.Lp, max(0.07, 0.30 * ctx.Lp))
    tis_p.visual.material = _mat(tm, TISSUE_COLOR, "tissue")
    tis_d = _tissue(tm, ctx.Ld, max(0.06, 0.26 * ctx.Ld))
    tis_d.visual.material = _mat(tm, TISSUE_COLOR, "tissue")
    scene.add_geometry(prox, node_name="bone_proximal",
                       geom_name="bone_proximal")
    scene.add_geometry(dist, node_name="bone_distal", geom_name="bone_distal")
    scene.add_geometry(tis_p, node_name="tissue_proximal",
                       geom_name="tissue_proximal")
    scene.add_geometry(tis_d, node_name="tissue_distal",
                       geom_name="tissue_distal")

    # unit mechanism primitives, stretched per frame by the viewer
    barrel = tm.creation.cylinder(radius=0.011, height=1.0, sections=20)
    barrel.visual.material = _mat(tm, BARREL_COLOR, "barrel")
    rod = tm.creation.cylinder(radius=0.004, height=1.0, sections=14)
    rod.visual.material = _mat(tm, ROD_COLOR, "rod")
    link = tm.creation.box(extents=(1.0, 0.016, 0.006))
    link.visual.material = _mat(tm, LINK_COLOR, "link")
    pin = tm.creation.icosphere(subdivisions=1, radius=0.007)
    pin.visual.material = _mat(tm, [0.15, 0.15, 0.15, 1.0], "pin")
    for name, geom in (("unit_barrel", barrel), ("unit_rod", rod),
                       ("unit_link", link), ("unit_pin", pin)):
        scene.add_geometry(geom, node_name=name, geom_name=name)

    glb_path = os.path.join(out_dir, "scene.glb")
    scene.export(glb_path)

    # ---- frames ------------------------------------------------------------
    F = len(frames_of_theta)
    frames = []
    for f in range(F):
        th = float(frames_of_theta[f])
        acts = []
        for ai in range(actuator_endpoints.shape[1]):
            A = actuator_endpoints[f, ai, 0]
            B = actuator_endpoints[f, ai, 1]
            acts.append({
                "a": [float(A[0]), float(A[1]), MECH_Z],
                "b": [float(B[0]), float(B[1]), MECH_Z],
                "util": (float(util_of_frame[f, ai])
                         if util_of_frame is not None else 0.0)})
        fr = {"theta_deg": round(np.degrees(th), 2), "actuators": acts}
        if link_segments is not None:
            fr["links"] = [[[float(s[0][0]), float(s[0][1]), MECH_Z],
                            [float(s[1][0]), float(s[1][1]), MECH_Z]]
                           for s in link_segments[f]]
        frames.append(fr)

    lm = []
    for (name, bone, fx, fy) in info["landmarks"]:
        L = ctx.Lp if bone == "P" else ctx.Ld
        d = max(0.018, 0.052 * L)
        lm.append({"name": name, "bone": bone,
                   "pos": [fx * L, fy * d * 1.6, 0.0]})

    doc = {
        "schema_version": 1,
        "meta": {"joint": joint, "side": side, "units": "m",
                 "prox_bone": info["prox"], "dist_bone": info["dist"],
                 "Lp": ctx.Lp, "Ld": ctx.Ld,
                 "caveat": CAVEAT, "axis_note": AXIS_NOTE,
                 "not_modelled": [
                     "ground reaction forces (ankle stance)",
                     "inter-joint reaction forces",
                     "out-of-sagittal-plane motion",
                     "migrating joint axis (fixed revolute assumed)"]},
        "rom_deg": [round(float(np.degrees(frames_of_theta.min())), 1),
                    round(float(np.degrees(frames_of_theta.max())), 1)],
        "rmse_deg": round(float(final.get("rmse_deg", 0.0)), 2),
        "landmarks": lm,
        "frames": frames,
    }
    json_path = os.path.join(out_dir, "scene.json")
    with open(json_path, "w") as fjs:
        json.dump(doc, fjs)

    _preview_png(tm, scene, frames_of_theta, actuator_endpoints,
                 os.path.join(out_dir, "scene_preview.png"), ctx, joint)
    return {"scene.glb": glb_path, "scene.json": json_path}


def _preview_png(tm, scene, thetas, act_ep, path, ctx, joint):
    """Offscreen fallback still (V-7 tier 3: matplotlib, always available)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111, projection="3d")
        th = float(np.median(thetas))
        for name in ("bone_proximal", "bone_distal"):
            mesh = scene.geometry[name]
            v, fc = mesh.vertices.copy(), mesh.faces
            if name == "bone_distal":
                c, s = np.cos(np.pi - th), np.sin(np.pi - th)
                v = v @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
            step = max(1, len(fc) // 4000)
            ax.plot_trisurf(v[:, 0], v[:, 1], fc[::step], v[:, 2],
                            color="#e8e2d0", edgecolor="none")
        f_mid = int(np.argmin(np.abs(thetas - th)))
        for ai in range(act_ep.shape[1]):
            A, B = act_ep[f_mid, ai]
            ax.plot([A[0], B[0]], [A[1], B[1]], [MECH_Z, MECH_Z],
                    color="#3a6ea5", lw=5)
        ax.set_title(f"{joint} - anatomical scene (mid flexion "
                     f"{np.degrees(th):.0f} deg)\n{CAVEAT}", fontsize=8)
        ax.set_box_aspect((1.6, 1.2, 0.6))
        ax.axis("off")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def scene_frames_from_final(final, n_frames=MAX_SCENE_FRAMES):
    """Decimate the final simulation onto the scene frame set."""
    t = final["t"]
    step = max(1, int(np.ceil(len(t) / n_frames)))
    idx = np.arange(0, len(t), step)
    return idx, np.deg2rad(np.asarray(final["theta_sim_deg"])[idx])
