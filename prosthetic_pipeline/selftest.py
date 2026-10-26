"""Quick physics/GA sanity checks: `python -m prosthetic_pipeline.selftest`."""

import numpy as np

from . import mechanism as mech
from .demo_data import make_demo_series
from .evolve import evolve
from .simulate import JointContext, simulate_final


def test_geometry():
    rng = np.random.default_rng(1)
    a, alpha = 0.12, 0.2
    b, beta = 0.05, -0.3
    th = rng.uniform(0.05, 2.4, 200)
    L, dL, r = mech.length_and_derivative(a, alpha, b, beta, th)
    # moment arm equals |dL/dtheta| by virtual work
    assert np.allclose(r, np.abs(dL), atol=1e-12), "moment arm != |dL/dtheta|"
    # analytic derivative matches finite differences
    h = 1e-6
    L2, _, _ = mech.length_and_derivative(a, alpha, b, beta, th + h)
    fd = (L2 - L) / h
    assert np.max(np.abs(fd - dL)) < 1e-4, f"dL/dtheta mismatch: {np.max(np.abs(fd - dL))}"
    # stroke fit brackets the length over the ROM
    L_ret, stroke = mech.fit_stroke(a, alpha, b, beta, 0.1, 2.2)
    grid = np.linspace(0.1, 2.2, 300)
    Lg, _, _ = mech.length_and_derivative(a, alpha, b, beta, grid)
    assert L_ret[0] <= Lg.min() and L_ret[0] + stroke[0] >= Lg.max()
    print("geometry checks passed")


def test_evolution_and_sim():
    pose = make_demo_series(seed=0)
    ctx = JointContext.from_pose(pose.joint, pose.Lp, pose.Ld, body_mass=80.0)
    res = evolve(pose.t, pose.theta_deg, ctx, pop_size=60, generations=40, seed=0)
    assert res.history_best[-1] < res.history_best[0], "evolution did not improve"
    final = simulate_final(res.best_mechanism, pose.t, pose.theta_deg, ctx)
    print(f"evolution: cost {res.history_best[0]:.1f} -> {res.history_best[-1]:.1f}, "
          f"final RMSE {final['rmse_deg']:.2f} deg")
    assert final["rmse_deg"] < 8.0, f"tracking too poor: {final['rmse_deg']:.2f} deg"
    m = res.best_mechanism
    assert m.stroke_ratio < mech.STROKE_RATIO_LIMIT + 0.2, "implausible stroke ratio"
    assert m.L_ret >= m.stroke + mech.DEAD_LENGTH_PARALLEL - 0.006, \
        f"packaging constraint violated: L_ret {m.L_ret*1000:.0f} mm vs stroke {m.stroke*1000:.0f} mm"
    print("evolution + simulation checks passed")
    return m, final, ctx


def test_actuator_design(m, final, ctx):
    import tempfile

    from .actuator_design import design_actuator, write_spec_files

    spec = design_actuator(m, final, ctx)
    assert spec["recommended"] in spec["variants"]
    for name, v in spec["variants"].items():
        assert v["energy_per_recording_J"] > 0 and np.isfinite(v["energy_per_recording_J"])
        assert v["battery_hours"] > 0
        assert v["drive"]["tau_motor_peak_Nm"] > 0
        assert v["drive"]["n_motor_rpm"] <= 12000 + 1
    assert spec["structure"]["rod_buckling_SF"] > 2, "rod would buckle"
    assert spec["packaging"]["margin_mm"] > -6, "actuator body cannot fit retracted length"
    lead = spec["variants"].get("leadscrew")
    if lead:
        s = lead["drive"]["screw"]
        lam_ok = (s["lead_mm"] / (np.pi * s["d_pitch_mm"]))  # tan(lead angle)
        assert s["self_locking"] == (lam_ok <= 0.16 / np.cos(np.deg2rad(14.5)) + 1e-9)
    with tempfile.TemporaryDirectory() as d:
        paths = write_spec_files(spec, m, ctx, d)
        import os
        assert all(os.path.exists(p) for p in paths.values())
    rec = spec["variants"][spec["recommended"]]
    print(f"actuator design: {spec['recommended']} "
          f"({rec['drive']['screw']['d_nom_mm']:.0f} mm x {rec['drive']['screw']['lead_mm']:.0f} mm, "
          f"{rec['drive']['motor_class']}, {rec['mass_est_kg']*1000:.0f} g, "
          f"{rec['battery_hours']:.1f} h battery)")
    print("actuator design checks passed")


def test_printable(m, final, ctx):
    import tempfile

    from .actuator_design import design_actuator

    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("printable: trimesh not installed, skipping")
        return
    from .printable import generate_printables

    spec = design_actuator(m, final, ctx)
    with tempfile.TemporaryDirectory() as d:
        out = generate_printables(m, spec, ctx, d)
        import os

        import trimesh
        n_stl = 0
        for k, p in out["paths"].items():
            assert os.path.exists(p)
            if p.endswith(".stl"):
                mesh = trimesh.load(p)
                assert mesh.volume > 0, f"{k}: empty mesh"
                n_stl += 1
        assert n_stl >= 6, f"expected 6+ STL files, got {n_stl}"
    print(f"printable: {n_stl} STL parts generated (scale {out['scale']:.2f})")
    print("printable checks passed")


def test_limb_demo():
    from .demo_data import make_limb_demo_series

    for limb, expect in (("arm", {"shoulder", "elbow"}), ("leg", {"hip", "knee", "ankle"})):
        series = make_limb_demo_series(limb, seed=1)
        assert set(series) == expect, f"{limb}: got {set(series)}"
        for j, s in series.items():
            rom = s.rom_deg
            assert rom[1] - rom[0] > 30, f"{limb}/{j}: ROM too small {rom}"
            assert len(s.t) == len(s.theta_deg) == len(s.norm_landmarks)
    print("limb demo data checks passed")


def test_usage_tasks(m, ctx):
    from .simulate import simulate_final
    from .usage_demo import gait_trajectories, pick_place_trajectories

    t, gait, _ = gait_trajectories()
    for j, th in gait.items():
        assert 0 <= th.min() and th.max() <= 160, f"gait {j} out of range"
    t2, arm, payload, ev_ = pick_place_trajectories()
    assert arm["shoulder"].min() > 90, "shoulder task leaves measured-angle space"
    assert payload.max() == ev_["payload_kg"] and payload.min() == 0

    # carrying a payload must demand more force than moving empty
    pose_t = t2
    th = arm["elbow"]
    empty = simulate_final(m, pose_t, th, ctx)
    loaded = simulate_final(m, pose_t, th, ctx, payload_t=pose_t, payload_kg=payload * 2)
    assert loaded["F_peak"] > empty["F_peak"], "payload did not increase force demand"
    print(f"usage tasks: gait + pick/place OK; payload raises peak force "
          f"{empty['F_peak']:.0f} -> {loaded['F_peak']:.0f} N")
    print("usage task checks passed")


if __name__ == "__main__":
    test_geometry()
    test_limb_demo()
    m, final, ctx = test_evolution_and_sim()
    test_actuator_design(m, final, ctx)
    test_printable(m, final, ctx)
    test_usage_tasks(m, ctx)
    print("ALL SELF-TESTS PASSED")
