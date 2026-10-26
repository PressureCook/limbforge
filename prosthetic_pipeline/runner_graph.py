"""Graph-engine joint processing (SPEC Phases 2-5): open-topology evolution
through the full downstream chain - spec, STLs, plots, GIF, design.json v2."""

import os

import numpy as np

from .actuator_design import design_from_requirements, requirements_markdown
from .graph_evolve import GraphResult, evolve_graph, simulate_graph_final
from .graph_viz import (make_gif_graph, plot_search, plot_tracking_graph,
                        plot_transmission)
from .simulate import JointContext
from .topology import get_topology


def _traces_pack(final):
    dt = float(final["t"][1] - final["t"][0])
    rates = np.gradient(final["q"], dt, axis=1)
    return {"efforts": final["u"], "rates": rates, "dt": dt,
            "t": final["t"], "theta": np.deg2rad(final["theta_sim_deg"])}


def _design_v2(cfg, pose, ctx, genome, res: GraphResult, final, spec) -> dict:
    table = final["table"]
    in_rom = (table.theta >= np.deg2rad(pose.rom_deg[0])) & \
             (table.theta <= np.deg2rad(pose.rom_deg[1]))
    r_rng = {f"a{i}": [float(table.r[i][in_rom].min()),
                       float(table.r[i][in_rom].max())]
             for i in range(table.n_act)}
    return {
        "schema_version": 2,
        "joint": pose.joint, "side": pose.side,
        "topology": {"kind": genome.topology, **genome.to_dict()},
        "transmission_summary": {
            "r_range": r_rng,
            "I_eff_range_kg_m2": [float(table.I_eff.min()), float(table.I_eff.max())],
            "min_reciprocal_condition": float(1.0 / max(table.cond[in_rom].max(), 1.0)),
        },
        "performance": {
            "rmse_deg": final["rmse_deg"], "rmse_fit_deg": res.rmse_fit,
            "rmse_holdout_deg": res.rmse_holdout,
            "effort_peak": final["u_peak"].tolist(),
            "effort_demand_peak": final["u_demand_peak"].tolist(),
            "energy_J": final["energy_J"],
        },
        "evolution": {"engine": "graph", "topology_flag": cfg.topology,
                      "population": cfg.pop, "generations": cfg.generations,
                      "seed": cfg.seed, "final_cost": res.cost,
                      "innovations": genome.innovations()},
        "robustness": res.robustness,
        "allocation_check": res.qp_check,
        "actuator_spec": spec,
    }


def process_joint_graph(pose, cfg, out_dir: str, tee, banner,
                        usage_task=None) -> dict:
    from .runner import _write_csv, _write_json  # late import (no cycle at module load)

    os.makedirs(out_dir, exist_ok=True)
    rom = pose.rom_deg
    ctx = JointContext.from_pose(pose.joint, pose.Lp, pose.Ld, cfg.body_mass,
                                 L_mass=pose.L_mass)
    banner(f"Open-topology evolution: {pose.side} {pose.joint} "
           f"[{cfg.topology}]")
    res = evolve_graph(pose.t, pose.theta_deg, ctx, topology=cfg.topology,
                       pop_size=cfg.pop, generations=cfg.generations,
                       seed=cfg.seed, log=tee)
    genome = res.genome
    topo = get_topology(res.topology)
    desc = topo.describe(genome)
    tee(f"  winner: {res.topology} ({desc['n_actuators']} actuator(s), "
        f"{desc['n_springs']} spring(s)) | fit RMSE {res.rmse_fit:.2f} deg, "
        f"hold-out {res.rmse_holdout:.2f} deg")
    tee(f"  robustness: rmse std {res.robustness['rmse_std_deg']:.2f} deg, "
        f"invalid {100*res.robustness['invalid_fraction']:.0f}% | "
        f"QP-greedy gap {res.qp_check['rmse_gap_deg']:.2f} deg")

    usage_check = None
    if usage_task is not None:
        t_task, th_task, pl_task = usage_task
        u = simulate_graph_final(genome, ctx, t_task, th_task,
                                 payload_t=t_task, payload_kg=pl_task)
        sat = float(np.mean(np.any(
            np.abs(u["u"]) >= 0.98 * u["F_max"][:, None], axis=0)))
        usage_check = {"pre_rmse_deg": round(u["rmse_deg"], 2),
                       "pre_saturation": round(sat, 4), "re_evolved": False}
        if u["rmse_deg"] > 4.0 or sat > 0.03:
            tee(f"  usage pre-check: task RMSE {u['rmse_deg']:.1f} deg, "
                f"saturated {100*sat:.0f}% -> re-evolving with the task included")
            t_comb = np.concatenate([pose.t, pose.t[-1] + 0.5 + t_task])
            th_comb = np.concatenate([pose.theta_deg, th_task])
            pl_comb = np.concatenate([np.zeros(len(pose.t)), pl_task])
            res = evolve_graph(t_comb, th_comb, ctx, topology=cfg.topology,
                               pop_size=cfg.pop, generations=cfg.generations,
                               seed=cfg.seed, log=tee, payload_t=t_comb,
                               payload_kg=pl_comb,
                               max_duration=float(t_comb[-1]) + 1.0)
            genome = res.genome
            topo = get_topology(res.topology)
            u2 = simulate_graph_final(genome, ctx, t_task, th_task,
                                      payload_t=t_task, payload_kg=pl_task)
            sat2 = float(np.mean(np.any(
                np.abs(u2["u"]) >= 0.98 * u2["F_max"][:, None], axis=0)))
            usage_check.update({"re_evolved": True,
                                "post_rmse_deg": round(u2["rmse_deg"], 2),
                                "post_saturation": round(sat2, 4)})
            tee(f"  after re-evolution: task RMSE {u2['rmse_deg']:.2f} deg, "
                f"saturated {100*sat2:.1f}%")

    final = simulate_graph_final(genome, ctx, pose.t, pose.theta_deg)
    tee(f"  simulation (QP): RMSE {final['rmse_deg']:.2f} deg | peak effort "
        + ", ".join(f"a{i} {p:.0f}" for i, p in enumerate(final["u_peak"])))

    reqs = topo.actuator_requirements(genome, ctx, final["table"],
                                      _traces_pack(final))
    spec = design_from_requirements(reqs, ctx)
    with open(os.path.join(out_dir, "actuator_spec.md"), "w") as f:
        f.write(requirements_markdown(spec, res.topology))
    _write_json(os.path.join(out_dir, "actuator_spec.json"), spec)
    roll = spec["rollup"]
    tee(f"  actuators: {roll['n_actuators']}x, feasible {roll['all_feasible']}, "
        f"~{roll['total_mass_kg']*1000:.0f} g total, "
        f"battery ~{roll['battery_hours']:.1f} h")

    stl_info = None
    if cfg.stl:
        try:
            from .printable import from_parts
            stl_info = from_parts(topo.solid_model(genome, ctx, 1.0), out_dir)
            tee(f"  3D print: {stl_info['part_count']} STL files -> print_plate.stl")
        except RuntimeError as e:
            tee(f"  3D print skipped: {e}")

    F_caps = final["F_max"]
    plot_search(res.history_best, res.history_species, res.history_parts,
                os.path.join(out_dir, "convergence.png"))
    plot_transmission(final["table"], F_caps,
                      os.path.join(out_dir, "mechanism_curves.png"),
                      title=f"Evolved {res.topology} transmission")
    plot_tracking_graph(final, os.path.join(out_dir, "tracking.png"))
    make_gif_graph(pose, genome, ctx, final,
                   os.path.join(out_dir, "demo.gif"),
                   title_extra=f", RMSE {final['rmse_deg']:.1f} deg")
    _write_csv(os.path.join(out_dir, "trajectory.csv"), pose, final)
    _write_json(os.path.join(out_dir, "design.json"),
                _design_v2(cfg, pose, ctx, genome, res, final, spec))

    a0 = spec["actuators"][0]
    kind0 = a0.get("kind", "linear")
    if kind0 == "rotary" and a0.get("feasible"):
        motor = a0["drive"]["motor_class"] + f" @ {a0['drive']['ratio']:.0f}:1"
    elif a0.get("feasible"):
        motor = a0["variants"][a0["recommended"]]["drive"]["motor_class"]
    else:
        motor = "infeasible"
    report_lines = [
        f"- flexion ROM **{rom[0]:.0f}..{rom[1]:.0f} deg**, tracking RMSE "
        f"**{final['rmse_deg']:.2f} deg** (hold-out {res.rmse_holdout:.2f} deg)",
        f"- evolved topology: **{res.topology}** - {desc['n_actuators']} actuator(s), "
        f"{desc['n_springs']} spring(s), {desc['n_bodies']} bodies",
        f"- actuators: {motor}; system ~{roll['total_mass_kg']*1000:.0f} g, "
        f"battery ~{roll['battery_hours']:.1f} h; all feasible: {roll['all_feasible']}",
        f"- robustness under printed tolerances: RMSE std "
        f"{res.robustness['rmse_std_deg']:.2f} deg, "
        f"invalid {100*res.robustness['invalid_fraction']:.0f}%",
    ]

    return {
        "pose": pose, "ctx": ctx, "genome": genome, "res": res, "final": final,
        "spec": spec, "stl": stl_info, "rom": rom,
        "th_range": (np.deg2rad(rom[0]), np.deg2rad(rom[1])),
        "mech": None, "report_lines": report_lines,
        "summary": {
            "joint": pose.joint, "side": pose.side,
            "usage_check": usage_check,
            "rom_deg": [round(rom[0], 1), round(rom[1], 1)],
            "rmse_deg": round(final["rmse_deg"], 2),
            "rmse_holdout_deg": round(res.rmse_holdout, 2),
            "topology_kind": res.topology,
            "n_actuators": desc["n_actuators"],
            "n_springs": desc["n_springs"],
            "mechanism": {
                "F_class_N": round(float(F_caps[0])),
                "v_class_mm_s": round(float(final["v_max"][0] * 1000)),
                "stroke_mm": round(float((final["table"].q[0].max()
                                          - final["table"].q[0].min()) * 1000)),
                "L_ret_mm": round(float(final["table"].q[0].min() * 1000)),
            },
            "actuator": {
                "recommended": res.topology, "motor": motor,
                "mass_g": round(roll["total_mass_kg"] * 1000),
                "battery_h": round(roll["battery_hours"], 1),
                "fits": bool(roll["all_feasible"]),
            },
        },
    }


def scene_data_for(genome, ctx, thetas, final, idx):
    """Per-frame actuator endpoints, coupler link segments and utilisation for
    a mechgraph design (consumed by runner._write_scene)."""
    import numpy as np
    from .opengraph import _dec_anchor
    from .topology import get_topology
    topo = get_topology("mechgraph")
    n_act = len([a for a in genome.actuators if a.kind != "rotary"])
    F = len(thetas)
    ep = np.zeros((F, max(n_act, 1), 2, 2))
    segs = None
    have_couplers = bool(genome.bodies)
    if have_couplers:
        n_links = None
    u_tr = np.abs(np.asarray(final["u"]))[:, idx]        # (n_act, F)
    Fmax = np.asarray(final["F_max"])
    util = (u_tr / np.maximum(Fmax[:, None], 1e-9)).T
    seg_frames = []
    for f, th in enumerate(thetas):
        poses = topo.solve_body_poses(genome, ctx, float(th)) \
            if have_couplers else {
                "ground": (np.zeros(2), 0.0),
                "output": (np.zeros(2), np.pi - float(th))}

        def world(body_id, loc):
            p, phi = poses[body_id]
            c, s = np.cos(phi), np.sin(phi)
            return p + np.array([c * loc[0] - s * loc[1],
                                 s * loc[0] + c * loc[1]])

        ai = 0
        for a in genome.actuators:
            if a.kind == "rotary":
                continue
            pa = world(a.mount["body_a"], _dec_anchor(
                ctx, a.mount["body_a"], a.genes["ax"], a.genes["ay"]))
            pb = world(a.mount["body_b"], _dec_anchor(
                ctx, a.mount["body_b"], a.genes["bx"], a.genes["by"]))
            ep[f, ai, 0], ep[f, ai, 1] = pa, pb
            ai += 1
        if have_couplers:
            fr = []
            for jd in genome.joints:
                pa = world(jd.body_a, _dec_anchor(ctx, jd.body_a,
                                                  jd.genes["ax"], jd.genes["ay"]))
                pb = world(jd.body_b, _dec_anchor(ctx, jd.body_b,
                                                  jd.genes["bx"], jd.genes["by"]))
                fr.append((pa, pb))
            seg_frames.append(fr)
    if have_couplers and seg_frames:
        segs = np.array(seg_frames)
    return ep, segs, util
