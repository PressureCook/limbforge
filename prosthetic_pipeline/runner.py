"""Pipeline orchestration: one call runs every stage into a self-contained
run folder under runs/, and maintains runs/index.json across runs.

Two run shapes:
- single joint: capture -> pose -> evolve -> simulate -> actuator spec -> STL -> viz
- full limb (arm/leg): capture once -> pose for every limb joint -> the whole
  single-joint chain per joint (into joints/<name>/) -> full-limb animation
  and combined report.

Used by both the CLI (run_pipeline.py) and the web UI (app.py).
"""

import json
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import actuator_design as act
from . import evolve as ev
from . import visualize as viz
from .joints import LIMB_PRESETS, MOVEMENTS, SEGMENT_ANTHROPOMETRY
from .simulate import JointContext, make_target, simulate_final

DEFAULT_RUNS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


@dataclass
class RunConfig:
    input: Optional[str] = None
    demo: bool = False
    joint: str = "auto"
    side: str = "auto"
    limb: Optional[str] = None          # "arm" | "leg" -> full-limb run
    movement: Optional[str] = None      # MOVEMENTS key the recording claims to contain
    capture_verified: Optional[bool] = None  # guided recorder: all live checks passed
    stl: bool = True                    # generate 3D-printable parts
    engine: str = "legacy"              # "legacy" | "graph" (open-topology search)
    topology: str = "auto"              # graph engine: fixed kind or "auto"
    photo_fps: float = 4.0
    rotate: int = 0
    body_mass: float = 80.0
    pop: int = 160
    generations: int = 120
    seed: int = 0
    model: Optional[str] = None
    label: Optional[str] = None
    run_name: Optional[str] = None      # exact folder name under runs_root
    runs_root: str = DEFAULT_RUNS_ROOT
    out: Optional[str] = None           # explicit output folder (overrides naming)


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:40] or "run"


def resolve_run_dir(cfg: RunConfig) -> str:
    """Create and return the run folder (callers may pre-create it for uploads)."""
    if cfg.out:
        os.makedirs(cfg.out, exist_ok=True)
        return os.path.abspath(cfg.out)
    if not cfg.run_name:
        label = cfg.label or (
            ("demo-" + (cfg.limb or "joint")) if cfg.demo else
            _slug(os.path.basename(os.path.normpath(cfg.input or "run"))))
        base = f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"
        name, k = base, 2
        while os.path.exists(os.path.join(cfg.runs_root, name)):
            name, k = f"{base}-{k}", k + 1
        cfg.run_name = name
    run_dir = os.path.join(cfg.runs_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.abspath(run_dir)


_INDEX_LOCK = threading.Lock()  # in-process writers (the UI) serialize here


def _update_index(runs_root: str, entry: dict):
    os.makedirs(runs_root, exist_ok=True)
    path = os.path.join(runs_root, "index.json")
    with _INDEX_LOCK:
        _update_index_locked(path, entry)


def _update_index_locked(path: str, entry: dict):
    index = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                index = json.load(f)
        except Exception:
            index = []
    index = [e for e in index if e.get("name") != entry["name"]]
    index.append(entry)
    index.sort(key=lambda e: e.get("created", ""), reverse=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, path)


def _write_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _validate_movement_rom(cfg: RunConfig, series_by_joint: dict, tee):
    """Enforce that an uploaded/recorded movement actually contains the task.

    Recordings that do not reach the movement's required range of motion are
    rejected with actionable advice, so only correct movements are designed on.
    """
    if cfg.demo or not cfg.movement or cfg.movement not in MOVEMENTS:
        return
    req = MOVEMENTS[cfg.movement].get("min_rom", {})
    problems = []
    missing = [j for j in req if j not in series_by_joint]
    if missing:
        raise RuntimeError(
            f"The '{MOVEMENTS[cfg.movement]['label']}' movement requires the "
            f"{', '.join(missing)} joint(s), which were not extracted from this "
            "recording - check that the right joint/limb is selected and the "
            "whole limb is visible.")
    for joint, need in req.items():
        s = series_by_joint[joint]
        lo, hi = s.rom_deg
        got = hi - lo
        if got + 1e-6 < need:
            problems.append(f"{joint}: recorded range {got:.0f} deg, task needs >= {need:.0f} deg")
    if problems:
        raise RuntimeError(
            f"The recording does not contain the '{MOVEMENTS[cfg.movement]['label']}' movement: "
            + "; ".join(problems)
            + ". Re-record with slower, larger movements, filmed side-on with the whole limb visible.")
    tee(f"  movement check: '{MOVEMENTS[cfg.movement]['label']}' requirements met "
        + ", ".join(f"{j} ROM >= {v} deg" for j, v in req.items()))


def _process_joint(pose, cfg: RunConfig, out_dir: str, tee, banner,
                   usage_task=None, ctx_override=None, coupling=None,
                   coupling_info=None) -> dict:
    """Evolve + simulate + actuator spec + plots (+ STL) for one PoseSeries.

    usage_task: optional (t, theta_deg, payload_kg) for the limb's real-world
    task. The evolved design is pre-checked against it and, if force/speed
    saturation makes it inadequate, re-evolved with the task appended to the
    training trajectory (designing for real-world usage, not just the recording).
    """
    if cfg.engine == "graph":
        from .runner_graph import process_joint_graph
        return process_joint_graph(pose, cfg, out_dir, tee, banner,
                                   usage_task=usage_task)
    os.makedirs(out_dir, exist_ok=True)
    rom = pose.rom_deg
    ctx = ctx_override or JointContext.from_pose(
        pose.joint, pose.Lp, pose.Ld, cfg.body_mass, L_mass=pose.L_mass)
    seg = SEGMENT_ANTHROPOMETRY[pose.joint]
    tee(f"  driven segment: {seg['name']}  mass {ctx.mass:.2f} kg, "
        f"length {ctx.Ld*100:.0f} cm, inertia {ctx.inertia*1000:.1f} g m^2")
    cpl_t = cpl_kg = None
    if coupling is not None:
        cpl_t, cpl_kg = coupling
        tee(f"  coupled design: carrying distal hardware + recorded distal "
            f"motion (equivalent payload {1000*cpl_kg.min():+.0f}..."
            f"{1000*cpl_kg.max():+.0f} g at L_mass)")

    banner(f"Evolutionary design: {pose.side} {pose.joint}")
    res = ev.evolve(pose.t, pose.theta_deg, ctx, pop_size=cfg.pop,
                    generations=cfg.generations, seed=cfg.seed, log=tee,
                    payload_t=cpl_t, payload_kg=cpl_kg)
    m = res.best_mechanism

    usage_check = None
    if usage_task is not None:
        t_task, th_task, pl_task = usage_task
        u = simulate_final(m, t_task, th_task, ctx, payload_t=t_task, payload_kg=pl_task)
        sat = float(np.mean(np.abs(u["F"]) >= 0.98 * m.F_max))
        usage_check = {"pre_rmse_deg": round(u["rmse_deg"], 2),
                       "pre_saturation": round(sat, 4), "re_evolved": False}
        if u["rmse_deg"] > 4.0 or sat > 0.03:
            tee(f"  usage pre-check: task RMSE {u['rmse_deg']:.1f} deg, saturated "
                f"{100*sat:.0f}% of the time -> re-evolving with the task included")
            gap = 0.5
            t_comb = np.concatenate([pose.t, pose.t[-1] + gap + t_task])
            th_comb = np.concatenate([pose.theta_deg, th_task])
            base_pl = (np.interp(pose.t, cpl_t, cpl_kg)
                       if cpl_kg is not None else np.zeros(len(pose.t)))
            pl_comb = np.concatenate([base_pl, pl_task])
            res = ev.evolve(t_comb, th_comb, ctx, pop_size=cfg.pop,
                            generations=cfg.generations, seed=cfg.seed, log=tee,
                            payload_t=t_comb, payload_kg=pl_comb,
                            max_duration=float(t_comb[-1]) + 1.0)
            m = res.best_mechanism
            u2 = simulate_final(m, t_task, th_task, ctx, payload_t=t_task, payload_kg=pl_task)
            sat2 = float(np.mean(np.abs(u2["F"]) >= 0.98 * m.F_max))
            usage_check.update({"re_evolved": True,
                                "post_rmse_deg": round(u2["rmse_deg"], 2),
                                "post_saturation": round(sat2, 4)})
            tee(f"  after re-evolution: task RMSE {u2['rmse_deg']:.2f} deg, "
                f"saturated {100*sat2:.1f}% of the time")
        else:
            tee(f"  usage pre-check: design already handles the task "
                f"(RMSE {u['rmse_deg']:.2f} deg)")

    tee(f"  best: anchors a={m.a*1000:.0f} mm @ {np.degrees(m.alpha):+.0f} deg, "
        f"b={m.b*1000:.0f} mm @ {np.degrees(m.beta):+.0f} deg | "
        f"{m.L_ret*1000:.0f} mm retracted + {m.stroke*1000:.0f} mm stroke, "
        f"{m.F_max:.0f} N, {m.v_max*1000:.0f} mm/s")

    final = simulate_final(m, pose.t, pose.theta_deg, ctx,
                           payload_t=cpl_t, payload_kg=cpl_kg)
    tee(f"  simulation: RMSE {final['rmse_deg']:.2f} deg | peak force "
        f"{final['F_peak']:.0f} N ({100*final['F_demand_peak']/m.F_max:.0f}% of capacity demanded)")

    spec = act.design_actuator(m, final, ctx)
    _, th_d, _ = make_target(pose.t, pose.theta_deg)
    th_lo, th_hi = float(th_d.min()), float(th_d.max())
    spec_paths = act.write_spec_files(spec, m, ctx, out_dir,
                                      th_mid=0.5 * (th_lo + th_hi))
    rec = spec["variants"][spec["recommended"]]
    tee(f"  actuator: {spec['recommended']} "
        f"({rec['drive']['screw']['d_nom_mm']:.0f} mm screw, "
        f"{rec['drive']['screw']['lead_mm']:.0f} mm lead, "
        f"{rec['drive']['gear_ratio']:.1f}:1, {rec['drive']['motor_class']}) | "
        f"{rec['bus_voltage_V']:.0f} V, ~{rec['mass_est_kg']*1000:.0f} g, "
        f"battery ~{rec['battery_hours']:.1f} h")
    if not spec["packaging"]["fits"]:
        tee("  WARNING: packaging check failed - see actuator_spec.md")

    stl_info = None
    if cfg.stl:
        try:
            from .printable import generate_printables
            stl_info = generate_printables(m, spec, ctx, out_dir,
                                           th_mid=0.5 * (th_lo + th_hi))
            tee(f"  3D print: {stl_info['part_count']} STL files"
                + (f" (scaled {stl_info['scale']:.2f}x)" if stl_info["scale"] < 1 else "")
                + " -> print_plate.stl")
        except RuntimeError as e:
            tee(f"  3D print skipped: {e}")

    viz.plot_convergence(res.history_best, res.history_mean,
                         os.path.join(out_dir, "convergence.png"))
    viz.plot_mechanism_curves(m, ctx, th_lo, th_hi,
                              os.path.join(out_dir, "mechanism_curves.png"))
    viz.plot_tracking(final, m, os.path.join(out_dir, "tracking.png"))
    _write_csv(os.path.join(out_dir, "trajectory.csv"), pose, final)
    design = _design_dict(cfg, pose, ctx, m, res, final, rom, seg)
    _write_json(os.path.join(out_dir, "design.json"), design)

    return {"pose": pose, "ctx": ctx, "mech": m, "res": res, "final": final,
            "spec": spec, "stl": stl_info, "rom": rom, "th_range": (th_lo, th_hi),
            "summary": {
                "joint": pose.joint, "side": pose.side,
                "usage_check": usage_check,
                "coupling": coupling_info,
                "rom_deg": [round(rom[0], 1), round(rom[1], 1)],
                "rmse_deg": round(final["rmse_deg"], 2),
                "mechanism": {"F_class_N": round(m.F_max),
                              "v_class_mm_s": round(m.v_max * 1000),
                              "stroke_mm": round(m.stroke * 1000),
                              "L_ret_mm": round(m.L_ret * 1000)},
                "actuator": {"recommended": spec["recommended"],
                             "motor": rec["drive"]["motor_class"],
                             "custom_drive": "custom" in rec["drive"]["motor_class"],
                             "mass_g": round(rec["mass_est_kg"] * 1000),
                             "battery_h": round(rec["battery_hours"], 1),
                             "fits": spec["packaging"]["fits"]},
            }}


def run(cfg: RunConfig, log: Callable[[str], None] = print) -> dict:
    if cfg.limb and cfg.engine == "graph":
        raise ValueError("--limb with --engine graph is not supported yet "
                         "(SPEC-2 E-12): use single joints or the legacy engine")
    """Execute the full pipeline. Returns the run summary (also saved as run.json)."""
    run_dir = resolve_run_dir(cfg)
    name = os.path.basename(run_dir)
    log_file = open(os.path.join(run_dir, "log.txt"), "a")

    def tee(msg=""):
        log_file.write(str(msg) + "\n")
        log_file.flush()
        log(msg)

    def banner(msg):
        tee(f"\n=== {msg} " + "=" * max(0, 60 - len(msg)))

    created = time.strftime("%Y-%m-%d %H:%M:%S")
    summary = {"name": name, "created": created, "status": "running",
               "limb": cfg.limb,
               "config": {k: v for k, v in cfg.__dict__.items() if k != "runs_root"},
               "artifacts": {}}
    _write_json(os.path.join(run_dir, "run.json"), summary)
    _update_index(cfg.runs_root, _index_entry(summary))
    t0 = time.time()

    try:
        # ---- capture + pose --------------------------------------------------
        frames = meta = None
        if cfg.demo:
            banner(f"Stage 1-2: synthetic demo movement ({cfg.limb or 'elbow curls'})")
            source_kind = "demo"
        else:
            banner("Stage 1: loading movement data")
            from .capture import load_frames
            frames, meta = load_frames(cfg.input, photo_fps=cfg.photo_fps, rotate=cfg.rotate)
            source_kind = meta["kind"]
            tee(f"  {len(frames)} frames from {meta['kind']}: {cfg.input}")
            if len(frames) < 4:
                raise RuntimeError("Need at least 4 frames of the movement (20+ recommended).")
            banner("Stage 2: pose estimation (MediaPipe)")

        if cfg.limb:
            summary_out = _run_limb(cfg, run_dir, frames, meta, tee, banner)
        else:
            summary_out = _run_single(cfg, run_dir, frames, meta, tee, banner)

        summary.update(summary_out)
        summary.update({"status": "done", "elapsed_s": round(time.time() - t0, 1),
                        "source": "demo" if cfg.demo else cfg.input,
                        "kind": source_kind, "artifacts": _artifact_map(run_dir)})
        banner(f"Done in {summary['elapsed_s']:.0f} s")
    except Exception as e:
        log_file.write(traceback.format_exc())
        log_file.flush()
        summary.update({"status": "failed", "error": str(e),
                        "elapsed_s": round(time.time() - t0, 1),
                        "artifacts": _artifact_map(run_dir)})
        tee(f"\nERROR: {e}")
    finally:
        log_file.close()

    _write_json(os.path.join(run_dir, "run.json"), summary)
    _update_index(cfg.runs_root, _index_entry(summary))
    summary["run_dir"] = run_dir
    return summary


def _run_single(cfg, run_dir, frames, meta, tee, banner) -> dict:
    if cfg.demo:
        from .demo_data import make_demo_series
        pose = make_demo_series(seed=cfg.seed)
    else:
        from .pose import extract_pose
        pose = extract_pose(frames, meta, joint=cfg.joint, side=cfg.side,
                            model_path=cfg.model, out_dir=run_dir)
        tee(f"  person detected in {pose.n_detected}/{pose.n_frames} frames")
    dur = float(pose.t[-1] - pose.t[0])
    rom = pose.rom_deg
    tee(f"  joint: {pose.side} {pose.joint} | {len(pose.t)} samples over {dur:.1f} s | "
        f"flexion {rom[0]:.0f}..{rom[1]:.0f} deg")
    for note in pose.notes:
        tee(f"  note: {note}")
    _validate_movement_rom(cfg, {pose.joint: pose}, tee)
    if cfg.capture_verified is False:
        tee("  note: guided recording had UNVERIFIED steps (participant skipped "
            "live checks); treat results with extra caution")

    r = _process_joint(pose, cfg, run_dir, tee, banner)

    banner("Visualization and report")
    th_lo, th_hi = r["th_range"]
    if cfg.engine == "graph":
        pass  # demo.gif already produced by the graph processor
    else:
        viz.make_gif(pose, r["mech"], r["ctx"], r["final"], th_lo, th_hi,
                 os.path.join(run_dir, "demo.gif"),
                 title_extra=f", RMSE {r['final']['rmse_deg']:.1f} deg")
    _make_thumb(os.path.join(run_dir, "demo.gif"), os.path.join(run_dir, "thumb.png"))
    try:   # 3D anatomical scene (SPEC-2 V-1): one geometric source of truth
        _write_scene(run_dir, pose, r, tee)
    except Exception as e:
        tee(f"  scene export skipped: {e}")
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(_report_md(cfg, pose, r, dur))
    tee("  wrote demo.gif, report.md, actuator_spec.md, plots, STL parts")

    out = dict(r["summary"])
    out.update({"duration_s": round(dur, 1)})
    return out


def _run_limb(cfg, run_dir, frames, meta, tee, banner) -> dict:
    side = cfg.side
    if cfg.demo:
        from .demo_data import make_limb_demo_series
        series_map = make_limb_demo_series(cfg.limb, seed=cfg.seed)
        errors = {}
        side = "right"
    else:
        from .pose import extract_limb
        series_map, errors, side = extract_limb(frames, meta, cfg.limb, cfg.side,
                                                model_path=cfg.model, out_dir=run_dir)
        first = next(iter(series_map.values()))
        tee(f"  person detected in {first.n_detected}/{first.n_frames} frames")
    tee(f"  limb: {side} {cfg.limb} -> joints: {', '.join(series_map)}"
        + (f" (failed: {', '.join(errors)})" if errors else ""))
    _validate_movement_rom(cfg, series_map, tee)
    if cfg.capture_verified is False:
        tee("  note: guided recording had UNVERIFIED steps (participant skipped "
            "live checks); treat results with extra caution")

    # the limb's real-world task, used both to pre-check and (if needed)
    # re-evolve each design, and for the final usage demonstration
    from .usage_demo import task_trajectories_for
    t_task, task_traj, task_payload = task_trajectories_for(cfg.limb)

    results = {}
    # full-body mode designs DISTAL-FIRST so every proximal joint carries the
    # hardware mass and the recorded motion of the joints below it (body.py)
    if cfg.limb == "body":
        from .body import (BODY_DESIGN_ORDER, coupled_context,
                           trajectory_coupling)
        order = [j for j in BODY_DESIGN_ORDER if j in series_map]
        tee(f"  coupled design order (distal-first): {' -> '.join(order)}")
    else:
        order = list(series_map)
    for jn in order:
        pose = series_map[jn]
        banner(f"Joint {jn} ({side})")
        rom = pose.rom_deg
        tee(f"  {len(pose.t)} samples | flexion {rom[0]:.0f}..{rom[1]:.0f} deg")
        jdir = os.path.join(run_dir, "joints", jn)
        usage_task = None
        if jn in task_traj:
            usage_task = (t_task, task_traj[jn], task_payload)
        ctx_ov = coupling = cinfo = None
        if cfg.limb == "body":
            ctx_ov, carried = coupled_context(jn, series_map, results,
                                              cfg.body_mass)
            ct, ckg, tinfo = trajectory_coupling(jn, series_map, cfg.body_mass)
            if ct is not None:
                coupling = (ct, ckg)
            cinfo = {"carried_hardware": carried,
                     "trajectory": tinfo or None,
                     "designed_after": [j for j in results]}
            if carried:
                tee("  carrying: " + ", ".join(
                    f"{c['joint']} hardware {c['mass_g']} g @ {c['at_mm']} mm"
                    for c in carried))
            if tinfo:
                tee(f"  distal-motion coupling from {tinfo['from']}: COM lever "
                    f"shifts {tinfo['lever_shift_mm'][0]}..{tinfo['lever_shift_mm'][1]} mm")
        results[jn] = _process_joint(pose, cfg, jdir, tee, banner,
                                     usage_task=usage_task, ctx_override=ctx_ov,
                                     coupling=coupling, coupling_info=cinfo)
    if cfg.limb == "body":   # report in anatomical order
        results = {j: results[j] for j in series_map if j in results}

    banner("Full-limb simulation and report")
    finals = {j: r["final"] for j, r in results.items()}
    mechs = {j: r["mech"] for j, r in results.items()}
    ctxs = {j: r["ctx"] for j, r in results.items()}
    if cfg.engine == "graph":
        tee("  limb/usage animations use the legacy engine's mechanism objects; "
            "graph runs ship per-joint demo GIFs instead (joints/<joint>/demo.gif)")
    viz.make_limb_gif(cfg.limb, side, series_map, finals, mechs, ctxs,
                      os.path.join(run_dir, "limb_demo.gif"))
    _make_thumb(os.path.join(run_dir, "limb_demo.gif"), os.path.join(run_dir, "thumb.png"))

    # real-world usage demonstration on the evolved limb
    usage_summary = None
    usage_need = LIMB_PRESETS["leg"] if cfg.limb == "body" else LIMB_PRESETS[cfg.limb]
    if set(usage_need) <= set(results):
        banner("Real-world usage demo")
        from .usage_demo import make_usage_gif, run_usage_scenario
        usage = run_usage_scenario(cfg.limb, mechs, ctxs)
        tee(f"  scenario: {usage['scenario']}")
        usage_summary = {"scenario": usage["scenario"], "joints": {}}
        for j, u in usage["joints"].items():
            tee(f"  {j}: tracking RMSE {u['rmse_deg']:.2f} deg | "
                f"peak force {100*u['utilization']:.0f}% of capacity | "
                f"saturated {100*u['saturation_frac']:.1f}% of the time")
            usage_summary["joints"][j] = {
                "rmse_deg": round(u["rmse_deg"], 2),
                "utilization": round(u["utilization"], 3),
                "saturation_frac": round(u["saturation_frac"], 4),
            }
        make_usage_gif(cfg.limb, side, usage, mechs, ctxs,
                       os.path.join(run_dir, "usage_demo.gif"))
        tee("  wrote usage_demo.gif")
    else:
        tee("  usage demo skipped: not all limb joints were extractable")

    if cfg.limb == "body":
        try:
            _write_body_scene(run_dir, side, series_map, results, tee)
        except Exception as e:
            tee(f"  body scene skipped: {e}")
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(_limb_report_md(cfg, side, results, errors, usage_summary))
    tee("  wrote limb_demo.gif, report.md, per-joint folders under joints/")

    dur = max(float(s.t[-1] - s.t[0]) for s in series_map.values())
    worst = max(results.values(), key=lambda r: r["final"]["rmse_deg"])
    return {
        "joint": f"{cfg.limb} ({len(results)} joints)", "side": side,
        "duration_s": round(dur, 1),
        "rmse_deg": round(worst["final"]["rmse_deg"], 2),
        "mechanism": worst["summary"]["mechanism"],
        "actuator": worst["summary"]["actuator"],
        "joints": {j: r["summary"] for j, r in results.items()},
        "joint_errors": errors,
        "usage": usage_summary,
    }


def _index_entry(summary: dict) -> dict:
    return {k: summary.get(k) for k in
            ("name", "created", "status", "kind", "joint", "side", "limb",
             "rmse_deg", "duration_s", "error")} | {
        "F_class_N": (summary.get("mechanism") or {}).get("F_class_N"),
        "topology_kind": summary.get("topology_kind"),
        "actuator": (summary.get("actuator") or {}).get("recommended"),
        "has_thumb": "thumb.png" in (summary.get("artifacts") or {}),
    }


def _artifact_map(run_dir: str) -> dict:
    """Relative paths of all known artifacts, including per-joint subfolders."""
    known = ["demo.gif", "limb_demo.gif", "usage_demo.gif", "thumb.png", "tracking.png",
             "mechanism_curves.png", "convergence.png", "actuator_drawing.png",
             "actuator_spec.md", "actuator_spec.json", "report.md", "design.json",
             "trajectory.csv", "log.txt", "scene.json", "scene.glb",
             "body_scene.json",
             "scene_preview.png", "print_plate.stl", "assembled_preview.stl",
             "printable_preview.png"]
    out = {}
    for base, _, files in os.walk(run_dir):
        rel_base = os.path.relpath(base, run_dir)
        if rel_base.startswith("input"):
            continue
        for f in sorted(files):
            rel = f if rel_base == "." else f"{rel_base}/{f}"
            if f in known or f.startswith("pose_check") or f.startswith("part_"):
                out[rel] = rel
    return out


def _write_body_scene(run_dir, side, series_map, results, tee):
    """Interactive full-body scene: per-frame skeleton + every joint's
    mechanism + embedded per-joint stats/spec for the click inspector."""
    joints = [j for j in ("shoulder", "elbow", "hip", "knee", "ankle")
              if j in results]
    ref = max(joints, key=lambda j: series_map[j].t[-1])
    t_ref = series_map[ref].t
    step = max(1, int(np.ceil(len(t_ref) / 110)))
    times = t_ref[::step]
    sim = {j: np.interp(times, results[j]["final"]["t"],
                        results[j]["final"]["theta_sim_deg"]) for j in joints}
    util = {}
    for j in joints:
        F = np.abs(np.interp(times, results[j]["final"]["t"],
                             results[j]["final"]["F"]))
        cap = results[j]["mech"].F_max if results[j]["mech"] is not None             else max(float(F.max()), 1e-9)
        util[j] = F / max(cap, 1e-9)
    seg_len = {j: results[j]["ctx"].Ld for j in joints}
    frames = []
    for k, tk in enumerate(times):
        angles = {j: float(sim[j][k]) for j in joints}
        chain = viz._chain_points("body", angles, seg_len)
        fr = {"t": round(float(tk), 3), "joints": {}}
        for j in joints:
            J, P, tip, phi = chain[j]
            entry = {"pos": [round(float(J[0]), 4), round(float(J[1]), 4)],
                     "tip": [round(float(tip[0]), 4), round(float(tip[1]), 4)],
                     "theta_deg": round(angles[j], 1),
                     "util": round(float(util[j][k]), 3)}
            m = results[j]["mech"]
            if m is not None:
                Aw, Bw = viz._mech_anchors_world(m, j, J, P,
                                                 np.deg2rad(angles[j]))
                entry["act"] = [[round(float(Aw[0]), 4), round(float(Aw[1]), 4)],
                                [round(float(Bw[0]), 4), round(float(Bw[1]), 4)]]
            fr["joints"][j] = entry
        frames.append(fr)

    panels = {}
    for j in joints:
        s = results[j]["summary"]
        spec = results[j]["spec"]
        rec = spec["variants"][spec["recommended"]] if "variants" in spec else {}
        panels[j] = {
            "summary": s,
            "spec_highlights": {
                "recommended": spec.get("recommended"),
                "screw": (rec.get("drive", {}).get("screw") or {}),
                "gear_ratio": rec.get("drive", {}).get("gear_ratio"),
                "motor": rec.get("drive", {}).get("motor_class"),
                "bus_V": rec.get("bus_voltage_V"),
                "mass_g": round(1000 * rec.get("mass_est_kg", 0)),
                "battery_h": round(rec.get("battery_hours", 0), 1),
                "custom_drive": "custom" in str(
                    rec.get("drive", {}).get("motor_class", "")),
                "fits": spec.get("packaging", {}).get("fits"),
            },
            "files": {
                "spec": f"joints/{j}/actuator_spec.md",
                "tracking": f"joints/{j}/tracking.png",
                "drawing": f"joints/{j}/actuator_drawing.png",
            },
        }
    doc = {
        "schema_version": 1, "side": side, "joints": joints,
        "seg_len": {j: round(float(seg_len[j]), 4) for j in joints},
        "coupling_note": (
            "Coupled design, distal-first: every proximal joint was evolved "
            "carrying the hardware mass of the joints below it, and against "
            "the load profile the recorded distal motion produces. Reaction "
            "torques from distal acceleration are NOT modelled."),
        "frames": frames, "panels": panels,
    }
    with open(os.path.join(run_dir, "body_scene.json"), "w") as f:
        json.dump(doc, f)
    tee("  wrote body_scene.json (interactive full-body inspector)")


def _write_scene(run_dir, pose, r, tee):
    """Export scene.glb/json/preview from the SAME data the simulator used."""
    from . import mechanism as mech_mod
    from .scene import build_scene_files, scene_frames_from_final
    final, ctx = r["final"], r["ctx"]
    idx, thetas = scene_frames_from_final(final)
    F_tr = np.abs(np.asarray(final["F"]))[idx]
    links = None
    if r.get("mech") is not None:                       # legacy engine
        m = r["mech"]
        A = np.array(mech_mod.anchor_A(m.a, m.alpha))
        ep = np.zeros((len(idx), 1, 2, 2))
        for i, th in enumerate(thetas):
            ep[i, 0, 0] = A
            ep[i, 0, 1] = np.array(mech_mod.anchor_B(m.b, m.beta, float(th)))
        util = (F_tr / max(m.F_max, 1e-9))[:, None]
    elif r.get("genome") is not None and r["genome"].topology == "mechgraph":
        from .runner_graph import scene_data_for
        ep, links, util = scene_data_for(r["genome"], ctx, thetas, final, idx)
    else:
        tee("  scene export: no geometry source for this topology (skipped)")
        return
    build_scene_files(run_dir, ctx, pose.joint, pose.side, final,
                      thetas, ep, link_segments=links, util_of_frame=util)
    tee("  wrote scene.glb + scene.json (3D anatomical viewer) + scene_preview.png")


def _make_thumb(gif_path: str, thumb_path: str):
    try:
        from PIL import Image
        im = Image.open(gif_path)
        im.seek(min(2, im.n_frames - 1))
        im = im.convert("RGB")
        im.thumbnail((360, 360))
        im.save(thumb_path)
    except Exception:
        pass


def _write_csv(path: str, pose, final):
    sim_at = np.interp(pose.t, final["t"], final["theta_sim_deg"])
    L_at = np.interp(pose.t, final["t"], final["L"])
    F_at = np.interp(pose.t, final["t"], final["F"])
    with open(path, "w") as f:
        f.write("t_s,theta_observed_deg,theta_sim_deg,actuator_length_m,actuator_force_N\n")
        for row in zip(pose.t, pose.theta_deg, sim_at, L_at, F_at):
            f.write(",".join(f"{v:.5f}" for v in row) + "\n")


def _design_dict(cfg, pose, ctx, m, res, final, rom, seg):
    return {
        "joint": pose.joint, "side": pose.side,
        "segment_lengths_m": {"proximal": ctx.Lp, "distal": ctx.Ld},
        "driven_segment": {"name": seg["name"], "mass_kg": ctx.mass,
                           "com_m": ctx.com, "inertia_kg_m2": ctx.inertia},
        "observed_rom_deg": {"min": rom[0], "max": rom[1]},
        "mechanism": m.to_dict(),
        "performance": {"rmse_deg": final["rmse_deg"], "F_peak_N": final["F_peak"],
                        "v_peak_m_s": final["v_peak"]},
        "evolution": {"population": cfg.pop, "generations": cfg.generations,
                      "seed": cfg.seed, "final_cost": res.best_cost,
                      "cost_breakdown": res.cost_breakdown},
    }


def _joint_section_md(r, title_prefix=""):
    if r.get("report_lines"):
        return list(r["report_lines"])
    m, spec, final = r["mech"], r["spec"], r["final"]
    rec = spec["variants"][spec["recommended"]]
    rom = r["rom"]
    lines = [
        f"- flexion ROM **{rom[0]:.0f}..{rom[1]:.0f} deg**, tracking RMSE **{final['rmse_deg']:.2f} deg**",
        f"- mechanism: anchors {m.a*1000:.0f} mm @ {np.degrees(m.alpha):+.0f} deg / "
        f"{m.b*1000:.0f} mm @ {np.degrees(m.beta):+.0f} deg; "
        f"{m.L_ret*1000:.0f} mm retracted + {m.stroke*1000:.0f} mm stroke; "
        f"class {m.F_max:.0f} N / {m.v_max*1000:.0f} mm/s",
        f"- actuator: **{spec['recommended']}**, {rec['drive']['screw']['d_nom_mm']:.0f} mm screw "
        f"x {rec['drive']['screw']['lead_mm']:.0f} mm lead, {rec['drive']['gear_ratio']:.1f}:1, "
        f"{rec['drive']['motor_class']}, {rec['bus_voltage_V']:.0f} V, "
        f"~{rec['mass_est_kg']*1000:.0f} g, battery ~{rec['battery_hours']:.1f} h",
        f"- packaging {'fits' if spec['packaging']['fits'] else 'DOES NOT FIT'} "
        f"(margin {spec['packaging']['margin_mm']:.0f} mm)",
    ]
    if r["stl"]:
        lines.append(f"- 3D print: {r['stl']['part_count']} STL parts"
                     + (f", scaled {r['stl']['scale']:.2f}x" if r['stl']['scale'] < 1 else ""))
    return lines


def _report_md(cfg, pose, r, dur):
    seg = SEGMENT_ANTHROPOMETRY[pose.joint]
    ctx = r["ctx"]
    lines = [
        "# Prosthetic actuator joint - run report\n",
        f"**Joint:** {pose.side} {pose.joint} &nbsp; **Source:** "
        + ("synthetic demo" if pose.synthetic else f"`{cfg.input}`") + "\n",
        "## Movement extracted",
        f"- {len(pose.t)} samples over {dur:.1f} s"
        + ("" if pose.synthetic else f" (person detected in {pose.n_detected}/{pose.n_frames} frames)"),
        f"- segment lengths from pose: proximal {ctx.Lp*100:.0f} cm, distal {ctx.Ld*100:.0f} cm",
        f"- driven segment model: {seg['name']}, {ctx.mass:.2f} kg (from {cfg.body_mass:.0f} kg body mass)",
        "",
        "## Results",
    ] + _joint_section_md(r) + [
        "",
        "## Files",
        "- `demo.gif` animated comparison; `actuator_spec.md` full engineering sheet;",
        "- `print_plate.stl` + `part_*.stl` 3D-printable demonstrator; `printable_preview.png`;",
        "- `tracking.png`, `mechanism_curves.png`, `convergence.png`, `design.json`, `trajectory.csv`.",
        "",
        "## Notes and limitations",
        "- 2D sagittal-plane model; gravity restores each joint toward its "
        "anatomical neutral pose (theta_neutral per joint).",
        "- Anthropometric mass properties (Winter tables) scaled by body mass.",
    ]
    for note in pose.notes:
        lines.append(f"- {note}")
    lines.append("- Exploration/demo tool only - not a medical device.")
    return "\n".join(lines) + "\n"


def _limb_report_md(cfg, side, results, errors, usage_summary=None):
    lines = [
        f"# Full {cfg.limb} - prosthetic joint breakdown\n",
        f"**Side:** {side} &nbsp; **Joints analyzed:** {', '.join(results)}"
        + (f" &nbsp; (not extractable: {', '.join(errors)})" if errors else "") + "\n",
        "Every joint gets its own evolved mechanism, physics validation, real-world",
        "actuator spec and 3D-printable parts in `joints/<joint>/`.\n",
    ]
    if cfg.movement and cfg.movement in MOVEMENTS:
        lines.append(f"**Recorded movement task:** {MOVEMENTS[cfg.movement]['label']}\n")
    for jn, r in results.items():
        lines.append(f"## {side} {jn}")
        lines += _joint_section_md(r)
        lines.append("")
    if usage_summary:
        lines += [f"## Real-world usage: {usage_summary['scenario']}",
                  "The evolved actuators driving the task (see `usage_demo.gif`):"]
        for j, u in usage_summary["joints"].items():
            sat = (f", saturated {100*u['saturation_frac']:.1f}% of the time"
                   if u["saturation_frac"] > 0.001 else "")
            lines.append(f"- {j}: RMSE **{u['rmse_deg']:.2f} deg**, peak force "
                         f"**{100*u['utilization']:.0f}%** of capacity{sat}")
        lines.append("")
    lines += [
        "## Files",
        "- `limb_demo.gif` - full-limb animation of the recorded movement",
        "- `usage_demo.gif` - the limb performing a real-world task (walk / pick & place)",
        "- `joints/<joint>/` - complete per-joint outputs (spec, STL, plots, data)",
        "",
        "## Notes",
        "- Joints are designed independently; interaction forces between joints are not modeled.",
        "- Usage-task trajectories are synthetic normative-style profiles.",
        "- Ankle numbers cover the hanging foot only: standing/push-off ankle torque is ground-reaction dominated and NOT modeled.",
        "- Ankle numbers cover the hanging foot only: standing/push-off ankle torque is ground-reaction dominated and NOT modeled - do not size a weight-bearing ankle from this.",
        "- 2D sagittal-plane model; not a medical device.",
    ]
    if errors:
        for jn, err in errors.items():
            lines.append(f"- {jn}: {err}")
    return "\n".join(lines) + "\n"
