#!/usr/bin/env python3
"""Wrapper CLI: photos/video -> pose -> evolved joint -> simulation -> actuator spec.

Each run gets its own folder under runs/ (runs/index.json tracks them all).

Examples
--------
    python run_pipeline.py --demo
    python run_pipeline.py --input ~/Pictures/elbow_curls --joint auto --label curls
    python run_pipeline.py --input curl.mov --joint elbow --side right --body-mass 75

Prefer a browser? `python app.py` starts the web UI at http://127.0.0.1:5001
"""

import argparse
import sys

from prosthetic_pipeline.joints import JOINT_NAMES, MOVEMENTS
from prosthetic_pipeline.runner import DEFAULT_RUNS_ROOT, RunConfig, run


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Evolve and simulate a prosthetic actuator joint from movement photos/video.")
    ap.add_argument("--input", help="folder of photos OR a video file of the movement")
    ap.add_argument("--demo", action="store_true",
                    help="run on a synthetic elbow-curl trajectory (no photos needed)")
    ap.add_argument("--joint", default="auto", choices=["auto"] + JOINT_NAMES)
    ap.add_argument("--side", default="auto", choices=["auto", "left", "right"])
    ap.add_argument("--limb", choices=["arm", "leg", "body"],
                    help="full-limb mode: every joint of the limb; 'body' = "
                         "full-body coupled design (distal-first, each joint "
                         "carrying the hardware and motion of those below)")
    ap.add_argument("--movement", choices=list(MOVEMENTS),
                    help="movement task the recording contains (enforces its ROM requirements)")
    ap.add_argument("--no-stl", action="store_true", help="skip 3D-printable STL generation")
    ap.add_argument("--engine", default="legacy", choices=["legacy", "graph"],
                    help="graph = open-topology mechanism evolution (SPEC)")
    ap.add_argument("--topology", default="auto",
                    choices=["auto", "direct_linear", "four_bar", "rotary_direct"],
                    help="graph engine: restrict the search to one topology")
    ap.add_argument("--photo-fps", type=float, default=4.0,
                    help="assumed photos per second for folder input (default 4)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--body-mass", type=float, default=80.0, help="body mass in kg (default 80)")
    ap.add_argument("--pop", type=int, default=160)
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", help="short label used in the auto-generated run folder name")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT,
                    help="folder that collects all runs (default ./runs)")
    ap.add_argument("--out", help="exact output folder (overrides the runs/<name> scheme)")
    ap.add_argument("--model", default=None, help="path to pose_landmarker .task model")
    args = ap.parse_args(argv)

    if not args.demo and not args.input:
        ap.error("provide --input (photos folder or video) or use --demo")

    if args.limb and args.engine == "graph":
        ap.error("--limb with --engine graph is not supported yet: the graph "
                 "engine has no mechanism objects for the limb/usage animations "
                 "(SPEC-2 E-12). Run per-joint graph designs, or use --engine legacy.")

    cfg = RunConfig(
        input=args.input, demo=args.demo, joint=args.joint, side=args.side,
        limb=args.limb, movement=args.movement, stl=not args.no_stl,
        engine=args.engine, topology=args.topology,
        photo_fps=args.photo_fps, rotate=args.rotate, body_mass=args.body_mass,
        pop=args.pop, generations=args.generations, seed=args.seed,
        model=args.model, label=args.label, runs_root=args.runs_root, out=args.out)
    summary = run(cfg, log=print)

    if summary["status"] != "done":
        return 1
    print(f"\nRun folder: {summary['run_dir']}")
    print("Open demo.gif for the animation, report.md for the summary, "
          "actuator_spec.md for the buildable design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
