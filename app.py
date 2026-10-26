#!/usr/bin/env python3
"""Local web UI for the prosthetic actuator pipeline.

    .venv/bin/python app.py     ->  http://127.0.0.1:5001

Upload a video or a set of photos (or launch the built-in demo), watch the
pipeline log live, and browse every run's outputs in one place.
"""

import json
import os
import re
import sys
import threading
import time
import uuid

from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory)
from werkzeug.utils import secure_filename

import numpy as np

from prosthetic_pipeline.capture import HEIC_EXTS, IMAGE_EXTS, VIDEO_EXTS
from prosthetic_pipeline.joints import (FREE_CAPTURE, JOINT_NAMES, JOINT_TRIPLETS,
                                        LIMB_PRESETS, MOVEMENTS, SAFETY_NOTE)
from prosthetic_pipeline.runner import (DEFAULT_RUNS_ROOT, RunConfig,
                                        resolve_run_dir)
from prosthetic_pipeline.runner import run as run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024  # 800 MB uploads
app.config["TEMPLATES_AUTO_RELOAD"] = True

RUNS_ROOT = DEFAULT_RUNS_ROOT
_JOBS = {}
_LOCK = threading.Lock()

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


class Job:
    def __init__(self, run_name):
        self.id = uuid.uuid4().hex[:12]
        self.run_name = run_name
        self.state = "running"
        self.lines = []
        self.error = None
        self.created = time.time()

    def log(self, msg=""):
        self.lines.append(str(msg))


def _busy():
    return any(j.state == "running" for j in _JOBS.values())


def _worker(job: Job, cfg: RunConfig):
    try:
        summary = run_pipeline(cfg, log=job.log)
        job.state = "done" if summary["status"] == "done" else "failed"
        job.error = summary.get("error")
    except Exception as e:  # runner handles its own errors; this is a backstop
        job.state = "failed"
        job.error = str(e)
        job.log(f"ERROR: {e}")


def _form_float(name, default, lo, hi):
    try:
        v = float(request.form.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(max(v, lo), hi)


def _form_int(name, default, lo, hi):
    return int(_form_float(name, default, lo, hi))


def _prune_jobs():
    """Keep only the most recent finished jobs (lines can hold a lot of log)."""
    done = [j for j in _JOBS.values() if j.state != "running"]
    for j in sorted(done, key=lambda j: j.created)[:-20]:
        _JOBS.pop(j.id, None)


def _build_config():
    """Parse the form into a RunConfig and save any uploads. Returns (cfg, error)."""
    mode = request.form.get("mode", "demo")
    joint = request.form.get("joint", "auto")
    if joint not in ["auto"] + JOINT_NAMES:
        joint = "auto"
    side = request.form.get("side", "auto")
    if side not in ("auto", "left", "right"):
        side = "auto"
    limb = None
    if request.form.get("target") == "limb":
        limb = request.form.get("limb", "arm")
        if limb not in LIMB_PRESETS:
            limb = "arm"
    label = re.sub(r"[^a-z0-9]+", "-", request.form.get("label", "").lower()).strip("-")[:40]

    movement = request.form.get("movement") or None
    if movement not in MOVEMENTS:
        movement = None
    if movement:  # the movement task must match what is being tested
        target = MOVEMENTS[movement]["target"]
        if "limb" in target:
            if limb != target["limb"]:
                return None, (f"movement '{movement}' is a whole-{target['limb']} task - "
                              "select that limb (or a matching movement)")
        else:
            if limb:
                return None, (f"movement '{movement}' targets the {target['joint']} joint, "
                              "not a whole limb")
            if joint == "auto":
                joint = target["joint"]
            elif joint != target["joint"]:
                return None, (f"movement '{movement}' targets the {target['joint']}, "
                              f"but the {joint} joint is selected")

    if limb and request.form.get("engine") == "graph":
        return None, ("full-limb runs are not supported on the graph engine yet - "
                      "pick a single joint, or switch the engine to legacy")

    verified = request.form.get("capture_verified")
    cfg = RunConfig(
        demo=(mode == "demo"),
        joint=joint, side=side, limb=limb, movement=movement,
        capture_verified=None if verified is None else verified == "1",
        stl=request.form.get("stl") is not None,
        engine=("graph" if request.form.get("engine") == "graph"
                and request.form.get("target") != "limb" else "legacy"),
        topology=(request.form.get("topology") if request.form.get("topology") in
                  ("auto", "direct_linear", "four_bar", "rotary_direct") else "auto"),
        photo_fps=_form_float("photo_fps", 4.0, 0.2, 120),
        rotate=_form_int("rotate", 0, 0, 270),
        body_mass=_form_float("body_mass", 80.0, 20, 250),
        pop=_form_int("pop", 160, 20, 600),
        generations=_form_int("generations", 120, 10, 1000),
        seed=_form_int("seed", 0, 0, 10 ** 6),
        runs_root=RUNS_ROOT,
    )

    if mode == "video":
        f = request.files.get("video")
        if not f or not f.filename:
            return None, "no video file uploaded"
        fname = secure_filename(f.filename) or "video.mov"
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VIDEO_EXTS:
            return None, f"unsupported video type {ext}"
        cfg.label = label or os.path.splitext(fname)[0].lower()[:40] or "video"
        dest = os.path.join(resolve_run_dir(cfg), "input")
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, fname)
        f.save(path)
        cfg.input = path
        clog = request.form.get("coach_log")
        if clog and len(clog) < 2_000_000:
            try:
                json.loads(clog)          # must be valid JSON
                with open(os.path.join(dest, "coach_log.json"), "w") as cf:
                    cf.write(clog)
            except ValueError:
                pass
    elif mode == "photos":
        files = [f for f in request.files.getlist("photos") if f and f.filename]
        if not files:
            return None, "no photos uploaded"
        cfg.label = label or "photos"
        dest = os.path.join(resolve_run_dir(cfg), "input", "photos")
        os.makedirs(dest, exist_ok=True)
        n_saved = 0
        for f in files:
            fname = secure_filename(f.filename)
            ext = os.path.splitext(fname)[1].lower()
            if fname and (ext in IMAGE_EXTS or ext in HEIC_EXTS):
                f.save(os.path.join(dest, fname))
                n_saved += 1
        if n_saved < 4:
            return None, "need at least 4 usable photos (jpg/png/heic)"
        cfg.input = dest
    else:  # demo
        cfg.label = label or ("demo-" + (limb or "joint"))
        resolve_run_dir(cfg)
    return cfg, None


@app.post("/api/run")
def api_run():
    with _LOCK:  # short critical section: reserve the single run slot
        if _busy():
            return jsonify(error="a run is already in progress - wait for it to finish"), 409
        job = Job(None)
        _JOBS[job.id] = job
        _prune_jobs()
    try:
        cfg, err = _build_config()  # upload saving happens outside the lock
        if err:
            job.state, job.error = "failed", err
            return jsonify(error=err), 400
        job.run_name = cfg.run_name
        threading.Thread(target=_worker, args=(job, cfg), daemon=True).start()
        return jsonify(job=job.id, run=job.run_name)
    except OSError as e:
        job.state, job.error = "failed", f"could not save upload: {e}"
        return jsonify(error=job.error), 500


# ---------------------------------------------------------------------------
# live movement coach: analyze single webcam frames during guided recording
# ---------------------------------------------------------------------------

_COACH = {"lm": None, "lock": threading.Lock()}


def _coach_landmarker():
    if _COACH["lm"] is None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        from prosthetic_pipeline.pose import DEFAULT_MODEL
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=DEFAULT_MODEL),
            running_mode=vision.RunningMode.IMAGE, num_poses=1)
        _COACH["lm"] = vision.PoseLandmarker.create_from_options(opts)
    return _COACH["lm"]


@app.post("/api/coach")
def api_coach():
    import cv2
    import mediapipe as mp

    from prosthetic_pipeline.pose import MIN_VISIBILITY, _interior_flexion_deg

    f = request.files.get("frame")
    if not f:
        return jsonify(error="no frame"), 400
    joints = [j for j in request.form.get("joints", "").split(",") if j in JOINT_TRIPLETS]
    side = request.form.get("side", "auto")
    buf = np.frombuffer(f.read(), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="bad frame"), 400
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with _COACH["lock"]:
        res = _coach_landmarker().detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return jsonify(detected=False, side=side, per_joint={}, tips=[
            "No person detected - step back so the whole body is in frame, and add light"])

    norm = np.array([[lm.x, lm.y, 1.0 if lm.visibility is None else lm.visibility]
                     for lm in res.pose_landmarks[0]])
    world = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_world_landmarks[0]])

    if side == "auto":
        score = {}
        for sd in ("left", "right"):
            vis = [norm[list(JOINT_TRIPLETS[j][sd]), 2].min() for j in joints] or [0]
            score[sd] = float(np.mean(vis))
        side = max(score, key=score.get)

    per_joint, tips = {}, []
    for j in joints:
        trip = list(JOINT_TRIPLETS[j][side])
        vis = float(norm[trip, 2].min())
        visible = vis >= MIN_VISIBILITY
        angle = float(_interior_flexion_deg(world, trip)) if visible else None
        per_joint[j] = {"visible": visible, "angle": angle}
        if not visible:
            tips.append(f"Your {side} {j} isn't clearly visible - turn side-on "
                        "and keep the whole limb in frame")
    return jsonify(detected=True, side=side, per_joint=per_joint, tips=tips)


@app.get("/api/status/<job_id>")
def api_status(job_id):
    job = _JOBS.get(job_id)
    if not job:
        abort(404)
    since = request.args.get("since", 0, type=int)
    return jsonify(state=job.state, run=job.run_name, error=job.error,
                   lines=job.lines[since:], total=len(job.lines))


@app.get("/api/runs")
def api_runs():
    path = os.path.join(RUNS_ROOT, "index.json")
    if not os.path.exists(path):
        return jsonify([])
    with open(path) as f:
        return jsonify(json.load(f))


@app.get("/")
def index():
    return render_template("index.html", joints=JOINT_NAMES,
                           movements=MOVEMENTS, free_capture=FREE_CAPTURE,
                           safety=SAFETY_NOTE)


def _stl_files(artifacts, prefix=""):
    return sorted(a for a in artifacts
                  if a.startswith(prefix) and a.endswith(".stl")
                  and a[len(prefix):].count("/") == 0)


@app.get("/run/<name>")
def run_page(name):
    if not _SAFE_NAME.match(name):
        abort(404)
    run_dir = os.path.join(RUNS_ROOT, name)
    run_json = os.path.join(run_dir, "run.json")
    if not os.path.exists(run_json):
        abort(404)
    with open(run_json) as f:
        data = json.load(f)
    artifacts = data.get("artifacts", {})

    spec_html = ""
    spec_md = os.path.join(run_dir, "actuator_spec.md")
    if os.path.exists(spec_md):
        with open(spec_md) as f:
            spec_html = _mini_markdown(f.read())

    # per-joint sections for limb runs
    joint_sections = []
    for jn, js in (data.get("joints") or {}).items():
        prefix = f"joints/{jn}/"
        joint_sections.append({
            "name": jn, "summary": js,
            "images": [a for a in (prefix + "tracking.png", prefix + "actuator_drawing.png",
                                   prefix + "printable_preview.png", prefix + "mechanism_curves.png")
                       if a in artifacts],
            "stls": _stl_files(artifacts, prefix),
            "files": sorted(a for a in artifacts if a.startswith(prefix)),
        })

    log_tail = ""
    log_path = os.path.join(run_dir, "log.txt")
    if data.get("status") != "done" and os.path.exists(log_path):
        with open(log_path) as f:
            log_tail = "".join(f.readlines()[-40:])
    pose_checks = sorted(k for k in artifacts
                         if os.path.basename(k).startswith("pose_check") and "/" not in k)
    return render_template("run.html", name=name, data=data, spec_html=spec_html,
                           log_tail=log_tail, pose_checks=pose_checks,
                           joint_sections=joint_sections,
                           root_stls=_stl_files(artifacts))


@app.get("/runs/<name>/f/<path:filename>")
def run_file(name, filename):
    if not _SAFE_NAME.match(name):
        abort(404)
    return send_from_directory(os.path.join(RUNS_ROOT, name), filename)


def _mini_markdown(md: str) -> str:
    """Small renderer for our own generated markdown (headers, bold, code, lists)."""
    import html as html_mod
    out, in_list = [], False
    for raw in md.splitlines():
        line = html_mod.escape(raw.rstrip())
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("### "):
            out.append(f"<h4>{line[4:]}</h4>")
        elif line.startswith("## "):
            out.append(f"<h3>{line[3:]}</h3>")
        elif line.startswith("# "):
            continue  # the page template provides the section heading
        elif line:
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


if __name__ == "__main__":
    os.makedirs(RUNS_ROOT, exist_ok=True)
    port = int(os.environ.get("PORT", "5001"))
    if len(sys.argv) > 2 and sys.argv[1] == "--port":
        port = int(sys.argv[2])
    print(f"Prosthetic pipeline UI ->  http://127.0.0.1:{port}")
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    except (OSError, SystemExit) as e:
        # werkzeug prints its own "Address already in use" then exits; add
        # the concrete way out before propagating
        is_busy = isinstance(e, SystemExit) or "in use" in str(e).lower()
        if is_busy:
            print(f"\nPort {port} is taken (an old app.py still running?).")
            print(f"  find it:   lsof -ti :{port}")
            print(f"  or re-run: .venv/bin/python app.py --port {port + 1}")
            sys.exit(1)
        raise
