"""Stage 2: pose estimation (MediaPipe) and joint-angle trajectory extraction.

Detection (running the model over frames) is separated from series extraction
(turning landmarks into one joint's flexion trajectory) so that full-limb runs
detect once and derive several joints from the same footage.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .joints import (JOINT_TRIPLETS, LIMB_PRESETS, MASS_END_LANDMARK,
                     POSE_CONNECTIONS, SEGMENT_ANTHROPOMETRY)

# fallback BONE segment lengths (m) where the anthropometry len_default is a
# whole-limb mass length instead
BONE_LEN_DEFAULT = {"shoulder": 0.30, "hip": 0.42}

MIN_VISIBILITY = 0.5
MAX_GAP_INTERP = 5      # interpolate runs of up to this many invalid frames
MIN_VALID_FRAMES = 8

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "pose_landmarker_full.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

TRIPLET_COLORS = [(0, 80, 255), (60, 200, 60), (255, 160, 0)]  # BGR per limb joint


@dataclass
class PoseSeries:
    """Uniformly-sampled flexion trajectory for one joint, plus context for viz."""
    joint: str
    side: str
    t: np.ndarray                 # (N,) seconds
    theta_deg: np.ndarray         # (N,) flexion angle
    Lp: float                     # m, proximal segment length (from world landmarks)
    Ld: float                     # m, distal segment length
    norm_landmarks: np.ndarray    # (N, 33, 3) image-normalized x, y, visibility (for viz)
    frame_wh: tuple               # (w, h) of source frames
    n_frames: int
    n_detected: int
    notes: List[str] = field(default_factory=list)
    overlay_paths: List[str] = field(default_factory=list)
    synthetic: bool = False
    L_mass: Optional[float] = None  # m, driven-mass reference length (whole limb
    #                                 for shoulder/hip); defaults to Ld

    def __post_init__(self):
        if self.L_mass is None:
            self.L_mass = self.Ld

    @property
    def rom_deg(self) -> tuple:
        return float(np.min(self.theta_deg)), float(np.max(self.theta_deg))


@dataclass
class Detection:
    """Raw landmark tracks for a frame sequence (one model pass)."""
    norm_all: np.ndarray    # (N, 33, 3) x, y, visibility
    world_all: np.ndarray   # (N, 33, 3) meters
    detected: np.ndarray    # (N,) bool
    times: np.ndarray       # (N,) seconds
    frame_wh: tuple
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.detected)

    @property
    def n_detected(self) -> int:
        return int(self.detected.sum())


def _make_landmarker(model_path: str, video_mode: bool):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pose model not found at {model_path}.\n"
            f"Download it (about 9 MB) with:\n  curl -L -o {model_path} \\\n    {MODEL_URL}")
    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mode, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp, vision.PoseLandmarker.create_from_options(opts)


def detect_all(frames, meta, model_path: Optional[str] = None,
               out_dir: str = "results") -> Detection:
    """Run the pose model over every frame once."""
    model_path = model_path or DEFAULT_MODEL
    video_mode = meta.get("kind") == "video"
    mp, landmarker = _make_landmarker(model_path, video_mode)

    n = len(frames)
    norm_all = np.full((n, 33, 3), np.nan)
    world_all = np.full((n, 33, 3), np.nan)
    detected = np.zeros(n, dtype=bool)
    frame_wh = (frames[0][1].shape[1], frames[0][1].shape[0]) if n else (0, 0)

    last_ms = -1
    for i, (t, bgr) in enumerate(frames):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if video_mode:
            ts_ms = max(int(round(t * 1000)), last_ms + 1)  # must be monotonic
            last_ms = ts_ms
            res = landmarker.detect_for_video(image, ts_ms)
        else:
            res = landmarker.detect(image)
        if not res.pose_landmarks:
            continue
        detected[i] = True
        for k, lm in enumerate(res.pose_landmarks[0]):
            vis = 1.0 if lm.visibility is None else float(lm.visibility)
            norm_all[i, k] = (lm.x, lm.y, vis)
        for k, lm in enumerate(res.pose_world_landmarks[0]):
            world_all[i, k] = (lm.x, lm.y, lm.z)
    landmarker.close()

    notes = list(meta.get("notes", []))
    if detected.any() and np.nanmax(norm_all[detected][:, :, 2]) < 1e-6:
        norm_all[detected, :, 2] = 1.0  # model variant without visibility scores
        notes.append("pose model reported no visibility scores; treating all landmarks as visible")

    if not detected.any():
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "pose_check_fail.jpg")
        _draw_overlay(frames[0][1], None, [], "NO POSE DETECTED", p)
        raise RuntimeError(
            f"No person detected in any of the {n} frames (see {p}). "
            "Use clear, well-lit footage with the full body or limb visible, "
            "ideally from the side of the moving joint.")

    times = np.array([frames[i][0] for i in range(n)])
    return Detection(norm_all=norm_all, world_all=world_all, detected=detected,
                     times=times, frame_wh=frame_wh, notes=notes)


def _interior_flexion_deg(world: np.ndarray, triplet) -> float:
    """Flexion = 180 - interior angle at the joint, from 3D world landmarks."""
    p, j, d = (world[i, :3] for i in triplet)
    u, v = p - j, d - j
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return np.nan
    cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return 180.0 - np.degrees(np.arccos(cosang))


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < win:
        return x
    kernel = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def _draw_overlay(img, norm_lm, triplets, label, path):
    h, w = img.shape[:2]
    out = img.copy()
    if norm_lm is not None:
        pts = [(int(x * w), int(y * h)) for x, y, _ in norm_lm]
        for i, j in POSE_CONNECTIONS:
            cv2.line(out, pts[i], pts[j], (180, 180, 180), 2)
        for k, triplet in enumerate(triplets):
            c = TRIPLET_COLORS[k % len(TRIPLET_COLORS)]
            for i in triplet:
                cv2.circle(out, pts[i], 7, c, -1)
            cv2.line(out, pts[triplet[0]], pts[triplet[1]], c, 3)
            cv2.line(out, pts[triplet[1]], pts[triplet[2]],
                     tuple(min(255, v + 55) for v in c), 3)
    cv2.putText(out, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(out, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(path, out)
    return path


def _candidate_series(det: Detection, triplet) -> Tuple[np.ndarray, np.ndarray]:
    n = det.n
    theta = np.full(n, np.nan)
    vis = np.zeros(n)
    for i in range(n):
        if not det.detected[i]:
            continue
        v = np.min(det.norm_all[i, list(triplet), 2])
        vis[i] = v
        if v >= MIN_VISIBILITY:
            theta[i] = _interior_flexion_deg(det.world_all[i], triplet)
    return theta, vis


def _build_series(det: Detection, frames, joint: str, side: str, theta: np.ndarray,
                  video_mode: bool, out_dir: str, tag: str = "") -> PoseSeries:
    """Clean one joint's raw angle track and package it with context."""
    n = det.n
    notes = list(det.notes)
    triplet = JOINT_TRIPLETS[joint][side]
    theta = theta.copy()
    valid = ~np.isnan(theta)
    idx = np.arange(n)

    # interpolate short gaps
    i = 0
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= MAX_GAP_INTERP:
                theta[i:j] = np.interp(idx[i:j], [i - 1, j], [theta[i - 1], theta[j]])
                valid[i:j] = True
            i = j
        else:
            i += 1

    # keep the longest contiguous valid run
    best_run, run_start = (0, 0), None
    for i in range(n + 1):
        if i < n and valid[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, i)
            run_start = None
    lo, hi = best_run
    if hi - lo < MIN_VALID_FRAMES:
        raise RuntimeError(
            f"Too few consecutive frames with a visible {side} {joint} to form a trajectory.")
    if hi - lo < valid.sum():
        notes.append(f"kept the longest continuous run of {hi - lo} frames "
                     f"(of {int(valid.sum())} valid)")

    t_arr = det.times[lo:hi] - det.times[lo]
    theta_run = _smooth(theta[lo:hi].copy(), 5 if video_mode else 3)

    # segment lengths from metric world landmarks
    p_i, j_i, d_i = triplet
    seg = det.world_all[lo:hi]
    Lp = float(np.nanmedian(np.linalg.norm(seg[:, p_i] - seg[:, j_i], axis=1)))
    Ld = float(np.nanmedian(np.linalg.norm(seg[:, d_i] - seg[:, j_i], axis=1)))
    bone_default = BONE_LEN_DEFAULT.get(joint, SEGMENT_ANTHROPOMETRY[joint]["len_default"])
    if not (0.05 < Ld < 1.2):
        Ld = bone_default
        notes.append(f"{joint}: distal segment length from pose looked implausible; used default")
    if not (0.05 < Lp < 1.2):
        Lp = bone_default

    # driven-mass reference length: full limb for shoulder/hip (mass end =
    # wrist/ankle), otherwise the distal bone itself
    mass_end = MASS_END_LANDMARK.get(joint, {}).get(side, d_i)
    L_mass = float(np.nanmedian(np.linalg.norm(seg[:, mass_end] - seg[:, j_i], axis=1)))
    if not (0.05 < L_mass < 1.5) or L_mass < Ld - 0.02:
        L_mass = SEGMENT_ANTHROPOMETRY[joint]["len_default"]
        notes.append(f"{joint}: mass reference length from pose looked implausible; used default")

    os.makedirs(out_dir, exist_ok=True)
    overlays = []
    for otag, i in [("first", lo), ("mid", (lo + hi) // 2), ("last", hi - 1)]:
        if det.detected[i]:
            p = os.path.join(out_dir, f"pose_check_{tag}{otag}.jpg")
            label = f"{side} {joint}  flexion={theta[i]:.0f} deg"
            overlays.append(_draw_overlay(frames[i][1], det.norm_all[i], [triplet], label, p))

    return PoseSeries(
        joint=joint, side=side, t=t_arr, theta_deg=theta_run, Lp=Lp, Ld=Ld,
        L_mass=L_mass, norm_landmarks=det.norm_all[lo:hi], frame_wh=det.frame_wh,
        n_frames=n, n_detected=det.n_detected, notes=notes, overlay_paths=overlays)


def extract_pose(frames, meta, joint: str = "auto", side: str = "auto",
                 model_path: Optional[str] = None, out_dir: str = "results") -> PoseSeries:
    """Single-joint extraction; auto picks the joint/side with the largest visible ROM."""
    det = detect_all(frames, meta, model_path, out_dir)
    video_mode = meta.get("kind") == "video"

    candidates = {}
    joint_names = [joint] if joint != "auto" else list(JOINT_TRIPLETS)
    sides = [side] if side != "auto" else ["left", "right"]
    for jn in joint_names:
        for sd in sides:
            theta, vis = _candidate_series(det, JOINT_TRIPLETS[jn][sd])
            valid = ~np.isnan(theta)
            if valid.sum() < MIN_VALID_FRAMES:
                continue
            rom = np.nanpercentile(theta, 95) - np.nanpercentile(theta, 5)
            candidates[(jn, sd)] = (rom * (0.25 + 0.75 * vis[valid].mean()), theta)
    if not candidates:
        raise RuntimeError(
            f"A person was detected in {det.n_detected}/{det.n} frames but no joint had "
            f"{MIN_VALID_FRAMES}+ frames with clearly visible landmarks. "
            "Check the pose_check images in the results folder.")
    (joint, side), (score, theta) = max(candidates.items(), key=lambda kv: kv[1][0])
    series = _build_series(det, frames, joint, side, theta, video_mode, out_dir)
    if len(candidates) > 1:
        series.notes.append(f"auto-selected joint: {side} {joint} "
                            f"(largest visible range of motion among {len(candidates)} candidates)")
    return series


def extract_limb(frames, meta, limb: str, side: str = "auto",
                 model_path: Optional[str] = None, out_dir: str = "results"):
    """Extract every joint of a limb preset from one recording.

    Returns (series_by_joint: Dict[str, PoseSeries], errors: Dict[str, str], side).
    """
    joints = LIMB_PRESETS[limb]
    det = detect_all(frames, meta, model_path, out_dir)
    video_mode = meta.get("kind") == "video"

    if side == "auto":
        score = {}
        for sd in ("left", "right"):
            vis = [np.nanmean(det.norm_all[det.detected][:, list(JOINT_TRIPLETS[j][sd]), 2])
                   for j in joints]
            score[sd] = np.nanmean(vis)
        side = max(score, key=score.get)

    series_by_joint: Dict[str, PoseSeries] = {}
    errors: Dict[str, str] = {}
    for jn in joints:
        theta, _ = _candidate_series(det, JOINT_TRIPLETS[jn][side])
        try:
            jdir = os.path.join(out_dir, "joints", jn)
            series_by_joint[jn] = _build_series(det, frames, jn, side, theta,
                                                video_mode, jdir)
        except RuntimeError as e:
            errors[jn] = str(e)
    if not series_by_joint:
        raise RuntimeError(
            f"None of the {limb} joints ({', '.join(joints)}) were extractable: "
            + "; ".join(errors.values()))

    # combined overlay highlighting all limb triplets
    mid = int(np.flatnonzero(det.detected)[len(np.flatnonzero(det.detected)) // 2])
    trips = [JOINT_TRIPLETS[j][side] for j in joints if j in series_by_joint]
    _draw_overlay(frames[mid][1], det.norm_all[mid], trips,
                  f"{side} {limb}: " + ", ".join(series_by_joint),
                  os.path.join(out_dir, "pose_check_limb.jpg"))
    return series_by_joint, errors, side
