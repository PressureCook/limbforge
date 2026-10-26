"""Stage 1: load movement data from a folder of photos or a video file."""

import os
import re
import subprocess
import sys
import tempfile

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HEIC_EXTS = {".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

MAX_FRAMES = 300  # cap on frames sent to pose estimation


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _convert_heic(path: str, workdir: str) -> str:
    """Convert HEIC to JPEG using macOS's built-in `sips`. Returns new path or ''."""
    if sys.platform != "darwin":
        return ""
    out = os.path.join(workdir, os.path.splitext(os.path.basename(path))[0] + ".jpg")
    try:
        subprocess.run(["sips", "-s", "format", "jpeg", path, "--out", out],
                       check=True, capture_output=True, timeout=30)
        return out
    except Exception:
        return ""


WORK_LONG_EDGE = 1080  # px; pose estimation gains nothing above this


def _downscale(frame, meta):
    """Resize so the long edge is at most WORK_LONG_EDGE (memory + speed)."""
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= WORK_LONG_EDGE:
        return frame
    s = WORK_LONG_EDGE / long_edge
    if not any(n.startswith("frames downscaled") for n in meta["notes"]):
        meta["notes"].append(f"frames downscaled to {WORK_LONG_EDGE} px long edge")
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _apply_rotation(frame, rotate: int):
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def load_frames(input_path: str, photo_fps: float = 4.0, rotate: int = 0):
    """Yield (t_seconds, bgr_image) and return metadata.

    Returns (frames, meta) where frames is a list of (t, image) and meta
    describes the source. Photos in a folder are ordered by filename (natural
    sort) and spaced at 1/photo_fps seconds; videos use their real fps.
    """
    input_path = os.path.expanduser(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    frames = []
    meta = {"source": input_path, "kind": None, "fps": None, "n_source": 0,
            "notes": []}

    if os.path.isdir(input_path):
        meta["kind"] = "photos"
        meta["fps"] = photo_fps
        names = sorted(os.listdir(input_path), key=_natural_key)
        heic_dir = None
        paths = []
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            full = os.path.join(input_path, n)
            if ext in IMAGE_EXTS:
                paths.append(full)
            elif ext in HEIC_EXTS:
                if heic_dir is None:
                    heic_dir = tempfile.mkdtemp(prefix="heic_conv_")
                conv = _convert_heic(full, heic_dir)
                if conv:
                    paths.append(conv)
                else:
                    meta["notes"].append(f"could not convert HEIC file {n}")
        meta["n_source"] = len(paths)
        indexed = list(enumerate(paths))  # original indices keep the time base honest
        if len(indexed) > MAX_FRAMES:
            stride = -(-len(indexed) // MAX_FRAMES)
            indexed = indexed[::stride]
            meta["notes"].append(f"subsampled photos by {stride}x to stay under {MAX_FRAMES} frames")
        for orig_i, p in indexed:
            img = cv2.imread(p)
            if img is None:
                meta["notes"].append(f"unreadable image skipped: {os.path.basename(p)}")
                continue
            frames.append((orig_i / photo_fps, _downscale(_apply_rotation(img, rotate), meta)))
        meta["notes"].append(
            f"photo timing assumed at {photo_fps:.1f} photos/sec (use --photo-fps to change)")
        return frames, meta

    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTS:
        meta["kind"] = "video"
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 240:
            fps = 30.0
            meta["notes"].append("video fps unreadable; assumed 30")
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(1, -(-n_total // MAX_FRAMES)) if n_total else 1
        meta["fps"] = fps / stride
        i = 0
        truncated = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                # hard cap regardless of whether the container reported a frame
                # count (webm/VFR clips often don't) - prevents unbounded memory
                if len(frames) >= MAX_FRAMES:
                    truncated = True
                    break
                frames.append((i / fps, _downscale(_apply_rotation(frame, rotate), meta)))
            i += 1
        cap.release()
        meta["n_source"] = i
        if stride > 1:
            meta["notes"].append(f"sampled every {stride}th video frame")
        if truncated:
            meta["notes"].append(
                f"video length unknown to the decoder; kept the first {MAX_FRAMES} frames "
                f"({frames[-1][0]:.0f} s) - trim the clip if the movement happens later")
        return frames, meta

    if ext in IMAGE_EXTS or ext in HEIC_EXTS:
        raise ValueError(
            "A single photo cannot describe movement. Pass a folder of photos "
            "(a burst / frames of the motion) or a video file.")
    raise ValueError(f"Unsupported input type: {input_path}")
